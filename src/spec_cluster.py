from copy import deepcopy
import os
import tempfile

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import pymeshlab
import torch
from IPython import embed
from plyfile import PlyData
from sklearnex import patch_sklearn
from tqdm import tqdm

from .config import CONFIGS, SCANNET_COLOR_MAP
from .utils import count_time, plydata_to_arrays, setup_mapping, timer, add_fields_online

patch_sklearn()
from sklearn.cluster import KMeans


class SpecClusterPipeline():

    def __init__(self, scan_path, count=200) -> None:
        self._load_plydata(scan_path)

    def _load_plydata(self, scan_path):
        self.full_plydata = PlyData.read(scan_path)
        return self

    @timer
    def downsample(self, dstarget=8000):
        assert hasattr(self, 'full_plydata') is not None

        with tempfile.NamedTemporaryFile(suffix='.ply') as temp_f:
            self.full_plydata.write(temp_f.name)
            meshset = pymeshlab.MeshSet()
            meshset.load_new_mesh(temp_f.name)

            meshset.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetperc=dstarget / len(self.full_plydata['vertex']),
                autoclean=True,
                qualitythr=0.8,
            )
            meshset.apply_filter('meshing_remove_unreferenced_vertices')

            meshset.save_current_mesh(temp_f.name)
            new_ply_data = PlyData.read(temp_f.name)

        self.sampled_plydata = new_ply_data
        return self

    @timer
    def setup_mapping(self):
        self.full2sampled, self.sampled2full = setup_mapping(self.full_plydata, self.sampled_plydata)
        return self

    @timer
    def calc_geod_dist(self):
        assert self.sampled_plydata is not None
        plydata = self.sampled_plydata
        vertices, faces = plydata_to_arrays(plydata)
        solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
        distances = []
        print(len(plydata['vertex']))
        for i in tqdm(range(len(plydata['vertex'])), disable=True):
            distances.append(solver.compute_distance(i))

        geod_mat = np.stack(distances)
        geod_mat = (geod_mat + geod_mat.T) / 2
        self.geod_mat = geod_mat
        return self

    @timer
    def calc_ang_dist(self, abs_inv=True, knn_range=None):
        assert self.sampled_plydata is not None
        knn = CONFIGS['normal_knn_range'] if knn_range is None else knn_range

        pcd = o3d.geometry.PointCloud()
        vertices, _ = plydata_to_arrays(self.sampled_plydata)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        center = vertices.mean(axis=0)
        pcd.orient_normals_towards_camera_location(center)
        normals = torch.as_tensor(np.asarray(pcd.normals))
        # use absolute result
        if abs_inv:
            t = 1 - (normals @ normals.T).abs()
        else:
            t = 1 - (normals @ normals.T)
        ang_mat = t / t.mean()
        self.ang_mat: torch.Tensor = ang_mat
        return self

    @timer
    def calc_aff_mat(self, ratio: float = None):
        assert self.geod_mat is not None
        assert self.ang_mat is not None
        ratio = CONFIGS['dist_proportion'] if ratio is None else ratio
        geod_mat = torch.as_tensor(self.geod_mat).cuda()
        ang_mat = self.ang_mat.cuda()
        dist_mat = ratio * geod_mat + (1 - ratio) * ang_mat
        dist_mat = (dist_mat + dist_mat.T) / 2
        sigma = dist_mat.mean() if CONFIGS['auto_temperature'] else CONFIGS['temperature']
        aff_mat = np.e**(-dist_mat / (2 * sigma**2))

        aff_iv_mat = torch.diag(1 / aff_mat.sum(dim=1).sqrt())
        n_mat = (aff_iv_mat @ aff_mat @ aff_iv_mat)
        self.aff_mat: torch.Tensor = n_mat
        return self

    @timer
    def calc_embedding(self, feature: int = 30):
        assert self.aff_mat is not None
        eigh_vals, eigh_vecs = torch.linalg.eigh(self.aff_mat)
        del self.aff_mat
        eigh_vecs = eigh_vecs.T
        embedding = eigh_vecs[-feature:, :].flip(dims=(0,)).T
        embedding_cpu = embedding.cpu()
        self.embedding_mat = np.asarray(embedding_cpu)
        return self

    @timer
    def knn_cluster(self, count=200):
        assert self.full_plydata is not None
        assert self.full2sampled is not None
        assert self.embedding_mat is not None
        mask = self.full_plydata['vertex']['label'][self.sampled2full] != 255
        # mask = self.sampled_plydata['vertex']['label'] != 255
        assert mask.any()
        assert count < mask.sum()
        selected_vertex_indices_in_sampled = np.random.permutation(np.nonzero(mask)[0])[:count]

        selected_vertex_labels = self.full_plydata['vertex']['label'][self.sampled2full][selected_vertex_indices_in_sampled]
        self.cluster_result = KMeans(
            n_clusters=count,
            init=self.embedding_mat[selected_vertex_indices_in_sampled],
        ).fit(self.embedding_mat)

        naive_indices = self.cluster_result.predict(self.embedding_mat)
        selected_predicted_labels = selected_vertex_labels[naive_indices]
        selected_predicted_distances = self.cluster_result.transform(self.embedding_mat)
        self.full_predicted_labels = selected_predicted_labels[self.full2sampled]
        full_naive_indices = naive_indices[self.full2sampled]
        self.full_predicted_distances = selected_predicted_distances[self.full2sampled]
        self.full_predicted_distances = self.full_predicted_distances[np.arange(len(self.full_predicted_distances)), full_naive_indices]
        print(self.full_predicted_distances.shape)
        return self

    def evaluate_cluster_result(self, correct_sum=None, total_sum=None):
        assert len(self.full_predicted_labels == self.full_plydata['vertex']['label'])
        correct = (self.full_plydata['vertex']['label'] == self.full_predicted_labels).sum()
        total = len(self.full_plydata['vertex'])
        if correct_sum is not None:
            correct_sum += correct
        if total_sum is not None:
            total_sum += total
        print(f'Correctness: {correct * 100/total:.2f}%')
        return self

    def evaluate_cluster_result_iou(self):
        Is = np.zeros((41))
        Os = np.zeros((41))
        for cls_idx in range(41):
            i = np.bitwise_and(self.full_predicted_labels == cls_idx, self.full_plydata['vertex']['label'] == cls_idx).sum()
            o = np.bitwise_or(self.full_predicted_labels == cls_idx, self.full_plydata['vertex']['label'] == cls_idx).sum()
            Is[cls_idx] = i
            Os[cls_idx] = o
        self.Is = Is
        self.Os = Os
        return self

    def save(self, save_dir="debug"):
        shot = len(self.sample_ids)
        os.makedirs(f"{save_dir}/spec_predictions", exist_ok=True)
        torch.save({
            "labels": self.full_predicted_labels,
            "confidence": self.full_predicted_distances,
        }, f"{save_dir}/spec_predictions/{self.scan_id}_{shot}.obj")
        return self

    def save_visualize(self, dir='debug'):
        assert self.full_predicted_labels is not None
        os.makedirs(dir, exist_ok=True)
        map_np = np.asarray(list(SCANNET_COLOR_MAP.values()))
        cloned_plydata = deepcopy(self.full_plydata)
        cloned_plydata['vertex']['red'] = map_np[:, 0][self.full_predicted_labels]
        cloned_plydata['vertex']['green'] = map_np[:, 1][self.full_predicted_labels]
        cloned_plydata['vertex']['blue'] = map_np[:, 2][self.full_predicted_labels]
        cloned_plydata.write(f'{dir}/output.ply')
        return self


if __name__ == '__main__':
    p = SpecClusterPipeline('misc/scene0000_00_vh_clean_2.labels.ply')
    p.downsample().setup_mapping().calc_geod_dist().calc_ang_dist().calc_aff_mat().calc_embedding().knn_cluster().evaluate_cluster_result(
    ).save_visualize()
    embed()
