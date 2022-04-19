from plyfile import PlyData

from src.spec_cluster import SpecClusterPipeline
import os

ply_path = 'misc/scene0000_00_vh_clean_2.labels.ply'


class TestSpectralCluster:
    pipeline = SpecClusterPipeline(ply_path)

    def test_downsample(self):
        for i in range(5):
            ds_target = 20000 - 3000 * i
            self.pipeline.downsample(ds_target)
            assert len(self.pipeline.sampled_plydata['vertex']) < ds_target + 1000
        assert True

    def test_calc_geod_dist(self):
        self.pipeline.calc_geod_dist()
        assert self.pipeline.geod_mat.shape[0] == len(self.pipeline.sampled_plydata['vertex'])
        assert self.pipeline.geod_mat.shape[1] == len(self.pipeline.sampled_plydata['vertex'])
        assert (self.pipeline.geod_mat == self.pipeline.geod_mat.T).all()

    def test_calc_ang_dist(self):
        self.pipeline.calc_ang_dist()
        assert self.pipeline.ang_mat.shape[0] == len(self.pipeline.sampled_plydata['vertex'])
        assert self.pipeline.ang_mat.shape[1] == len(self.pipeline.sampled_plydata['vertex'])
        assert (self.pipeline.ang_mat == self.pipeline.ang_mat.T).all()
