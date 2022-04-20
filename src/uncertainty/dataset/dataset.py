from abc import ABC
from enum import Enum
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import transforms as t
from .sampler import DistributedInfSampler, InfSampler
from ..lib.distributed_utils import get_world_size
from .voxelizer import Voxelizer


class DatasetPhase(Enum):
    Train = 0
    Val = 1
    Val2 = 2
    TrainVal = 3
    Test = 4


def cache(func):

    def wrapper(self, *args, **kwargs):
        # Assume that args[0] is index
        index = args[0]
        if self.cache:
            if index not in self.cache_dict[func.__name__]:
                results = func(self, *args, **kwargs)
                self.cache_dict[func.__name__][index] = results
            return self.cache_dict[func.__name__][index]
        else:
            return func(self, *args, **kwargs)

    return wrapper


class DictDataset(Dataset, ABC):

    def __init__(
        self,
        input_transform=None,
        target_transform=None,
    ):
        """
        data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
        """
        Dataset.__init__(self)

        self.input_transform, self.target_transform = input_transform, target_transform
        self.data_loader_dict = {
            'input': (self.load_input, self.input_transform),
            'target': (self.load_target, self.target_transform),
        }

    def load_input(self, index):
        raise NotImplementedError

    def load_target(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        out_array = []
        for load, transform in self.data_loader_dict:
            v = load(index)
            if transform is not None:
                v = transform(v)
            out_array.append(v)
        return out_array

    def __len__(self):
        raise NotImplementedError


class VoxelizationConfig:
    VOXEL_SIZE = 0.05  # 5cm

    IS_TEMPORAL = False
    CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
    ROTATION_AXIS = None
    NUM_IN_CHANNEL = None
    NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
    IGNORE_LABELS = None  # labels that are not evaluated

    # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
    # augmentation has to be done before voxelization
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
    ELASTIC_DISTORT_PARAMS = None

    # MISC.
    PREVOXELIZATION_VOXEL_SIZE = None

    # Augment coords to feats
    AUGMENT_COORDS_TO_FEATS = False


class VoxelizationDataset(VoxelizationConfig, DictDataset):
    """
    This dataset loads RGB point clouds and their labels as a list of points
    and voxelizes the pointcloud with sufficient data augmentation.
    """

    def __init__(self,
                 input_transform=None,
                 target_transform=None,
                 prevoxel_transform=None,
                 ignore_label=255,
                 return_transformation=False,
                 return_mapping=False,
                 augment_data=False):

        DictDataset.__init__(
            self,
            input_transform=input_transform,
            target_transform=target_transform,
        )

        self.prevoxel_transform = prevoxel_transform
        self.ignore_mask = ignore_label
        self.return_transformation = return_transformation
        self.return_mapping = return_mapping
        self.augment_data = augment_data

        # Prevoxel transformations
        self.voxelizer = Voxelizer(voxel_size=self.VOXEL_SIZE,
                                   clip_bound=self.CLIP_BOUND,
                                   use_augmentation=augment_data,
                                   scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
                                   rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
                                   translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
                                   ignore_label=ignore_label)

        # map labels not evaluated to ignore_label
        # TODO check
        label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                label_map[l] = self.ignore_mask
            else:
                label_map[l] = n_used
                n_used += 1
        label_map[self.ignore_mask] = self.ignore_mask
        self.label_map = label_map
        self.NUM_LABELS -= len(self.IGNORE_LABELS)

    def __getitem__(self, index):
        coords, feats, labels, center = self.load_ply(index)
        # Downsample the pointcloud with finer voxel size before transformation for memory and speed
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            inds = ME.utils.sparse_quantize(coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
            coords = coords[inds]
            feats = feats[inds]
            labels = labels[inds]

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords, feats, labels = self.prevoxel_transform(coords, feats, labels)

        # For saving mappings
        # coords, feats, labels, transformation, mapping, inverse_mapping = self.voxelizer.voxelize(coords, feats, labels, center=center, )
        coords, feats, labels, transformation = self.voxelizer.voxelize(coords, feats, labels, center=center, )

        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, feats, labels = self.input_transform(coords, feats, labels)
        if self.target_transform is not None:
            coords, feats, labels = self.target_transform(coords, feats, labels)
        if self.IGNORE_LABELS is not None:
            labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

        # Use coordinate features if config is set
        if self.AUGMENT_COORDS_TO_FEATS:
            norm_coords = coords - coords.mean(0)
            # color must come first.
            if isinstance(coords, np.ndarray):
                feats = np.concatenate((feats, norm_coords), 1)
            else:
                feats = torch.cat((feats, norm_coords), 1)

        return_args = [coords, feats, labels]
        if self.return_transformation:
            return_args.append(transformation.astype(np.float32))

        if self.return_mapping:
            return_args.append(mapping.cpu())
            return_args.append(inverse_mapping.cpu())

        return tuple(return_args)

    def load_ply(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           input_transform=None,
                           target_transform=None,
                           collate=True):
    if collate and config.return_transformation:
        collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
    elif collate:
        collate_fn = t.cfl_collate_fn_factory(limit_numpoints)
    else:
        collate_fn = None

    prevoxel_transform_train = []
    if augment_data:
        prevoxel_transform_train.append(t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS))

    if len(prevoxel_transform_train) > 0:
        prevoxel_transforms = t.Compose(prevoxel_transform_train)
    else:
        prevoxel_transforms = None

    input_transforms = []
    if input_transform is not None:
        input_transforms += input_transform

    if augment_data:
        input_transforms += [
            t.RandomDropout(0.2),
            t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
            t.ChromaticAutoContrast(),
            t.ChromaticTranslation(config.data_aug_color_trans_ratio),
            t.ChromaticJitter(config.data_aug_color_jitter_std),
            # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
        ]

    if len(input_transforms) > 0:
        input_transforms = t.Compose(input_transforms)
    else:
        input_transforms = None

    dataset = DatasetClass(config,
                           prevoxel_transform=prevoxel_transforms,
                           input_transform=input_transforms,
                           target_transform=target_transform,
                           augment_data=augment_data,
                           phase=phase)

    data_args = {
        'dataset': dataset,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }

    if repeat:
        if get_world_size() > 1:
            data_args['sampler'] = DistributedInfSampler(dataset, shuffle=shuffle)  # torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            data_args['sampler'] = InfSampler(dataset, shuffle)
    else:
        data_args['shuffle'] = shuffle

    data_loader = DataLoader(**data_args)

    return data_loader
