import os
from pathlib import Path

import numpy as np
from plyfile import PlyData

from ..dataset import VoxelizationDataset, DatasetPhase
from ...lib.utils import read_txt

CLASS_LABELS = (
    'wall',
    'floor',
    'cabinet',
    'bed',
    'chair',
    'sofa',
    'table',
    'door',
    'window',
    'bookshelf',
    'picture',
    'counter',
    'desk',
    'curtain',
    'refrigerator',
    'shower curtain',
    'toilet',
    'sink',
    'bathtub',
    'otherfurniture',
)
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (26., 13., 201.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    31: (182., 56., 128.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


class ScannetVoxelizationConfig:
    VOXEL_SIZE = 0.02

    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    ROTATION_AXIS = 'z'
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    LOCFEAT_IDX = 2


class InferenceDataset(ScannetVoxelizationConfig, VoxelizationDataset):

    def __init__(
        self,
        config,
        prevoxel_transform=None,
        input_transform=None,
        target_transform=None,
        augment_data=True,
        phase=None,
    ) -> None:
        self.data_paths = [(config.data_root + '/' + i) for i in sorted(os.listdir(config.data_root), key=lambda x: int(x.split('.')[0]))]
        VoxelizationDataset.__init__(
            self,
            input_transform=input_transform,
            target_transform=target_transform,
            prevoxel_transform=prevoxel_transform,
            ignore_label=config.ignore_label,
            return_transformation=config.return_transformation,
            augment_data=augment_data,
        )

    def load_ply(self, index):
        filepath = self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32) if 'label' in data.dtype.names else None
        return coords, feats, labels, None

    def __len__(self):
        return len(self.data_paths)


class ScannetVoxelizationDatasetBase(ScannetVoxelizationConfig, VoxelizationDataset):
    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'scannetv2_train.txt',
        DatasetPhase.Val: 'scannetv2_val.txt',
        DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
        DatasetPhase.Test: 'scannetv2_test.txt'
    }

    def __init__(
        self,
        config,
        prevoxel_transform=None,
        input_transform=None,
        target_transform=None,
        augment_data=True,
        phase=DatasetPhase.Train,
    ):
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        self.data_root = Path(self.get_root_path(config))
        self.data_paths = read_txt(os.path.join('./splits/scannet', self.DATA_PATH_FILE[phase]))
        VoxelizationDataset.__init__(input_transform=input_transform,
                                     target_transform=target_transform,
                                     prevoxel_transform=prevoxel_transform,
                                     ignore_label=config.ignore_label,
                                     return_transformation=config.return_transformation,
                                     augment_data=augment_data)

    def load_ply(self, index):
        filepath = self.data_root / self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        return coords, feats, labels, None

    def __len__(self):
        return len(self.data_paths)


class ScannetVoxelizationDataset(ScannetVoxelizationDatasetBase):
    get_root_path = lambda self, config: config.scannet_path


class ScannetVoxelizationtestDataset(ScannetVoxelizationDatasetBase):
    get_root_path = lambda self, config: config.scannet_test_path
