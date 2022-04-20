from argparse import Namespace
from ..src.uncertainty.lib.datasets.scannet import InferenceDataset


def test_inference_dataset():
    config = Namespace(**{'ignore_label': 255, 'return_transformation': False, 'data_paths': ['misc/scene0000_00_vh_clean_2.labels.ply']})
    InferenceDataset(config)
    assert True