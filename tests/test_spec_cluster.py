from plyfile import PlyData

from src.spec_cluster import SpecClusterPipeline
import os
ply_path = 'misc/scene0000_00_vh_clean_2.labels.ply'

def test_downsample():
    pipeline = SpecClusterPipeline(ply_path)
    for i in range(5):
        ds_target = 20000 - 3000 * i
        pipeline.downsample(ds_target)
        assert len(pipeline.sampled_plydata['vertex']) < ds_target + 1000
    assert True