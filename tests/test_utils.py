import os
import numpy as np
from plyfile import PlyData
from src.utils import add_fields_online, plydata_to_arrays

ply_path = 'misc/scene0000_00_vh_clean_2.labels.ply'


def test_add_fields_online():
    plydata = PlyData.read(ply_path)
    ret_plydata = add_fields_online(plydata, [('new_field1', 'ushort'), ('new_field2', 'ushort')], clear=False)
    assert 'new_field1' in ret_plydata['vertex'].dtype().fields.keys()
    assert 'new_field2' in ret_plydata['vertex'].dtype().fields.keys()


def test_add_fields_online_clear():
    plydata = PlyData.read(ply_path)
    ret_plydata = add_fields_online(plydata, [('new_field1', 'ushort'), ('new_field2', 'ushort')], clear=True)
    assert 'new_field1' in ret_plydata['vertex'].dtype().fields.keys()
    assert not ret_plydata['vertex']['new_field1'].any()
    assert not ret_plydata['vertex']['new_field2'].any()


def test_plydata_to_arrays():
    plydata = PlyData.read(ply_path)
    vertice_count = len(plydata['vertex'])
    face_count = len(plydata['face'])
    v, f = plydata_to_arrays(plydata)
    assert v.shape[0] == vertice_count
    assert v.shape[1] == 3
    assert f.shape[0] == face_count
    assert f.shape[1] == 3
