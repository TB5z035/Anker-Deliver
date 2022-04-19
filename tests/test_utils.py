from plyfile import PlyData
from src.utils import add_fields_online

def test_add_fields_online():
    ply_path = 'misc/scene0000_00_vh_clean_2.labels.ply'
    plydata = PlyData.read(ply_path)
    ret_plydata = add_fields_online(plydata, [('new_field', 'ushort')], clear=False)
    assert 'new_field' in ret_plydata['vertex'].dtype().fields.keys()

def test_add_fields_online_clear():
    ply_path = 'misc/scene0000_00_vh_clean_2.labels.ply'
    plydata = PlyData.read(ply_path)
    ret_plydata = add_fields_online(plydata, [('new_field', 'ushort')], clear=False)
    assert 'new_field' in ret_plydata['vertex'].dtype().fields.keys()