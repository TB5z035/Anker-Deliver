import functools
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def add_fields_online(plydata: PlyData, fields: List[Tuple], clear=True) -> PlyData:
    p = plydata
    v, f = p.elements
    a = np.empty(len(v.data), v.data.dtype.descr + fields)
    for name in v.data.dtype.fields:
        a[name] = v[name]
    if clear:
        for i in fields:
            a[i[0]] = 0
    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v, f], text=True)

    return p


def plydata_to_arrays(plydata: PlyData) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    faces = np.stack(plydata['face']['vertex_indices'])
    return vertices, faces


def setup_mapping(ply_origin: PlyData, ply_down_sampled: PlyData):
    full_coords, _ = plydata_to_arrays(ply_origin)
    sampled_coords, _ = plydata_to_arrays(ply_down_sampled)
    full_coords = torch.as_tensor(full_coords).cuda()
    sampled_coords = torch.as_tensor(sampled_coords).cuda()
    full2sampled = torch.as_tensor([((sampled_coords - coord)**2).sum(dim=1).min(dim=0)[1] for coord in full_coords
                                   ])  # use index in full mesh to find index of the closest in sampled mesh
    sampled2full = torch.as_tensor([((full_coords - coord)**2).sum(dim=1).min(dim=0)[1] for coord in sampled_coords
                                   ])  # use index in smapled mesh to find index of the closest in full mesh
    del full_coords, sampled_coords
    return full2sampled, sampled2full


def timer(fn):

    @functools.wraps(fn)
    def inner(*args, **kw):
        start = datetime.now()
        r = fn(*args, **kw)
        end = datetime.now()
        print(f"{fn.__name__} spent: {(end - start).seconds}s {(end-start).microseconds // 1000} ms")
        return r

    return inner


@contextmanager
def count_time(name=None, file=sys.stdout):
    print(f"Process {(name+' ') if name is not None else ''}start", file=file)
    start = datetime.now()
    yield
    end = datetime.now()
    print(f"Process {(name+' ') if name is not None else ''}spent: {(end - start).seconds}s {(end-start).microseconds // 1000} ms", file=file)
