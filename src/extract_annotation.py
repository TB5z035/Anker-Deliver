import os
import numpy as np

from plyfile import PlyData

from .uncertainty.config import SCANNET_COLOR_MAP


def run():
    files = [int(i.split('.')[0]) for i in sorted(os.listdir('input/init'))]
    checkpoint_list = [int(i.split('.')[0]) for i in os.listdir('checkpoints') if i != 'init.pth']
    if len(checkpoint_list) == 0:
        path = 'input/0'
    else:
        path = f'input/{max(checkpoint_list)}'
    os.makedirs(path, exist_ok=True)
    for file in files:
        plydata = PlyData.read(f'input/init/{file}.ply')
        if os.path.isfile(f'annotated/{file}.ply'):
            plydata_label = PlyData.read(f'annotated/{file}.ply')
            colors = np.stack([
                plydata_label['vertex']['red'],
                plydata_label['vertex']['green'],
                plydata_label['vertex']['blue'],
            ], axis=1)
            labels = np.zeros((len(plydata_label['vertex']), 1), dtype='u1')
            for idx, val in SCANNET_COLOR_MAP.items():
                labels[(colors == np.asarray(val)).all(axis=1, keepdims=True)] = idx
            plydata['vertex']['label'] = labels.squeeze()
        plydata.write(f'{path}/{file}.ply')

if __name__ == '__main__':
    run()