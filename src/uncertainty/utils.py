import functools
import json
import logging
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from plyfile import PlyData, PlyElement

from .dataset.datasets.scannet import COLOR_MAP
from .distributed_utils import get_rank, get_world_size


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


def setup_logger(config):
    """
    Logger setup function
    This function should only be called by main process in DDP
    """
    logging.root.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(name)s\t][%(levelname)s\t] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    log_level = logging.INFO

    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(formatter)
    cli_handler.setLevel(log_level)

    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H:%M', time.localtime(now / 1000))
        filename = os.path.join(config.log_dir, f"{config.run_name}-{timestr}.log")

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

    logging.root.addHandler(cli_handler)
    logging.root.addHandler(file_handler)


def distributed_init(init_method, rank, world_size):
    """
    torch distributed iniitialized
    create a multiprocess group and initialize nccl communicator
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda(rank))
        else:
            dist.all_reduce(torch.zeros(1))
        return dist.get_rank()
    logging.getLogger().warn("Distributed already initialized!")


def fast_hist(predictions, labels, num_labels):
    selector = (labels >= 0) & (labels < num_labels)  # ndarray of bool type, selector
    stat = torch.bincount(num_labels * labels[selector].int() + predictions[selector], minlength=num_labels**2).reshape(num_labels, num_labels)
    return stat


def calc_iou(stat):
    with np.errstate(divide='ignore', invalid='ignore'):
        return stat.diagonal() / (stat.sum(dim=0) + stat.sum(dim=1) - stat.diagonal() + 1e-7)


def save_prediction(coords, feats, path, mode='unc', in_fn=None):
    """
    save prediction for **ONE** scene
    mode:
    - 'unc': treat feats as the uncertainty of each class (batchsize * num_cls)
    - 'prediction': treat feats as the predicted label (batchsize * 1)s
    """
    if mode == 'unc':
        # feats = feats.min(dim=1)[0]
        # feats = feats[:, 2]
        feats = feats.mean(dim=1)

        coef = 3  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        # feats = feats/feats.max() * 128

        def rgbl_fn(id, row, feat):
            return 127 + int(feat), 255 - int(feat), 127, 0
    elif mode == 'unc_select':
        feats, labels = feats
        feats = feats.gather(1, labels.unflatten(0, (labels.shape[0], 1)))
        coef = 3  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        def rgbl_fn(id, row, feat):
            return 127 + int(feat), 255 - int(feat), 127, 0
    elif mode == 'unc_reinforce':
        print('reinforce render')
        feats, labels, truths = feats
        feats = feats.gather(1, labels.unflatten(0, (labels.shape[0], 1)))
        coef = 3  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        def rgbl_fn(id, row, feat):
            if truths[id].item() == 255:
                return 127 + int(feat), 255 - int(feat), 64, 0
            else:
                return 0, 0, 255, 0

    elif mode == 'prediction':

        def rgbl_fn(id, row, feat):
            return (*[int(i) for i in COLOR_MAP[feat.item()]], feat)
    else:
        rgbl_fn = in_fn

    print(f'Saved to {path}')
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'w') as f:  # pylint: disable=invalid-name
        f.write('ply\n'
                'format ascii 1.0\n'
                f'element vertex {coords.shape[0]}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n'
                'property uchar red\n'
                'property uchar green\n'
                'property uchar blue\n'
                'property uchar label\n'
                'end_header\n')
        for id, (row, feat) in enumerate(zip(coords, feats)):
            f.write(f'{row[0]} {row[1]} {row[2]} '
                    f'{rgbl_fn(id, row, feat)[0]} '
                    f'{rgbl_fn(id, row, feat)[1]} '
                    f'{rgbl_fn(id, row, feat)[2]} '
                    f'{rgbl_fn(id, row, feat)[3]}\n')


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ExpTimer(Timer):
    """ Exponential Moving Average Timer """

    def __init__(self, alpha=0.5):
        super(ExpTimer, self).__init__()
        self.alpha = alpha

    def toc(self):
        self.diff = time.time() - self.start_time
        self.average_time = self.alpha * self.diff + \
            (1 - self.alpha) * self.average_time
        return self.average_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def precision_at_one(pred, target, ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return float('nan')


def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None, name=None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    # only works for rank == 0
    if get_world_size() > 1 and get_rank() > 0:
        return

    # os.makedirs('weights', exist_ok=True)
    # if config.overwrite_weights:
    #     if postfix is not None:
    #         filename = f"checkpoint_{config.model}{postfix}.pth"
    #     else:
    #         filename = f"checkpoint_{config.model}.pth"
    # else:
    #     filename = f"checkpoint_{config.model}_iter_{iteration}.pth"
    filename = name
    checkpoint_file = config.checkpoint_dir + '/' + filename

    _model = model.module if get_world_size() > 1 else model
    state = {'iteration': iteration, 'epoch': epoch, 'arch': config.model, 'state_dict': _model.state_dict(), 'optimizer': optimizer.state_dict()}
    if best_val is not None:
        state['best_val'] = best_val
        state['best_val_iter'] = best_val_iter
    # json.dump(vars(config), open(config.checkpoint_dir + '/config.json', 'w'), indent=4)
    logging.info(f"Checkpoint saved to {checkpoint_file}")
    torch.save(state, checkpoint_file)
    # Delete symlink if it exists
    # if os.path.exists(f'{config.log_dir}/weights.pth'):
    #     os.remove(f'{config.log_dir}/weights.pth')
    # Create symlink
    # os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')
