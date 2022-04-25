SCANNET_COLOR_MAP = {
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
    13: (124., 232., 109.),
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
    31: (56., 23, 131.),
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
# DistanceProportion = 0.8
# TargetFaceNum = 10000
# QualityThreshold = 0.8
# MeshHeatT = 1.0
# NormalKnnRange = 10
# AutoTemperature = yes
# Temperature = 0.1
# Shots = 20
CONFIGS = {'normal_knn_range': 10, 'dist_proportion': 0.8, 'auto_temperature': True, 'temperature': 0.1}

import argparse


def str2opt(arg):
    assert arg in ['SGD', 'Adam']
    return arg


def str2scheduler(arg):
    assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR']
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


def str2list(l):
    return [int(i) for i in l.split(',')]


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='Res16UNet34C', help='Model name')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default=None, help='Saved weights to load')
net_arg.add_argument('--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')

# Wrappers
net_arg.add_argument('--wrapper_type', default='None', type=str, help='Wrapper on the network')
net_arg.add_argument('--wrapper_region_type', default=1, type=int, help='Wrapper connection types 0: hypercube, 1: HYPER_CROSS, (default: 1)')
net_arg.add_argument('--wrapper_kernel_size', default=3, type=int, help='Wrapper kernel size')

# Meanfield arguments
net_arg.add_argument('--meanfield_iterations', type=int, default=10, help='Number of meanfield iterations')
net_arg.add_argument('--crf_spatial_sigma', default=1, type=int, help='Trilateral spatial sigma')
net_arg.add_argument('--crf_chromatic_sigma', default=12, type=int, help='Trilateral chromatic sigma')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=1e-2)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR')
opt_arg.add_argument('--max_iter', type=int, default=120000)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='log/')
dir_arg.add_argument('--data_root', type=str, default='outputs/default')
dir_arg.add_argument('--scannet_path', type=str, help='Scannet online voxelization dataset root dir')
dir_arg.add_argument('--scannet_test_path', type=str, help='Scannet online voxelization dataset root dir')
dir_arg.add_argument('--eval_result_dir', type=str, default='/')
dir_arg.add_argument('--checkpoint_dir', type=str, default='/')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--return_transformation', type=str2bool, default=False)
data_arg.add_argument('--ignore_label', type=int, default=255)
data_arg.add_argument('--train_dataset', type=str, default='')
data_arg.add_argument('--train_batch_size', type=int, default=8)
data_arg.add_argument('--val_dataset', type=str, default='')
data_arg.add_argument('--val_batch_size', type=int, default=1)
data_arg.add_argument('--test_batch_size', type=int, default=1)
data_arg.add_argument('--num_workers', type=int, default=0, help='num workers for train/test dataloader')

# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument('--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
data_aug_arg.add_argument('--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
data_aug_arg.add_argument('--data_aug_hue_max', type=float, default=0.5, help='Hue translation range. [0, 1]')
data_aug_arg.add_argument('--data_aug_saturation_max', type=float, default=0.20, help='Saturation translation range, [0, 1]')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--save_freq', type=int, default=None, help='save frequency')
train_arg.add_argument('--val_freq', type=int, default=None, help='validation frequency')
train_arg.add_argument('--stat_freq', type=int, default=None, help='validation frequency')
train_arg.add_argument('--empty_cache_freq', type=int, default=1, help='Clear pytorch cache frequency')
train_arg.add_argument('--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument('--resume_optimizer', default=True, type=str2bool, help='Use checkpoint optimizer states when resume training')

# Distributed Training configurations
distributed_arg = add_argument_group('Distributed')
distributed_arg.add_argument('--distributed_world_size', type=int, default=2)
distributed_arg.add_argument('--distributed_rank', type=int, default=0)
distributed_arg.add_argument('--distributed_backend', type=str, default='nccl')
distributed_arg.add_argument('--distributed_init_method', type=str, default='')
distributed_arg.add_argument('--distributed_port', type=int, default=10010)
distributed_arg.add_argument('--device_id', type=int, default=0)
distributed_arg.add_argument('--distributed_no_spawn', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument('--run_name', type=str, default='')
misc_arg.add_argument('--unc_round', type=int, default=20)
misc_arg.add_argument('--unc_result_dir', type=str, default='/')


misc_arg.add_argument('--optim_step', type=int, default=1)
misc_arg.add_argument('--validate_step', type=int, default=500)
misc_arg.add_argument('--train_epoch', type=int, default=20000)


misc_arg.add_argument('--unc_stat_path', type=str, default='/')
misc_arg.add_argument('--unc_dataset', type=str, default="")
misc_arg.add_argument('--ignore_index', type=int, default=255)
misc_arg.add_argument('--do_train', action='store_true')
misc_arg.add_argument('--do_validate', action='store_true')
misc_arg.add_argument('--do_unc_inference', action='store_true')
misc_arg.add_argument('--do_unc_demo', action='store_true')
misc_arg.add_argument('--do_verbose_inference', action='store_true')
misc_arg.add_argument('--do_unc_render', action='store_true')
misc_arg.add_argument('--round', type=int, default=None)


def get_config():
    config = parser.parse_args()
    return config  # Training settings
