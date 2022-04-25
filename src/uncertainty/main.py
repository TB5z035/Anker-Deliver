import logging
import os
import random
import time

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from .config import get_config
from .dataset.dataset import initialize_data_loader
from .dataset.datasets import load_dataset
from .dataset.datasets.scannet import COLOR_MAP
from .lib.solvers import initialize_optimizer, initialize_scheduler
from .models import load_model
from .utils import distributed_init, setup_logger


def main(action, mod_config):
    """
    Program entry
    Branch based on number of available GPUs
    """
    device_count = torch.cuda.device_count()
    if device_count > 1:
        port = random.randint(10000, 20000)
        init_method = f'tcp://localhost:{port}'
        mp.spawn(
            fn=main_worker,
            args=(device_count, init_method, action, mod_config),
            nprocs=device_count,
        )
    else:
        main_worker(0, 1, None, action, mod_config)


def main_worker(rank=0, world_size=1, init_method=None, action=None, mod_config=None):
    """
    Top pipeline
    """

    # + Device and distributed setup
    if not torch.cuda.is_available():
        raise Exception("No GPU Found")
    device = rank
    if world_size > 1:
        distributed_init(init_method, rank, world_size)

    config = get_config()
    config = mod_config(config)
    setup_logger(config)
    logger = logging.getLogger(__name__)
    if rank == 0:
        logger.info(f'Run with {world_size} cpu')

    torch.cuda.set_device(device)

    # Set seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    if rank == 0:
        logger.info("Running config")
        for key, value in vars(config).items():
            logger.info(f"---> {key:>30}: {value}")  # pylint: disable=W1203

    # Setup model
    num_in_channel = 3  # RGB
    # num_labels = val_dataloader.dataset.NUM_LABELS
    num_labels = 20
    model_class = load_model(config.model)
    model = model_class(num_in_channel, num_labels, config)

    # Load pretrained weights
    if config.weights:
        state = torch.load(config.weights, map_location=f'cuda:{device}')
        model.load_state_dict({k: v for k, v in state['state_dict'].items() if not k.startswith('projector.')})
        if rank == 0:
            logger.info(f"Weights loaded from {config.weights}")  # pylint: disable=W1203

    model = model.to(device)
    if rank == 0:
        logger.info("Model setup done")
        logger.info(f"\n{model}")  # pylint: disable=W1203

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[device],
            output_device=[device],
            broadcast_buffers=False,
            # bucket_cap_mb=
        )

    action(model, config)
    return 
    # wandb

    # # Action switch
    # if config.do_train:
    #     # Set up test dataloader
    #     train_dataset_cls = load_dataset(config.train_dataset)
    #     val_dataset_cls = load_dataset(config.val_dataset)

    #     # hint: use phase to select different split of data
    #     train_dataloader = initialize_data_loader(
    #         train_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='train',
    #         augment_data=True,
    #         shuffle=True,
    #         repeat=True,
    #         batch_size=config.train_batch_size,
    #         limit_numpoints=False,
    #     )

    #     val_dataloader = initialize_data_loader(
    #         val_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='val',
    #         augment_data=False,
    #         shuffle=False,
    #         repeat=False,
    #         batch_size=config.val_batch_size,
    #         limit_numpoints=False,
    #     )
    #     if rank == 0:
    #         logger.info("Dataloader setup done")
    #     train(model, train_dataloader, val_dataloader, config, logger, rank=rank, world_size=world_size)
    # elif config.do_validate:
    #     val_dataset_cls = load_dataset(config.val_dataset)
    #     val_dataloader = initialize_data_loader(
    #         val_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='val',
    #         augment_data=False,
    #         shuffle=False,
    #         repeat=False,
    #         batch_size=config.val_batch_size,
    #         limit_numpoints=False,
    #     )
    #     if rank == 0:
    #         logger.info("Dataloader setup done")
    #     val_loss, val_miou, iou_per_class = inference(model, val_dataloader, config, logger, rank=rank, world_size=world_size, evaluate=True)
    #     logger.info(f"VAL: loss (avg): {val_loss.item():.4f}, iou (avg): {val_miou.item():.4f}")
    #     for idx, i in enumerate(iou_per_class):
    #         logger.info(f"VAL: iou (cls#{idx}): {i.item():.4f}")
    # elif config.do_unc_demo:
    #     val_dataset_cls = load_dataset(config.unc_dataset)
    #     unc_dataloader = initialize_data_loader(
    #         val_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='train',
    #         augment_data=False,
    #         shuffle=False,
    #         repeat=False,
    #         batch_size=config.test_batch_size,
    #         limit_numpoints=False,
    #     )
    #     unc_demo(model, unc_dataloader, config, logger)
    #     # unc_inference(model, unc_dataloader, config, logger)
    # elif config.do_unc_inference:
    #     val_dataset_cls = load_dataset(config.unc_dataset)
    #     unc_dataloader = initialize_data_loader(
    #         val_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='train',
    #         augment_data=False,
    #         shuffle=False,
    #         repeat=False,
    #         batch_size=config.test_batch_size,
    #         limit_numpoints=False,
    #     )
    #     unc_inference(model, unc_dataloader, config, logger)
    # elif config.do_verbose_inference:
    #     val_dataset_cls = load_dataset(config.unc_dataset)
    #     unc_dataloader = initialize_data_loader(
    #         val_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='train',
    #         augment_data=False,
    #         shuffle=False,
    #         repeat=False,
    #         batch_size=config.test_batch_size,
    #         limit_numpoints=False,
    #     )
    #     verbose_inference(model, unc_dataloader, config, logger)
    # elif config.do_unc_render:
    #     val_dataset_cls = load_dataset(config.unc_dataset)
    #     unc_dataloader = initialize_data_loader(
    #         val_dataset_cls,
    #         config,
    #         num_workers=config.num_workers,
    #         phase='train',
    #         augment_data=False,
    #         shuffle=False,
    #         repeat=False,
    #         batch_size=config.test_batch_size,
    #         limit_numpoints=False,
    #     )
    #     unc_render(unc_dataloader, config, logger, option=2)
    # if world_size > 1:
    #     dist.destroy_process_group()
