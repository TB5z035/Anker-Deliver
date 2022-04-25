import logging
import os

import MinkowskiEngine as ME
import numpy as np
import torch
from IPython import embed
from torch import nn

from .dataset.dataset import initialize_data_loader
from .dataset.datasets import load_dataset
from .distributed_utils import all_gather_list, get_rank, get_world_size
from .lib.solvers import initialize_optimizer, initialize_scheduler
from .main import main
from .utils import AverageMeter, Timer, checkpoint, precision_at_one


def _set_seed(config, step):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    seed = config.seed + step
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn):
    return
    # v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config, transform_data_fn)
    # if writer:
    #     writer.add_scalar('validation/mIoU', v_mIoU, curr_iter)
    #     writer.add_scalar('validation/loss', v_loss, curr_iter)
    #     writer.add_scalar('validation/precision_at_1', v_score, curr_iter)

    # return v_mIoU


def train(
    model,
    data_loader,
    val_data_loader,
    config,
    transform_data_fn=None,
):
    device = config.device_id
    distributed = get_world_size() > 1
    model.train()

    # Configuration
    data_timer, iter_timer = Timer(), Timer()
    fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()

    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    fw_time_avg, bw_time_avg, ddp_time_avg = AverageMeter(), AverageMeter(), AverageMeter()

    losses, scores = AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    # Train the network
    logging.info('===> Start training on {} GPUs, batch-size={}'.format(get_world_size(), config.train_batch_size * get_world_size()))
    best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    data_iter = data_loader.__iter__()
    while is_training:
        for iteration in range(len(data_loader) // config.iter_size):
            optimizer.zero_grad()
            data_time, batch_loss, batch_score = 0, 0, 0
            iter_timer.tic()

            # set random seed for every iteration for trackability
            _set_seed(config, curr_iter)

            for sub_iter in range(config.iter_size):
                # Get training data
                data_timer.tic()
                coords, input, target, _, _ = data_iter.next()
                # For some networks, making the network invariant to even, odd coords is important. Random translation
                coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                # Preprocess input
                color = input[:, :3].int()
                if config.normalize_color:
                    input[:, :3] = input[:, :3] / 255. - 0.5
                sinput = ME.SparseTensor(input.to(device), coords.to(device))
                #sinput = ME.SparseTensor(input,coords).to(device)
                data_time += data_timer.toc(False)

                # Feed forward
                fw_timer.tic()

                inputs = (sinput,) 
                # model.initialize_coords(*init_args)
                soutput = model(*inputs)
                # The output of the network is not sorted
                target = target.long().to(device)

                loss = criterion(soutput.F, target.long())

                # Compute and accumulate gradient
                loss /= config.iter_size
                pred = soutput.F.max(1)[1]
                score = precision_at_one(pred, target)

                fw_timer.toc(False)
                bw_timer.tic()

                loss.backward()
                bw_timer.toc(False)

                logging_output = {'loss': loss.item(), 'score': score / config.iter_size}
                logging.warn(f"{loss.item()}")

                ddp_timer.tic()
                if distributed:
                    logging_output = all_gather_list(logging_output)
                    logging_output = {w: np.mean([a[w] for a in logging_output]) for w in logging_output[0]}

                batch_loss += logging_output['loss']
                batch_score += logging_output['score']
                ddp_timer.toc(False)

            # Update number of steps
            optimizer.step()
            scheduler.step()

            data_time_avg.update(data_time)
            iter_time_avg.update(iter_timer.toc(False))
            fw_time_avg.update(fw_timer.diff)
            bw_time_avg.update(bw_timer.diff)
            ddp_time_avg.update(ddp_timer.diff)

            losses.update(batch_loss, target.size(0))
            scores.update(batch_score, target.size(0))

            if curr_iter >= config.max_iter:
                is_training = False
                break

            if config.stat_freq is not None and (curr_iter % config.stat_freq == 0 or curr_iter == 1):
                lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(epoch, curr_iter,
                                                                                  len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tData time: {:.4f}, Forward time: {:.4f}, Backward time: {:.4f}, DDP time: {:.4f}, Total iter time: {:.4f}".format(
                    scores.avg, data_time_avg.avg, fw_time_avg.avg, bw_time_avg.avg, ddp_time_avg.avg, iter_time_avg.avg)
                logging.info(debug_str)
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            # if config.save_freq is not None and curr_iter % config.save_freq == 0:
            #     checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

            if curr_iter % config.empty_cache_freq == 0:
                # Clear cache
                torch.cuda.empty_cache()

            # End of iteration
            curr_iter += 1

        epoch += 1

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    # if not distributed or get_rank() == 0:
    #     val_miou = validate(model, val_data_loader, None, curr_iter, config, transform_data_fn)
    #     if val_miou is not None and val_miou > best_val_miou:
    #         best_val_miou = val_miou
    #         best_val_iter = curr_iter
    #         checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")

    if not distributed or get_rank() == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, postfix="last", name=f"{config.round}.pth")

    logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

def mod_config(config):
    config.stat_freq = 1
    config.checkpoint_dir = 'checkpoints'
    ###### weights
    if config.round is None:
        checkpoint_list = [int(i.split('.')[0]) for i in os.listdir('checkpoints') if i != 'init.pth']
        if len(checkpoint_list) == 0:
            config.weights = f'checkpoints/init.pth'
            config.run_name = f'finetune-round1'
            config.data_root = 'input/0/' # fixme
            config.round = 1
        else:
            config.weights = f'checkpoints/{max(checkpoint_list)}.pth'
            config.data_root = f'input/{max(checkpoint_list)}/' # fixme
            config.run_name = f'finetune-round{max(checkpoint_list) + 1}'
            config.round = max(checkpoint_list) + 1
    elif config.round == 0:
        config.weights = f'checkpoints/init.pth'
        config.data_root = 'input/0/' # fixme
        config.run_name = f'finetune-round1'
        config.round = 1
    elif isinstance(config.round, int):
        config.weights = f'checkpoints/{config.round}.pth'
        config.data_root = f'input/{max(checkpoint_list)}/' # fixme
        config.run_name = f'finetune-round{config.round + 1}'
        config.round = config.round + 1
    else:
        assert False, 'round parameter should be int or None'
    return config

def action(model, config):
    rank = get_rank()
    logger = logging.getLogger(__name__)

    train_dataset_cls = load_dataset(config.train_dataset)

    # hint: use phase to select different split of data
    train_dataloader = initialize_data_loader(
        train_dataset_cls,
        config,
        num_workers=config.num_workers,
        phase='train',
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.train_batch_size,
        limit_numpoints=False,
    )
    if rank == 0:
        logger.info("Dataloader setup done")
    train(model, train_dataloader, None, config)


if __name__ == '__main__':
    main(action, mod_config)
