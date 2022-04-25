import os
from logging import getLogger
from posixpath import split

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from torch import nn

from .dataset.dataset import initialize_data_loader
from .dataset.datasets import load_dataset
from .distributed_utils import get_rank, get_world_size
from .main import main
from .utils import calc_iou, fast_hist, save_prediction


def inference(
    model,
    dataloader,
    config,
    save=False,
    evaluate=True,
):
    """Evaluation on val / test dataset"""
    torch.cuda.empty_cache()
    rank = get_rank()
    world_size = get_world_size
    device = f'cuda:{rank}'
    logger = getLogger(__name__)
    model.eval()

    with torch.no_grad():
        global_cnt = 0

        losses = []
        num_labels = dataloader.dataset.NUM_LABELS
        iu_hist = torch.zeros((num_labels, num_labels))
        # One epoch
        for step, (batched_coords, batched_feats, batched_targets, unique_map, inverse_map) in enumerate(dataloader):

            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))
            batched_sparse_output = model(batched_sparse_input)
            batched_outputs = batched_sparse_output.F.cpu()
            batched_prediction = batched_outputs.max(dim=1)[1].int()

            # Save predictions for each scene
            if save:
                for i in range(config.test_batch_size):
                    plydata = PlyData.read('input/init/' + sorted(os.listdir('input/init'), key=lambda x: int(x.split('.')[0]))[global_cnt])
                    selector = batched_coords[:, 0] == i
                    # single_scene_coords = batched_coords[selector][:, 1:]
                    single_scene_coords = torch.as_tensor(np.stack([
                        plydata['vertex']['x'],
                        plydata['vertex']['y'],
                        plydata['vertex']['z'],
                    ], axis=1))
                    single_scene_predictions = batched_prediction[selector][inverse_map[i]]
                    save_prediction(single_scene_coords,
                                    single_scene_predictions + 1,
                                    f"{config.eval_result_dir}/{global_cnt}.ply",
                                    mode="prediction")
                    global_cnt += 1
            else:
                global_cnt += config.test_batch_size

            # Evaluate prediction
            if evaluate:
                criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
                loss = criterion(batched_outputs, batched_targets.long())
                iu_hist += fast_hist(batched_prediction.flatten(), batched_targets.flatten(), num_labels)
                iou = calc_iou(iu_hist)
                # ious_perclass.append(iou)

                if world_size == 1 or rank == 0:
                    logger.info(f"---> inference #{step} of {len(dataloader)} loss: {loss:.4f} iou: {iou.mean():.4f}")
                losses.append(loss)

                # TODO calculate average precision
                probablities = F.softmax(batched_outputs, dim=1)  # pylint: disable=unused-variable
                # avg_precision =

    if evaluate:
        return torch.stack(losses).mean(), iou.mean(), iou

def mod_config(config):
    ###### weights
    if config.round is None:
        checkpoint_list = [int(i.split('.')[0]) for i in os.listdir('checkpoints') if i != 'init.pth']
        if len(checkpoint_list) == 0:
            config.weights = f'checkpoints/init.pth'
            config.run_name = f'inference-round0'
            config.eval_result_dir = f'predicted/round0'
        else:
            config.weights = f'checkpoints/{max(checkpoint_list)}.pth'
            config.run_name = f'inference-round{max(checkpoint_list)}'
            config.eval_result_dir = f'predicted/round{max(checkpoint_list)}'
        
        # input_data_list = [int(i) for i in os.listdir('input') if i != 'init']
        config.data_root = 'input/init'
    elif config.round == 0:
        config.weights = f'checkpoints/init.pth'
        config.data_root = 'input/init'
        config.run_name = f'inference-round0'
        config.eval_result_dir = f'predicted/round0'
    elif isinstance(config.round, int):
        config.weights = f'checkpoints/{config.round}.pth'
        config.data_root = 'input/init'
        config.run_name = f'inference-round{config.round}'
        config.eval_result_dir = f'predicted/round{config.round}'
    else:
        assert False, 'round parameter should be int or None'
    return config

def action(model, config):
    rank = get_rank()
    logger = getLogger(__name__)
    val_dataset_cls = load_dataset(config.val_dataset)
    val_dataloader = initialize_data_loader(
        val_dataset_cls,
        config,
        num_workers=config.num_workers,
        phase='val',
        augment_data=False,
        shuffle=False,
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False,
    )
    if rank == 0:
        logger.info("Dataloader setup done")
    val_loss, val_miou, iou_per_class = inference(model, val_dataloader, config, save=True)
    logger.info(f"VAL: loss (avg): {val_loss.item():.4f}, iou (avg): {val_miou.item():.4f}")
    for idx, i in enumerate(iou_per_class):
        logger.info(f"VAL: iou (cls#{idx}): {i.item():.4f}")


if __name__ == '__main__':
    main(action, mod_config)
