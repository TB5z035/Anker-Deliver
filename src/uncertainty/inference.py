import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from torch import nn

from ..utils import calc_iou, fast_hist, save_prediction
from .dataset.dataset import initialize_data_loader
from .dataset.datasets import load_dataset
from .main import main


def inference(
    model,
    dataloader,
    config,
    logger,
    rank=0,
    world_size=1,
    save=False,
    evaluate=True,
):
    """Evaluation on val / test dataset"""
    torch.cuda.empty_cache()
    device = f'cuda:{rank}'

    model.eval()

    with torch.no_grad():
        global_cnt = 0

        losses = []
        num_labels = dataloader.dataset.NUM_LABELS
        iu_hist = torch.zeros((num_labels, num_labels))
        # One epoch
        for step, (batched_coords, batched_feats, batched_targets) in enumerate(dataloader):

            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))
            batched_sparse_output = model(batched_sparse_input)
            batched_outputs = batched_sparse_output.F.cpu()
            batched_prediction = batched_outputs.max(dim=1)[1].int()

            # Save predictions for each scene
            if save:
                for i in range(config.test_batch_size):
                    selector = batched_coords[:, 0] == i
                    single_scene_coords = batched_coords[selector][:, 1:]
                    single_scene_predictions = batched_prediction[selector]
                    save_prediction(single_scene_coords,
                                    single_scene_predictions,
                                    f"{config.eval_result_dir}/{global_cnt}_predicted_inference.ply",
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


def action(rank, world_size, model, logger, config):
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
    val_loss, val_miou, iou_per_class = inference(model, val_dataloader, config, logger, rank=rank, world_size=world_size, save=True)
    logger.info(f"VAL: loss (avg): {val_loss.item():.4f}, iou (avg): {val_miou.item():.4f}")
    for idx, i in enumerate(iou_per_class):
        logger.info(f"VAL: iou (cls#{idx}): {i.item():.4f}")


if __name__ == '__main__':
    main(action)
