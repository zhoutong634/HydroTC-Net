import os
import time
import warnings

import torch
import torch.optim
from tqdm import tqdm

from utils import compute_flood_metrics, finalize_flood_metrics, init_flood_metric_sums, update_flood_metric_sums
import Config as config

warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, batch_metrics,
                  average_metrics, mode, lr, logger):
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'MAE:{:.4f} '.format(batch_metrics['mae'])
    string += '(Avg {:.4f}) '.format(average_metrics['mae'])
    string += 'RMSE:{:.4f} '.format(batch_metrics['rmse'])
    string += '(Avg {:.4f}) '.format(average_metrics['rmse'])
    string += 'R2:{:.4f} '.format(batch_metrics['r2'])
    string += '(Avg {:.4f}) '.format(average_metrics['r2'])
    string += 'Bias:{:.4f} '.format(batch_metrics['bias'])
    string += '(Avg {:.4f}) '.format(average_metrics['bias'])
    string += 'NSE:{:.4f} '.format(batch_metrics['nse'])
    string += '(Avg {:.4f}) '.format(average_metrics['nse'])
    string += 'F1:{:.4f} '.format(batch_metrics['f1'])
    string += '(Avg {:.4f}) '.format(average_metrics['f1'])
    string += 'IoU:{:.4f} '.format(batch_metrics['iou'])
    string += '(Avg {:.4f}) '.format(average_metrics['iou'])
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)


def save_predictions(inputs, targets, preds, vis_path, batch_idx):
    import matplotlib.pyplot as plt

    input_data = inputs[0].cpu().numpy()
    target_data = targets[0, 0].cpu().numpy()
    pred_data = preds[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(input_data[0], cmap='viridis')
    axes[0].set_title('Input (Channel 0)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(target_data, cmap='jet')
    axes[1].set_title('Target Water Depth')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pred_data, cmap='jet')
    axes[2].set_title('Predicted Water Depth')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    save_file = os.path.join(vis_path, f'batch_{batch_idx}_sample_0.png')
    plt.savefig(save_file)
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'
    device = torch.device("cuda")

    if model.training and optimizer is None:
        raise ValueError("Optimizer is not initialized. Please check the optimizer setup.")

    end = time.time()
    time_sum = 0.0
    loss_sum = 0.0
    metric_sums = init_flood_metric_sums()
    desc = f"{logging_mode} Epoch {epoch + 1}"
    pbar = tqdm(loader, total=len(loader), dynamic_ncols=True, leave=False, desc=desc)

    for i, batch in enumerate(pbar, 1):
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        if len(batch) == 3:
            inputs, targets, masks = batch
            masks = masks.to(device, non_blocking=True)
        else:
            inputs, targets = batch
            masks = None

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        grad_enabled = model.training
        with torch.set_grad_enabled(grad_enabled):
            preds = model(inputs)
            try:
                out_loss = criterion(preds, targets.float(), mask=masks)
            except TypeError:
                out_loss = criterion(preds, targets.float())

        if model.training:
            optimizer.zero_grad(set_to_none=True)
            out_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_metrics = compute_flood_metrics(targets, preds, threshold=config.flood_threshold, mask=masks)
        update_flood_metric_sums(metric_sums, targets, preds, threshold=config.flood_threshold, mask=masks)
        average_metrics = finalize_flood_metrics(metric_sums)

        batch_time = time.time() - end

        if epoch % config.vis_frequency == 0 and logging_mode == 'Val' and i == 1:
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_predictions(inputs, targets, preds, vis_path, i)

        time_sum += len(inputs) * batch_time
        loss_sum += len(inputs) * out_loss.item()

        if i == len(loader):
            denom = config.batch_size * (i - 1) + len(inputs)
        else:
            denom = i * config.batch_size

        average_loss = loss_sum / denom
        average_time = time_sum / denom

        end = time.time()

        lr = min(g["lr"] for g in optimizer.param_groups) if optimizer is not None else 0.0
        postfix = {
            "loss": f"{out_loss.item():.4f}",
            "avg": f"{average_loss:.4f}",
            "mae": f"{batch_metrics['mae']:.4f}",
            "rmse": f"{batch_metrics['rmse']:.4f}",
            "f1": f"{batch_metrics['f1']:.4f}",
            "iou": f"{batch_metrics['iou']:.4f}",
        }
        if logging_mode == 'Train':
            postfix["lr"] = f"{lr:.2e}"
        pbar.set_postfix(postfix)

        if i % config.print_frequency == 0:
            pbar.write(
                f"[{logging_mode}] Epoch {epoch + 1}/{config.epochs} "
                f"Iter {i}/{len(loader)} "
                f"Loss {out_loss.item():.4f} Avg {average_loss:.4f} "
                f"MAE {batch_metrics['mae']:.4f} RMSE {batch_metrics['rmse']:.4f} "
                f"F1 {batch_metrics['f1']:.4f} IoU {batch_metrics['iou']:.4f}"
                + (f" LR {lr:.2e}" if logging_mode == 'Train' else "")
            )

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)
            writer.add_scalar(logging_mode + '_mae', batch_metrics['mae'], step)
            writer.add_scalar(logging_mode + '_rmse', batch_metrics['rmse'], step)
            writer.add_scalar(logging_mode + '_r2', batch_metrics['r2'], step)
            writer.add_scalar(logging_mode + '_bias', batch_metrics['bias'], step)
            writer.add_scalar(logging_mode + '_nse', batch_metrics['nse'], step)
            writer.add_scalar(logging_mode + '_f1', batch_metrics['f1'], step)
            writer.add_scalar(logging_mode + '_iou', batch_metrics['iou'], step)
            if hasattr(criterion, 'get_latest_components'):
                for component_name, component_value in criterion.get_latest_components().items():
                    writer.add_scalar(f"{logging_mode}_loss_{component_name}", component_value, step)

    pbar.close()

    if lr_scheduler is not None:
        lr_scheduler.step()
    return average_loss, finalize_flood_metrics(metric_sums)
