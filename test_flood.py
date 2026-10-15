# 测试终端命令行
# python .\test_flood.py  --checkpoint .\FloodPrediction\UNet\FloodPrediction_06.23_15h22\models\best_model-UNet.pth.tar `  --test_dir .\data\test  --dem_path .\data\static\dem.input  --save_dir .\results_unet_6231522
import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

import Config as config
from flood_dataset import build_flood_tensors, load_dem, load_rain_series, load_target_time_minutes
from nets.UNet import UNet
from nets.HydroTC_Net import HydroTC_Net
from nets.WideUNet import WideUNet
from utils import (
    finalize_flood_metrics,
    init_flood_metric_sums,
    merge_flood_metric_sums,
    update_flood_metric_sums,
)

matplotlib.use("Agg")


def clean_state_dict(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def build_model(model_type, device, base_channel=None, bottleneck_channel=None):
    if model_type == "UNet":
        model = UNet(n_channels=config.n_channels, n_classes=config.n_labels)
    elif model_type == "HydroTC_Net":
        model = HydroTC_Net(
            config.get_CTranS_config(),
            n_channels=config.n_channels,
            n_classes=config.n_labels,
            img_size=config.img_size,
        )
    elif model_type == "wide_unet":
        model = WideUNet(
            n_channels=config.n_channels,
            n_classes=config.n_labels,
            base_channel=base_channel or 68,
            bottleneck_channel=bottleneck_channel or 1936,
        )
    else:
        raise ValueError("Unsupported model type: %s" % model_type)
    return model.to(device)


def resize_to_match(input_tensor, target_size):
    if input_tensor.shape[2:] != target_size:
        return F.interpolate(input_tensor, size=target_size, mode="bilinear", align_corners=False)
    return input_tensor


def load_single_event_data(rain_path, nc_path, dem_path, normalize=True, pad_multiple=16,
                           rain_norm_factor=config.rain_norm_factor):
    dem, valid_mask, building_mask = load_dem(dem_path, normalize=normalize)
    rain_series, rain_times = load_rain_series(rain_path, normalize=normalize, return_times=True)
    target_times = load_target_time_minutes(nc_path)

    with xr.open_dataset(nc_path) as ds:
        water_depth = ds["h"].values

    inputs_list = []
    labels_list = []
    for time_index in range(water_depth.shape[0]):
        input_tensor, label = build_flood_tensors(
            rain_series,
            dem,
            water_depth[time_index],
            time_index,
            rain_times=rain_times,
            target_time_minutes=float(target_times[time_index]),
            normalize=normalize,
            pad_multiple=pad_multiple,
            rain_norm_factor=rain_norm_factor,
        )
        inputs_list.append(input_tensor)
        labels_list.append(label)

    return inputs_list, labels_list, dem.shape, valid_mask, building_mask


def save_comparison_plot(event_save_dir, sample_idx, preds_cpu, targ_cpu, orig_shape=None):
    if orig_shape is not None:
        h_orig, w_orig = orig_shape
        preds_cpu = preds_cpu[:h_orig, :w_orig]
        targ_cpu = targ_cpu[:h_orig, :w_orig]

    comp_dir = os.path.join(event_save_dir, "comparison")
    pred_npy_dir = os.path.join(event_save_dir, "pred_npy")
    target_npy_dir = os.path.join(event_save_dir, "target_npy")
    pred_png_dir = os.path.join(event_save_dir, "pred_png")
    target_png_dir = os.path.join(event_save_dir, "target_png")
    for d in [comp_dir, pred_npy_dir, target_npy_dir, pred_png_dir, target_png_dir]:
        os.makedirs(d, exist_ok=True)

    np.save(os.path.join(pred_npy_dir, f"pred_{sample_idx}.npy"), preds_cpu)
    np.save(os.path.join(target_npy_dir, f"target_{sample_idx}.npy"), targ_cpu)

    cmap_blue = LinearSegmentedColormap.from_list(
        "water_depth",
        ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"],
        N=256,
    )
    cmap_blue.set_bad(color=(1, 1, 1, 0))
    cmap_diff = LinearSegmentedColormap.from_list("diff_rb", ["blue", "white", "red"], N=256)
    cmap_diff.set_bad(color=(1, 1, 1, 0))

    vmax_depth = 0.3
    time_min = (sample_idx - 1) * 6

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.ma.masked_invalid(preds_cpu), cmap=cmap_blue, vmin=0, vmax=vmax_depth)
    ax.set_title(f"Predicted Water Depth (t={time_min} min)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Water Depth (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(pred_png_dir, f"pred_{sample_idx}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.ma.masked_invalid(targ_cpu), cmap=cmap_blue, vmin=0, vmax=vmax_depth)
    ax.set_title(f"Target Water Depth (t={time_min} min)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Water Depth (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(target_png_dir, f"target_{sample_idx}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    signed_diff = preds_cpu - targ_cpu

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(np.ma.masked_invalid(targ_cpu), cmap=cmap_blue, vmin=0, vmax=vmax_depth)
    axes[0].set_title(f"Target Water Depth (t={time_min} min)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Water Depth (m)")

    im1 = axes[1].imshow(np.ma.masked_invalid(preds_cpu), cmap=cmap_blue, vmin=0, vmax=vmax_depth)
    axes[1].set_title(f"Predicted Water Depth (t={time_min} min)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Water Depth (m)")

    im2 = axes[2].imshow(np.ma.masked_invalid(signed_diff), cmap=cmap_diff, vmin=-0.5, vmax=0.5)
    axes[2].set_title(f"Prediction - Target (t={time_min} min)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Water Depth (m)")

    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, f"comparison_{sample_idx}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def write_metric_csv(metrics_file, metrics, event_count=None):
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["Flood Threshold (m)", f"{config.flood_threshold:.2f}"])
        for key in ["mae", "rmse", "r2", "bias", "nse", "f1", "iou"]:
            writer.writerow([key.upper(), f"{metrics[key]:.6f}"])
        writer.writerow(["Samples", f"{metrics['samples']}"])
        if event_count is not None:
            writer.writerow(["Total Events", f"{event_count}"])


def evaluate_single_event(model, device, train_grid_size, rain_path, nc_path, dem_path,
                          event_name, save_dir, batch_size=1, normalize=True, pad_multiple=16,
                          rain_norm_factor=config.rain_norm_factor):
    print(f"\n{'=' * 60}")
    print(f"Evaluating Event: {event_name}")
    print(f"{'=' * 60}")

    inputs_list, labels_list, orig_shape, valid_mask, building_mask = load_single_event_data(
        rain_path,
        nc_path,
        dem_path,
        normalize=normalize,
        pad_multiple=pad_multiple,
        rain_norm_factor=rain_norm_factor,
    )

    event_save_dir = os.path.join(save_dir, event_name)
    os.makedirs(event_save_dir, exist_ok=True)

    metric_sums = init_flood_metric_sums()

    with torch.no_grad():
        for i in range(0, len(inputs_list), batch_size):
            batch_end = min(i + batch_size, len(inputs_list))
            inputs = torch.stack(inputs_list[i:batch_end]).to(device)
            targets = torch.stack(labels_list[i:batch_end]).to(device)

            if inputs.shape[2:] != train_grid_size:
                inputs = resize_to_match(inputs, train_grid_size)

            preds = model(inputs)
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if preds.shape[2:] != targets.shape[2:]:
                preds = resize_to_match(preds, targets.shape[2:])

            h_orig, w_orig = orig_shape
            metric_mask = valid_mask.astype(bool) & (~building_mask.astype(bool))
            metric_mask = torch.tensor(metric_mask, device=device).unsqueeze(0).unsqueeze(0)
            update_flood_metric_sums(
                metric_sums,
                targets[:, :, :h_orig, :w_orig],
                preds[:, :, :h_orig, :w_orig],
                threshold=config.flood_threshold,
                mask=metric_mask,
            )
            running_metrics = finalize_flood_metrics(metric_sums)

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"Processed {i + 1}/{len(inputs_list)} samples - "
                    f"MAE: {running_metrics['mae']:.6f}, "
                    f"RMSE: {running_metrics['rmse']:.6f}, "
                    f"F1@{config.flood_threshold:.2f}m: {running_metrics['f1']:.6f}, "
                    f"IoU@{config.flood_threshold:.2f}m: {running_metrics['iou']:.6f}"
                )

            for j in range(inputs.size(0)):
                sample_idx = i + j + 1
                preds_cpu = preds[j, 0].detach().cpu().numpy()
                targ_cpu = targets[j, 0].detach().cpu().numpy()

                h_orig, w_orig = orig_shape
                preds_cpu = preds_cpu[:h_orig, :w_orig]
                targ_cpu = targ_cpu[:h_orig, :w_orig]
                display_mask = valid_mask.astype(bool) & (~building_mask.astype(bool))
                preds_cpu[~display_mask] = np.nan
                targ_cpu[~display_mask] = np.nan

                save_comparison_plot(event_save_dir, sample_idx, preds_cpu, targ_cpu, orig_shape=orig_shape)

    event_metrics = finalize_flood_metrics(metric_sums)
    write_metric_csv(os.path.join(event_save_dir, "metrics.csv"), event_metrics)

    print(f"\nEvent {event_name} finished")
    print(f"Samples: {event_metrics['samples']}")
    print(f"MAE: {event_metrics['mae']:.6f}")
    print(f"RMSE: {event_metrics['rmse']:.6f}")
    print(f"R2: {event_metrics['r2']:.6f}")
    print(f"Bias: {event_metrics['bias']:.6f}")
    print(f"NSE: {event_metrics['nse']:.6f}")
    print(f"F1@{config.flood_threshold:.2f}m: {event_metrics['f1']:.6f}")
    print(f"IoU@{config.flood_threshold:.2f}m: {event_metrics['iou']:.6f}")

    return event_metrics, metric_sums


def main():
    parser = argparse.ArgumentParser(description="Evaluate flood model on multiple events")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "test"))
    parser.add_argument("--dem_path", type=str, default=os.path.join(os.path.dirname(__file__), "data", "static", "dem.input"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results_multi_event"))
    parser.add_argument("--train_grid_size", type=int, nargs=2, default=[560, 624])
    parser.add_argument("--rain_norm_factor", type=float, default=config.rain_norm_factor)
    args = parser.parse_args()

    device = torch.device(args.device)
    train_grid_size = tuple(args.train_grid_size)

    if args.checkpoint is None:
        checkpoint_path = os.path.join(config.model_path, f"best_model-{config.model_name}.pth.tar")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No checkpoint provided and default not found: %s" % checkpoint_path)
    else:
        checkpoint_path = args.checkpoint

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Prefer model metadata stored in the checkpoint (e.g. wide_unet widths)
    ckpt_model_name = checkpoint.get("model_name", config.model_name) if isinstance(checkpoint, dict) else config.model_name
    base_channel = checkpoint.get("base_channel") if isinstance(checkpoint, dict) else None
    bottleneck_channel = checkpoint.get("bottleneck_channel") if isinstance(checkpoint, dict) else None
    print(f"Model type: {ckpt_model_name} (base_channel={base_channel}, bottleneck_channel={bottleneck_channel})")
    model = build_model(ckpt_model_name, device, base_channel=base_channel, bottleneck_channel=bottleneck_channel)
    model.load_state_dict(clean_state_dict(checkpoint.get("state_dict", checkpoint)))
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    event_dirs = []
    for item in os.listdir(args.test_dir):
        item_path = os.path.join(args.test_dir, item)
        rain_path = os.path.join(item_path, "rain.txt")
        nc_path = os.path.join(item_path, "output.nc")
        if os.path.isdir(item_path) and os.path.exists(rain_path) and os.path.exists(nc_path):
            event_dirs.append((item, rain_path, nc_path))

    if not event_dirs:
        raise FileNotFoundError("No valid event directories found in: %s" % args.test_dir)

    overall_metric_sums = init_flood_metric_sums()
    event_results = []

    for event_name, rain_path, nc_path in event_dirs:
        event_metrics, event_metric_sums = evaluate_single_event(
            model,
            device,
            train_grid_size,
            rain_path,
            nc_path,
            args.dem_path,
            event_name,
            args.save_dir,
            args.batch_size,
            normalize=True,
            pad_multiple=16,
            rain_norm_factor=args.rain_norm_factor,
        )
        merge_flood_metric_sums(overall_metric_sums, event_metric_sums)
        event_results.append({"event": event_name, **event_metrics})

    overall_metrics = finalize_flood_metrics(overall_metric_sums)
    write_metric_csv(os.path.join(args.save_dir, "overall_metrics.csv"), overall_metrics, event_count=len(event_dirs))

    with open(os.path.join(args.save_dir, "event_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event", "MAE", "RMSE", "R2", "Bias", "NSE",
            f"F1@{config.flood_threshold:.2f}m",
            f"IoU@{config.flood_threshold:.2f}m",
            "samples",
        ])
        for result in event_results:
            writer.writerow([
                result["event"],
                f"{result['mae']:.6f}",
                f"{result['rmse']:.6f}",
                f"{result['r2']:.6f}",
                f"{result['bias']:.6f}",
                f"{result['nse']:.6f}",
                f"{result['f1']:.6f}",
                f"{result['iou']:.6f}",
                result["samples"],
            ])

    print(f"\n{'=' * 60}")
    print("Multi-Event Evaluation Finished")
    print(f"{'=' * 60}")
    print(f"Total Events: {len(event_dirs)}")
    print(f"Total Samples: {overall_metrics['samples']}")
    print(f"Overall MAE: {overall_metrics['mae']:.6f}")
    print(f"Overall RMSE: {overall_metrics['rmse']:.6f}")
    print(f"Overall R2: {overall_metrics['r2']:.6f}")
    print(f"Overall Bias: {overall_metrics['bias']:.6f}")
    print(f"Overall NSE: {overall_metrics['nse']:.6f}")
    print(f"Overall F1@{config.flood_threshold:.2f}m: {overall_metrics['f1']:.6f}")
    print(f"Overall IoU@{config.flood_threshold:.2f}m: {overall_metrics['iou']:.6f}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
