# -*- coding: utf-8 -*-
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from tensorboardX import SummaryWriter
from flood_dataset import MultiEventFloodDataset
import logging

# ===============================

# 导入你的模块
# ===============================
from nets.UNet import UNet
from nets.HydroTC_Net import HydroTC_Net
from Train_one_epoch import train_one_epoch
from utils import FloodCompositeLoss
import Config as config

# ===============================
# 基础设置
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ===============================
# 日志配置
# ===============================
def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

# ===============================
# 固定随机种子
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===============================
# 验证
# ===============================
def validate(loader, model, criterion, writer, epoch, model_type):
    """
    验证函数，使用Train_one_epoch中的train_one_epoch函数
    """
    # 设置模型为评估模式
    model.eval()
    
    # 调用Train_one_epoch中的train_one_epoch函数进行验证
    val_loss, val_metrics = train_one_epoch(
        model,
        loader, 
        None, 
        criterion,   # 验证时不需要optimizer
        writer, 
        epoch, 
        None,  # 验证时不需要lr_scheduler
        model_type, 
        logger
    )
    
    return val_loss, val_metrics


def format_metric_line(metrics):
    return (
        f"MAE: {metrics['mae']:.6f}, "
        f"RMSE: {metrics['rmse']:.6f}, "
        f"R2: {metrics['r2']:.6f}, "
        f"Bias: {metrics['bias']:.6f}, "
        f"NSE: {metrics['nse']:.6f}, "
        f"F1@{config.flood_threshold:.2f}m: {metrics['f1']:.6f}, "
        f"IoU@{config.flood_threshold:.2f}m: {metrics['iou']:.6f}"
    )


def format_duration(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ===============================
# 保存检查点
# ===============================
def save_checkpoint(state, save_path):
    """
    保存当前模型状态
    如果是最佳模型，则保存为best_model
    """
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + 'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

# ===============================
# 主函数
# ===============================
def main():
    train_start_time = time.time()

    set_seed(42)

    # ===============================
    # 数据集
    # ===============================
    # 导入洪涝数据集类
    # 创建训练和验证数据集
    # 注意：需要根据实际数据位置修改路径
    train_dataset = MultiEventFloodDataset(
        root_dir=config.train_dataset,
        dem_path=config.dem_path,
        normalize=True,
        pad_multiple=config.pad_multiple,
        rain_norm_factor=config.rain_norm_factor
    )

    val_dataset = MultiEventFloodDataset(
        root_dir=config.val_dataset,
        dem_path=config.dem_path,
        normalize=True,
        pad_multiple=config.pad_multiple,
        rain_norm_factor=config.rain_norm_factor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # ===============================
    # 模型选择
    # ===============================
    model_type = config.model_name

    if model_type == "UNet":
        model = UNet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == "HydroTC_Net":
        config_vit = config.get_CTranS_config()
        # Lujiazui DEM: 549 rows x 610 cols, padded to 560x624 (multiples of 16)
        # Transformer部分接收的是下采样后的特征图，但位置嵌入是基于原始输入尺寸计算的
        img_size = config.img_size
        model = HydroTC_Net(config_vit,
                           n_channels=config.n_channels,
                           n_classes=config.n_labels,
                           img_size=img_size)

    else:
        raise ValueError("Unsupported model type")

    model = model.to(DEVICE)

    # ===============================
    # 损失函数（回归）
    # ===============================
    criterion = FloodCompositeLoss(
        threshold=config.flood_threshold,
        temperature=config.loss_extent_temperature,
        depth_alpha=config.loss_depth_alpha,
        depth_beta=config.loss_depth_beta,
        underestimate_weight=config.loss_underestimate_weight,
        depth_y_max=config.max_water_depth,
        smooth_l1_beta=config.loss_smooth_l1_beta,
        depth_weight=config.loss_depth_weight,
        flood_bce_weight=config.loss_flood_bce_weight,
        flood_dice_weight=config.loss_flood_dice_weight,
        gradient_weight=config.loss_gradient_weight,
    )

    # ===============================
    # 优化器
    # ===============================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5  # 添加L2正则化，防止过拟合
    )

    # ===============================
    # 训练循环
    # ===============================
    best_val_loss = float("inf")
    best_epoch = 1
    epochs = config.epochs
    early_stopping_patience = config.early_stopping_patience

    # 创建保存目录
    checkpoint_dir = config.model_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # TensorBoard日志目录
    log_dir = config.tensorboard_folder
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 训练日志
    logger_path = config.logger_path
    global logger
    logger = logger_config(logger_path)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Training on device: {DEVICE}")
    logger.info(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_time))}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Rain normalization factor: {config.rain_norm_factor:.3f} mm/h")
    logger.info(
        "Loss weights: depth=%.2f, flood_bce=%.2f, flood_dice=%.2f, gradient=%.2f, underestimate=%.2f",
        config.loss_depth_weight,
        config.loss_flood_bce_weight,
        config.loss_flood_dice_weight,
        config.loss_gradient_weight,
        config.loss_underestimate_weight,
    )

    # 学习率调度器
    if config.cosineLR:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)
    else:
        lr_scheduler = None

    for epoch in range(epochs):

        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, epochs))

        # 训练一个epoch
        model.train(True)
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            writer,
            epoch,
            lr_scheduler,
            model_type,
            logger
        )

        # 验证
        val_loss, val_metrics = validate(
            val_loader,
            model,
            criterion,
            writer,
            epoch,
            model_type
        )

        logger.info(f"Train Loss: {train_loss:.6f}, {format_metric_line(train_metrics)}")
        logger.info(f"Val Loss:   {val_loss:.6f}, {format_metric_line(val_metrics)}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            if epoch + 1 > 5:  # 至少训练5个epoch后才保存最佳模型
                logger.info('\t Saving best model, val loss decreased from: {:.4f} to {:.4f}'.format(best_val_loss, val_loss))
                best_val_loss = val_loss
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch,
                    'best_model': True,
                    'model': model_type,
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict()
                }, checkpoint_dir)
        else:
            logger.info('\t Val loss:{:.4f} does not decrease, the best is still: {:.4f} in epoch {}'.format(val_loss, best_val_loss, best_epoch))
        
        # 早停机制
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, early_stopping_patience))

        if early_stopping_count > early_stopping_patience:
            logger.info('\t Early stopping!')
            break

    logger.info("🎉 Training Finished")
    total_training_time = time.time() - train_start_time
    logger.info("Training Finished")
    logger.info(f"Total training time: {format_duration(total_training_time)}")
    writer.close()


if __name__ == "__main__":
    main()
