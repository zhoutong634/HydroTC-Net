import math

import torch
from torch import nn
import torch.nn.functional as F


def init_flood_metric_sums():
    return {
        'abs_error_sum': 0.0,
        'sq_error_sum': 0.0,
        'error_sum': 0.0,
        'target_sum': 0.0,
        'target_sq_sum': 0.0,
        'pred_sum': 0.0,
        'pred_sq_sum': 0.0,
        'pred_target_sum': 0.0,
        'value_count': 0,
        'sample_count': 0,
        'tp': 0.0,
        'fp': 0.0,
        'fn': 0.0,
    }


def update_flood_metric_sums(metric_sums, targets, preds, threshold=0.05, mask=None):
    with torch.no_grad():
        targets = targets.detach().float()
        preds = preds.detach().float()
        batch_size = targets.shape[0]

        if mask is not None:
            mask = mask.detach().to(device=targets.device).bool()
            while mask.ndim < targets.ndim:
                mask = mask.unsqueeze(0)
            mask = torch.broadcast_to(mask, targets.shape)
            if not torch.any(mask):
                return
            targets = targets[mask]
            preds = preds[mask]

        error = preds - targets

        metric_sums['abs_error_sum'] += torch.sum(torch.abs(error)).item()
        metric_sums['sq_error_sum'] += torch.sum(error * error).item()
        metric_sums['error_sum'] += torch.sum(error).item()
        metric_sums['target_sum'] += torch.sum(targets).item()
        metric_sums['target_sq_sum'] += torch.sum(targets * targets).item()
        metric_sums['pred_sum'] += torch.sum(preds).item()
        metric_sums['pred_sq_sum'] += torch.sum(preds * preds).item()
        metric_sums['pred_target_sum'] += torch.sum(preds * targets).item()
        metric_sums['value_count'] += targets.numel()
        metric_sums['sample_count'] += batch_size

        target_flood = targets >= threshold
        pred_flood = preds >= threshold
        metric_sums['tp'] += torch.sum(target_flood & pred_flood).item()
        metric_sums['fp'] += torch.sum(~target_flood & pred_flood).item()
        metric_sums['fn'] += torch.sum(target_flood & ~pred_flood).item()


def merge_flood_metric_sums(dst, src):
    for key in dst:
        dst[key] += src[key]
    return dst


def finalize_flood_metrics(metric_sums):
    eps = 1e-8
    count = max(metric_sums['value_count'], 1)

    mae = metric_sums['abs_error_sum'] / count
    rmse = math.sqrt(metric_sums['sq_error_sum'] / count)
    bias = metric_sums['error_sum'] / count

    target_sum = metric_sums['target_sum']
    pred_sum = metric_sums['pred_sum']
    target_sq_sum = metric_sums['target_sq_sum']
    pred_sq_sum = metric_sums['pred_sq_sum']
    pred_target_sum = metric_sums['pred_target_sum']

    target_ss_tot = target_sq_sum - (target_sum * target_sum) / count
    nse = 1.0 - (metric_sums['sq_error_sum'] / (target_ss_tot + eps))

    covariance = pred_target_sum - (pred_sum * target_sum) / count
    pred_var = pred_sq_sum - (pred_sum * pred_sum) / count
    target_var = target_ss_tot
    r2 = (covariance * covariance) / (max(pred_var * target_var, 0.0) + eps)

    precision = metric_sums['tp'] / (metric_sums['tp'] + metric_sums['fp'] + eps)
    recall = metric_sums['tp'] / (metric_sums['tp'] + metric_sums['fn'] + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = metric_sums['tp'] / (metric_sums['tp'] + metric_sums['fp'] + metric_sums['fn'] + eps)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'nse': nse,
        'f1': f1,
        'iou': iou,
        'samples': metric_sums['sample_count'],
    }


def compute_flood_metrics(targets, preds, threshold=0.05, mask=None):
    metric_sums = init_flood_metric_sums()
    update_flood_metric_sums(metric_sums, targets, preds, threshold=threshold, mask=mask)
    return finalize_flood_metrics(metric_sums)


class FloodCompositeLoss(nn.Module):
    def __init__(
        self,
        threshold=0.05,
        temperature=30.0,
        depth_alpha=2.0,
        depth_beta=2.0,
        underestimate_weight=1.0,
        depth_y_max=10.0,
        smooth_l1_beta=1.0,
        depth_weight=1.0,
        flood_bce_weight=0.5,
        flood_dice_weight=0.5,
        gradient_weight=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.depth_alpha = depth_alpha
        self.depth_beta = depth_beta
        self.underestimate_weight = underestimate_weight
        self.depth_y_max = max(depth_y_max, eps)
        self.smooth_l1_beta = smooth_l1_beta
        self.depth_weight = depth_weight
        self.flood_bce_weight = flood_bce_weight
        self.flood_dice_weight = flood_dice_weight
        self.gradient_weight = gradient_weight
        self.eps = eps
        self.latest_components = {}

    def _prepare_mask(self, mask, targets):
        if mask is None:
            return None
        mask = mask.to(device=targets.device).bool()
        while mask.ndim < targets.ndim:
            mask = mask.unsqueeze(0)
        return torch.broadcast_to(mask, targets.shape)

    def _masked_mean(self, values, mask):
        if mask is None:
            return values.mean()
        if not torch.any(mask):
            return values.sum() * 0.0
        return values[mask].mean()

    def _weighted_smooth_l1(self, preds, targets, mask=None):
        diff = torch.abs(preds - targets)
        if self.smooth_l1_beta < self.eps:
            smooth_l1 = diff
        else:
            smooth_l1 = torch.where(
                diff < self.smooth_l1_beta,
                0.5 * diff * diff / self.smooth_l1_beta,
                diff - 0.5 * self.smooth_l1_beta,
            )

        flood_mask = (targets >= self.threshold).float()
        depth_scale = torch.clamp(targets / self.depth_y_max, 0.0, 1.0)
        weights = 1.0 + self.depth_alpha * flood_mask + self.depth_beta * depth_scale
        under_mask = ((preds < targets) & (targets >= self.threshold)).float()
        weights = weights * (1.0 + self.underestimate_weight * under_mask)
        return self._masked_mean(weights * smooth_l1, mask)

    def _soft_flood_probs(self, preds):
        return torch.sigmoid(self.temperature * (preds - self.threshold))

    def _dice_loss(self, probs, target_mask, mask=None):
        if mask is not None:
            mask = mask.float()
            probs = probs * mask
            target_mask = target_mask * mask
        probs = probs.reshape(probs.shape[0], -1)
        target_mask = target_mask.reshape(target_mask.shape[0], -1)
        intersection = (probs * target_mask).sum(dim=1)
        denominator = probs.sum(dim=1) + target_mask.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()

    def _gradient_loss(self, preds, targets, mask=None):
        pred_dx = preds[:, :, :, 1:] - preds[:, :, :, :-1]
        target_dx = targets[:, :, :, 1:] - targets[:, :, :, :-1]
        pred_dy = preds[:, :, 1:, :] - preds[:, :, :-1, :]
        target_dy = targets[:, :, 1:, :] - targets[:, :, :-1, :]
        loss_x_values = torch.abs(pred_dx - target_dx)
        loss_y_values = torch.abs(pred_dy - target_dy)

        if mask is None:
            loss_x = loss_x_values.mean()
            loss_y = loss_y_values.mean()
        else:
            mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
            mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
            loss_x = self._masked_mean(loss_x_values, mask_x)
            loss_y = self._masked_mean(loss_y_values, mask_y)
        return loss_x + loss_y

    def forward(self, preds, targets, mask=None):
        preds = preds.float()
        targets = targets.float()
        mask = self._prepare_mask(mask, targets)

        depth_loss = self._weighted_smooth_l1(preds, targets, mask)

        target_mask = (targets >= self.threshold).float()
        flood_probs = self._soft_flood_probs(preds)
        flood_bce_values = F.binary_cross_entropy_with_logits(
            self.temperature * (preds - self.threshold),
            target_mask,
            reduction='none',
        )
        flood_bce = self._masked_mean(flood_bce_values, mask)
        flood_dice = self._dice_loss(flood_probs, target_mask, mask)

        gradient_loss = self._gradient_loss(preds, targets, mask)

        total_loss = (
            self.depth_weight * depth_loss
            + self.flood_bce_weight * flood_bce
            + self.flood_dice_weight * flood_dice
            + self.gradient_weight * gradient_loss
        )

        self.latest_components = {
            'depth': depth_loss.detach().item(),
            'flood_bce': flood_bce.detach().item(),
            'flood_dice': flood_dice.detach().item(),
            'gradient': gradient_loss.detach().item(),
            'total': total_loss.detach().item(),
        }
        return total_loss

    def get_latest_components(self):
        return dict(self.latest_components)
