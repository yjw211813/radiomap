import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image import image_gradients


class MixedLoss_SSIM_MSE_Grad(nn.Module):
    def __init__(self, ssim_weight=0.05, mse_weight=1.0, grad_weight=1.0, l1_weight=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.l1_weight = l1_weight

        # 初始化损失函数和指标
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        """
        计算混合损失
        Args:
            pred: 模型预测输出
            target: 真实标签
        Returns:
            total_loss: 最终的混合损失
            loss_components: 包含所有子损失的字典
        """
        # 梯度损失 (GDL)
        target_dy, target_dx = image_gradients(target)
        pred_dy, pred_dx = image_gradients(pred)
        grad_loss = (torch.abs(pred_dy - target_dy) + torch.abs(pred_dx - target_dx)).sum() / pred.numel()

        # SSIM损失
        ssim_loss = 1.0 - structural_similarity_index_measure(
            pred,
            target,
            data_range=(0,1)
        )

        # L1损失
        l1_loss = self.l1_loss(pred, target)

        # MSE损失
        mse_loss = self.mse_loss(pred, target)

        # 计算总损失
        total_loss = self.ssim_weight * ssim_loss + self.l1_weight * l1_loss + self.grad_weight * grad_loss

        # 整理所有子损失
        loss_components = {
            'total_loss': total_loss,
            'grad_loss': self.grad_weight * grad_loss,
            'ssim_loss': self.ssim_weight * ssim_loss,
            'l1_loss': self.l1_weight * l1_loss,
            'mse_loss': self.mse_weight * mse_loss
        }

        return total_loss, loss_components


class ScaleAwareMixedLoss(nn.Module):
    def __init__(
        self,
        mse_w=1.0,
        ssim_w=0.0,
        grad_w=0.0,
        data_range=(0,1)
    ):
        super().__init__()
        self.mse_w = mse_w
        self.ssim_w = ssim_w
        self.grad_w = grad_w
        self.data_range = data_range
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = 0.0
        comps = {}

        # -------- MSE (all scales) --------
        mse = self.mse(pred, target)
        loss += self.mse_w * mse
        comps["mse"] = mse.detach()

        # -------- SSIM (mid / high scales) --------
        if self.ssim_w > 0:
            ssim = 1.0 - structural_similarity_index_measure(
                pred, target, data_range=self.data_range
            )
            loss += self.ssim_w * ssim
            comps["ssim"] = ssim.detach()

        # -------- Gradient loss (high scales only) --------
        if self.grad_w > 0:
            ty, tx = image_gradients(target)
            py, px = image_gradients(pred)
            grad = (torch.abs(py - ty) + torch.abs(px - tx)).mean()
            loss += self.grad_w * grad
            comps["grad"] = grad.detach()

        return loss, comps


def build_scale_losses():
    return {
        "32": ScaleAwareMixedLoss(
            mse_w=1.0,
            ssim_w=0.0,
            grad_w=0.0
        ),
        "64": ScaleAwareMixedLoss(
            mse_w=1.0,
            ssim_w=0.2,
            grad_w=0.0
        ),
        "128": ScaleAwareMixedLoss(
            mse_w=1.0,
            ssim_w=0.5,
            grad_w=0.2
        ),
        "256": ScaleAwareMixedLoss(
            mse_w=1.0,
            ssim_w=1.0,
            grad_w=1.0
        ),
    }


class MultiScalePhysicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_losses = build_scale_losses()
        self.scale_weights = {
            "32":  0.1,
            "64":  0.2,
            "128": 0.5,
            "256": 1.0
        }

    def forward(self, preds, target):
        targets = {
            "256": target,
            "128": F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False),
            "64":  F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False),
            "32":  F.interpolate(target, scale_factor=0.125, mode="bilinear", align_corners=False),
        }

        total = 0.0
        logs = {}

        for k in ["32", "64", "128", "256"]:
            loss, comps = self.scale_losses[k](preds[k], targets[k])
            total += self.scale_weights[k] * loss

            logs[f"loss_{k}"] = loss.detach()
            for ck, cv in comps.items():
                logs[f"{ck}_{k}"] = cv

        logs["total_loss"] = total
        return total, logs
