# loss.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Focal Loss ---
# (Phần FocalLoss giữ nguyên, không thay đổi)
class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)

        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1).view(-1, 1)

        if self.alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            alpha = torch.FloatTensor(self.alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(num_class, 1) * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Unsupported alpha type')

        alpha = alpha.to(logit.device)
        idx = target.cpu().long()

        one_hot = torch.zeros(target.size(0), num_class).scatter_(1, idx, 1).to(logit.device)
        if self.smooth:
            one_hot = torch.clamp(one_hot, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx].squeeze()
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        return loss.mean() if self.size_average else loss.sum()

# --- Dice Loss ---
# (Phần BinaryDiceLoss giữ nguyên, không thay đổi)
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        B = target.size(0)
        input_flat = input.view(B, -1)
        target_flat = target.view(B, -1)
        intersection = input_flat * target_flat
        dice = (2 * intersection.sum(1) + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        return 1 - dice.mean()


# --- Multi-channel BCE + Dice for Segmentation (ĐÃ CẬP NHẬT) ---
def focal_dice_loss(pred_logits, target, valid_region_mask, gamma=2.0, alpha=0.25):
    """
    Kết hợp Focal Loss và Dice Loss, được triển khai đúng cho bài toán multi-label segmentation.
    - pred_logits: Đầu ra của model, CHƯA qua sigmoid.
    """
    # --- 1. Tính Focal Loss (chỉ trên vùng hợp lệ) ---
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    
    # Tính pt
    pred_probs = torch.sigmoid(pred_logits)
    p_t = pred_probs * target + (1 - pred_probs) * (1 - target)
    
    # Tính modulating factor
    modulating_factor = (1.0 - p_t).pow(gamma)
    
    # Tính alpha weight
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    focal_loss_pixelwise = alpha_t * modulating_factor * bce_loss
    
    # Áp dụng valid_region_mask
    focal_loss_masked = focal_loss_pixelwise * valid_region_mask.unsqueeze(1)
    
    # Lấy trung bình trên toàn bộ các pixel hợp lệ trong batch
    # Thêm 1e-6 để tránh chia cho 0
    focal_loss = focal_loss_masked.sum() / (valid_region_mask.sum() * target.shape[1] + 1e-6)

    # --- 2. Tính Dice Loss (giữ nguyên, đã đúng) ---
    pred_probs_masked = pred_probs * valid_region_mask.unsqueeze(1)
    target_masked = target * valid_region_mask.unsqueeze(1)
    
    intersection = (pred_probs_masked * target_masked).sum(dim=(2, 3))
    union = pred_probs_masked.sum(dim=(2, 3)) + target_masked.sum(dim=(2, 3))
    
    dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
    
    mask_has_gt = (target_masked.sum(dim=(2, 3)) > 0)
    if mask_has_gt.sum() > 0:
        dice_loss = (1. - dice_score[mask_has_gt]).mean()
    else:
        dice_loss = torch.tensor(0.0, device=pred_logits.device)

    return focal_loss, dice_loss