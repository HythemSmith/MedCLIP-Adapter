import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Focal Loss ---
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

# --- Multi-channel BCE + Dice for Segmentation ---
def multi_channel_focal_dice_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    A combination of Focal Loss and Dice Loss for multi-label segmentation.
    pred, target: [B, C, H, W] where C = number of classes (18)
    """
    # --- Focal Loss Calculation ---
    # This is the correct implementation for multi-label segmentation
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pred_probs_for_focal = torch.sigmoid(pred)
    p_t = pred_probs_for_focal * target + (1 - pred_probs_for_focal) * (1 - target)
    focal_modulator = (1 - p_t)**gamma
    
    # Apply alpha weighting
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_loss = (alpha_t * focal_modulator * bce_loss).mean()

    # --- Dice Loss Calculation ---
    pred_probs = torch.sigmoid(pred)
    target_sum = target.sum(dim=[2, 3])
    has_mask = target_sum > 0
    intersection = (pred_probs * target).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_all = (2. * intersection + 1e-5) / (union + 1e-5)
    if has_mask.any():
        dice_loss = 1 - dice_all[has_mask].mean()
    else:
        # Nếu không có mask nào trong batch này, Dice loss là 0
        dice_loss = 0.0

    return focal_loss + dice_loss
