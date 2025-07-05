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
def multi_channel_focal_dice_loss(pred, target):
    """
    Phiên bản kết hợp BCE và Dice Loss ổn định về mặt số học.
    Hàm này tính toán Dice loss trên từng mẫu và chỉ lấy trung bình
    các giá trị loss hợp lệ (nơi có ground truth mask), ngăn ngừa mất ổn định.
    """
    # 1. Loss theo từng pixel (BCE)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)

    # 2. Loss theo hình dạng (Dice - Phiên bản ổn định)
    pred_probs = torch.sigmoid(pred)
    
    intersection = (pred_probs * target).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Dice score cho từng mẫu trong batch và từng class
    dice_score = (2. * intersection + 1e-6) / (union + 1e-6)

    # <<< THAY ĐỔI DUY NHẤT VÀ QUAN TRỌNG NHẤT Ở ĐÂY >>>
    # Kẹp giá trị dice_score để đảm bảo nó luôn nằm trong khoảng [0, 1]
    # Điều này ngăn chặn lỗi làm tròn gây ra score > 1.
    dice_score = torch.clamp(dice_score, 0.0, 1.0)
    
    # Bây giờ, dice_loss chắc chắn sẽ không bao giờ âm
    dice_loss_per_sample = 1. - dice_score

    # Chỉ tính loss trên các mẫu có mask (target.sum > 0)
    mask_has_gt = (target.sum(dim=(2, 3)) > 0)

    if mask_has_gt.sum() > 0:
        # Lấy trung bình của các giá trị loss HỢP LỆ
        dice_loss = dice_loss_per_sample[mask_has_gt].mean()
    else:
        dice_loss = torch.tensor(0.0, device=pred.device)

    # Trả về 2 thành phần loss riêng biệt
    return bce_loss, dice_loss