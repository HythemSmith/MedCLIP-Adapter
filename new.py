import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from MedClip_Adapter import CLIP_Swin_Implanted
from zero_shot_dataset import LEVEL3_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== CONFIG ====
MODEL_PATH = "checkpoints_BTXRD_Cleaned/model_best.pth"
PROMPT_CACHE = "prompt/prompt_cache.pt"
IMAGE_PATH = r"C:\Users\vanlo\Desktop\organized_cleaned\bone_tumor\benign\multiple_osteochondromas\images\IMG000456.jpeg"  # <- THAY ĐƯỜNG DẪN Ở ĐÂY
RESIZE = 224
TOPK = 3

# ==== LOAD MODEL ====
model = CLIP_Swin_Implanted().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ==== LOAD TEXT EMBEDDINGS ====
prompt_cache = torch.load(PROMPT_CACHE)
text_embeddings_l3 = []
for name in LEVEL3_NAMES:
    emb = prompt_cache[name]["embedding"].mean(dim=0)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    text_embeddings_l3.append(emb)
text_embeddings_l3 = torch.stack(text_embeddings_l3).to(device)

# ==== TEXT PROJECTORS ====
seg_feature_dims = [model.c_in_stages[i] // 2 for i in range(len(model.c_in_stages))]
text_projectors = torch.nn.ModuleList([
    torch.nn.Linear(text_embeddings_l3.shape[-1], dim).to(device)
    for dim in seg_feature_dims
])
text_projectors.load_state_dict(ckpt['text_projectors_state_dict'])

# ==== LOAD IMAGE ====
transform_x = T.Compose([
    T.Resize(RESIZE, interpolation=Image.BICUBIC),
    T.CenterCrop((RESIZE, RESIZE)),  # Hoặc dùng transform pad nếu cần
    T.ToTensor()
])
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform_x(image).unsqueeze(0).to(device)

# ==== FORWARD PASS ====
with torch.no_grad():
    _, _, seg_intermediates = model(image_tensor)
    pred_mask_list = []
    for i, projector in enumerate(text_projectors):
        patch_tokens = seg_intermediates[i]  # [1, N, D]
        text_proj = projector(text_embeddings_l3)  # [C, D]

        patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
        text_proj = text_proj / text_proj.norm(dim=-1, keepdim=True)

        similarity = patch_tokens @ text_proj.T  # [1, N, C]
        B, N, C = similarity.shape
        H_feat = W_feat = int(N ** 0.5)

        heatmap = similarity.permute(0, 2, 1).reshape(1, C, H_feat, W_feat)
        upsampled = torch.nn.functional.interpolate(heatmap, size=(RESIZE, RESIZE), mode="bilinear", align_corners=False)
        pred_mask_list.append(upsampled)

    # Averaging multi-scale predictions
    pred_heatmap = torch.stack(pred_mask_list).mean(dim=0).squeeze(0)  # [C, H, W]

# ==== SHOW TOP-K PREDICTED SEGMENTATION MAPS ====
prob_map = pred_heatmap.sigmoid()
mean_score = prob_map.view(prob_map.shape[0], -1).mean(dim=-1)
topk = torch.topk(mean_score, k=TOPK)

fig, axes = plt.subplots(1, TOPK + 1, figsize=(5*(TOPK+1), 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

for i, class_idx in enumerate(topk.indices):
    heat = prob_map[class_idx].cpu().numpy()
    axes[i+1].imshow(image)
    axes[i+1].imshow(heat, cmap='jet', alpha=0.5)
    axes[i+1].set_title(f"{LEVEL3_NAMES[class_idx]} ({mean_score[class_idx]:.2f})")
    axes[i+1].axis("off")

plt.tight_layout()
plt.show()
