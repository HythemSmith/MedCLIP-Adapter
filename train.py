import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg') # Sử dụng backend không cần GUI
import logging

from torch.cuda.amp import GradScaler, autocas
t
from torch.optim.lr_scheduler import OneCycleLR

from MedClip_Adapter import CLIP_Swin_Implanted
from zero_shot_dataset import MedicalTrainDataset, LEVEL1_NAMES, LEVEL2_NAMES, LEVEL3_NAMES
from loss import multi_channel_focal_dice_loss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(50)

def collate_fn_skip_corrupted(batch):
    """
    Hàm collate_fn tùy chỉnh để lọc ra các mẫu bị lỗi (trả về None) từ batch.
    Điều này cần thiết khi sử dụng num_workers > 0 trên Windows.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Trả về None nếu cả batch đều bị lỗi
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    logging.basicConfig(
        filename="training_run.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    PRETRAINED_MEDCLIP_CKPT_PATH = r"D:\Thesis\MedCLIP-Adapter\checkpoints\pytorch_model.bin"
    PROMPT_CACHE_PATH = r"prompt\prompt_cache.pt"
    OUTPUT_CHECKPOINT_DIR = "checkpoints_final"
    VIS_OUTPUT_DIR = "training_visualizations" #  Thư mục lưu heatmap

    # --- Cấu hình Dataset ---
    # Sử dụng metadata.csv thay vì quét thư mục
    METADATA_PATH = "metadata.csv"
    DATASET_SOURCE_FILTER = "BTXRD_cleaned" # Chọn 'large_dataset', 'BTXRD_dataset', hoặc None để dùng tất cả

    MAX_LEARNING_RATE_ADAPTERS = 1e-4 # Tăng LR cho các thành phần mới
    MAX_LEARNING_RATE_BACKBONE = 1e-5
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 8
    EPOCHS = 50
    NUM_WORKERS = 4
    EXCLUDED_CLASS_FOR_ZERO_SHOT = "osteosarcoma"

    LOSS_WEIGHT_L1 = 0.25
    LOSS_WEIGHT_L2 = 0.5
    LOSS_WEIGHT_L3 = 1.0
    SEG_LOSS_WEIGHT = 1.0 # Tăng mạnh trọng số để ưu tiên segmentation
    GRAD_CLIP_VALUE = 1.0 # Thêm giá trị cho gradient clipping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Using device: {device}")
    logging.info(f"🔄 Using device: {device}")

    print("🚀 Initializing Zero-Shot Model (CLIP_Swin_Implanted)...")
    model = CLIP_Swin_Implanted().to(device)
    logging.info("🚀 Initializing Zero-Shot Model...")

    print(f"📦 Loading pre-trained backbone from: {PRETRAINED_MEDCLIP_CKPT_PATH}")
    state_dict = torch.load(PRETRAINED_MEDCLIP_CKPT_PATH, map_location=device)
    image_state_dict = {k.replace("vision_model.", ""): v for k, v in state_dict.items() if k.startswith("vision_model.")}
    model.backbone.load_state_dict(image_state_dict, strict=False)

    print("🧺 Freezing early backbone stages...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("✔ Backbone configured for fine-tuning.")

    print(f"🔠 Loading prompt cache from: {PROMPT_CACHE_PATH}")
    if not os.path.exists(PROMPT_CACHE_PATH):
        raise FileNotFoundError(f"'{PROMPT_CACHE_PATH}' not found.")
    prompt_cache = torch.load(PROMPT_CACHE_PATH)

    def create_text_embeddings(level_names, cache):
        embeddings = []
        for name in level_names:
            if name not in cache:
                raise KeyError(f"Class name '{name}' not found in prompt_cache.")
            emb = cache[name]["embedding"].mean(dim=0)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        return torch.stack(embeddings).to(device).detach()

    text_embeddings_l1 = create_text_embeddings(LEVEL1_NAMES, prompt_cache)
    text_embeddings_l2 = create_text_embeddings(LEVEL2_NAMES, prompt_cache)
    text_embeddings_l3 = create_text_embeddings(LEVEL3_NAMES, prompt_cache)
    print("✔ Text embeddings created for all levels.")

    print("🚀 Initializing Text Projectors for Segmentation Loss...")
    seg_feature_dims = [model.c_in_stages[i] // 2 for i in range(len(model.c_in_stages))]
    text_embedding_dim = text_embeddings_l3.shape[-1]
    text_projectors_l3 = nn.ModuleList([
        nn.Linear(text_embedding_dim, seg_dim).to(device) for seg_dim in seg_feature_dims
    ])

    print(f"📂 Loading training data from '{METADATA_PATH}', excluding class: '{EXCLUDED_CLASS_FOR_ZERO_SHOT}'")
    if DATASET_SOURCE_FILTER:
        print(f"   Filtering for dataset source: '{DATASET_SOURCE_FILTER}'")

    train_dataset = MedicalTrainDataset(
        metadata_path=METADATA_PATH,
        excluded_class=EXCLUDED_CLASS_FOR_ZERO_SHOT,
        dataset_source_filter=DATASET_SOURCE_FILTER
    )
    
    # Sử dụng collate_fn được định nghĩa ở top-level để tránh lỗi pickle với multiprocessing trên Windows.
    # Hàm này cũng sẽ bỏ qua các mẫu bị lỗi (ví dụ: không tìm thấy ảnh).
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        collate_fn=collate_fn_skip_corrupted
    )
    print(f"✔ Training data loaded with {len(train_dataset)} samples.")

    total_steps_in_epoch = len(train_loader)
    log_interval = max(1, total_steps_in_epoch // 10)
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    print(f"✍️ Logging loss details every {log_interval} steps.")

    params_to_train = [
        {"params": model.backbone.layers[-1].parameters(), "lr": MAX_LEARNING_RATE_BACKBONE},
        {"params": model.backbone.layers[-2].parameters(), "lr": MAX_LEARNING_RATE_BACKBONE},
        {"params": model.cls_adapters.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": model.seg_adapters.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": model.image_projection.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": [model.logit_scale], "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": text_projectors_l3.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS}
    ]
    optimizer = torch.optim.AdamW(params_to_train, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[group['lr'] for group in params_to_train], 
        total_steps=total_steps,
        pct_start=0.2 # Rút ngắn pha warm-up, mặc định là 0.3
    )
    print("✔ Optimizer (AdamW) and Scheduler (OneCycleLR) are set up.")

    scaler = GradScaler()
    print("✔ GradScaler for Mixed Precision Training is ready.")

    # Lấy một batch cố định để visualize
    print("📸 Preparing a fixed batch for visualization...")
    vis_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_skip_corrupted)
    fixed_vis_batch = next(iter(vis_loader))
    print("✔ Fixed visualization batch is ready.")

    def save_visualization(model, text_projectors, text_embeddings, vis_batch, epoch, device, output_dir):
        """Hàm sinh và lưu heatmap cho một batch cố định."""
        model.eval() # Chuyển sang chế độ đánh giá
        with torch.no_grad():
            # GIẢI NÉN THÊM valid_region_mask
            image, mask_gt, _, _, label3, valid_region_mask = vis_batch
            image, mask_gt, label3 = image.to(device), mask_gt.to(device), label3.to(device)
            # ĐƯA MASK LÊN DEVICE
            valid_region_mask = valid_region_mask.to(device)

            _, _, seg_intermediates = model(image)

            pred_mask_list = []
            for i, projector in enumerate(text_projectors):
                patch_tokens = seg_intermediates[i]
                text_embed_proj = projector(text_embeddings)

                patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
                text_embed_proj = text_embed_proj / text_embed_proj.norm(dim=-1, keepdim=True)

                anomaly_scores = patch_tokens @ text_embed_proj.T
                B, N, C = anomaly_scores.shape
                H_feat = W_feat = int(N ** 0.5)
                anomaly_scores = anomaly_scores.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
                heatmap_pred = F.interpolate(anomaly_scores, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                pred_mask_list.append(heatmap_pred)

            # Lấy trung bình các heatmap từ các tầng
            avg_heatmap = torch.stack(pred_mask_list).mean(dim=0).sigmoid()

            # <<< THAY ĐỔI QUAN TRỌNG >>>
            # Áp dụng valid_region_mask vào heatmap trước khi hiển thị
            # unsqueeze(1) để broadcast shape từ [B, H, W] thành [B, 1, H, W]
            masked_avg_heatmap = avg_heatmap * valid_region_mask.unsqueeze(1)

            # Vẽ và lưu
            num_images = image.shape[0]
            fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
            fig.suptitle(f'Epoch {epoch+1} Visualization', fontsize=16)

            for i in range(num_images):
                img_display = image[i].cpu().permute(1, 2, 0).numpy()
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

                true_class_idx = label3[i].nonzero(as_tuple=True)[0]
                if len(true_class_idx) > 0:
                    idx_to_show = true_class_idx[0]
                    class_name = LEVEL3_NAMES[idx_to_show]
                    gt_mask_display = mask_gt[i, idx_to_show].cpu().numpy()
                    # SỬ DỤNG HEATMAP ĐÃ ĐƯỢC MASK
                    pred_heatmap_display = masked_avg_heatmap[i, idx_to_show].cpu().numpy()
                else:
                    idx_to_show = LEVEL3_NAMES.index('normal')
                    class_name = 'normal (no GT mask)'
                    gt_mask_display = np.zeros_like(mask_gt[i, 0].cpu().numpy())
                    # SỬ DỤNG HEATMAP ĐÃ ĐƯỢC MASK
                    pred_heatmap_display = masked_avg_heatmap[i, idx_to_show].cpu().numpy()

                axes[i, 0].imshow(img_display)
                axes[i, 0].set_title("Original Image")
                axes[i, 1].imshow(gt_mask_display, cmap='gray')
                axes[i, 1].set_title(f"Ground Truth: {class_name}")
                axes[i, 2].imshow(pred_heatmap_display, cmap='jet')
                axes[i, 2].set_title(f"Predicted Heatmap: {class_name}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(output_dir, f"epoch_{epoch+1:03d}.png")
            plt.savefig(save_path)
            plt.close(fig)

    print("\n🔥 Starting training loop...")
    best_loss = float('inf')
    os.makedirs(OUTPUT_CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch_data in enumerate(pbar):
            # Bỏ qua batch nếu nó hoàn toàn rỗng (tất cả mẫu đều bị lỗi)
            if batch_data is None:
                logging.warning(f"Skipping empty batch at step {step+1} in epoch {epoch+1}.")
                continue
            # Nhận thêm valid_region_mask từ loader
            image, mask_gt, label1, label2, label3, valid_region_mask = batch_data
            image, mask_gt = image.to(device), mask_gt.to(device)
            label1, label2, label3 = label1.to(device), label2.to(device), label3.to(device)
            # Đưa valid_region_mask lên device
            valid_region_mask = valid_region_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

             with autocast():
                # ... (phần tính logits không đổi)
                clf_loss = (LOSS_WEIGHT_L1 * loss1) + (LOSS_WEIGHT_L2 * loss2) + (LOSS_WEIGHT_L3 * loss3)
                
                # --- Segmentation Loss Calculation ---
                active_class_idxs = (label3.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
                
                seg_loss_val = torch.tensor(0.0, device=device)
                
                if active_class_idxs.numel() > 0:
                    total_bce_seg, total_dice_seg = 0.0, 0.0
                    num_stages = len(text_projectors_l3)
                    
                    for i, projector in enumerate(text_projectors_l3):
                        # ... (phần code tính heatmap_pred giữ nguyên)
                        patch_tokens = seg_intermediates[i]
                        # ...
                        heatmap_pred = F.interpolate(anomaly_scores, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                        
                        masked_heatmap_pred = heatmap_pred * valid_region_mask.unsqueeze(1)
                        masked_mask_gt = mask_gt[:, active_class_idxs] * valid_region_mask.unsqueeze(1)
                        
                        # <<< THAY ĐỔI CÁCH GỌI HÀM LOSS >>>
                        bce_seg, dice_seg = multi_channel_focal_dice_loss(masked_heatmap_pred, masked_mask_gt)
                        
                        total_bce_seg += bce_seg
                        total_dice_seg += dice_seg
                    
                    # Lấy trung bình loss từ các stage
                    avg_bce_seg = total_bce_seg / num_stages
                    avg_dice_seg = total_dice_seg / num_stages

                    # Kết hợp chúng lại (bạn có thể thêm trọng số cho dice nếu muốn, ví dụ: 0.5 * avg_dice_seg)
                    seg_loss_val = avg_bce_seg + avg_dice_seg

                total_loss = clf_loss + SEG_LOSS_WEIGHT * seg_loss_val
             # Thêm kiểm tra loss âm để đảm bảo an toàn
            if torch.isnan(total_loss) or (total_loss.is_finite() and total_loss.item() < 0):
                logging.warning(f"Invalid loss detected (NaN or Negative: {total_loss.item()}). Skipping step.")
                continue
            if torch.isnan(total_loss):
                print("⚠️ NaN loss encountered! Skipping step.")
                continue

            scaler.scale(total_loss).backward()
            # Unscale gradients before clipping
            # Áp dụng clipping cho TẤT CẢ các tham số có thể huấn luyện
            scaler.unscale_(optimizer)
            all_trainable_params = []
            for param_group in params_to_train:
                all_trainable_params.extend(param_group['params'])
            torch.nn.utils.clip_grad_norm_(all_trainable_params, GRAD_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += total_loss.item()
            pbar.set_postfix({"loss": total_loss.item(), "lr": scheduler.get_last_lr()[0]})

            if (step + 1) % log_interval == 0 or (step + 1) == total_steps_in_epoch:
                logging.info(
                    f"Epoch {epoch+1} | Step [{step+1}/{total_steps_in_epoch}] | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"CLF Loss: {clf_loss.item():.4f} | "
                    f"SEG Loss: {(SEG_LOSS_WEIGHT * seg_loss).item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch+1} finished. Avg Train Loss: {avg_loss:.4f}")
        logging.info(f"--- Epoch {epoch+1} Summary | Avg Train Loss: {avg_loss:.4f} ---")

        # --- Lưu hình ảnh visualization ---
        try:
            save_visualization(model, text_projectors_l3, text_embeddings_l3, fixed_vis_batch, epoch, device, VIS_OUTPUT_DIR)
            logging.info(f"🖼️  Saved visualization for epoch {epoch+1} to '{VIS_OUTPUT_DIR}'")
        except Exception as e:
            logging.error(f"Could not save visualization for epoch {epoch+1}: {e}")

        # ------------------------------------

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(OUTPUT_CHECKPOINT_DIR, "model_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'text_projectors_state_dict': text_projectors_l3.state_dict(),
                'epoch': epoch + 1
            }, save_path)
            print(f"💾 Best model saved to {save_path} (loss: {best_loss:.4f})")
            logging.info(f"💾 Best model saved to {save_path}")

    final_save_path = os.path.join(OUTPUT_CHECKPOINT_DIR, "model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_projectors_state_dict': text_projectors_l3.state_dict(),
        'epoch': EPOCHS
    }, final_save_path)
    print(f"💾 Final model (with projectors) saved to {final_save_path}")
    print("\n🎉 Training finished!")

if __name__ == '__main__':
    main()
