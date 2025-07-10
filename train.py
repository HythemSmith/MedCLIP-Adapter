import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # Sử dụng backend không cần GUI
import logging

from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from MedClip_Adapter import CLIP_Swin_Implanted
from zero_shot_dataset import ZeroShotTrainDataset, LEVEL1_NAMES, LEVEL2_NAMES, LEVEL3_NAMES, POSITION_NAMES
# <<< THAY ĐỔI: Chỉ import hàm focal_dice_loss >>>
from loss import focal_dice_loss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(45)

def collate_fn_skip_corrupted(batch):
    """Hàm collate_fn tùy chỉnh để lọc ra các mẫu bị lỗi (trả về None) từ batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    logging.basicConfig(
        filename="training_run_final.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # --- Cấu hình ---
    PRETRAINED_MEDCLIP_CKPT_PATH = r"checkpoints\pytorch_model.bin"
    PROMPT_CACHE_PATH = r"prompt\prompt_cache_structured.pt"
    OUTPUT_CHECKPOINT_DIR = "checkpoints_final"
    VIS_OUTPUT_DIR = "training_visualizations"
    DATASET_ROOT_DIR = r"C:\Users\vanlo\Desktop\BTXRD_cleaned"
    LOAD_CHECKPOINT_PATH = r"D:\Thesis\MedCLIP-Adapter\checkpoints_final\model_best.pth" # Để rỗng nếu muốn train từ đầu

    # --- Hyperparameters ---
    MAX_LEARNING_RATE_ADAPTERS = 1e-4
    MAX_LEARNING_RATE_BACKBONE = 1e-5
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 12
    EPOCHS = 50
    NUM_WORKERS = 4
    EXCLUDED_CLASS_FOR_ZERO_SHOT = "osteosarcoma"

    LOSS_WEIGHT_L1 = 0.25
    LOSS_WEIGHT_L2 = 0.5
    LOSS_WEIGHT_L3 = 1.0
    LOSS_WEIGHT_POS = 0.5
    # <<< THAY ĐỔI QUAN TRỌNG 1: Tăng mạnh trọng số loss phân đoạn >>>
    # Giá trị 5.0 quá thấp, khiến mô hình "lờ đi" loss phân đoạn.
    # Tăng lên 50.0 để buộc mô hình phải tập trung tạo ra heatmap chính xác.
    SEG_LOSS_WEIGHT = 50.0
    GRAD_CLIP_VALUE = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Using device: {device}")
    logging.info(f"🔄 Using device: {device}")

    # --- Khởi tạo Model và các thành phần phụ ---
    print("🚀 Initializing Model & Components...")
    model = CLIP_Swin_Implanted().to(device)
    seg_feature_dims = [model.c_in_stages[i] // 2 for i in range(len(model.c_in_stages))]
    text_embedding_dim = 768
    text_projectors_l3 = nn.ModuleList([
        nn.Linear(text_embedding_dim, seg_dim).to(device) for seg_dim in seg_feature_dims
    ])
    pos_classifier = nn.Linear(text_embedding_dim, len(POSITION_NAMES)).to(device)
    # <<< THAY ĐỔI: Xóa instance FocalLoss không cần thiết >>>
    print("✔ Model and Components Initialized.")

    # --- Tải Pretrained Weights ---
    if LOAD_CHECKPOINT_PATH and os.path.exists(LOAD_CHECKPOINT_PATH):
        pass # Sẽ xử lý ở phần Resume
    elif os.path.exists(PRETRAINED_MEDCLIP_CKPT_PATH):
        print(f"📦 Loading pre-trained backbone from: {PRETRAINED_MEDCLIP_CKPT_PATH}")
        state_dict = torch.load(PRETRAINED_MEDCLIP_CKPT_PATH, map_location=device)
        image_state_dict = {k.replace("vision_model.", ""): v for k, v in state_dict.items() if k.startswith("vision_model.")}
        model.backbone.load_state_dict(image_state_dict, strict=False)
    
    for param in model.backbone.parameters(): param.requires_grad = False
    for param in model.backbone.layers[-1].parameters(): param.requires_grad = True
    for param in model.backbone.layers[-2].parameters(): param.requires_grad = True
    print("✔ Backbone configured for fine-tuning.")

    # --- Tải Text Embeddings từ Cache (Logic Hybrid Caching) ---
    print(f"🔠 Loading structured prompt cache from: {PROMPT_CACHE_PATH}")
    if not os.path.exists(PROMPT_CACHE_PATH):
        raise FileNotFoundError(f"'{PROMPT_CACHE_PATH}' not found. Please run the updated prompt_encoder.py first.")
    
    prompt_cache = torch.load(PROMPT_CACHE_PATH, map_location=device)
    text_embeddings_l1 = prompt_cache['l1_embeddings'].detach()
    text_embeddings_l2 = prompt_cache['l2_embeddings'].detach()
    text_embeddings_l3_base = prompt_cache['l3_pathology_embeddings'].detach()
    text_embeddings_pos_base = prompt_cache['location_embeddings'].detach()
    print("✔ All required text embeddings loaded from structured cache.")

    # --- Tải Dữ liệu ---
    print(f"📂 Loading training data from '{DATASET_ROOT_DIR}', excluding: '{EXCLUDED_CLASS_FOR_ZERO_SHOT}'")
    train_dataset = ZeroShotTrainDataset(root_dir=DATASET_ROOT_DIR, excluded_class=EXCLUDED_CLASS_FOR_ZERO_SHOT)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_corrupted
    )
    print(f"✔ Training data loaded with {len(train_dataset)} samples.")

    # --- Thiết lập Optimizer và Scheduler ---
    params_to_train = [
        {"params": model.backbone.layers[-1].parameters(), "lr": MAX_LEARNING_RATE_BACKBONE},
        {"params": model.backbone.layers[-2].parameters(), "lr": MAX_LEARNING_RATE_BACKBONE},
        {"params": model.cls_adapters.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": model.seg_adapters.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": model.image_projection.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": [model.logit_scale], "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": text_projectors_l3.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS},
        {"params": pos_classifier.parameters(), "lr": MAX_LEARNING_RATE_ADAPTERS}
    ]
    optimizer = torch.optim.AdamW(params_to_train, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=[group['lr'] for group in params_to_train], total_steps=len(train_loader) * EPOCHS, pct_start=0.2)
    scaler = GradScaler()
    print("✔ Optimizer, Scheduler, and GradScaler are set up.")

    # --- Logic Tải Checkpoint để Resume ---
    start_epoch = 0
    best_loss = float('inf')
    if LOAD_CHECKPOINT_PATH and os.path.exists(LOAD_CHECKPOINT_PATH):
        logging.info(f"Resuming training from checkpoint: {LOAD_CHECKPOINT_PATH}")
        print(f"🔄 Resuming from checkpoint: {LOAD_CHECKPOINT_PATH}")
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        text_projectors_l3.load_state_dict(checkpoint['text_projectors_state_dict'])
        pos_classifier.load_state_dict(checkpoint['pos_classifier_state_dict'])
        if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"✅ Resumed from Epoch {start_epoch}. Best loss so far: {best_loss:.4f}")
    else:
        print("📂 No checkpoint found, starting from scratch.")

    # --- Chuẩn bị và hàm Visualization ---
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    if len(train_dataset) > 0:
        vis_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_skip_corrupted)
        fixed_vis_batch = next(iter(vis_loader))
    else:
        fixed_vis_batch = None
    
    # (Hàm Visualization giữ nguyên)
    def save_visualization(model, text_projectors, text_embeddings_l3_base, pos_classifier, vis_batch, epoch, device, output_dir):
        if vis_batch is None: return
        model.eval(); pos_classifier.eval()
        with torch.no_grad(), autocast():
            image, mask_gt, _, _, label3, valid_region_mask, label_pos = vis_batch
            image, mask_gt, label3, valid_region_mask, label_pos = image.to(device), mask_gt.to(device), label3.to(device), valid_region_mask.to(device), label_pos.to(device)
            image_features, _, seg_intermediates = model(image)
            pred_mask_list = []
            for i, projector in enumerate(text_projectors):
                patch_tokens = seg_intermediates[i]
                text_embed_proj = projector(text_embeddings_l3_base)
                patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
                text_embed_proj = text_embed_proj / text_embed_proj.norm(dim=-1, keepdim=True)
                anomaly_scores = patch_tokens @ text_embed_proj.T
                B, N, C = anomaly_scores.shape; H_feat = W_feat = int(N ** 0.5)
                anomaly_scores = anomaly_scores.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
                heatmap_pred = F.interpolate(anomaly_scores, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                pred_mask_list.append(heatmap_pred)
            avg_heatmap = torch.stack(pred_mask_list).mean(dim=0).sigmoid()
            masked_avg_heatmap = avg_heatmap * valid_region_mask.unsqueeze(1)
            pos_logits = pos_classifier(image_features)
            pos_pred = (pos_logits.sigmoid() > 0.5)
            num_images = image.shape[0]; fig, axes = plt.subplots(num_images, 3, figsize=(18, 6 * num_images))
            fig.suptitle(f'Epoch {epoch+1} Visualization', fontsize=16)
            for i in range(num_images):
                img_display = image[i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
                img_display = std * img_display + mean; img_display = np.clip(img_display, 0, 1)
                true_class_idx = label3[i].nonzero(as_tuple=True)[0]
                class_name = LEVEL3_NAMES[true_class_idx[0]] if len(true_class_idx) > 0 else 'normal'
                idx_to_show = true_class_idx[0] if len(true_class_idx) > 0 else 0
                gt_mask_display = mask_gt[i, idx_to_show].cpu().numpy()
                pred_heatmap_display = masked_avg_heatmap[i, idx_to_show].cpu().numpy()
                true_pos_names = [POSITION_NAMES[j] for j, val in enumerate(label_pos[i]) if val > 0]
                pred_pos_names = [POSITION_NAMES[j] for j, val in enumerate(pos_pred[i]) if val > 0]
                pos_label = f"GT: {', '.join(true_pos_names) or 'N/A'}\nPred: {', '.join(pred_pos_names) or 'N/A'}"
                axes[i, 0].imshow(img_display); axes[i, 0].set_title("Original Image")
                axes[i, 1].imshow(gt_mask_display, cmap='gray'); axes[i, 1].set_title(f"GT Mask: {class_name}\n{pos_label}")
                axes[i, 2].imshow(pred_heatmap_display, cmap='jet'); axes[i, 2].set_title(f"Predicted Heatmap: {class_name}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1:03d}.png")); plt.close(fig)

    # --- Vòng lặp huấn luyện ---
    if len(train_loader) == 0:
        print("❌ Training loader is empty. Exiting.")
        return
        
    print("\n🔥 Starting training loop...")
    log_interval = max(1, len(train_loader) // 10)

    for epoch in range(start_epoch, EPOCHS):
        model.train(); pos_classifier.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch_data in enumerate(pbar):
            if batch_data is None: continue
            try:
                image, mask_gt, label1, label2, label3, valid_region_mask, label_pos = [d.to(device) for d in batch_data]
            except Exception as e:
                logging.warning(f"Failed to unpack or move to device at step {step}: {e}"); continue
            
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                image_features, logit_scale, seg_intermediates = model(image)
                B = image_features.shape[0]
                
                # --- Classification Loss Calculation ---
                pos_indices = torch.argmax(label_pos, dim=1)
                location_embeds_for_batch = text_embeddings_pos_base[pos_indices]
                combined_text_embeddings_l3 = text_embeddings_l3_base.unsqueeze(0) + location_embeds_for_batch.unsqueeze(1)
                combined_text_embeddings_l3 = F.normalize(combined_text_embeddings_l3, p=2, dim=2)
                
                logits3 = torch.bmm(image_features.unsqueeze(1), combined_text_embeddings_l3.transpose(1, 2)).squeeze(1) * logit_scale.exp()
                logits1 = image_features @ text_embeddings_l1.T * logit_scale.exp()
                logits2 = image_features @ text_embeddings_l2.T * logit_scale.exp()
                logits_pos = pos_classifier(image_features)

                loss1 = F.binary_cross_entropy_with_logits(logits1, label1)
                loss2 = F.binary_cross_entropy_with_logits(logits2, label2)
                loss3 = F.binary_cross_entropy_with_logits(logits3, label3)
                loss_pos = F.binary_cross_entropy_with_logits(logits_pos, label_pos)
                clf_loss = (LOSS_WEIGHT_L1*loss1) + (LOSS_WEIGHT_L2*loss2) + (LOSS_WEIGHT_L3*loss3) + (LOSS_WEIGHT_POS*loss_pos)

                # --- Segmentation Loss Calculation ---
                active_class_idxs = (label3.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
                seg_loss_val = torch.tensor(0.0, device=device)
                if active_class_idxs.numel() > 0:
                    total_focal_seg, total_dice_seg = 0.0, 0.0
                    C_active = active_class_idxs.numel()

                    for i, projector in enumerate(text_projectors_l3):
                        patch_tokens = seg_intermediates[i]
                        seg_dim = patch_tokens.shape[-1]
                        
                        # <<< THAY ĐỔI QUAN TRỌNG 2: Cung cấp text embedding có thông tin vị trí cho nhánh segmentation >>>
                        # Lấy text embedding kết hợp cho các lớp active
                        seg_text_embeds = combined_text_embeddings_l3[:, active_class_idxs, :] # Shape: [B, C_active, D]
                        # Chiếu embedding này xuống không gian của patch_token
                        # Reshape để chiếu tất cả cùng lúc
                        reshaped_embeds = seg_text_embeds.reshape(B * C_active, text_embedding_dim)
                        projected_embeds = projector(reshaped_embeds)
                        text_embed_proj = projected_embeds.view(B, C_active, seg_dim) # Shape: [B, C_active, seg_dim]

                        # Chuẩn hóa và tính anomaly scores bằng BMM
                        patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
                        text_embed_proj = text_embed_proj / text_embed_proj.norm(dim=-1, keepdim=True)
                        anomaly_scores = torch.bmm(patch_tokens, text_embed_proj.transpose(1, 2)) # Shape: [B, N, C_active]
                        
                        H_feat = W_feat = int(anomaly_scores.shape[1] ** 0.5)
                        anomaly_scores = anomaly_scores.permute(0, 2, 1).reshape(B, C_active, H_feat, W_feat)
                        heatmap_pred = F.interpolate(anomaly_scores, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                        
                        focal_seg, dice_seg = focal_dice_loss(
                            heatmap_pred, mask_gt[:, active_class_idxs], valid_region_mask
                        )
                        total_focal_seg += focal_seg; total_dice_seg += dice_seg
                    
                    avg_focal_seg = total_focal_seg / len(text_projectors_l3)
                    avg_dice_seg = total_dice_seg / len(text_projectors_l3)
                    seg_loss_val = avg_focal_seg + avg_dice_seg

                total_loss = clf_loss + SEG_LOSS_WEIGHT * seg_loss_val

            if not torch.isfinite(total_loss):
                logging.warning(f"Invalid loss detected (NaN or Inf: {total_loss.item()}). Skipping step."); continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            all_trainable_params = [p for group in params_to_train for p in group['params']]
            torch.nn.utils.clip_grad_norm_(all_trainable_params, GRAD_CLIP_VALUE)
            scaler.step(optimizer); scaler.update(); scheduler.step()

            running_loss += total_loss.item()
            pbar.set_postfix({"loss": total_loss.item(), "lr": scheduler.get_last_lr()[0]})

            if (step + 1) % log_interval == 0:
                logging.info(
                    f"Epoch {epoch+1} | Step [{step+1}/{len(train_loader)}] | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"CLF Loss: {clf_loss.item():.4f} | "
                    f"Seg Loss (Weighted): {(SEG_LOSS_WEIGHT * seg_loss_val).item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch+1} finished. Avg Train Loss: {avg_loss:.4f}")
        logging.info(f"--- Epoch {epoch+1} Summary | Avg Train Loss: {avg_loss:.4f} ---")

        save_visualization(model, text_projectors_l3, text_embeddings_l3_base, pos_classifier, fixed_vis_batch, epoch, device, VIS_OUTPUT_DIR)
        logging.info(f"🖼️ Saved visualization for epoch {epoch+1}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(OUTPUT_CHECKPOINT_DIR, "model_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_projectors_state_dict': text_projectors_l3.state_dict(),
                'pos_classifier_state_dict': pos_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }, save_path)
            print(f"💾 Best model saved to {save_path} (loss: {best_loss:.4f})")
            logging.info(f"💾 Best model saved to {save_path}")

    final_save_path = os.path.join(OUTPUT_CHECKPOINT_DIR, "model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_projectors_state_dict': text_projectors_l3.state_dict(),
        'pos_classifier_state_dict': pos_classifier.state_dict(),
        'epoch': EPOCHS,
    }, final_save_path)
    print(f"💾 Final model saved to {final_save_path}")
    print("\n🎉 Training finished!")

if __name__ == '__main__':
    main()