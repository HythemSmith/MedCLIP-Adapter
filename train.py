import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # Use non-GUI backend
import logging

from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from MedClip_Adapter import CLIP_Swin_Implanted
from zero_shot_dataset import MedicalTrainDataset, LEVEL1_NAMES, LEVEL2_NAMES, LEVEL3_NAMES, POSITION_NAMES
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
    Custom collate_fn to filter out corrupted samples (returning None) from the batch.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    logging.basicConfig(
        filename="training_run.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # --- Configuration ---
    PRETRAINED_MEDCLIP_CKPT_PATH = r"E:\MedCLIP-Adapter\MedCLIP-Adapter\checkpoints\pytorch_model.bin"
    PROMPT_CACHE_PATH = r"prompt\prompt_cache2.pt"
    OUTPUT_CHECKPOINT_DIR = "checkpoints_final"
    VIS_OUTPUT_DIR = "training_visualizations"
    METADATA_PATH = "metadata.csv"
    DATASET_SOURCE_FILTER = "BTXRD_cleaned"

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
    LOSS_WEIGHT_POS = 0.5  # Weight for position classification loss
    SEG_LOSS_WEIGHT = 5.0  # Weight for segmentation loss
    GRAD_CLIP_VALUE = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Using device: {device}")
    logging.info(f"🔄 Using device: {device}")

    # --- Initialize Model ---
    print("🚀 Initializing Model (CLIP_Swin_Implanted)...")
    model = CLIP_Swin_Implanted().to(device)
    logging.info("🚀 Initializing Model...")

    print(f"📦 Loading pre-trained backbone from: {PRETRAINED_MEDCLIP_CKPT_PATH}")
    state_dict = torch.load(PRETRAINED_MEDCLIP_CKPT_PATH, map_location=device)
    image_state_dict = {k.replace("vision_model.", ""): v for k, v in state_dict.items() if k.startswith("vision_model.")}
    model.backbone.load_state_dict(image_state_dict, strict=False)

    print("🧺 Freezing early backbone stages...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("✔ Backbone configured for fine-tuning.")

    # --- Load Text Embeddings ---
    print(f"🔠 Loading prompt cache from: {PROMPT_CACHE_PATH}")
    if not os.path.exists(PROMPT_CACHE_PATH):
        raise FileNotFoundError(f"'{PROMPT_CACHE_PATH}' not found.")
    prompt_cache = torch.load(PROMPT_CACHE_PATH)

    def create_text_embeddings(level_names, cache):
        embeddings = [cache[name]["embedding"].mean(dim=0) for name in level_names]
        embeddings = torch.stack(embeddings).to(device).detach()
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    text_embeddings_l1 = create_text_embeddings(LEVEL1_NAMES, prompt_cache)
    text_embeddings_l2 = create_text_embeddings(LEVEL2_NAMES, prompt_cache)
    text_embeddings_l3 = create_text_embeddings(LEVEL3_NAMES, prompt_cache)
    text_embeddings_pos = create_text_embeddings(POSITION_NAMES, prompt_cache)
    print("✔ Text embeddings created for all levels and positions.")

    # --- Initialize Text Projectors ---
    print("🚀 Initializing Text Projectors for Segmentation...")
    seg_feature_dims = [model.c_in_stages[i] // 2 for i in range(len(model.c_in_stages))]
    text_embedding_dim = text_embeddings_l3.shape[-1]
    text_projectors_l3 = nn.ModuleList([
        nn.Linear(text_embedding_dim, seg_dim).to(device) for seg_dim in seg_feature_dims
    ])

    # --- Initialize Position Classifier ---
    print("🚀 Initializing Position Classifier...")
    pos_classifier = nn.Linear(text_embedding_dim, len(POSITION_NAMES)).to(device)

    # --- Load Data ---
    print(f"📂 Loading training data from '{METADATA_PATH}', excluding: '{EXCLUDED_CLASS_FOR_ZERO_SHOT}'")
    train_dataset = MedicalTrainDataset(
        metadata_path=METADATA_PATH,
        excluded_class=EXCLUDED_CLASS_FOR_ZERO_SHOT,
        dataset_source_filter=DATASET_SOURCE_FILTER
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_corrupted
    )
    print(f"✔ Training data loaded with {len(train_dataset)} samples.")

    # --- Set up Optimizer and Scheduler ---
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
    scheduler = OneCycleLR(
        optimizer, max_lr=[group['lr'] for group in params_to_train],
        total_steps=len(train_loader) * EPOCHS, pct_start=0.2
    )
    scaler = GradScaler()
    print("✔ Optimizer, Scheduler, and GradScaler are set up.")

    # --- Prepare Visualization ---
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    print("📸 Preparing a fixed batch for visualization...")
    vis_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_skip_corrupted)
    fixed_vis_batch = next(iter(vis_loader))
    print("✔ Fixed visualization batch is ready.")

    # --- Visualization Function ---
    def save_visualization(model, text_projectors, text_embeddings, pos_classifier, text_embeddings_pos, vis_batch, epoch, device, output_dir):
        model.eval()
        with torch.no_grad():
            image, mask_gt, _, _, label3, valid_region_mask, label_pos = vis_batch
            image, mask_gt, label3, valid_region_mask, label_pos = image.to(device), mask_gt.to(device), label3.to(device), valid_region_mask.to(device), label_pos.to(device)

            image_features, _, seg_intermediates = model(image)
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

            avg_heatmap = torch.stack(pred_mask_list).mean(dim=0).sigmoid()
            masked_avg_heatmap = avg_heatmap * valid_region_mask.unsqueeze(1)

            # Position prediction
            pos_logits = pos_classifier(image_features)
            pos_probs = pos_logits.sigmoid()
            pos_pred = (pos_probs > 0.5).float()

            num_images = image.shape[0]
            fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
            fig.suptitle(f'Epoch {epoch+1} Visualization', fontsize=16)

            for i in range(num_images):
                img_display = image[i].cpu().permute(1, 2, 0).numpy()
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) if img_display.max() > img_display.min() else img_display

                true_class_idx = label3[i].nonzero(as_tuple=True)[0]
                if len(true_class_idx) > 0:
                    idx_to_show = true_class_idx[0]
                    class_name = LEVEL3_NAMES[idx_to_show]
                    gt_mask_display = mask_gt[i, idx_to_show].cpu().numpy()
                    pred_heatmap_display = masked_avg_heatmap[i, idx_to_show].cpu().numpy()
                else:
                    idx_to_show = 0
                    class_name = 'normal'
                    gt_mask_display = np.zeros_like(mask_gt[i, 0].cpu().numpy())
                    pred_heatmap_display = masked_avg_heatmap[i, idx_to_show].cpu().numpy()

                # Get position labels
                true_pos_idx = label_pos[i].nonzero(as_tuple=True)[0]
                pred_pos_idx = pos_pred[i].nonzero(as_tuple=True)[0]
                true_pos_names = [POSITION_NAMES[idx] for idx in true_pos_idx]
                pred_pos_names = [POSITION_NAMES[idx] for idx in pred_pos_idx]
                pos_label = f"GT Pos: {', '.join(true_pos_names) if true_pos_names else 'None'}\nPred Pos: {', '.join(pred_pos_names) if pred_pos_names else 'None'}"

                axes[i, 0].imshow(img_display)
                axes[i, 0].set_title("Original Image")
                axes[i, 1].imshow(gt_mask_display, cmap='gray')
                axes[i, 1].set_title(f"Ground Truth: {class_name}\n{pos_label}")
                axes[i, 2].imshow(pred_heatmap_display, cmap='jet')
                axes[i, 2].set_title(f"Predicted Heatmap: {class_name}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1:03d}.png"))
            plt.close(fig)

    # --- Training Loop ---
    print("\n🔥 Starting training loop...")
    best_loss = float('inf')
    os.makedirs(OUTPUT_CHECKPOINT_DIR, exist_ok=True)
    total_steps_in_epoch = len(train_loader)
    log_interval = max(1, total_steps_in_epoch // 10)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch_data in enumerate(pbar):
            if batch_data is None:
                continue

            image, mask_gt, label1, label2, label3, valid_region_mask, label_pos = [d.to(device) for d in batch_data]

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                image_features, logit_scale, seg_intermediates = model(image)

                logits1 = image_features @ text_embeddings_l1.T * logit_scale.exp()
                logits2 = image_features @ text_embeddings_l2.T * logit_scale.exp()
                logits3 = image_features @ text_embeddings_l3.T * logit_scale.exp()
                logits_pos = pos_classifier(image_features)  # Position classification

                loss1 = F.binary_cross_entropy_with_logits(logits1, label1)
                loss2 = F.binary_cross_entropy_with_logits(logits2, label2)
                loss3 = F.binary_cross_entropy_with_logits(logits3, label3)
                loss_pos = F.binary_cross_entropy_with_logits(logits_pos, label_pos)
                clf_loss = (LOSS_WEIGHT_L1 * loss1) + (LOSS_WEIGHT_L2 * loss2) + (LOSS_WEIGHT_L3 * loss3) + (LOSS_WEIGHT_POS * loss_pos)

                # Segmentation Loss
                active_class_idxs = (label3.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
                seg_loss_val = torch.tensor(0.0, device=device)
                if active_class_idxs.numel() > 0:
                    total_bce_seg, total_dice_seg = 0.0, 0.0
                    for i, projector in enumerate(text_projectors_l3):
                        patch_tokens = seg_intermediates[i]
                        text_embed_proj = projector(text_embeddings_l3[active_class_idxs])
                        patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
                        text_embed_proj = text_embed_proj / text_embed_proj.norm(dim=-1, keepdim=True)
                        anomaly_scores = patch_tokens @ text_embed_proj.T
                        B, N, C = anomaly_scores.shape
                        H_feat = W_feat = int(N ** 0.5)
                        anomaly_scores = anomaly_scores.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
                        heatmap_pred = F.interpolate(anomaly_scores, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                        masked_heatmap_pred = heatmap_pred * valid_region_mask.unsqueeze(1)
                        masked_mask_gt = mask_gt[:, active_class_idxs] * valid_region_mask.unsqueeze(1)
                        bce_seg, dice_seg = multi_channel_focal_dice_loss(masked_heatmap_pred, masked_mask_gt)
                        total_bce_seg += bce_seg
                        total_dice_seg += dice_seg
                    avg_bce_seg = total_bce_seg / len(text_projectors_l3)
                    avg_dice_seg = total_dice_seg / len(text_projectors_l3)
                    seg_loss_val = avg_bce_seg + avg_dice_seg

                total_loss = clf_loss + SEG_LOSS_WEIGHT * seg_loss_val

            if not torch.isfinite(total_loss):
                logging.warning(f"Invalid loss detected (NaN or Negative: {total_loss.item()}). Skipping step.")
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            all_trainable_params = [p for group in params_to_train for p in group['params']]
            torch.nn.utils.clip_grad_norm_(all_trainable_params, GRAD_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += total_loss.item()
            pbar.set_postfix({"loss": total_loss.item(), "lr": scheduler.get_last_lr()[0]})

            if (step + 1) % log_interval == 0:
                logging.info(
                    f"Epoch {epoch+1} | Step [{step+1}/{total_steps_in_epoch}] | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"CLF Loss: {clf_loss.item():.4f} (L1: {loss1.item():.4f}, L2: {loss2.item():.4f}, L3: {loss3.item():.4f}, Pos: {loss_pos.item():.4f}) | "
                    f"SEG Loss: {(SEG_LOSS_WEIGHT * seg_loss_val).item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"\n✅ Epoch {epoch+1} finished. Avg Train Loss: {avg_loss:.4f}")
        logging.info(f"--- Epoch {epoch+1} Summary | Avg Train Loss: {avg_loss:.4f} ---")

        # Save visualization
        try:
            save_visualization(model, text_projectors_l3, text_embeddings_l3, pos_classifier, text_embeddings_pos, fixed_vis_batch, epoch, device, VIS_OUTPUT_DIR)
            logging.info(f"🖼️ Saved visualization for epoch {epoch+1}")
        except Exception as e:
            logging.error(f"Could not save visualization for epoch {epoch+1}: {e}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(OUTPUT_CHECKPOINT_DIR, "model_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_projectors_state_dict': text_projectors_l3.state_dict(),
                'pos_classifier_state_dict': pos_classifier.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }, save_path)
            print(f"💾 Best model saved to {save_path} (loss: {best_loss:.4f})")
            logging.info(f"💾 Best model saved to {save_path}")

    # Save final model
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