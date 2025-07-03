# test.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# IMPORT CÁC MODULE CỦA BẠN
from MedClip_Adapter import CLIP_Swin_Implanted
from zero_shot_dataset import MedicalTestDataset, LEVEL1_NAMES, LEVEL2_NAMES, LEVEL3_NAMES

def main():
    logging.basicConfig(
        filename="testing_zeroshot.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    CHECKPOINT_DIR = "checkpoints_BTXRD_Cleaned"
    CHECKPOINT_FILENAME = "model_best.pth"
    MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    PROMPT_CACHE_PATH = r"prompt\prompt_cache.pt"
    ROOT_DATASET_PATH = r"C:\Users\vanlo\Desktop\organized_cleaned" # <<< KIỂM TRA LẠI ĐƯỜNG DẪN NÀY
    UNSEEN_TEST_CLASS = "multiple_osteochondromas"
    BATCH_SIZE = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Using device for testing: {device}")
    logging.info(f"🔄 Using device for testing: {device}")

    print("🚀 Initializing Zero-Shot Model...")
    model = CLIP_Swin_Implanted().to(device)
    logging.info("🚀 Initializing Zero-Shot Model...")

    print(f"📦 Loading trained checkpoint from: {MODEL_CHECKPOINT_PATH}")
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at '{MODEL_CHECKPOINT_PATH}'.")
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✔ Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")

    model.eval()

    print(f"🔠 Loading prompt cache from: {PROMPT_CACHE_PATH}")
    prompt_cache = torch.load(PROMPT_CACHE_PATH)

    def create_text_embeddings(level_names, cache):
        embeddings = []
        for name in level_names:
            prompt_embeddings = cache[name]["embedding"]
            emb = prompt_embeddings.mean(dim=0)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        return torch.stack(embeddings).to(device)

    text_embeddings_l1 = create_text_embeddings(LEVEL1_NAMES, prompt_cache)
    text_embeddings_l2 = create_text_embeddings(LEVEL2_NAMES, prompt_cache)
    text_embeddings_l3 = create_text_embeddings(LEVEL3_NAMES, prompt_cache)

    print("🚀 Initializing Text Projectors for Segmentation Evaluation...")
    seg_feature_dims = [model.c_in_stages[i] // 2 for i in range(len(model.c_in_stages))]
    text_embedding_dim = text_embeddings_l3.shape[-1]
    text_projectors_l3 = torch.nn.ModuleList([
        torch.nn.Linear(text_embedding_dim, seg_dim).to(device) for seg_dim in seg_feature_dims
    ])
    # Tải các trọng số đã được huấn luyện cho projector
    if 'text_projectors_state_dict' in checkpoint:
        text_projectors_l3.load_state_dict(checkpoint['text_projectors_state_dict'])
        print("✔ Loaded trained text projectors.")

    print(f"📂 Loading test data for UNSEEN class: '{UNSEEN_TEST_CLASS}'")
    test_dataset = MedicalTestDataset(ROOT_DATASET_PATH, target_class=UNSEEN_TEST_CLASS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n🔬 Starting evaluation loop...")
    all_labels, all_preds, all_logits = {k: [] for k in ['l1', 'l2', 'l3']}, {k: [] for k in ['l1', 'l2', 'l3']}, {k: [] for k in ['l1', 'l2', 'l3']}
    total_loss = 0.0
    # Khởi tạo danh sách lưu Dice score bên ngoài vòng lặp
    dice_scores_all_samples = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating on '{UNSEEN_TEST_CLASS}'")
        for image, mask_gt, label1, label2, label3 in pbar:
            image, mask_gt = image.to(device), mask_gt.to(device)
            label1, label2, label3 = label1.to(device), label2.to(device), label3.to(device)

            image_features, logit_scale, seg_intermediates = model(image)

            logits1 = image_features @ text_embeddings_l1.T * logit_scale.exp()
            logits2 = image_features @ text_embeddings_l2.T * logit_scale.exp()
            logits3 = image_features @ text_embeddings_l3.T * logit_scale.exp()

            loss1 = F.binary_cross_entropy_with_logits(logits1, label1)
            loss2 = F.binary_cross_entropy_with_logits(logits2, label2)
            loss3 = F.binary_cross_entropy_with_logits(logits3, label3)
            total_loss += (loss1 + loss2 + loss3).item()

            probs1 = torch.sigmoid(logits1)
            probs2 = torch.sigmoid(logits2)
            probs3 = torch.sigmoid(logits3)

            all_preds['l1'].append((probs1 > 0.5).cpu().numpy())
            all_preds['l2'].append((probs2 > 0.5).cpu().numpy())
            all_preds['l3'].append((probs3 > 0.5).cpu().numpy())
            all_labels['l1'].append(label1.cpu().numpy())
            all_labels['l2'].append(label2.cpu().numpy())
            all_labels['l3'].append(label3.cpu().numpy())
            all_logits['l1'].append(probs1.cpu().numpy())
            all_logits['l2'].append(probs2.cpu().numpy())
            all_logits['l3'].append(probs3.cpu().numpy())

            # === Segmentation prediction ===
            seg_loss_batch = 0.0
            pred_mask_list = []
            true_mask_list = []

            for i, projector in enumerate(text_projectors_l3):
                patch_tokens = seg_intermediates[i]
                text_embed_proj = projector(text_embeddings_l3)

                patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
                text_embed_proj = text_embed_proj / text_embed_proj.norm(dim=-1, keepdim=True)

                anomaly_scores = patch_tokens @ text_embed_proj.T
                B, N, C = anomaly_scores.shape
                H_feat = W_feat = int(N ** 0.5)

                anomaly_scores = anomaly_scores.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
                heatmap_pred = F.interpolate(anomaly_scores, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                pred_mask_list.append(heatmap_pred.sigmoid())
                true_mask_list.append(mask_gt)

            all_pred_masks = torch.stack(pred_mask_list).mean(dim=0)
            all_true_masks = true_mask_list[0]

            smooth = 1e-5
            pred_bin = (all_pred_masks > 0.5).float()
            intersection = (pred_bin * all_true_masks).sum(dim=(2, 3))
            union = pred_bin.sum(dim=(2, 3)) + all_true_masks.sum(dim=(2, 3))
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores_all_samples.append(dice.cpu())

    for level in ['l1', 'l2', 'l3']:
        all_labels[level] = np.vstack(all_labels[level])
        all_preds[level] = np.vstack(all_preds[level])
        all_logits[level] = np.vstack(all_logits[level])

    avg_test_loss = total_loss / len(test_loader)
    print(f"\n--- 📊 Test Results for Unseen Class: '{UNSEEN_TEST_CLASS}' ---")
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    def print_metrics_for_level(level_name, labels_true, preds_pred, logits_pred, class_names):
        print(f"\n--- Metrics for {level_name} ---")
        support_per_class = labels_true.sum(axis=0)
        relevant_classes_idx = np.where(support_per_class > 0)[0]
        if len(relevant_classes_idx) == 0:
            print("No positive samples for any class.")
            return

        p, r, f1, s = precision_recall_fscore_support(labels_true, preds_pred, labels=relevant_classes_idx, average=None, zero_division=0)
        print(f"{'Class':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
        print("-" * 75)
        for idx, (prec, rec, f1_score, sup) in enumerate(zip(p, r, f1, s)):
            class_idx = relevant_classes_idx[idx]
            print(f"{class_names[class_idx]:<30} | {prec:<10.3f} | {rec:<10.3f} | {f1_score:<10.3f} | {int(sup):<10}")

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels_true, preds_pred, average='macro', zero_division=0)
        print(f"\nMacro-Average - P: {p_macro:.3f}, R: {r_macro:.3f}, F1: {f1_macro:.3f}")

        print("\n--- AUC Scores ---")
        for idx in relevant_classes_idx:
            if len(np.unique(labels_true[:, idx])) > 1:
                auc_score = roc_auc_score(labels_true[:, idx], logits_pred[:, idx])
                print(f"AUC for class {class_names[idx]}: {auc_score:.3f}")
            else:
                print(f"AUC for class {class_names[idx]}: N/A")

    print_metrics_for_level("LEVEL 1", all_labels['l1'], all_preds['l1'], all_logits['l1'], LEVEL1_NAMES)
    print_metrics_for_level("LEVEL 2", all_labels['l2'], all_preds['l2'], all_logits['l2'], LEVEL2_NAMES)
    print_metrics_for_level("LEVEL 3", all_labels['l3'], all_preds['l3'], all_logits['l3'], LEVEL3_NAMES)

    if dice_scores_all_samples:
        # Ghép các tensor dice score từ các batch lại
        dice_scores_all_samples = torch.cat(dice_scores_all_samples, dim=0)
        # Tính Dice trung bình cho mỗi lớp
        dice_per_class = dice_scores_all_samples.mean(dim=0)
        
        # Lấy chỉ số của lớp unseen
        unseen_class_idx = LEVEL3_NAMES.index(UNSEEN_TEST_CLASS)
        unseen_class_dice = dice_per_class[unseen_class_idx]
        print("\n--- 🎯 Zero-Shot Segmentation Metric ---")
        print(f"Dice Score for UNSEEN class '{UNSEEN_TEST_CLASS}': {unseen_class_dice:.4f}")

    print("\n🎉 Evaluation finished!")

if __name__ == '__main__':
    main()
