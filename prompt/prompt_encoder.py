# --- START OF FILE prompt_encoder.py (VERSION 4.0 - HYBRID CACHING) ---

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import sys
# Lấy đường dẫn của thư mục cha (nơi chứa zero_shot_dataset.py)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Thêm thư mục cha vào sys.path để Python có thể tìm thấy module
sys.path.insert(0, parent_dir)
from zero_shot_dataset import LEVEL1_NAMES, LEVEL2_NAMES, LEVEL3_NAMES, POSITION_NAMES


def encode_prompts(prompt_bank_path, ckpt_path, output_path="prompt_cache.pt", device="cuda"):
    """
    Mã hóa các prompt từ prompt_bank.json và lưu chúng vào một file cache có cấu trúc,
    tách biệt các embedding bệnh lý và vị trí để hỗ trợ 'Hybrid Caching'.
    """
    device = torch.device(device)
    print(f"Using device: {device}")

    # 1. Tải Text Encoder từ MedCLIP (Giữ nguyên, không thay đổi)
    print("Loading MedCLIP text encoder (BioClinicalBERT)...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

    print(f"Loading checkpoint weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    text_state_dict = {k.replace("text_model.", ""): v for k, v in state_dict.items() if k.startswith("text_model.")}
    
    if text_state_dict:
        text_encoder.load_state_dict(text_state_dict, strict=False)
        print("Successfully loaded text encoder weights from MedCLIP checkpoint.")
    else:
        print("Warning: No 'text_model' weights found in checkpoint. Using base Bio_ClinicalBERT weights.")
    
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    print("Text encoder is ready and frozen.")

    # 2. Tải prompt bank từ file JSON (Giữ nguyên, không thay đổi)
    print(f"Loading prompts from: {prompt_bank_path}")
    if not os.path.exists(prompt_bank_path):
        raise FileNotFoundError(f"Prompt bank file not found at: {prompt_bank_path}. Please create it first.")
        
    with open(prompt_bank_path, "r", encoding="utf-8") as f:
        prompt_bank = json.load(f)
    print(f"Loaded {len(prompt_bank)} classes from prompt bank.")

    # 3. Mã hóa tất cả các prompt và lưu vào một cache tạm thời
    temp_cache = {}
    for name, entry in tqdm(prompt_bank.items(), desc="Encoding all prompts"):
        if "prompts" not in entry or not entry["prompts"]:
            print(f"Warning: No prompts found for key '{name}'. Skipping.")
            continue
            
        prompts = list(set(entry["prompts"]))
        embeddings = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = text_encoder(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu())
        
        if embeddings:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            temp_cache[name] = avg_embedding

    # <<< THAY ĐỔI: TÁI CẤU TRÚC CACHE TẠM THỜI THÀNH CÁC TENSOR CUỐI CÙNG >>>
    print("Re-organizing cache into structured tensors...")
    
    # Sắp xếp các embedding theo đúng thứ tự của các danh sách đã import
    # Dùng .squeeze() để bỏ chiều thừa (nếu có) do stack
    l1_embeddings_tensor = torch.stack([temp_cache[name] for name in LEVEL1_NAMES]).squeeze()
    l2_embeddings_tensor = torch.stack([temp_cache[name] for name in LEVEL2_NAMES]).squeeze()
    l3_embeddings_tensor = torch.stack([temp_cache[name] for name in LEVEL3_NAMES]).squeeze()
    pos_embeddings_tensor = torch.stack([temp_cache[name] for name in POSITION_NAMES]).squeeze()
    
    # Tạo dictionary cuối cùng để lưu
    final_structured_cache = {
        'l1_embeddings': l1_embeddings_tensor,
        'l2_embeddings': l2_embeddings_tensor,
        'l3_pathology_embeddings': l3_embeddings_tensor,
        'location_embeddings': pos_embeddings_tensor
    }

    # 4. Lưu cache có cấu trúc
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_structured_cache, output_path)
    print("\nSuccessfully saved structured prompt cache to:", output_path)
    print("Cache content summary:")
    for key, tensor in final_structured_cache.items():
        print(f"- {key}: Tensor with shape {tensor.shape}")


if __name__ == "__main__":
    PROMPT_BANK_FILE = r"prompt\prompt_bank.json"
    MEDCLIP_CHECKPOINT = r"checkpoints\pytorch_model.bin"
    CACHE_OUTPUT_FILE = r"prompt\prompt_cache_structured.pt" # Đổi tên file để tránh nhầm lẫn

    encode_prompts(
        prompt_bank_path=PROMPT_BANK_FILE,
        ckpt_path=MEDCLIP_CHECKPOINT,
        output_path=CACHE_OUTPUT_FILE,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )