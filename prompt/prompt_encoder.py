# --- START OF FILE prompt_encoder.py (VERSION 3.0 - MINIMALIST) ---

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def encode_prompts(prompt_bank_path, ckpt_path, output_path="prompt_cache2.pt", device="cuda"):
    """
    Chỉ đọc file prompt_bank.json được cung cấp và mã hóa nội dung của nó.
    Không tự động tạo thêm bất kỳ prompt nào.
    """
    device = torch.device(device)
    print(f"Using device: {device}")

    # 1. Tải Text Encoder từ MedCLIP
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

    # 2. Tải prompt bank từ file JSON
    print(f"Loading prompts from: {prompt_bank_path}")
    if not os.path.exists(prompt_bank_path):
        raise FileNotFoundError(f"Prompt bank file not found at: {prompt_bank_path}. Please create it first.")
        
    with open(prompt_bank_path, "r", encoding="utf-8") as f:
        prompt_bank = json.load(f)
    print(f"Loaded {len(prompt_bank)} classes from prompt bank.")

    # 3. Mã hóa tất cả các prompt trong ngân hàng
    prompt_cache = {}
    for name, entry in tqdm(prompt_bank.items(), desc="Encoding prompts"):
        # Đảm bảo có key 'prompts' và nó không rỗng
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
        
        # Lấy trung bình embedding của tất cả các câu prompt cho một lớp
        if embeddings:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            prompt_cache[name] = {
                "embedding": avg_embedding
            }
    
    # 4. Lưu cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(prompt_cache, output_path)
    print(f"\nSuccessfully saved prompt cache with {len(prompt_cache)} keys to: {output_path}")

if __name__ == "__main__":
    # Đảm bảo các đường dẫn này chính xác
    PROMPT_BANK_FILE = r"prompt\prompt_bank.json"
    MEDCLIP_CHECKPOINT = r"checkpoints\pytorch_model.bin"
    CACHE_OUTPUT_FILE = r"prompt\prompt_cache.pt"

    encode_prompts(
        prompt_bank_path=PROMPT_BANK_FILE,
        ckpt_path=MEDCLIP_CHECKPOINT,
        output_path=CACHE_OUTPUT_FILE,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )