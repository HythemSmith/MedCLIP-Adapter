import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def encode_prompts(prompt_bank_path, ckpt_path, output_path="prompt_cache2.pt", device="cuda"):
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load BioClinicalBERT tokenizer and model
    print("Loading MedCLIP text encoder (BioClinicalBERT)...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

    # Load checkpoint weights
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    text_state_dict = {k.replace("text_model.", ""): v for k, v in state_dict.items() if k.startswith("text_model.")}
    text_encoder.load_state_dict(text_state_dict, strict=False)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    print("Text encoder loaded and frozen.")

    # Load prompt_bank.json
    with open(prompt_bank_path, "r", encoding="utf-8") as f:
        prompt_bank = json.load(f)

    prompt_cache = {}

    for disease, entry in tqdm(prompt_bank.items(), desc="Encoding prompts"):
        prompts = list(set(entry["prompts"]))  # remove duplicate
        embeddings = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = text_encoder(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze(0).cpu())

        print(f"{disease} - embedding dim: {embeddings[-1].shape}")

        prompt_cache[disease] = {
            "prompts": prompts,
            "embedding": torch.stack(embeddings),
            "level1": entry.get("level1"),
            "level2": entry.get("level2"),
            "position": entry.get("position")
        }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(prompt_cache, output_path)
    print(f"Saved prompt cache with {len(prompt_cache)} classes to: {output_path}")

if __name__ == "__main__":
    encode_prompts(
        prompt_bank_path=r"E:\MedCLIP-Adapter\MedCLIP-Adapter\prompt\prompts_bank.json",
        ckpt_path=r"E:\MedCLIP-Adapter\MedCLIP-Adapter\checkpoints\pytorch_model.bin",
        output_path=r"prompt\prompt_cache2.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
