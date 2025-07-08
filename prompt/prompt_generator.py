# --- START OF FILE prompt_generator.py ---

import json

def generate_descriptive_prompts():
    """
    Tạo một file JSON chứa các câu prompt mô tả đặc trưng y tế chất lượng cao.
    Bộ prompt này được thiết kế riêng cho cấu trúc dataset của bạn.
    """
    print("Generating high-quality, descriptive prompts for specified diseases...")
    
    prompt_bank = {}

    # === BONE TUMOR / BENIGN ===

    prompt_bank["giant_cell_tumor"] = {
        "prompts": [
            "An X-ray of a giant cell tumor, a benign but locally aggressive bone tumor.",
            "A lytic, eccentric lesion in the epiphysis of a long bone, extending to the articular surface.",
            "Radiograph showing a geographic bone lesion with a well-defined, non-sclerotic border.",
            "A medical scan presenting a 'soap bubble' appearance, characteristic of a giant cell tumor."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }

    prompt_bank["multiple_osteochondromas"] = {
        "prompts": [
            "An X-ray showing multiple osteochondromas, a condition known as hereditary multiple exostoses.",
            "Multiple bony projections with cartilage caps, typically seen in a young patient.",
            "Radiograph showing multiple exostoses pointing away from the nearest joint, causing bone deformity."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }

    prompt_bank["osteochondroma"] = {
        "prompts": [
            "An X-ray of an osteochondroma, the most common benign bone tumor.",
            "A cartilage-capped bony projection on the external surface of a bone.",
            "This radiograph shows a pedunculated or sessile lesion with cortical and medullary continuity with the host bone."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }
    
    prompt_bank["osteofibroma"] = {
        "prompts": [
            "An X-ray of an osteofibrous dysplasia, a benign fibro-osseous lesion.",
            "A well-circumscribed, intracortical, lytic lesion, typically in the tibia.",
            "Radiograph showing a hazy, ground-glass appearance within the bone lesion."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }

    prompt_bank["other_bt"] = {
        "prompts": [
            "An X-ray of a benign bone tumor, not otherwise specified.",
            "A non-aggressive bone lesion with well-defined borders and a sclerotic rim.",
            "A focal benign process within the bone, showing no signs of aggressive features."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }

    prompt_bank["simple_bone_cyst"] = {
        "prompts": [
            "An X-ray of a simple or unicameral bone cyst.",
            "A centrally located, well-defined lytic lesion in the metaphysis of a long bone.",
            "This scan shows a fluid-filled cavity with a thin sclerotic rim and may show a 'fallen fragment' sign after a fracture."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }

    prompt_bank["synovial_osteochondroma"] = {
        "prompts": [
            "An X-ray of synovial osteochondromatosis, a benign condition of the joint lining.",
            "Multiple small, calcified or ossified loose bodies within a joint or bursa.",
            "Radiograph showing multiple intra-articular nodules of similar size, causing a 'bag of marbles' appearance."
        ],
        "level1": "bone_tumor",
        "level2": "benign"
    }

    # === BONE TUMOR / MALIGNANT ===
    
    prompt_bank["osteosarcoma"] = {
        "prompts": [
            "An X-ray of osteosarcoma, a highly malignant bone tumor that produces bone.",
            "An aggressive, poorly defined lesion in the metaphysis of a long bone, often around the knee.",
            "Radiograph showing a 'sunburst' or 'sun-ray' periosteal reaction and extensive bone destruction.",
            "This scan presents a finding of Codman's triangle and production of tumor osteoid, classic for osteosarcoma."
        ],
        "level1": "bone_tumor",
        "level2": "malignant"
    }
    
    prompt_bank["other_mt"] = {
        "prompts": [
            "An X-ray of a malignant bone tumor, not otherwise specified.",
            "An aggressive bone lesion with ill-defined borders, cortical destruction, and a soft tissue mass.",
            "A focal malignant process within the bone, suspicious for a primary sarcoma or a metastatic lesion."
        ],
        "level1": "bone_tumor",
        "level2": "malignant"
    }

    # === MISC ===

    prompt_bank["normal"] = {
        "prompts": [
            "A normal X-ray with no significant findings or abnormalities.",
            "This radiograph shows a healthy bone and joint structure with intact cortex and normal density.",
            "No evidence of fracture, dislocation, degenerative changes, or tumor."
        ],
        "level1": "misc",
        "level2": "none"  # Hoặc bạn có thể để là "normal" nếu bạn có một lớp như vậy
    }
    
    return prompt_bank

def main():
    """Hàm chính để tạo và lưu file JSON."""
    
    # Đường dẫn file output
    output_json_path = "prompt_bank.json"
    
    # 1. Tạo dictionary chứa các câu prompt chất lượng cao
    prompts_to_save = generate_descriptive_prompts()
    
    # 2. Lưu vào file JSON
    print(f"Saving generated prompts to: {output_json_path}...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(prompts_to_save, f, ensure_ascii=False, indent=4)
        
        print(f"Prompts successfully saved to '{output_json_path}'.")
        print(f"Total classes with prompts: {len(prompts_to_save)}")
        print("\nNext step: Run 'prompt_encoder.py' to create the cache file.")

    except Exception as e:
        print(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()