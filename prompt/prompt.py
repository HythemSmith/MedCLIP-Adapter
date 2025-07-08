import pandas as pd
import json

# Load sheet 1
df = pd.read_excel(r"prompt\dataset.xlsx", sheet_name=0)

# Xác định cột disease thật sự
disease_cols = [
    "osteochondroma", "osteosarcoma", "giant cell tumor",
    "osteofibroma", "simple bone cyst", "multiple osteochondromas",
    "synovial osteochondroma", "other bt", "other mt"
]

# Xác định vị trí giải phẫu
position_cols = ['hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'pelvis']

# Bảng phân loại mức độ
disease_level2_map = {
    "osteochondroma": "benign",
    "osteosarcoma": "malignant",
    "giant cell tumor": "benign",
    "osteofibroma": "benign",
    "simple bone cyst": "benign",
    "multiple osteochondromas": "benign",
    "synovial osteochondroma": "benign",
    "other bt": "benign",
    "other mt": "malignant"
}

prompt_dict = {}

for disease in disease_cols:
    disease_clean = disease.strip().lower().replace(" ", "_")
    subset = df[df[disease]==1]

    # các vị trí có bệnh
    positions_for_disease = []
    for pos in position_cols:
        if subset[pos].sum() > 0:
            positions_for_disease.append(pos)

    prompts = []

    if positions_for_disease:
        for pos in positions_for_disease:
            prompts.append(
                f"An X-ray showing a case of {disease} affecting the {pos}, a {disease_level2_map[disease]} bone tumor."
            )
            prompts.append(
                f"Radiograph demonstrates {disease} in the {pos}, consistent with {disease_level2_map[disease]} nature."
            )
            prompts.append(
                f"Medical image of {disease} involving the {pos}, classified as {disease_level2_map[disease]}."
            )
            prompts.append(
                f"Typical radiographic features of {disease} located in the {pos}, which is {disease_level2_map[disease]}."
            )
    else:
        prompts.append(
            f"An X-ray demonstrating {disease}, a {disease_level2_map[disease]} bone tumor."
        )
        prompts.append(
            f"Radiograph showing features of {disease}, consistent with {disease_level2_map[disease]}."
        )

    prompt_dict[disease_clean] = {
        "prompts": prompts,
        "position": positions_for_disease,
        "level1": "bone_tumor",
        "level2": disease_level2_map[disease]
    }

# Save to JSON
with open("prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompt_dict, f, indent=4, ensure_ascii=False)

print("✅ prompts.json file saved with cleaned disease-focused prompts.")
