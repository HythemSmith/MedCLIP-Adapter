import os
import pandas as pd

# Config
dataset_path = "dataset.xlsx"
image_folder = "E:\MedCLIP-Adapter\MedCLIP-Adapter\dataset_nostructure\images"
mask_folder = "E:\MedCLIP-Adapter\MedCLIP-Adapter\dataset_nostructure\masks"

# Load excel
df = pd.read_excel(dataset_path, sheet_name=0)

# disease and position columns
disease_cols = [
    "osteochondroma", "osteosarcoma", "giant cell tumor",
    "osteofibroma", "simple bone cyst", "multiple osteochondromas",
    "synovial osteochondroma", "other bt", "other mt"
]
position_cols = ['hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'pelvis']

# results
errors = []

for idx, row in df.iterrows():
    img_name = row['q']
    # Construct mask name by replacing the image extension with '_mask.png'
    mask_name = os.path.splitext(img_name)[0] + "_mask.png"

    # check mask exists
    if not os.path.exists(os.path.join(mask_folder, mask_name)):
        errors.append((img_name, "Missing mask file"))

    # check only one disease
    active_disease = [d for d in disease_cols if row[d] == 1]
    if len(active_disease) != 1:
        errors.append((img_name, f"Has {len(active_disease)} diseases active: {active_disease}"))

    # check only one position
    active_pos = [p for p in position_cols if row[p] == 1]
    if len(active_pos) != 1:
        errors.append((img_name, f"Has {len(active_pos)} positions active: {active_pos}"))

# log to csv
report = pd.DataFrame(errors, columns=["image", "error"])
report.to_csv("consistency_report.csv", index=False)
print("✅ Consistency check complete. See consistency_report.csv")