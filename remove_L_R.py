import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = r"C:\Users\vanlo\Desktop\xray_rl_detection\runs\detect\train3\weights\best.pt"
SOURCE_ROOT = r"C:\Users\vanlo\Desktop\organized"
OUTPUT_ROOT = r"C:/Users/vanlo/Desktop/organized_cleaned"
IMG_EXTS = ['.jpg', '.jpeg', '.png']

# === LOAD YOLOv8 MODEL ===
model = YOLO(MODEL_PATH)

# === WALK THROUGH ALL IMAGES FOLDERS ===
image_files = []
for root, _, files in os.walk(SOURCE_ROOT):
    if os.path.basename(root).lower() != "images":
        continue
    for fname in files:
        if os.path.splitext(fname)[1].lower() in IMG_EXTS:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, SOURCE_ROOT)
            image_files.append((full_path, rel_path))

# === PROCESS EACH IMAGE ===
for img_path, rel_path in tqdm(image_files, desc="Predicting and cleaning"):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Run detection
    results = model(img, verbose=False)[0]  # one image only

    # Prepare mask for inpainting
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        mask[y1:y2, x1:x2] = 255  # mark all boxes

    # Use TELEA inpainting instead of black fill
    inpainted = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    # Save result
    save_path = os.path.join(OUTPUT_ROOT, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, inpainted)

print(f"\n✅ Đã xử lý xong {len(image_files)} ảnh. Ảnh sạch lưu ở: {OUTPUT_ROOT}")
