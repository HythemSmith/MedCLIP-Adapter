import os
import cv2
import numpy as np
from PIL import Image

def black_outside_film(image: Image.Image) -> Image.Image:
    # Convert to grayscale for contour detection
    gray = np.array(image.convert("L"))
    inverted = 255 - gray

    # Threshold to extract dark region (the film)
    _, thresh = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠️ No contours found, skipping")
        return image

    # Create a mask of the detected film region
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Dilate the mask to avoid cutting parts of the bone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel)

    # Apply mask: keep pixels inside film, black out everything else
    img_np = np.array(image)
    if img_np.ndim == 2:  # grayscale
        img_np = np.expand_dims(img_np, axis=2)

    if img_np.ndim == 3:
        img_masked = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            img_masked[..., c] = np.where(mask == 255, img_np[..., c], 0)
    else:
        img_masked = np.where(mask == 255, img_np, 0)

    return Image.fromarray(img_masked.squeeze())  # remove singleton dim if grayscale

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                img = Image.open(in_path).convert("RGB")
                out = black_outside_film(img)
                out.save(out_path)
                print(f"✅ Saved: {out_path}")
            except Exception as e:
                print(f"❌ Error on {in_path}: {e}")

if __name__ == "__main__":
    input_dir = r"C:\Users\vanlo\Desktop\BTXRD_cleaned"              # Thư mục gốc
    output_dir = r"C:\Users\vanlo\Desktop\categories_cleaned"      # thư mục kết quả

    process_folder(input_dir, output_dir)
