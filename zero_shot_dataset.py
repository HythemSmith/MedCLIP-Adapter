import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import glob

# ... (Các danh sách và map không đổi) ...
LEVEL1_NAMES = ['bone_tumor', 'degenerative', 'misc', 'trauma']
LEVEL2_NAMES = ['benign', 'malignant']
LEVEL3_NAMES = [
    'giant_cell_tumor', 'multiple_osteochondromas', 'osteochondroma', 'osteofibroma',
    'other_bt', 'simple_bone_cyst', 'synovial_osteochondroma', 'osteosarcoma', 'other_mt',
    'disc_space_narrowing', 'foraminal_stenosis', 'osteophytes', 'normal', 'other_lesions',
    'spondylolysthesis', 'surgical_implant', 'vertebral_collapse', 'broken'
]
POSITION_NAMES = ['hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'pelvis', 'unknown']

LEVEL1_MAP = {v: i for i, v in enumerate(LEVEL1_NAMES)}
LEVEL2_MAP = {v: i for i, v in enumerate(LEVEL2_NAMES)}
LEVEL3_MAP = {v: i for i, v in enumerate(LEVEL3_NAMES)}
POSITION_MAP = {v: i for i, v in enumerate(POSITION_NAMES)}

class ResizePadToSquare:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        w, h = img.size; scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        pad_w = self.size - new_w; pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img_padded = TF.pad(img_resized, padding, fill=0)
        valid_mask = torch.zeros(self.size, self.size)
        valid_mask[padding[1]:padding[1]+new_h, padding[0]:padding[0]+new_w] = 1
        return img_padded, valid_mask

class ZeroShotDatasetBase(Dataset):
    def __init__(self, root_dir, resize=224, position_csv="prompt/position.csv"):
        self.root_dir = root_dir
        self.resize = resize
        self.resizer = ResizePadToSquare(resize)
        
        # <<< TẠO PIPELINE TRANSFORM HOÀN CHỈNH >>>
        # Pipeline này sẽ:
        # 1. Chuyển ảnh RGB đầu vào thành ảnh thang độ xám.
        # 2. Sao chép kênh thang độ xám đó thành 3 kênh để tương thích với model.
        # 3. Chuyển ảnh PIL thành Tensor PyTorch.
        self.image_transform = T.Compose([
            T.ToTensor()
        ])
        
        # Transform riêng cho mask (chỉ cần ToTensor)
        self.mask_transform = T.ToTensor()

        self.position_map = self._load_position_map(position_csv)
        self.image_info = []

    def _load_position_map(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"[WARN] position.csv not found at {csv_path}")
            return {}
        df = pd.read_csv(csv_path)
        return dict(zip(df['name'], df['position']))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        if idx >= len(self.image_info): raise IndexError("Index out of range")
        img_path, class_infos = self.image_info[idx]
        
        # 1. Mở ảnh. Vì ảnh đã là RGB, không cần .convert()
        image_pil = Image.open(img_path).convert("RGB")
        
        # 2. Resize và pad (vẫn thực hiện trên ảnh PIL)
        image_padded, valid_mask = self.resizer(image_pil)
        
        # 3. Áp dụng pipeline transform đã định nghĩa
        # Thao tác này sẽ xử lý cả việc chuyển sang grayscale và to_tensor
        image = self.image_transform(image_padded)

        # Chuẩn bị các tensor nhãn
        H, W = image.shape[1], image.shape[2]
        mask_tensor = torch.zeros((len(LEVEL3_NAMES), H, W), dtype=torch.float32)
        label_level1 = torch.zeros(len(LEVEL1_NAMES)); label_level2 = torch.zeros(len(LEVEL2_NAMES))
        label_level3 = torch.zeros(len(LEVEL3_NAMES)); label_pos = torch.zeros(len(POSITION_NAMES))

        # Xử lý các nhãn bệnh lý và mask
        for info in class_infos:
            class_name, l1, l2, mask_path = info["class_name"], info["level1"], info["level2"], info["mask_path"]
            if class_name in LEVEL3_MAP:
                c = LEVEL3_MAP[class_name]; label_level3[c] = 1.0
                if mask_path and os.path.exists(mask_path):
                    mask_pil = Image.open(mask_path).convert("L")
                    mask_padded, _ = self.resizer(mask_pil)
                    mask_tensor[c] = self.mask_transform(mask_padded).squeeze(0)
            if l1 in LEVEL1_MAP: label_level1[LEVEL1_MAP[l1]] = 1.0
            if l2 in LEVEL2_MAP: label_level2[LEVEL2_MAP[l2]] = 1.0

        # Xử lý nhãn vị trí
        has_position = False
        if class_infos:
            image_name = class_infos[0]["image_name"]
            if image_name in self.position_map:
                pos_str = self.position_map.get(image_name, None)
                if isinstance(pos_str, str) and pos_str.strip():
                    for pos in pos_str.split(','):
                        pos = pos.strip().lower()
                        if pos in POSITION_MAP and pos != 'unknown':
                            label_pos[POSITION_MAP[pos]] = 1.0; has_position = True
        if not has_position:
            label_pos[POSITION_MAP['unknown']] = 1.0

        return image, mask_tensor, label_level1, label_level2, label_level3, valid_mask, label_pos

# --- Các class ZeroShotTrainDataset và ZeroShotTestDataset không cần thay đổi ---
# Chúng sẽ tự động kế thừa logic xử lý ảnh mới từ ZeroShotDatasetBase.
class ZeroShotTrainDataset(ZeroShotDatasetBase):
    def __init__(self, root_dir="BTXRD_cleaned", excluded_class=None, resize=224, position_csv="prompt/position.csv"):
        super().__init__(root_dir, resize, position_csv); self.excluded_class = excluded_class
        self._build_dataset(exclude=True)
    def _build_dataset(self, exclude=True):
        image_dict = {}; 
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
                img_path = os.path.join(root, file); parts = img_path.split(os.sep)
                if "images" not in parts: continue
                try: l3, l2, l1 = parts[-3], parts[-4], parts[-5]
                except IndexError: continue
                if l3 == self.excluded_class: continue
                dir_name, image_filename = os.path.split(img_path)
                image_name_base, _ = os.path.splitext(image_filename)
                mask_dir = dir_name.replace(os.sep + "images", os.sep + "masks")
                search_pattern = os.path.join(mask_dir, f"{image_name_base}_mask.*")
                found_masks = glob.glob(search_pattern)
                mask_path = found_masks[0] if found_masks else ""
                image_name = os.path.basename(img_path)
                if img_path not in image_dict: image_dict[img_path] = []
                image_dict[img_path].append({"class_name": l3, "level1": l1, "level2": l2, "mask_path": mask_path, "image_name": image_name})
        filtered = {k: v for k, v in image_dict.items() if all(rec["class_name"] != self.excluded_class for rec in v)}
        self.image_info = list(filtered.items())
        print(f"✅ Train dataset loaded: {len(self.image_info)} samples (excluded: '{self.excluded_class}')")

class ZeroShotTestDataset(ZeroShotDatasetBase):
    def __init__(self, root_dir="BTXRD_cleaned", target_class=None, resize=224, position_csv="position.csv"):
        super().__init__(root_dir, resize, position_csv); self.target_class = target_class
        self._build_dataset()
    def _build_dataset(self):
        image_dict = {}; 
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
                img_path = os.path.join(root, file); parts = img_path.split(os.sep)
                if "images" not in parts: continue
                try: l3, l2, l1 = parts[-3], parts[-4], parts[-5]
                except IndexError: continue
                if l3 != self.target_class: continue
                dir_name, image_filename = os.path.split(img_path)
                image_name_base, _ = os.path.splitext(image_filename)
                mask_dir = dir_name.replace(os.sep + "images", os.sep + "masks")
                search_pattern = os.path.join(mask_dir, f"{image_name_base}_mask.*")
                found_masks = glob.glob(search_pattern)
                mask_path = found_masks[0] if found_masks else ""
                image_name = os.path.basename(img_path)
                if img_path not in image_dict: image_dict[img_path] = []
                image_dict[img_path].append({"class_name": l3, "level1": l1, "level2": l2, "mask_path": mask_path, "image_name": image_name})
        self.image_info = list(image_dict.items())
        print(f"✅ Test dataset loaded: {len(self.image_info)} samples for class '{self.target_class}'")