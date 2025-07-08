# FILE: zero_shot_dataset.py
# VERSION: Hoàn chỉnh với Data Augmentation

import os
import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchvision.transforms as T
import numpy as np # Thêm import numpy
# Thêm import cho Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# --- Các hằng số định nghĩa lớp không thay đổi ---
# Đảm bảo các danh sách này khớp với cấu hình trong file train.py
LEVEL1_NAMES = ['bone_tumor', 'degenerative', 'misc', 'trauma']
LEVEL2_NAMES = ['benign', 'malignant']
LEVEL3_NAMES = [
    'giant_cell_tumor', 'multiple_osteochondromas', 'osteochondroma', 'osteofibroma',
    'other_bt', 'simple_bone_cyst', 'synovial_osteochondroma', 'osteosarcoma', 'other_mt',
    'disc_space_narrowing', 'foraminal_stenosis', 'osteophytes', 'normal', 'other_lesions',
    'spondylolysthesis', 'surgical_implant', 'vertebral_collapse', 'broken'
]
POSITION_NAMES = [
    'hand', 'ulna', 'radius', 'humerus', 'foot', 'tibia', 'fibula', 'femur', 'pelvis'
]
LEVEL1_MAP = {name: i for i, name in enumerate(LEVEL1_NAMES)}
LEVEL2_MAP = {name: i for i, name in enumerate(LEVEL2_NAMES)}
LEVEL3_MAP = {name: i for i, name in enumerate(LEVEL3_NAMES)}
# Bản đồ vị trí tương ứng với chỉ số trong tensor
POSITION_MAP = {name: i for i, name in enumerate(POSITION_NAMES)}

def get_image_position(image_name, csv_path=r'prompt\position.csv'):
    try:
        # Load the position.csv file
        df = pd.read_csv(csv_path)
        
        # Create a dictionary mapping image names to positions
        position_map = df.set_index('name')['position'].to_dict()
        
        # Return the position for the given image name, or None if not found
        return position_map.get(image_name)
    
    except FileNotFoundError:
        print(f"Error: Could not find file at '{csv_path}'.")
        return None
    except Exception as e:
        print(f"Error: Failed to process position.csv. {str(e)}")
        return None

class ResizePadToSquare:
    """
    Resize ảnh về kích thước mong muốn mà vẫn giữ tỷ lệ, sau đó thêm padding.
    Đồng thời trả về một "valid_region_mask" để xác định vùng ảnh gốc.
    """
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        # Giữ tỷ lệ khung hình
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize ảnh
        img_resized = img.resize((new_w, new_h), resample=self.interpolation)
        
        # Tính toán padding để đưa ảnh vào giữa
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        # Áp dụng padding
        img_padded = TF.pad(img_resized, padding, fill=0)

        # Tạo valid_region_mask: 1 cho vùng ảnh gốc, 0 cho vùng padding
        valid_mask = torch.zeros(self.size, self.size, dtype=torch.float32)
        valid_mask[pad_h // 2 : pad_h // 2 + new_h, pad_w // 2 : pad_w // 2 + new_w] = 1
        
        return img_padded, valid_mask


# --- Lớp Dataset cơ sở đã được tích hợp Data Augmentation ---
class MedicalImageDatasetBase(Dataset):
    def __init__(self, metadata_path, resize=224, dataset_source_filter=None, is_train=False):
        try:
            df_all = pd.read_csv(metadata_path)
        except FileNotFoundError:

            print(f"Lỗi: Không tìm thấy file metadata tại '{metadata_path}'.")
            raise

        if dataset_source_filter:
            self.df = df_all[df_all['source'] == dataset_source_filter].copy()
        else:
            self.df = df_all.copy()
        
        print(f"Đã tải {len(self.df)} bản ghi từ metadata.")
        self.resize = resize
        self.is_train = is_train

        # Pipeline Augmentation đơn giản và an toàn
        if self.is_train:
            print("INFO: Chế độ Train - Bật Augmentation (phiên bản đơn giản hóa).")
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=0, value=0),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.LongestMaxSize(max_size=self.resize),
                A.PadIfNeeded(min_height=self.resize, min_width=self.resize, border_mode=0, value=0),
                ToTensorV2(),
            ])
        else:
            print("INFO: Chế độ Test/Validation - Chỉ Resize và Pad.")
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=self.resize),
                A.PadIfNeeded(min_height=self.resize, min_width=self.resize, border_mode=0, value=0),
                ToTensorV2(),
            ])
        self.image_paths = []
        self.image_info_map = {}

    def _prepare_data(self, filtered_df):
        self.image_info_map = filtered_df.groupby('image_path').apply(lambda x: x.to_dict('records')).to_dict()
        self.image_paths = list(self.image_info_map.keys())
        print(f"Đã chuẩn bị dataset với {len(self.image_paths)} ảnh độc nhất.")

    def __len__(self):
        return len(self.image_paths)
    
    def _find_xray_film_bbox(self, pil_image):
        """
        Logic crop mạnh mẽ hơn: Đảo ngược ảnh để tìm phim X-quang.
        """
        gray_image = np.array(pil_image.convert('L'))
        
        # Đảo ngược ảnh: nền trắng -> đen, phim đen -> trắng
        inverted_image = cv2.bitwise_not(gray_image)
        
        # Ngưỡng hóa để làm nổi bật phim (giờ đã là màu trắng)
        _, thresh = cv2.threshold(inverted_image, 50, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Tìm contour lớn nhất, đây chính là phim X-quang
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Lấy bounding box của nó
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Thêm một chút padding để không bị cắt sát viền (tùy chọn)
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + (padding * 2)
        h = h + (padding * 2)
        
        return (x, y, x + w, y + h)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        info_list = self.image_info_map[img_path]
        img_name = os.path.basename(img_path)
        pos_names = get_image_position(img_name)
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IOError):
            return None

        # 1. TÍNH TOÁN BBOX TỪ ẢNH GỐC BẰNG LOGIC MỚI
        bbox = self._find_xray_film_bbox(image_pil)
        if bbox is None:
            print(f"Cảnh báo: Không tìm thấy phim X-quang trong ảnh {img_path}. Bỏ qua.")
            return None

        # 2. CROP ẢNH VÀ MASK BẰNG CÙNG MỘT BBOX
        image_pil_cropped = image_pil.crop(bbox)
        
        record = info_list[0]
        mask_path = record.get('mask_path')
        mask_pil_cropped = None
        if mask_path and pd.notna(mask_path) and os.path.exists(mask_path):
            try:
                mask_pil = Image.open(mask_path).convert("L")
                mask_pil_cropped = mask_pil.crop(bbox)
            except Exception:
                mask_pil_cropped = None
        
        if mask_pil_cropped is None:
            mask_pil_cropped = Image.new('L', image_pil_cropped.size, 0)

        # 3. CHUYỂN SANG NUMPY ĐỂ ĐƯA VÀO ALBUMENTATIONS
        image_np = np.array(image_pil_cropped)
        mask_np = np.array(mask_pil_cropped)

        # 4. ÁP DỤNG AUGMENTATION
        augmented = self.transform(image=image_np, mask=mask_np)
        image = augmented['image'].float() / 255.0
        mask_transformed = augmented['mask'].float().unsqueeze(0)

        # 5. BỎ SOFT-MASKING (Không cần thiết nữa vì ảnh đầu vào đã rất sạch)
        # Chỉ cần tạo valid_region_mask để dùng trong vòng lặp huấn luyện
        valid_region_mask = (image[0, :, :] > 0).float()
        
        # ... Phần còn lại của hàm để tạo label và mask_tensor không đổi
        H, W = image.shape[1], image.shape[2]
        mask_tensor = torch.zeros((len(LEVEL3_NAMES), H, W), dtype=torch.float32)
        label_level1 = torch.zeros(len(LEVEL1_NAMES), dtype=torch.float32)
        label_level2 = torch.zeros(len(LEVEL2_NAMES), dtype=torch.float32)
        label_level3 = torch.zeros(len(LEVEL3_NAMES), dtype=torch.float32)
        label_pos = torch.zeros(len(POSITION_NAMES), dtype=torch.float32)

        for rec in info_list:
            class_name = rec.get('class_name')
            l1_name = rec.get('level1')
            l2_name = rec.get('level2')
            pos_names = rec.get('position')
            print(f"Processing position: {pos_names}")
            if class_name and class_name in LEVEL3_MAP:
                l3_idx = LEVEL3_MAP[class_name]
                label_level3[l3_idx] = 1.0
                if mask_transformed is not None:
                     mask_tensor[l3_idx] = mask_transformed.squeeze(0)
            if l1_name and l1_name in LEVEL1_MAP:
                label_level1[LEVEL1_MAP[l1_name]] = 1.0
            if l2_name and l2_name in LEVEL2_MAP:
                label_level2[LEVEL2_MAP[l2_name]] = 1.0
            for pos_name in pos_names.split(','):
                
                pos_name = pos_name.strip().lower()
                if pos_name in POSITION_MAP:
                    label_pos[POSITION_MAP[pos_name]] = 1.0
        return image, mask_tensor, label_level1, label_level2, label_level3, valid_region_mask, label_pos


# --- Lớp MedicalTrainDataset kế thừa và BẬT augmentation ---
class MedicalTrainDataset(MedicalImageDatasetBase):
    def __init__(self, metadata_path, excluded_class=None, resize=224, dataset_source_filter=None):
        # Truyền is_train=True vào lớp cha để bật augmentation
        super().__init__(
            metadata_path, 
            resize=resize, 
            dataset_source_filter=dataset_source_filter, 
            is_train=True
        )

        df_filtered = self.df
        if excluded_class:
            print(f"Loại bỏ lớp '{excluded_class}' cho quá trình huấn luyện zero-shot.")
            # Tìm tất cả các ảnh có chứa lớp bị loại trừ
            paths_to_exclude = df_filtered[df_filtered['class_name'] == excluded_class]['image_path'].unique()
            # Loại bỏ tất cả các dòng có đường dẫn ảnh đó
            df_filtered = df_filtered[~df_filtered['image_path'].isin(paths_to_exclude)]
            print(f"Đã loại bỏ {len(paths_to_exclude)} ảnh chứa lớp bị loại trừ.")

        self._prepare_data(df_filtered)


# --- Lớp MedicalTestDataset kế thừa và TẮT augmentation ---
class MedicalTestDataset(MedicalImageDatasetBase):
    def __init__(self, metadata_path, target_class, resize=224, dataset_source_filter=None):
        # Truyền is_train=False vào lớp cha để tắt augmentation
        super().__init__(
            metadata_path, 
            resize=resize, 
            dataset_source_filter=dataset_source_filter, 
            is_train=False
        )
        
        print(f"Tạo tập test cho lớp unseen: '{target_class}'")
        
        df_source_filtered = self.df
        # Tìm các ảnh thuộc lớp mục tiêu
        test_image_paths = df_source_filtered[df_source_filtered['class_name'] == target_class]['image_path'].unique()
        
        # Lấy tất cả các nhãn (đa nhãn) cho những ảnh đã được chọn vào tập test
        df_filtered = df_source_filtered[df_source_filtered['image_path'].isin(test_image_paths)]
        
        print(f"Đã tìm thấy {len(test_image_paths)} ảnh để kiểm thử.")
        self._prepare_data(df_filtered)