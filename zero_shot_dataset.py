import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
from torchvision.transforms import functional as TF
import pandas as pd

# Các hằng số định nghĩa lớp không thay đổi
LEVEL1_NAMES = ['bone_tumor', 'degenerative', 'misc', 'trauma']
LEVEL2_NAMES = ['benign', 'malignant']
LEVEL3_NAMES = [
    'giant_cell_tumor', 'multiple_osteochondromas', 'osteochondroma', 'osteofibroma',
    'other_bt', 'simple_bone_cyst', 'synovial_osteochondroma', 'osteosarcoma', 'other_mt',
    'disc_space_narrowing', 'foraminal_stenosis', 'osteophytes', 'normal', 'other_lesions',
    'spondylolysthesis', 'surgical_implant', 'vertebral_collapse', 'broken'
]
LEVEL1_MAP = {name: i for i, name in enumerate(LEVEL1_NAMES)}
LEVEL2_MAP = {name: i for i, name in enumerate(LEVEL2_NAMES)}
LEVEL3_MAP = {name: i for i, name in enumerate(LEVEL3_NAMES)}

# Lớp tiện ích để resize ảnh không thay đổi
class ResizePadToSquare:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize ảnh
        img_resized = img.resize((new_w, new_h), resample=self.interpolation)
        
        # Tính toán padding
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        # Áp dụng padding cho ảnh
        img_padded = TF.pad(img_resized, padding, fill=0)

        # Tạo valid_region_mask
        # Mask này sẽ có giá trị 1 ở vùng ảnh gốc và 0 ở vùng padding
        valid_mask = torch.zeros(self.size, self.size, dtype=torch.float32)
        valid_mask[pad_h // 2 : pad_h // 2 + new_h, pad_w // 2 : pad_w // 2 + new_w] = 1
        
        return img_padded, valid_mask

# --- Lớp Dataset cơ sở được viết lại hoàn toàn ---
class MedicalImageDatasetBase(Dataset):
    def __init__(self, metadata_path, transform=None, resize=224, dataset_source_filter=None):
        # Tải toàn bộ file metadata bằng pandas
        try:
            df_all = pd.read_csv(metadata_path)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file metadata tại '{metadata_path}'.")
            print("Vui lòng chạy script 'create_metadata.py' trước.")
            raise

        # Lọc theo nguồn dataset nếu được chỉ định (ví dụ: 'large_dataset' hoặc 'cleaned_dataset')
        if dataset_source_filter:
            print(f"Đang lọc dataset theo nguồn: '{dataset_source_filter}'")
            self.df = df_all[df_all['source'] == dataset_source_filter].copy()
        else:
            self.df = df_all.copy()
        
        print(f"Đã tải {len(self.df)} bản ghi từ metadata.")

        # Các phép biến đổi (transforms) không thay đổi
        self.resizer = ResizePadToSquare(resize, interpolation=Image.BICUBIC)
        self.resizer_mask = ResizePadToSquare(resize, interpolation=Image.NEAREST)

        self.transform_x = T.ToTensor()
        self.transform_mask_to_tensor = T.ToTensor()

        # Các thuộc tính này sẽ được các lớp con (subclass) điền vào
        self.image_paths = []
        self.image_info_map = {}

    def _prepare_data(self, filtered_df):
        """
        Hàm trợ giúp để các lớp con gọi sau khi đã lọc xong dataframe.
        Nó nhóm các bản ghi theo đường dẫn ảnh để xử lý đa nhãn.
        """
        # Nhóm theo 'image_path' để xử lý các ảnh có nhiều nhãn
        self.image_info_map = filtered_df.groupby('image_path').apply(lambda x: x.to_dict('records')).to_dict()
        self.image_paths = list(self.image_info_map.keys())
        print(f"Đã chuẩn bị dataset với {len(self.image_paths)} ảnh độc nhất.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        info_list = self.image_info_map[img_path]

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file ảnh tại {img_path}")
            return None

        # Áp dụng resizer để nhận ảnh đã pad và valid_mask
        image_pil_padded, valid_region_mask = self.resizer(image_pil)
        image = self.transform_x(image_pil_padded)

        H, W = image.shape[1], image.shape[2]
        mask_tensor = torch.zeros((len(LEVEL3_NAMES), H, W))
        label_level1 = torch.zeros(len(LEVEL1_NAMES))
        label_level2 = torch.zeros(len(LEVEL2_NAMES))
        label_level3 = torch.zeros(len(LEVEL3_NAMES))

        for record in info_list:
            class_name = record['class_name']
            mask_path = record['mask_path']
            l1_name = record.get('level1')
            l2_name = record.get('level2')

            if class_name not in LEVEL3_MAP:
                continue
            
            l3_idx = LEVEL3_MAP[class_name]
            label_level3[l3_idx] = 1

            if l1_name and l1_name in LEVEL1_MAP:
                label_level1[LEVEL1_MAP[l1_name]] = 1
            if l2_name and l2_name in LEVEL2_MAP:
                label_level2[LEVEL2_MAP[l2_name]] = 1

            if mask_path and pd.notna(mask_path) and os.path.exists(mask_path):
                try:
                    mask_pil = Image.open(mask_path).convert("L")
                    # Chỉ resize mask, không cần valid_mask của mask
                    mask_pil_padded, _ = self.resizer_mask(mask_pil)
                    mask = self.transform_mask_to_tensor(mask_pil_padded)
                    mask_tensor[l3_idx] = mask.squeeze(0)
                except Exception as e:
                    print(f"Lỗi khi xử lý mask {mask_path}: {e}")

        # Trả về thêm valid_region_mask
        return image, mask_tensor, label_level1, label_level2, label_level3, valid_region_mask

# --- Lớp MedicalTrainDataset được viết lại ---
class MedicalTrainDataset(MedicalImageDatasetBase):
    def __init__(self, metadata_path, excluded_class=None, transform=None, resize=224, dataset_source_filter=None):
        super().__init__(metadata_path, transform=transform, resize=resize, dataset_source_filter=dataset_source_filter)

        df_filtered = self.df
        if excluded_class:
            print(f"Loại bỏ lớp '{excluded_class}' cho quá trình huấn luyện zero-shot.")
            paths_to_exclude = df_filtered[df_filtered['class_name'] == excluded_class]['image_path'].unique()
            df_filtered = df_filtered[~df_filtered['image_path'].isin(paths_to_exclude)]
            print(f"Đã loại bỏ {len(paths_to_exclude)} ảnh chứa lớp bị loại trừ.")

        self._prepare_data(df_filtered)

# --- Lớp MedicalTestDataset được viết lại ---
class MedicalTestDataset(MedicalImageDatasetBase):
    def __init__(self, metadata_path, target_class, transform=None, resize=224, dataset_source_filter=None):
        super().__init__(metadata_path, transform=transform, resize=resize, dataset_source_filter=dataset_source_filter)
        
        print(f"Tạo tập test cho lớp unseen: '{target_class}'")
        
        df_source_filtered = self.df
        test_image_paths = df_source_filtered[df_source_filtered['class_name'] == target_class]['image_path'].unique()
        
        # Lấy TẤT CẢ các nhãn cho những ảnh được chọn vào tập test
        df_filtered = df_source_filtered[df_source_filtered['image_path'].isin(test_image_paths)]
        
        print(f"Đã tìm thấy {len(test_image_paths)} ảnh để kiểm thử.")
        self._prepare_data(df_filtered)
