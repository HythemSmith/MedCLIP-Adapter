import os
import csv
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
# Thêm các đường dẫn đến thư mục gốc của dataset vào đây
# Ví dụ: ['C:/Users/vanlo/Desktop/dataset_cleaned', 'C:/Users/vanlo/Desktop/BTXRD_cleaned']
DATA_ROOTS = [
    r'C:\Users\vanlo\Desktop\dataset_cleaned',
    r'C:\Users\vanlo\Desktop\BTXRD_cleaned'
]

# Đường dẫn file CSV đầu ra
OUTPUT_CSV_PATH = 'metadata.csv'

# Các phần mở rộng file ảnh và mask cần tìm
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
MASK_SUFFIX = '_mask'
MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg')
# --- KẾT THÚC CẤU HÌNH ---

def find_mask_path(image_path: Path) -> str:
    """
    Tìm đường dẫn file mask tương ứng với file ảnh.
    Quy tắc: IMG001.jpg -> .../masks/IMG001_mask.png
    """
    image_stem = image_path.stem
    mask_dir = image_path.parent.parent / 'masks'

    if not mask_dir.exists():
        return ''

    for ext in MASK_EXTENSIONS:
        potential_mask_name = f"{image_stem}{MASK_SUFFIX}{ext}"
        potential_mask_path = mask_dir / potential_mask_name
        if potential_mask_path.exists():
            return str(potential_mask_path.resolve())
            
    return ''

def main():
    """
    Quét các thư mục trong DATA_ROOTS và tạo file metadata.csv.
    """
    print(f"Bắt đầu quét các thư mục: {DATA_ROOTS}")
    all_data = []
    
    for root_str in DATA_ROOTS:
        root_path = Path(root_str)
        if not root_path.exists():
            print(f"⚠️  Cảnh báo: Đường dẫn không tồn tại, bỏ qua: {root_path}")
            continue

        source_name = root_path.name
        print(f"Đang xử lý nguồn: {source_name}")

        # Tìm tất cả các file ảnh trong các thư mục con 'images'
        image_files = list(root_path.rglob('images/*'))
        
        for image_path in tqdm(image_files, desc=f"  -> Quét {source_name}"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    parts = image_path.parts
                    source_index = parts.index(source_name)
                    
                    level1 = parts[source_index + 1]
                    class_name = parts[source_index + 3]

                    # Logic đặc biệt cho level2 dựa trên level1
                    if level1 in ['degenerative', 'trauma', 'misc']:
                        level2 = 'other'
                    else:
                        level2 = parts[source_index + 2]

                    mask_path = find_mask_path(image_path)

                    all_data.append({
                        'image_path': str(image_path.resolve()),
                        'mask_path': mask_path,
                        'level1': level1,
                        'level2': level2,
                        'class_name': class_name,
                        'source': source_name
                    })
                except (ValueError, IndexError) as e:
                    print(f"\n❌ Lỗi khi xử lý đường dẫn: {image_path}. Cấu trúc thư mục có thể không đúng. Lỗi: {e}")

    if not all_data:
        print("Không tìm thấy file ảnh nào. Vui lòng kiểm tra lại cấu hình DATA_ROOTS.")
        return

    print(f"\nTổng cộng tìm thấy {len(all_data)} mẫu. Đang ghi ra file {OUTPUT_CSV_PATH}...")
    
    # Ghi dữ liệu ra file CSV
    try:
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'mask_path', 'level1', 'level2', 'class_name', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"✅ Hoàn thành! File '{OUTPUT_CSV_PATH}' đã được tạo thành công.")
    except IOError as e:
        print(f"❌ Lỗi khi ghi file CSV: {e}")

if __name__ == '__main__':
    main()

