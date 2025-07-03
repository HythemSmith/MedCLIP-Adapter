import os
import csv
from tqdm import tqdm

def create_metadata_file():
    """
    Quét qua các thư mục dataset nguồn, trích xuất thông tin về ảnh, mask, và lớp bệnh,
    sau đó ghi vào một file CSV duy nhất.
    """
    # --- CẤU HÌNH ---
    # Định nghĩa các thư mục dataset nguồn của bạn.
    # Mỗi mục là một tuple: (tên định danh, đường dẫn tuyệt đối)
    SOURCE_DIRECTORIES = [
        ('dataset_cleaned', r"C:\Users\vanlo\Desktop\dataset_cleaned"),
        ('BTXRD_cleaned', r"C:\Users\vanlo\Desktop\BTXRD_cleaned")
    ]

    # Đường dẫn đến file metadata.csv sẽ được tạo
    OUTPUT_CSV_PATH = "metadata.csv"

    # Các đuôi file ảnh hợp lệ
    VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

    print("🚀 Bắt đầu quá trình tạo file metadata...")

    metadata_records = []
    processed_images = set() # Dùng để tránh ghi trùng lặp nếu ảnh có ở cả 2 dataset

    for source_name, root_dir in SOURCE_DIRECTORIES:
        print(f"\n🔍 Đang quét thư mục: '{source_name}' tại '{root_dir}'")
        if not os.path.isdir(root_dir):
            print(f"⚠️  Cảnh báo: Thư mục không tồn tại, bỏ qua: {root_dir}")
            continue

        for dirpath, _, filenames in tqdm(list(os.walk(root_dir)), desc=f"Scanning {source_name}"):
            # Chỉ xử lý các thư mục có tên là 'images' ở cuối đường dẫn
            if not dirpath.endswith(os.sep + 'images'):
                continue

            relative_path = os.path.relpath(dirpath, root_dir)
            parts = relative_path.split(os.sep)

            level1, level2, level3 = None, None, None

            # Logic trích xuất các cấp nhãn từ đường dẫn tương đối
            # Ví dụ: bone_tumor/benign/osteosarcoma/images
            if len(parts) == 4 and parts[-1] == 'images':
                level1, level2, level3 = parts[0], parts[1], parts[2]
            # Ví dụ: trauma/broken/images
            elif len(parts) == 3 and parts[-1] == 'images':
                level1, level3 = parts[0], parts[1]
                level2 = 'other'  # Gán giá trị mặc định nếu không có level2
            else:
                continue # Bỏ qua nếu cấu trúc thư mục không như mong đợi

            for fname in filenames:
                file_ext = os.path.splitext(fname)[1].lower()
                if file_ext not in VALID_IMAGE_EXTENSIONS:
                    continue

                img_path = os.path.abspath(os.path.join(dirpath, fname))
                print(f"📷 Đang xử lý ảnh: {img_path}")
                # Tránh xử lý lại ảnh đã có trong dataset trước đó
                if img_path in processed_images:
                    continue
                processed_images.add(img_path)

                # Tìm đường dẫn mask tương ứng
                mask_path = img_path.replace("images", "masks")
                print(f"🗺️  Đang tìm mask cho ảnh: {mask_path}")
                base, _ = os.path.splitext(mask_path)
                found_mask_path = ""
                for ext in VALID_IMAGE_EXTENSIONS:
                    potential_mask_path = base + ext
                    if os.path.exists(potential_mask_path):
                        found_mask_path = potential_mask_path
                        break

                metadata_records.append({
                    'image_path': img_path,
                    'mask_path': found_mask_path,
                    'level1': level1,
                    'level2': level2,
                    'class_name': level3,  # class_name chính là level3
                    'source': source_name
                })

    print(f"\n✍️  Đang ghi {len(metadata_records)} bản ghi vào file '{OUTPUT_CSV_PATH}'...")
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['image_path', 'mask_path', 'level1', 'level2', 'class_name', 'source']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    print(f"\n✅ Hoàn thành! File metadata đã được tạo tại: {os.path.abspath(OUTPUT_CSV_PATH)}")

if __name__ == '__main__':
    create_metadata_file()