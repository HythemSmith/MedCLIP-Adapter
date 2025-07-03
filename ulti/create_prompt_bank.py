import os
import json
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# BƯỚC 1: CẤU HÌNH
# ==============================================================================

# --- THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP ---
DATASET_ROOT_PATH = r"C:\Users\vanlo\Desktop\dataset"  # Đường dẫn đến thư mục dataset gốc của bạn
OUTPUT_JSON_PATH = "prompt_bank.json"  # Tên file JSON output

# ==============================================================================
# BƯỚC 2: CÁC HÀM TIỆN ÍCH
# ==============================================================================

def format_class_name(name):
    """Chuyển tên lớp từ 'snake_case' thành 'human readable'."""
    return name.replace('_', ' ').replace('-', ' ')

def discover_hierarchy(root_dir):
    """Quét thư mục dataset để tìm cấu trúc phân cấp."""
    print("Discovering hierarchy from directory structure...")
    hierarchy_info = {}  # {l3_name: {'level1': l1, 'level2': l2}}
    level1_classes = set()
    level2_classes = set()

    # Sử dụng os.walk để duyệt qua cây thư mục
    for dirpath, dirnames, _ in tqdm(os.walk(root_dir), desc="Scanning dataset"):
        # Chúng ta tìm các thư mục lá (thư mục chứa 'images' hoặc 'masks')
        if 'images' in dirnames or 'masks' in dirnames:
            # Lấy đường dẫn tương đối và chuẩn hóa
            relative_path = os.path.relpath(dirpath, root_dir)
            parts = relative_path.split(os.sep)
            
            # Bỏ qua các thư mục không mong muốn hoặc ở gốc
            if parts == ['.']:
                continue

            l1, l2, l3 = None, None, None
            
            # Cấu trúc: L1/L2/L3
            if len(parts) == 3:
                l1, l2, l3 = parts
            # Cấu trúc: L1/L3 (ví dụ: misc/normal)
            elif len(parts) == 2:
                l1, l3 = parts
                l2 = 'none' # Sử dụng một giá trị đặc biệt để chỉ không có level 2
            else:
                # Bỏ qua các cấu trúc thư mục không mong muốn
                continue

            # Chỉ lưu thông tin nếu l3 chưa được xử lý
            if l3 and l3 not in hierarchy_info:
                hierarchy_info[l3] = {'level1': l1, 'level2': l2}
                level1_classes.add(l1)
                if l2 and l2 != 'none':
                    level2_classes.add(l2)
    
    print("✔ Hierarchy discovered.")
    return hierarchy_info, sorted(list(level1_classes)), sorted(list(level2_classes))


def generate_all_prompts(hierarchy_info, l1_names, l2_names):
    """Tạo ra một dictionary chứa tất cả các câu prompt cho tất cả các lớp."""
    print("Generating prompt strings...")
    # Sử dụng defaultdict(dict) để dễ dàng gán các key con
    all_prompts = defaultdict(dict)

    # --- Tạo prompt cho Level 3 (chi tiết nhất) ---
    for l3_name, info in hierarchy_info.items():
        l1_name = info['level1']
        l2_name = info['level2']
        
        l1_fmt = format_class_name(l1_name)
        l3_fmt = format_class_name(l3_name)
        
        # Tạo ngữ cảnh phân cấp
        if l2_name and l2_name != 'none':
            l2_fmt = format_class_name(l2_name)
            context = f"a type of {l2_fmt} {l1_fmt}"
        else:
            context = f"a type of {l1_fmt}"

        all_prompts[l3_name]['prompts'] = [
            f"An X-ray of {l3_fmt}, which is {context}.",
            f"Radiograph showing a case of {l3_fmt}.",
            f"This medical scan presents a finding of {l3_fmt}."
        ]
        all_prompts[l3_name]['level1'] = l1_name
        all_prompts[l3_name]['level2'] = l2_name

    # --- Tạo prompt cho Level 2 (tổng quát hơn) ---
    for l2_name in l2_names:
        l2_fmt = format_class_name(l2_name)
        all_prompts[l2_name]['prompts'] = [
            f"An X-ray showing a {l2_fmt} condition or finding.",
            f"This radiograph shows a {l2_fmt} lesion.",
            f"A medical scan indicating a {l2_fmt} process."
        ]
        
    # --- Tạo prompt cho Level 1 (tổng quát nhất) ---
    for l1_name in l1_names:
        l1_fmt = format_class_name(l1_name)
        all_prompts[l1_name]['prompts'] = [
            f"An X-ray image showing a type of {l1_fmt}.",
            f"This medical scan presents a finding related to {l1_fmt}.",
            f"A radiograph with signs of {l1_fmt}."
        ]

    print("✔ All prompts generated.")
    return all_prompts


# ==============================================================================
# BƯỚC 3: HÀM CHÍNH ĐỂ THỰC THI
# ==============================================================================

def main():
    # 1. Khám phá cấu trúc thư mục
    hierarchy_info, level1_names, level2_names = discover_hierarchy(DATASET_ROOT_PATH)
    
    # In ra để kiểm tra
    print("\n--- Discovered Hierarchy ---")
    print(f"Level 1 classes found: {level1_names}")
    print(f"Level 2 classes found: {level2_names}")
    print(f"Level 3 classes found: {list(hierarchy_info.keys())}")
    
    # 2. Tạo tất cả các câu prompt cần thiết
    prompts_to_save = generate_all_prompts(hierarchy_info, level1_names, level2_names)
    
    # 3. Lưu dictionary chứa prompt vào file JSON
    print(f"\nSaving generated prompts to: {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        # indent=4 để file JSON dễ đọc hơn
        json.dump(prompts_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"✔ Prompts successfully saved to {OUTPUT_JSON_PATH}.")
    print("You can now use this JSON file to generate text embeddings in a separate step.")


if __name__ == "__main__":
    main()