# generate_prompts.py

import os
import json
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# BƯỚC 1: CẤU HÌNH
# ==============================================================================

# --- THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP ---
# Đường dẫn đến thư mục dataset gốc của bạn
DATASET_ROOT_PATH = r"C:\Users\vanlo\Desktop\dataset"
# Tên file JSON output sẽ được tạo ra
OUTPUT_JSON_PATH = "prompt_bank.json" 

# ==============================================================================
# BƯỚC 2: CÁC HÀM TIỆN ÍCH
# ==============================================================================

def format_class_name(name):
    """
    Chuyển tên lớp từ dạng 'snake_case' hoặc 'kebab-case' 
    thành dạng con người có thể đọc được, ví dụ: 'giant_cell_tumor' -> 'giant cell tumor'.
    """
    return name.replace('_', ' ').replace('-', ' ')

def discover_hierarchy(root_dir):
    """
    Quét thư mục dataset để tự động tìm cấu trúc phân cấp (L1, L2, L3)
    và danh sách các lớp ở mỗi cấp độ.
    """
    print("🔄 Discovering hierarchy from directory structure...")
    
    # {l3_name: {'level1': l1_name, 'level2': l2_name}}
    hierarchy_info = {} 
    
    level1_classes = set()
    level2_classes = set()
    level3_classes = set()

    # Sử dụng os.walk để duyệt qua toàn bộ cây thư mục một cách hiệu quả
    for dirpath, dirnames, _ in tqdm(os.walk(root_dir), desc="Scanning dataset"):
        # Chúng ta chỉ quan tâm đến các thư mục chứa dữ liệu ảnh
        if 'images' not in dirnames and 'masks' not in dirnames:
            continue
            
        # Lấy đường dẫn tương đối so với thư mục gốc và chuẩn hóa
        relative_path = os.path.relpath(dirpath, root_dir)
        parts = relative_path.split(os.sep)
        
        # Bỏ qua thư mục gốc
        if parts == ['.']:
            continue
            
        l1, l2, l3 = None, None, None
        
        # Phân tích đường dẫn để xác định các level
        if len(parts) == 3: # Cấu trúc đầy đủ: L1/L2/L3
            l1, l2, l3 = parts
        elif len(parts) == 2: # Cấu trúc thiếu L2: L1/L3
            l1, l3 = parts
            l2 = 'none' # Sử dụng một giá trị đặc biệt để chỉ không có L2
        else:
            # Bỏ qua các cấu trúc không mong muốn (ví dụ: quá nông hoặc quá sâu)
            continue
        
        # Thêm các lớp đã khám phá vào các tập hợp (set) để đảm bảo tính duy nhất
        if l1: level1_classes.add(l1)
        if l2 and l2 != 'none': level2_classes.add(l2)
        if l3: level3_classes.add(l3)

        # Lưu thông tin phân cấp cho lớp Level 3
        if l3 and l3 not in hierarchy_info:
            hierarchy_info[l3] = {'level1': l1, 'level2': l2}
            
    print("✔ Hierarchy discovered.")
    # Trả về thông tin phân cấp và danh sách các lớp đã được sắp xếp
    return hierarchy_info, sorted(list(level1_classes)), sorted(list(level2_classes)), sorted(list(level3_classes))


def generate_all_prompts(hierarchy_info, l1_names, l2_names, l3_names):
    """
    Tạo ra một dictionary chứa tất cả các câu prompt cho tất cả các lớp ở cả 3 level.
    """
    print("✍️ Generating prompt strings for all levels...")
    all_prompts = defaultdict(dict)

    # --- A. TẠO PROMPT CHO LEVEL 3 (CHI TIẾT NHẤT) ---
    # Các prompt này sẽ chứa ngữ cảnh từ các level cao hơn
    for l3_name in l3_names:
        info = hierarchy_info.get(l3_name)
        if not info:
            print(f"Warning: No hierarchy info found for L3 class '{l3_name}'. Skipping.")
            continue

        l1_name = info['level1']
        l2_name = info['level2']
        
        l1_fmt = format_class_name(l1_name)
        l3_fmt = format_class_name(l3_name)
        
        # Tạo ngữ cảnh phân cấp thông minh
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
        # Lưu lại thông tin phân cấp để dễ dàng tra cứu sau này
        all_prompts[l3_name]['level1'] = l1_name
        all_prompts[l3_name]['level2'] = l2_name

    # --- B. TẠO PROMPT CHO LEVEL 2 (TỔNG QUÁT HƠN) ---
    for l2_name in l2_names:
        l2_fmt = format_class_name(l2_name)
        all_prompts[l2_name]['prompts'] = [
            f"An X-ray showing a {l2_fmt} condition or finding.",
            f"This radiograph shows a {l2_fmt} lesion.",
            f"A medical scan indicating a {l2_fmt} process."
        ]
        
    # --- C. TẠO PROMPT CHO LEVEL 1 (TỔNG QUÁT NHẤT) ---
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
    """Hàm chính điều phối toàn bộ quy trình."""
    
    # 1. Khám phá cấu trúc thư mục để lấy thông tin về các lớp
    hierarchy_info, level1_names, level2_names, level3_names = discover_hierarchy(DATASET_ROOT_PATH)
    
    # In ra thông tin đã khám phá để người dùng kiểm tra
    print("\n--- Discovered Hierarchy Summary ---")
    print(f"Total Level 1 classes found: {len(level1_names)} -> {level1_names}")
    print(f"Total Level 2 classes found: {len(level2_names)} -> {level2_names}")
    print(f"Total Level 3 classes found: {len(level3_names)}")
    
    # 2. Dựa trên thông tin đã khám phá, tạo ra dictionary chứa các câu prompt
    prompts_to_save = generate_all_prompts(hierarchy_info, level1_names, level2_names, level3_names)
    
    # 3. Lưu dictionary này vào một file JSON có định dạng đẹp, dễ đọc
    print(f"\n💾 Saving generated prompts to: {OUTPUT_JSON_PATH}...")
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            # indent=4 giúp file JSON được định dạng với thụt lề 4 dấu cách, rất dễ đọc
            json.dump(prompts_to_save, f, ensure_ascii=False, indent=4)
            
        print(f"✔ Prompts successfully saved to '{OUTPUT_JSON_PATH}'.")
        print(f"Total unique classes with prompts generated: {len(prompts_to_save)}")
        print("\nNext steps:")
        print("1. Review the generated 'prompts.json' file.")
        print("2. Run 'prompt_encoder.py' to convert these text prompts into embeddings and create 'prompt_cache.pt'.")

    except Exception as e:
        print(f"\n❌ Error saving JSON file: {e}")


if __name__ == "__main__":
    main()