import pandas as pd

def loc_anh_da_chan_doan_excel(file_path):
    """
    Đọc file EXCEL (.xlsx) chứa thông tin ảnh và in ra những dòng có nhiều hơn 1
    chẩn đoán (giá trị '1') trong khoảng cột được chỉ định.
    
    Args:
        file_path (str): Đường dẫn đến file Excel của bạn.
    """
    start_column_name = 'osteochondroma'
    end_column_name = 'other mt'

    try:
        # --- THAY ĐỔI QUAN TRỌNG NHẤT LÀ Ở ĐÂY ---
        # Sử dụng pd.read_excel() để đọc file .xlsx
        # Thông thường không cần chỉ định 'encoding' với file Excel
        df = pd.read_excel(file_path) 
        
        all_columns = df.columns.tolist()

        try:
            start_index = all_columns.index(start_column_name)
            end_index = all_columns.index(end_column_name)
        except ValueError as e:
            print(f"Lỗi: Không tìm thấy tên cột trong file Excel.")
            print(f"Vui lòng kiểm tra xem cột '{start_column_name}' và '{end_column_name}' có tồn tại trong sheet đầu tiên không.")
            return

        cols_to_check = all_columns[start_index : end_index + 1]

        print("Chương trình sẽ kiểm tra các dòng có > 1 giá trị '1' trong các cột sau:")
        print(cols_to_check)
        print("-" * 50)

        df_subset = df[cols_to_check]
        # Chuyển đổi sang chuỗi để so sánh '1' cho chắc chắn
        count_ones = df_subset.astype(str).eq('1').sum(axis=1)
        ket_qua = df[count_ones > 1]

        if not ket_qua.empty:
            print("Đã tìm thấy các dòng thỏa mãn điều kiện (có nhiều hơn 1 chẩn đoán):")
            print(ket_qua.to_string())
        else:
            print("Không tìm thấy dòng nào thỏa mãn điều kiện.")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'")
    except Exception as e:
        print(f"Đã xảy ra một lỗi không mong muốn: {e}")
        print("Gợi ý: Hãy đảm bảo bạn đã cài đặt thư viện 'openpyxl' bằng lệnh: pip install openpyxl")

# --- CÁCH SỬ DỤNG ---
ten_file_csv = r"C:\Users\vanlo\Desktop\BTXRD-20250621T110139Z-1-001\BTXRD\dataset.xlsx"
loc_anh_da_chan_doan_excel(ten_file_csv)