import os

def print_tree(path, prefix=""):
    if not os.path.isdir(path):
        print(f"{path} không phải là thư mục hợp lệ.")
        return

    folder_name = os.path.basename(os.path.abspath(path))
    print(folder_name + "/")

    def inner_print(current_path, prefix):
        items = sorted(
            [item for item in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, item))]
        )
        count = len(items)
        for index, item in enumerate(items):
            connector = "├── " if index < count - 1 else "└── "
            print(prefix + connector + item + "/")
            new_prefix = prefix + ("│   " if index < count - 1 else "    ")
            inner_print(os.path.join(current_path, item), new_prefix)

    inner_print(path, prefix)

# Sử dụng
if __name__ == "__main__":
    print_tree(r"C:\Users\vanlo\Desktop\BTXRD_cleaned")
