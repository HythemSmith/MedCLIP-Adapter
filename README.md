# README: Bone Tumor Classification Prompt Pipeline

## 1️⃣ Mục tiêu dự án

Xây dựng pipeline AI giúp phân loại bệnh lý xương từ ảnh X-quang, trả về:

* Loại bệnh (diagnosis)
* Vị trí bệnh (anatomical location)
* Mức độ lành/ác tính (benign/malignant)

## 2️⃣ Các bước đã thực hiện

### 2.1 prompt.py

* Đọc file `dataset.xlsx` (sheet 1) chứa metadata ảnh, nhãn bệnh và vị trí.
* Chọn lọc **chỉ các cột bệnh lý** (osteosarcoma, osteochondroma, giant cell tumor, …)
* Loại bỏ các cột anatomical region làm key chính (như "ulna", "femur")
* Sinh prompt hợp lý, ví dụ:

  * *"An X-ray showing a case of osteosarcoma affecting the tibia, a malignant bone tumor."*
* Gán `level1` (bone\_tumor), `level2` (benign/malignant) chính xác theo từng bệnh.
* Xuất file `prompts.json` chứa dict:

```json
{
  "osteosarcoma": {
     "prompts": [...],
     "position": ["femur", "tibia", ...],
     "level1": "bone_tumor",
     "level2": "malignant"
  },
  ...
}
```

### 2.2 prompt\_encoder.py

* Đọc lại `prompts.json`
* Sử dụng **BioClinicalBERT** từ MedCLIP làm text encoder
* Mean pooling toàn bộ token thay vì chỉ dùng CLS token, giúp mô tả tốt hơn thông tin vị trí
* Lưu lại các field:

  * `prompts`
  * `embedding`
  * `level1`
  * `level2`
  * `position`
* Normalize embedding
* Lưu thành file `prompt_cache.pt` để tăng tốc training và zero-shot retrieval.

### 2.3 check\_consistency.py

* Kiểm tra dữ liệu:

  * Ảnh phải có đúng 1 nhãn bệnh active
  * Ảnh phải có đúng 1 vị trí active
  * Kiểm tra file mask tồn tại khớp tên với ảnh
* Ghi lại báo cáo `consistency_report.csv` để dễ dàng rà soát dữ liệu trước khi huấn luyện.

## 3️⃣ Quy trình huấn luyện

✅ Bước 1: chạy `check_consistency.py` để chắc chắn dữ liệu không lỗi
✅ Bước 2: chạy `prompt.py` để sinh `prompts.json`
✅ Bước 3: chạy `prompt_encoder.py` để sinh `prompt_cache.pt`
✅ Bước 4: dùng `prompt_cache.pt` trong training pipeline để embedding text prompt.

## 4️⃣ Ghi chú mở rộng

* Nếu sau này bạn bổ sung prompt dạng case report (triệu chứng, tuổi, giới), có thể mở rộng `prompt.py`
* Nếu dataset có thêm bounding box hoặc mask chi tiết (epiphysis, metaphysis), có thể update `position` sâu hơn.

---

**Liên hệ hỗ trợ**: (bạn điền email dev tại đây nếu cần)

🚀 *Chúc bạn build mô hình bone tumor AI thành công!*
