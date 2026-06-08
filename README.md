# Đồ án Môn học: Nhận diện và Phân loại Rác Thải Tự Động
*(Môn: Học Sâu - Deep Learning)*

Đây là mã nguồn chính thức cho Đồ án cuối kỳ, được thiết kế bám sát 100% yêu cầu Rubric: Tối ưu hóa tài nguyên (Data-centric AI, Transfer Learning) và Quản lý thực nghiệm (W&B, Checkpoints, Reproducibility).

## 1. Yêu cầu Hệ thống
- Python 3.9+
- Khuyến khích thực thi trên **Kaggle** để tận dụng tài nguyên GPU T4.

## 2. Hướng dẫn Triển khai trên Kaggle

### Bước 1: Khởi tạo Môi trường
1. Khởi tạo một Notebook mới trên Kaggle.
2. Kích hoạt GPU: **Session Options** -> **Accelerator** -> Chọn **GPU T4 x2**.
3. Thêm bộ dữ liệu TACO gốc bằng cách bấm **Add Data**, tìm kiếm và thêm: `sohamchaudhari2004/taco-trash-detection-dataset`.

### Bước 2: Tải Mã nguồn & Cài đặt Thư viện
Tạo ô code đầu tiên và chạy lệnh sau:

```bash
# Lấy mã nguồn từ kho lưu trữ
!git clone https://github.com/realer23kdl/Waste-Detection-and-Classification.git
%cd Waste-Detection-and-Classification

# Cài đặt các thư viện phụ thuộc
!pip install -r requirements.txt
```

---

## 3. Quy trình Thực thi (Pipeline 4 Giai đoạn)

### Giai đoạn 1: Tiền xử lý & Gom cụm nhãn (Label Clustering)
Thực hiện đọc file COCO JSON nguyên bản, gom nhóm 60 loại rác thành 6 siêu nhóm chuẩn và xuất ra định dạng YOLO TXT.

```bash
# Thuật toán Gom cụm nhãn và Chuyển đổi JSON sang YOLO (Tiền xử lý cực xịn)
!python src/data_loaders/coco_to_yolo.py

# Trực quan hóa phân bố 6 nhãn sau khi đã gom cụm (Phân tích Class Imbalance)
!python src/data_loaders/eda_analysis.py --data_path /kaggle/working/taco-trash-detection-dataset/labels
```

### Giai đoạn 2: Huấn luyện Khối Định vị (Object Detection)
Huấn luyện mô hình YOLOv8s để trích xuất Bounding Box. Hệ thống tự động cấu hình `data.yaml` cho 6 nhãn gốc:
```bash
!python src/trainers/train_yolo.py --data_path /kaggle/working/taco-trash-detection-dataset/ --epochs 50 --batch 16
```
*(Có thể điều chỉnh `--epochs` hoặc thuật toán Optimizer để thực hiện Ablation Study)*

### Giai đoạn 3: Tối ưu Dữ liệu & Huấn luyện Khối Phân loại (Classification)
Trích xuất phân vùng ảnh (ROI) từ Bounding Box để tối ưu hóa quá trình huấn luyện:
```bash
# Tiền xử lý trích xuất vùng ảnh rác (Chiến lược Data-Centric AI)
!python src/data_loaders/batch_crop.py --images_dir /kaggle/input/datasets/sohamchaudhari2004/taco-trash-detection-dataset/data/images --labels_dir /kaggle/working/taco-trash-detection-dataset/labels

# Huấn luyện mô hình Phân loại (Sử dụng PyTorch Custom Dataset & DataLoader)
# Tích hợp Transfer Learning, Weighted Loss và cố định Random Seed
!python src/trainers/train_classifier.py
```

### Giai đoạn 4: Đánh giá & Phân tích lỗi (Error Analysis & XAI)
Hệ thống tự động in các chỉ số định lượng (F1-score, Precision) và vẽ Ma trận nhầm lẫn (Confusion Matrix) sau Giai đoạn 3. Để triển khai AI Giải thích (Explainable AI), chạy lệnh:
```bash
# Trích xuất bản đồ nhiệt Grad-CAM và phân tích lỗi Bounding Box
!python test_demo.py
```
