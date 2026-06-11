# Đồ án Môn học: Nhận diện và Phân loại Rác Thải Tự Động
*(Môn: Học Sâu - Deep Learning)*

Hệ thống được thiết kế theo quy trình hai giai đoạn (Two-Stage Pipeline), kết hợp thuật toán Phát hiện đối tượng (Object Detection) để tìm vùng chứa rác và thuật toán Phân loại hình ảnh (Image Classification) để gán nhãn loại rác. Mã nguồn được tổ chức theo phương pháp Lập trình Hướng đối tượng (OOP).

---

## 1. Hướng dẫn Cài đặt Môi trường (Cơ bản)

Để chạy mã nguồn, môi trường cần cài đặt các thư viện cần thiết. Khởi chạy lệnh sau tại thư mục gốc:
```bash
pip install -r requirements.txt
```
*(Hệ thống sử dụng các thư viện chính như: PyTorch, Ultralytics, Grad-CAM và Scikit-Learn).*

---

## 2. Kiến trúc thư mục (OOP Pipeline)

Hệ thống được chia thành 3 khối chức năng chính, với luồng dữ liệu (Data Flow) được truyền lần lượt qua các bước xử lý:

**Khối 1: `data_pipeline/` (Tiền xử lý Dữ liệu)**
Khối này đảm nhiệm việc xử lý dữ liệu đầu vào. Tệp `main_prep.py` và `roboflow_coco_prep.py` chịu trách nhiệm đồng nhất 60 phân lớp ban đầu về 7 phân lớp chung. Tiếp theo, tệp `crop_from_yolo.py` sử dụng kích thước Bounding Box để cắt (crop) các phần hình ảnh chứa rác, tạo dữ liệu đầu vào cho mô hình phân loại.

**Khối 2: `src/` (Mã nguồn Cốt lõi & Thuật toán)**
Khối này chứa các lớp (Classes) quản lý dữ liệu và mô hình:
- Dữ liệu được nạp vào mô hình thông qua tệp `data_prep/dataset.py`. Lớp này kế thừa từ `Dataset` của PyTorch, thực hiện tiền xử lý ảnh (Augmentation) và tính toán trọng số (Class Weights) để xử lý dữ liệu mất cân bằng.
- Các mô hình được khởi tạo tại `models/detector.py` (YOLO) và `models/classifier.py` (ResNet50). Cấu trúc này giúp dễ dàng tùy chỉnh số lượng nhãn hoặc cấu hình mô hình.
- Lớp `ClassifierTrainer` trong `trainers/train_classifier.py` quản lý quá trình huấn luyện, gọi dữ liệu từ bộ nạp và cập nhật trọng số cho mô hình. Hàm huấn luyện cũng tích hợp cơ chế Dừng sớm (Early Stopping) và Điều chỉnh tốc độ học (ReduceLROnPlateau).

**Khối 3: `evaluation/` (Đánh giá & Triển khai)**
Tiếp nhận các tệp trọng số (Weights) từ bước huấn luyện, Khối 3 thực hiện tính toán các chỉ số (F1-Score, mAP, Accuracy) thông qua các tệp `test_yolo.py` và `test_classifier.py`, đồng thời hỗ trợ phân tích kết quả mô hình.

---

## 3. Quy trình chạy Code từ đầu đến cuối (End-to-End)

Quy trình huấn luyện có thể được thực thi thông qua dòng lệnh với thư viện `argparse`.

### BƯỚC 1: Tiền xử lý Dữ liệu (Data Preparation)
Đồng bộ nhãn và chia cấu trúc thư mục:
```bash
python data_pipeline/roboflow_coco_prep.py \
    --dataset_dir "/đường_dẫn_tới_thư_mục_chứa_ảnh" \
    --mapping_label src/config/mapping_label.json
```

### BƯỚC 2: Huấn luyện Mô hình (Training)

**2.1 Huấn luyện Khối Định vị (YOLOv8)**
Khởi chạy huấn luyện mạng YOLO:
```bash
python src/trainers/train_yolo.py \
    --data_path datasets/yolo_data/data.yaml \
    --model yolov8m.pt \
    --epochs 50 \
    --batch 16 \
    --patience 25 \
    --lr 0.01
```

**2.2 Huấn luyện Khối Phân loại (CNN)**
Khởi chạy huấn luyện mạng ResNet50, có thể theo dõi qua TensorBoard:
```bash
python src/trainers/train_classifier.py \
    --data_path datasets/classifier_data/train \
    --model resnet50 \
    --epochs 30 \
    --batch 32 \
    --learning_rate 0.001 \
    --dropout 0.3 \
    --patience 5 \
    --use_tensorboard
```

### BƯỚC 3: Đánh giá Độc lập (Chạy Test tách biệt)

**3.1 Chấm điểm Mô hình Định vị (YOLO)**
```bash
python evaluation/test_yolo.py \
    --weights runs/detect/runs/detect/yolov8m_trashnet/weights/best.pt \
    --data_path datasets/yolo_data/data.yaml
```

**3.2 Chấm điểm Mô hình Phân loại (CNN)**
Kiểm thử mô hình phân loại và in ra Ma trận nhầm lẫn (Confusion Matrix):
```bash
python evaluation/test_classifier.py \
    --weights weights/best_resnet50.pth \
    --data_path datasets/classifier_data/test \
    --model resnet50 \
    --dropout 0.3
```

---

## 4. Hướng dẫn Dự đoán (Inference / Testing)

Tệp `evaluation/run_inference.py` kết nối trực tiếp hai Mô hình thành một luồng (Pipeline). Bức ảnh đầu vào sẽ đi qua mạng YOLO để khoanh vùng đối tượng, phần khoanh vùng đó được đưa tiếp vào mạng CNN để phân loại, cuối cùng trả ra bức ảnh tổng hợp kết quả.
```bash
python evaluation/run_inference.py \
    --pipeline detect_and_classify \
    --detector runs/detect/runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50 \
    --image test_image.jpg
```

---

## 5. Ablation Study: Đánh giá Hiệu quả của bước Cắt rác (Crop)

Việc chạy Ablation Study nhằm mục đích so sánh hai trường hợp: phân loại trực tiếp trên ảnh gốc (bị lẫn phông nền) so với phân loại trên ảnh đã được cắt bỏ phông nền bởi YOLO. Kết quả thu được giúp làm rõ tác dụng của việc chia hệ thống thành hai giai đoạn.
```bash
python evaluation/ablation_comparison.py \
    --image test_image.jpg \
    --detector runs/detect/runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50
```

---

## 6. Giải thích Mô hình bằng Grad-CAM (XAI)

Hệ thống sử dụng kỹ thuật Grad-CAM để trực quan hóa quá trình dự đoán. Khi chạy tệp `error_analysis.py`, thuật toán sẽ tạo ra một bản đồ nhiệt (Heatmap) chỉ ra các vùng pixel trên đồ vật mà mạng CNN sử dụng làm cơ sở để phân loại. Điều này giúp hiểu rõ hơn cách mô hình hoạt động thay vì chỉ nhận kết quả đầu ra.
```bash
python evaluation/error_analysis.py \
    --image test_image.jpg \
    --detector runs/detect/runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50
```
