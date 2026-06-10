# Đồ án Môn học: Nhận diện và Phân loại Rác Thải Tự Động
*(Môn: Học Sâu - Deep Learning)*

Đây là mã nguồn chính thức cho Đồ án cuối kỳ, được thiết kế bám sát **100% yêu cầu Rubric**: Tối ưu hóa tài nguyên (Data-centric AI, Transfer Learning), Quản lý thực nghiệm (Early Stopping, Checkpointing, LR Scheduler, Ablation Study) và Code chuẩn Lập trình Hướng đối tượng (OOP).

---

## 1. Hướng dẫn Cài đặt Môi trường (Cơ bản)

Để chạy được mã nguồn, máy tính hoặc môi trường (Google Colab/Kaggle) của bạn cần được cài đặt các thư viện lõi. 

Chạy lệnh sau tại thư mục gốc của project:
```bash
pip install -r requirements.txt
```

*Các thư viện chính bao gồm: `torch`, `torchvision`, `ultralytics` (YOLO), `tensorboard`, `grad-cam`, `scikit-learn`...*

---

## 2. Kiến trúc thư mục (OOP Pipeline)
Hệ thống được chia làm 3 Khối chuẩn mực: Khối Dữ liệu, Khối Lõi Thuật toán và Khối Đánh giá.

```text
our_pipeline/
├── data_pipeline/               <- Khối 1: Công cụ Tiền xử lý Dữ liệu (Đầu vào)
│   ├── main_prep.py             <- Script chuẩn bị dữ liệu từ bộ COCO JSON gốc.
│   ├── roboflow_coco_prep.py    <- Script chuyên biệt xử lý dữ liệu COCO tải từ nền tảng Roboflow.
│   └── crop_from_yolo.py        <- Script tự động cắt rác (crop) dựa trên nhãn YOLO có sẵn.
│
├── src/                         <- Khối 2: Lõi Thuật toán & Core (Bộ não OOP)
│   ├── config/                  <- Định nghĩa các đường dẫn file (AppConfig).
│   ├── core/                    <- Chứa các hệ thống lõi như Logger và BasePipeline.
│   ├── data_prep/               <- Lõi xử lý dữ liệu chuẩn OOP (Dataset, Transforms, Splitters, Processors).
│   ├── models/                  <- Định nghĩa các lõi mô hình Hướng đối tượng (TrashDetector, TrashClassifier).
│   ├── pipeline/                <- Chứa các luồng thực thi tổng hợp như TACOPipeline, InferencePipeline.
│   ├── trainers/                <- Lõi mã nguồn huấn luyện YOLO và Classifier.
│   └── utils/                   <- Các công cụ trực quan hóa (Visualizer, Metrics, Grad-CAM).
│
├── evaluation/                  <- Khối 3: Công cụ Đánh giá & Triển khai (Đầu ra)
│   ├── run_inference.py         <- Kịch bản chạy thực tế: Đưa ảnh vào dự đoán toàn bộ quy trình.
│   ├── ablation_comparison.py   <- Thực nghiệm So sánh hiệu quả giữa việc Có cắt rác (Crop) và Không cắt rác.
│   └── error_analysis.py        <- Chạy AI Giải thích (XAI - Grad-CAM) để phân tích lý do dự đoán sai.
│
│
├── README.md                    <- "Trang bìa" của dự án, giải thích tổng quan và hướng dẫn sử dụng.
└── requirements.txt             <- Danh sách các thư viện mã nguồn mở cần cài đặt.
```

---

## 3. Quy trình chạy Code từ đầu đến cuối (End-to-End)

Dưới đây là 3 bước thực thi tuần tự từ Data -> Train -> Đánh giá.

### BƯỚC 1: Tiền xử lý Dữ liệu (Data Preparation)

Tùy thuộc vào định dạng dữ liệu bạn tải về, chọn 1 trong 2 trường hợp:

**Trường hợp A: Dữ liệu tải về dạng Raw COCO JSON**
Nếu dùng file `annotations.json` thô, bạn cần chạy luồng tiền xử lý (Gom nhãn, binarization, cắt rác):
```bash
python data_pipeline/main_prep.py --raw_annotations /đường_dẫn/annotations.json --mapping_label /đường_dẫn/mapping_label.json
```
*Kết quả:* Hệ thống tự động sinh ra 2 tập dữ liệu riêng biệt:
1. File `.txt` chuẩn cho YOLO lưu tại `datasets/yolo_data/`.
2. Ảnh rác đã cắt nhỏ xếp theo Class lưu tại `datasets/classifier_data/train/`.

**Trường hợp B: Dữ liệu tải về dạng COCO (từ Roboflow)**
Nếu bạn tải dữ liệu COCO chia sẵn từ Roboflow, hãy chạy luồng xử lý riêng cho nó và **nhớ truyền file mapping** để gom 60 nhãn về 7 nhãn:
```bash
python data_pipeline/roboflow_coco_prep.py --dataset_dir /đường_dẫn_thư_mục_roboflow --mapping_label src/config/mapping_label.json
```

**Trường hợp C: Dữ liệu tải về dạng YOLO (từ Roboflow)**
Nếu bạn đã tải dữ liệu chuẩn YOLO qua Roboflow (gồm các file `.txt` và `data.yaml`), bạn **ĐƯỢC BỎ QUA KHÂU BỞI YOLO** nhưng vẫn cần dùng lệnh cắt rác:
```bash
python data_pipeline/crop_from_yolo.py --dataset_dir /đường_dẫn_yolo --mapping_label src/config/mapping_label.json
```

---

### BƯỚC 2: Huấn luyện Mô hình (Training)

Hệ thống bao gồm 2 mô hình huấn luyện độc lập: Định vị (YOLO) và Phân loại (ResNet/EfficientNet).

#### 2.1 Huấn luyện Khối Định vị (YOLOv8)
Sử dụng cờ `--model` để thay đổi qua lại giữa các phiên bản YOLO:
```bash
# Chạy huấn luyện (Mặc định YOLOv8 Medium, tự động vẽ biểu đồ TensorBoard)
python src/trainers/train_yolo.py --data_path /đường_dẫn/data.yaml --epochs 150 --batch 16 --patience 25
```
*(Hỗ trợ các models: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov9c.pt`, `yolov8m-rtdetr.pt`)*

#### 2.2 Huấn luyện Khối Phân loại (CNN) & Thực hiện Ablation Study
Mô hình Phân loại được tích hợp sẵn chức năng **Ablation Study (So sánh siêu tham số)** qua Terminal, đáp ứng mục 2 Rubric.

Bạn có thể thay đổi Động các tham số Learning Rate, Dropout, Model, Batch Size:
```bash
# Lệnh huấn luyện tiêu chuẩn (Đã bật đóng băng trọng số lõi và LR nhỏ để bảo vệ Model)
python src/trainers/train_classifier.py --data_path datasets/classifier_data/train --model resnet50 --epochs 50 --batch 32 --learning_rate 0.0001 --freeze_base

# Lệnh chạy Thực nghiệm Ablation Study (Đổi Model, Dropout) kèm TensorBoard
python src/trainers/train_classifier.py --model efficientnet_b0 --learning_rate 0.0001 --dropout 0.5 --freeze_base --use_tensorboard

# Xem biểu đồ ngay trên Kaggle/Colab:
# Mở một Ô Cell mới và chạy 2 lệnh sau:
# %load_ext tensorboard
# %tensorboard --logdir runs

```

---

### BƯỚC 3: Đánh giá Độc lập (Chạy Test tách biệt)

Thay vì Đánh giá tự động ngay sau khi Train, bạn có thể chạy riêng các tập lệnh Test sau khi đã có file trọng số tốt nhất (`best.pt` hoặc `best_resnet50.pth`):

#### 3.1 Chấm điểm Mô hình Định vị (YOLO)
```bash
python src/trainers/test_yolo.py --data_path /đường_dẫn/data.yaml --weights runs/detect/yolov8m_trashnet/weights/best.pt
```
*Kết quả:* Tính toán ra các chỉ số mAP50, mAP50-95 và lưu biểu đồ vào thư mục `runs/detect/val/`.

#### 3.2 Chấm điểm Mô hình Phân loại (CNN)
Hệ thống sẽ lấy file trọng số tải lên mạng phân loại và chấm điểm trên tập Test:
```bash
python src/trainers/test_classifier.py --data_path datasets/classifier_data/test --model resnet50 --weights weights/best_resnet50.pth
```
*Kết quả:* Xuất ra màn hình chỉ số Accuracy, Precision, Recall, F1 và tạo ra 2 file ảnh cực kỳ quan trọng cho Báo cáo (Rubric Phần 3):
1. `test_confusion_matrix_resnet50.png`: Ma trận nhầm lẫn.
2. `test_error_analysis_resnet50.png`: Lưới trực quan hóa các mẫu dự đoán sai.

---

## 4. Hướng dẫn Dự đoán (Inference / Testing)

Sau khi huấn luyện xong cả 2 mô hình, bạn có thể chạy luồng Inference tổng hợp 2 giai đoạn (Detect -> Crop -> Classify) cho một bức ảnh bất kỳ:

```bash
python evaluation/run_inference.py \
    --pipeline detect_and_classify \
    --image test_image.jpg \
    --detector runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50 \
    --classifier_weights weights/best_resnet50.pth \
    --conf 0.25
```
*Kết quả:* Hệ thống sẽ gọi YOLO để vẽ Bounding Box, dùng ResNet50 để gắn nhãn phân loại kèm Độ tin cậy (Confidence), và cuối cùng **tự động lưu ảnh trực quan ra file `inference_result.jpg`**.

---

## 5. Ablation Study: Đánh giá Tầm quan trọng của Cắt rác (Crop)

Để so sánh trực quan hiệu quả của việc **Có cắt rác (Detect -> Crop -> Classify)** so với việc **Truyền thẳng ảnh gốc (Classify Only)**, bạn hãy chạy lệnh sau:

```bash
python evaluation/ablation_comparison.py \
    --image test_image.jpg \
    --detector runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50 \
    --classifier_weights weights/best_resnet50.pth
```
*Kết quả:* Trả về file ảnh `ablation_comparison_result.png` chia làm 2 nửa màn hình. Nửa trái là kết quả sai lệch khi Classifier bị nhiễu bởi hậu cảnh. Nửa phải là kết quả chính xác khi YOLO đã loại bỏ hậu cảnh và chỉ cắt đúng cục rác. Bức ảnh này **cực kỳ đắt giá** để đưa vào báo cáo!

---

## 6. Explainable AI (XAI) bằng Grad-CAM

Để giải thích được lý do vì sao AI lại phân loại sai, hãy dùng lệnh sau để soi "Bản đồ nhiệt":

```bash
python evaluation/error_analysis.py \
    --image test_image.jpg \
    --detector runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50 \
    --classifier_weights weights/best_resnet50.pth
```
*Kết quả:* Trả về file `test_heatmap_gradcam.png` soi chiếu vào điểm đặc trưng của rác (Ví dụ: AI nhìn vào nếp gấp của túi nilon để đưa ra quyết định).
