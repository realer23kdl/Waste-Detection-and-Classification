# Đồ án Môn học: Nhận diện và Phân loại Rác Thải Tự Động
*(Môn: Học Sâu - Deep Learning)*

Đây là mã nguồn chính thức cho Đồ án cuối kỳ, được thiết kế bám sát 100% yêu cầu Rubric: Tối ưu hóa tài nguyên (Data-centric AI, Transfer Learning) và Quản lý thực nghiệm (Early Stopping, Checkpointing, LR Scheduler).

Đặc biệt, toàn bộ quy trình tiền xử lý dữ liệu đã được **Tái cấu trúc theo chuẩn Lập trình Hướng đối tượng (OOP)** để tách biệt rõ ràng các khâu cấu hình, xử lý và huấn luyện.

---

## 1. Kiến trúc thư mục (OOP Pipeline)
Hệ thống được module hóa cao độ để dễ dàng bảo trì và mở rộng:
- `src/config/`: Nơi định nghĩa các đường dẫn file (AppConfig, SystemConfig).
- `src/core/`: Chứa các hệ thống lõi như `LoggerSetup` và `BasePipeline`.
- `src/data_prep/`: Trái tim xử lý dữ liệu:
  - `processors/`: Chứa các thuật toán `CategoryMappingProcessor`, `BinarizationProcessor`.
  - `splitters/`: Chứa thuật toán cắt tập dữ liệu `MultiLabelSplitter`, `SingleLabelSplitter`.
  - `transforms.py`: Chứa các bộ Augmentations và Resize.
- `src/pipeline/`: Chứa `WastePreprocessingPipeline` dùng để lắp ghép các processors bên trên.
- `src/trainers/`: Mã nguồn huấn luyện YOLO và Classifier.

---

## 2. Hướng dẫn Tiền xử lý Dữ liệu (Data Preparation)

Tùy thuộc vào định dạng dữ liệu bạn tải về, hãy chọn 1 trong 2 trường hợp sau:

### Trường hợp A: Dữ liệu tải về dạng Raw COCO JSON
Nếu bạn sử dụng file `annotations.json` thô, bạn CẦN chạy Pipeline tiền xử lý để làm sạch, Gom nhãn, và San phẳng nhãn (Binarization).

**Cách chạy trên Kaggle:**
Dùng câu lệnh sau, truyền trực tiếp đường dẫn file gốc vào:
```bash
!python main_prep.py --raw_annotations /kaggle/input/.../annotations.json --mapping_label /kaggle/input/.../mapping_label.json
```
*(Bổ sung tùy chỉnh chia Train/Test)*: Bạn có thể thêm `--test_size 0.3` (chia 30% test) hoặc `--random_state 42` để cố định tập chia.
*Kết quả:* Hệ thống sẽ tự động sinh ra các file `.json` đã được xử lý chuẩn mực.

### Trường hợp B: Dữ liệu tải về dạng YOLO (từ Roboflow)
Nếu bạn đã tải dữ liệu qua Roboflow ở dạng YOLO (gồm các file `.txt` và `data.yaml`), Roboflow đã làm hộ phần chia Train/Test.
👉 **BẠN ĐƯỢC BỎ QUA BƯỚC NÀY!** Không cần chạy `main_prep.py`, hãy đi thẳng xuống phần Huấn luyện.

---

## 3. Hướng dẫn Huấn luyện (Training)

Mô hình YOLOv8 Medium đã được tùy chỉnh cấu hình để đáp ứng tối đa tiêu chí Rubric của đồ án, bao gồm:
- **Early Stopping (patience=25):** Tự động dừng nếu mAP không tăng.
- **Checkpointing (save_period=10):** Lưu backup mỗi 10 epochs để chống mất mát.
- **Cosine LR Scheduler (cos_lr=True):** Giúp đồ thị hàm Loss hội tụ đẹp.

**Lệnh huấn luyện YOLO/RT-DETR:**
Sử dụng cờ `--model` để thay đổi qua lại giữa các phiên bản YOLO cực kỳ tiện lợi:
```bash
# Tắt wandb để tránh lỗi hỏi API Key trên Kaggle
%env WANDB_MODE=disabled

# Khởi động quy trình Train YOLOv8 Medium (Mặc định)
!python src/trainers/train_yolo.py --data_path /đường/dẫn/đến/file/data.yaml --epochs 150 --batch 16 --patience 25
```

**Các tham số có thể tùy chỉnh qua dòng lệnh (Argparse):**
- `--data_path`: (Bắt buộc) Đường dẫn tới file `data.yaml` hoặc thư mục chứa data.
- `--model`: Tên mô hình Ultralytics. **Các model được hỗ trợ:** `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` (mặc định), `yolov9c.pt`, `yolov8m-rtdetr.pt` (mô hình Transformer).
- `--epochs`: Số epoch huấn luyện (mặc định: `50`).
- `--batch`: Kích thước batch size (mặc định: `16`).
- `--patience`: Số lượng epoch tối đa chờ mAP không tăng trước khi dừng (mặc định: `25`).
- `--lr`: Tốc độ học (Learning Rate, mặc định: `0.01`).
- `--optimizer`: Tối ưu hóa (VD: `SGD`, `AdamW`, mặc định: `auto`).

**Lệnh huấn luyện Classifier:**
Hệ thống cho phép bạn chuyển đổi kiến trúc mạng dễ dàng chỉ bằng cờ `--model` (Hỗ trợ: `efficientnet_b0`, `resnet50`, `mobilenet_v3`):
```bash
!python src/trainers/train_classifier.py --data_path datasets/classifier_data/train --model resnet50 --epochs 50 --patience 5
```

**Các tham số có thể tùy chỉnh qua dòng lệnh (Argparse):**
- `--data_path`: Đường dẫn tới thư mục ảnh đã cắt rác (Mặc định: `datasets/classifier_data/train`).
- `--model`: Kiến trúc mạng CNN. **Các model được hỗ trợ:** `efficientnet_b0` (mặc định), `resnet50`, `mobilenet_v3`.
- `--epochs`: Số epoch huấn luyện (mặc định: `50`).
- `--batch`: Kích thước batch size (mặc định: `32`).
- `--patience`: Số lượng epoch tối đa chờ Validation Loss không giảm trước khi dừng sớm (mặc định: `5`).

---

## 4. Hướng dẫn Nhận diện (Inference)

Sau khi huấn luyện xong cả Khối Định vị (YOLO) và Khối Phân loại (Classifier), bạn có thể chạy luồng Inference tổng hợp 2 giai đoạn (Detect -> Crop -> Classify) bằng lệnh sau:
```bash
!python main_inference.py \
    --pipeline detect_and_classify \
    --detector runs/detect/yolov8m_trashnet/weights/best.pt \
    --classifier resnet50 \
    --image test_image.jpg
```
**Ý nghĩa:** File này sẽ gọi YOLO ra khoanh vùng rác trong bức ảnh `test_image.jpg`, sau đó cắt (crop) từng cục rác ra và ném cho mạng ResNet50 để xác định chính xác nó là rác gì. Kết quả cuối cùng sẽ được in ra màn hình hoặc vẽ trực tiếp lên ảnh.
