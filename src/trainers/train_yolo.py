import argparse
import yaml
import os
from ultralytics import YOLO
import wandb

def create_yaml_if_needed(data_path):
    """Sử dụng data.yaml có sẵn hoặc tự tạo mới nếu cần"""
    if data_path.endswith('.yaml'):
        return data_path
        
    # Kiểm tra xem trong thư mục có sẵn data.yaml không (Thường Roboflow sẽ cung cấp sẵn)
    existing_yaml = os.path.join(data_path, 'data.yaml')
    if os.path.exists(existing_yaml):
        print(f"Đã tìm thấy và sử dụng file cấu hình có sẵn tại: {existing_yaml}")
        return existing_yaml
        
    # Nếu không có, tự động tạo mới (Dành cho bộ data cũ)
    yaml_path = 'data.yaml'
    if os.path.isdir(data_path):
        print(f"Đang tự động tạo tệp {yaml_path} trỏ tới bộ dữ liệu: {data_path}")
        data_yaml = {
            'train': os.path.join(data_path, 'images'),
            'val': os.path.join(data_path, 'images'),
            'test': os.path.join(data_path, 'images'),
            'nc': 6,
            'names': ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        return yaml_path
    else:
        raise ValueError("Đường dẫn data không hợp lệ! Phải là thư mục hoặc file .yaml")

def train_yolo_model(data_path: str, epochs: int = 50, batch_size: int = 16, optimizer='auto', lr=0.01):
    """
    Code Python huấn luyện YOLOv8.
    """
    yaml_path = create_yaml_if_needed(data_path)
    
    print("Khởi tạo Weights & Biases để ghi log biểu đồ...")
    wandb.init(project="trashnet-yolo-detection", job_type="training")
    
    print("Nạp mô hình YOLOv8 Medium (Não to hơn, chuyên trị rác nhỏ)...")
    model = YOLO('yolov8m.pt')
    
    print("Bắt đầu huấn luyện...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        project='runs/detect',
        name='yolov8m_trashnet',
        exist_ok=True,
        plots=True,
        
        # --- BỔ SUNG CÁC TIÊU CHÍ RUBRIC (CHECKPOINT, EARLY STOPPING, LR SCHEDULER) ---
        
        # 1. Early Stopping (Dừng sớm tránh Overfitting)
        patience=25,       # Dừng huấn luyện nếu mAP không tăng sau 25 epochs
        
        # 2. Checkpointing (Lưu trọng số tự động)
        save=True,         # Tự động lưu best.pt và last.pt
        save_period=10,    # Lưu thêm 1 file checkpoint dự phòng mỗi 10 epochs
        
        # 3. Learning Rate Scheduler (Điều chỉnh tốc độ học)
        cos_lr=True,       # Kích hoạt Cosine Annealing Scheduler (Giảm LR theo hình sin)
        lr0=0.01,          # Tốc độ học khởi tạo ban đầu
        lrf=0.01,          # Tốc độ học cuối cùng (lr0 * lrf = 0.0001)
        
        # --- BỘ KỸ THUẬT TĂNG CƯỜNG DỮ LIỆU (ĐÃ TỐI ƯU CHO RÁC NHỎ) ---
        mosaic=1.0,  # Vẫn giữ nguyên ghép 4 ảnh để học bối cảnh
        degrees=10.0, # Xoay nhẹ 10 độ
        
        # Tắt các hiệu ứng phá hủy rác nhỏ:
        scale=0.0,    # Không zoom nhỏ rác lại
        perspective=0.0, # Không bóp méo 3D
        mixup=0.0,    # Không làm mờ rác
        flipud=0.0,   # Không lật lộn ngược rác (rác rơi trên đất hiếm khi lộn ngược)
        
        # Thay đổi màu sắc và lật ngang vẫn an toàn
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )
    
    print(f"Huấn luyện hoàn tất! Trọng số tốt nhất được lưu tại: runs/detect/yolov8s_trashnet/weights/best.pt")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Đường dẫn tới thư mục data trên Kaggle (VD: /kaggle/input/...) hoặc file data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='auto', help='Tối ưu Optimizer (VD: SGD, AdamW)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    train_yolo_model(args.data_path, args.epochs, args.batch, args.optimizer, args.lr)
