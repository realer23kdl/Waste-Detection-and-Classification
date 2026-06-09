import argparse
import yaml
import os
import sys
from ultralytics import YOLO
# Thêm đường dẫn gốc để import file detector.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.detector import TrashDetector

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
            'train': os.path.join(data_path, 'train/images'),
            'val': os.path.join(data_path, 'val/images'),
            'test': os.path.join(data_path, 'val/images'),
            'nc': 6,
            'names': ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        return yaml_path
    else:
        raise ValueError("Đường dẫn data không hợp lệ! Phải là thư mục hoặc file .yaml")

def train_yolo_model(data_path: str, model_name: str = 'yolov8m.pt', epochs: int = 50, batch_size: int = 16, patience: int = 25, optimizer='auto', lr=0.01):
    """
    Code Python huấn luyện YOLOv8.
    """
    yaml_path = create_yaml_if_needed(data_path)
    
    print("Sử dụng TensorBoard mặc định của YOLO để ghi log biểu đồ...")    
    print(f"Nạp mô hình {model_name} thông qua OOP TrashDetector...")
    detector = TrashDetector(model_path=model_name, model_type='yolo' if 'yolo' in model_name else 'rtdetr')
    
    print("Bắt đầu huấn luyện...")
    results = detector.train(
        data_yaml_path=yaml_path,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        name='yolov8m_trashnet'
    )
    
    print(f"Huấn luyện hoàn tất! Trọng số tốt nhất được lưu tại: runs/detect/yolov8s_trashnet/weights/best.pt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Đường dẫn tới thư mục data trên Kaggle (VD: /kaggle/input/...) hoặc file data.yaml')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='Tên mô hình Ultralytics (VD: yolov8m.pt, yolov9c.pt, yolov8m-rtdetr.pt)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=25, help='Early Stopping patience')
    parser.add_argument('--optimizer', type=str, default='auto', help='Tối ưu Optimizer (VD: SGD, AdamW)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    train_yolo_model(args.data_path, args.model, args.epochs, args.batch, args.patience, args.optimizer, args.lr)
