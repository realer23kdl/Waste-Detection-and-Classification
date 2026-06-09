import argparse
import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.detector import TrashDetector

def test_yolo_model(data_yaml_path: str, weights_path: str):
    print(f"=== BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH YOLO ===")
    print(f"Trọng số: {weights_path}")
    print(f"Data: {data_yaml_path}")
    
    # Nạp mô hình
    model_type = 'rtdetr' if 'rtdetr' in weights_path.lower() else 'yolo'
    detector = TrashDetector(model_path=weights_path, model_type=model_type)
    
    print("\nĐang chấm điểm trên tập Test (hoặc Validation)...")
    # Gọi hàm val() của Ultralytics
    metrics = detector.model.val(data=data_yaml_path)
    
    print("\n[KẾT QUẢ ĐÁNH GIÁ YOLO]")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    
    print("\nCảnh báo: YOLO mặc định tự động lưu các file biểu đồ (Confusion Matrix, F1_Curve) vào thư mục runs/detect/val/")
    print("=== HOÀN TẤT ĐÁNH GIÁ ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Đường dẫn tới file data.yaml")
    parser.add_argument('--weights', type=str, required=True, help="Đường dẫn tới file trọng số best.pt")
    args = parser.parse_args()
    
    test_yolo_model(args.data_path, args.weights)
