import os
import sys

# Thêm đường dẫn src để import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trainers.train_yolo import train_yolo_model

if __name__ == "__main__":
    print("=== BẮT ĐẦU CHẠY THỬ NGHIỆM TRAINING (DÙNG DATASET MẪU) ===")
    print("Quá trình này sẽ tự động tải bộ Data mẫu siêu nhỏ (coco8) của YOLO để test.")
    
    try:
        # data='coco8.yaml' là bộ dữ liệu có sẵn của Ultralytics gồm 8 bức ảnh
        # epochs=3 để chạy lướt qua thật nhanh, chỉ để kiểm tra code có lỗi không
        train_yolo_model(data_yaml_path='coco8.yaml', epochs=3, batch_size=2)
        print("\n=== TEST THÀNH CÔNG! HỆ THỐNG HUẤN LUYỆN ĐÃ HOẠT ĐỘNG HOÀN HẢO ===")
        print("Trọng số (model.pth) sinh ra trong quá trình test này sẽ nằm trong thư mục 'runs/detect/'")
    except Exception as e:
        print(f"\n[LỖI] Quá trình test bị lỗi: {e}")
        print("Hãy đảm bảo bạn đã cài thư viện: pip install ultralytics wandb")
