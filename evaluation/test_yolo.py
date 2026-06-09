import argparse
import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình YOLO")
    parser.add_argument('--weights', type=str, required=True, help="Đường dẫn file weights (.pt)")
    parser.add_argument('--data_path', type=str, required=True, help="Đường dẫn file data.yaml")
    args = parser.parse_args()

    print(f"Đang đánh giá mô hình YOLO với weights: {args.weights}")
    model = YOLO(args.weights)
    
    # Chạy validation trên tập test
    metrics = model.val(data=args.data_path, split='test')
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ YOLO TRÊN TẬP TEST")
    print("="*50)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")

if __name__ == "__main__":
    main()
