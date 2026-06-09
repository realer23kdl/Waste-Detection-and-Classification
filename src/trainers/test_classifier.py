import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_prep.dataset import TrashDataset
from src.models.classifier import TrashClassifier
from src.utils.metrics import Evaluator

def test_model(data_path: str, model_name: str, weights_path: str, batch_size: int = 32):
    print(f"=== BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH {model_name.upper()} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Nạp tập dữ liệu Test từ: {data_path}")
    test_dataset = TrashDataset(data_path, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Khởi tạo mạng phân loại và nạp trọng số...")
    classifier = TrashClassifier(model_name=model_name, num_classes=6)
    classifier.model.load_state_dict(torch.load(weights_path, map_location=device))
    classifier.model = classifier.model.to(device)
    classifier.model.eval()
    
    print("Đang chạy dự đoán trên tập Test để lấy chỉ số F1, Precision, Recall...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = classifier.model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # ĐÁNH GIÁ (RUBRIC PHẦN 3)
    evaluator = Evaluator(class_names=test_dataset.classes)
    
    print("\n[KẾT QUẢ ĐÁNH GIÁ]")
    evaluator.calculate_metrics(y_true, y_pred)
    
    # Vẽ Ma trận nhầm lẫn
    cm_path = f"test_confusion_matrix_{model_name}.png"
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    print(f"Đã lưu Ma trận nhầm lẫn tại: {cm_path}")
    
    # Trực quan hóa ảnh phân loại sai (Rubric: Error Analysis)
    print("\nĐang trích xuất và trực quan hóa các mẫu dự đoán sai...")
    error_path = f"test_error_analysis_{model_name}.png"
    evaluator.plot_wrong_predictions(test_dataset, y_true, y_pred, num_samples=9, save_path=error_path)
    print(f"Đã lưu Phân tích lỗi tại: {error_path}")
    
    print("\n=== HOÀN TẤT ĐÁNH GIÁ ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Đường dẫn tới thư mục data test")
    parser.add_argument('--model', type=str, default='resnet50', help="Kiến trúc mạng (resnet50, efficientnet_b0...)")
    parser.add_argument('--weights', type=str, required=True, help="Đường dẫn tới file .pth tốt nhất")
    parser.add_argument('--batch', type=int, default=32, help="Kích thước batch size")
    args = parser.parse_args()

    test_model(args.data_path, args.model, args.weights, args.batch)
