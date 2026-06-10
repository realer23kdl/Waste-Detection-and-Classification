import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep.dataset import TrashDataset
from src.models.classifier import TrashClassifier

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Classifier")
    parser.add_argument('--weights', type=str, required=True, help="Đường dẫn file weights (.pth)")
    parser.add_argument('--data_path', type=str, required=True, help="Thư mục chứa ảnh test")
    parser.add_argument('--model', type=str, default='resnet50', help="Tên model")
    parser.add_argument('--dropout', type=float, default=0.3, help="Tỷ lệ Dropout (phải khớp với lúc train)")
    args = parser.parse_args()

    print(f"Đang nạp tập dữ liệu từ {args.data_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = TrashDataset(root_dir=args.data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Khởi tạo mô hình {args.model}...")
    num_classes = len(test_dataset.classes)
    
    model_wrapper = TrashClassifier(model_name=args.model, num_classes=num_classes, dropout_rate=args.dropout)
    model = model_wrapper.model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Đang tiến hành đánh giá trên tập test...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*50)
    print("BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT)")
    print("="*50)
    target_names = [k for k, v in sorted(test_dataset.class_to_idx.items(), key=lambda item: item[1])]
    print(classification_report(all_labels, all_preds, labels=range(len(target_names)), target_names=target_names, zero_division=0))

    # Vẽ Confusion Matrix
    print("\nĐang vẽ Ma trận nhầm lẫn (Confusion Matrix)...")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(target_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix_resnet50.png', dpi=150)
    print("Đã lưu ma trận nhầm lẫn tại: confusion_matrix_resnet50.png")

if __name__ == "__main__":
    main()
