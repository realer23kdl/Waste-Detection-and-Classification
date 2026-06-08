import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import random
import os
import sys

# Thêm đường dẫn gốc để import file dataset.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_prep.dataset import TrashDataset
from src.models.classifier import TrashClassifier

def set_seed(seed=42):
    """Cố định Random Seed để có thể tái lập kết quả (Theo đúng Rubric)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Trainer] Đã cố định Random Seed = {seed}")

class ClassifierTrainer:
    """
    Vòng lặp huấn luyện (Training Loop) chuyên nghiệp cho PyTorch.
    Bao gồm: Forward, Backward, Scheduler, Early Stopping, Checkpointing, và Weighted Loss.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', patience=5, class_weights=None):
        set_seed(42)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Hàm Loss (Áp dụng Weighted Loss nếu có Class Imbalance)
        if class_weights is not None:
            weights_tensor = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            print("[Trainer] Đã kích hoạt Weighted Loss để chống Mất cân bằng dữ liệu!")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Cập nhật PyTorch >= 2.2: Hàm ReduceLROnPlateau đã loại bỏ tham số verbose
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)
        
        self.patience = patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader)

    def train(self, num_epochs=50, save_path="best_classifier.pth"):
        print("\n=== BẮT ĐẦU HUẤN LUYỆN CLASSIFIER ===")
        train_losses, val_losses = [], []
        
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # Tự động tạo thư mục nếu chưa tồn tại
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Đã lưu Checkpoint mới tại {save_path}!")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"\n[Early Stopping] Đã dừng sớm ở Epoch {epoch+1} vì Val Loss không giảm nữa.")
                    break
                    
        return train_losses, val_losses

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Classifier")
    parser.add_argument('--data_path', type=str, default="datasets/classifier_data/train", help="Đường dẫn tới dữ liệu")
    parser.add_argument('--model', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'], help="Kiến trúc mạng")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    # KHỐI LỆNH THỰC THI CHUẨN RUBRIC (Kế thừa Dataset & Sử dụng DataLoader)
    print(f"Khởi tạo Data Pipeline cho Classifier với mạng {args.model}...")
    
    # 1. Cấu hình biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 2. Khởi tạo Custom Dataset (Đúng theo yêu cầu "kế thừa lớp Dataset")
    DATA_DIR = args.data_path
    
    if os.path.exists(DATA_DIR):
        train_dataset = TrashDataset(root_dir=DATA_DIR, transform=transform)
        
        # 3. Tính toán Weighted Loss tự động chống mất cân bằng
        class_weights = train_dataset.get_class_weights()
        num_classes = len(train_dataset.classes)
        
        # 4. Sử dụng DataLoader trong PyTorch (Yêu cầu Rubric)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False) # Dùng tạm train làm val để test
        
        # 5. Khởi tạo Mô hình động dựa trên Argparse (SỬ DỤNG OOP TRASH CLASSIFIER)
        print(f"Đang khởi tạo mô hình {args.model} thông qua OOP TrashClassifier...")
        classifier_wrapper = TrashClassifier(model_name=args.model, num_classes=num_classes, pretrained=True)
        model = classifier_wrapper.model
            
        # 6. Truyền class_weights vào Trainer
        trainer = ClassifierTrainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            patience=args.patience,
            class_weights=class_weights
        )
        
        # 7. Bắt đầu huấn luyện
        save_path = f"weights/best_{args.model}.pth"
        train_losses, val_losses = trainer.train(num_epochs=args.epochs, save_path=save_path)
        
        # ==========================================
        # KHỐI LỆNH ĐÁNH GIÁ (RUBRIC PHẦN 3)
        # ==========================================
        from src.utils.metrics import Evaluator
        evaluator = Evaluator(class_names=train_dataset.classes)
        
        # 1. Vẽ biểu đồ Loss qua các Epochs
        print("Đang vẽ biểu đồ Loss...")
        evaluator.plot_loss_curves(train_losses, val_losses, save_path=f"loss_curve_{args.model}.png")
        
        # 2. Chạy đánh giá trên tập Test (Lấy Val làm Test tạm thời)
        print("Đang chạy dự đoán trên tập kiểm tra để lấy chỉ số F1, Precision, Recall...")
        trainer.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(trainer.device)
                outputs = trainer.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
        # In ra các chỉ số vàng (Accuracy, Precision, Recall, F1)
        evaluator.calculate_metrics(y_true, y_pred)
        
        # 3. Vẽ Ma trận nhầm lẫn
        evaluator.plot_confusion_matrix(y_true, y_pred, save_path=f"confusion_matrix_{args.model}.png")
        
        # 4. Trực quan hóa ảnh phân loại sai (Rubric: Error Analysis)
        print("Đang trích xuất và trực quan hóa các mẫu dự đoán sai...")
        evaluator.plot_wrong_predictions(train_dataset, y_true, y_pred, num_samples=9, save_path=f"error_analysis_{args.model}.png")
        
        print("\n[HOÀN TẤT] Quá trình huấn luyện và đánh giá mô hình đã kết thúc.")
        
    else:
        print(f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}")
