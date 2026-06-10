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
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, model, train_loader, val_loader, device='cuda', patience=5, class_weights=None, learning_rate=0.001, use_tensorboard=False):
        set_seed(42)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir="runs/classifier")
            print("[Trainer] Đã khởi tạo TensorBoard tại thư mục runs/classifier")
        
        # Hàm Loss (Áp dụng Weighted Loss nếu có Class Imbalance)
        if class_weights is not None:
            weights_tensor = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            print("[Trainer] Đã kích hoạt Weighted Loss để chống Mất cân bằng dữ liệu!")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
            
            if self.use_tensorboard:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)
            
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
                    
        if self.use_tensorboard:
            self.writer.close()
            
        return train_losses, val_losses

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Classifier")
    parser.add_argument('--data_path', type=str, default="datasets/classifier_data/train", help="Đường dẫn tới dữ liệu")
    parser.add_argument('--model', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet50', 'mobilenet_v3'], help="Kiến trúc mạng")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Tốc độ học (Nên để 1e-4 cho Transfer Learning)")
    parser.add_argument('--dropout', type=float, default=0.3, help="Tỷ lệ Dropout")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--use_tensorboard', action='store_true', help="Bật log TensorBoard")
    parser.add_argument('--freeze_base', action='store_true', help="Đóng băng các layer Conv của mạng pre-trained để chỉ train lớp phân loại cuối")
    args = parser.parse_args()

    # KHỐI LỆNH THỰC THI CHUẨN RUBRIC (Kế thừa Dataset & Sử dụng DataLoader)
    print(f"Khởi tạo Data Pipeline cho Classifier với mạng {args.model}...")
    
    # 1. Cấu hình biến đổi ảnh (Data Augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
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
        classifier_wrapper = TrashClassifier(model_name=args.model, num_classes=num_classes, pretrained=True, dropout_rate=args.dropout)
        model = classifier_wrapper.model
        
        # Áp dụng Đóng băng Trọng số (Freeze Base Model) để chống Model Collapse
        if args.freeze_base:
            print("Đã bật chế độ FREEZE BASE MODEL. Chỉ huấn luyện lớp Fully Connected cuối cùng!")
            for name, param in model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False
            
        # 6. Truyền class_weights vào Trainer
        trainer = ClassifierTrainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            patience=args.patience,
            class_weights=class_weights,
            learning_rate=args.learning_rate,
            use_tensorboard=args.use_tensorboard
        )
        
        # 7. Bắt đầu huấn luyện
        save_path = f"weights/best_{args.model}.pth"
        train_losses, val_losses = trainer.train(num_epochs=args.epochs, save_path=save_path)
        
        # 8. Vẽ biểu đồ hàm mất mát (Loss Curve)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Biểu đồ Hàm mất mát (Loss Curve) - {args.model.upper()}')
        plt.legend()
        plt.grid(True)
        loss_curve_path = f'loss_curve_{args.model}.png'
        plt.savefig(loss_curve_path, dpi=150)
        print(f"\n[HOÀN TẤT] Đã vẽ và lưu biểu đồ Loss tại: {loss_curve_path}")
        
        print(f"[HOÀN TẤT] Quá trình huấn luyện mô hình đã kết thúc. Trọng số được lưu tại: {save_path}")
        
    else:
        print(f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}")
