import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.data_prep.dataset import TrashDataset
from src.models.classifier import TrashClassifier
from src.trainers.train_classifier import ModelTrainer

def run_demo():
    print("======================================================")
    print("DEMO: DATALOADER, EARLY STOPPING & LR SCHEDULER (OOP)")
    print("======================================================\n")

    # 1. DATACLASS & DATALOADER
    print("[1/3] Khởi tạo Dataset và DataLoader...")
    try:
        train_dataset = TrashDataset("datasets/classifier_data/train")
        val_dataset = TrashDataset("datasets/classifier_data/test")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        print(f" -> Load thành công {len(train_dataset)} ảnh train và {len(val_dataset)} ảnh test!")
    except Exception as e:
        print(f"Lỗi load data (Hãy chắc chắn thư mục datasets/classifier_data đã có ảnh): {e}")
        return

    # 2. KHỞI TẠO MODEL
    print("\n[2/3] Khởi tạo mạng ResNet50...")
    classifier = TrashClassifier(model_name='resnet50', num_classes=6)

    # 3. GỌI CLASS TRAINER (CHỨA EARLY STOPPING & LR SCHEDULER)
    print("\n[3/3] Bắt đầu quá trình Train Demo với các kỹ thuật tối ưu...")
    print(" - Early Stopping: Dừng sớm nếu 5 Epoch không cải thiện (patience=5)")
    print(" - LR Scheduler: Tự động giảm Learning Rate nếu chững lại (ReduceLROnPlateau)")
    
    trainer = ModelTrainer(
        model=classifier.model,
        train_loader=train_loader,
        val_loader=val_loader,
        patience=5,             # <--- EARLY STOPPING
        learning_rate=0.001     # <--- LR SCHEDULER được kích hoạt ngầm
    )

    # Chạy thử 3 Epochs để nộp Demo cho nhóm
    train_losses, val_losses = trainer.train(num_epochs=3, save_path="demo_weights.pth")
    print("\n[HOÀN TẤT] Quá trình Demo kết thúc thành công!")

if __name__ == "__main__":
    run_demo()
