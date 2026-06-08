import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class Evaluator:
    """
    Khối đánh giá (Evaluation): Tính toán toàn bộ các chỉ số định lượng
    và trực quan hóa bằng biểu đồ theo đúng chuẩn báo cáo khoa học.
    """
    def __init__(self, class_names):
        """
        Args:
            class_names: Danh sách tên các nhãn (ví dụ: ['Nhựa', 'Giấy', ...])
        """
        self.class_names = class_names

    def calculate_metrics(self, y_true, y_pred):
        """
        Tính 4 chỉ số vàng: Accuracy, Precision, Recall, F1-Score.
        """
        acc = accuracy_score(y_true, y_pred)
        # Sử dụng macro average để tính trung bình đều cho các nhãn (quan trọng với dữ liệu mất cân bằng)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print("\n=== BẢNG THÀNH TÍCH MÔ HÌNH ===")
        print(f"1. Accuracy (Độ chính xác tổng): {acc*100:.2f}%")
        print(f"2. Precision (Độ chuẩn xác):    {prec*100:.2f}%")
        print(f"3. Recall (Độ phủ):           {rec*100:.2f}%")
        print(f"4. F1-Score (Cân bằng):       {f1*100:.2f}%")
        print("===============================\n")
        
        return acc, prec, rec, f1

    def plot_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        """
        Vẽ Ma trận nhầm lẫn (Confusion Matrix) cực đẹp bằng Seaborn.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        
        plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
        plt.ylabel('Nhãn Thực Tế (True)')
        plt.xlabel('Nhãn Dự Đoán (Predicted)')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300)
        print(f"[Metrics] Đã lưu biểu đồ ma trận nhầm lẫn tại: {save_path}")
        # plt.show() # Tạm tắt show() tự động để chạy mượt trên Kaggle/Server

    def plot_wrong_predictions(self, dataset, y_true, y_pred, num_samples=9, save_path="wrong_predictions.png"):
        """
        [Rubric: Error Analysis]
        Trực quan hóa các mẫu dự đoán sai để phân tích điểm yếu của mô hình.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Tìm các index dự đoán sai
        wrong_indices = np.where(y_true != y_pred)[0]
        
        if len(wrong_indices) == 0:
            print("[Error Analysis] Tuyệt vời! Không có mẫu nào dự đoán sai.")
            return
            
        print(f"[Error Analysis] Đã tìm thấy {len(wrong_indices)} mẫu dự đoán sai. Đang vẽ {min(num_samples, len(wrong_indices))} mẫu tiêu biểu...")
        
        # Chọn ngẫu nhiên (hoặc đầu tiên) num_samples
        np.random.shuffle(wrong_indices)
        selected_indices = wrong_indices[:num_samples]
        
        cols = 3
        rows = int(np.ceil(len(selected_indices) / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i, idx in enumerate(selected_indices):
            # Lấy đường dẫn ảnh từ dataset (vì DataLoader đã xáo trộn nên cần gọi trực tiếp ảnh gốc bằng index)
            # Lưu ý: DataLoader shuffle=True sẽ làm mất index, nên truyền Dataset không shuffle vào hàm này.
            image, _ = dataset[idx] 
            
            # Un-normalize image
            img_np = image.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            ax = axes[i] if len(axes) > 1 else axes[0]
            ax.imshow(img_np)
            ax.set_title(f"Thực tế: {true_label}\nDự đoán: {pred_label}", color='red', fontsize=10)
            ax.axis('off')
            
        # Ẩn các ô thừa
        for j in range(i + 1, len(axes)):
            if len(axes) > 1:
                axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[Error Analysis] Đã lưu báo cáo trực quan ảnh lỗi tại: {save_path}")

    def plot_loss_curves(self, train_losses, val_losses, save_path="loss_curve.png"):
        """
        Vẽ đường cong Loss qua các Epochs để kiểm tra mô hình có bị Overfitting hay không.
        (Thỏa mãn yêu cầu vẽ biểu đồ Loss của rubric)
        """
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
        plt.title('Biểu đồ Hàm Mất Mát (Loss Curve)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=300)
        print(f"[Metrics] Đã lưu biểu đồ Loss tại: {save_path}")
        plt.show()

    def calculate_iou(self, boxA, boxB):
        """
        Tính toán Intersection over Union (IoU) giữa 2 Bounding Box.
        Box format: [x_min, y_min, x_max, y_max]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def evaluate_detection(self, y_true_boxes, y_pred_boxes):
        """
        (Cho Detection) Đánh giá chất lượng khoanh vùng bằng IoU trung bình.
        Trong thực tế, mAP sẽ được tính tự động bởi YOLO (results.box.map).
        Hàm này hỗ trợ tính mIoU thủ công để báo cáo.
        """
        ious = []
        for true_box, pred_box in zip(y_true_boxes, y_pred_boxes):
            iou = self.calculate_iou(true_box, pred_box)
            ious.append(iou)
        
        mean_iou = np.mean(ious)
        print(f"\n[Detection Metrics] Trung bình IoU (Độ khớp khung vuông): {mean_iou*100:.2f}%")
        return mean_iou

    def evaluate_segmentation(self, y_true_mask, y_pred_mask):
        """
        (Cho Segmentation) Đánh giá mặt nạ bằng mIoU và Dice Coefficient.
        y_true_mask, y_pred_mask: Mảng Numpy 2D chứa nhãn pixel (0 là nền, 1 là rác).
        """
        intersection = np.logical_and(y_true_mask, y_pred_mask)
        union = np.logical_or(y_true_mask, y_pred_mask)
        
        iou = np.sum(intersection) / np.sum(union)
        dice = (2. * np.sum(intersection)) / (np.sum(y_true_mask) + np.sum(y_pred_mask))
        
        print(f"\n[Segmentation Metrics] mIoU (Độ khớp mặt nạ): {iou*100:.2f}%")
        print(f"[Segmentation Metrics] Dice Score (Chỉ số F1 cho pixel): {dice*100:.2f}%")
        return iou, dice
