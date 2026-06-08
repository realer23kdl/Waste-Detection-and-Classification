import cv2
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Khối trực quan hóa phân tích lỗi (Error Analysis) 
    dành riêng cho Object Detection và Segmentation.
    """
    @staticmethod
    def plot_bounding_boxes(image_path, true_boxes, pred_boxes, save_path="error_analysis_detection.png"):
        """
        Vẽ đè Khung thực tế (Ground Truth) và Khung dự đoán (Prediction) lên cùng 1 ảnh.
        Màu xanh lá: Thực tế (Đúng)
        Màu đỏ: Dự đoán (Mô hình)
        Giúp thấy rõ mô hình khoanh to quá hay nhỏ quá.
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Vẽ Ground Truth (Xanh lá)
        for box in true_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'Thuc Te', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Vẽ Dự đoán (Đỏ)
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, 'Du Doan', (x2-60, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title("Phân tích lỗi Khoanh Vùng (Xanh: Đúng, Đỏ: Mô hình)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[Visualizer] Đã lưu ảnh so sánh Bounding Box tại: {save_path}")
        plt.show()

    @staticmethod
    def plot_segmentation_masks(image_path, true_mask, pred_mask, save_path="error_analysis_segmentation.png"):
        """
        Vẽ đè Mặt nạ thực tế và Mặt nạ dự đoán lên ảnh.
        Hiển thị 3 khung hình: Ảnh Gốc | Ảnh Thực Tế | Ảnh Dự Đoán
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tạo lớp overlay màu
        true_overlay = img.copy()
        pred_overlay = img.copy()
        
        # Bôi màu xanh lá cho mask đúng, màu đỏ cho mask dự đoán
        true_overlay[true_mask == 1] = [0, 255, 0]
        pred_overlay[pred_mask == 1] = [255, 0, 0]
        
        # Trộn màu (alpha blending) để nhìn xuyên thấu
        img_true = cv2.addWeighted(true_overlay, 0.5, img, 0.5, 0)
        img_pred = cv2.addWeighted(pred_overlay, 0.5, img, 0.5, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Ảnh Gốc")
        axes[0].axis('off')
        
        axes[1].imshow(img_true)
        axes[1].set_title("Mask Thực Tế (Ground Truth)")
        axes[1].axis('off')
        
        axes[2].imshow(img_pred)
        axes[2].set_title("Mask Dự Đoán (Prediction)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[Visualizer] Đã lưu ảnh phân tích Mặt nạ tại: {save_path}")
        plt.show()
