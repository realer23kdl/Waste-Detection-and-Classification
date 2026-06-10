import sys
import os

# Thêm đường dẫn gốc để import các module dễ dàng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import TrashDetector
from utils.cropper import ImageCropper
from models.classifier import TrashClassifier

class DetectAndClassifyPipeline:
    """
    Kẻ điều phối vĩ đại: Nối 3 công đoạn Detection -> Crop -> Classification lại thành 1 luồng.
    """
    def __init__(self, yolo_weights='yolov8n.pt', resnet_weights=None, classifier_model_name='resnet50'):
        """
        Khởi tạo toàn bộ dây chuyền.
        """
        print("=== KHỞI ĐỘNG DÂY CHUYỀN XỬ LÝ RÁC ===")
        # 1. Mạng tìm rác (Của bạn)
        self.detector = TrashDetector(model_path=yolo_weights, model_type='yolo')
        
        # 2. Cái kéo cắt ảnh (Của bạn)
        self.cropper = ImageCropper(padding=10)
        
        # 3. Mạng phân loại (Của người số 1)
        # Sửa thành num_classes=7 vì tập data có thêm thư mục "unknown"
        self.classifier = TrashClassifier(model_name=classifier_model_name, num_classes=7, model_weights_path=resnet_weights)
        print("======================================\n")

    def run(self, image_path: str, conf_threshold: float = 0.05):
        """
        Chạy toàn bộ dây chuyền.
        """
        print(f"-> Đang xử lý bức ảnh: {image_path}")
        
        # BƯỚC 1: Tìm rác (Đã hạ ngưỡng độ tin cậy xuống 0.05 để bù đắp cho mô hình train ít epoch)
        bounding_boxes = self.detector.predict(image_path, conf_threshold=conf_threshold)
        
        if not bounding_boxes:
            print("Không tìm thấy rác trong ảnh này!")
            return
            
        # BƯỚC 2: Cắt rác ra khỏi bối cảnh
        cropped_images = self.cropper.crop_objects(image_path, bounding_boxes)
        print(f"[Pipeline] Đã cắt thành công {len(cropped_images)} mảnh rác mini.")
        
        # BƯỚC 3: Đưa rác mini đi đọc tên và lấy xác suất (Confidence)
        predictions = self.classifier.predict(cropped_images, return_prob=True)
        
        # TỔNG KẾT VÀ VẼ ẢNH TRỰC QUAN
        import cv2
        img = cv2.imread(image_path)
        labels_only = []
        
        print("\n=== KẾT QUẢ CUỐI CÙNG ===")
        for i, (label, prob) in enumerate(predictions):
            box = bounding_boxes[i]
            x_min, y_min, x_max, y_max = map(int, box[:4])
            labels_only.append(label)
            
            # Text hiển thị (Ví dụ: NHỰA (95.5%))
            display_text = f"{label.upper()} ({prob*100:.1f}%)"
            print(f"Cục rác thứ {i+1} ở tọa độ {box} là: {display_text}")
            
            # Vẽ khung Bounding Box màu xanh lá
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Vẽ nền chữ
            (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x_min, y_min - 30), (x_min + w, y_min), (0, 255, 0), -1)
            # Viết chữ đen lên nền xanh
            cv2.putText(img, display_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        print("=========================")
        
        # Lưu ảnh
        output_path = "inference_result.jpg"
        cv2.imwrite(output_path, img)
        print(f"\n[Visualizer] TADA! Đã vẽ xong khung, nhãn và độ tin cậy.")
        print(f"[Visualizer] Ảnh trực quan đã được lưu tại: {output_path}")
        
        return bounding_boxes, labels_only
