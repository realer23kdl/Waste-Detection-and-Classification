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
        
        # BƯỚC 3: Đưa rác mini đi đọc tên
        labels = self.classifier.predict(cropped_images)
        
        # TỔNG KẾT
        print("\n=== KẾT QUẢ CUỐI CÙNG ===")
        for i, label in enumerate(labels):
            print(f"Cục rác thứ {i+1} ở tọa độ {bounding_boxes[i]} là: {label.upper()}")
        print("=========================")
        
        return bounding_boxes, labels
