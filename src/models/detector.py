from ultralytics import YOLO, RTDETR

class TrashDetector:
    """
    Lớp định vị (Detection) bọc thư viện Ultralytics.
    Hỗ trợ dễ dàng chuyển đổi giữa YOLOv8 (Mạng CNN) và RT-DETR (Mạng Transformer).
    """
    def __init__(self, model_path: str = 'yolov8n.pt', model_type: str = 'yolo'):
        """
        Khởi tạo mô hình định vị rác.
        
        Args:
            model_path: Đường dẫn tới file trọng số (ví dụ: 'best.pt' sau khi train).
            model_type: 'yolo' hoặc 'rtdetr'.
        """
        self.model_type = model_type.lower()
        print(f"[Detector] Đang nạp mô hình {self.model_type.upper()} từ {model_path}...")
        
        if self.model_type == 'yolo':
            self.model = YOLO(model_path)
        elif self.model_type == 'rtdetr':
            self.model = RTDETR(model_path)
        else:
            raise ValueError("Chỉ hỗ trợ model_type là 'yolo' hoặc 'rtdetr'")

    def predict(self, image_path: str, conf_threshold: float = 0.25):
        """
        Dự đoán vị trí cục rác trong ảnh.
        
        Args:
            image_path: Đường dẫn bức ảnh.
            conf_threshold: Ngưỡng độ tin cậy (nhỏ hơn mức này thì vứt bỏ).
            
        Returns:
            Danh sách các tọa độ [x_min, y_min, x_max, y_max].
        """
        bounding_boxes = []
        
        if self.model_type in ['yolo', 'rtdetr']:
            # Gọi thư viện Ultralytics dự đoán
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            for result in results:
                boxes_data = result.boxes.xyxy.cpu().numpy().tolist()
                bounding_boxes.extend(boxes_data)
            
        print(f"[Detector] Đã phát hiện {len(bounding_boxes)} cục rác trong ảnh.")
        return bounding_boxes
