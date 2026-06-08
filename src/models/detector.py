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
        elif self.model_type == 'efficientdet':
            try:
                import torch
                from effdet import create_model_from_config, get_efficientdet_config
            except ImportError:
                raise ImportError("Vui lòng cài đặt thư viện effdet: pip install effdet timm")
            
            # Cấu trúc mẫu để nạp mô hình EfficientDet (D0)
            config = get_efficientdet_config('tf_efficientdet_d0')
            config.num_classes = 6
            config.image_size = (640, 640)
            
            self.model = create_model_from_config(config, bench_task='predict', num_classes=6)
            # Nếu model_path không tồn tại (chưa train), thì nạp model trống để code không báo lỗi
            import os
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
        else:
            raise ValueError("Chỉ hỗ trợ model_type là 'yolo', 'rtdetr' hoặc 'efficientdet'")

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
                
        elif self.model_type == 'efficientdet':
            # Logic dự đoán dành cho mô hình PyTorch thuần (EfficientDet)
            import cv2
            import torch
            
            # 1. Đọc và tiền xử lý ảnh (Resize thủ công)
            img = cv2.imread(image_path)
            h_orig, w_orig = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            
            # 2. Chuyển thành Tensor chuẩn hóa
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0) # (1, 3, 640, 640)
            
            # 3. Đưa vào não bộ dự đoán
            with torch.no_grad():
                output = self.model(img_tensor)
                
            # 4. Bóc tách kết quả [x_min, y_min, x_max, y_max, score, class_id]
            if output is not None and len(output) > 0:
                detections = output[0] # Lấy ảnh đầu tiên
                for det in detections:
                    score = float(det[4])
                    if score >= conf_threshold:
                        # Quan trọng: Scale (bóp) tọa độ hộp trả về kích thước ảnh gốc
                        scale_x = w_orig / 640.0
                        scale_y = h_orig / 640.0
                        
                        x1, y1 = float(det[0]) * scale_x, float(det[1]) * scale_y
                        x2, y2 = float(det[2]) * scale_x, float(det[3]) * scale_y
                        bounding_boxes.append([x1, y1, x2, y2])
            
        print(f"[Detector] Đã phát hiện {len(bounding_boxes)} cục rác trong ảnh.")
        return bounding_boxes
