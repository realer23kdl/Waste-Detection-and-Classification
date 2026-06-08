import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List

class TrashClassifier:
    """
    Lớp Phân loại (Classification) hỗ trợ cả ResNet50 và EfficientNet-B0.
    Chuyên đọc các bức ảnh mini (đã được cắt rác) để phân loại thành Nhựa, Giấy, Kim loại...
    """
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 6, model_weights_path: str = None):
        """
        Khởi tạo mạng phân loại.
        
        Args:
            model_name: Tên mạng muốn sử dụng ('resnet50' hoặc 'efficientnet_b0').
            num_classes: Số lượng loại rác (TrashNet là 6).
            model_weights_path: Đường dẫn tới file trọng số sau khi train (VD: best_resnet.pth).
        """
        self.model_name = model_name.lower()
        print(f"[Classifier] Đang nạp mô hình {self.model_name.upper()}...")
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif self.model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        elif self.model_name == 'mobilenet_v3':
            self.model = models.mobilenet_v3_small(pretrained=False)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = torch.nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("Chỉ hỗ trợ 'resnet50', 'efficientnet_b0', hoặc 'mobilenet_v3'")
        
        # Nếu có file trọng số tự train thì nạp vào
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
            print(f"[Classifier] Đã nạp thành công trọng số từ {model_weights_path}")
            
        self.model.eval() # Bật chế độ đánh giá (không train)
        
        # Cấu hình tiền xử lý ảnh theo chuẩn của ResNet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Ánh xạ ID ra tên chữ (Ví dụ)
        self.class_names = ["Thủy tinh", "Giấy", "Bìa cứng", "Nhựa", "Kim loại", "Rác hỗn hợp"]

    def predict(self, cropped_images: List[np.ndarray]) -> List[str]:
        """
        Dự đoán danh sách ảnh mini.
        
        Args:
            cropped_images: Danh sách các ảnh Numpy do thằng Cropper cắt ra.
            
        Returns:
            Danh sách tên các loại rác tương ứng.
        """
        results = []
        for img_array in cropped_images:
            # Tiền xử lý
            input_tensor = self.preprocess(img_array)
            input_batch = input_tensor.unsqueeze(0) # Tạo batch size = 1
            
            # Dự đoán
            with torch.no_grad():
                output = self.model(input_batch)
                
            # Lấy vị trí có xác suất cao nhất
            _, predicted_idx = torch.max(output, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            results.append(predicted_class)
            
        return results
