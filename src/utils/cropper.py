import cv2
import numpy as np
from typing import List

class ImageCropper:
    """
    Lớp xử lý hình ảnh: Chuyên nhận tọa độ Bounding Box và cắt vật thể ra khỏi nền.
    (Lấy ý tưởng từ file cut_bbox_litter.py của tác giả repo gốc)
    """
    def __init__(self, padding: int = 5):
        """
        Args:
            padding: Số lượng pixel viền dư ra khi cắt (để ảnh không bị cụt quá sát).
        """
        self.padding = padding

    def crop_objects(self, image_path: str, bounding_boxes: List[List[float]]) -> List[np.ndarray]:
        """
        Đọc ảnh và cắt ra các bức ảnh mini dựa trên danh sách tọa độ.
        
        Args:
            image_path: Đường dẫn tới bức ảnh gốc.
            bounding_boxes: Danh sách các tọa độ [x_min, y_min, x_max, y_max].
            
        Returns:
            Danh sách các bức ảnh mini (dạng Numpy array) đã được cắt.
        """
        # 1. Đọc ảnh bằng OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc được ảnh tại: {image_path}")
            
        # Chuyển BGR sang RGB để đưa vào model Classification sau này cho chuẩn màu
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_height, img_width = img.shape[:2]
        cropped_images = []

        # 2. Duyệt qua từng tọa độ để cắt
        for box in bounding_boxes:
            # Làm tròn số thành số nguyên (pixel)
            x1, y1, x2, y2 = map(int, box)
            
            # Thêm Padding (viền) nhưng không được tràn ra ngoài bức ảnh gốc
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(img_width, x2 + self.padding)
            y2 = min(img_height, y2 + self.padding)
            
            # 3. Cắt (Crop) bằng ma trận Numpy
            crop = img[y1:y2, x1:x2]
            
            # Kiểm tra nếu cắt ra ảnh bị rỗng (tọa độ lỗi)
            if crop.size > 0:
                cropped_images.append(crop)
                
        return cropped_images
