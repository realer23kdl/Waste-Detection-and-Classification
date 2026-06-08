import os
import cv2
import logging
from tqdm import tqdm
from src.data_prep.processors.base_processor import BaseAnnotationProcessor

logger = logging.getLogger("ImageCropProcessor")

class ImageCropProcessor(BaseAnnotationProcessor):
    """
    Trích xuất phân vùng ảnh (ROI) dựa trên thông tin Bounding Box từ dataset_dict.
    Cắt ảnh và lưu vào thư mục phân loại tương ứng (để huấn luyện Classifier).
    """
    def __init__(self, dataset_dict: dict, images_source_dir: str, output_dir: str):
        super().__init__(dataset_dict)
        self.images_source_dir = images_source_dir
        self.output_dir = output_dir

    def transform(self):
        logger.info(f"ImageCropProcessor: Bắt đầu quá trình cắt ảnh tại {self.output_dir}...")
        
        # Tạo từ điển tra cứu ID -> Tên Class
        class_dict = {}
        for cat in self.dataset.get('categories', []):
            class_dict[cat['id']] = cat['name']
            
        # Tạo cấu trúc thư mục đích cho từng Class
        for cat_name in class_dict.values():
            os.makedirs(os.path.join(self.output_dir, cat_name), exist_ok=True)

        # Nhóm annotations theo image_id
        ann_by_img = {}
        for ann in self.dataset.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in ann_by_img:
                ann_by_img[img_id] = []
            ann_by_img[img_id].append(ann)

        crop_count = 0
        
        # Xử lý từng bức ảnh
        for img_info in tqdm(self.dataset.get('images', []), desc="Cắt ảnh"):
            img_id = img_info['id']
            if img_id not in ann_by_img:
                continue
                
            file_name = img_info['file_name']
            src_img_path = os.path.join(self.images_source_dir, file_name)
            
            if not os.path.exists(src_img_path):
                continue
                
            img = cv2.imread(src_img_path)
            if img is None:
                continue
                
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            
            for i, ann in enumerate(ann_by_img[img_id]):
                cat_id = ann['category_id']
                cat_name = class_dict.get(cat_id, "unknown")
                
                x_min, y_min, bbox_w, bbox_h = map(int, map(float, ann['bbox']))
                x_max = x_min + bbox_w
                y_max = y_min + bbox_h
                
                # Bỏ qua nếu kích thước quá nhỏ hoặc lỗi
                if bbox_w <= 0 or bbox_h <= 0:
                    continue
                    
                # Cắt ảnh
                cropped_img = img[y_min:y_max, x_min:x_max]
                if cropped_img.size == 0:
                    continue
                    
                save_name = f"{base_name}_crop{i}.jpg"
                save_path = os.path.join(self.output_dir, cat_name, save_name)
                
                cv2.imwrite(save_path, cropped_img)
                crop_count += 1
                
        logger.info(f"ImageCropProcessor: Hoàn tất! Đã cắt và lưu {crop_count} phân vùng ảnh.")
