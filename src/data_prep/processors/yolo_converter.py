import os
import shutil
import logging
from tqdm import tqdm
from src.data_prep.processors.base_processor import BaseAnnotationProcessor

logger = logging.getLogger("YoloFormatProcessor")

class YoloFormatProcessor(BaseAnnotationProcessor):
    """
    Chuyển đổi dataset_dict (COCO format) thành định dạng TXT của YOLO
    và sao chép ảnh vào thư mục cấu trúc chuẩn.
    """
    def __init__(self, dataset_dict: dict, images_source_dir: str, output_dir: str):
        super().__init__(dataset_dict)
        self.images_source_dir = images_source_dir
        self.output_dir = output_dir

    def transform(self):
        logger.info(f"YoloFormatProcessor: Bắt đầu sinh dữ liệu YOLO tại {self.output_dir}...")
        images_out_dir = os.path.join(self.output_dir, "images")
        labels_out_dir = os.path.join(self.output_dir, "labels")
        
        os.makedirs(images_out_dir, exist_ok=True)
        os.makedirs(labels_out_dir, exist_ok=True)

        # Tạo từ điển tra cứu nhanh thông tin ảnh
        img_dict = {}
        for img in self.dataset.get('images', []):
            img_dict[img['id']] = img

        # Nhóm annotations theo image_id
        ann_by_img = {}
        for ann in self.dataset.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in ann_by_img:
                ann_by_img[img_id] = []
            ann_by_img[img_id].append(ann)

        processed_count = 0
        
        # Xử lý từng bức ảnh
        for img_id, img_info in tqdm(img_dict.items(), desc="Chuyển đổi YOLO"):
            file_name = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']
            
            # Base name (vd: batch_1/000006.jpg -> 000006)
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            txt_path = os.path.join(labels_out_dir, base_name + '.txt')
            
            # Lưu tọa độ vào file .txt
            with open(txt_path, 'w') as f:
                if img_id in ann_by_img:
                    for ann in ann_by_img[img_id]:
                        cat_id = ann['category_id']
                        # Lưu ý: nếu đã chạy Binarization, cat_id sẽ là 0
                        # Nếu chưa chạy, cat_id có thể là 1-7. YOLO bắt đầu từ 0.
                        # Do đó, cần trừ 1 nếu cat_id > 0 (để 1->0, 2->1).
                        # Binarization gán thẳng 0, nên nếu là 0 thì giữ nguyên.
                        final_cat_id = cat_id - 1 if cat_id > 0 else 0
                        
                        x_min, y_min, bbox_w, bbox_h = ann['bbox']
                        
                        # Chuẩn hóa (Normalize)
                        x_center = (x_min + bbox_w / 2.0) / img_w
                        y_center = (y_min + bbox_h / 2.0) / img_h
                        width = bbox_w / img_w
                        height = bbox_h / img_h
                        
                        # Kẹp trong khoảng [0, 1]
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))
                        
                        f.write(f"{final_cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        processed_count += 1
            
            # Sao chép ảnh (Flatten cấu trúc thư mục)
            src_img_path = os.path.join(self.images_source_dir, file_name)
            if os.path.exists(src_img_path):
                dst_img_path = os.path.join(images_out_dir, base_name + os.path.splitext(file_name)[1])
                shutil.copy(src_img_path, dst_img_path)

        logger.info(f"YoloFormatProcessor: Hoàn tất! Đã xuất {processed_count} bounding boxes sang TXT.")
