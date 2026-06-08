import os
import cv2
import albumentations as A
from tqdm import tqdm

def apply_augmentation(images_dir, labels_dir, output_images_dir, output_labels_dir):
    """
    Sử dụng Albumentations để tự động xoay/lật ảnh và tính toán lại Bounding Box.
    Code này dùng để phô diễn kỹ năng xử lý Data cho báo cáo.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Định nghĩa 파íp biến đổi: Lật ngang, Đổi độ sáng, Thêm nhiễu
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    print(f"Đang tiến hành Augmentation cho {len(image_files)} ảnh...")
    
    count = 0
    for img_name in tqdm(image_files):
        img_path = os.path.join(images_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        
        if not os.path.exists(txt_path):
            continue
            
        # Đọc ảnh
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Đọc nhãn YOLO
        bboxes = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Format: [x_center, y_center, width, height, class_id]
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), class_id])
                    
        # Nếu ảnh không có nhãn rác nào hợp lệ thì bỏ qua
        if len(bboxes) == 0:
            continue
            
        try:
            # Thực hiện Augmentation (Lật ảnh + Tự động dời Bounding box)
            transformed = transform(image=image, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            
            # Lưu ảnh mới
            new_img_name = f"aug_{img_name}"
            save_img_path = os.path.join(output_images_dir, new_img_name)
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_img_path, transformed_image)
            
            # Lưu nhãn mới
            new_txt_name = f"aug_{txt_name}"
            save_txt_path = os.path.join(output_labels_dir, new_txt_name)
            with open(save_txt_path, 'w') as f:
                for bbox in transformed_bboxes:
                    # Ghi lại format YOLO: class_id x_c y_c w h
                    class_id = bbox[4]
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            count += 1
        except Exception as e:
            # Albumentations có thể lỗi nếu BBox tràn ra ngoài ảnh
            continue
            
    print(f"\n[HOÀN THÀNH] Đã nhân bản thành công {count} ảnh (kèm Bounding Box mới)!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    args = parser.parse_args()
    
    OUT_IMG_DIR = "datasets/train_augmented/images"
    OUT_LBL_DIR = "datasets/train_augmented/labels"
    
    if os.path.exists(args.images_dir) and os.path.exists(args.labels_dir):
        apply_augmentation(args.images_dir, args.labels_dir, OUT_IMG_DIR, OUT_LBL_DIR)
    else:
        print(f"Không tìm thấy dữ liệu tại: {args.images_dir}")
