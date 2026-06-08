import os
import cv2
from tqdm import tqdm

def crop_and_save_dataset(images_dir, labels_dir, output_dir, class_names):
    """
    Trích xuất phân vùng ảnh (ROI) từ Bounding Box.
    Đọc tọa độ chuẩn từ tệp nhãn YOLO, trích xuất vùng ảnh và lưu vào cấu trúc thư mục phân loại.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Khởi tạo cấu trúc thư mục phân loại
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
    # Trích xuất danh sách tệp ảnh từ tất cả các thư mục con (VD: batch_1, batch_2)
    import glob
    image_files = glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True)
    image_files.extend(glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True))
    
    print(f"Tiến hành trích xuất phân vùng trên tổng số {len(image_files)} ảnh...")
    
    crop_count = 0
    for img_path in tqdm(image_files, desc="Đang trích xuất"):
        # Tương thích với cấu trúc phẳng (Flatten) của thư mục labels
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        
        # Bỏ qua nếu ảnh này không có file nhãn (không có rác)
        if not os.path.exists(txt_path):
            continue
            
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img, _ = img.shape
        
        # Đọc file txt chứa tọa độ (Format YOLO: class_id x_center y_center width height)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            
            # Chuyển đổi tọa độ YOLO (tỷ lệ 0-1) sang Pixel thực tế
            x_center = int(x_c * w_img)
            y_center = int(y_c * h_img)
            width = int(w * w_img)
            height = int(h * h_img)
            
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(w_img, int(x_center + width / 2))
            y2 = min(h_img, int(y_center + height / 2))
            
            # Bỏ qua nếu nhãn dán bị lỗi (rác quá nhỏ dẫn đến width=0 hoặc height=0)
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Cắt ảnh
            cropped_img = img[y1:y2, x1:x2]
            
            # Lưu ảnh vào đúng thư mục tên rác
            class_name = class_names[class_id]
            save_name = f"{os.path.splitext(img_name)[0]}_crop{i}.jpg"
            save_path = os.path.join(output_dir, class_name, save_name)
            
            cv2.imwrite(save_path, cropped_img)
            crop_count += 1
            
    print(f"\n[HOÀN TẤT] Quá trình trích xuất kết thúc với {crop_count} phân vùng ảnh.")
    print(f"Dữ liệu phân loại được lưu trữ tại: {output_dir}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='/kaggle/working/taco-trash-detection-dataset/images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/working/taco-trash-detection-dataset/labels')
    args = parser.parse_args()
    
    OUTPUT_DIR = "datasets/classifier_data/train"
    CLASS_NAMES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]
    
    if os.path.exists(args.images_dir):
        crop_and_save_dataset(args.images_dir, args.labels_dir, OUTPUT_DIR, CLASS_NAMES)
    else:
        print(f"Không tìm thấy thư mục ảnh: {args.images_dir}")
