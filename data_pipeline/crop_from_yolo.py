import os
import sys
import cv2
import yaml
import glob
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def crop_yolo_dataset(yolo_dir, output_dir):
    """
    Đọc thư mục dữ liệu chuẩn YOLO (từ Roboflow) và cắt tất cả các Bounding Box
    thành các bức ảnh nhỏ để lưu vào thư mục cho mạng Classifier huấn luyện.
    """
    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Lỗi: Không tìm thấy file {yaml_path}. Vui lòng trỏ đúng thư mục chứa data.yaml!")
        return
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        
    names = data_yaml.get('names', [])
    print(f"Tìm thấy {len(names)} nhãn: {names}")
    
    # Tạo sẵn các thư mục con chứa ảnh từng loại rác (Nhựa, Giấy, Kim loại...)
    for name in names:
        os.makedirs(os.path.join(output_dir, str(name)), exist_ok=True)
        
    splits = ['train', 'valid', 'test']
    total_crops = 0
    
    for split in splits:
        img_dir = os.path.join(yolo_dir, split, 'images')
        lbl_dir = os.path.join(yolo_dir, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        print(f"\nĐang lấy kéo cắt rác trong tập [{split.upper()}]...")
        for img_path in glob.glob(os.path.join(img_dir, '*.*')):
            img_name = os.path.basename(img_path)
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(lbl_dir, txt_name)
            
            if not os.path.exists(txt_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])
                
                # Chuyển đổi tọa độ YOLO (tỉ lệ 0-1) về tọa độ Pixel của ảnh
                x1 = int((x_center - bbox_w/2) * w)
                y1 = int((y_center - bbox_h/2) * h)
                x2 = int((x_center + bbox_w/2) * w)
                y2 = int((y_center + bbox_h/2) * h)
                
                # Mở rộng (Padding) 10px để không cắt phạm vào rác
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(w, x2 + 10)
                y2 = min(h, y2 + 10)
                
                crop_img = img[y1:y2, x1:x2]
                if crop_img.size == 0: continue
                
                cls_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                
                # Lưu tấm ảnh mini vào thư mục tương ứng của Classifier
                # VD: datasets/classifier_data/train/Plastic/img_0.jpg
                out_path = os.path.join(output_dir, str(cls_name), f"{split}_{os.path.splitext(img_name)[0]}_{idx}.jpg")
                cv2.imwrite(out_path, crop_img)
                total_crops += 1

    print(f"\n[HOÀN TẤT] Đã cắt thành công {total_crops} cục rác và lưu vào: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool cắt ảnh rác từ data YOLO cho Classifier")
    parser.add_argument('--yolo_dir', type=str, required=True, help="Thư mục gốc chứa file data.yaml")
    parser.add_argument('--output_dir', type=str, default='datasets/classifier_data/train', help="Thư mục xuất ảnh cắt")
    args = parser.parse_args()
    
    crop_yolo_dataset(args.yolo_dir, args.output_dir)
