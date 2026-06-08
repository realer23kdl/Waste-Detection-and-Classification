import os
import json
import argparse
from tqdm import tqdm

# Bảng tra cứu tĩnh (Hardcoded Mapping Dictionary)
# Ánh xạ chính xác tuyệt đối 60 ID gốc của TACO sang 6 phân lớp chính (Super Categories).
# 0: Glass, 1: Paper, 2: Cardboard, 3: Plastic, 4: Metal, 5: Trash
TACO_MAPPING = {
    # 0: Thủy tinh (Glass)
    0: [6, 9, 23, 26],
    # 1: Giấy (Paper)
    1: [20, 30, 31, 32, 33, 34, 35, 56],
    # 2: Bìa cứng (Cardboard)
    2: [13, 14, 15, 16, 17, 18, 19],
    # 3: Nhựa (Plastic)
    3: [4, 5, 7, 21, 24, 27, 29, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 54, 55],
    # 4: Kim loại (Metal)
    4: [0, 1, 2, 8, 10, 11, 12, 28, 50, 52],
    # 5: Rác hỗn hợp/Không tái chế (Trash) - Gồm cả xốp (Foam), hữu cơ, rác vụn...
    5: [3, 22, 25, 46, 51, 53, 57, 58, 59]
}

# Tạo từ điển tra cứu ngược (Reverse Lookup) O(1)
ID_TO_SUPER_ID = {}
for super_id, old_ids in TACO_MAPPING.items():
    for old_id in old_ids:
        ID_TO_SUPER_ID[old_id] = super_id

def map_taco_to_6_classes(old_cat_id):
    """
    Thuật toán Gom cụm nhãn (Label Clustering) dựa trên Bảng tra cứu ID tĩnh.
    Đảm bảo tính chặt chẽ 100% cho báo cáo khoa học.
    """
    # Nếu một ID lạ xuất hiện (do lỗi dữ liệu), mặc định phân vào nhóm Rác hỗn hợp (5)
    return ID_TO_SUPER_ID.get(old_cat_id, 5)

def convert_coco_json_to_yolo_txt(json_path, images_dir, output_labels_dir):
    """
    Trích xuất dữ liệu từ tệp annotations_unofficial.json, thực hiện gom nhóm nhãn,
    và chuyển đổi định dạng tọa độ sang YOLO TXT.
    """
    if not os.path.exists(json_path):
        print(f"[LỖI] Không tìm thấy tệp JSON tại: {json_path}")
        return

    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"Đang tiến hành nạp dữ liệu COCO JSON vào bộ nhớ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Bước 1: Xây dựng bảng tra cứu ánh xạ danh mục (60 -> 6)
    print("Đang khởi chạy thuật toán Label Clustering (60 -> 6)...")
    id_mapping = {}
    for cat in data['categories']:
        old_id = cat['id']
        new_id = map_taco_to_6_classes(old_id)
        id_mapping[old_id] = new_id
        
    # Bước 2: Khởi tạo thông tin không gian của ảnh
    img_dict = {}
    for img in data['images']:
        img_dict[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
        
    # Bước 3: Chuyển đổi định dạng Bounding Box
    print("Đang thực hiện chuyển đổi tọa độ Bounding Box sang định dạng YOLO...")
    annotations_count = 0
    
    # Khởi tạo các tệp nhãn trống
    for img_id, img_info in img_dict.items():
        txt_name = os.path.splitext(os.path.basename(img_info['file_name']))[0] + '.txt'
        open(os.path.join(output_labels_dir, txt_name), 'w').close()
        
    for ann in tqdm(data['annotations'], desc="Tiến trình chuyển đổi"):
        img_id = ann['image_id']
        if img_id not in img_dict:
            continue
            
        old_cat_id = ann['category_id']
        if old_cat_id not in id_mapping:
            continue
            
        new_cat_id = id_mapping[old_cat_id]
        
        # Tiêu chuẩn hóa tọa độ (Normalized Coordinates)
        img_w = img_dict[img_id]['width']
        img_h = img_dict[img_id]['height']
        
        x_min, y_min, bbox_w, bbox_h = ann['bbox']
        
        x_center = (x_min + bbox_w / 2.0) / img_w
        y_center = (y_min + bbox_h / 2.0) / img_h
        width = bbox_w / img_w
        height = bbox_h / img_h
        
        x_center, y_center = max(0.0, min(1.0, x_center)), max(0.0, min(1.0, y_center))
        width, height = max(0.0, min(1.0, width)), max(0.0, min(1.0, height))
        
        # Sử dụng basename để làm phẳng (Flatten) toàn bộ file nhãn ra thư mục gốc
        txt_name = os.path.splitext(os.path.basename(img_dict[img_id]['file_name']))[0] + '.txt'
        txt_path = os.path.join(output_labels_dir, txt_name)
        
        with open(txt_path, 'a') as f:
            f.write(f"{new_cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
        annotations_count += 1
        
    # Sao chép và làm phẳng (Flatten) toàn bộ ảnh sang thư mục Working
    # (Loại bỏ các thư mục con batch_X để đồng bộ 1:1 với thư mục labels)
    images_working_dir = os.path.join(os.path.dirname(output_labels_dir), 'images')
    if not os.path.exists(images_working_dir):
        print("\nĐang sao chép và làm phẳng (Flatten) thư mục ảnh... (Quá trình này tốn khoảng 5 giây)")
        os.makedirs(images_working_dir, exist_ok=True)
        import glob
        import shutil
        for img_path in tqdm(glob.glob(os.path.join(images_dir, "**", "*.*"), recursive=True), desc="Copy ảnh"):
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(img_path, os.path.join(images_working_dir, os.path.basename(img_path)))
        
    print(f"\n[HOÀN TẤT] Quá trình tiền xử lý dữ liệu đã kết thúc thành công.")
    print(f"Tổng số phân vùng đã xử lý: {annotations_count} Bounding Boxes.")
    print(f"Đường dẫn lưu trữ dữ liệu đầu ra: {output_labels_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='/kaggle/input/datasets/sohamchaudhari2004/taco-trash-detection-dataset/data/annotations_unofficial.json')
    parser.add_argument('--images_dir', type=str, default='/kaggle/input/datasets/sohamchaudhari2004/taco-trash-detection-dataset/data')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/taco-trash-detection-dataset/labels')
    
    args = parser.parse_args()
    
    convert_coco_json_to_yolo_txt(args.json_path, args.images_dir, args.output_dir)
