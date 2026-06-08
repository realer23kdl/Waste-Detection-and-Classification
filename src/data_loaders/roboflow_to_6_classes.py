import os
import yaml
import glob
import argparse

def get_super_id(name):
    """
    Thuật toán AI tự động ánh xạ 59 loại rác chi tiết của Roboflow 
    về 6 phân lớp chuẩn của dự án.
    """
    n = name.lower()
    
    # 0. Thủy tinh (Glass)
    if 'glass' in n: return 0
    
    # 1. Giấy (Paper)
    if 'paper' in n or 'magazine' in n or 'tissue' in n: return 1
    
    # 2. Bìa cứng (Cardboard)
    if 'carton' in n or 'corrugated' in n or 'box' in n or 'toilet tube' in n: return 2
    
    # 4. Kim loại (Metal)
    if 'metal' in n or 'can' in n or 'aluminium' in n or 'aluminum' in n or 'foil' in n or 'aerosol' in n or 'pop tab' in n: return 4
    
    # 3. Nhựa & Xốp (Plastic)
    if 'plastic' in n or 'styrofoam' in n or 'foam' in n or 'polypropylene' in n: return 3
    if 'bag' in n or 'bottle' in n or 'cup' in n or 'lid' in n or 'straw' in n or 'wrapper' in n: return 3
    if 'container' in n or 'tube' in n or 'tub' in n or 'tupperware' in n or 'crisp packet' in n or 'blister' in n or 'six pack' in n: return 3
    
    # 5. Rác không tái chế (Trash)
    return 5

def process_dataset(data_path):
    yaml_path = os.path.join(data_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"❌ Không tìm thấy {yaml_path}")
        return

    print("Đang nạp file data.yaml...")
    with open(yaml_path, 'r', encoding='utf8') as f:
        data_cfg = yaml.safe_load(f)

    old_names = data_cfg.get('names', [])
    id_map = {}
    for i, name in enumerate(old_names):
        id_map[i] = get_super_id(name)
    print(f"Đã quét và lập sơ đồ chuyển đổi cho {len(old_names)} loại rác chi tiết!")

    print("Đang tiến hành gán nhãn lại (Re-labeling) toàn bộ file TXT...")
    label_files = glob.glob(os.path.join(data_path, '**', 'labels', '*.txt'), recursive=True)
    count_boxes = 0

    for txt_file in label_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_id = int(parts[0])
                new_id = id_map.get(old_id, 5) # Mặc định là Trash nếu không có
                new_line = f"{new_id} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)
                count_boxes += 1
                
        with open(txt_file, 'w') as f:
            f.writelines(new_lines)

    print(f"Đã sửa xong nhãn cho {count_boxes} Bounding Box!")

    # Cập nhật lại trái tim của YOLO
    data_cfg['nc'] = 6
    data_cfg['names'] = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    with open(yaml_path, 'w', encoding='utf8') as f:
        yaml.dump(data_cfg, f, default_flow_style=False)

    print("\n🎉 BIẾN ĐỔI DATASET HOÀN TẤT 100%! BẠN CÓ THỂ BẮT ĐẦU HUẤN LUYỆN YOLO.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Đường dẫn tới thư mục Roboflow vừa tải')
    args = parser.parse_args()
    process_dataset(args.data_path)
