import os
import matplotlib.pyplot as plt
from collections import Counter

def run_eda(labels_dir, class_names):
    """
    Quét toàn bộ thư mục nhãn YOLO và đếm số lượng từng loại rác.
    Xuất ra biểu đồ phân bố (Bar chart) để dán vào báo cáo.
    """
    print(f"Đang quét thư mục nhãn: {labels_dir} ...")
    txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    class_counts = Counter()
    
    for txt in txt_files:
        with open(os.path.join(labels_dir, txt), 'r') as file:
            lines = file.readlines()
            for line in lines:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
                
    # Hiển thị số liệu
    print("\n[KẾT QUẢ PHÂN TÍCH EDA]")
    for class_id, count in class_counts.items():
        print(f"- {class_names[class_id]}: {count} mẫu")
        
    # Vẽ biểu đồ
    ids = list(class_counts.keys())
    counts = list(class_counts.values())
    labels = [class_names[i] for i in ids]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])
    plt.title('Phân bố Số lượng Rác thải (Class Distribution)', fontsize=14)
    plt.xlabel('Loại Rác', fontsize=12)
    plt.ylabel('Số lượng (Mẫu)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = "eda_class_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nĐã lưu biểu đồ phân tích EDA tại: {save_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="datasets/train/labels", help="Đường dẫn tới thư mục labels")
    args = parser.parse_args()
    
    CLASS_NAMES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]
    
    if os.path.exists(args.data_path):
        run_eda(args.data_path, CLASS_NAMES)
    else:
        print(f"Không tìm thấy thư mục: {args.data_path}")
