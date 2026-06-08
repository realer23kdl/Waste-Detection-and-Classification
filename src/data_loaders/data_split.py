import os
import random
import shutil
from tqdm import tqdm

def split_dataset(source_img_dir, source_lbl_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Chia ngẫu nhiên tập dữ liệu thành Train (80%), Val (10%), Test (10%).
    """
    print("Đang tiến hành xáo trộn và chia tách Dữ liệu (Train/Val/Test)...")
    
    # Tạo các thư mục đích
    splits = ['train', 'val', 'test']
    for s in splits:
        os.makedirs(os.path.join(output_dir, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', s), exist_ok=True)
        
    # Lấy danh sách file ảnh
    images = [f for f in os.listdir(source_img_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images) # Xáo trộn ngẫu nhiên
    
    total = len(images)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    split_dict = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    # Copy file
    for split_name, img_list in split_dict.items():
        for img_name in tqdm(img_list, desc=f"Đang copy vào tập {split_name.upper()}"):
            # Đường dẫn gốc
            src_img = os.path.join(source_img_dir, img_name)
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            src_txt = os.path.join(source_lbl_dir, txt_name)
            
            # Nếu có ảnh mà không có nhãn thì bỏ qua
            if not os.path.exists(src_txt):
                continue
                
            # Copy ảnh
            shutil.copy(src_img, os.path.join(output_dir, 'images', split_name, img_name))
            # Copy nhãn
            shutil.copy(src_txt, os.path.join(output_dir, 'labels', split_name, txt_name))
            
    print("\n[HOÀN THÀNH] Việc chia tập dữ liệu đã xong. Bạn đã sẵn sàng chạy Train YOLO!")

if __name__ == "__main__":
    random.seed(42) # Cố định seed
    SRC_IMG = "datasets/raw_images"
    SRC_LBL = "datasets/raw_labels"
    OUT_DIR = "datasets/yolo_format"
    
    if os.path.exists(SRC_IMG) and os.path.exists(SRC_LBL):
        split_dataset(SRC_IMG, SRC_LBL, OUT_DIR)
    else:
        print("Tạo sẵn file Data Split. Chạy khi đã tải Data về.")
