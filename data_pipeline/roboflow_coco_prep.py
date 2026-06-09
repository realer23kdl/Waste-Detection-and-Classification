import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.io_utils import IOUtils
from src.data_prep.processors.category_mapping import CategoryMappingProcessor
from src.data_prep.processors.binarization import BinarizationProcessor
from src.data_prep.processors.image_cropper import ImageCropProcessor
from src.data_prep.processors.yolo_converter import YoloFormatProcessor
from src.config.app_config import AppConfig

def process_roboflow_coco(dataset_dir: str, mapping_label_path: str = None):
    """
    Xử lý trực tiếp bộ dữ liệu COCO tải từ Roboflow (chia sẵn train, valid, test với _annotations.coco.json)
    """
    splits = ['train', 'valid', 'test']
    
    mapping_dict = None
    if mapping_label_path and os.path.exists(mapping_label_path):
        mapping_dict = IOUtils.load_json(mapping_label_path)
        print("Đã nạp file mapping_label.json")
    
    config = AppConfig()
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        json_path = os.path.join(split_dir, '_annotations.coco.json')
        
        if not os.path.exists(json_path):
            print(f"Bỏ qua tập {split}, không tìm thấy: {json_path}")
            continue
            
        print(f"\n=== ĐANG XỬ LÝ TẬP {split.upper()} ===")
        # 1. Nạp JSON
        coco_data = IOUtils.load_json(json_path)
        
        # 2. Gom nhãn (Nếu có truyền file mapping)
        if mapping_dict:
            print("Đang tiến hành gom nhãn (Mapping)...")
            mapper = CategoryMappingProcessor(dataset_dict=coco_data, mapping_dict=mapping_dict)
            mapper.transform()
            coco_data = mapper.get_dataset()
            
        # 3. Cắt ảnh rác cho Classifier
        if split in ['train', 'valid']:
            cls_out = config.PATH_CLASSIFIER_TRAIN
        else:
            cls_out = config.PATH_CLASSIFIER_TEST
            
        print("Đang sử dụng ImageCropper để cắt rác cho Classifier...")
        crop_processor = ImageCropProcessor(
            dataset_dict=coco_data,
            images_source_dir=split_dir,  # Trên Roboflow COCO, ảnh nằm chung thư mục với file JSON
            output_dir=cls_out
        )
        crop_processor.transform()
        
        # 4. Gom tất cả về nhãn 'litter' cho YOLO (Binarization)
        print("Đang Binarize nhãn cho mạng YOLO...")
        binarizer = BinarizationProcessor(coco_data)
        binarizer.transform()
        binary_data = binarizer.get_dataset()
        
        # 5. Xuất ra định dạng YOLO (Tự động copy ảnh và tạo file .txt)
        print("Đang chuyển đổi cấu trúc sang YOLO TXT...")
        # Đổi tên thư mục valid thành val cho chuẩn YOLO
        yolo_out = os.path.join(config.PATH_YOLO_BASE, 'val' if split == 'valid' else split)
        yolo_processor = YoloFormatProcessor(
            dataset_dict=binary_data,
            images_source_dir=split_dir,
            output_dir=yolo_out
        )
        yolo_processor.transform()
        
    # 6. Sinh file data.yaml cho YOLO
    yaml_path = os.path.join(config.PATH_YOLO_BASE, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['litter']\n")
        
    print("\n[HOÀN TẤT] Đã chuyển đổi thành công bộ data COCO của Roboflow!")
    print(f"Dữ liệu YOLO: {config.PATH_YOLO_BASE} (Đã tự động tạo {yaml_path})")
    print(f"Dữ liệu Classifier: {config.PATH_CLASSIFIER_TRAIN}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help="Thư mục gốc tải từ Roboflow (chứa train/valid/test)")
    parser.add_argument('--mapping_label', type=str, default=None, help="File mapping_label.json (Chỉ truyền vào nếu cần gom nhãn)")
    args = parser.parse_args()
    
    process_roboflow_coco(args.dataset_dir, args.mapping_label)
