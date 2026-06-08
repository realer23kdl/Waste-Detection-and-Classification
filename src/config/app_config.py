from dataclasses import dataclass, field

class SystemConfig:
    """Cấu hình vận hành hệ thống."""
    log_file_path: str = './logs/pipeline.log'

@dataclass
class DataConfig:
    TACO_TO_7_CLASSES_MAP: str = './datasets/config/mapping_label.json'
    RAW_ANNOTATIONS_PATH: str = './datasets/raw/annotations.json'
    PATH_7_CLASSES: str = './datasets/processed/taco_to_detectwaste_annotations.json'
    PATH_MULTI_TRAIN: str = './datasets/train/multi_train_annotations.json'
    PATH_MULTI_TEST: str = './datasets/test/multi_test_annotations.json'
    PATH_BINARY_TRAIN: str = './datasets/train/binary_train_annotations.json'
    PATH_BINARY_TEST: str = './datasets/test/binary_test_annotations.json'
    
    # Thư mục gốc chứa ảnh (để copy/crop)
    PATH_IMAGES_DIR: str = './datasets/raw/images'
    
    # Đầu ra cho mô hình
    PATH_YOLO_BASE: str = './datasets/yolo_data'
    PATH_CLASSIFIER_TRAIN: str = './datasets/classifier_data/train'

@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
