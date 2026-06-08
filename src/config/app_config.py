from dataclasses import dataclass

class SystemConfig:
    """Cấu hình vận hành hệ thống."""
    log_file_path: str = './logs/pipeline.log'

@dataclass(frozen=True)
class DataConfig:
    TACO_TO_7_CLASSES_MAP: str = './datasets/config/mapping_label.json'
    RAW_ANNOTATIONS_PATH: str = './datasets/raw/annotations.json'
    PATH_7_CLASSES: str = './datasets/processed/taco_to_detectwaste_annotations.json'
    PATH_MULTI_TRAIN: str = './datasets/train/multi_train_annotations.json'
    PATH_MULTI_TEST: str = './datasets/test/multi_test_annotations.json'
    PATH_BINARY_TRAIN: str = './datasets/train/binary_train_annotations.json'
    PATH_BINARY_TEST: str = './datasets/test/binary_test_annotations.json'

@dataclass
class AppConfig:
    system: SystemConfig = SystemConfig()
    data: DataConfig = DataConfig()
