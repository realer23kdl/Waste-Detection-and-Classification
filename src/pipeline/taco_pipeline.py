import logging
import os
from src.core.base_pipeline import BasePipeline
from src.utils.io_utils import IOUtils
from src.data_prep.processors.category_mapping import CategoryMappingProcessor
from src.data_prep.processors.binarization import BinarizationProcessor
from src.data_prep.splitters.multi_label_splitter import MultiLabelSplitter
from src.data_prep.processors.yolo_converter import YoloFormatProcessor
from src.data_prep.processors.image_cropper import ImageCropProcessor
from src.config.app_config import AppConfig

logger = logging.getLogger("TACO_Pipeline")

class WastePreprocessingPipeline(BasePipeline):
    def __init__(self, config):
        self.config = config

    def execute(self, test_size=0.2, random_state=2020):
        logger.info("--- GIAI ĐOẠN TIỀN XỬ LÝ TĨNH (XỬ LÝ RAW ANNS) ---")
        
        # Bước 1
        logger.info("Bước 1: Nạp JSON...")
        raw_dataset = IOUtils.load_json(self.config.RAW_ANNOTATIONS_PATH)
        mapping_dict = IOUtils.load_json(self.config.TACO_TO_7_CLASSES_MAP)
        
        # Bước 2
        logger.info("Bước 2: Gom nhãn 60 -> 7...")
        mapper = CategoryMappingProcessor(dataset_dict=raw_dataset, mapping_dict=mapping_dict)
        mapper.transform()
        dataset_7_classes = mapper.get_dataset()
        IOUtils.save_coco_json(self.config.PATH_7_CLASSES, dataset_7_classes)
        
        # Bước 3
        logger.info("Bước 3: Chia Train/Test...")
        splitter = MultiLabelSplitter(test_size=test_size, random_state=random_state)
        train_7_classes, test_7_classes = splitter.split(dataset_7_classes)
        IOUtils.save_coco_json(self.config.PATH_MULTI_TRAIN, train_7_classes)
        IOUtils.save_coco_json(self.config.PATH_MULTI_TEST, test_7_classes)
        
        # --- BƯỚC 4: CẮT ẢNH CHO CLASSIFIER ---
        # Chỉ cắt ảnh ở tập Train để làm dữ liệu huấn luyện Classifier
        logger.info("Bước 4: Cắt vùng ảnh rác (ROI) để chuẩn bị cho Huấn luyện Classifier...")
        crop_processor = ImageCropProcessor(
            dataset_dict=train_7_classes, 
            images_source_dir=self.config.PATH_IMAGES_DIR,
            output_dir=self.config.PATH_CLASSIFIER_TRAIN
        )
        crop_processor.transform()

        # --- BƯỚC 5: BINARIZATION (DÀNH RIÊNG CHO YOLO DETECTOR) ---
        logger.info("Bước 5: Binarization (Chuyển tất cả về 1 class 'litter' cho YOLO)...")
        binarizer_train = BinarizationProcessor(train_7_classes)
        binarizer_train.transform()
        train_binary = binarizer_train.get_dataset()
        
        binarizer_test = BinarizationProcessor(test_7_classes)
        binarizer_test.transform()
        test_binary = binarizer_test.get_dataset()

        # --- BƯỚC 6: SINH FILE YOLO TXT ---
        logger.info("Bước 6: Sinh file định dạng YOLO TXT...")
        yolo_train_processor = YoloFormatProcessor(
            dataset_dict=train_binary,
            images_source_dir=self.config.PATH_IMAGES_DIR,
            output_dir=os.path.join(self.config.PATH_YOLO_BASE, "train")
        )
        yolo_train_processor.transform()

        yolo_test_processor = YoloFormatProcessor(
            dataset_dict=test_binary,
            images_source_dir=self.config.PATH_IMAGES_DIR,
            output_dir=os.path.join(self.config.PATH_YOLO_BASE, "val")
        )
        yolo_test_processor.transform()

        # --- BƯỚC 7: LƯU TRỮ JSON DỰ PHÒNG ---
        logger.info("Bước 7: Lưu dữ liệu (JSON format dự phòng)...")
        IOUtils.save_coco_json(self.config.PATH_BINARY_TRAIN, train_binary)
        IOUtils.save_coco_json(self.config.PATH_BINARY_TEST, test_binary)

        logger.info("--- PIPELINE KẾT THÚC THÀNH CÔNG ---")
