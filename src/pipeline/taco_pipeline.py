import logging
from src.core.base_pipeline import BasePipeline
from src.utils.io_utils import IOUtils
from src.data_prep.processors.category_mapping import CategoryMappingProcessor
from src.data_prep.processors.binarization import BinarizationProcessor
from src.data_prep.splitters.multi_label_splitter import MultiLabelSplitter

logger = logging.getLogger("TACO_Pipeline")

class WastePreprocessingPipeline(BasePipeline):
    def __init__(self, config):
        self.config = config

    def execute(self, test_size=0.2, random_state=2020):
        logger.info("--- GIAI ĐOẠN TIỀN XỬ LÝ TĨNH (XỬ LÝ RAW ANNS) ---")
        
        # Bước 1
        raw_dataset = IOUtils.load_json(self.config.RAW_ANNOTATIONS_PATH)
        mapping_dict = IOUtils.load_json(self.config.TACO_TO_7_CLASSES_MAP)
        
        # Bước 2
        mapper = CategoryMappingProcessor(dataset_dict=raw_dataset, mapping_dict=mapping_dict)
        mapper.transform()
        dataset_7_classes = mapper.get_dataset()
        IOUtils.save_coco_json(self.config.PATH_7_CLASSES, dataset_7_classes)
        
        # Bước 3
        splitter = MultiLabelSplitter(test_size=test_size, random_state=random_state)
        train_7_classes, test_7_classes = splitter.split(dataset_7_classes)
        IOUtils.save_coco_json(self.config.PATH_MULTI_TRAIN, train_7_classes)
        IOUtils.save_coco_json(self.config.PATH_MULTI_TEST, test_7_classes)
        
        # Bước 4
        for tag, ds, path in [('TRAIN', train_7_classes, self.config.PATH_BINARY_TRAIN), ('TEST', test_7_classes, self.config.PATH_BINARY_TEST)]:
            processor = BinarizationProcessor(dataset_dict=ds)
            processor.transform()
            IOUtils.save_coco_json(path, processor.get_dataset())

        logger.info("--- PIPELINE KẾT THÚC THÀNH CÔNG ---")
