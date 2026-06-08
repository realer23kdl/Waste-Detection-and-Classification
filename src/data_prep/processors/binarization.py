import logging
from src.data_prep.processors.base_processor import BaseAnnotationProcessor

logger = logging.getLogger("BinarizationProcessor")

class BinarizationProcessor(BaseAnnotationProcessor):
    def __init__(self, dataset_dict: dict):
        super().__init__(dataset_dict)
        self.binary_category_id = 1
        self.binary_name = 'litter'

    def transform(self):
        logger.info("Bắt đầu tiến trình San phẳng Nhãn (Binarization)...")
        if 'info' not in self.dataset:
            self.dataset['info'] = {}

        self.dataset['info']['description'] = 'detectwaste_binary'
        self.dataset['categories'] = [{
            'id': self.binary_category_id,
            'name': self.binary_name,
            'supercategory': self.binary_name
        }]

        for ann in self.dataset.get('annotations', []):
            ann['category_id'] = self.binary_category_id
            ann.pop('segmentation', None)
