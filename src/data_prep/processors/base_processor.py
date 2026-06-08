from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("BaseProcessor")

class BaseAnnotationProcessor(ABC):
    def __init__(self, dataset_dict: dict):
        self.dataset = dataset_dict
        logger.info(f"[{self.__class__.__name__}] Đã khởi tạo processor với {len(self.dataset.get('images', []))} ảnh.")

    def get_dataset(self) -> dict:
        return self.dataset

    @abstractmethod
    def transform(self):
        pass
