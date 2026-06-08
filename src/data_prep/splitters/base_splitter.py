from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import logging
from src.utils.dataset_utils import DatasetUtils

logger = logging.getLogger("BaseDatasetSplitter")

class BaseDatasetSplitter(ABC):
    def __init__(self, test_size: float = 0.2, random_state: int = 2020):
        self.test_size = test_size
        self.random_state = random_state

    @abstractmethod
    def split(self, dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    def _build_coco_subset(self, original_dataset: Dict[str, Any], subset_images: List[Dict]) -> Dict[str, Any]:
        subset_annotations = DatasetUtils.filter_annotations(
            annotations=original_dataset.get('annotations', []),
            images=subset_images
        )
        subset_dataset = {
            'info': original_dataset.get('info', {}),
            'licenses': original_dataset.get('licenses', []),
            'categories': original_dataset.get('categories', []),
            'images': subset_images,
            'annotations': subset_annotations
        }
        return subset_dataset
