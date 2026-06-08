import numpy as np
import logging
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import Dict, Any, Tuple, List
from src.data_prep.splitters.base_splitter import BaseDatasetSplitter

logger = logging.getLogger("MultiLabelSplitter")

class MultiLabelSplitter(BaseDatasetSplitter):
    def _build_dense_feature_matrix(self, images: List[Dict], annotations: List[Dict]) -> np.ndarray:
        cat_to_idx = {cat['id']: i for i, cat in enumerate(self.categories)}
        matrix = np.zeros((len(images), len(self.categories)))
        img_id_to_idx = {img['id']: i for i, img in enumerate(images)}

        for ann in annotations:
            if ann['image_id'] in img_id_to_idx:
                row = img_id_to_idx[ann['image_id']]
                col = cat_to_idx[ann['category_id']]
                matrix[row, col] += 1
        return matrix

    def split(self, dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = dataset.get('images', [])
        annotations = dataset.get('annotations', [])
        self.categories = dataset.get('categories', [])

        feature_matrix = self._build_dense_feature_matrix(images, annotations)
        strat_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        
        train_index, test_index = next(strat_split.split(images, feature_matrix))
        train_images = [images[i] for i in train_index]
        test_images = [images[i] for i in test_index]

        return self._build_coco_subset(dataset, train_images), self._build_coco_subset(dataset, test_images)
