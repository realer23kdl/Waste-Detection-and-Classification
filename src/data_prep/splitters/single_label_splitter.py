import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Any, Tuple, List
import logging
from src.data_prep.splitters.base_splitter import BaseDatasetSplitter

logger = logging.getLogger("SingleLabelSplitter")

class SingleLabelSplitter(BaseDatasetSplitter):
    def _build_dominant_label_array(self, images: List[Dict], annotations: List[Dict]) -> np.ndarray:
        categories_per_image = defaultdict(Counter)
        for ann in annotations:
            categories_per_image[ann['image_id']][ann['category_id']] += 1

        max_category = []
        for im in images:
            im_id = im['id']
            if im_id in categories_per_image:
                dominant_cat = categories_per_image[im_id].most_common(1)[0][0]
                max_category.append(dominant_cat)
            else:
                max_category.append(0)
        return np.array(max_category)

    def split(self, dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = dataset.get('images', [])
        annotations = dataset.get('annotations', [])
        dominant_labels = self._build_dominant_label_array(images, annotations)
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_index, test_index = next(strat_split.split(images, dominant_labels))

        train_images = [images[i] for i in train_index]
        test_images = [images[i] for i in test_index]

        return self._build_coco_subset(dataset, train_images), self._build_coco_subset(dataset, test_images)
