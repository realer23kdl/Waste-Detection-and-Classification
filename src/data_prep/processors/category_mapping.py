import logging
from src.data_prep.processors.base_processor import BaseAnnotationProcessor

logger = logging.getLogger("CategoryMappingProcessor")

class CategoryMappingProcessor(BaseAnnotationProcessor):
    def __init__(self, dataset_dict: dict, mapping_dict: dict):
        super().__init__(dataset_dict)
        self.fast_lookup_dict = self._build_fast_lookup(mapping_dict)

    def _build_fast_lookup(self, config: dict) -> dict:
        lookup = {}
        for new_label, old_labels_list in config.items():
            for old_label in old_labels_list:
                lookup[old_label] = new_label
        return lookup

    def transform(self):
        logger.info("CategoryMappingProcessor: Bắt đầu quá trình đồng bộ hóa danh mục...")
        new_categories = []
        new_group_to_new_id = {}
        old_id_to_new_id = {}
        current_new_id = 1

        for old_cat in self.dataset.get('categories', []):
            old_name = old_cat['name']
            new_name = self.fast_lookup_dict.get(old_name, 'unknown')

            if new_name not in new_group_to_new_id:
                new_group_to_new_id[new_name] = current_new_id
                new_categories.append({"id": current_new_id, "name": new_name, "supercategory": new_name})
                current_new_id += 1

            old_id_to_new_id[old_cat['id']] = new_group_to_new_id[new_name]

        old_count = len(self.dataset.get('categories', []))
        self.dataset['categories'] = new_categories

        for ann in self.dataset.get('annotations', []):
            old_cat_id = ann['category_id']
            ann['category_id'] = old_id_to_new_id.get(old_cat_id, old_cat_id)
            ann.pop('segmentation', None)

        logger.info(f"CategoryMappingProcessor: Hoàn tất! Thu gọn {old_count} loại về {len(new_categories)} loại nhãn.")
