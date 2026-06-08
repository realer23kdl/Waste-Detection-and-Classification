import logging
from typing import List, Dict, Any

logger = logging.getLogger("DatasetUtils")

class DatasetUtils:
    @staticmethod
    def filter_annotations(annotations: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Bắt đầu lọc nhãn cho {len(images)} bức ảnh...")
        valid_image_ids = {int(im['id']) for im in images}
        filtered_anns = [ann for ann in annotations if int(ann['image_id']) in valid_image_ids]
        drop_rate = len(annotations) - len(filtered_anns)
        logger.info(f"Hoàn tất lọc: Giữ lại {len(filtered_anns)} / {len(annotations)} nhãn. Đã loại bỏ {drop_rate} nhãn rác.")
        return filtered_anns

    @staticmethod
    def concatenate_datasets(list_of_datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not list_of_datasets:
            logger.warning("Danh sách dataset rỗng.")
            return {}

        logger.info(f"Tiến hành gộp {len(list_of_datasets)} bộ dữ liệu...")
        last_im_id = 1
        last_ann_id = 1

        concat_dataset = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}

        for index, dataset in enumerate(list_of_datasets):
            if index == 0:
                concat_dataset['info'] = dataset.get('info', {})
                concat_dataset['categories'] = dataset.get('categories', [])

            img_id_mapping = {}
            for im in dataset.get('images', []):
                old_id = im['id']
                img_id_mapping[old_id] = last_im_id
                im['id'] = last_im_id
                last_im_id += 1

            for ann in dataset.get('annotations', []):
                ann['image_id'] = img_id_mapping.get(ann['image_id'], ann['image_id'])
                ann['id'] = last_ann_id
                last_ann_id += 1

            concat_dataset['images'].extend(dataset.get('images', []))
            concat_dataset['annotations'].extend(dataset.get('annotations', []))

        logger.info(f"Gộp thành công! Kết quả: {len(concat_dataset['images'])} ảnh, {len(concat_dataset['annotations'])} nhãn.")
        return concat_dataset
