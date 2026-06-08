import os
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger("IOUtils")

class IOUtils:
    @staticmethod
    def save_coco_json(dest_path: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Chuẩn bị lưu COCO JSON tại: {dest_path}")
        data_dict = {
            'info': dataset.get('info', {}),
            'licenses': dataset.get('licenses', []),
            'images': dataset.get('images', []),
            'annotations': dataset.get('annotations', []),
            'categories': dataset.get('categories', [])
        }
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, sort_keys=True)
            logger.info(f"Lưu file thành công! {len(data_dict['images'])} ảnh, {len(data_dict['annotations'])} nhãn.")
        except Exception as e:
            logger.exception(f"Lỗi lưu file: {e}")
            raise e
        return data_dict

    @staticmethod
    def load_json(source_path: str) -> Dict[str, Any]:
        logger.info(f"Đang nạp dữ liệu từ: {source_path}...")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File {source_path} không tồn tại!")
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            return data_dict
        except Exception as e:
            logger.exception(f"Lỗi nạp file JSON: {e}")
            raise e
