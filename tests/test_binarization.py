import sys
import os
import pytest

# Thêm đường dẫn để import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep.processors.binarization import BinarizationProcessor

def test_binarization_processor():
    # Tạo dummy dataset (Dữ liệu giả lập)
    dummy_data = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [50, 50, 20, 20]}
        ],
        "categories": [
            {"id": 3, "name": "plastic"},
            {"id": 5, "name": "metal"}
        ]
    }

    processor = BinarizationProcessor(dummy_data)
    processor.transform()
    result = processor.get_dataset()

    # Kiểm tra xem danh mục (categories) có bị ép về 1 class không
    assert len(result['categories']) == 1
    assert result['categories'][0]['id'] == 1
    assert result['categories'][0]['name'] == 'litter'

    # Kiểm tra xem tất cả annotation có bị ép về category_id = 1 không
    for ann in result['annotations']:
        assert ann['category_id'] == 1
