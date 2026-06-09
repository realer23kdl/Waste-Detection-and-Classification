import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Tạo thư mục dummy
os.makedirs("datasets/raw/images", exist_ok=True)
os.makedirs("datasets/config", exist_ok=True)

# 1. Tạo file ảnh dummy
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:] = (255, 255, 255) # Trắng
cv2.imwrite("datasets/raw/images/dummy_img.jpg", img)

# 2. Tạo mapping_label.json
mapping = {
    "Glass": ["glass_bottle", "broken_glass"],
    "Plastic": ["plastic_bottle", "plastic_bag"],
    "Paper": ["paper_cup"]
}
with open("datasets/config/mapping_label.json", "w") as f:
    json.dump(mapping, f)

# 3. Tạo annotations.json dummy
anns = {
    "images": [{"id": 1, "file_name": "dummy_img.jpg", "width": 100, "height": 100}],
    "categories": [
        {"id": 10, "name": "plastic_bottle"},
        {"id": 11, "name": "glass_bottle"}
    ],
    "annotations": [
        {"id": 100, "image_id": 1, "category_id": 10, "bbox": [10, 10, 40, 40]}, # Plastic
        {"id": 101, "image_id": 1, "category_id": 11, "bbox": [50, 50, 20, 20]}  # Glass
    ]
}
with open("datasets/raw/annotations.json", "w") as f:
    json.dump(anns, f)

print("Đã tạo Dummy Data thành công! Sẵn sàng chạy thử main_prep.py...")
