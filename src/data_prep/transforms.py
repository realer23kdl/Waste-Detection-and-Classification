import torch
import cv2
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
      pass

class Compose(BaseTransform):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(BaseTransform):
    def __call__(self, image, target):
        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
        if isinstance(image, torch.ByteTensor):
            image = image.float().div(255)
        return image, target

class Resize(BaseTransform):
    def __init__(self, size=(640, 640)):
        self.size = size
    def __call__(self, image, target):
        h, w = image.shape[:2]
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if target is not None and 'annotations' in target:
            scale_x = self.size[0] / w
            scale_y = self.size[1] / h
            for ann in target['annotations']:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    ann['bbox'] = [bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y]
        return image, target

class Normalize(BaseTransform):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    def __call__(self, image, target):
        image = (image - self.mean) / self.std
        return image, target

class RandomHorizontalFlip(BaseTransform):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, target):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            h, w = image.shape[:2]
            if target is not None and 'annotations' in target:
                for ann in target['annotations']:
                    if 'bbox' in ann:
                        x_min, y_min, width, height = ann['bbox']
                        ann['bbox'] = [w - x_min - width, y_min, width, height]
        return image, target

class RandomColorJitter(BaseTransform):
    def __init__(self, brightness=0.2, saturation=0.2):
        self.brightness = brightness
        self.saturation = saturation
    def __call__(self, image, target):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        s_ratio = random.uniform(1 - self.saturation, 1 + self.saturation)
        img_hsv[:, :, 1] *= s_ratio
        v_ratio = random.uniform(1 - self.brightness, 1 + self.brightness)
        img_hsv[:, :, 2] *= v_ratio
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(img_hsv, cv2.HSV2BGR)
        return image, target
