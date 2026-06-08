import os
from PIL import Image
from torch.utils.data import Dataset

class TrashDataset(Dataset):
    """
    Kế thừa torch.utils.data.Dataset theo đúng chuẩn PyTorch và yêu cầu Rubric.
    Đọc dữ liệu ảnh đã được phân loại vào các thư mục con (ví dụ: Glass, Paper, Plastic...).
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Đường dẫn tới thư mục chứa các thư mục con (class).
            transform (callable, optional): Các phép biến đổi ảnh (Resize, Normalize...).
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        # Ánh xạ tên class sang ID (0, 1, 2...)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Quét toàn bộ ảnh trong các thư mục con
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        """Trả về tổng số lượng ảnh"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Lấy ra 1 mẫu dữ liệu (ảnh + nhãn) tại vị trí idx.
        Hàm này sẽ được DataLoader gọi liên tục khi train.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Đọc ảnh bằng PIL để tương thích tốt với torchvision.transforms
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_class_weights(self):
        """
        Tự động tính toán tỷ lệ mất cân bằng dữ liệu để truyền vào Weighted Loss.
        """
        import numpy as np
        class_counts = np.bincount(self.labels, minlength=len(self.classes))
        total_samples = len(self.labels)
        
        # Tính trọng số: Class nào càng ít ảnh thì trọng số càng cao
        weights = total_samples / (len(self.classes) * class_counts)
        return weights.tolist()
