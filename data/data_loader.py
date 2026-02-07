import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from data.prepare_data import preprocess_image

class AestheticDataset(Dataset):
    def __init__(self, data_dir, annotations_file=None, transform=None):
        """
        美学数据集
        
        Args:
            data_dir (str): 数据目录
            annotations_file (str): 标注文件路径
            transform (callable): 数据变换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 如果有标注文件，加载标注
        if annotations_file and os.path.exists(annotations_file):
            self.annotations = self._load_annotations(annotations_file)
        else:
            # 如果没有标注文件，使用假数据
            self.annotations = self._generate_fake_annotations()
        
        self.image_files = list(self.annotations.keys())
    
    def _load_annotations(self, annotations_file):
        """
        加载标注
        
        Args:
            annotations_file (str): 标注文件路径
        
        Returns:
            dict: 标注字典
        """
        annotations = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    image_file = parts[0]
                    # 两参数版本：曝光和饱和度
                    exposure = float(parts[1])
                    saturation = float(parts[2])
                    annotations[image_file] = {
                        'exposure': exposure,
                        'saturation': saturation
                    }
        return annotations
    
    def _generate_fake_annotations(self, num_samples=100):
        """
        生成假数据标注
        
        Args:
            num_samples (int): 生成的样本数量
        
        Returns:
            dict: 假数据标注
        """
        annotations = {}
        # 假设数据目录中有一些图像
        image_files = [f for f in os.listdir(self.data_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            # 如果没有图像，生成假的图像文件名
            for i in range(num_samples):
                image_file = f'fake_image_{i}.jpg'
                # 随机生成参数
                exposure = np.random.uniform(-1.0, 1.0)
                saturation = np.random.uniform(0.0, 2.0)
                annotations[image_file] = {
                    'exposure': exposure,
                    'saturation': saturation
                }
        else:
            # 使用实际图像文件
            for image_file in image_files:
                # 随机生成参数
                exposure = np.random.uniform(-1.0, 1.0)
                saturation = np.random.uniform(0.0, 2.0)
                annotations[image_file] = {
                    'exposure': exposure,
                    'saturation': saturation
                }
        
        return annotations
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取数据
        
        Args:
            idx (int): 索引
        
        Returns:
            tuple: (image, params)
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)
        
        # 加载和预处理图像
        if os.path.exists(image_path):
            image = preprocess_image(image_path)
        else:
            # 如果图像不存在，生成假图像
            image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # 获取标注
        annotation = self.annotations.get(image_file, {
            'exposure': 0.0,
            'saturation': 1.0
        })
        
        # 提取参数
        params = torch.tensor([
            annotation['exposure'],
            annotation['saturation']
        ], dtype=torch.float32)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return torch.from_numpy(image).squeeze(0), params

def create_data_loader(data_dir, annotations_file=None, batch_size=32, shuffle=True):
    """
    创建数据加载器
    
    Args:
        data_dir (str): 数据目录
        annotations_file (str): 标注文件路径
        batch_size (int): 批量大小
        shuffle (bool): 是否打乱
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = AestheticDataset(
        data_dir=data_dir,
        annotations_file=annotations_file
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Windows系统建议设置为0
    )
    
    return data_loader