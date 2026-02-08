import torch
from torch.utils.data import Dataset
import sqlite3
import requests
import cv2
import numpy as np
from collections import OrderedDict

class UnsplashDynamicDataset(Dataset):
    def __init__(self, db_path, transform=None, cache_size=100, split='train'):
        """
        动态加载Unsplash数据集
        
        Args:
            db_path (str): 数据库文件路径
            transform (callable): 数据变换
            cache_size (int): 缓存大小
            split (str): 数据集划分，可选值: 'train', 'val', 'test'
        """
        self.db_path = db_path
        self.transform = transform
        self.cache_size = cache_size
        self.split = split
        self.image_urls = self._load_image_urls()
        self.cache = OrderedDict()  # LRU缓存
    
    def _load_image_urls(self):
        """
        从数据库加载图像URL
        
        Returns:
            list: 包含photo_id和image_url的列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT photo_id, photo_image_url FROM photos")
            urls = cursor.fetchall()
            conn.close()
            
            # 根据split参数划分数据集
            total_count = len(urls)
            train_end = int(total_count * 0.6)  # 训练集：前60%
            val_end = train_end + int(total_count * 0.2)  # 验证集：中间20%
            # 测试集：后20%
            
            if self.split == 'train':
                return urls[:train_end]
            elif self.split == 'val':
                return urls[train_end:val_end]
            elif self.split == 'test':
                return urls[val_end:]
            else:
                return urls
        except Exception as e:
            print(f"加载图像URL时出错: {e}")
            return []
    
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 数据集大小
        """
        return len(self.image_urls)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 索引
        
        Returns:
            tuple: (image, params)
        """
        photo_id, image_url = self.image_urls[idx]
        
        # 检查缓存
        if photo_id in self.cache:
            # 更新缓存顺序（LRU）
            image = self.cache.pop(photo_id)
            self.cache[photo_id] = image
        else:
            # 下载图像
            image = self._download_image(image_url)
            
            # 缓存图像
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # 移除最旧的项
            self.cache[photo_id] = image
        
        # 预处理图像
        image = self._preprocess_image(image)
        
        # 生成标注（两参数：曝光和饱和度）
        params = self._generate_params(image)
        
        return image, params
    
    def _download_image(self, image_url):
        """
        下载图像
        
        Args:
            image_url (str): 图像URL
        
        Returns:
            np.ndarray: 图像数组
        """
        try:
            # 确保URL格式正确
            if not image_url.startswith(('http://', 'https://')):
                # 尝试构建正确的Unsplash URL
                if image_url.startswith('images.unsplash.com'):
                    image_url = f"https://{image_url}"
                else:
                    # 对于其他格式，返回随机图像
                    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 添加尺寸参数，减小下载大小
            if '?' in image_url:
                resized_url = f"{image_url}&w=224&h=224"
            else:
                resized_url = f"{image_url}?w=224&h=224"
            
            response = requests.get(resized_url, timeout=10)
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is not None:
                    return image
        except Exception as e:
            print(f"Error downloading image: {e}")
        
        # 如果下载失败，返回随机图像
        return np.ones((224, 224, 3), dtype=np.uint8) * 128  # 灰色图像 占位符
    
    def _preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image (np.ndarray): 原始图像
        
        Returns:
            torch.Tensor: 预处理后的图像
        """
        # 调整尺寸
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化
        image = image / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # 转换为张量
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def _generate_params(self, image):
        """
        生成调色参数
        
        Args:
            image (torch.Tensor): 预处理后的图像
        
        Returns:
            torch.Tensor: 调色参数 [exposure, contrast, saturation, highlight, shadow]
        """
        # 将张量转换为numpy数组
        img_np = image.permute(1, 2, 0).numpy()
        
        # 反归一化
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # 转换为HSV
        img_hsv = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # 计算图像统计特征
        brightness = np.mean(img_hsv[:, :, 2]) / 255.0
        saturation = np.mean(img_hsv[:, :, 1]) / 255.0
        
        # 计算对比度（基于标准差）
        img_lab = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        contrast = np.std(img_lab[:, :, 0]) / 255.0
        
        # 计算高光和阴影（基于亮度分布）
        l_channel = img_lab[:, :, 0].astype(np.float32) / 255.0
        highlight = np.mean(l_channel[l_channel > 0.6])  # 亮部平均值
        shadow = 1.0 - np.mean(l_channel[l_channel < 0.4])  # 暗部平均值（反转）
        
        # 生成参数
        
        # 曝光调整：使亮度更加适中
        if brightness < 0.3:
            exposure = 0.6  # 大幅提亮
        elif brightness > 0.7:
            exposure = -0.4  # 大幅变暗
        else:
            exposure = 0.0  # 保持不变
        
        # 对比度调整
        if contrast < 0.15:
            contrast_param = 1.8  # 增加对比度
        elif contrast > 0.25:
            contrast_param = 0.7  # 降低对比度
        else:
            contrast_param = 1.0  # 保持不变
        
        # 饱和度调整：使色彩更加自然
        if saturation < 0.3:
            saturation_param = 1.6  # 增加饱和度
        elif saturation > 0.7:
            saturation_param = 0.9  # 降低饱和度
        else:
            saturation_param = 1.0  # 保持不变
        
        # 高光调整
        if highlight < 0.7:
            highlight_param = 0.8  # 增强高光
        elif highlight > 0.85:
            highlight_param = 0.3  # 降低高光
        else:
            highlight_param = 0.5  # 保持不变
        
        # 阴影调整
        if shadow < 0.3:
            shadow_param = 0.8  # 增强阴影
        elif shadow > 0.5:
            shadow_param = 0.4  # 降低阴影
        else:
            shadow_param = 0.5  # 保持不变
        
        # 添加一些随机性
        exposure += np.random.normal(0, 0.08)
        contrast_param += np.random.normal(0, 0.08)
        saturation_param += np.random.normal(0, 0.08)
        highlight_param += np.random.normal(0, 0.08)
        shadow_param += np.random.normal(0, 0.08)
        
        # 限制范围
        exposure = np.clip(exposure, -1.0, 1.0)
        contrast_param = np.clip(contrast_param, 0.5, 2.0)
        saturation_param = np.clip(saturation_param, 0.0, 2.0)
        highlight_param = np.clip(highlight_param, 0.0, 1.0)
        shadow_param = np.clip(shadow_param, 0.0, 1.0)
        
        return torch.tensor([exposure, contrast_param, saturation_param, highlight_param, shadow_param], dtype=torch.float32)

if __name__ == '__main__':
    # 测试数据集
    db_path = 'data/unsplash/db/unsplash.db'
    dataset = UnsplashDynamicDataset(db_path)
    
    # 测试获取样本
    for i in range(5):
        image, params = dataset[i]
        print(f"Sample {i}: Image shape: {image.shape}, Params: {params}")