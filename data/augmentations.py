import torch
import numpy as np
import cv2

class AestheticAugmentations:
    def __init__(self, 
                 horizontal_flip_prob=0.5, 
                 vertical_flip_prob=0.0, 
                 rotate_range=10, 
                 brightness_range=0.2, 
                 contrast_range=0.2,
                 saturation_range=0.2):
        """
        美学数据增强
        
        Args:
            horizontal_flip_prob (float): 水平翻转概率
            vertical_flip_prob (float): 垂直翻转概率
            rotate_range (float): 旋转角度范围
            brightness_range (float): 亮度调整范围
            contrast_range (float): 对比度调整范围
            saturation_range (float): 饱和度调整范围
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotate_range = rotate_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
    
    def __call__(self, image, params):
        """
        应用数据增强
        
        Args:
            image (np.ndarray): 输入图像
            params (torch.Tensor): 调色参数
        
        Returns:
            tuple: (增强后的图像, 参数)
        """
        # 确保图像是numpy数组
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # 移除批次维度
        if image.ndim == 4:
            image = image[0]
        
        # 调整通道顺序 (C, H, W) -> (H, W, C)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        # 反转归一化，恢复到0-255范围
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std + mean) * 255
        image = image.astype(np.uint8)
        
        # 水平翻转
        if np.random.random() < self.horizontal_flip_prob:
            image = cv2.flip(image, 1)
        
        # 垂直翻转
        if np.random.random() < self.vertical_flip_prob:
            image = cv2.flip(image, 0)
        
        # 旋转
        if self.rotate_range > 0:
            angle = np.random.uniform(-self.rotate_range, self.rotate_range)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # 亮度调整
        if self.brightness_range > 0:
            brightness_factor = np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        
        # 对比度调整
        if self.contrast_range > 0:
            contrast_factor = np.random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        
        # 饱和度调整
        if self.saturation_range > 0:
            saturation_factor = np.random.uniform(1 - self.saturation_range, 1 + self.saturation_range)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 重新归一化
        image = image / 255.0
        image = (image - mean) / std
        
        # 调整通道顺序 (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # 转换回torch.Tensor
        image = torch.from_numpy(image.astype(np.float32))
        
        return image, params

def get_default_augmentations():
    """
    获取默认的数据增强
    
    Returns:
        AestheticAugmentations: 默认数据增强
    """
    return AestheticAugmentations(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.0,
        rotate_range=10,
        brightness_range=0.2,
        contrast_range=0.2,
        saturation_range=0.2
    )