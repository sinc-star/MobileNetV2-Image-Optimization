import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """
    预处理图像
    
    Args:
        image_path (str): 图像路径
        target_size (tuple): 目标尺寸，默认(224, 224)
    
    Returns:
        np.ndarray: 预处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整尺寸
    image = cv2.resize(image, target_size)
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 归一化
    image = image / 255.0
    
    # 使用ImageNet均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    
    # 调整维度
    image = np.expand_dims(image, axis=0)
    
    # 转换为float32
    image = image.astype(np.float32)
    
    return image

def preprocess_image_for_inference(image, target_size=(224, 224)):
    """
    为推理预处理图像
    
    Args:
        image (np.ndarray): 输入图像
        target_size (tuple): 目标尺寸，默认(224, 224)
    
    Returns:
        np.ndarray: 预处理后的图像
    """
    # 调整尺寸
    image = cv2.resize(image, target_size)
    
    # BGR转RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 归一化
    image = image / 255.0
    
    # 使用ImageNet均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    
    # 调整维度
    image = np.expand_dims(image, axis=0)
    
    # 转换为float32
    image = image.astype(np.float32)
    
    return image

def normalize_params(params, param_ranges):
    """
    归一化参数
    
    Args:
        params (dict): 原始参数
        param_ranges (dict): 参数范围
    
    Returns:
        dict: 归一化后的参数
    """
    normalized_params = {}
    for key, value in params.items():
        min_val, max_val = param_ranges[key]
        # 归一化到 [-1, 1]
        normalized_params[key] = 2 * (value - min_val) / (max_val - min_val) - 1
    return normalized_params

def denormalize_params(normalized_params, param_ranges):
    """
    反归一化参数
    
    Args:
        normalized_params (dict): 归一化后的参数
        param_ranges (dict): 参数范围
    
    Returns:
        dict: 反归一化后的参数
    """
    denormalized_params = {}
    for key, value in normalized_params.items():
        min_val, max_val = param_ranges[key]
        # 从 [-1, 1] 映射回原始范围
        denormalized_params[key] = (value + 1) * (max_val - min_val) / 2 + min_val
    return denormalized_params