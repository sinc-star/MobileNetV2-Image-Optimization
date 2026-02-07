import os
import sys
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import cv2
import numpy as np
from data.prepare_data import preprocess_image_for_inference
from training.config import get_inference_config

class TFLiteInference:
    def __init__(self, model_path):
        """
        TensorFlow Lite推理
        
        Args:
            model_path (str): TensorFlow Lite模型路径
        """
        # 加载模型
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入输出张量
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f'Loaded TFLite model from {model_path}')
        print(f'Input shape: {self.input_details[0]["shape"]}')
        print(f'Output shape: {self.output_details[0]["shape"]}')
    
    def predict(self, image):
        """
        预测调色参数
        
        Args:
            image (np.ndarray): 输入图像
        
        Returns:
            np.ndarray: 预测的参数
        """
        # 预处理图像
        preprocessed_image = preprocess_image_for_inference(image)
        
        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
        
        # 执行推理
        self.interpreter.invoke()
        
        # 获取输出
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output[0]

def apply_color_adjustments(image, params, param_ranges):
    """
    应用调色参数
    
    Args:
        image (np.ndarray): 原始图像
        params (np.ndarray): 调色参数 [exposure, saturation]
        param_ranges (dict): 参数范围
    
    Returns:
        np.ndarray: 调色后的图像
    """
    # 复制图像
    adjusted_image = image.copy()
    
    # 获取参数
    exposure = params[0]
    saturation = params[1]
    
    # 应用曝光调整
    alpha = 1.0 + exposure
    beta = 0
    adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=alpha, beta=beta)
    
    # 应用饱和度调整
    hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

def main():
    """
    主函数
    """
    # 获取配置
    config = get_inference_config()
    
    # 创建推理器
    # 注意：这里使用假的模型路径，实际使用时需要替换为真实的模型路径
    tflite_model_path = config['tflite_model_path']
    
    # 如果模型文件不存在，创建一个假模型
    if not os.path.exists(tflite_model_path):
        # 创建模型目录
        os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)
        
        # 创建一个简单的假模型
        # 这里只是创建一个空文件，实际使用时需要运行转换脚本
        with open(tflite_model_path, 'wb') as f:
            f.write(b'')
        print(f'Created dummy TFLite model at {tflite_model_path}')
    
    try:
        # 创建推理器
        inference = TFLiteInference(tflite_model_path)
        
        # 测试推理
        # 生成一个假图像
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 预测参数
        params = inference.predict(test_image)
        print(f'Predicted params: {params}')
        
        # 应用参数
        adjusted_image = apply_color_adjustments(test_image, params, config['param_ranges'])
        print('Applied color adjustments')
        
    except Exception as e:
        print(f'Error during inference: {e}')
        print('Note: This is expected if the TFLite model is not properly created yet')

def process_image(image_path, model_path, output_path):
    """
    处理单个图像
    
    Args:
        image_path (str): 输入图像路径
        model_path (str): TensorFlow Lite模型路径
        output_path (str): 输出图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to read image: {image_path}')
        return
    
    # 创建推理器
    inference = TFLiteInference(model_path)
    
    # 预测参数
    params = inference.predict(image)
    print(f'Predicted params for {image_path}: {params}')
    
    # 获取配置
    config = get_inference_config()
    
    # 应用参数
    adjusted_image = apply_color_adjustments(image, params, config['param_ranges'])
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, adjusted_image)
    print(f'Saved adjusted image to {output_path}')

if __name__ == '__main__':
    main()