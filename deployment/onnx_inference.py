import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
import base64
from io import BytesIO
from PIL import Image

class ONNXInference:
    def __init__(self, model_path):
        """
        ONNX模型推理
        
        Args:
            model_path (str): ONNX模型路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f'Loaded ONNX model from {model_path}')
        print(f'Input shape: {self.session.get_inputs()[0].shape}')
        print(f'Output shape: {self.session.get_outputs()[0].shape}')
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: 输入图像 (可以是路径、numpy数组或PIL Image)
        
        Returns:
            np.ndarray: 预处理后的图像
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to read image from path: {image}")
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """
        预测调色参数
        
        Args:
            image: 输入图像
        
        Returns:
            dict: 预测的参数
        """
        preprocessed_image = self.preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed_image})
        params = outputs[0][0]
        
        output_dim = len(params)
        
        if output_dim == 2:
            return {
                'exposure': float(params[0]),
                'saturation': float(params[1])
            }
        elif output_dim == 5:
            return {
                'exposure': float(params[0]),
                'contrast': float(params[1]),
                'saturation': float(params[2]),
                'highlight': float(params[3]),
                'shadow': float(params[4])
            }
        else:
            raise ValueError(f"Unsupported output dimension: {output_dim}")
    
    def apply_color_adjustments(self, image, params):
        """
        应用调色参数
        
        Args:
            image: 输入图像
            params (dict): 调色参数
        
        Returns:
            np.ndarray: 调色后的图像
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        adjusted_image = image.astype(np.float32)
        
        exposure = params.get('exposure', 0.0)
        saturation = params.get('saturation', 1.0)
        
        alpha = 1.0 + exposure
        adjusted_image = np.clip(adjusted_image * alpha, 0, 255)
        
        hsv = cv2.cvtColor(adjusted_image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        adjusted_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        if 'contrast' in params:
            contrast = params.get('contrast', 1.0)
            mean_val = np.mean(adjusted_image)
            adjusted_image = np.clip((adjusted_image - mean_val) * contrast + mean_val, 0, 255)
        
        if 'highlight' in params and 'shadow' in params:
            highlight = params.get('highlight', 0.5)
            shadow = params.get('shadow', 0.5)
            
            lab = cv2.cvtColor(adjusted_image.astype(np.uint8), cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            l_normalized = l_channel / 255.0
            
            highlight_weight = np.clip((l_normalized - 0.5) * 2, 0, 1)
            shadow_weight = np.clip((0.5 - l_normalized) * 2, 0, 1)
            
            highlight_adjustment = 1.0 + highlight * 0.3
            shadow_adjustment = 0.7 + shadow * 0.3
            
            l_channel = l_channel * (highlight_weight * highlight_adjustment + shadow_weight * shadow_adjustment + 
                                      (1 - highlight_weight - shadow_weight))
            
            l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
            lab[:, :, 0] = l_channel
            
            adjusted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        
        adjusted_image = cv2.bilateralFilter(adjusted_image, 5, 50, 50)
        
        return adjusted_image
    
    def image_to_base64(self, image):
        """
        将图像转换为base64编码
        
        Args:
            image: 输入图像
        
        Returns:
            str: base64编码的图像
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def base64_to_image(self, base64_string):
        """
        将base64编码转换为图像
        
        Args:
            base64_string (str): base64编码的图像
        
        Returns:
            np.ndarray: 图像数组
        """
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

def main():
    """
    主函数 - 测试推理
    """
    model_path = r"E:\ai\MobileNetV2\models\onnx\model_5param_epoch_19.onnx"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    inference = ONNXInference(model_path)
    
    test_image_path = 'test_img/test01.png'
    if os.path.exists(test_image_path):
        params = inference.predict(test_image_path)
        print(f'Predicted params: {params}')
        
        adjusted_image = inference.apply_color_adjustments(test_image_path, params)
        output_path = 'test_img/test01_adjusted.png'
        cv2.imwrite(output_path, adjusted_image)
        print(f'Saved adjusted image to {output_path}')
    else:
        print(f"Test image not found at {test_image_path}")

if __name__ == '__main__':
    main()
