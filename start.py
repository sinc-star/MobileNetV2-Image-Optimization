import os
import sys
import cv2
import numpy as np
import onnxruntime as rt

# 添加父目录到Python路径，确保能找到models模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def preprocess_image(image_path):
    """
    预处理图像，使其符合模型输入要求
    
    Args:
        image_path (str): 图像路径
    
    Returns:
        np.ndarray: 预处理后的图像数组
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 调整图像大小为224x224
    image = cv2.resize(image, (224, 224))
    
    # 转换颜色空间：BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 归一化到0-1范围
    image = image / 255.0
    
    # 应用ImageNet均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # 调整通道顺序：(H, W, C) -> (C, H, W)
    image = image.transpose(2, 0, 1)
    
    # 添加批次维度
    image = np.expand_dims(image, axis=0)
    
    # 转换为float32
    image = image.astype(np.float32)
    
    return image


def apply_color_adjustments(image_path, exposure, saturation):
    """
    应用颜色调整到图像
    
    Args:
        image_path (str): 图像路径
        exposure (float): 曝光调整值
        saturation (float): 饱和度调整值
    
    Returns:
        np.ndarray: 调整后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 调整曝光（通过调整V通道）
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + exposure), 0, 255)
    
    # 调整饱和度（通过调整S通道）
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    
    # 转换回BGR颜色空间
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_image


def main():
    """
    主函数，加载模型并处理测试图像
    """
    # 测试图像列表
    image_files = ["test_img/test01.png", "test_img/test02.png", "test_img/test03.png", "test_img/test04.png", "test_img/test05.png", "test_img/test06.png"]
    # 模型路径
    model_path = "models/checkpoints/model.onnx"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    # 加载模型
    print("Loading model...")
    try:
        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        print(f"Model loaded successfully. Input name: {input_name}, Output name: {output_name}")
        # 硬编码输入名称以确保正确性
        input_name = "input"
        output_name = "output"
        print(f"Using hardcoded input/output names: {input_name}, {output_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 处理每个图像
    for image_file in image_files:
        # 检查图像是否存在
        if not os.path.exists(image_file):
            print(f"Error: Image file '{image_file}' not found. Skipping.")
            continue
        
        print(f"\nProcessing image: {image_file}")
        
        # 预处理图像
        print("  Preprocessing image...")
        try:
            preprocessed_image = preprocess_image(image_file)
            print(f"  Image preprocessed successfully. Shape: {preprocessed_image.shape}, Dtype: {preprocessed_image.dtype}")
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            continue
        
        # 预测参数
        print("  Predicting color adjustment parameters...")
        try:
            params = sess.run([output_name], {input_name: preprocessed_image})
            print(f"  Prediction completed successfully. Output shape: {np.array(params).shape}")
        except Exception as e:
            print(f"Error predicting parameters: {e}")
            continue
        
        # 提取曝光和饱和度参数
        try:
            exposure = params[0][0][0]
            saturation = params[0][0][1]
            
            print(f"  Predicted parameters:")
            print(f"    Exposure: {exposure:.4f}")
            print(f"    Saturation: {saturation:.4f}")
        except Exception as e:
            print(f"Error extracting parameters: {e}")
            continue
        
        # 应用颜色调整
        print("  Applying color adjustments...")
        try:
            adjusted_image = apply_color_adjustments(image_file, exposure, saturation)
            print(f"  Color adjustments applied successfully. Shape: {adjusted_image.shape}")
        except Exception as e:
            print(f"Error applying color adjustments: {e}")
            continue
        
        # 保存调整后的图像
        # 从文件路径中提取文件名
        base_name = os.path.basename(image_file)
        output_file = f"optimized_{base_name}"
        
        # 确保输出目录存在
        output_dir = "optimized_test_img"
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建完整的输出路径
        output_path = os.path.join(output_dir, output_file)
        
        # 保存图像
        try:
            success = cv2.imwrite(output_path, adjusted_image)
            if success:
                print(f"  Optimized image saved to '{output_path}'.")
            else:
                print(f"  Error: Failed to save optimized image to '{output_path}'.")
        except Exception as e:
            print(f"Error saving image: {e}")
            continue
    
    print("\nAll images processed!")


if __name__ == "__main__":
    main()
