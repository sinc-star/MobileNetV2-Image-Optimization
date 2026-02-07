import torch
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import AestheticRegressionModel
from training.config import get_config

def export_onnx(model_path, output_path, input_shape=(1, 3, 224, 224)):
    """
    导出模型为ONNX格式
    
    Args:
        model_path (str): PyTorch模型路径
        output_path (str): ONNX模型输出路径
        input_shape (tuple): 输入形状
    """
    # 获取配置
    config = get_config()
    
    # 创建模型
    model = AestheticRegressionModel(
        pretrained=False,
        output_dim=config['output_dim']
    )
    
    # 加载模型权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {model_path}')
    else:
        raise ValueError(f'Model path not found: {model_path}')
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 导出为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f'Exported ONNX model to {output_path}')

def main():
    """
    主函数
    """
    # 获取配置
    config = get_config()
    
    # 创建保存目录
    onnx_save_dir = 'models/onnx'
    os.makedirs(onnx_save_dir, exist_ok=True)
    
    # 导出ONNX模型
    model_path = os.path.join(config['save_dir'], 'best_model_epoch_1.pth')
    output_path = os.path.join(onnx_save_dir, 'model.onnx')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    # 导出ONNX
    export_onnx(model_path, output_path)

if __name__ == '__main__':
    main()