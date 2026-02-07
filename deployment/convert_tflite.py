import os
import sys
import torch
import tensorflow as tf

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import AestheticRegressionModel
from training.config import get_config

def convert_tflite(model_path, output_path, quantize=False):
    """
    将PyTorch模型转换为TensorFlow Lite格式
    
    Args:
        model_path (str): PyTorch模型路径
        output_path (str): TensorFlow Lite模型输出路径
        quantize (bool): 是否进行量化
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
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 导出为ONNX格式
    onnx_path = 'models/onnx/model.onnx'
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13
    )
    print(f'Exported ONNX model to {onnx_path}')
    
    # 尝试使用TensorFlow的ONNX导入功能
    try:
        import onnx
        from tensorflow.python.onnx.keras_tensor import keras_tensor_to_tf_tensor
        from tensorflow.python.onnx.tf_onnx_api import convert_graph
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 转换为TensorFlow图
        tf_graph = convert_graph(onnx_model.SerializeToString())
        print('Converted ONNX model to TensorFlow graph')
        
        # 保存为SavedModel格式
        saved_model_path = 'models/tf/saved_model'
        os.makedirs(saved_model_path, exist_ok=True)
        
        # 创建签名函数
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, 224, 224], dtype=tf.float32, name='input')])
        def serving_fn(input):
            return tf_graph(input)
        
        # 保存模型
        tf.saved_model.save(tf_graph, saved_model_path, signatures={'serving_default': serving_fn})
        print(f'Exported TensorFlow SavedModel to {saved_model_path}')
        
    except Exception as e:
        print(f'Error converting ONNX to TensorFlow: {e}')
        print('Falling back to using tf2onnx...')
        
        # 安装tf2onnx
        os.system('pip install tf2onnx')
        
        # 使用tf2onnx转换
        saved_model_path = 'models/tf/saved_model'
        os.makedirs(saved_model_path, exist_ok=True)
        
        os.system(f'python -m tf2onnx.convert --onnx {onnx_path} --output {saved_model_path} --opset 13')
        print(f'Exported TensorFlow SavedModel to {saved_model_path}')
    
    # 转换为TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # 配置量化
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print('Enabled quantization')
    
    # 转换模型
    tflite_model = converter.convert()
    print('Converted TensorFlow model to TensorFlow Lite format')
    
    # 保存TensorFlow Lite模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f'Saved TensorFlow Lite model to {output_path}')

def main():
    """
    主函数
    """
    # 获取配置
    config = get_config()
    
    # 创建保存目录
    tflite_save_dir = 'models/tflite'
    os.makedirs(tflite_save_dir, exist_ok=True)
    
    # 转换TFLite模型
    model_path = os.path.join(config['save_dir'], 'best_model_epoch_6.pth')
    output_path = os.path.join(tflite_save_dir, 'model.tflite')
    
    # 转换为TFLite
    convert_tflite(model_path, output_path, quantize=False)

if __name__ == '__main__':
    main()