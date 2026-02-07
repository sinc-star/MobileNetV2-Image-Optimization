# 基于MobileNetV2的美学回归模型

## 项目简介

本项目实现了一个基于MobileNetV2的美学回归模型，用于预测图像的调色参数。模型输出两个关键的调色参数：曝光偏移和饱和度系数，然后使用这些参数对原始图像进行调色处理。

## 项目结构

```
├── data/
│   ├── prepare_data.py      # 数据预处理脚本
│   ├── data_loader.py       # 数据加载器
│   └── augmentations.py     # 数据增强
├── models/
│   ├── mobilenetv2.py       # MobileNetV2模型定义
│   ├── regression_head.py   # 回归头部定义
│   └── model.py             # 完整模型组装
├── training/
│   ├── train.py             # 训练主脚本
│   ├── evaluate.py          # 评估脚本
│   └── config.py            # 训练配置
├── deployment/
│   ├── export_onnx.py       # 导出ONNX模型
│   ├── convert_tflite.py    # 转换为TFLite
│   └── inference.py         # 推理脚本
├── utils/
│   ├── metrics.py           # 评估指标
├── README.md                # 项目说明
└── training_flow.md         # 训练流程文档
```

## 环境配置

### 依赖包安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
env\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装依赖
pip install torch torchvision torchaudio
pip install opencv-python tensorflow onnx onnx-tf pillow matplotlib scikit-learn
```

## 模型架构

### 基础网络
- **MobileNetV2**：作为特征提取 backbone，使用ImageNet预训练权重

### 回归头部
- **两参数回归**：输出曝光偏移和饱和度系数
- **网络结构**：MobileNetV2 → Global Average Pooling → Dense(128) → ReLU → Dense(64) → ReLU → Dense(2)

## 训练流程

### 1. 数据准备
- 准备训练、验证和测试数据集
- 数据集应包含图像和对应的调色参数标注

### 2. 配置训练参数
修改 `training/config.py` 中的训练配置：

```python
def get_config():
    return {
        'pretrained': True,
        'output_dim': 2,
        'train_data_dir': 'data/train',
        'val_data_dir': 'data/val',
        'batch_size': 32,
        'learning_rate': 1e-4,
        'max_epochs': 100,
        'early_stop_patience': 10,
        'save_dir': 'models/checkpoints',
    }
```

### 3. 开始训练

```bash
python training/train.py
```

### 4. 模型评估

```bash
python training/evaluate.py
```

## 模型部署

### 1. 导出ONNX模型

```bash
python deployment/export_onnx.py
```

### 2. 转换为TensorFlow Lite

```bash
python deployment/convert_tflite.py
```

## 推理流程

### 1. 处理单个图像

```python
from deployment.inference import process_image

image_path = 'path/to/image.jpg'
model_path = 'models/tflite/model.tflite'
output_path = 'path/to/output.jpg'

process_image(image_path, model_path, output_path)
```

### 2. 批量处理

可以使用 `deployment/inference.py` 中的 `TFLiteInference` 类进行批量处理：

```python
from deployment.inference import TFLiteInference
import cv2

# 创建推理器
inference = TFLiteInference('models/tflite/model.tflite')

# 处理多个图像
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for image_path in image_paths:
    # 读取图像
    image = cv2.imread(image_path)
    # 预测参数
    params = inference.predict(image)
    # 应用参数
    adjusted_image = apply_color_adjustments(image, params, config['param_ranges'])
    # 保存结果
    cv2.imwrite(f'output_{image_path}', adjusted_image)
```

## 参数说明

### 模型输出参数
- **曝光偏移（Exposure）**：范围 [-1.0, 1.0]
  - 负值：降低曝光（变暗）
  - 正值：增加曝光（变亮）
  - 0：保持原始曝光

- **饱和度系数（Saturation）**：范围 [0.0, 2.0]
  - 0.0：完全去饱和（灰度图像）
  - 1.0：保持原始饱和度
  - 2.0：增加饱和度（更鲜艳）

### 调色效果

| 曝光值 | 效果 |
|-------|------|
| -1.0  | 最暗 |
| -0.5  | 较暗 |
| 0.0   | 原始 |
| 0.5   | 较亮 |
| 1.0   | 最亮 |

| 饱和度值 | 效果 |
|---------|------|
| 0.0     | 灰度 |
| 0.5     | 低饱和 |
| 1.0     | 原始 |
| 1.5     | 高饱和 |
| 2.0     | 最高饱和 |

## 模型优化

### 训练优化
- **学习率调度**：使用余弦退火策略
- **早停策略**：防止过拟合
- **数据增强**：随机翻转、旋转、亮度调整等

### 推理优化
- **TensorFlow Lite量化**：减小模型大小，提高推理速度
- **批处理推理**：提高处理效率
- **硬件加速**：利用GPU或NPU加速推理

## 注意事项

1. **数据集质量**：模型性能依赖于数据集的质量和多样性
2. **参数范围**：确保训练数据中的参数值在合理范围内
3. **模型选择**：根据部署环境选择合适的模型大小和量化级别
4. **推理速度**：在移动设备上，建议使用量化后的TFLite模型

## 未来工作

1. **多参数回归**：扩展模型以预测更多调色参数（对比度、高光、阴影等）
2. **领域适应**：针对不同场景（人像、风景、夜景等）进行模型适应
3. **实时推理**：进一步优化推理速度，实现实时调色
4. **用户反馈**：集成用户反馈机制，持续改进模型性能

## 参考资料

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Lite官方文档](https://www.tensorflow.org/lite)

## 许可证

本项目采用MIT许可证。