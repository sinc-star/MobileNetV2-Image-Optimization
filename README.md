# 基于MobileNetV2的美学回归模型

## 项目简介

本项目实现了一个基于MobileNetV2的美学回归模型，用于预测图像的调色参数。模型输出两个关键的调色参数：曝光偏移和饱和度系数，然后使用这些参数对原始图像进行调色处理。

## 项目结构

```
├── data/
│   ├── prepare_data.py      # 数据预处理脚本
│   ├── data_loader.py       # 数据加载器
│   ├── dynamic_dataset.py   # 动态数据集类
│   ├── augmentations.py     # 数据增强
│   ├── create_database.py   # 数据库创建脚本
│   ├── unsplash/          # Unsplash数据集
│   │   └── db/
│   │       └── unsplash.db
│   └── upsplash/         # Unsplash数据集文件
│       ├── DOCS.md
│       ├── README.md
│       └── TERMS.md
├── models/
│   ├── mobilenetv2.py       # MobileNetV2模型定义
│   ├── regression_head.py   # 回归头部定义
│   ├── model.py             # 完整模型组装
│   ├── checkpoints/         # 模型检查点
│   ├── onnx/              # ONNX模型
│   │   └── model.onnx
│   └── tflite/            # TensorFlow Lite模型
│       └── model.tflite
├── training/
│   ├── train.py             # 训练主脚本
│   ├── evaluate.py          # 评估脚本
│   └── config.py            # 训练配置
├── deployment/
│   ├── export_onnx.py       # 导出ONNX模型
│   ├── convert_tflite.py    # 转换为TFLite
│   └── inference.py         # 推理脚本
├── utils/
│   └── metrics.py           # 评估指标
├── test_img/               # 测试图像
├── start.py               # 图像优化主脚本
├── README.md              # 项目说明
└── .gitignore            # Git忽略文件配置
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
pip install opencv-python onnxruntime pillow matplotlib scikit-learn
```

## 模型架构

### 基础网络
- **MobileNetV2**：作为特征提取 backbone，使用ImageNet预训练权重

### 回归头部
- **两参数回归**：输出曝光偏移和饱和度系数
- **网络结构**：MobileNetV2 → Global Average Pooling → Dense(128) → ReLU → Dense(64) → ReLU → Dense(2)


### 数据集来源
- **官方仓库**: https://github.com/unsplash/datasets

### 数据集使用
本项目使用 Unsplash 数据集中的图像作为训练样本，通过人工标注或自动生成的方式为每张图像分配调色参数（曝光偏移和饱和度系数）。

**注意**: 使用 Unsplash 数据集时，请遵守其使用条款和许可证要求。

## 训练流程

### 1. 数据准备
- 准备训练、验证和测试数据集
- 数据集应包含图像和对应的调色参数标注
- 使用 `data/prepare_data.py` 脚本进行数据预处理

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

### 导出ONNX模型

```bash
python deployment/export_onnx.py
```

导出的ONNX模型可以用于：
- **移动端部署**：使用ONNX Runtime在Android或iOS上运行
- **桌面应用**：使用ONNX Runtime进行推理
- **Web应用**：通过ONNX.js在浏览器中运行

### ONNX Runtime安装

```bash
# Python环境
pip install onnxruntime

# Android (在app/build.gradle中添加)
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'

# iOS (在Podfile中添加)
pod 'onnxruntime-objc', '~> 1.18.0'
```

## 推理流程

### 1. 处理单个图像

```python
import onnxruntime as rt
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

def apply_color_adjustments(image, exposure, saturation):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + exposure), 0, 255)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 加载模型
sess = rt.InferenceSession("models/checkpoints/model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 处理图像
image_path = 'path/to/image.jpg'
preprocessed = preprocess_image(image_path)
params = sess.run([output_name], {input_name: preprocessed})
exposure = params[0][0][0]
saturation = params[0][0][1]

# 应用颜色调整
original_image = cv2.imread(image_path)
adjusted_image = apply_color_adjustments(original_image, exposure, saturation)
cv2.imwrite('path/to/output.jpg', adjusted_image)
```

### 2. 批量处理

```python
import onnxruntime as rt
import cv2
import os

# 创建推理会话
sess = rt.InferenceSession("models/checkpoints/model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 处理多个图像
image_dir = 'path/to/images'
output_dir = 'path/to/output'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        
        # 预处理和推理
        preprocessed = preprocess_image(image_path)
        params = sess.run([output_name], {input_name: preprocessed})
        exposure = params[0][0][0]
        saturation = params[0][0][1]
        
        # 应用颜色调整
        original_image = cv2.imread(image_path)
        adjusted_image = apply_color_adjustments(original_image, exposure, saturation)
        
        # 保存结果
        output_path = os.path.join(output_dir, f'optimized_{filename}')
        cv2.imwrite(output_path, adjusted_image)
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
- **ONNX Runtime优化**：使用ONNX Runtime的高性能推理引擎
- **批处理推理**：提高处理效率
- **硬件加速**：利用GPU、CPU或NPU加速推理
- **模型量化**：通过ONNX量化工具减小模型大小，提高推理速度

## 注意事项

1. **数据集质量**：模型性能依赖于数据集的质量和多样性
2. **参数范围**：确保训练数据中的参数值在合理范围内
3. **模型选择**：根据部署环境选择合适的模型大小和优化级别
4. **推理速度**：在移动设备上，建议使用ONNX Runtime进行推理
5. **ONNX兼容性**：确保ONNX Runtime版本与导出的ONNX模型版本兼容

## 未来工作

1. **多参数回归**：扩展模型以预测更多调色参数（对比度、高光、阴影等）
2. **领域适应**：针对不同场景（人像、风景、夜景等）进行模型适应
3. **实时推理**：进一步优化推理速度，实现实时调色
4. **用户反馈**：集成用户反馈机制，持续改进模型性能

## 参考资料

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [ONNX Runtime官方文档](https://onnxruntime.ai/docs/)
- [ONNX官方文档](https://onnx.ai/onnx/intro/)
