# 将 MobileNetV2 ONNX 模型集成到 Dify 使用指南

## 概述

本指南将帮助你将 MobileNetV2 图像美学增强模型集成到 Dify 平台中，实现在 Dify 中使用该模型进行图像美学调整。

## 文件结构

```
MobileNetV2/
├── models/
│   └── onnx/
│       └── model.onnx          # ONNX 模型文件
├── deployment/
│   └── onnx_inference.py       # ONNX 推理脚本
└── dify/
    ├── api_server.py            # Flask API 服务器
    ├── tool.yaml               # Dify 工具配置文件
    ├── requirements.txt        # Python 依赖
    ├── start_server.bat       # Windows 启动脚本
    └── start_server.sh        # Linux/Mac 启动脚本
```

## 前置要求

1. Python 3.8 或更高版本
2. 已训练好的 ONNX 模型文件（位于 `models/onnx/model.onnx`）
3. Dify 平台访问权限

## 安装步骤

### 1. 安装依赖

在 `dify` 目录下安装所需的 Python 包：

```bash
cd dify
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install flask flask-cors onnxruntime opencv-python Pillow numpy
```

### 2. 验证模型文件

确保 ONNX 模型文件存在于正确位置：

```bash
ls ../models/onnx/model.onnx
```

如果模型文件不存在，请先运行导出脚本：

```bash
cd ..
python deployment/export_onnx.py
```

### 3. 启动 API 服务器

#### Windows 系统

双击运行 `start_server.bat` 或在命令行中执行：

```bash
cd dify
python api_server.py
```

#### Linux/Mac 系统

```bash
cd dify
chmod +x start_server.sh
./start_server.sh
```

服务器默认在 `http://localhost:5000` 启动。

### 4. 测试 API

使用 curl 测试 API：

```bash
# 健康检查
curl http://localhost:5000/health

# 预测测试（需要准备一张测试图片）
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "return_base64": true
  }'
```

## 在 Dify 中集成

### 方法一：使用自定义工具（推荐）

1. **登录 Dify 平台**

2. **创建自定义工具**

   - 进入 Dify 的"工具"页面
   - 点击"创建自定义工具"
   - 选择"导入工具配置"

3. **上传工具配置**

   - 上传 `dify/tool.yaml` 文件
   - 或者手动配置：
     - 工具名称：`image_aesthetic_enhancement`
     - API 类型：OpenAPI
     - API 地址：`http://localhost:5000`
     - 端点：`/predict`

4. **配置参数**

   - 模型路径：`models/onnx/model.onnx`（如果需要自定义）
   - 其他参数保持默认

5. **测试工具**

   - 在 Dify 中测试工具是否正常工作
   - 上传一张图片，验证返回的预测参数

### 方法二：使用 API 调用

在 Dify 的代码节点中直接调用 API：

```python
import requests
import base64

def enhance_image(image_base64):
    url = "http://localhost:5000/predict"
    
    payload = {
        "image": image_base64,
        "return_base64": True
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result['success']:
        return {
            'params': result['params'],
            'adjusted_image': result['adjusted_image']
        }
    else:
        raise Exception(result['error'])
```

## API 接口说明

### POST /predict

预测图像的美学参数

**请求体：**

```json
{
  "image": "data:image/png;base64,...",
  "return_base64": false
}
```

**参数说明：**

- `image` (必需): Base64 编码的图像数据
- `return_base64` (可选): 是否返回调整后的图像，默认为 false

**响应：**

```json
{
  "success": true,
  "params": {
    "exposure": 0.123,
    "saturation": 1.456
  },
  "adjusted_image": "data:image/png;base64,...",
  "message": "Successfully predicted aesthetic parameters"
}
```

**参数范围：**

- `exposure`: -1.0 到 1.0（曝光调整）
- `saturation`: 0.0 到 2.0（饱和度调整）

### GET /health

健康检查端点

**响应：**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 使用示例

### 在 Dify 工作流中使用

1. **创建工作流**

2. **添加图像输入节点**

3. **添加自定义工具节点**

   - 选择 `image_aesthetic_enhancement` 工具
   - 配置输入参数

4. **添加输出节点**

   - 显示预测的参数
   - 或显示调整后的图像

### Python 脚本调用示例

```python
import requests
import base64
from PIL import Image
import io

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 读取图片并转换为 base64
image_base64 = image_to_base64('test_img/test01.png')
image_base64 = f"data:image/png;base64,{image_base64}"

# 调用 API
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'image': image_base64,
        'return_base64': True
    }
)

result = response.json()
print(f"Exposure: {result['params']['exposure']}")
print(f"Saturation: {result['params']['saturation']}")

# 保存调整后的图像
if 'adjusted_image' in result:
    adjusted_data = result['adjusted_image'].split(',')[1]
    adjusted_image = Image.open(io.BytesIO(base64.b64decode(adjusted_data)))
    adjusted_image.save('output_adjusted.png')
```

## 故障排除

### 问题 1：模型加载失败

**错误信息：** `Model not found at models/onnx/model.onnx`

**解决方案：**

1. 检查模型文件是否存在
2. 运行导出脚本生成 ONNX 模型：
   ```bash
   python deployment/export_onnx.py
   ```

### 问题 2：端口被占用

**错误信息：** `Address already in use`

**解决方案：**

1. 修改 `api_server.py` 中的端口号
2. 或停止占用 5000 端口的进程

### 问题 3：依赖安装失败

**解决方案：**

1. 使用虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r dify/requirements.txt
   ```

2. 单独安装依赖：
   ```bash
   pip install flask flask-cors onnxruntime opencv-python Pillow numpy
   ```

### 问题 4：Dify 无法连接到 API

**解决方案：**

1. 确认 API 服务器正在运行
2. 检查防火墙设置
3. 确认 Dify 可以访问 `http://localhost:5000`
4. 如果使用 Docker，确保端口映射正确

## 性能优化

1. **批量处理：** 修改 API 以支持批量图像处理
2. **缓存：** 对相同图像的预测结果进行缓存
3. **GPU 加速：** 使用 GPU 版本的 ONNX Runtime
4. **模型量化：** 对 ONNX 模型进行量化以减小模型大小

## 安全建议

1. **认证：** 添加 API 密钥认证
2. **速率限制：** 实现请求速率限制
3. **输入验证：** 验证输入图像的大小和格式
4. **HTTPS：** 在生产环境中使用 HTTPS

## 扩展功能

可以考虑添加以下功能：

1. 支持更多美学参数（对比度、高光、阴影等）
2. 支持多种图像格式
3. 添加图像质量评估
4. 支持实时视频流处理
5. 添加批量处理接口

## 联系支持

如有问题，请查阅：
- Dify 官方文档：https://docs.dify.ai
- ONNX Runtime 文档：https://onnxruntime.ai/docs/
