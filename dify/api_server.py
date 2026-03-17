import os
import sys
import json
import base64
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deployment.onnx_inference import ONNXInference

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'onnx', 'model.onnx')

if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}")
    inference = None
else:
    inference = ONNXInference(model_path)

def load_image_from_url(image_url):
    """
    从URL加载图像
    
    Args:
        image_url (str): 图像URL
    
    Returns:
        PIL.Image: 加载的图像
    """
    try:
        print(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        print(f"Downloaded image, size: {len(response.content)} bytes")
        image = Image.open(BytesIO(response.content))
        print(f"Opened image: {image.size}, {image.mode}")
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        raise ValueError(f"Failed to load image from URL: {str(e)}")

def decode_base64_image(base64_string):
    """
    解码base64图像
    
    Args:
        base64_string (str): base64编码的图像
    
    Returns:
        PIL.Image: 解码后的图像
    """
    try:
        print(f"Input base64 string length: {len(base64_string)}")
        
        if base64_string.startswith('data:image'):
            parts = base64_string.split(',')
            print(f"Found data:image prefix, parts count: {len(parts)}")
            if len(parts) > 1:
                base64_string = parts[1]
            else:
                base64_string = base64_string[len('data:image/png;base64,'):]
        
        print(f"Base64 string after prefix removal: {len(base64_string)} chars")
        
        image_data = base64.b64decode(base64_string)
        print(f"Decoded image data size: {len(image_data)} bytes")
        
        image = Image.open(BytesIO(image_data))
        print(f"Image opened: {image.size}, {image.mode}")
        return image
    except Exception as e:
        print(f"Error in decode_base64_image: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

def get_image_from_input(image_input):
    """
    从输入获取图像（支持URL或base64）
    
    Args:
        image_input: 图像输入（URL或base64）
    
    Returns:
        PIL.Image: 加载的图像
    """
    if isinstance(image_input, str):
        if image_input.startswith('http://') or image_input.startswith('https://'):
            return load_image_from_url(image_input)
        elif image_input.startswith('data:image'):
            return decode_base64_image(image_input)
        else:
            raise ValueError(f"Unsupported image input format: {image_input[:50]}...")
    else:
        raise ValueError(f"Image input must be a string, got {type(image_input)}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    预测美学参数
    """
    try:
        data = request.get_json()
        
        print(f"Received request data keys: {data.keys() if data else 'None'}")
        print(f"Full request data: {json.dumps(data, indent=2) if data else 'None'}")
        
        if not data or 'image' not in data:
            return jsonify({
                'text': 'Missing image data',
                'files': [],
                'json': json.dumps({'success': False, 'error': 'Missing image data'})
            }), 400
        
        if inference is None:
            return jsonify({
                'text': 'Model not loaded',
                'files': [],
                'json': json.dumps({'success': False, 'error': 'Model not loaded'})
            }), 500
        
        image_input = data['image']
        return_base64 = data.get('return_base64', False)
        
        print(f"Image input type: {type(image_input)}")
        print(f"Image input length: {len(str(image_input))}")
        
        image = get_image_from_input(image_input)
        
        params = inference.predict(image)
        print(f"Predicted params: {params}")
        
        result = {
            'success': True,
            'params': params,
            'message': 'Successfully predicted aesthetic parameters'
        }
        
        if return_base64:
            adjusted_image = inference.apply_color_adjustments(image, params)
            print(f"Adjusted image shape: {adjusted_image.shape}")
            
            adjusted_pil = Image.fromarray(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            adjusted_pil.save(buffered, format="PNG")
            adjusted_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result['adjusted_image'] = f"data:image/png;base64,{adjusted_base64}"
            print(f"Generated adjusted image base64: {len(adjusted_base64)} chars")
        
        # 构建返回的文本内容
        exposure = params.get('exposure', 0.0)
        saturation = params.get('saturation', 1.0)
        text_content = f"图像已成功美化！\n曝光: {exposure:.2f}\n饱和度: {saturation:.2f}"
        
        # 构建最终响应
        response = {
            'text': text_content,
            'files': [],
            'json': json.dumps(result)
        }
        
        print(f"Returning response: {response}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = {
            'text': f"Error: {str(e)}",
            'files': [],
            'json': json.dumps({'success': False, 'error': str(e)})
        }
        
        return jsonify(error_response), 500

@app.route('/health', methods=['GET'])
def health():
    """
    健康检查
    """
    return jsonify({
        'status': 'healthy' if inference is not None else 'unhealthy',
        'model_loaded': inference is not None
    })

@app.route('/', methods=['GET'])
def index():
    """
    首页
    """
    return jsonify({
        'service': 'Image Aesthetic Enhancement API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Predict aesthetic parameters',
            '/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
