def main(http_response_body):
    import json
    
    text = "无响应数据"
    exposure = 0.0
    saturation = 1.0
    adjusted_image_url = ""
    success = False
    error = ""
    
    try:
        data = json.loads(http_response_body)
        
        if 'json' in data and data['json']:
            json_str = data['json']
            result = json.loads(json_str)
            
            success = result.get('success', False)
            
            if success:
                params = result.get('params', {})
                exposure = params.get('exposure', 0.0)
                saturation = params.get('saturation', 1.0)
                adjusted_image_url = result.get('adjusted_image_url', "")
                
                text = data.get('text', "图像已成功美化！")
            else:
                error = result.get('error', '未知错误')
                text = f"处理失败: {error}"
        else:
            text = data.get('text', "无响应数据")
            error = "缺少json数据"
            
    except Exception as e:
        error = str(e)
        text = f"解析错误: {error}"
    
    return {
        "text": text,
        "exposure": exposure,
        "saturation": saturation,
        "adjusted_image_url": adjusted_image_url,
        "success": success,
        "error": error
    }
