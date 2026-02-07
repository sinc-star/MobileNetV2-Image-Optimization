def get_config():
    """
    获取训练配置
    
    Returns:
        dict: 训练配置
    """
    return {
        # 模型配置
        'pretrained': True,
        'output_dim': 5,  # 五参数版本：曝光、对比度、饱和度、高光、阴影
        
        # 数据配置
        'db_path': 'data/unsplash/db/unsplash.db',  # 数据库路径
        'batch_size': 32,
        
        # 训练配置
        'learning_rate': 1e-5,  # 学习率
        'weight_decay': 1e-5,  # 权重衰减
        'max_epochs': 20,  # 最大训练轮数
        'early_stop_patience': 3,  # 早停耐心值，超过此轮数无改进则停止训练
        'log_interval': 10,  # 日志打印间隔，每训练此轮数打印一次日志
        
        # 保存配置
        'save_dir': 'models/checkpoints',
    }

def get_eval_config():
    """
    获取评估配置
    
    Returns:
        dict: 评估配置
    """
    return {
        # 模型配置
        'output_dim': 5,  # 五参数版本：曝光、对比度、饱和度、高光、阴影
        
        # 数据配置
        'test_data_dir': 'data/test',
        'test_annotations_file': None,
        'batch_size': 32,
        
        # 模型路径
        'model_path': 'models/checkpoints/best_model_epoch_1.pth',# 评估时使用的模型路径
    }

def get_inference_config():
    """
    获取推理配置
    
    Returns:
        dict: 推理配置
    """
    return {
        # 模型配置
        'output_dim': 2,
        
        # 图像配置
        'image_size': (224, 224),
        
        # 模型路径
        'model_path': 'models/checkpoints/best_model.pth',
        'tflite_model_path': 'models/tflite/model.tflite',
        
        # 参数范围
        'param_ranges': {
            'exposure': [-1.0, 1.0],     # 曝光偏移范围
            'contrast': [0.5, 2.0],       # 对比度系数范围
            'saturation': [0.0, 2.0],     # 饱和度系数范围
            'highlight': [0.0, 1.0],       # 高光调整范围
            'shadow': [0.0, 1.0]          # 阴影调整范围
        }
    }