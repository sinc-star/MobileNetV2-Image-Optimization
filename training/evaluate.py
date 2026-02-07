import os
import sys
import numpy as np
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import AestheticRegressionModel
from data.dynamic_dataset import UnsplashDynamicDataset
from utils.metrics import calculate_metrics
from training.train import create_dynamic_data_loader
from training.config import get_config


class Evaluator:
    def __init__(self, config):
        """
        评估器
        
        Args:
            config (dict): 评估配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = AestheticRegressionModel(
            pretrained=False,
            output_dim=config['output_dim']
        ).to(self.device)
        
        # 加载模型权重
        if os.path.exists(config['model_path']):
            checkpoint = torch.load(config['model_path'], map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded model from {config["model_path"]}')
        else:
            raise ValueError(f'Model path not found: {config["model_path"]}')
        
        # 创建数据加载器
        # 获取训练配置中的数据库路径
        train_config = get_config()
        db_path = train_config['db_path']
        
        # 创建测试数据加载器
        self.test_loader = create_dynamic_data_loader(
            db_path=db_path,
            batch_size=config['batch_size'],
            shuffle=False,
            split='test'
        )
    
    def evaluate(self):
        """
        评估模型
        
        Returns:
            dict: 评估指标
        """
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.test_loader):
                # 移动到设备
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    print(f'Processing batch {batch_idx+1}/{len(self.test_loader)}')
        
        # 计算指标
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_outputs, all_targets)
        
        # 打印指标
        print('Evaluation Results:')
        print(f'RMSE: {metrics["rmse"]:.4f}')
        print(f'MAE: {metrics["mae"]:.4f}')
        print(f'R2: {metrics["r2"]:.4f}')
        
        # 打印参数预测示例
        print('\nPrediction Examples:')
        for i in range(min(5, len(all_outputs))):
            print(f'Example {i+1}:')
            print(f'  Predicted: Exposure={all_outputs[i, 0]:.4f}, Saturation={all_outputs[i, 1]:.4f}')
            print(f'  Ground Truth: Exposure={all_targets[i, 0]:.4f}, Saturation={all_targets[i, 1]:.4f}')
        
        return metrics

def main():
    """
    主函数
    """
    # 导入配置
    from training.config import get_eval_config
    
    # 获取配置
    config = get_eval_config()
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    # 开始评估
    evaluator.evaluate()

if __name__ == '__main__':
    main()