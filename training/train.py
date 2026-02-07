import os
import sys
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from models.model import AestheticRegressionModel
from data.dynamic_dataset import UnsplashDynamicDataset
from data.augmentations import get_default_augmentations
from utils.metrics import calculate_metrics

def create_dynamic_data_loader(db_path, batch_size=32, shuffle=True, split='train'):
    """
    创建动态数据加载器
    
    Args:
        db_path (str): 数据库路径
        batch_size (int): 批量大小
        shuffle (bool): 是否打乱
        split (str): 数据集划分，可选值: 'train', 'val', 'test'
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = UnsplashDynamicDataset(db_path, split=split)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 在Windows上设置为0以避免多线程问题
        pin_memory=False  # 当num_workers=0时，pin_memory设置为False
    )
    return data_loader

class Trainer:
    def __init__(self, config):
        """
        训练器
        
        Args:
            config (dict): 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = AestheticRegressionModel(
            pretrained=config['pretrained'],
            output_dim=config['output_dim']
        ).to(self.device)
        
        # 创建损失函数
        self.criterion = nn.MSELoss()
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 创建学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['max_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # 创建数据加载器
        db_path = config['db_path']
        self.train_loader = create_dynamic_data_loader(
            db_path=db_path,
            batch_size=config['batch_size'],
            shuffle=True,
            split='train'
        )
        
        # 对于验证集，使用划分好的验证集
        self.val_loader = create_dynamic_data_loader(
            db_path=db_path,
            batch_size=config['batch_size'],
            shuffle=False,
            split='val'
        )
        
        # 创建数据增强
        self.augmentations = get_default_augmentations()
        
        # 创建保存目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化最佳验证损失
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
    
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        
        Args:
            epoch (int): 当前 epoch
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # 移动到设备
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 应用数据增强
            augmented_images = []
            augmented_targets = []
            for img, tgt in zip(images, targets):
                aug_img, aug_tgt = self.augmentations(img, tgt)
                augmented_images.append(aug_img)
                augmented_targets.append(aug_tgt)
            
            images = torch.stack(augmented_images)
            targets = torch.stack(augmented_targets)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                print(f'Epoch [{epoch+1}/{self.config["max_epochs"]}], '\
                      f'Batch [{batch_idx+1}/{len(self.train_loader)}], '\
                      f'Loss: {loss.item():.4f}, '\
                      f'LR: {self.scheduler.get_last_lr()[0]:.6f}')
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        print(f'Epoch [{epoch+1}/{self.config["max_epochs"]}], '\
              f'Train Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def validate(self, epoch):
        """
        验证
        
        Args:
            epoch (int): 当前 epoch
        
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # 移动到设备
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算指标
        import numpy as np
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_outputs, all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        print(f'Epoch [{epoch+1}/{self.config["max_epochs"]}], '\
              f'Val Loss: {avg_loss:.4f}, '\
              f'RMSE: {metrics["rmse"]:.4f}, '\
              f'MAE: {metrics["mae"]:.4f}, '\
              f'R2: {metrics["r2"]:.4f}')
        
        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.early_stop_counter = 0
            model_path = os.path.join(self.save_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': metrics
            }, model_path)
            print(f'Saved best model to {model_path}')
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.config['early_stop_patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                return True
        
        return False
    
    def train(self):
        """
        开始训练
        """
        print(f'Starting training on {self.device}')
        print(f'Model: AestheticRegressionModel')
        print(f'Output dim: {self.config["output_dim"]}')
        print(f'Train batches: {len(self.train_loader)}')
        print(f'Val batches: {len(self.val_loader)}')
        
        for epoch in range(self.config['max_epochs']):
            start_time = time.time()
            
            # 训练
            self.train_epoch(epoch)
            
            # 验证
            early_stop = self.validate(epoch)
            
            if early_stop:
                break
            
            # 打印时间
            end_time = time.time()
            print(f'Epoch {epoch+1} took {end_time - start_time:.2f} seconds')
        
        print('Training completed!')

def main():
    """
    主函数
    """
    # 导入配置
    from training.config import get_config
    
    # 获取配置
    config = get_config()
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()