import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(RegressionHead, self).__init__()
        """
        回归头部，用于预测调色参数
        
        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出参数数量，默认2（曝光和饱和度）
        """
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x