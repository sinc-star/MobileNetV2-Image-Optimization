import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2FeatureExtractor, self).__init__()
        # 加载预训练的MobileNetV2模型
        base_model = mobilenet_v2(pretrained=pretrained)
        # 移除分类头部，只保留特征提取部分
        self.features = base_model.features
        # 获取特征维度
        self.feature_dim = 1280  # MobileNetV2最后一层的输出通道数
    
    def forward(self, x):
        # 前向传播获取特征
        x = self.features(x)
        # 全局平均池化
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 展平特征
        x = torch.flatten(x, 1)
        return x