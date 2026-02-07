import torch
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2FeatureExtractor
from models.regression_head import RegressionHead

class AestheticRegressionModel(nn.Module):
    def __init__(self, pretrained=True, output_dim=2):
        super(AestheticRegressionModel, self).__init__()
        """
        完整的美学回归模型
        
        Args:
            pretrained (bool): 是否使用预训练权重
            output_dim (int): 输出参数数量，默认2（曝光和饱和度）
        """
        # 特征提取器
        self.feature_extractor = MobileNetV2FeatureExtractor(pretrained=pretrained)
        # 回归头部
        self.regression_head = RegressionHead(
            input_dim=self.feature_extractor.feature_dim,
            output_dim=output_dim
        )
    
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        # 预测参数
        params = self.regression_head(features)
        return params
    
    def get_params_range(self):
        """
        获取参数的范围
        
        Returns:
            dict: 每个参数的范围
        """
        if self.regression_head.fc3.out_features == 2:
            return {
                'exposure': [-1.0, 1.0],  # 曝光偏移范围
                'saturation': [0.0, 2.0]   # 饱和度系数范围
            }
        else:
            return {
                'exposure': [-1.0, 1.0],
                'contrast': [0.5, 2.0],
                'saturation': [0.0, 2.0],
                'highlight': [0.0, 1.0],
                'shadow': [0.0, 1.0]
            }