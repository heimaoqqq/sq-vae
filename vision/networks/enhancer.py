import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroDopplerEnhancer(nn.Module):
    """
    专为微多普勒时频图设计的细节增强模块
    增强垂直特征(多普勒频移)和水平特征(频谱连续性)
    """
    def __init__(self):
        super().__init__()
        # 特征提取和增强网络
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 垂直特征增强（针对微多普勒垂直细节）
        self.vertical_enhance = nn.Conv2d(32, 16, kernel_size=(5, 1), padding=(2, 0))
        
        # 水平特征增强（针对基频连续性）
        self.horizontal_enhance = nn.Conv2d(32, 16, kernel_size=(1, 5), padding=(0, 2))
        
        # 合成最终增强结果
        self.fusion = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # 允许正负增强
        )
        
        # 可学习的增强强度因子
        self.enhance_factor = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, x):
        # 提取特征
        features = self.feature_net(x)
        
        # 垂直和水平特征增强
        v_features = self.vertical_enhance(features)
        h_features = self.horizontal_enhance(features)
        
        # 合并增强特征
        combined = torch.cat([v_features, h_features], dim=1)
        enhancement = self.fusion(combined)
        
        # 应用增强，并限制在有效范围内
        enhanced = x + self.enhance_factor * enhancement
        return torch.clamp(enhanced, 0, 1)


class ResidualDetailEnhancer(nn.Module):
    """
    基于残差学习的细节增强器
    """
    def __init__(self, channels=3):
        super().__init__()
        # 细节提取网络
        self.detail_net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, padding=1)
        )
        # 可学习的增强系数
        self.enhance_factor = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # 提取并增强细节
        residual_details = self.detail_net(x)
        enhanced = x + self.enhance_factor * residual_details
        return torch.clamp(enhanced, 0, 1) 