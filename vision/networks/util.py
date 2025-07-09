import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish激活函数 x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(name):
    """获取激活函数"""
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif name.lower() == "swish":
        return Swish()
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif name.lower() == "tanh":
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)  # 默认使用ReLU


class ResBlock(nn.Module):
    def __init__(self, dim, activation="relu"):
        super(ResBlock, self).__init__()
        
        # 使用与原始实现相同的模块名称，以确保兼容性
        act = get_activation(activation)
        
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            act,
            nn.Conv2d(dim, dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(dim)
        )
        
        self.activation = act

    def forward(self, x):
        return x + self.block(x)
