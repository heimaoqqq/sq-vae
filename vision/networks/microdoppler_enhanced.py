import torch
from torch import nn
import torch.nn.functional as F
from networks.util import ResBlock
from networks.enhancer import MicroDopplerEnhancer, ResidualDetailEnhancer
from networks.net_64 import EncoderVqResnet64, DecoderVqResnet64

class EncoderVq_enhanced_resnet(EncoderVqResnet64):
    """
    使用标准编码器结构，保持与原始网络一致
    """
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_enhanced_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"


class DecoderVq_enhanced_resnet(nn.Module):
    """
    针对微多普勒时频图优化的增强型解码器
    在标准解码器基础上添加了细节增强模块
    """
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVq_enhanced_resnet, self).__init__()
        
        # 获取增强器类型配置（如果存在）
        enhancer_type = getattr(cfgs, 'enhancer_type', 'microdoppler')
        
        # 使用标准解码器的ResBlocks部分
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        
        # 标准解码器的上采样部分
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)
        
        # 添加细节增强模块
        if enhancer_type == 'microdoppler':
            self.detail_enhancer = MicroDopplerEnhancer()
        else:
            self.detail_enhancer = ResidualDetailEnhancer(channels=3)
            
        self.dataset = "MicroDoppler"
        
    def forward(self, z):
        # 基本解码过程
        out_res = self.res(z)
        base_output = self.convt(out_res)
        
        # 应用细节增强
        enhanced_output = self.detail_enhancer(base_output)
        return enhanced_output 