import torch
from torch import nn
import torch.nn.functional as F
from networks.util import ResBlock


class EncoderVq_midres(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVq_midres, self).__init__()
        self.flg_variance = flg_var_q
        # 初始卷积层
        layers_conv = []
        # 第一层：256x256 -> 128x128
        layers_conv.append(nn.Sequential(nn.Conv2d(3, dim_z // 4, 4, stride=2, padding=1)))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 4))
        layers_conv.append(nn.ReLU())
        
        # 第二层：128x128 -> 64x64
        layers_conv.append(nn.Conv2d(dim_z // 4, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        
        # 最后一层卷积，保持空间维度
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        
        self.conv = nn.Sequential(*layers_conv)
        
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVq_midres(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVq_midres, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        
        # 转置卷积层
        layers_convt = []
        # 初始卷积，保持空间维度
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        
        # 第一层上采样：64x64 -> 128x128
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 4, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 4))
        layers_convt.append(nn.ReLU())
        
        # 第二层上采样：128x128 -> 256x256
        layers_convt.append(nn.ConvTranspose2d(dim_z // 4, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        
        self.convt = nn.Sequential(*layers_convt)
        
    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out


# 导出接口函数，与原来的microdoppler.py保持一致的接口
class EncoderVq_midres_resnet(EncoderVq_midres):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_midres_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"


class DecoderVq_midres_resnet(DecoderVq_midres):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_midres_resnet, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler" 