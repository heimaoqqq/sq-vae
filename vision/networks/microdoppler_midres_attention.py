import torch
from torch import nn
import torch.nn.functional as F
from networks.util import ResBlock

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, in_channels, out_channels=None, heads=8):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        
        # 计算每个头的维度
        self.head_channels = self.out_channels // heads
        
        # 定义查询、键、值的线性层
        self.q = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        
        # 输出投影
        self.proj = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        
        self.scale = (self.head_channels) ** -0.5
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 生成查询、键、值
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # 重塑形状用于多头注意力 [b, c, h, w] -> [b, heads, head_channels, h*w]
        q = q.reshape(b, self.heads, self.head_channels, h*w)
        k = k.reshape(b, self.heads, self.head_channels, h*w)
        v = v.reshape(b, self.heads, self.head_channels, h*w)
        
        # 计算注意力
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, heads, h*w, h*w]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)  # [b, heads, h*w, head_channels]
        out = out.reshape(b, self.out_channels, h, w)
        
        # 输出投影
        out = self.proj(out)
        
        return out

class EncoderVq_midres_attention(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVq_midres_attention, self).__init__()
        self.flg_variance = flg_var_q
        
        # 初始卷积层
        layers_conv = []
        # 第一层：256x256 -> 128x128
        layers_conv.append(nn.Sequential(nn.Conv2d(3, dim_z // 4, 4, stride=2, padding=1)))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 4))
        layers_conv.append(nn.ReLU())
        
        # 添加通道注意力
        layers_conv.append(ChannelAttention(dim_z // 4))
        
        # 第二层：128x128 -> 64x64
        layers_conv.append(nn.Conv2d(dim_z // 4, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        
        # 添加通道注意力和空间注意力
        layers_conv.append(ChannelAttention(dim_z))
        layers_conv.append(SpatialAttention())
        
        # 最后一层卷积，保持空间维度
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        
        self.conv = nn.Sequential(*layers_conv)
        
        # 添加自注意力层
        self.self_attention = SelfAttention(dim_z)
        
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
        out_attn = self.self_attention(out_conv)
        out_res = self.res(out_attn)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVq_midres_attention(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVq_midres_attention, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        
        # 添加自注意力层
        self.self_attention = SelfAttention(dim_z)
        
        # 转置卷积层
        layers_convt = []
        # 初始卷积，保持空间维度
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        
        # 添加通道注意力
        layers_convt.append(ChannelAttention(dim_z))
        
        # 第一层上采样：64x64 -> 128x128
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 4, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 4))
        layers_convt.append(nn.ReLU())
        
        # 添加通道注意力
        layers_convt.append(ChannelAttention(dim_z // 4))
        
        # 第二层上采样：128x128 -> 256x256
        layers_convt.append(nn.ConvTranspose2d(dim_z // 4, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        
        self.convt = nn.Sequential(*layers_convt)
        
    def forward(self, z):
        out_res = self.res(z)
        out_attn = self.self_attention(out_res)
        out = self.convt(out_attn)
        return out


# 导出接口函数，与原来的microdoppler.py保持一致的接口
class EncoderVq_midres_attn_resnet(EncoderVq_midres_attention):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_midres_attn_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"


class DecoderVq_midres_attn_resnet(DecoderVq_midres_attention):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_midres_attn_resnet, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler" 