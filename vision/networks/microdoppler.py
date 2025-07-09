from networks.net_64 import EncoderVqResnet64, DecoderVqResnet64
import torch.nn as nn
from networks.util import ResBlock, get_activation


class EncoderVq_resnet(EncoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        # 获取通道乘数和激活函数类型
        channel_multiplier = getattr(cfgs, "channel_multiplier", 1.0)
        activation_type = getattr(cfgs, "activation", "relu")
        
        # 应用通道乘数
        adjusted_dim_z = int(dim_z * channel_multiplier)
        
        # 保存激活函数类型供后续使用
        self.activation_type = activation_type
        
        # 调用父类构造函数
        super(EncoderVq_resnet, self).__init__(adjusted_dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"
        
        # 如果使用非默认激活函数，重新构建ResBlock部分
        if activation_type != "relu":
            # 获取ResBlock数量
            num_rb = getattr(cfgs, "num_rb", 6)  # 默认为6个ResBlock
            
            # 重建ResBlock层
            layers_resblocks = []
            for i in range(num_rb-1):
                layers_resblocks.append(ResBlock(adjusted_dim_z, activation=activation_type))
            self.res = nn.Sequential(*layers_resblocks)
            self.res_m = ResBlock(adjusted_dim_z, activation=activation_type)
            if self.flg_variance:
                self.res_v = ResBlock(adjusted_dim_z, activation=activation_type)


class DecoderVq_resnet(DecoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn):
        # 获取通道乘数和激活函数类型
        channel_multiplier = getattr(cfgs, "channel_multiplier", 1.0)
        activation_type = getattr(cfgs, "activation", "relu")
        
        # 应用通道乘数
        adjusted_dim_z = int(dim_z * channel_multiplier)
        
        # 保存激活函数类型供后续使用
        self.activation_type = activation_type
        
        # 调用父类构造函数
        super(DecoderVq_resnet, self).__init__(adjusted_dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler"
        
        # 如果使用非默认激活函数，重新构建ResBlock部分
        if activation_type != "relu":
            # 获取ResBlock数量
            num_rb = getattr(cfgs, "num_rb", 6)  # 默认为6个ResBlock
            
            # 重建ResBlock层
            layers_resblocks = []
            for i in range(num_rb-1):
                layers_resblocks.append(ResBlock(adjusted_dim_z, activation=activation_type))
            self.res = nn.Sequential(*layers_resblocks) 