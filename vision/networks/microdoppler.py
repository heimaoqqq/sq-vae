from networks.net_64 import EncoderVqResnet64, DecoderVqResnet64
import torch.nn as nn
from networks.util import ResBlock, get_activation


class EncoderVq_resnet(EncoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        # 安全获取通道乘数，确保cfgs有network属性
        channel_multiplier = 1.0
        activation = "relu"
        
        # 检查cfgs是否包含network属性
        if hasattr(cfgs, 'network'):
            channel_multiplier = getattr(cfgs.network, "channel_multiplier", 1.0)
            activation = getattr(cfgs.network, "activation", "relu")
        
        adjusted_dim_z = int(dim_z * channel_multiplier)
        self.activation = activation
        
        super(EncoderVq_resnet, self).__init__(adjusted_dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"
        
        # 如果指定了自定义激活函数，重新构建ResBlock
        if self.activation != "relu" and hasattr(cfgs, 'network') and hasattr(cfgs.network, 'num_rb'):
            # 重新构建ResBlock部分
            num_rb = cfgs.network.num_rb
            layers_resblocks = []
            for i in range(num_rb-1):
                layers_resblocks.append(ResBlock(adjusted_dim_z, activation=self.activation))
            self.res = nn.Sequential(*layers_resblocks)
            self.res_m = ResBlock(adjusted_dim_z, activation=self.activation)
            if self.flg_variance:
                self.res_v = ResBlock(adjusted_dim_z, activation=self.activation)


class DecoderVq_resnet(DecoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn):
        # 安全获取通道乘数，确保cfgs有network属性
        channel_multiplier = 1.0
        activation = "relu"
        
        # 检查cfgs是否包含network属性
        if hasattr(cfgs, 'network'):
            channel_multiplier = getattr(cfgs.network, "channel_multiplier", 1.0)
            activation = getattr(cfgs.network, "activation", "relu")
        
        adjusted_dim_z = int(dim_z * channel_multiplier)
        self.activation = activation
        
        super(DecoderVq_resnet, self).__init__(adjusted_dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler"
        
        # 如果指定了自定义激活函数，重新构建ResBlock
        if self.activation != "relu" and hasattr(cfgs, 'network') and hasattr(cfgs.network, 'num_rb'):
            # 重新构建ResBlock部分
            num_rb = cfgs.network.num_rb
            layers_resblocks = []
            for i in range(num_rb-1):
                layers_resblocks.append(ResBlock(adjusted_dim_z, activation=self.activation))
            self.res = nn.Sequential(*layers_resblocks) 