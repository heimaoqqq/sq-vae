from networks.net_64 import EncoderVqResnet64, DecoderVqResnet64
import torch.nn as nn
from networks.util import ResBlock, get_activation


class EncoderVq_resnet(EncoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        # 获取通道乘数
        channel_multiplier = getattr(cfgs.network, "channel_multiplier", 1.0)
        adjusted_dim_z = int(dim_z * channel_multiplier)
        
        # 获取激活函数
        self.activation = getattr(cfgs.network, "activation", "relu")
        
        super(EncoderVq_resnet, self).__init__(adjusted_dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"
        
        # 如果指定了自定义激活函数，重新构建ResBlock
        if self.activation != "relu":
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
        # 获取通道乘数
        channel_multiplier = getattr(cfgs.network, "channel_multiplier", 1.0)
        adjusted_dim_z = int(dim_z * channel_multiplier)
        
        # 获取激活函数
        self.activation = getattr(cfgs.network, "activation", "relu")
        
        super(DecoderVq_resnet, self).__init__(adjusted_dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler"
        
        # 如果指定了自定义激活函数，重新构建ResBlock
        if self.activation != "relu":
            # 重新构建ResBlock部分
            num_rb = cfgs.network.num_rb
            layers_resblocks = []
            for i in range(num_rb-1):
                layers_resblocks.append(ResBlock(adjusted_dim_z, activation=self.activation))
            self.res = nn.Sequential(*layers_resblocks) 