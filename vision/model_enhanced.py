import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model import SQVAE, GaussianSQVAE, weights_init
from perceptual_loss import VGGPerceptualLoss, SpectralLoss
from quantizer import GaussianVectorQuantizer, VmfVectorQuantizer
import networks.microdoppler_enhanced as net_microdoppler_enhanced


class EnhancedSQVAE(SQVAE):
    """
    增强型SQ-VAE基类，添加感知损失支持和增强型解码器
    """
    def __init__(self, cfgs, flgs):
        super(EnhancedSQVAE, self).__init__(cfgs, flgs)
        
        # 获取是否使用感知损失的配置
        self.use_perceptual = getattr(cfgs.model, 'use_perceptual', False)
        self.use_spectral = getattr(cfgs.model, 'use_spectral', False)
        
        # 如果启用感知损失，初始化感知损失模块
        if self.use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss(resize=True).cuda()
            self.perceptual_weight = getattr(cfgs.model, 'perceptual_weight', 0.1)
            print(f"使用VGG感知损失，权重={self.perceptual_weight}")
            
        # 如果启用频谱损失，初始化频谱损失模块
        if self.use_spectral:
            self.spectral_loss = SpectralLoss().cuda()
            self.spectral_weight = getattr(cfgs.model, 'spectral_weight', 0.1)
            print(f"使用频谱损失，权重={self.spectral_weight}")


class EnhancedGaussianSQVAE(EnhancedSQVAE):
    """
    增强型高斯SQ-VAE，结合感知损失和微多普勒增强功能
    """
    def __init__(self, cfgs, flgs):
        super(EnhancedGaussianSQVAE, self).__init__(cfgs, flgs)
        self.flg_arelbo = flgs.arelbo
        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))
    
    def _calc_loss(self, x_reconst, x, loss_latent):
        bs = x.shape[0]
        # 基本MSE损失
        mse = F.mse_loss(x_reconst, x, reduction="sum") / bs
        
        # 根据模型配置计算重建损失
        if self.flg_arelbo:
            loss_reconst = self.dim_x * torch.log(mse) / 2
        else:
            loss_reconst = mse / (2*self.logvar_x.exp()) + self.dim_x * self.logvar_x / 2
        
        # 添加感知损失
        p_loss = torch.tensor(0.0).to(x.device)
        if hasattr(self, 'perceptual_loss'):
            p_loss = self.perceptual_loss(x_reconst, x)
            loss_reconst = loss_reconst + self.perceptual_weight * p_loss
        
        # 添加频谱损失
        s_loss = torch.tensor(0.0).to(x.device)
        if hasattr(self, 'spectral_loss'):
            s_loss = self.spectral_loss(x_reconst, x)
            loss_reconst = loss_reconst + self.spectral_weight * s_loss
        
        # 总损失 = 重建损失 + 潜在损失
        loss_all = loss_reconst + loss_latent
        
        # 构建损失字典，包含各项损失值便于监控
        loss = dict(all=loss_all, mse=mse)
        
        # 如果使用了感知损失，记录它
        if hasattr(self, 'perceptual_loss'):
            loss['perceptual'] = p_loss  # 保持张量形式，不调用.item()
            
        # 如果使用了频谱损失，记录它
        if hasattr(self, 'spectral_loss'):
            loss['spectral'] = s_loss  # 保持张量形式，不调用.item()

        return loss 