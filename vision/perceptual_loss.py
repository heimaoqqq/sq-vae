import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class VGGPerceptualLoss(nn.Module):
    """
    基于VGG网络提取特征计算感知损失
    可以更好地保留图像的视觉质量和细节
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # 使用预训练的VGG16网络提取特征
        vgg = models.vgg16(pretrained=True).features[:23].eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.feature_extractor = vgg
        self.resize = resize
        
        # 标准化参数（ImageNet预训练模型使用的归一化参数）
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x, target, feature_layers=[0, 5, 10, 17, 22]):
        """
        计算输入图像和目标图像在VGG特征空间的差异
        
        参数:
            x: 输入图像张量 [B, C, H, W]
            target: 目标图像张量 [B, C, H, W]
            feature_layers: 使用的VGG层索引
        """
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        # 标准化处理
        x = (x - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # 如果需要，将输入调整为VGG期望的224x224大小
        if self.resize:
            x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        # 计算特征损失
        loss = 0.0
        features_x = x
        features_target = target
        
        # 逐层计算特征差异
        for i, layer in enumerate(self.feature_extractor):
            features_x = layer(features_x)
            features_target = layer(features_target)
            
            if i in feature_layers:
                loss += F.l1_loss(features_x, features_target)
        
        return loss


class SpectralLoss(nn.Module):
    """
    频谱损失，专门针对微多普勒时频图特性设计
    确保重建图像在频域上与原始图像相似
    """
    def __init__(self):
        super(SpectralLoss, self).__init__()
        
    def forward(self, x, target):
        """
        计算输入图像和目标图像在频域上的差异
        
        参数:
            x: 输入图像张量 [B, C, H, W]
            target: 目标图像张量 [B, C, H, W]
        """
        # 计算傅里叶变换
        x_fft = torch.fft.fft2(x)
        target_fft = torch.fft.fft2(target)
        
        # 计算幅度谱
        x_mag = torch.abs(x_fft)
        target_mag = torch.abs(target_fft)
        
        # 计算幅度谱差异
        mag_loss = F.mse_loss(x_mag, target_mag)
        
        # 计算相位谱
        x_phase = torch.angle(x_fft)
        target_phase = torch.angle(target_fft)
        
        # 计算相位差异，注意相位是周期性的
        phase_diff = torch.abs(torch.remainder(x_phase - target_phase + torch.pi, 2 * torch.pi) - torch.pi)
        phase_loss = phase_diff.mean()
        
        # 综合损失，幅度差异占主导
        return mag_loss + 0.1 * phase_loss 