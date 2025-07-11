import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import SQVAE, GaussianSQVAE, weights_init
from quantizer import GaussianVectorQuantizer, VmfVectorQuantizer

# 导入diffusers的组件
try:
    # 更新为使用新版diffusers库的导入路径
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    # 从AutoencoderKL获取编码器和解码器组件
    from diffusers.models.autoencoders.autoencoder_kl import Encoder, Decoder
    from diffusers.models.unets.unet_2d_condition import ResnetBlock2D, Downsample2D, Upsample2D
    
    DIFFUSERS_AVAILABLE = True
    print("成功导入diffusers库组件")
except ImportError as e:
    print(f"Diffusers库导入错误: {e}")
    
    # 尝试使用pip安装diffusers（适用于Kaggle环境）
    try:
        import sys
        import subprocess
        print("尝试安装diffusers库...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers", "--quiet"])
        print("diffusers库安装完成，尝试再次导入...")
        
        # 再次尝试导入
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL, Encoder, Decoder
        from diffusers.models.unets.unet_2d_condition import ResnetBlock2D, Downsample2D, Upsample2D
        DIFFUSERS_AVAILABLE = True
        print("成功导入diffusers库组件")
    except Exception as install_error:
        print(f"无法安装或导入diffusers库: {install_error}")
        DIFFUSERS_AVAILABLE = False
        print("将使用自定义实现替代diffusers组件")


class DiffusersEncoder(nn.Module):
    """
    基于diffusers的编码器架构，但修改为输出均值和方差
    """
    def __init__(self, in_channels=3, out_channels=64, down_block_types=None,
                 block_out_channels=None, layers_per_block=2, act_fn="silu",
                 norm_num_groups=32, flg_variance=True):
        super().__init__()
        self.flg_variance = flg_variance
        
        if down_block_types is None:
            down_block_types = ["DownEncoderBlock2D", "DownEncoderBlock2D"]
        if block_out_channels is None:
            block_out_channels = [128, 256]
            
        if DIFFUSERS_AVAILABLE:
            # 使用diffusers的Encoder但不包括最后的输出层
            self.encoder = Encoder(
                in_channels=in_channels,
                out_channels=out_channels,  # 临时值，后面会替换
                down_block_types=down_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                double_z=False  # 不使用double_z，我们会自己处理均值和方差
            )
            # 移除原始输出层
            self.encoder.conv_out = nn.Identity()
            
            # 添加自定义输出层
            last_channel = block_out_channels[-1]
            if self.flg_variance:
                # 输出均值和对数方差
                self.mean_out = nn.Conv2d(last_channel, out_channels, kernel_size=3, padding=1)
                self.logvar_out = nn.Conv2d(last_channel, out_channels, kernel_size=3, padding=1)
            else:
                # 只输出均值
                self.mean_out = nn.Conv2d(last_channel, out_channels, kernel_size=3, padding=1)
        else:
            # 如果没有diffusers，创建自定义编码器
            self._create_custom_encoder(in_channels, out_channels, block_out_channels, layers_per_block)
    
    def _create_custom_encoder(self, in_channels, out_channels, block_out_channels, layers_per_block):
        """创建自定义编码器，模仿diffusers的设计"""
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([])
        
        # 修复：确保通道数正确匹配
        current_channels = block_out_channels[0]
        
        for i in range(len(block_out_channels)):
            # 添加残差块
            for j in range(layers_per_block):
                # 第一层残差块输入通道是current_channels，输出通道是block_out_channels[i]
                if j == 0:
                    self.blocks.append(self._create_resnet_block(current_channels, block_out_channels[i]))
                else:
                    self.blocks.append(self._create_resnet_block(block_out_channels[i], block_out_channels[i]))
            
            # 更新当前通道数
            current_channels = block_out_channels[i]
            
            # 添加下采样层
            if i < len(block_out_channels) - 1:
                self.blocks.append(nn.Conv2d(current_channels, block_out_channels[i+1], kernel_size=4, stride=2, padding=1))
                self.blocks.append(nn.LeakyReLU(0.2))
                current_channels = block_out_channels[i+1]
        
        # 输出层
        last_channel = block_out_channels[-1]
        if self.flg_variance:
            # 输出均值和对数方差
            self.mean_out = nn.Conv2d(last_channel, out_channels, kernel_size=3, padding=1)
            self.logvar_out = nn.Conv2d(last_channel, out_channels, kernel_size=3, padding=1)
        else:
            # 只输出均值
            self.mean_out = nn.Conv2d(last_channel, out_channels, kernel_size=3, padding=1)
    
    def _create_resnet_block(self, in_channels, out_channels):
        """创建简单的残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        if DIFFUSERS_AVAILABLE:
            h = self.encoder(x)
            
            # 计算均值和方差
            if self.flg_variance:
                mean = self.mean_out(h)
                logvar = self.logvar_out(h)
                return mean, logvar
            else:
                mean = self.mean_out(h)
                return mean
        else:
            # 自定义编码器前向传播
            h = self.conv_in(x)
            for block in self.blocks:
                h = block(h)
            
            # 计算均值和方差
            if self.flg_variance:
                mean = self.mean_out(h)
                logvar = self.logvar_out(h)
                return mean, logvar
            else:
                mean = self.mean_out(h)
                return mean


class DiffusersDecoder(nn.Module):
    """
    基于diffusers的解码器架构
    """
    def __init__(self, in_channels=64, out_channels=3, up_block_types=None,
                 block_out_channels=None, layers_per_block=2, act_fn="silu",
                 norm_num_groups=32):
        super().__init__()
        
        if up_block_types is None:
            up_block_types = ["UpDecoderBlock2D", "UpDecoderBlock2D"]
        if block_out_channels is None:
            block_out_channels = [128, 256]
            
        if DIFFUSERS_AVAILABLE:
            # 使用diffusers的Decoder
            self.decoder = Decoder(
                in_channels=in_channels,
                out_channels=out_channels,
                up_block_types=up_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups
            )
        else:
            # 如果没有diffusers，创建自定义解码器
            self._create_custom_decoder(in_channels, out_channels, block_out_channels, layers_per_block)
    
    def _create_custom_decoder(self, in_channels, out_channels, block_out_channels, layers_per_block):
        """创建自定义解码器，模仿diffusers的设计"""
        # 反转通道列表以匹配上采样过程
        reversed_block_out_channels = list(reversed(block_out_channels))
        
        # 输入转换层
        self.conv_in = nn.Conv2d(in_channels, reversed_block_out_channels[0], kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([])
        
        # 跟踪当前通道数
        current_channels = reversed_block_out_channels[0]
        
        for i in range(len(reversed_block_out_channels)):
            # 添加残差块
            for j in range(layers_per_block):
                self.blocks.append(self._create_resnet_block(current_channels, reversed_block_out_channels[i]))
                current_channels = reversed_block_out_channels[i]
            
            # 添加上采样层
            if i < len(reversed_block_out_channels) - 1:
                next_channels = reversed_block_out_channels[i+1]
                self.blocks.append(nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1))
                self.blocks.append(nn.LeakyReLU(0.2))
                current_channels = next_channels
        
        # 输出层
        self.conv_out = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _create_resnet_block(self, in_channels, out_channels):
        """创建简单的残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, z):
        if DIFFUSERS_AVAILABLE:
            return self.decoder(z)
        else:
            # 自定义解码器前向传播
            h = self.conv_in(z)
            for block in self.blocks:
                h = block(h)
            return self.conv_out(h)


class DiffusersSQVAE(SQVAE):
    """
    结合diffusers的VQModel骨架架构和SQ-VAE的核心元素
    """
    def __init__(self, cfgs, flgs):
        nn.Module.__init__(self)  # 不调用SQVAE的初始化，我们将自己实现
        
        # 数据空间
        dataset = cfgs.dataset.name
        self.dim_x = cfgs.dataset.dim_x
        self.dataset = cfgs.dataset.name
        
        # SQ-VAE核心参数
        self.param_var_q = cfgs.model.param_var_q
        
        # 创建编码器和解码器
        encoder_config = getattr(cfgs.network, 'encoder', {})
        decoder_config = getattr(cfgs.network, 'decoder', {})
        
        # 从配置中获取参数，如果没有则使用默认值
        down_block_types = getattr(encoder_config, 'down_block_types', ["DownEncoderBlock2D", "DownEncoderBlock2D"])
        encoder_block_out_channels = getattr(encoder_config, 'block_out_channels', [128, 256])
        encoder_layers_per_block = getattr(encoder_config, 'layers_per_block', 2)
        
        up_block_types = getattr(decoder_config, 'up_block_types', ["UpDecoderBlock2D", "UpDecoderBlock2D"])
        decoder_block_out_channels = getattr(decoder_config, 'block_out_channels', [128, 256])
        decoder_layers_per_block = getattr(decoder_config, 'layers_per_block', 2)
        
        # 创建编码器和解码器
        self.encoder = DiffusersEncoder(
            in_channels=3,
            out_channels=cfgs.quantization.dim_dict,
            down_block_types=down_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            flg_variance=flgs.var_q
        )
        
        self.decoder = DiffusersDecoder(
            in_channels=cfgs.quantization.dim_dict,
            out_channels=3,
            up_block_types=up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block
        )
        
        # 应用权重初始化
        self.apply(weights_init)
        
        # 码本参数
        self.size_dict = cfgs.quantization.size_dict
        self.dim_dict = cfgs.quantization.dim_dict
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.log_param_q_scalar = nn.Parameter(torch.tensor(cfgs.model.log_param_q_init))
        
        # 创建量化器
        if self.param_var_q == "vmf":
            self.quantizer = VmfVectorQuantizer(
                self.size_dict, self.dim_dict, cfgs.quantization.temperature.init)
        else:
            self.quantizer = GaussianVectorQuantizer(
                self.size_dict, self.dim_dict, cfgs.quantization.temperature.init, self.param_var_q)
    
    def forward(self, x, flg_train=False, flg_quant_det=True):
        # 添加全局级别调试控制
        # 检查是否为主进程 (在分布式训练中很重要)
        is_main_process = True
        if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
            is_main_process = torch.distributed.get_rank() == 0
        
        # 仅在主进程中打印一次模型信息
        if is_main_process and not hasattr(DiffusersSQVAE, '_model_info_printed'):
            DiffusersSQVAE._model_info_printed = True
            print("\n===== 模型信息 =====")
            print(f"输入形状: {x.shape}")
            print(f"量化参数: {self.param_var_q}")
            print(f"量化器: {self.quantizer.__class__.__name__}")
            print("===================\n")
            
        # 编码
        try:
            if self.param_var_q == "vmf":
                z_from_encoder = F.normalize(self.encoder(x), p=2.0, dim=1)
                self.param_q = (self.log_param_q_scalar.exp() + torch.tensor([1.0], device=x.device))
            else:
                if self.param_var_q == "gaussian_1":
                    z_from_encoder = self.encoder(x)
                    log_var_q = torch.tensor([0.0], device=x.device)
                else:
                    z_from_encoder, log_var = self.encoder(x)
                    if self.param_var_q == "gaussian_2":
                        log_var_q = log_var.mean(dim=(1,2,3), keepdim=True)
                    elif self.param_var_q == "gaussian_3":
                        log_var_q = log_var.mean(dim=1, keepdim=True)
                    elif self.param_var_q == "gaussian_4":
                        log_var_q = log_var
                    else:
                        raise Exception(f"未定义的param_var_q: {self.param_var_q}")
                self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
            
            # 仅在主进程中打印一次形状信息
            if is_main_process and not hasattr(DiffusersSQVAE, '_shape_info_printed'):
                DiffusersSQVAE._shape_info_printed = True
                print("\n===== 形状信息 =====")
                print(f"输入: {x.shape}")
                print(f"潜在向量: {z_from_encoder.shape}")
                
                # 量化
                z_quantized, loss_latent, perplexity = self.quantizer(
                    z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det)
                
                # 解码
                x_reconst = self.decoder(z_quantized)
                print(f"重建输出: {x_reconst.shape}")
                print("====================\n")
                loss = self._calc_loss(x_reconst, x, loss_latent)
                loss["perplexity"] = perplexity
                return x_reconst, dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized), loss
            
            # 正常前向传播
            z_quantized, loss_latent, perplexity = self.quantizer(
                z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det)
            latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)

            # 解码
            x_reconst = self.decoder(z_quantized)
            
            # 损失
            loss = self._calc_loss(x_reconst, x, loss_latent)
            loss["perplexity"] = perplexity
            
            return x_reconst, latents, loss
        except Exception as e:
            import traceback
            print(f"DiffusersSQVAE前向传播错误: {e}")
            print(f"输入形状: {x.shape}")
            print(f"量化参数: {self.param_var_q}")
            traceback.print_exc()
            raise


class DiffusersGaussianSQVAE(DiffusersSQVAE):
    """
    使用Gaussian分布的DiffusersSQVAE
    """
    def __init__(self, cfgs, flgs):
        super(DiffusersGaussianSQVAE, self).__init__(cfgs, flgs)
        self.flg_arelbo = flgs.arelbo
        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))
    
    def _calc_loss(self, x_reconst, x, loss_latent):
        bs = x.shape[0]
        # 重建损失 - 添加数值稳定性检查
        try:
            # 检查输入数据有效性
            if torch.isnan(x_reconst).any() or torch.isinf(x_reconst).any():
                print("警告: 重建图像包含NaN/Inf，将替换为随机值")
                x_reconst = torch.where(torch.isnan(x_reconst) | torch.isinf(x_reconst),
                                      torch.rand_like(x_reconst), x_reconst)
                
            mse = F.mse_loss(x_reconst, x, reduction="none")
            
            # 检查是否有异常值
            if torch.isnan(mse).any() or torch.isinf(mse).any():
                print("警告: MSE计算产生NaN/Inf值，将替换为合理值")
                mse = torch.where(torch.isnan(mse) | torch.isinf(mse),
                                torch.tensor(1.0, device=mse.device), mse)
                
            # 使用更稳定的方法计算均值
            mse_sum = mse.sum() / bs
            
            # 最终检查
            if torch.isnan(mse_sum) or torch.isinf(mse_sum):
                print("警告: MSE总和为NaN/Inf，使用默认值0.1")
                mse_sum = torch.tensor(0.1, device=x.device)
        except Exception as e:
            print(f"MSE计算错误: {e}")
            mse_sum = torch.tensor(0.1, device=x.device)
            
        # 计算最终损失
        try:
            if self.flg_arelbo:
                # 使用arelbo损失
                # 限制mse_sum的范围以避免log(0)
                mse_sum = torch.clamp(mse_sum, min=1e-10, max=1e10)
                loss_reconst = self.dim_x * torch.log(mse_sum) / 2
                
                # 检查arelbo损失
                if torch.isnan(loss_reconst) or torch.isinf(loss_reconst):
                    print("警告: ARELBO损失为NaN/Inf，使用MSE作为替代")
                    loss_reconst = mse_sum
            else:
                # 确保logvar_x不会导致数值问题
                logvar_x_safe = torch.clamp(self.logvar_x, min=-10, max=10)
                var_x = torch.exp(logvar_x_safe)
                
                # 使用稳定的计算方法
                loss_reconst = mse_sum / (2*var_x) + self.dim_x * logvar_x_safe / 2
                
                # 检查传统损失
                if torch.isnan(loss_reconst) or torch.isinf(loss_reconst):
                    print("警告: 重建损失为NaN/Inf，使用MSE作为替代")
                    loss_reconst = mse_sum
        except Exception as e:
            print(f"重建损失计算错误: {e}")
            loss_reconst = mse_sum
            
        # 确保潜在损失有效
        if torch.isnan(loss_latent) or torch.isinf(loss_latent):
            print("警告: 潜在损失为NaN/Inf，设置为0")
            loss_latent = torch.tensor(0.0, device=x.device)
            
        # 总损失
        loss_all = loss_reconst + loss_latent
        
        # 最终安全检查
        if torch.isnan(loss_all) or torch.isinf(loss_all):
            print("警告: 总损失为NaN/Inf，设置为MSE")
            loss_all = mse_sum
            
        loss = dict(all=loss_all, mse=mse_sum)
        
        return loss 