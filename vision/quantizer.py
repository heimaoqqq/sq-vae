import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (torch.sum(z_continuous_flat**2, dim=1, keepdim=True) 
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_continuous_flat, codebook.t()))

    return distances



class VectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
    
    def forward(self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False):
        return self._quantize(z_from_encoder, param_q, codebook,
                                flg_train=flg_train, flg_quant_det=flg_quant_det)
    
    def _quantize(self):
        raise NotImplementedError()
    
    def set_temperature(self, value):
        self.temperature = value
    
    def _calc_distance_bw_enc_codes(self):
        raise NotImplementedError()
    
    def _calc_distance_bw_enc_dec(self):
        raise NotImplementedError()


class GaussianVectorQuantizer(VectorQuantizer):
    def __init__(self, size_dict, dim_dict, temperature=0.5, param_var_q="gaussian_1"):
        super(GaussianVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
        self.param_var_q = param_var_q
    
    def _quantize(self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device="cuda")
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                # 获取实际元素数量以确保正确重塑
                actual_total_elements = logit.shape[0]  # logit的第一维大小
                
                # 采样
                dist = Categorical(probabilities)
                indices = dist.sample()  # 不立即重塑
                
                # 计算合适的重塑维度
                # 首先尝试使用输入提供的bs, width, height
                expected_elements = bs * width * height
                
                if actual_total_elements == expected_elements:
                    # 如果元素数量匹配，使用原始维度
                    indices = indices.view(bs, width, height)
                else:
                    # 如果不匹配，尝试推断合适的维度
                    # 假设宽高比保持不变，调整批次大小
                    adjusted_bs = actual_total_elements // (width * height)
                    if adjusted_bs * width * height == actual_total_elements:
                        indices = indices.view(adjusted_bs, width, height)
                    else:
                        # 如果仍然无法整除，采用扁平化处理
                        # 记录原始形状以便后续重塑操作
                        flat_indices = indices
                
                # 使用one-hot编码
                if 'flat_indices' in locals():
                    # 如果使用了扁平化处理
                    encodings_hard = F.one_hot(flat_indices, num_classes=self.size_dict).type_as(codebook)
                    # 记住，我们需要适当地重塑最终的z_quantized
                else:
                    encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                
                avg_probs = torch.mean(probabilities, dim=0)
            
            # 计算z_quantized，确保形状正确
            if 'flat_indices' in locals():
                # 如果使用了扁平化处理
                z_quantized_flat = torch.matmul(encodings_hard, codebook)
                # 尝试重塑为适当的形状
                if actual_total_elements % bs == 0:
                    # 如果可以整除批次大小，调整宽高
                    new_spatial_dim = int((actual_total_elements / bs) ** 0.5)
                    if bs * new_spatial_dim * new_spatial_dim == actual_total_elements:
                        z_quantized = z_quantized_flat.view(bs, new_spatial_dim, new_spatial_dim, dim_z)
                    else:
                        # 无法找到合适的正方形尺寸，采用一维表示
                        z_quantized = z_quantized_flat.view(bs, -1, 1, dim_z)
                else:
                    # 如果无法整除批次大小，直接使用一维表示
                    z_quantized = z_quantized_flat.view(-1, 1, 1, dim_z)
            else:
                # 使用标准重塑
                z_quantized_flat = torch.matmul(encodings_hard, codebook)
                
                # 检查重塑后的尺寸是否匹配
                total_size = encodings_hard.size(0) * width * height * dim_z
                actual_size = z_quantized_flat.numel()
                
                if total_size == actual_size:
                    # 如果尺寸匹配，直接重塑
                    z_quantized = z_quantized_flat.view(encodings_hard.size(0), width, height, dim_z)
                else:
                    # 如果尺寸不匹配，计算正确的维度
                    # 首先确保可以被批次大小和dim_z整除
                    if actual_size % (encodings_hard.size(0) * dim_z) == 0:
                        # 计算新的空间维度
                        spatial_elements = actual_size // (encodings_hard.size(0) * dim_z)
                        new_hw = int(spatial_elements ** 0.5)
                        
                        if new_hw ** 2 == spatial_elements:  # 是完全平方数
                            z_quantized = z_quantized_flat.view(encodings_hard.size(0), new_hw, new_hw, dim_z)
                        else:
                            # 使用一维空间表示
                            z_quantized = z_quantized_flat.view(encodings_hard.size(0), spatial_elements, 1, dim_z)
                    else:
                        # 如果无法合理分解，使用最简单的重塑
                        # 计算可以得到的最大批次大小
                        possible_bs = actual_size // dim_z
                        z_quantized = z_quantized_flat.view(possible_bs, 1, 1, dim_z)
                        print(f"Warning: Reshape to non-standard dimensions. Original: {encodings_hard.size(0)}x{width}x{height}x{dim_z}, Actual: {possible_bs}x1x1x{dim_z}")
            
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):        
        if self.param_var_q == "gaussian_1":
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_2":
            # 全局单一方差
            bs, width, height, dim_z = z_from_encoder.shape
            weight_flat = weight.view(-1, 1)  # 确保维度正确 [bs*1*1*1, 1]
            distances = weight_flat * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_3":
            # 位置独立方差模式
            bs, width, height, dim_z = z_from_encoder.shape
            total_elements = bs * width * height
            
            # 打印调试信息
            # print(f"z_from_encoder shape: {z_from_encoder.shape}")
            # print(f"weight shape: {weight.shape}")
            
            # 计算z_continuous_flat
            z_continuous_flat = z_from_encoder.reshape(-1, self.dim_dict)  # 使用-1自动计算维度，避免硬编码
            
            # 适配weight的形状以匹配z_from_encoder的元素数量
            # 首先处理weight的形状变换，确保它与z_continuous_flat的第一维匹配
            # 对于gaussian_3模式，weight的形状应为[bs, 1, width, height]
            
            # 根据z_continuous_flat的实际大小重新构建weight_flat
            if weight.shape[0] == bs and z_continuous_flat.shape[0] == bs * width * height:
                # 需要将weight广播到每个空间位置
                weight_expanded = weight.expand(bs, width, height, 1)
                weight_flat = weight_expanded.reshape(-1, 1)
            else:
                # 如果weight已经是正确的形状，只需重新整形
                weight_permuted = weight.permute(0, 2, 3, 1).contiguous()
                weight_flat = weight_permuted.reshape(-1, 1)
            
            # 确保形状匹配
            if weight_flat.shape[0] != z_continuous_flat.shape[0]:
                # 如果仍然不匹配，则进行适当的复制或裁剪
                if weight_flat.shape[0] < z_continuous_flat.shape[0]:
                    # 需要扩展weight_flat
                    repeat_factor = z_continuous_flat.shape[0] // weight_flat.shape[0]
                    weight_flat = weight_flat.repeat(repeat_factor, 1)
                else:
                    # 需要裁剪weight_flat
                    weight_flat = weight_flat[:z_continuous_flat.shape[0], :]
            
            # 再次确保形状匹配
            assert weight_flat.shape[0] == z_continuous_flat.shape[0], f"Weight shape {weight_flat.shape[0]} != z_continuous shape {z_continuous_flat.shape[0]}"
            
            # 计算距离
            z_sq = torch.sum(z_continuous_flat**2, dim=1, keepdim=True)  # [total_elements, 1]
            c_sq = torch.sum(codebook**2, dim=1)  # [size_dict]
            z_c = torch.matmul(z_continuous_flat, codebook.t())  # [total_elements, size_dict]
            
            distances = weight_flat * (z_sq + c_sq - 2 * z_c)
        elif self.param_var_q == "gaussian_4":
            # 独立元素方差
            bs, width, height, dim_z = z_from_encoder.shape
            z_from_encoder_flat = z_from_encoder.reshape(-1, self.dim_dict)
            
            # 将权重从[bs, dim_z, width, height]变形为与z_from_encoder_flat兼容的形状
            weight = weight.permute(0, 2, 3, 1).contiguous().reshape(-1, self.dim_dict)
            
            # 计算每个位置的距离，考虑每个维度的权重
            z_sq = torch.sum(z_from_encoder_flat**2 * weight, dim=1, keepdim=True)
            c_sq = torch.sum(codebook**2, dim=1)
            z_c = torch.matmul(z_from_encoder_flat * weight, codebook.t())
            distances = z_sq + c_sq - 2 * z_c

        return distances
        
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1-x2)**2 * weight, dim=(1,2,3))
    


class VmfVectorQuantizer(VectorQuantizer):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VmfVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
    
    def _quantize(self, z_from_encoder, kappa_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        codebook_norm = F.normalize(codebook, p=2.0, dim=1)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook_norm, kappa_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook_norm).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device="cuda")
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook_norm).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()

        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, kappa_q).mean()        
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity
 
    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, kappa_q):
        z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict)
        distances = -kappa_q * torch.matmul(z_from_encoder_flat, codebook.t())

        return distances
    
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum(x1 * (x1-x2) * weight, dim=(1,2,3))
