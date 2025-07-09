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
        
        # 添加数值稳定性检查
        # 1. 确保var_q没有无效值
        if torch.isnan(var_q).any() or torch.isinf(var_q).any():
            print("警告: 检测到var_q中有NaN或Inf值，将替换为默认值")
            var_q = torch.where(torch.isnan(var_q) | torch.isinf(var_q), 
                               torch.tensor(1.0, device=var_q.device), var_q)
        
        # 2. 确保z_from_encoder没有无效值
        if torch.isnan(z_from_encoder).any() or torch.isinf(z_from_encoder).any():
            print("警告: 检测到z_from_encoder中有NaN或Inf值，将替换为0")
            z_from_encoder = torch.where(torch.isnan(z_from_encoder) | torch.isinf(z_from_encoder),
                                       torch.tensor(0.0, device=z_from_encoder.device), z_from_encoder)
            z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        
        # 安全地计算精度，防止除以零
        precision_q = 1. / torch.clamp(var_q, min=1e-5)
        
        # 限制精度上限，防止数值溢出
        precision_q = torch.clamp(precision_q, max=1e5)

        # 计算距离并应用精度
        try:
            logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
            
            # 检查logit是否包含NaN或Inf
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                print("警告: 距离计算产生NaN/Inf，使用替代值")
                # 使用接近于零的小值替代无效值
                logit = torch.where(torch.isnan(logit) | torch.isinf(logit),
                                  torch.tensor(0.0, device=logit.device), logit)
        except Exception as e:
            print(f"距离计算错误: {e}")
            # 创建安全的默认logit
            logit = torch.zeros((z_from_encoder_permuted.shape[0] * z_from_encoder_permuted.shape[1] * 
                                z_from_encoder_permuted.shape[2], self.size_dict), device=z_from_encoder.device)
        
        # 添加数值稳定性
        # 限制logit范围，防止softmax溢出
        logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
        logit = logit - logit_max  # 数值稳定性技巧
        logit = torch.clamp(logit, min=-50.0, max=50.0)  # 进一步限制范围
        
        # 使用更稳定的softmax计算
        try:
            probabilities = F.softmax(logit, dim=-1)
            log_probabilities = F.log_softmax(logit, dim=-1)
            
            # 确保概率和为1且不含NaN
            if torch.isnan(probabilities).any() or torch.sum(probabilities, dim=-1).min() < 0.99:
                print("警告: 检测到概率计算中的NaN值或概率和不为1，将使用均匀概率")
                probabilities = torch.ones_like(probabilities) / self.size_dict
                log_probabilities = torch.log(probabilities + 1e-10)
        except Exception as e:
            print(f"概率计算错误: {e}")
            # 创建安全的均匀分布
            probabilities = torch.ones_like(logit) / self.size_dict
            log_probabilities = torch.log(probabilities + 1e-10)
        
        # Quantization
        if flg_train:
            try:
                encodings = gumbel_softmax_sample(logit, self.temperature)
                # 确保encodings的有效性
                if torch.isnan(encodings).any() or torch.isinf(encodings).any():
                    print("警告: Gumbel softmax产生无效值，使用标准softmax")
                    encodings = F.softmax(logit / max(self.temperature, 1e-5), dim=-1)
                    
                z_quantized = torch.matmul(encodings, codebook)
                z_quantized = z_quantized.view(bs, width, height, dim_z)
                avg_probs = torch.mean(probabilities.detach(), dim=0)
            except Exception as e:
                print(f"量化编码错误: {e}")
                # 使用更安全的方法
                encodings = F.softmax(logit / max(self.temperature, 1e-5), dim=-1)
                z_quantized = torch.matmul(encodings, codebook).view(bs, width, height, dim_z)
                avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                try:
                    indices = torch.argmax(logit, dim=1).unsqueeze(1)
                    encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=z_from_encoder.device)
                    encodings_hard.scatter_(1, indices, 1)
                    avg_probs = torch.mean(encodings_hard, dim=0)
                    z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
                except Exception as e:
                    print(f"确定性量化错误: {e}")
                    # 安全回退：使用最简单的方式
                    uniform_indices = torch.randint(0, self.size_dict, (indices.shape[0], 1), device=z_from_encoder.device)
                    encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=z_from_encoder.device)
                    encodings_hard.scatter_(1, uniform_indices, 1)
                    z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
                    avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                try:
                    # 添加安全检查
                    valid_probs = torch.isfinite(probabilities).all(dim=1)
                    if not valid_probs.all():
                        print(f"警告: 检测到{(~valid_probs).sum().item()}个样本的概率分布无效，使用均匀分布替代")
                        # 对于无效的概率行，使用均匀分布
                        uniform_probs = torch.ones((~valid_probs).sum().item(), self.size_dict, 
                                                 device=probabilities.device) / self.size_dict
                        probabilities[~valid_probs] = uniform_probs
                    
                    # 确保概率和为1
                    prob_sum = probabilities.sum(dim=1, keepdim=True)
                    probabilities = probabilities / torch.clamp(prob_sum, min=1e-10)
                    
                    # 使用安全的采样方法
                    try:
                        dist = Categorical(probabilities)
                        indices = dist.sample().view(bs, width, height)
                    except Exception as e:
                        print(f"分类采样错误: {e}，使用argmax替代")
                        indices = torch.argmax(probabilities, dim=1).view(bs, width, height)
                        
                    encodings_hard = F.one_hot(indices, num_classes=self.size_dict).float()
                    avg_probs = torch.mean(probabilities, dim=0)
                    z_quantized = torch.matmul(encodings_hard.view(-1, self.size_dict), codebook).view(bs, width, height, dim_z)
                except Exception as e:
                    print(f"随机量化错误: {e}")
                    # 安全回退：使用均匀采样
                    indices = torch.randint(0, self.size_dict, (bs, width, height), device=z_from_encoder.device)
                    encodings_hard = F.one_hot(indices, num_classes=self.size_dict).float()
                    z_quantized = torch.matmul(encodings_hard.view(-1, self.size_dict), codebook).view(bs, width, height, dim_z)
                    avg_probs = torch.ones(self.size_dict, device=z_from_encoder.device) / self.size_dict
        
        # 安全地排列张量
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # 检查量化后的张量是否有效
        if torch.isnan(z_to_decoder).any() or torch.isinf(z_to_decoder).any():
            print("警告: 量化后的张量包含NaN/Inf，替换为随机值")
            z_to_decoder = torch.where(torch.isnan(z_to_decoder) | torch.isinf(z_to_decoder),
                                     torch.randn_like(z_to_decoder) * 0.01, z_to_decoder)
        
        # Latent loss - 添加数值稳定性
        try:
            kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
            # 确保kld_discrete不是NaN
            if torch.isnan(kld_discrete) or torch.isinf(kld_discrete):
                print("警告: KLD离散项计算出NaN/Inf，设置为0")
                kld_discrete = torch.tensor(0.0, device=z_from_encoder.device)
        except Exception as e:
            print(f"KLD离散项计算错误: {e}")
            kld_discrete = torch.tensor(0.0, device=z_from_encoder.device)
        
        try:    
            kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
            # 确保kld_continuous不是NaN
            if torch.isnan(kld_continuous) or torch.isinf(kld_continuous):
                print("警告: KLD连续项计算出NaN/Inf，设置为0")
                kld_continuous = torch.tensor(0.0, device=z_from_encoder.device)
        except Exception as e:
            print(f"KLD连续项计算错误: {e}")
            kld_continuous = torch.tensor(0.0, device=z_from_encoder.device)
            
        loss = kld_discrete + kld_continuous
        
        # 确保loss是有限的
        if torch.isnan(loss) or torch.isinf(loss):
            print("警告: 总损失为NaN/Inf，设置为小的正值")
            loss = torch.tensor(0.1, device=z_from_encoder.device)
        
        # 确保avg_probs不含NaN且非零
        avg_probs = torch.clamp(avg_probs, min=1e-10)
        # 安全地计算perplexity
        try:
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            if torch.isnan(perplexity) or torch.isinf(perplexity):
                perplexity = torch.tensor(float(self.size_dict), device=z_from_encoder.device)
        except Exception as e:
            print(f"困惑度计算错误: {e}")
            perplexity = torch.tensor(float(self.size_dict), device=z_from_encoder.device)

        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):        
        if self.param_var_q == "gaussian_1":
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_2":
            # 获取实际特征图大小
            bs, width, height, dim_z = z_from_encoder.shape
            # 动态调整weight的大小以匹配特征图
            weight = weight.tile(1, 1, width, height).view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == "gaussian_3":
            # 在gaussian_3中，权重是在通道维度上计算均值
            # 从错误信息可以看出，weight的形状为[bs, 1, width, height]
            # 而不是我们之前假设的[bs, 1, 1, 1]
            bs, width, height, dim_z = z_from_encoder.shape
            
            # 将权重从[bs, 1, width, height]转换为[bs*width*height, 1]
            # 首先将通道维度放到最后
            weight_permuted = weight.permute(0, 2, 3, 1).contiguous()  # [bs, width, height, 1]
            weight_flat = weight_permuted.view(-1, 1)  # [bs*width*height, 1]
            
            # 计算z_continuous_flat
            z_continuous_flat = z_from_encoder.view(-1, self.dim_dict)
            
            # 计算距离
            z_sq = torch.sum(z_continuous_flat**2, dim=1, keepdim=True)
            c_sq = torch.sum(codebook**2, dim=1)
            z_c = torch.matmul(z_continuous_flat, codebook.t())
            distances = weight_flat * (z_sq + c_sq - 2 * z_c)
        elif self.param_var_q == "gaussian_4":
            # 在gaussian_4中，权重直接使用网络预测的每个位置的方差
            # 我们需要将z_from_encoder和权重重塑为适当的形状进行计算
            bs, width, height, dim_z = z_from_encoder.shape
            z_from_encoder_flat = z_from_encoder.reshape(-1, self.dim_dict)
            # 将权重从[bs, dim_z, width, height]变形为[bs*width*height, dim_z]
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
        
        # 添加数值稳定性检查
        # 1. 确保kappa_q没有无效值
        if torch.isnan(kappa_q).any() or torch.isinf(kappa_q).any():
            print("警告: 检测到kappa_q中有NaN或Inf值，将替换为默认值")
            kappa_q = torch.where(torch.isnan(kappa_q) | torch.isinf(kappa_q), 
                                torch.tensor(1.0, device=kappa_q.device), kappa_q)
        
        # 2. 确保z_from_encoder没有无效值
        if torch.isnan(z_from_encoder).any() or torch.isinf(z_from_encoder).any():
            print("警告: 检测到z_from_encoder中有NaN或Inf值，将替换为0")
            z_from_encoder = torch.where(torch.isnan(z_from_encoder) | torch.isinf(z_from_encoder),
                                      torch.tensor(0.0, device=z_from_encoder.device), z_from_encoder)
            z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        
        # 限制kappa_q范围，防止数值问题
        kappa_q = torch.clamp(kappa_q, min=1e-5, max=1e5)
        
        codebook_norm = F.normalize(codebook, p=2.0, dim=1)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook_norm, kappa_q)
        
        # 添加数值稳定性
        # 限制logit范围，防止softmax溢出
        logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
        logit = logit - logit_max  # 数值稳定性技巧
        logit = torch.clamp(logit, min=-50.0, max=50.0)  # 进一步限制范围
        
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # 确保概率和为1且不含NaN
        if torch.isnan(probabilities).any():
            print("警告: 检测到概率计算中的NaN值，将使用均匀概率")
            probabilities = torch.ones_like(probabilities) / self.size_dict
            log_probabilities = torch.log(probabilities + 1e-10)
        
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
                # 添加安全检查
                valid_probs = torch.isfinite(probabilities).all(dim=1)
                if not valid_probs.all():
                    print(f"警告: 检测到{(~valid_probs).sum().item()}个样本的概率分布无效，使用均匀分布替代")
                    # 对于无效的概率行，使用均匀分布
                    uniform_probs = torch.ones((~valid_probs).sum().item(), self.size_dict, 
                                            device=probabilities.device) / self.size_dict
                    probabilities[~valid_probs] = uniform_probs
                
                # 确保概率和为1
                prob_sum = probabilities.sum(dim=1, keepdim=True)
                probabilities = probabilities / prob_sum.clamp(min=1e-10)
                
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook_norm).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()

        # Latent loss - 添加数值稳定性
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        # 确保kld_discrete不是NaN
        if torch.isnan(kld_discrete) or torch.isinf(kld_discrete):
            print("警告: KLD离散项计算出NaN/Inf，设置为0")
            kld_discrete = torch.tensor(0.0, device=z_from_encoder.device)
            
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, kappa_q).mean()
        # 确保kld_continuous不是NaN
        if torch.isnan(kld_continuous) or torch.isinf(kld_continuous):
            print("警告: KLD连续项计算出NaN/Inf，设置为0")
            kld_continuous = torch.tensor(0.0, device=z_from_encoder.device)
            
        loss = kld_discrete + kld_continuous
        
        # 确保avg_probs不含NaN且非零
        avg_probs = torch.clamp(avg_probs, min=1e-10)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_to_decoder, loss, perplexity
 
    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, kappa_q):
        z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict)
        distances = -kappa_q * torch.matmul(z_from_encoder_flat, codebook.t())

        return distances
    
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum(x1 * (x1-x2) * weight, dim=(1,2,3))
