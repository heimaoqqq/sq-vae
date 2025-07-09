import sys
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from trainer_base import TrainerBase
from model import GaussianSQVAE, VmfSQVAE
from model_enhanced import EnhancedGaussianSQVAE
from third_party.semseg import SegmentationMetric
from util import *
import time


class GaussianSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(GaussianSQVAETrainer, self).__init__(cfgs, flgs, train_loader, val_loader, test_loader)
        self.model_class = GaussianSQVAE
        self.plots = {
            "loss_train": [], "mse_train": [], "perplexity_train": [],
            "loss_val": [], "mse_val": [], "perplexity_val": [],
            "loss_test": [], "mse_test": [], "perplexity_test": []
        }

    def _train(self, epoch):
        train_loss = []
        ms_error = []
        perplexity = []
        self.model.train()
        start_time = time.time()
        print(f"Epoch {epoch}: Training...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.cuda()
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                self.model.module.quantizer.set_temperature(temperature_current)
            
            _, _, loss = self.model(x, flg_train=True, flg_quant_det=False)
            
            # 确保所有损失值都是标量
            for key in loss:
                if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                    loss[key] = loss[key].mean()
                    
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].item())
            ms_error.append(loss["mse"].item())
            perplexity.append(loss["perplexity"].item())
                
        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
        sys.stdout.flush()  # 强制刷新输出缓冲区
                
        return result    
    
    def _test(self, mode="validation"):
        self.model.eval()
        print(f"Epoch evaluation ({mode})...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        _ = self._test_sub(False, mode)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result

    def _test_sub(self, flg_quant_det, mode="validation"):
        test_loss = []
        ms_error = []
        perplexity = []
        if mode == "validation":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(data_loader):
                x = x.cuda()
                _, _, loss = self.model(x, flg_quant_det=flg_quant_det)
                
                # 确保所有损失值都是标量
                for key in loss:
                    if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                        loss[key] = loss[key].mean()
                
                test_loss.append(loss["all"].item())
                ms_error.append(loss["mse"].item())
                perplexity.append(loss["perplexity"].item())
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, mode, time.time()-start_time)
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        return result
        
    def _make_epoch_logger(self, train_result, val_result):
        self.plots["loss_train"].append(train_result["loss"])
        self.plots["mse_train"].append(train_result["mse"])
        self.plots["perplexity_train"].append(train_result["perplexity"])
        self.plots["loss_val"].append(val_result["loss"])
        self.plots["mse_val"].append(val_result["mse"])
        self.plots["perplexity_val"].append(val_result["perplexity"])
    
    def generate_reconstructions(self, filename, nrows=4, ncols=8, individual=False):
        print(f"Generating reconstructions...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        if individual:
            self._generate_individual_reconstructions(filename, num_images=8)
        else:
            self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        message = mode.capitalize().ljust(16) + \
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec" \
            .format(
                result["loss"], result["mse"], result["perplexity"], time_interval
            )
        print(message)  # 总是打印，忽略noprint标志
        sys.stdout.flush()  # 强制刷新输出缓冲区


class EnhancedGaussianSQVAETrainer(GaussianSQVAETrainer):
    """
    增强型高斯SQ-VAE训练器，扩展基本训练器以支持增强型模型
    并添加额外损失项的记录和显示
    """
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        # 首先调用父类初始化
        super(EnhancedGaussianSQVAETrainer, self).__init__(cfgs, flgs, train_loader, val_loader, test_loader)
        # 替换模型类为增强型模型
        self.model_class = EnhancedGaussianSQVAE
        
        # 扩展plots字典以包含额外损失项
        self.plots.update({
            "perceptual_train": [], "perceptual_val": [], "perceptual_test": [],
            "spectral_train": [], "spectral_val": [], "spectral_test": []
        })
        
        print("使用增强型高斯SQ-VAE训练器")
        
    def _train(self, epoch):
        train_loss = []
        ms_error = []
        perplexity = []
        perceptual_loss = []
        spectral_loss = []
        
        self.model.train()
        start_time = time.time()
        print(f"Epoch {epoch}: Training...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.cuda()
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                self.model.module.quantizer.set_temperature(temperature_current)
            
            _, _, loss = self.model(x, flg_train=True, flg_quant_det=False)
            
            # 确保所有损失值都是标量
            for key in loss:
                if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                    loss[key] = loss[key].mean()
                    
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].item())
            ms_error.append(loss["mse"].item())
            perplexity.append(loss["perplexity"].item())
            
            # 记录额外损失项（如果存在）
            if "perceptual" in loss:
                perceptual_loss.append(loss["perceptual"])
            if "spectral" in loss:
                spectral_loss.append(loss["spectral"])
                
        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        
        # 添加额外损失项到结果
        if perceptual_loss:
            result["perceptual"] = np.array(perceptual_loss).mean(0)
        if spectral_loss:
            result["spectral"] = np.array(spectral_loss).mean(0)
            
        self.print_loss(result, "train", time.time()-start_time)
        sys.stdout.flush()  # 强制刷新输出缓冲区
                
        return result
        
    def _test_sub(self, flg_quant_det, mode="validation"):
        test_loss = []
        ms_error = []
        perplexity = []
        perceptual_loss = []
        spectral_loss = []
        
        if mode == "validation":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(data_loader):
                x = x.cuda()
                _, _, loss = self.model(x, flg_quant_det=flg_quant_det)
                
                # 确保所有损失值都是标量
                for key in loss:
                    if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                        loss[key] = loss[key].mean()
                
                test_loss.append(loss["all"].item())
                ms_error.append(loss["mse"].item())
                perplexity.append(loss["perplexity"].item())
                
                # 记录额外损失项（如果存在）
                if "perceptual" in loss:
                    perceptual_loss.append(loss["perceptual"])
                if "spectral" in loss:
                    spectral_loss.append(loss["spectral"])
                    
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["mse"] = np.array(ms_error).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        
        # 添加额外损失项到结果
        if perceptual_loss:
            result["perceptual"] = np.array(perceptual_loss).mean(0)
        if spectral_loss:
            result["spectral"] = np.array(spectral_loss).mean(0)
            
        self.print_loss(result, mode, time.time()-start_time)
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        return result
    
    def print_loss(self, result, mode, time_interval):
        # 首先构建基本信息
        message = mode.capitalize().ljust(16) + \
            "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}" \
            .format(
                result["loss"], result["mse"], result["perplexity"]
            )
            
        # 添加额外损失项信息（如果存在）
        if "perceptual" in result:
            message += ", Perceptual: {:5.4f}".format(result["perceptual"])
        if "spectral" in result:
            message += ", Spectral: {:5.4f}".format(result["spectral"])
            
        # 添加时间信息
        message += ", Time: {:5.3f} sec".format(time_interval)
        
        print(message)
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
    def _make_epoch_logger(self, train_result, val_result):
        # 调用父类方法记录基本指标
        super()._make_epoch_logger(train_result, val_result)
        
        # 记录额外损失项
        if "perceptual" in train_result:
            self.plots["perceptual_train"].append(train_result["perceptual"])
        if "perceptual" in val_result:
            self.plots["perceptual_val"].append(val_result["perceptual"])
        if "spectral" in train_result:
            self.plots["spectral_train"].append(train_result["spectral"])
        if "spectral" in val_result:
            self.plots["spectral_val"].append(val_result["spectral"])
    
    def generate_reconstructions(self, filename, nrows=4, ncols=8, individual=False):
        print(f"Generating enhanced reconstructions...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        if individual:
            self._generate_individual_reconstructions(filename, num_images=8)
        else:
            # 使用父类方法生成重建图像
            self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
            
            # 另外生成一个带有"enhanced_"前缀的对比图
            # 这样可以直观比较增强前后的效果
            enhanced_filename = "enhanced_" + filename
            self._generate_enhanced_reconstructions(enhanced_filename, nrows=nrows, ncols=ncols)

    def _generate_enhanced_reconstructions(self, filename, nrows=4, ncols=8):
        """
        生成增强重建图像对比图，显示原图、普通重建和增强重建的效果对比
        """
        print("生成增强重建对比图...")
        sys.stdout.flush()
        
        # 准备图像数据
        n_samples = nrows * ncols
        with torch.no_grad():
            # 从测试集中获取样本
            data_iter = iter(self.test_loader)
            x, _ = next(data_iter)
            x = x[:n_samples].cuda()
            
            # 使用模型进行普通重建和增强重建
            if isinstance(self.model, nn.DataParallel):
                decoder = self.model.module.decoder
                encoder = self.model.module.encoder
                quantizer = self.model.module.quantizer
                codebook = self.model.module.codebook
                param_q = self.model.module.param_q
            else:
                decoder = self.model.decoder
                encoder = self.model.encoder
                quantizer = self.model.quantizer
                codebook = self.model.codebook
                param_q = self.model.param_q
            
            # 获取编码结果
            if hasattr(encoder, 'flg_variance') and encoder.flg_variance:
                z_e, _ = encoder(x)
            else:
                z_e = encoder(x)
            
            # 获取量化结果
            z_q, _, _ = quantizer(z_e, param_q, codebook, False, True)
            
            # 生成重建图像
            x_reconst = decoder(z_q)
            
            # 创建图像网格
            nrows = nrows * 3  # 每个样本显示三行：原图、标准重建、增强重建
            figsize = (ncols * 2, nrows * 2)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            
            # 绘制图像
            for i in range(n_samples):
                row = i // ncols * 3
                col = i % ncols
                
                # 显示原始图像
                orig_img = x[i].cpu().permute(1, 2, 0).numpy()
                axes[row, col].imshow(orig_img)
                axes[row, col].set_title('Original')
                axes[row, col].axis('off')
                
                # 显示普通重建
                recon_img = x_reconst[i].cpu().permute(1, 2, 0).detach().numpy()
                axes[row + 1, col].imshow(recon_img)
                axes[row + 1, col].set_title('Basic')
                axes[row + 1, col].axis('off')
                
                # 显示增强重建 (应用细节增强器)
                if hasattr(decoder, 'detail_enhancer'):
                    enhanced_img = decoder.detail_enhancer(x_reconst[i].unsqueeze(0))
                    enhanced_img = enhanced_img.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
                    axes[row + 2, col].imshow(enhanced_img)
                    axes[row + 2, col].set_title('Enhanced')
                    axes[row + 2, col].axis('off')
                else:
                    # 如果没有增强器，显示相同的重建结果
                    axes[row + 2, col].imshow(recon_img)
                    axes[row + 2, col].set_title('No Enhancer')
                    axes[row + 2, col].axis('off')
                    
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()


class VmfSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(VmfSQVAETrainer, self).__init__(cfgs, flgs, train_loader, val_loader, test_loader)
        self.model_class = VmfSQVAE
        self.metric_semseg = SegmentationMetric(cfgs.network.num_class)
        self.plots = {
            "loss_train": [], "acc_train": [], "perplexity_train": [],
            "loss_val": [], "acc_val": [], "perplexity_val": [], "miou_val": [],
            "loss_test": [], "acc_test": [], "perplexity_test": [], "miou_test": []
        }
    
    def _train(self, epoch):
        train_loss = []
        acc = []
        perplexity = []
        self.model.train()
        start_time = time.time()
        print(f"Epoch {epoch}: Training...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            y = self.preprocess(x, y)
            if self.flgs.decay:
                step = (epoch - 1) * len(self.train_loader) + batch_idx + 1
                temperature_current = self._set_temperature(
                    step, self.cfgs.quantization.temperature)
                self.model.module.quantizer.set_temperature(temperature_current)
            _, _, loss = self.model(y, flg_train=True, flg_quant_det=False)
            
            # 确保所有损失值都是标量
            for key in loss:
                if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                    loss[key] = loss[key].mean()
            
            self.optimizer.zero_grad()
            loss["all"].backward()
            self.optimizer.step()

            train_loss.append(loss["all"].item())
            acc.append(loss["acc"].item())
            perplexity.append(loss["perplexity"].item())

        result = {}
        result["loss"] = np.asarray(train_loss).mean(0)
        result["acc"] = np.array(acc).mean(0)
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, "train", time.time()-start_time)
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        return result
    
    def _test(self, mode="val"):
        print(f"Epoch evaluation ({mode})...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        _ = self._test_sub(False)
        result = self._test_sub(True, mode)
        self.scheduler.step(result["loss"])
        return result
    
    def _test_sub(self, flg_quant_det, mode="val"):
        test_loss = []
        acc = []
        perplexity = []
        self.metric_semseg.reset()
        if mode == "val":
            data_loader = self.val_loader
        elif mode == "test":
            data_loader = self.test_loader
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                y = self.preprocess(x, y)
                x_reconst, _, loss = self.model(y, flg_quant_det=flg_quant_det)
                
                # 确保所有损失值都是标量
                for key in loss:
                    if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                        loss[key] = loss[key].mean()
                
                self.metric_semseg.update(x_reconst, y)
                pixAcc, mIoU, _ = self.metric_semseg.get()
                test_loss.append(loss["all"].item())
                acc.append(loss["acc"].item())
                perplexity.append(loss["perplexity"].item())
            pixAcc, mIoU, _ = self.metric_semseg.get()
        result = {}
        result["loss"] = np.asarray(test_loss).mean(0)
        result["acc"] = np.array(acc).mean(0)
        result["miou"] = mIoU
        result["perplexity"] = np.array(perplexity).mean(0)
        self.print_loss(result, mode, time.time()-start_time)
        print("%15s"%"PixAcc: {:5.4f} mIoU: {:5.4f}".format(
            pixAcc, mIoU
        ))
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        return result
    
    def _make_epoch_logger(self, train_result, val_result):
        self.plots["loss_train"].append(train_result["loss"])
        self.plots["acc_train"].append(train_result["acc"])
        self.plots["perplexity_train"].append(train_result["perplexity"])
        self.plots["loss_val"].append(val_result["loss"])
        self.plots["acc_val"].append(val_result["acc"])
        self.plots["miou_val"].append(val_result["miou"])
        self.plots["perplexity_val"].append(val_result["perplexity"])
        
    def generate_reconstructions(self, filename, nrows=4, ncols=8, individual=False):
        print(f"Generating reconstructions...")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        if individual:
            self._generate_individual_reconstructions(filename, num_images=8)
        else:
            self._generate_reconstructions_discrete(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        message = mode.capitalize().ljust(16) + \
            "Loss: {:5.4f}, ACC: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec" \
            .format(
            result["loss"], result["acc"], result["perplexity"], time_interval
            )
        print(message)  # 总是打印，忽略noprint标志
        sys.stdout.flush()  # 强制刷新输出缓冲区


