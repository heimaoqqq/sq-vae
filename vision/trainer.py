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
from third_party.semseg import SegmentationMetric
from util import *
import time


class GaussianSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(GaussianSQVAETrainer, self).__init__(cfgs, flgs, train_loader, val_loader, test_loader)
        
        # 从model导入GaussianSQVAE
        from model import GaussianSQVAE
        self.model_class = GaussianSQVAE
        
        # 初始化模型
        self._initialize_model()
        
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
            
            # 添加异常处理
            try:
                _, _, loss = self.model(x, flg_train=True, flg_quant_det=False)
                
                # 确保所有损失值都是标量
                for key in loss:
                    if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                        loss[key] = loss[key].mean()
                        
                # 检查损失是否为NaN或Inf
                if torch.isnan(loss["all"]) or torch.isinf(loss["all"]):
                    print(f"警告: 批次 {batch_idx} 损失为NaN/Inf，跳过此批次")
                    continue
                    
                self.optimizer.zero_grad()
                loss["all"].backward()
                
                # 应用梯度裁剪，防止梯度爆炸
                if hasattr(self, 'use_gradient_clip') and self.use_gradient_clip:
                    # 检查梯度是否包含NaN
                    has_nan = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan = True
                                break
                    
                    if has_nan:
                        print(f"警告: 批次 {batch_idx} 梯度包含NaN/Inf，跳过更新")
                        # 重置梯度
                        self.optimizer.zero_grad()
                        continue
                    else:
                        # 应用梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                self.optimizer.step()

                train_loss.append(loss["all"].item())
                ms_error.append(loss["mse"].item())
                perplexity.append(loss["perplexity"].item())
            except Exception as e:
                print(f"批次 {batch_idx} 训练出错: {e}")
                # 跳过这个批次，继续训练
                continue

        # 如果没有成功训练任何批次，返回默认结果
        if len(train_loss) == 0:
            print("警告: 所有批次都失败，返回默认结果")
            result = {
                "loss": 1.0, 
                "mse": 1.0, 
                "perplexity": float(self.model.module.size_dict)
            }
        else:
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
                
                # 收集损失值，转换为标量
                test_loss.append(loss["all"].item())
                ms_error.append(loss["mse"].item())
                perplexity.append(loss["perplexity"].item())
                
                # 记录额外损失项（如果存在），确保转换为标量
                if "perceptual" in loss:
                    perceptual_loss.append(loss["perceptual"].item() if torch.is_tensor(loss["perceptual"]) else loss["perceptual"])
                if "spectral" in loss:
                    spectral_loss.append(loss["spectral"].item() if torch.is_tensor(loss["spectral"]) else loss["spectral"])
                    
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


class VmfSQVAETrainer(TrainerBase):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(VmfSQVAETrainer, self).__init__(cfgs, flgs, train_loader, val_loader, test_loader)
        
        # 从model导入VmfSQVAE
        from model import VmfSQVAE
        self.model_class = VmfSQVAE
        
        # 初始化模型
        self._initialize_model()
        
        self.plots = {
            "loss_train": [], "mse_train": [], "perplexity_train": [],
            "loss_val": [], "mse_val": [], "perplexity_val": [],
            "loss_test": [], "mse_test": [], "perplexity_test": []
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
                
            # 添加异常处理
            try:
                _, _, loss = self.model(y, flg_train=True, flg_quant_det=False)
                
                # 确保所有损失值都是标量
                for key in loss:
                    if isinstance(loss[key], torch.Tensor) and loss[key].numel() > 1:
                        loss[key] = loss[key].mean()
                
                # 检查损失是否为NaN或Inf
                if torch.isnan(loss["all"]) or torch.isinf(loss["all"]):
                    print(f"警告: 批次 {batch_idx} 损失为NaN/Inf，跳过此批次")
                    continue
                    
                self.optimizer.zero_grad()
                loss["all"].backward()
                
                # 应用梯度裁剪，防止梯度爆炸
                if hasattr(self, 'use_gradient_clip') and self.use_gradient_clip:
                    # 检查梯度是否包含NaN
                    has_nan = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan = True
                                break
                    
                    if has_nan:
                        print(f"警告: 批次 {batch_idx} 梯度包含NaN/Inf，跳过更新")
                        # 重置梯度
                        self.optimizer.zero_grad()
                        continue
                    else:
                        # 应用梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                self.optimizer.step()

                train_loss.append(loss["all"].item())
                acc.append(loss["acc"].item())
                perplexity.append(loss["perplexity"].item())
            except Exception as e:
                print(f"批次 {batch_idx} 训练出错: {e}")
                # 跳过这个批次，继续训练
                continue

        # 如果没有成功训练任何批次，返回默认结果
        if len(train_loss) == 0:
            print("警告: 所有批次都失败，返回默认结果")
            result = {
                "loss": 1.0, 
                "acc": 0.5, 
                "perplexity": float(self.model.module.size_dict)
            }
        else:
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
        print(f"Generating VmfSQVAE reconstructions...")
        sys.stdout.flush()
        if individual:
            self._generate_individual_reconstructions(filename, num_images=8)
        else:
            self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
    
    def print_loss(self, result, mode, time_interval):
        message = mode.capitalize().ljust(16) + \
            "Loss: {:5.4f}, ACC: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec" \
            .format(
            result["loss"], result["acc"], result["perplexity"], time_interval
            )
        print(message)  # 总是打印，忽略noprint标志
        sys.stdout.flush()  # 强制刷新输出缓冲区


