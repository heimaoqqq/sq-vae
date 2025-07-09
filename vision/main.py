import os
import argparse
from configs.defaults import get_cfgs_defaults
import torch
import sys
import torch.nn as nn
import time
import numpy as np

from trainer import GaussianSQVAETrainer, VmfSQVAETrainer, EnhancedGaussianSQVAETrainer
# 在顶部导入，使其在全局命名空间可用
try:
    # 使用绝对导入路径
    from vision.model_diffusers_sq import DiffusersGaussianSQVAE
    DIFFUSERS_MODEL_AVAILABLE = True
    print("成功导入DiffusersGaussianSQVAE模型")
except ImportError as e:
    # 尝试相对导入
    try:
        # 将当前目录添加到导入路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from model_diffusers_sq import DiffusersGaussianSQVAE
        DIFFUSERS_MODEL_AVAILABLE = True
        print("成功导入DiffusersGaussianSQVAE模型")
    except ImportError as e:
        print(f"无法导入DiffusersGaussianSQVAE模型: {e}")
        DIFFUSERS_MODEL_AVAILABLE = False
from util import set_seeds, get_loader


def arg_parse():
    parser = argparse.ArgumentParser(
            description="main.py")
    parser.add_argument(
        "-c", "--config_file", default="", help="config file")
    parser.add_argument(
        "-ts", "--timestamp", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "--save", action="store_true", help="save trained model")
    parser.add_argument(
        "--dbg", action="store_true", help="print losses per epoch")
    parser.add_argument(
        "--gpu", default="0", help="index of gpu to be used")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed number for randomness")
    args = parser.parse_args()
    return args


def load_config(args):
    cfgs = get_cfgs_defaults()
    config_path = os.path.join(os.path.dirname(__file__), "configs", args.config_file)
    print(config_path)
    
    # 直接使用原始配置文件
    try:
        cfgs.merge_from_file(config_path)
    except Exception as e:
        print(f"配置加载失败: {e}")
        # 尝试手动解析YAML
        try:
            import yaml
            # 尝试使用UTF-8编码读取
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            except UnicodeDecodeError:
                # 如果失败，尝试使用latin-1编码
                with open(config_path, 'r', encoding='latin-1') as f:
                    config_data = yaml.safe_load(f)
            
            # 手动设置配置
            for section, values in config_data.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        setattr(getattr(cfgs, section), key, value)
                else:
                    setattr(cfgs, section, values)
            print("已使用手动解析的方式加载配置")
        except Exception as e2:
            print(f"手动解析配置也失败: {e2}")
            raise e
    
    cfgs.train.seed = args.seed
    cfgs.flags.save = args.save
    cfgs.flags.noprint = False  # 强制启用日志输出，覆盖--dbg参数
    cfgs.path_data = cfgs.path
    cfgs.path = os.path.join(cfgs.path, cfgs.path_specific)
    if cfgs.model.name.lower() == "vmfsqvae":
        cfgs.quantization.dim_dict += 1
    cfgs.flags.var_q = not(cfgs.model.param_var_q=="gaussian_1" or
                                      cfgs.model.param_var_q=="vmf")
    cfgs.freeze()
    flgs = cfgs.flags
    return cfgs, flgs


if __name__ == "__main__":
    print("main.py")
    
    # 启用更详细的输出
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    
    ## Experimental setup
    args = arg_parse()
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfgs, flgs = load_config(args)
    print("[Checkpoint path] "+cfgs.path)
    print(cfgs)
    
    ## Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    set_seeds(args.seed)

    ## Data loader
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loader(
        cfgs.dataset.name, cfgs.path_dataset, cfgs.train.bs, cfgs.nworker)
    print("Complete dataload")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    ## Trainer
    print("=== {} ===".format(cfgs.model.name.upper()))
    if cfgs.model.name == "GaussianSQVAE":
        trainer = GaussianSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    elif cfgs.model.name == "VmfSQVAE":
        trainer = VmfSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    elif cfgs.model.name == "EnhancedGaussianSQVAE":
        trainer = EnhancedGaussianSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    elif cfgs.model.name == "DiffusersGaussianSQVAE":
        # 检查Diffusers模型是否可用
        if not DIFFUSERS_MODEL_AVAILABLE:
            print("=" * 50)
            print("错误: 无法使用DiffusersGaussianSQVAE模型")
            print("请确认以下几点:")
            print("1. diffusers库已正确安装且可访问")
            print("2. vision/model_diffusers_sq.py文件存在且可正确导入")
            print("3. 环境中的Python路径正确配置")
            print("=" * 50)
            raise ImportError("无法使用DiffusersGaussianSQVAE模型，请检查日志中的详细信息")
        
        # 确保model_diffusers_sq.py中的模型可以被导入
        try:
            # 再次验证模型类可以被正确导入和使用
            print("正在验证DiffusersGaussianSQVAE模型...")
            from model_diffusers_sq import DiffusersGaussianSQVAE
            print(f"成功验证模型类: DiffusersGaussianSQVAE")
            
            # 创建一个专用的trainer，不使用GaussianSQVAETrainer
            from trainer_base import TrainerBase
            
            # 创建一个简单的训练器类
            class DiffusersTrainer(TrainerBase):
                def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
                    super(DiffusersTrainer, self).__init__(cfgs, flgs, train_loader, val_loader, test_loader)
                    
                    # 直接创建DiffusersGaussianSQVAE模型
                    print("创建DiffusersGaussianSQVAE模型...")
                    model_instance = DiffusersGaussianSQVAE(cfgs, flgs).cuda()
                    self.model = nn.DataParallel(model_instance)
                    
                    # 创建优化器
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=self.lr, amsgrad=False)
                    
                    # 创建学习率调度器
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, **self.scheduler_params)
                    
                    # 初始化plots字典
                    self.plots = {
                        "loss_train": [], "mse_train": [], "perplexity_train": [],
                        "loss_val": [], "mse_val": [], "perplexity_val": [],
                        "loss_test": [], "mse_test": [], "perplexity_test": []
                    }
                    
                    print("DiffusersGaussianSQVAE模型创建完成")
                
                # 实现必要的方法
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
                            
                            # 收集损失值，转换为标量
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
                
                def generate_reconstructions(self, filename, nrows=4, ncols=8, individual=False):
                    print(f"Generating reconstructions...")
                    sys.stdout.flush()  # 强制刷新输出缓冲区
                    if individual:
                        self._generate_individual_reconstructions(filename, num_images=8)
                    else:
                        self._generate_reconstructions_continuous(filename, nrows=nrows, ncols=ncols)
                    
                def _generate_reconstructions_continuous(self, filename, nrows=4, ncols=8):
                    self.model.eval()
                    x = next(iter(self.test_loader))[0]
                    x = x[:nrows*ncols].cuda()
                    output = self.model(x, flg_train=False, flg_quant_det=True)
                    x_tilde = output[0]
                    x_cat = torch.cat([x, x_tilde], 0)
                    images = x_cat.cpu().data.numpy()
                    from vision.util import plot_images
                    plot_images(images, filename+".png", nrows=nrows, ncols=ncols)
                
                def _generate_individual_reconstructions(self, filename, num_images=8):
                    """生成单独的原始图像和重建图像文件"""
                    self.model.eval()
                    x = next(iter(self.test_loader))[0]
                    x = x[:num_images].cuda()
                    output = self.model(x, flg_train=False, flg_quant_det=True)
                    x_tilde = output[0]
                    
                    import os
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # 创建保存目录
                    save_dir = filename + "_individual"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        
                    # 保存原始图像和重建图像
                    for i in range(num_images):
                        # 获取原始图像和重建图像
                        original = x[i].cpu().data.numpy()
                        reconstructed = x_tilde[i].cpu().data.numpy()
                        
                        # 如果是单通道图像，转换为三通道
                        if original.shape[0] == 1:
                            original = np.repeat(original, 3, axis=0)
                            reconstructed = np.repeat(reconstructed, 3, axis=0)
                        
                        # 保存原始图像
                        plt.figure(figsize=(5, 5))
                        plt.axis('off')
                        plt.imshow(original.transpose((1, 2, 0)))
                        plt.savefig(os.path.join(save_dir, f"original_{i}.png"), bbox_inches="tight")
                        plt.close()
                        
                        # 保存重建图像
                        plt.figure(figsize=(5, 5))
                        plt.axis('off')
                        plt.imshow(reconstructed.transpose((1, 2, 0)))
                        plt.savefig(os.path.join(save_dir, f"reconstructed_{i}.png"), bbox_inches="tight")
                        plt.close()
                
                def print_loss(self, result, mode, time_interval):
                    message = mode.capitalize().ljust(16) + \
                        "Loss: {:5.4f}, MSE: {:5.4f}, Perplexity: {:5.4f}, Time: {:5.3f} sec" \
                        .format(
                            result["loss"], result["mse"], result["perplexity"], time_interval
                        )
                    print(message)  # 总是打印，忽略noprint标志
                    sys.stdout.flush()  # 强制刷新输出缓冲区
            
            # 创建训练器实例
            trainer = DiffusersTrainer(cfgs, flgs, train_loader, val_loader, test_loader)
            print("成功初始化DiffusersGaussianSQVAE训练器")
        except Exception as e:
            print(f"初始化DiffusersGaussianSQVAE模型时出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        raise Exception(f"Undefined model: {cfgs.model.name}")

    ## Main
    print("Starting training...")
    if args.timestamp == "":
        trainer.main_loop()
    if flgs.save:
        trainer.load(args.timestamp)
        print("Best models were loaded!!")
        res_test = trainer.test()

