import shutil
import json
import datetime
from torch import nn
import matplotlib.pyplot as plt

from model import GaussianSQVAE, VmfSQVAE
from util import *

class TrainerBase(nn.Module):
    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(TrainerBase, self).__init__()
        self.cfgs = cfgs
        self.flgs = flgs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 初始化模型和优化器属性，但不创建实例
        self.model = None
        self.model_class = None  # 子类需要设置此属性
        self.optimizer = None
        
        # 设置优化器参数
        self.lr = cfgs.train.lr
        
        # 创建调度器
        self.scheduler_params = {
            "mode": "min", 
            "factor": 0.5, 
            "patience": 3,
            "verbose": True, 
            "threshold": 0.0001, 
            "threshold_mode": "rel",
            "cooldown": 0, 
            "min_lr": 0, 
            "eps": 1e-08
        }
        
    def _initialize_model(self):
        """初始化模型和优化器，需要在子类中调用"""
        if self.model_class is None:
            raise ValueError("子类必须设置model_class属性")
        
        # 输出模型信息
        print(f"初始化模型: {self.model_class.__name__}")
        
        # 创建模型实例
        model_instance = self.model_class(self.cfgs, self.flgs).cuda()
        self.model = nn.DataParallel(model_instance)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, amsgrad=False)
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self.scheduler_params)
    
    def load(self, timestamp=""):
        if timestamp != "":
            self.path = os.path.join(self.cfgs.path, timestamp)
        self.model.load_state_dict(
            torch.load(os.path.join(self.path, "best.pt")))
        self.plots = np.load(
            os.path.join(self.path, "plots.npy"), allow_pickle=True).item()
        print(self.path)
        self.model.eval()
    

    ## Methods for main loop

    def main_loop(self, max_iter=None, timestamp=None):
        if timestamp == None:
            self._make_path()
        else:
            self.path = os.path.join(self.cfgs.path, timestamp)
        BEST_LOSS = 1e+20
        LAST_SAVED = -1

        if max_iter == None:
            max_iter = self.cfgs.train.epoch_max
        for epoch in range(1, max_iter+1):
            myprint("[Epoch={}]".format(epoch), self.flgs.noprint)
            res_train = self._train(epoch)
            if self.flgs.save:
                self._writer_train(res_train, epoch)
            res_test = self._test()
            if self.flgs.save:
                self._writer_val(res_test, epoch)
            
            if self.flgs.save:
                if res_test["loss"] <= BEST_LOSS:
                    BEST_LOSS = res_test["loss"]
                    LAST_SAVED = epoch
                    myprint("----Saving model!", self.flgs.noprint)
                    torch.save(
                        self.model.state_dict(), os.path.join(self.path, "best.pt"))
                    self.generate_reconstructions(
                        os.path.join(self.path, "reconstructions_best"), individual=True)
                else:
                    myprint("----Not saving model! Last saved: {}"
                        .format(LAST_SAVED), self.flgs.noprint)
                torch.save(
                    self.model.state_dict(), os.path.join(self.path, "current.pt"))
                self.generate_reconstructions(
                    os.path.join(self.path, "reconstructions_current"), individual=True)
    
    def preprocess(self, x, y):
        if self.cfgs.dataset.name == "CelebAMask_HQ":
            y[:, 0, :, :] = y[:, 0, :, :] * 255.0
            y = torch.round(y[:, 0, :, :]).cuda()
        return y
    
    def test(self, mode="test"):
        result = self._test(mode)
        if mode == "test":
            self._writer_test(result)
        return result
    
    def _set_temperature(self, step, param):
        temperature = np.max([param.init * np.exp(-param.decay*step), param.min])
        return temperature

    
    def _save_config(self):
        tf = open(self.path + "/configs.json", "w")
        json.dump(self.cfgs, tf)
        tf.close()
    
    def _train(self):
        raise NotImplementedError()
    
    def _test(self):
        raise NotImplementedError()
    
    def print_loss(self):
        raise NotImplementedError()
    

    ## Visualization    

    def generate_reconstructions(self):
        raise NotImplementedError()
    
    def generate_reconstructions_paper(self, nrows=1, ncols=10, off_set=0):
        self.model.eval()
        x = next(iter(self.test_loader))[0]
        x = x[off_set:off_set+nrows*ncols].cuda()
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        images_original = x.cpu().data.numpy()
        images_reconst = x_tilde.cpu().data.numpy()
        plot_images_paper(images_original,
            os.path.join(self.path, "paper_original"), nrows=nrows, ncols=ncols)
        plot_images_paper(images_reconst,
            os.path.join(self.path, "paper_reconst"), nrows=nrows, ncols=ncols)

    def _generate_reconstructions_continuous(self, filename, nrows=4, ncols=8):
        self.model.eval()
        x = next(iter(self.test_loader))[0]
        x = x[:nrows*ncols].cuda()
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        x_cat = torch.cat([x, x_tilde], 0)
        images = x_cat.cpu().data.numpy()
        plot_images(images, filename+".png", nrows=nrows, ncols=ncols)
    
    def _generate_individual_reconstructions(self, filename, num_images=8):
        """生成单独的原始图像和重建图像文件"""
        self.model.eval()
        x = next(iter(self.test_loader))[0]
        x = x[:num_images].cuda()
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        
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
    
    def _generate_reconstructions_discrete(self, filename, nrows=4, ncols=8):
        self.model.eval()
        x, y = next(iter(self.test_loader))
        x = x[:nrows*ncols].cuda()
        y = y[:nrows*ncols].cuda()
        y[:, 0, :, :] = y[:, 0, :, :] * 255.0
        y_long = y
        y = y[:, 0, :, :]
        output = self.model(y, flg_train=False, flg_quant_det=True)
        label_tilde = output[0]
        label_real = idx_to_onehot(y_long)
        label_batch_predict = generate_label(label_tilde[:,:19,:,:], x.shape[-1])
        label_batch_real = generate_label(label_real, x.shape[-1])
        x_cat = torch.cat([label_batch_real, label_batch_predict], 0)
        images = x_cat.cpu().data.numpy()
        plot_images(images, filename+".png", nrows=nrows, ncols=ncols)


    ## Saving

    def _make_path(self):
        import glob
        dt_now = datetime.datetime.now()
        timestamp = dt_now.strftime("%m%d_%H%M")
        self.path = os.path.join(self.cfgs.path, "{}_seed{}_{}".format(
            self.cfgs.network.name, self.cfgs.train.seed,timestamp))
        print(self.path)
        if self.flgs.save:
            self._makedir(self.path)
            list_dir = self.cfgs.list_dir_for_copy
            files = []
            for dirname in list_dir:
                files.append(glob.glob(dirname+"*.py"))
            target = os.path.join(self.path, "codes")
            for i, dirname in enumerate(list_dir):
                if not os.path.exists(os.path.join(target, dirname)):
                    os.makedirs(os.path.join(target, dirname))
                for file in  files[i]:
                    shutil.copyfile(file, os.path.join(target, file))

    def _makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            i = 1
            while True:
                path += "_{}".format(i)
                if not os.path.exists(path):
                    os.makedirs(path)
                    break
                print(i)
                i += 1
        self._save_config()
        self.path = path
        
    def _writer_train(self, result, epoch):
        self._append_writer_train(result)
        np.save(os.path.join(self.path, "plots.npy"), self.plots)
    
    def _writer_val(self, result, epoch):
        self._append_writer_val(result)
        np.save(os.path.join(self.path, "plots.npy"), self.plots)
    
    def _writer_test(self, result):
        self._append_writer_test(result)
        np.save(os.path.join(self.path, "plots.npy"), self.plots)
    
    def _append_writer_train(self, result):
        for metric in result:
            self.plots[metric+"_train"].append(result[metric])
        
    def _append_writer_val(self, result):
        for metric in result:
            self.plots[metric+"_val"].append(result[metric])
    
    def _append_writer_test(self, result):
        for metric in result:
            self.plots[metric+"_test"].append(result[metric])

