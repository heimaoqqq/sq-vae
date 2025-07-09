import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class MicroDopplerDataset(Dataset):
    """自定义数据集用于加载微多普勒时频图像"""
    
    def __init__(self, root_dir, transform=None, resize=None):
        """
        初始化数据集
        
        参数:
            root_dir (str): 包含所有ID_x文件夹的根目录
            transform (callable, optional): 可选的图像变换
            resize (tuple, optional): 如果需要调整图像大小，提供(height, width)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        
        # 获取所有ID文件夹并按数字排序
        self.folders = []
        folder_ids = []
        for folder in os.listdir(root_dir):
            if folder.startswith('ID_') and os.path.isdir(os.path.join(root_dir, folder)):
                try:
                    # 提取ID数字部分并转换为整数
                    id_num = int(folder.split('_')[1])
                    folder_ids.append((id_num, folder))
                except ValueError:
                    continue
        
        # 按ID数字大小排序
        folder_ids.sort()
        self.folders = [folder for _, folder in folder_ids]
        
        # 创建ID到标签的映射
        self.id_to_label = {folder: idx for idx, folder in enumerate(self.folders)}
        
        # 收集所有图像路径和对应标签
        self.image_paths = []
        self.labels = []
        for folder in self.folders:
            folder_path = os.path.join(root_dir, folder)
            folder_label = self.id_to_label[folder]
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(folder_label)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 调整图像大小(如果需要)
        if self.resize:
            image = image.resize(self.resize, Image.BILINEAR)
        
        # 应用变换(如果有)
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换为ToTensor
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor(label)


def get_microdoppler_dataloaders(dataset_path, batch_size=32, num_workers=2, 
                               image_size=None, train_ratio=0.7, val_ratio=0.15):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        dataset_path (str): 数据集路径
        batch_size (int): 批量大小
        num_workers (int): 数据加载工作线程数
        image_size (tuple): 可选的调整图像大小(height, width)
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        
    返回:
        train_loader, val_loader, test_loader: 三个数据加载器
    """
    # 定义变换
    if image_size:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        resize = None  # 通过transform调整大小
    else:
        # 如果不指定大小，则保持原始大小
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        resize = None  # 保持原始大小
    
    # 创建数据集
    full_dataset = MicroDopplerDataset(dataset_path, transform=transform, resize=resize)
    
    # 打印数据集信息
    print(f"找到 {len(full_dataset)} 张图像，来自 {len(full_dataset.folders)} 个用户ID")
    print(f"用户ID映射到标签: {full_dataset.id_to_label}")
    
    # 计算分割大小
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子以便复现
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 