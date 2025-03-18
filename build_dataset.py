import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

import dataloader

ts = 0.0082
window_length = int(30 / ts)  # 30s
step =int(10 / ts)  # 10s
num_classes = 10

def load_csv_file(filepath):
    """读取 CSV 文件，并转换为 float32 数组"""
    return pd.read_csv(filepath, header=None).values.astype(np.float32)

def sliding_window_with_hann(data, LBL, window_length, step):
    """
    对输入数据进行滑动窗口分割，并应用 Hann 窗加权。
    Args:
        data: numpy 数组，形状 [T_total, F]
        LBL
        window_length: 每个窗口长度（时间步数）
        step: 滑动步长（窗口之间的间隔），例如 window_length - overlap
    Returns:
        windows: numpy 数组，形状 [N_windows, window_length, F]
    """
    F,T_total = data.shape
    windows = []
    labels=[]
    hann_win = np.hanning(window_length).reshape(-1, 1)  # shape: [window_length, 1]
    # 计算窗口个数
    for start in range(0, T_total - window_length + 1, step):
        win = data[:, start:start+window_length].T  # shape: [window_length, F]
        win = win * hann_win  # 应用 Hann 窗
        windows.append(win)
        win_label=LBL[:,start:start+window_length].flatten().reshape(1,-1)
        window_label = np.zeros(num_classes, dtype=int)
        for label in win_label[0,:]:
            if 0 <= label < num_classes:
                window_label[int(label)] = 1
        labels.append(window_label)

    return np.stack(windows, axis=0), np.stack(labels,axis=0)  # shape: [N_windows, window_length, F],[N_windows, num_classes]


def normalize_data(data):
    """
    对数据进行归一化（min-max归一化）
    参数:
        data: numpy 数组
    返回:
        归一化后的数据, 值域为 [0, 1]
    """
    min_val = np.min(data)
    max_val = np.max(data)
    # 防止除零
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)


class RadarDynamicDataset(Dataset):
    def __init__(self, root_dir, window_length=30, step=10):
        """
        Args:
            root_dir (str): 数据根目录，例如 "Radar_gnn-main/processed_data/Zxx"
            window_length (int): 窗口长度，单位为时间步（例如30秒）
            overlap (int): 窗口重叠部分长度（例如20秒）
        """
        # 获取所有样本文件夹路径，假设命名规则为 "*_mon_*"
        self.sample_folders = glob.glob(os.path.join(root_dir, "*_mon_*"))
        self.window_length = window_length
        self.step = step
        
    def __len__(self):
        return len(self.sample_folders)
    
    def __getitem__(self, idx):
        """
        对于每个样本，动态加载数据，并对每个雷达和标签进行滑动窗口分割
        返回:
            radar_tensor: [N_windows, C, T, F]  (C: 雷达数, T=window_length, F:频谱bin数)
            label_tensor: [N_windows, L]  (L: 标签维度，一般为1或多维)
        """
        sample_folder = self.sample_folders[idx]
        radar_list = []
        label_list = []
        num_radars = 5  # 假设每个样本有 5 个雷达数据（radar_0 ~ radar_4）
        
        # 对每个雷达
        for r in range(num_radars):
            radar_folder = os.path.join(sample_folder, f"radar_{r}")
            real_path = os.path.join(radar_folder, "Zxx_real.csv")
            imag_path = os.path.join(radar_folder, "Zxx_imag.csv")
            
            # 读取数据： shape [T_total, F]
            real_part = load_csv_file(real_path)
            imag_part = load_csv_file(imag_path)

            # 读取标签文件
            if(r==0):
                lbl_path = os.path.join(radar_folder, "LBL.csv")
                LBL = pd.read_csv(lbl_path, header=None).values.astype(np.float32)  # shape: [T_total, L]

            # 计算幅值
            amplitude = np.sqrt(real_part**2 + imag_part**2)
            amplitude = normalize_data(amplitude)

            # 对幅值图应用滑动窗口和 Hann 加权，得到 [N_windows, window_length, F]
            radar_windows, labels = sliding_window_with_hann(amplitude,LBL, self.window_length, self.step)
            radar_list.append(radar_windows)
            label_list.append(labels)
        
        # 现在 radar_list 中每个元素形状为 [N_windows, window_length, F]，共有 num_radars = 5 个
        # 需要将它们堆叠到一起作为通道维度：
        # 首先确保所有雷达窗口数量一致（假设 T_total 固定，滑动窗口数量一致）
        radar_data = np.stack(radar_list, axis=1)  # shape: [N_windows, num_radars, window_length, F]
        window_labels = label_list[0]  # shape: [N_windows, num_classes]
        
        # 转换为 tensor
        radar_tensor = torch.tensor(radar_data)  # [N_windows, num_radars, window_length, F]
        label_tensor = torch.tensor(window_labels)  # [N_windows, num_classes]
        return radar_tensor, label_tensor


if __name__ == "__main__":
    root_dir = "Radar_gnn-main/processed_data/Zxx"  
    dataset = RadarDynamicDataset(root_dir, window_length=30, step=10)
    
    # 数据集划分示例：80%训练, 10%验证, 10%测试（这里只给出示例）
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:train_size+val_size])
    test_dataset = Subset(dataset, indices[train_size+val_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 检查一个 batch
    for radar_batch, label_batch in train_loader:
        print("Radar batch shape:", radar_batch.shape)  # 期望: [B, N_windows, C, T, F]
        print("Label batch shape:", label_batch.shape)    # 期望: [B, N_windows, L]
        break



