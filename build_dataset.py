import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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

    return np.stack(windows, axis=0), np.stack(labels,axis=0)  # shape: [N_windows, window_length, F]


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


if __name__ == "__main__":
    # 数据路径
    data_dir = "processed_data"
    Zxx_real = load_csv_file(os.path.join(data_dir, "Zxx", "Zxx_real.csv"))
    Zxx_imag = load_csv_file(os.path.join(data_dir, "Zxx", "Zxx_imag.csv"))
    LBL = load_csv_file(os.path.join(data_dir, "Zxx", "LBL.csv"))
    
    # 将实部和虚部组合成复数
    Zxx = Zxx_real + 1j * Zxx_imag  #(128, 14634)
    
    # 滑动窗口分割
    windows,window_labels = sliding_window_with_hann(Zxx,LBL, window_length, step)
        

    print(window_labels)  # [N_windows, num_classes]   (10, 10)
    print(windows.shape)  # [N_windows, window_length, F]   (10, 3658, 128)



