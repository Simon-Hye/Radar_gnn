import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import shutil

ts = 0.0082
window_length = int(30 / ts)  # 30s
step = int(10 / ts)  # 10s
num_classes = 10


def load_csv_file(filepath):
    """读取 CSV 文件，并转换为 float32 数组"""
    return pd.read_csv(filepath, header=None).values.astype(np.float32)


def sliding_window_with_hann(data, LBL, window_length, step):
    """
    对输入数据进行滑动窗口分割，并应用 Hann 窗加权。
    Args:
        data: numpy 数组，形状 [F, T]
        LBL
        window_length: 每个窗口长度（时间步数）
        step: 滑动步长（窗口之间的间隔），例如 window_length - overlap
    Returns:
        windows: numpy 数组，形状 [N_windows, window_length, F]
    """
    F, T_total = data.shape
    windows = []
    labels = []
    hann_win = np.hanning(window_length).reshape(-1, 1)  # shape: [window_length, 1]
    # 计算窗口个数
    for start in range(0, T_total - window_length + 1, step):
        win = data[:, start:start + window_length].T  # shape: [window_length, F]
        win = win * hann_win  # 应用 Hann 窗
        windows.append(win)
        win_label = LBL[:, start:start + window_length].flatten().reshape(1, -1)
        window_label = np.zeros(num_classes, dtype=int)
        for label in win_label[0, :]:
            if 0 <= label < num_classes:
                window_label[int(label)] = 1
        labels.append(window_label)

    return np.stack(windows, axis=0), np.stack(labels,
                                               axis=0)  # shape: [N_windows, window_length, F],[N_windows, num_classes]


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


class RadarDataset(Dataset):
    def __init__(self, root_dir,mode='train' ,window_length=int(30 / 0.0082), step=int(10 / 0.0082), num_radars=5):
        """
        Args:
            root_dir (str): 数据根目录，例如 "processed_data/Zxx"
            window_length (int): 窗口长度，单位为时间步（例如30秒）
            step (int): 滑动窗口步长，单位为时间步（例如10秒）
        """
        # 获取所有样本文件夹路径，假设命名规则为 "*_mon_*"
        all_folders = glob.glob(os.path.join(root_dir, "*_mon_*"))

        fr2_folders = [f for f in all_folders if 'Fr2' in os.path.basename(f)]
        other_folders = [f for f in all_folders if 'Fr2' not in os.path.basename(f)]
        if(mode == 'train'):
            self.sample_folders = other_folders
        elif (mode == 'test'):
            self.sample_folders = fr2_folders
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")
        # 选择 Fr2 文件夹作为测试集，其他文件夹作为训练集+验证集
        self.window_length = window_length
        self.step = step
        self.num_radars = num_radars
        self.mode = mode
        # 构建全局索引映射：每个元素为 (sample_idx, window_idx)
        self.indices = []
        for i, folder in enumerate(self.sample_folders):
            # 以 radar_0 的数据作为参考
            radar_folder = os.path.join(folder, "radar_0")
            real_path = os.path.join(radar_folder, "Zxx_real.csv")
            imag_path = os.path.join(radar_folder, "Zxx_imag.csv")
            real_part = load_csv_file(real_path)
            imag_part = load_csv_file(imag_path)
            Zxx = real_part + 1j * imag_part
            # Zxx.shape = [F, T_total]
            F, T_total = Zxx.shape
            N_windows = (T_total - window_length) // step + 1
            for win_idx in range(N_windows):
                self.indices.append((i, win_idx))
        print(f"{mode}dataset 总窗口样本数: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        根据全局索引返回一个窗口作为单个样本
        输出:
            radar_tensor: [num_radars, window_length, F]
            label_tensor: [num_classes]
        """
        sample_idx, win_idx = self.indices[idx]
        folder = self.sample_folders[sample_idx]
        radar_windows_list = []
        label_windows = None
        for r in range(self.num_radars):
            radar_folder = os.path.join(folder, f"radar_{r}")
            real_path = os.path.join(radar_folder, "Zxx_real.csv")
            imag_path = os.path.join(radar_folder, "Zxx_imag.csv")
            real_part = load_csv_file(real_path)
            imag_part = load_csv_file(imag_path)
            Zxx = real_part + 1j * imag_part

            Zxx_swapped = np.concatenate((Zxx[64:-1], Zxx[0:63]), axis=0)
            Zxx_magnitude = np.abs(Zxx_swapped)
            Zxx_db = 20 * np.log10(Zxx_magnitude + 1e-10)  # 避免 log(0) 错误
            vmin = np.percentile(Zxx_db, 5)  # 计算 5% 分位数
            vmax = np.percentile(Zxx_db, 95)  # 计算 95% 分位数
            Zxx_processed = np.clip(Zxx_db, vmin, vmax)  # 限制范围
            Zxx_processed = normalize_data(Zxx_processed)  # 归一化

            # 对每个雷达进行滑动窗口分割
            # 读取标签只一次
            if r == 0:
                lbl_path = os.path.join(radar_folder, "LBL.csv")
                LBL = pd.read_csv(lbl_path, header=None).values.astype(np.float32)
                if LBL.ndim == 1:
                    LBL = LBL.reshape(1, -1)

            windows, labels = sliding_window_with_hann(Zxx_processed, LBL, self.window_length, self.step)
            radar_windows_list.append(windows)  # windows: [N_windows, window_length, F]

            if r == 0:
                label_windows = labels  # [N_windows, num_classes]

        # 取出当前窗口 win_idx，构造每个窗口样本
        radar_sample = np.stack([w[win_idx] for w in radar_windows_list],
                                axis=0)  # shape: [num_radars, window_length, F]
        label_sample = label_windows[win_idx]  # shape: [num_classes]

        return torch.tensor(radar_sample, dtype=torch.float32), torch.tensor(label_sample, dtype=torch.float32)


def generate_dataset(save_dir):
    """
    leave-one-person-out (LOPO) 数据集的生成函数：
    将处理好的数据集中的每个窗口样本分别保存到对应的文件夹中：
    先按 80% 训练 / 10% 验证 / 10% 测试 划分，然后在每个部分中将每个窗口保存为单独的 .pt 文件。
    """
    mode1= 'train'
    mode2= 'test'

    dataset_train = RadarDataset(root_dir=root_dir, mode=mode1)

    # 划分数据集
    total_samples = len(dataset_train)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    train_size = int(0.8 * total_samples)
    val_size = int(0.2 * total_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    splits = {
        "train": Subset(dataset_train, train_indices),
        "val": Subset(dataset_train, val_indices),
    }

    for split, subset in splits.items():
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        count = 0
        for i in range(len(subset)):
            # 这里每个样本返回的是： radar_data: [num_radars, window_length, F] 和
            # labels: [num_classes]
            sample, label = subset[i]

            file_path = os.path.join(split_dir, f"{count}.pt")

            torch.save({"sample": torch.tensor(sample, dtype=torch.float32),
                        "label": torch.tensor(label, dtype=torch.float32)},
                       file_path)
            count += 1
            print(f"Processing {split} dataset_train---------- num {i + 1}/{len(subset)}")

        print(f"{split} dataset_train: 保存了 {count} 个窗口样本到 {split_dir}")

    # 处理测试集
    dataset_test = RadarDataset(root_dir=root_dir, mode=mode2)
    test_dir = os.path.join(save_dir, mode2)
    os.makedirs(test_dir, exist_ok=True)
    count = 0
    for i in range(len(dataset_test)):
        sample, label = dataset_test[i]
        file_path = os.path.join(test_dir, f"{count}.pt")
        torch.save({"sample": torch.tensor(sample, dtype=torch.float32),
                    "label": torch.tensor(label, dtype=torch.float32)},
                   file_path)
        count += 1
        print(f"Processing {mode2} dataset_train---------- num {i + 1}/{len(dataset_test)}")
    

if __name__ == "__main__":
    root_dir = "processed_data/Zxx"
    output_dir = "processed_data/Datasets_LOPO"

    generate_dataset(output_dir)

    """
    Sample data shape:     [num_radars, window_length, F]     [5, 3658, 126]
    Sample label shape:    [num_classes]    [10]
    """


