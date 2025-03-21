import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

class Wins_Dataset(Dataset):
    def __init__(self, folder):
        """
        Args:
            folder (str): 存储 .pt 文件的文件夹路径， "processed_data/Datasets/train"
        """
        # 获取文件夹下所有 .pt 文件
        self.files = glob.glob(os.path.join(folder, "*.pt"))
        self.files.sort()  # 可选：排序保证顺序一致

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载 .pt 文件，返回字典
        data_dict = torch.load(self.files[idx],weights_only=False)
        sample = data_dict['sample']  # radar 数据，例如 shape: [num_radars, window_length, F]
        label = data_dict['label']    # 标签，例如 shape: [num_classes]
        return sample, label

if __name__ == "__main__":
    # 定义训练、验证、测试集的文件夹
    train_folder = "processed_data/Datasets/train"
    valid_folder = "processed_data/Datasets/valid"
    test_folder  = "processed_data/Datasets/test"

    # 创建数据集实例
    train_dataset = Wins_Dataset(train_folder)
    valid_dataset = Wins_Dataset(valid_folder)
    test_dataset  = Wins_Dataset(test_folder)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 检查一个 batch 的数据形状
    for samples, labels in train_loader:
        print("Train batch sample shape:", samples.shape)  # torch.Size([4, 5, 3658, 128])
        print("Train batch label shape:", labels.shape)      # torch.Size([4, 10])
        break

