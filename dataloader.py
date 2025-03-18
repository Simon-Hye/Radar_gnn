import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RadarDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (numpy array): 形状 (N, 5, T, F)，N 是样本数，5是五个雷达维度，T 是时间步长，F 是特征数
            labels (numpy array): 形状 (N, num_classes)，多标签分类
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class RadarDataModule:
    def __init__(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size=32):
        self.train_dataset = RadarDataset(train_data, train_labels)
        self.val_dataset = RadarDataset(val_data, val_labels)
        self.test_dataset = RadarDataset(test_data, test_labels)
        self.batch_size = batch_size

    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
if __name__ == "__main__":
    # 生成随机数据
    N = 1000
    C = 5
    T = 30
    F = 128
    num_classes = 10
    train_data = np.random.randn(N, C, T, F)
    train_labels = np.random.randint(0, 2, size=(N, num_classes))
    
    # 创建数据模块
    data_module = RadarDataModule(train_data, train_labels, train_data, train_labels, train_data, train_labels)
    
    # 测试数据加载器
    train_loader = data_module.train_loader()
    for batch in train_loader:
        inputs, targets = batch
        print(inputs.size(), targets.size())
        break