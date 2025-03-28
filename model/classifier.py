
import torch.nn as nn

class RadarClassifier(nn.Module):
    def __init__(self, in_features=512, hidden_features=128, num_classes=10,lamda=0.3):
        """
        分类头，用于将全局图特征映射到类别概率
        Args:
            in_features (int): 输入特征维度
            hidden_features (int): 隐藏层维度
            num_classes (int): 类别数
            lamda (float): Dropout概率
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(lamda),
            nn.Linear(hidden_features, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(x)