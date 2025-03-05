import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_dim=64):
        """
        CNN用于提取单个雷达的Doppler-time图特征
        Args:
            input_channels (int): 输入图像的通道数
            output_dim (int): 输出特征向量维度
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, T, F]，单个雷达的 Doppler-time图
        Returns:
            Tensor: [B, output_dim]
        """
        x = self.cnn(x)  # 输出 [B, 64, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
if __name__ == '__main__':

    model = CNNFeatureExtractor()

    # 生成随机输入 [B, 1, T, F]，假设 T=64, F=64, B=4
    B, C, T, F = 4, 1, 64, 64  # 批量大小、通道数、时间步、频率维度
    input_tensor = torch.randn(B, C, T, F)

    # 前向传播测试
    output = model(input_tensor)

    # 输出形状
    print("Output shape:", output.shape)  # 预期输出: [B, output_dim]，即 [4, 64]
    