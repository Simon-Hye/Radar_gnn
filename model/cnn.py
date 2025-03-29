import torch
import torch.nn as nn

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        """
        CNN用于提取单个雷达的Doppler-time图特征
        输入: [B, 1, T, F].
        输出: [B, output_dim].
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),  # 降维过快的问题
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4))  # 保留一定的空间信息
        )
        self.fc = nn.Linear(128 * 8 * 4, output_dim)  # 线性层映射到更高维度

    def forward(self, x):
        x = self.cnn(x)  # [B, 128, 8, 4]
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # [B, output_dim]
        return x


class RadarFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, cnn_out_dim=256):
        super().__init__()
        self.shared_cnn = CNNFeatureExtractor(input_channels, cnn_out_dim)

    def forward(self, x):
        # x 形状: [B, 5, T, F]
        B, C, T, F = x.shape
        x = x.view(B * C, 1, T, F)  # 合并 batch 和通道
        x = self.shared_cnn(x)  # CNN 处理, [B*5, output_dim]
        x = x.view(B, C, -1)  # 恢复通道维度
        return x  # 输出 [B, 5, output_dim]


if __name__ == '__main__':
    model = RadarFeatureExtractor()

    # 生成随机输入 [B, 1, T, F]，
    B, C, T, F = 8, 5, 3658, 126  # 批量大小、通道数、时间步、频率维度
    input_tensor = torch.randn(B, C, T, F)

    # 前向传播测试
    output = model(input_tensor)

    # 输出形状
    print("Output shape:", output.shape)  # 预期输出: [8, 5, 256], 8个样本，每个样本5个雷达，每个雷达256维特征
