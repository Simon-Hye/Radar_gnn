import torch
import torch.nn as nn
import torch.nn.functional as F


from .cnn import CNNFeatureExtractor
from .gat import RadarGATAggregator
from .classifier import RadarClassifier
from .graph_builder import StaticGraphBuilder

class SingleWindowRadarGNN(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 time_steps=30, 
                 doppler_bins=128,
                 radar_coords=None,
                 radar_feature_dim=64,
                 gat_out_channels=128,
                 gat_heads=2):
        """
        单窗口雷达图神经网络模型
        输入: [B, 5, time_steps, doppler_bins]，5个雷达的 Doppler-time 图
        输出: [B, num_classes] 多标签概率
        """
        super().__init__()
        
        # 构建静态图结构（基于雷达坐标）
        if radar_coords is None:
            # 默认雷达坐标，5个雷达对应
            radar_coords = torch.tensor([
                                [3.19,0.0], 
                                [3.19*1.414/2.0, -3.19*1.414/2.0,],
                                [0.0, -3.19],
                                [-3.19*1.414/2.0, -3.19*1.414/2.0,],
                                [-3.19,0.0]
                            ], dtype=torch.float)
            
        edge_index, edge_attr = StaticGraphBuilder.build(radar_coords)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        
        # 单雷达特征提取 CNN（输入：单个雷达的 Doppler-time 图 [B, 1, T, F]）
        self.radar_cnn = CNNFeatureExtractor(input_channels=1, output_dim=radar_feature_dim)
        
        # GAT 聚合模块：聚合 5 个雷达的特征
        self.gat_aggregator = RadarGATAggregator(in_channels=radar_feature_dim, 
                                                 out_channels=gat_out_channels, 
                                                 heads=gat_heads, 
                                                 edge_dim=1)
        
        # 全局特征聚合后，进入分类器
        global_feature_dim = 2 * (gat_out_channels * gat_heads)  # max pooling + mean pooling 拼接
        self.classifier = RadarClassifier(in_features=global_feature_dim, num_classes=num_classes)
    
    def forward(self, x):
        """
        Args:
            x: [B, 5, time_steps, doppler_bins]，5个雷达的 Doppler-time 图
        Returns:
            Tensor: [B, num_classes] 多标签概率
        """

        B = x.size(0)
        radar_feats = []
        # 分别处理 5 个雷达的数据
        for r in range(x.size(1)):  # x.size(1) 应为 5
            # 取出单个雷达数据：[B, time_steps, doppler_bins] -> 增加通道维度 -> [B, 1, T, F]
            single_radar = x[:, r].unsqueeze(1)
            feat = self.radar_cnn(single_radar)  # 得到 [B, radar_feature_dim]
            radar_feats.append(feat)
        
        # 构建节点特征矩阵：[B, 5, radar_feature_dim]
        node_feats = torch.stack(radar_feats, dim=1)
        
        # GAT 聚合：输出 [B, 5, gat_out_channels * gat_heads]
        graph_feats = self.gat_aggregator(node_feats, self.edge_index, self.edge_attr)
        
        # 全局特征聚合：对 5 个节点进行最大池化和均值池化，然后拼接 -> [B, 2 * (gat_out_channels*gat_heads)]
        max_pool = graph_feats.max(dim=1)[0]
        mean_pool = graph_feats.mean(dim=1)
        global_feat = torch.cat([max_pool, mean_pool], dim=1)
        
        # 分类输出
        return self.classifier(global_feat)



if __name__ == "__main__":
    # 参数设置
    batch_size = 32
    time_steps = 30
    doppler_bins = 128
    
    # 模拟输入：[B, 5, time_steps, doppler_bins]
    dummy_input = torch.randn(batch_size, 5, time_steps, doppler_bins)
    
    # 初始化模型（可通过构造函数参数调参）
    model = SingleWindowRadarGNN(num_classes=10)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # 预期输出: [32, 10]，即 10 个类别的概率分布