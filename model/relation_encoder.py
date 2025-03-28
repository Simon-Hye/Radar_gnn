import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelRelationEncoder(nn.Module):
    """标签关系建模模块"""
    def __init__(self, feature_dim=2048, num_classes=10, relation_dim=128):
        """
        Args:
            feature_dim (int): GNN 提取的特征维度
            num_classes (int): 标签数量
            relation_dim (int): 关系映射的隐藏维度
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 标签关系编码器，适配 CNN 输出的 feature_dim
        self.relation_encoder = nn.Sequential(
            nn.Linear(feature_dim, relation_dim),  
            nn.ReLU(),
            nn.Linear(relation_dim, num_classes * num_classes)  
        )
        
        # 关系矩阵初始化
        nn.init.xavier_uniform_(self.relation_encoder[-1].weight)  # Xavier 初始化
        self.relation_encoder[-1].bias.data.zero_()  # 偏置设为0
        
    def forward(self, features, logits):
        """
        Args:
            features: [B, feature_dim] GNN 提取的特征
            logits: [B, C] 原始分类 logits
        Returns:
            refined_logits: [B, C] 优化后的 logits
        """
        # 生成关系矩阵 [B, C*C] → [B, C, C]
        relation_weights = self.relation_encoder(features)  
        relation_matrix = relation_weights.view(-1, self.num_classes, self.num_classes)
        relation_matrix = F.softmax(relation_matrix, dim=-1)  # 归一化到概率分布

        # 应用关系调整 [B, C] -> [B, C, 1]
        logits = logits.unsqueeze(-1)  
        refined_logits = torch.bmm(relation_matrix, logits).squeeze(-1)  
        
        return refined_logits + logits.squeeze(-1)  # 残差连接
if __name__ == '__main__':
    batch_size = 16
    feature_dim = 2048  
    num_classes = 10  

    # 随机生成特征和logits
    features = torch.randn(batch_size, feature_dim)  # 模拟 GNN/CNN 提取的特征
    logits = torch.randn(batch_size, num_classes)  # 模拟初始 logits

    # 创建模型
    model = LabelRelationEncoder(feature_dim=feature_dim, num_classes=num_classes)

    # 前向传播
    refined_logits = model(features, logits)
    
    print(refined_logits.shape) # torch.Size([16, 10])