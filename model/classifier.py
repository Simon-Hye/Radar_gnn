import torch.nn.functional as F
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, in_features=2048, hidden_features=128, num_classes=10,lamda=0.3):
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

        # 标签关系编码器，适配 GNN 输出的 feature_dim
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

        return (refined_logits + logits.squeeze(-1))  # 残差连接


class RadarClassifier(nn.Module):

    def __init__(self, feature_dim, num_classes, hidden_dim=128, relation_dim=128, dropout_prob=0.3):
        super().__init__()
        self.MLP = MLPClassifier(feature_dim, hidden_dim, num_classes, dropout_prob)
        self.label_relation_encoder = LabelRelationEncoder(feature_dim, num_classes, relation_dim)

    def forward(self, features):
        """
        Args:
            features: [B, feature_dim] GNN 提取的特征
        Returns:
            final_logits: [B, num_classes] 最终调整后的 logits
        """
        # 先用分类头生成初始 logits
        logits = self.MLP(features)  # [B, num_classes]

        # 再用 LabelRelationEncoder 调整 logits
        final_logits = self.label_relation_encoder(features, logits)
        return final_logits


if __name__ == "__main__":
    batch_size = 16
    feature_dim = 2048  # GNN 提取的特征维度
    num_classes = 10  # 分类类别数

    # 随机生成 batch_size 组 feature 向量，模拟 GNN 提取的特征
    features = torch.randn(batch_size, feature_dim)

    # 初始化模型
    model = RadarClassifier(feature_dim=feature_dim, num_classes=num_classes)

    # 前向传播
    final_logits = model(features)

    # 打印输出
    print("Final logits shape:", final_logits.shape)  # 预期：[batch_size, num_classes]