import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class RadarGATAggregator(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, heads=2, edge_dim=1):
        """
        利用 GATv2Conv 层聚合来自多个雷达的特征
        Args:
            in_channels (int): 每个节点的输入特征维度
            out_channels (int): 每个注意力头的输出维度
            heads (int): 注意力头数量
            edge_dim (int): 边属性维度
        """
        super().__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads=heads, edge_dim=edge_dim, concat=True)
        self.register_buffer("offsets", None, persistent=False)  # 缓存偏移量
    
    def precompute_offsets(self, B, num_nodes, device):
        if self.offsets is None or self.offsets.size(0) < B:
            self.offsets = torch.arange(B, device=device).view(-1, 1) * num_nodes

    def forward(self, node_feats, edge_index, edge_attr):
        """
        Args:
            node_feats: [B, num_nodes, in_channels] - 每个样本的节点特征
            edge_index: [2, num_edges] - 静态图的边索引（所有样本共享）
            edge_attr: [num_edges, edge_dim] - 每条边的属性（所有样本共享）
        Returns:
            Tensor: [B, num_nodes, out_channels * heads] - 经过 GAT 处理后的节点特征
        """
        B, num_nodes, feat_dim = node_feats.shape
        
        self.precompute_offsets(B, num_nodes, node_feats.device)

        edge_index_batch = (edge_index.unsqueeze(0) + self.offsets[:B].unsqueeze(-1)).view(2, -1)

        # 将节点特征扁平化以适应 GAT 层
        node_feats_flat = node_feats.view(-1, feat_dim)

        # 扩展 edge_attr
        edge_attr_batch = edge_attr.repeat(B, 1)

        # GAT 前向传播
        out = self.gat(node_feats_flat, edge_index_batch, edge_attr_batch)

        # 重新调整形状为 [B, num_nodes, out_channels * heads]
        out = out.view(B, num_nodes, -1)

        return out
    
if(__name__ == '__main__'):
    B = 4  # batch size
    num_nodes = 5  # 每个样本的雷达数量
    in_channels = 64  # 输入特征维度
    out_channels = 128  # GAT输出维度
    heads = 2  # 注意力头数
    edge_dim = 1  # 边属性维度
    num_edges = 6  # 图中的边数

    # 生成随机节点特征 [B, num_nodes, in_channels]
    node_feats = torch.randn(B, num_nodes, in_channels,requires_grad=True)

    # 生成静态图的 edge_index [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 生成边属性 [num_edges, edge_dim]
    edge_attr = torch.randn(num_edges, edge_dim)

    # 创建 GAT 聚合模型
    model = RadarGATAggregator(in_channels, out_channels, heads, edge_dim)

    model.forward(node_feats, edge_index, edge_attr)
    # 期望的偏移量
    expected_offsets = torch.tensor([0, num_nodes, 2 * num_nodes, 3 * num_nodes]).view(-1, 1)

    # 断言检查
    assert torch.equal(model.offsets, expected_offsets), f"Expected {expected_offsets}, but got {model.offsets}"
    print("recompute_offsets 计算正确")

    # 前向传播
    output = model(node_feats, edge_index, edge_attr)

    # 输出形状检查
    print("输入 node_feats 形状:", node_feats.shape)  # [B, num_nodes, in_channels]
    print("输出特征 形状:", output.shape)  # [B, num_nodes, out_channels * heads]

     # 计算损失
    loss = output.sum()
    loss.backward()  # 计算梯度

    # 检查梯度是否计算
    print("node_feats.grad is None:", node_feats.grad is None)
    if node_feats.grad is not None:
        print("node_feats.grad 形状:", node_feats.grad.shape)
