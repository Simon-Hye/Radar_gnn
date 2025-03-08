import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import dense_to_sparse

class StaticGraphBuilder:
    @staticmethod
    def build(coords, 
             mode='gaussian', 
             sigma=None,
             k_nearest=None,
             threshold=None,
             add_self_loops=True,
             normalize_weights=True):
        """
        
        参数:
            coords (Tensor): [num_nodes, 2] 雷达二维坐标
            mode (str): 相似度计算模式 ('gaussian', 'threshold', 'knn')
            sigma (float): 高斯核参数，None表示自动计算
            k_nearest (int): knn模式下的最近邻数
            threshold (float): 相似度阈值
            add_self_loops (bool): 是否添加自环
            normalize_weights (bool): 是否归一化边权重
            
        返回:
            edge_index (Tensor): [2, num_edges]
            edge_attr (Tensor): [num_edges, 1] 边权重
        """
        # 输入验证
        assert coords.dim() == 2 and coords.size(1) == 2, "坐标应为[N,2]矩阵"
        assert mode in ['gaussian', 'threshold', 'knn'], "不支持的建图模式"
        
        num_nodes = coords.size(0)
        device = coords.device
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(coords, coords)  # [N,N]
        
        # 自动计算sigma (平均距离的1/2)
        if sigma is None and mode == 'gaussian':
            sigma = torch.mean(dist_matrix) / 2
            sigma = max(sigma.item(), 1e-6)  # 防止除零
            
        # 构建邻接矩阵
        if mode == 'gaussian':
            adj = torch.exp(-dist_matrix**2 / (2 * sigma**2))
        elif mode == 'threshold':
            assert threshold is not None, "threshold模式需要指定阈值"
            adj = (dist_matrix < threshold).float()
        elif mode == 'knn':
            assert k_nearest is not None and 0 < k_nearest < num_nodes, "需要有效的k值"
            _, topk_indices = torch.topk(-dist_matrix, k=k_nearest+1, dim=1)  # +1包含自身
            adj = torch.zeros_like(dist_matrix)
            for i, indices in enumerate(topk_indices):
                adj[i, indices] = 1.0
            adj.fill_diagonal_(0)  # 后续统一处理自环
        
        # 添加自环
        if add_self_loops:
            adj += torch.eye(num_nodes, device=device)
            
        # 边过滤 (移除零权边)
        edge_index, edge_attr = dense_to_sparse(adj)
        
        # 权重归一化
        if normalize_weights:
            row_sum = torch.zeros(num_nodes, device=device)
            row_sum.scatter_add_(0, edge_index[0], edge_attr)
            edge_attr = edge_attr / (row_sum[edge_index[0]] + 1e-6)
        
        return edge_index, edge_attr.unsqueeze(-1)  # 边属性维度扩展

    @staticmethod
    def visualize(coords, edge_index):
        """可视化图结构 """
        
        plt.figure(figsize=(8, 6))
        # 绘制节点
        plt.scatter(coords[:,0], coords[:,1], s=200, c='red', zorder=2)
        
        # 绘制边
        for (src, dst) in edge_index.t().tolist():
            if src != dst:  # 不绘制自环
                plt.plot([coords[src,0], coords[dst,0]],
                        [coords[src,1], coords[dst,1]], 
                        'gray', alpha=0.5, zorder=1)
        
        plt.axis('equal')
        plt.title(f"Graph Structure ({edge_index.size(1)} edges)")
        plt.show()

if __name__ == "__main__":
    # 5个雷达节点坐标
    coords = torch.tensor([
        [3.19,0.0], 
        [3.19*1.414/2.0, -3.19*1.414/2.0,],
        [0.0, -3.19],
        [-3.19*1.414/2.0, -3.19*1.414/2.0,],
        [-3.19,0.0]
    ], dtype=torch.float)

    # 案例1：高斯相似度建图
    edge_index, edge_attr = StaticGraphBuilder.build(
        coords, 
        mode='gaussian',
        add_self_loops=True
    )
    print("高斯模式边数量:", edge_index.size(1))
    StaticGraphBuilder.visualize(coords, edge_index)

    # 案例2：KNN建图 (k=2)
    edge_index_knn, _ = StaticGraphBuilder.build(
        coords,
        mode='knn',
        k_nearest=2,
        add_self_loops=False
    )
    print("KNN模式边数量:", edge_index_knn.size(1))