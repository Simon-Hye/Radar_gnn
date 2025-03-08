import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader


from model.radar_gnn import SingleWindowRadarGNN
from loading_data import RadarDataModule


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "best_model.pth"
):
    """多标签分类模型训练验证流程
    
    Args:
        model: 初始化的模型实例
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        num_epochs: 训练轮次
        lr: 初始学习率
        device: 训练设备
        save_path: 最佳模型保存路径
    Returns:
        Tuple: (训练损失记录, 验证指标记录, 最佳模型)
    """
    # 初始化配置
    model = model.to(device)
    criterion = nn.BCELoss()  # 配合最后的Sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    train_loss_history = []
    val_metrics_history = []

    for epoch in range(num_epochs):
        # ================= 训练阶段 =================
        model.train()
        epoch_train_loss = []
        
        for batch in train_loader:
            # 数据加载 (假设数据加载器返回 (inputs, targets))
            inputs, targets = batch
            inputs = inputs.to(device)  # [B,5,T,F]
            targets = targets.float().to(device)  # [B,10]
            
            # 前向传播
            outputs = model(inputs)  # [B,10]
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
            optimizer.step()
            
            epoch_train_loss.append(loss.item())
        
        # 记录平均训练损失
        avg_train_loss = np.mean(epoch_train_loss)
        train_loss_history.append(avg_train_loss)
        
        # ================= 验证阶段 =================
        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics_history.append(val_metrics)
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: 保存最佳模型 (val_loss={best_val_loss:.4f})")
        
        # 打印日志
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f}")
    
    return train_loss_history, val_metrics_history, model

def evaluate_model(model, data_loader: DataLoader, device: str = "cuda"):
    """模型验证/测试函数
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
    Returns:
        Dict: 包含loss, f1, auc等指标
    """
    model.eval()
    criterion = nn.BCELoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 转换为numpy数组
            preds = outputs.cpu().numpy()
            labels = targets.cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(labels)
            total_loss += loss.item() * inputs.size(0)
    
    # 合并所有批次的预测结果
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # 计算指标
    avg_loss = total_loss / len(data_loader.dataset)
    pred_labels = (all_preds > 0.5).astype(int)  # 阈值设为0.5
    
    metrics = {
        'loss': avg_loss,
        'f1': f1_score(all_targets, pred_labels, average='macro'),
        'auc': roc_auc_score(all_targets, all_preds, average='macro'),
        'accuracy': accuracy_score(all_targets, pred_labels)
    }
    return metrics


if __name__ == "__main__":
    # 生成随机数据
    N = 1000
    C = 5
    T = 30
    F = 128
    num_classes = 10
    train_data = np.random.randn(N ,C, T, F)
    train_labels = np.random.randint(0, 2, size=(N, num_classes))
    
    # 创建数据模块
    data_module = RadarDataModule(train_data, train_labels, train_data, train_labels, train_data, train_labels)
    
    # 测试数据加载器
    train_loader = data_module.train_loader()
    for batch in train_loader:
        inputs, targets = batch
        print(f"inputs size: {inputs.size()} targets size: {targets.size()}")
        break
    
    # 初始化模型
    model = SingleWindowRadarGNN( num_classes=num_classes)
    
    # 训练模型
    train_loss_history, val_metrics_history, best_model = train_model(
        model, train_loader, train_loader, num_epochs=10, device="cpu"
    )
    print("训练完成")
