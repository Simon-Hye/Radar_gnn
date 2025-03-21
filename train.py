import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


from model.radar_gnn import SingleWindowRadarGNN
from loading_data import Wins_Dataset

def train_model(
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: str = "best_model.pth",
        log_dir: str = "runs/experiment"
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
        log_dir: Tensorboard 日志目录
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

    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(num_epochs):
        # ================= 训练阶段 =================
        model.train()
        epoch_train_loss = []

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)  # [B, 5, T, F]
            targets = targets.float().to(device)  # [B, num_classes]

            outputs = model(inputs)  # [B, num_classes]
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_train_loss.append(loss.item())
            writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)
            global_step += 1

        # 记录平均训练损失
        avg_train_loss = np.mean(epoch_train_loss)
        train_loss_history.append(avg_train_loss)
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)

        # ================= 验证阶段 =================
        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics_history.append(val_metrics)
        writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
        writer.add_scalar("Metrics/F1", val_metrics['f1'], epoch)
        writer.add_scalar("Metrics/AUC", val_metrics['auc'], epoch)
        writer.add_scalar("Metrics/Accuracy", val_metrics['accuracy'], epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # 学习率调度
        scheduler.step(val_metrics['loss'])

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch + 1}: save the best model (val_loss={best_val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")

    writer.close()

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

    train_folder = "processed_data/Datasets/train"
    valid_folder = "processed_data/Datasets/val"

    train_dataset = Wins_Dataset(train_folder)
    valid_dataset = Wins_Dataset(valid_folder)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    model = SingleWindowRadarGNN()

    train_loss_history, val_metrics_history, best_model = train_model(
        model, train_loader, valid_loader, num_epochs=10, lr=0.001, device="cuda", save_path="best_model.pth"
    )
    print("训练完成")
