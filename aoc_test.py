import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch_geometric.data import DataLoader
from model.radar_gnn import SingleWindowRadarGNN
from loading_data import Wins_Dataset


def apply_thresholds(probs, thresholds):
    # probs: [N, num_classes], thresholds: [num_classes]
    return (probs >= thresholds).astype(int)


def find_best_threshold_per_class(y_true, y_scores):
    """
    对于多标签任务的每个类别，计算最佳阈值
    返回一个数组，每个元素为对应类别的最佳阈值。
    """
    num_classes = y_true.shape[1]
    best_thresholds = []
    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_scores[:, i])
        # Youden 指数: J = tpr - fpr, 找到使 J 最大的阈值
        J = tpr - fpr
        best_thresh = thresholds[np.argmax(J)]
        best_thresholds.append(best_thresh)
    return np.array(best_thresholds)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
valid_folder = "processed_data/Datasets/val"
valid_dataset = Wins_Dataset(valid_folder)
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
model = SingleWindowRadarGNN()
model.load_state_dict(torch.load("best_model.pth"))
model=model.to(device)
model.eval()  # 设置为评估模式

all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:  # val_loader 是验证数据集
        inputs = inputs.to(device)  # 移动到 GPU 或 CPU
        labels = labels.cpu().numpy()  # 保存真实标签

        outputs = model(inputs)  # 获取模型输出
        probs = torch.sigmoid(outputs).cpu().numpy()  # 应用 Sigmoid 获取概率

        all_probs.append(probs)
        all_labels.append(labels)

# 转换为 NumPy 数组
all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

thresholds = find_best_threshold_per_class(all_labels, all_probs)

print("最佳阈值 per class:", thresholds) #[0.529212   0.44142497 0.5154443  0.29661912 0.2954413  0.37115073  0.3486727  0.26898766 0.4172198  0.37225443]

val_pred_labels = apply_thresholds(all_probs, thresholds)
f1 = f1_score(all_labels, val_pred_labels, average='macro')
accuracy = accuracy_score(all_labels, val_pred_labels)

print(f"Validation F1: {f1:.4f}, Accuracy: {accuracy:.4f}")