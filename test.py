import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch_geometric.data import DataLoader
from model.radar_gnn import SingleWindowRadarGNN
from loading_data import Wins_Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from train import evaluate_model, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_folder = "processed_data/Datasets_LOPO/test"
test_dataset = Wins_Dataset(test_folder)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
model = SingleWindowRadarGNN()
model.load_state_dict(torch.load("best_model.pth"))
model=model.to(device)

metrics, y_pred, y_true=evaluate_model(model,test_loader,'cuda')

print(f"Test Loss: {metrics['loss']:.4f} | F1: {metrics['f1']:.4f} | "
      f"AUC: {metrics['auc']:.4f} | Hamming accuaracy:{metrics['hamming_accuracy']:.4f} |"
      f" Accuracy: {metrics['accuracy']:.4f}")

conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

num_classes = y_true.shape[1]
fig, axes = plt.subplots(1, num_classes, figsize=(12, 4))

for i in range(num_classes):
    sns.heatmap(conf_matrices[i], annot=True, fmt="d", cmap="Blues",
                annot_kws={"size": 14},
                cbar=False, ax=axes[i])

    axes[i].set_title(f"Label {i}", fontsize=14)
    axes[i].set_xlabel("Predicted", fontsize=12)
    axes[i].set_ylabel("Actual", fontsize=12)

plt.tight_layout()
plt.show()