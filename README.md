# RadarGNN - 基于图神经网络的雷达信号分类系统

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.10%2B-orange)

本项目是基于图神经网络（GNN）的雷达信号分类系统，能够处理多雷达协同采集的时频图数据，实现高效的多标签动作分类。系统通过时空特征提取和图结构建模，有效捕捉雷达网络的空间关系与时序特征。

## 功能特性

-  **多雷达协同处理**：支持5个雷达的时频图联合分析
-  **混合神经网络架构**：CNN特征提取 + GAT图聚合
-  **时序建模**：支持30秒时间窗口的时空特征融合
-  **多标签分类**：10类动作的联合识别
-  **动态图构建**：基于雷达物理位置的动态邻接矩阵

## 环境依赖

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- NumPy 1.21+
- scikit-learn 1.0+

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-geometric
pip install scikit-learn
pip intall matplotlib


- best_model_v2 is literally the best one.
- 关于tensorboard存档的说明：./runs 下experience为LOPO训练过程，experience_v1为在原数据集上的多次训练过程，v2为对GAT的消融实验
- 对于Label3-6区分效果较差的可能原因：spectrograms are incapable of directly discriminating between translational movements such as walking and in-place movements such as falls.
- 改进后的CNN（更深，权重共享）
  - Test Loss: 0.0603 | F1: 0.6778 | AUC: 0.9259 | Hamming accuaracy:0.8216 | Accuracy: 0.2861

- 改进的classifier（标签关系建模）：
  - hidden_layer=256 效果很差 改为128之后效果明显
  - Test Loss: 0.0575 | F1: 0.7033 | AUC: 0.9344 | Hamming accuaracy:0.8520 | Total Accuracy: 0.3657


- 新增学习率调度ReduceLROnPlateau（patience=10） 
- LOPO在新数据集Datasets_LOPO上进行，Datasets_LOPO的test集为Fr2经过处理的所有样本，其余人的数据经过处理后随机抽取百分之八十为训练集，百分之二十为训练集(v2 and the newer versions are all employed in Datasets_LOPO)
  - Test Loss: 0.0463 | F1: 0.7621 | AUC: 0.9656 | Hamming accuaracy:0.8922 | Accuracy: 0.4925
  - CM_LOPO生成在.\figures 文件夹下
  - 消融实验：去除GAT组件，改为简单拼接5个CNN产生的特征向量，其余不变
    - Test Loss: 0.0533 | F1: 0.7341 | AUC: 0.9477 | Hamming accuaracy:0.8799 | Accuracy: 0.4478
