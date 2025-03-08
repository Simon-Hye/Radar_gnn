# RadarGNN - 基于图神经网络的雷达信号分类系统

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.10%2B-orange)

本项目是基于图神经网络（GNN）的雷达信号分类系统，能够处理多雷达协同采集的时频图数据，实现高效的多标签动作分类。系统通过时空特征提取和图结构建模，有效捕捉雷达网络的空间关系与时序特征。

## 功能特性

- 🛰️ **多雷达协同处理**：支持5个雷达的时频图联合分析
- 🧠 **混合神经网络架构**：CNN特征提取 + GAT图聚合
- ⏱️ **时序建模**：支持30秒时间窗口的时空特征融合
- 🏷️ **多标签分类**：10类动作的联合识别
- 📊 **动态图构建**：基于雷达物理位置的动态邻接矩阵

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
