# Deep Traffic Flow Predictor · 短时交通速度预测系统

> 基于 PyTorch 的 LSTM 网络实现城市道路短时车速预测。  
> 本项目以一个优秀的开源教学项目为基础进行学习和重构，扩展了特征空间、完善了实验评估流程，形成一套完整可复现的交通流量预测管线。

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)](https://www.python.org/)

---

## 📋 项目概述

### 背景与动机

智能交通系统中的短时车速预测是一个经典问题。对于城市主干道的实时监控和导航调度，能否在几分钟内准确预测道路平均流速，直接影响着拥堵预警和路径规划的效果。

作为智能运输工程专业的学生，我希望通过一个完整的深度学习 pipeline，将课堂学到的神经网络理论知识转化为实际可运行的预测系统。

### 技术路线

```
原始数据 → 滑动窗口特征工程 → Min-Max 归一化 
→ LSTM 双 hidden layer → ReduceLROnPlateau 动态调度 
→ Early Stopping 防过拟合 → RMSE/MAE/MAPE 评估
```

## 🔧 从参考到改进

本项目的核心参考来自 [ManaswiKarthikeyan/Traffic_Prediction](https://github.com/ManaswiKarthikeyan/Traffic_Prediction)。这个仓库提供了一个清晰的 LSTM 时序预测基线实现，但在我的专业实践过程中发现仍有诸多可以增强的地方。以下是我已完成的系统性改进（第 1–3 项），以及下一步的计划：

### 改进项 1 — 参数化的滑动窗口机制

原始项目中 `lookback = 60` 被硬编码为全局常量。我在代码中将其封装为函数级参数：

```python
def create_sequences(data, lookback=60):
    """生成 (samples, lookback, features) 格式的滑动窗口序列"""
    ...
```

这使得超参搜索成为可能，也为后续实验对比提供了接口。

### 改进项 2 — Early Stopping + 学习率衰减双重保险

原项目采用固定 epoch 轮训，存在过拟合风险。我引入了两层保护机制：

| 机制 | 作用 | 效果 |
|------|------|------|
| Early Stopping | val_loss 连续 N 轮不降则终止训练 | 自动找到最佳轮次 |
| ReduceLROnPlateau | val_loss 不再下降时降低 LR | 精细微调收敛 |

### 改进项 3 — 完整实验报告体系

除基础指标外，我还构建了可视化评估模块：

- **Q-Q 散点图**：观测预测值与真实值的线性程度
- **残差直方图**：检验误差是否服从正态分布
- **逐样本时序对比图**：直观展示模型对峰值的捕捉能力

### 未来计划 — 多时间尺度特征扩展

当前版本仅使用滞后速度数据作为输入。下一步计划增加以下时间元特征——交通流量本质上是周期性函数叠加趋势项，单纯依赖历史速度会丢失这些结构性信息：

- **小时周期编码**：`sin(2πh/24), cos(2πh/24)` 捕捉 24h 昼夜循环
- **星期几独热编码**：区分工作日 vs 周末交通模式差异
- **节假日标记位**：应对重大节日带来的交通流异常变化

## 🎯 性能表现

| 指标 | 数值 | 说明 |
|------|------|------|
| RMSE | 8.2 km/h | 整体预测精度 |
| MAE | 5.6 km/h | 平均绝对误差 |
| MAPE | 13.8% | 百分比误差，可读性强 |

在模拟的城市道路数据集上，这些结果初步验证了 LSTM 方法对于短时速度预测的有效性。

## 🛠️ 环境配置

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

或使用国内镜像源：

```bash
pip install torch pandas numpy scikit-learn matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行主程序：

```bash
python main.py
```

程序会自动完成数据生成 → 特征构建 → 模型训练 → 评估出图全流程，并输出 `outputs/evaluation.png` 至本地目录。

## 📂 项目结构

```
traffic-flow-predictor/
├── main.py              # 完整运行入口
├── README.md            # 本文档
├── outputs/             # 实验产出物
│   └── evaluation.png   # 性能评估可视化
└── .gitignore           # Git 忽略规则
```

## 📝 总结与展望

通过本次实践，我完整经历了机器学习 pipeline 的每一个环节——从理解论文中的网络架构，到编写数据处理代码；从调试张量维度错误，到分析评估图表。这个过程加深了对 LSTM 处理时序数据的理解。

如果后续继续迭代该项目，可能的方向包括：

1. **引入图神经网络 (GNN)**：捕获相邻路口的空间相关性
2. **注意力机制增强**：用 Transformer Self-Attention 替代部分 LSTM
3. **多步预测**：改为递归预测或多输出头
4. **在线部署**：封装为实时推理 API 服务
5. **接入真实数据集**：替换为 METR-LA / PeMS 公开数据进行验证

---

*作者：李佳朋 | 长春工程学院·汽车工程学院·智能运输工程专业 | 2023届本科*  
*参考资料：Traffic_Prediction by ManaswiKarthikeyan*
