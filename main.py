"""
基于 LSTM 的深度短时交通速度预测系统
======================================

项目概述：
    针对城市主干道交通流量时空特性，构建深度学习模型实现未来短时（5分钟粒度）
    路段车速预测。完整的机器学习管线：数据生成→特征工程→模型训练→评估部署。

技术栈：Python · PyTorch · Scikit-learn · Pandas · NumPy
数据集：合成城市道路时序数据（模拟真实传感器采样频率），含高峰/平峰周期模式

核心改进（对比原始参考项目 ManaswiKarthikeyan/Traffic_Prediction）：
    1. 多尺度滑动窗口：支持灵活配置 lookback 窗口长度
    2. 早停机制 + 学习率动态衰减：自动识别最佳训练轮次，防止过拟合
    3. 可复现实验报告：自动生成 RMSE/MAE/MAPE 指标与预测可视化
未来计划：
    - 扩展特征空间：除滞后速度外，增加小时周期编码、星期独热、节假日标记等时间元特征

作者：李佳朋 · 智能运输工程专业 · 长春工程学院 · 2023届本科
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ─── 实验配置 ─────────────────────────────────────────────
CONFIG = {
    'seed': 42,
    'num_samples': 8000,
    'freq_min': 5,                # 采样间隔（分钟）
    'lookback': 12,               # 回溯窗口（历史时间步数）
    'horizon': 1,                 # 预测步长（向前时间步数）
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'max_epochs': 20,
    'patience': 7,                # 早停耐心值
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 节：合成交通速度数据集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_traffic_data(config):
    """
    基于周期性模式的合成交通速度序列。
    
    设计思路：
    - 早晚高峰时段 (7:00-9:00, 17:00-19:00) 车速显著下降至 ~25 km/h
    - 午间时段 (11:00-13:00) 车速中等 ~45 km/h  
    - 夜间及清晨车速平稳高企 ~60 km/h
    - 叠加正态噪声模拟真实传感器的随机波动
    
    Returns:
        pd.DataFrame: 含 datetime 和 speed 列的完整时序表
    """
    seed = config['seed']
    np.random.seed(seed)
    
    n = config['num_samples']
    timestamps = pd.date_range('2025-01-01', periods=n, freq=f"{config['freq_min']}min")
    
    speeds = []
    for ts in timestamps:
        h = ts.hour
        # 分时段设定基准速度并叠加高斯噪声
        if 7 <= h <= 9 or 17 <= h <= 19:       # 通勤高峰
            base = 25 + np.random.normal(0, 3)
        elif 11 <= h <= 13:                     # 午间
            base = 45 + np.random.normal(0, 5)
        else:                                     # 非高峰
            base = 60 + np.random.normal(0, 5)
        
        speeds.append(np.clip(base, 5, 120))      # 物理约束：速度 [5, 120]
    
    return pd.DataFrame({'datetime': timestamps, 'speed': speeds})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 节：滑动窗口特征工程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TrafficFeatureEngineer:
    """
    从一维速度序列构造多变量训练样本。
    
    每个样本包含两个部分：
      X — 过去 LOOKBACK 个时间步的速度历史（形状: [n_steps, n_features]）
      y — HORIZON 步之后的真实速度值
    """
    
    def __init__(self, lookback, horizon):
        self.lookback = lookback
        self.horizon = horizon
    
    def transform(self, speed_series):
        """将一维数组转为 (samples, timesteps, features) 格式"""
        arr = np.asarray(speed_series).flatten()
        X_list, y_list = [], []
        
        for i in range(self.lookback, len(arr) - self.horizon):
            segment = arr[i - self.lookback : i]
            X_list.append(segment.reshape(-1, 1))
            y_list.append(arr[i + self.horizon])
        
        return np.array(X_list), np.array(y_list)
    
    def split(self, X, y):
        """按 train / val / test 比例划分数据集"""
        total = len(X)
        tr_end = int(total * CONFIG['train_ratio'])
        va_end = int(total * (CONFIG['train_ratio'] + CONFIG['val_ratio']))
        
        return (
            X[:tr_end], y[:tr_end],
            X[tr_end:va_end], y[tr_end:va_end],
            X[va_end:],   y[va_end:]
        )
    
    @staticmethod
    def normalize(split_tuple):
        """
        对 (X_train, y_train, X_val, y_val, X_test, y_test) 
        做 Min-Max 归一化到 [0, 1]。
        
        注意：仅用训练集参数 fit_transform，验证/测试集只用 transform，
              防止数据泄露 (Data Leakage)。
        """
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_tuple
        
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # reshape → flatten → fit → reshape（保留三维结构）
        X_tr_s = scaler_X.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
        X_va_s = scaler_X.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
        X_te_s = scaler_X.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)
        
        y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
        y_va_s = scaler_y.transform(y_va.reshape(-1, 1)).flatten()
        y_te_s = scaler_y.transform(y_te.reshape(-1, 1)).flatten()
        
        return (
            X_tr_s, y_tr_s,
            X_va_s, y_va_s,
            X_te_s, y_te_s
        ), (scaler_X, scaler_y)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 节：LSTM 预测模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TrafficSpeedLSTM(torch.nn.Module):
    """
    双层 LSTM + 全连接层的短时交通速度预测网络。
    
    结构图：
        Input [batch, seq_len, 1]
              │
          ┌───┴───┐
          │ LSTM x2│  hidden=64, dropout=0.2
          └───┬───┘
          取最后一个 time_step → [batch, 64]
              │
          Linear(64 → 32) → ReLU → Dropout(0.2)
              │
          Linear(32 → 1) → squeeze
              │
         Output [batch]  预测未来 HORIZON 步的速度
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)          # 输出最后一层所有 time_step
        last_hidden = out[:, -1, :]    # 只取末时刻隐藏状态
        return self.head(last_hidden).squeeze(-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 节：训练管线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, train_loader, val_loader, device):
    """
    带早停和学习率调度的训练循环。
    
    Returns:
        dict: 包含 train_losses, val_losses, best_epoch, early_stop_reason
    """
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    history = {
        'train_losses': [], 'val_losses': [], 
        'best_epoch': 0, 'early_stop_reason': None,
        'lr_schedule': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'─'*50}")
    print("开始训练 ...")
    print(f"{'─'*50}")
    print(f"{'Epoch':>6s}  {'Train Loss':>12s}  {'Val Loss':>11s}  {'LR':>8s}  {'记录'}")
    
    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        train_loss_sum = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()           # 清空旧梯度
            pred = model(X_batch)           # 前向
            loss = criterion(pred, y_batch) # MSE 损失
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新参数
            
            train_loss_sum += loss.item()
        
        avg_train = train_loss_sum / len(train_loader)
        
        # ── 验证 ──
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss_sum += criterion(model(X_b), y_b).item()
        
        avg_val = val_loss_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        
        history['train_losses'].append(avg_train)
        history['val_losses'].append(avg_val)
        history['lr_schedule'].append(current_lr)
        
        mark = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            patience_counter = 0
            mark = " ★ 新最佳"
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                history['early_stop_reason'] = f"连续{patience_counter}轮未改善"
                print(f"\n⏹ 早停! {history['early_stop_reason']}")
                break
        
        print(f"{epoch:>6d}  {avg_train:>12.6f}  {avg_val:>11.6f}  {current_lr:>8.1e}{mark}")
    
    history['final_epoch'] = epoch
    print(f"\n✓ 训练完成 | 最佳 Epoch={history['best_epoch']} | ValLoss={best_val_loss:.6f}")
    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 节：评估与可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate_and_visualize(model, test_X, test_y, scalers, history, device):
    """在测试集上评估并生成三张核心图：收敛曲线 · 预测对比 · 误差分布"""
    scaler_X, scaler_y = scalers
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    model.eval()
    
    with torch.no_grad():
        preds_scaled = model(test_X.to(device)).cpu().numpy()
    
    # 反归一化回物理单位 (km/h)
    preds_orig = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    true_orig  = scaler_y.inverse_transform(test_y.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((preds_orig - true_orig) ** 2))
    mae  = np.mean(np.abs(preds_orig - true_orig))
    mape = np.mean(np.abs((true_orig - preds_orig) / true_orig)) * 100
    
    print(f"\n{'─'*50}")
    print("测试集评估指标:")
    print(f"  RMSE: {rmse:.2f} km/h   (均方根误差，越小越好)")
    print(f"  MAE:  {mae:.2f} km/h   (平均绝对误差)")
    print(f"  MAPE: {mape:.2f}%       (平均绝对百分比误差)")
    print(f"  解读: 预测均值偏离真实值约 {mae:.1f} km/h ({mape:.1f}% 相对误差)\n")
    
    # ── 绘图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (1) 损失收敛
    ax = axes[0]
    ax.plot(history['train_losses'], label='Train', color='#3498db', linewidth=1.5)
    ax.plot(history['val_losses'],   label='Validation', color='#e74c3c', linewidth=1.5)
    ax.axvline(x=history['best_epoch']-1, color='green', linestyle='--', alpha=0.6, label='Best')
    ax.set_title('训练收敛曲线', fontsize=12)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # (2) 最后 200 个时间步：预测 vs 真实
    ax = axes[1]
    show_n = min(200, len(true_orig))
    offset = len(true_orig) - show_n
    ax.plot(range(show_n), true_orig[offset:], color='#3498db', lw=1.5, label='Ground Truth')
    ax.plot(range(show_n), preds_orig[offset:], color='#e74c3c', lw=1.5, label='Prediction')
    ax.set_title('测试末尾段 · 预测对比', fontsize=12)
    ax.set_xlabel('Time Step'); ax.set_ylabel('Speed (km/h)')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # (3) 残差分布
    ax = axes[2]
    residuals = preds_orig - true_orig
    ax.hist(residuals, bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', lw=2)
    ax.set_title('预测残差分布 (越窄越集中越好)', fontsize=12)
    ax.set_xlabel('Error (km/h)'); ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'evaluation.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"📊 评估图已保存: {fig_path}\n")
    
    # 返回指标字典（方便后续写进 README）
    return {'RMSE': f'{rmse:.2f}', 'MAE': f'{mae:.2f}', 'MAPE': f'{mape:.2f}%'}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    cfg = CONFIG
    np.random.seed(cfg['seed'])
    
    print("=" * 60)
    print("  短时交通速度预测系统 · Deep Traffic Flow Predictor")
    print("  作者: 李佳朋 | 智能运输工程 · 2023届")
    print(f"  框架: PyTorch {torch.__version__} | GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # --- 1. 生成数据 ---
    print("\n[1/5] 生成交通速度数据集 ...")
    df = generate_traffic_data(cfg)
    print(f"  ✓ {len(df)} 条记录 · 平均速度 {df['speed'].mean():.1f} km/h")
    
    # --- 2. 特征工程 ---
    print("\n[2/5] 滑动窗口特征构造 ...")
    engineer = TrafficFeatureEngineer(cfg['lookback'], cfg['horizon'])
    X, y = engineer.transform(df['speed'])
    print(f"  ✓ 序列化后: {len(X)} 样本 · 输入 shape={X.shape}")
    
    # --- 2b. 数据划分 & 归一化 ---
    print("\n[2b/5] 数据划分 & 归一化 ...")
    split_tuple = engineer.split(X, y)
    normalized, scalers = engineer.normalize(split_tuple)
    train_X, train_y, val_X, val_y, test_X, test_y = normalized
    scaler_X, scaler_y = scalers
    print(f"  Train:{len(train_X)}  Val:{len(val_X)}  Test:{len(test_X)}")
    
    # --- 3. 构建模型 ---
    print("\n[3/5] 构建 LSTM 预测网络 ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficSpeedLSTM(input_dim=1, hidden_dim=cfg['hidden_dim'], 
                             num_layers=cfg['num_layers'], dropout=cfg['dropout'])
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ {params:,} 可训练参数\n{model}")
    
    # --- 4. 加载数据 & 训练 ---
    print("\n[4/5] 数据加载 & 模型训练 ...")
    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32)
    ), batch_size=cfg['batch_size'], shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32)
    ), batch_size=cfg['batch_size'], shuffle=False)
    
    history = train_model(model, train_loader, val_loader, device)
    
    # --- 5. 评估 & 出图 ---
    print("\n[5/5] 测试集评估 ...")
    metrics = evaluate_and_visualize(
        model, torch.tensor(test_X, dtype=torch.float32).to(device),
        test_y, (scaler_X, scaler_y), history, device
    )
    
    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("  ✅ 全部流程完成")
    print("=" * 60)
    print(f"  📦 输出目录: {OUTPUT_DIR}")
    print(f"  🗂️  包含:")
    print(f"     - best_model.pth    (最佳模型权重)")
    print(f"     - evaluation.png    (三合一评估图)")
    print(f"  📊 最终指标: RMSE={metrics['RMSE']} · MAE={metrics['MAE']} · MAPE={metrics['MAPE']}")


if __name__ == '__main__':
    main()
