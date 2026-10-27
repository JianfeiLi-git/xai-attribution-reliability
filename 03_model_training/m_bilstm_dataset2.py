
"""
bilstm
修改：
1. Dataset1 使用5个随机种子，Dataset2 使用5个随机种子
2. 停牌过滤阈值从110天改为100天（更严格）
3. Bootstrap 验证
4. 保存所有训练过程数据和结果数据（用于后期可视化）
"""

import math
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# XAI相关库
import shap
from captum.attr import IntegratedGradients, DeepLift, GradientShap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==================== 设置随机种子 ====================

def set_seed(seed):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 1. 数据加载与序列构建 ====================

def load_data_raw(data_dir: Path):
    print(f"  [IO] Reading data from: {data_dir.name}")
    X_train = pd.read_csv(data_dir / "train_X.csv")
    X_val = pd.read_csv(data_dir / "val_X.csv")
    X_test = pd.read_csv(data_dir / "test_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv")
    y_val = pd.read_csv(data_dir / "val_y.csv")
    y_test = pd.read_csv(data_dir / "test_y.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



# ============ 新增：时间过滤函数 ============
def filter_data_by_date(X_df, y_df, start_date='2010-01-01'):
    """
    过滤训练数据，只保留指定日期之后的数据
    start_date: 起始日期，格式 'YYYY-MM-DD'
    """
    X_df['trade_date'] = pd.to_datetime(X_df['trade_date'])
    y_df['trade_date'] = pd.to_datetime(y_df['trade_date'])
    
    start_datetime = pd.to_datetime(start_date)
    
    X_filtered = X_df[X_df['trade_date'] >= start_datetime].copy()
    y_filtered = y_df[y_df['trade_date'] >= start_datetime].copy()
    
    print(f"  ⏰ 时间过滤: {start_date} 之后")
    print(f"     原始样本: {len(X_df):,} → 过滤后: {len(X_filtered):,}")
    
    return X_filtered, y_filtered
# ==========================================

def calculate_thresholds(y_df: pd.DataFrame, target_col="pct_chg"):
    train_returns = y_df[target_col].dropna()
    q33 = np.percentile(train_returns, 33.33)
    q67 = np.percentile(train_returns, 66.67)
    return q33, q67

def build_sequence_data_with_meta(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
    """构建序列数据，同时返回元数据"""
    # 1. 标签处理
    y_work = y_df.copy()
    y_work['trade_date'] = pd.to_datetime(y_work['trade_date'])
    y_work['future_change'] = y_work.groupby('stock_code')['pct_chg'].shift(-1)
    
    def get_label(x):
        if pd.isna(x): return -999
        if x < q33: return 0
        elif x > q67: return 2
        else: return 1
    y_work['label'] = y_work['future_change'].apply(get_label)
    
    # 2. 特征处理
    X_work = X_df.copy()
    X_work['trade_date'] = pd.to_datetime(X_work['trade_date'])
    
    # 3. 合并对齐
    merged = pd.merge(X_work, y_work[['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']], 
                      on=['stock_code', 'trade_date'], how='inner')
    
    # 确定特征列
    if feature_cols == 'ALL':
        exclude = ['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']
        valid_feats = [c for c in X_work.columns if c not in exclude]
    else:
        valid_feats = [f for f in feature_cols if f in X_work.columns]
    
    print(f"  Feature Count: {len(valid_feats)}")
    
    all_seqs = []
    all_labels = []
    all_meta = []
    
    print(f"  → 扫描 {merged['stock_code'].nunique()} 只股票序列...")
    
    for stock, g in merged.groupby('stock_code'):
        if len(g) < max_seq_len: continue
        
        g = g.sort_values('trade_date')
        feats = g[valid_feats].values
        dates = g['trade_date'].values
        labels = g['label'].values
        
        # 提取元数据
        meta_data = g[['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']].values
        
        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len
            
            # 【修正后的停牌过滤逻辑】
            # 停牌数据过滤 (使用自然日天数跨度检查)
            days_diff = (dates[end-1] - dates[start_idx]) / np.timedelta64(1, 'D')
            
            # max_seq_len=60 是交易日数量
            # 正常情况约84天 (60 ÷ 5 × 7，含周末)
            # 允许16天缓冲（长假+少量停牌）
            # 总阈值：100天
            # 原阈值110天过于宽松，改为100天过滤更多停牌数据
            if days_diff > 100:
                continue
            
            target = labels[end-1]
            if target == -999: continue
            
            # 特征序列
            seq_x = feats[start_idx : end]
            
            all_seqs.append(seq_x)
            all_labels.append(target)
            all_meta.append(meta_data[end-1])
            
    if not all_seqs:
        return np.empty((0, max_seq_len, len(valid_feats))), np.array([]), pd.DataFrame()
    
    # 重构 Meta DataFrame
    meta_df = pd.DataFrame(all_meta, columns=['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change'])
    
    return np.stack(all_seqs), np.array(all_labels, dtype=np.int64), meta_df

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==================== 2. 模型定义 (GRU & BiLSTM) ====================

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = torch.nan_to_num(x)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        logits = self.fc(last_output)
        return logits

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        # 双向，所以输入维度是 hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = torch.nan_to_num(x)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits

# ==================== 3. 训练与评估 ====================

def evaluate_model(model, loader, device, return_probs=False):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)
            pred = logits.argmax(-1)
            
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
            probs.append(prob.cpu().numpy())
            
    if not preds: return 0.0, 0.0, None, None
    
    y_p = np.concatenate(preds)
    y_t = np.concatenate(trues)
    y_prob = np.concatenate(probs, axis=0)
    
    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average='macro')
    
    if return_probs:
        return acc, f1, y_p, y_prob
    return acc, f1

def train_model(model, train_loader, val_loader, device, epochs=400, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    best_f1 = 0
    patience = 80
    counter = 0
    best_state = None
    
    # 历史记录容器
    history = []
    
    print(f"🚀 Start Training (Max {epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            
        # 验证
        val_acc, val_f1 = evaluate_model(model, val_loader, device)
        train_loss = np.mean(losses)
        curr_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_acc': float(val_acc),
            'val_f1': float(val_f1),
            'lr': float(curr_lr)
        })
        
        scheduler.step(val_f1)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d} | Loss={train_loss:.4f} | ValF1={val_f1:.4f} | LR={curr_lr:.2e}")
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break
    
    if best_state: model.load_state_dict(best_state)
    return model, history
#适用于dataset1
# def train_model_with_weight(model, train_loader, val_loader, device, class_weights, epochs=400, lr=5e-5):
#     """带类别权重的训练"""
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.7, patience=15, verbose=False
#     )
    
#     best_f1 = 0
#     patience = 50
#     counter = 0
#     best_state = None
#     history = []
#适用于dataset2
def train_model_with_weight(model, train_loader, val_loader, device, class_weights, epochs=400, lr=5e-5):
    """带类别权重的训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    best_f1 = 0
    patience = 60
    counter = 0
    best_state = None
    history = []
    
    print(f"🚀 Start Training with Class Weight (Max {epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            
        val_acc, val_f1 = evaluate_model(model, val_loader, device)
        train_loss = np.mean(losses)
        curr_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_acc': float(val_acc),
            'val_f1': float(val_f1),
            'lr': float(curr_lr)
        })
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d} | Loss={train_loss:.4f} | ValF1={val_f1:.4f} | LR={curr_lr:.2e}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        
        scheduler.step(val_f1)
        
        if counter >= patience:
            print(f"  🛑 Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history


# ==================== 4. Bootstrap 验证 ====================

def bootstrap_validation(y_test, y_pred, dataset_name, model_type, feat_desc, seed, save_dir, n_iterations=100):
    """Bootstrap验证"""
    print("\n" + "=" * 80)
    print(f"【Bootstrap验证 - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}】")
    print("=" * 80)
    print(f"执行 {n_iterations} 次Bootstrap重采样...")
    
    bootstrap_results = {'accuracy': [], 'f1': []}
    
    n_samples = len(y_test)
    
    for i in range(n_iterations):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{n_iterations}")
        
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_test_boot = y_test[indices]
        y_pred_boot = y_pred[indices]
        
        acc = accuracy_score(y_test_boot, y_pred_boot)
        f1 = f1_score(y_test_boot, y_pred_boot, average='macro')
        
        bootstrap_results['accuracy'].append(acc)
        bootstrap_results['f1'].append(f1)
    
    print(f"\nBootstrap结果 (95% 置信区间):")
    print(f"{'Metric':<10} {'Mean':<8} {'Std':<8} {'95% CI':<25}")
    print("-" * 60)
    
    bootstrap_summary = []
    
    for metric in ['accuracy', 'f1']:
        values = bootstrap_results[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        
        bootstrap_summary.append({
            'Metric': metric,
            'Mean': mean_val,
            'Std': std_val,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })
        
        print(f"{metric:<10} {mean_val:.4f}   {std_val:.4f}   [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    bootstrap_df = pd.DataFrame(bootstrap_summary)
    filename = f"bootstrap_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
    bootstrap_df.to_csv(save_dir / filename, index=False)
    print(f"\n✓ Bootstrap结果已保存到 {filename}")
    
    return bootstrap_df

# ==================== 6. XAI分析 (DL专用) ====================

def attention_based_importance(model, test_loader, device, feature_names):
    """
    基于梯度的特征重要性分析 (已修复 cuDNN 报错与引用报错)
    """
    import torch.nn as nn # 确保函数内能识别 nn 关键字
    
    model.train() 
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
            
    model.zero_grad()
    all_grads = []
    
    for xb, yb in test_loader:
        xb = xb.to(device)
        xb.requires_grad = True 
        
        with torch.set_grad_enabled(True):
            logits = model(xb)
            pred_class = logits.argmax(dim=1)
            
            for i in range(len(xb)):
                if xb.grad is not None:
                    xb.grad.zero_()
                model.zero_grad()
                
                # [cite_start]--- 核心修改：删除 [cite: 39] 标签 ---
                logits[i, pred_class[i]].backward(retain_graph=True)
                
                # 取该样本在所有时间步上的平均梯度
                grad = xb.grad[i].abs().mean(dim=0).cpu().detach().numpy()
                all_grads.append(grad)
    
    mean_grads = np.mean(all_grads, axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Gradient_Importance': mean_grads
    }).sort_values('Gradient_Importance', ascending=False)

    # 添加
    print("\n=== Gradient 完整结果 ===")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']:25s} {row['Gradient_Importance']:.6f}")
    # 添加结束
    
    model.eval()
    return importance_df



def integrated_gradients_analysis(model, train_loader, test_loader, device, feature_names, 
                                  dataset_name, model_type, feat_desc, seed, save_dir, n_samples=500):
    """
    Integrated Gradients分析 (彻底修复 cuDNN RNN 状态报错)
    """
    import torch.nn as nn
    
    # 1. 强制进入 train 模式以允许 backward,但手动锁定 Dropout 消除随机性
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    
    # 2. 局部禁用 cuDNN;这是解决此报错的最有效保险手段
    # 它会迫使 PyTorch 使用不依赖训练状态检查的普通算子计算梯度
    orig_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        # 准备数据 (保持不变)
        X_samples, y_samples = [], []
        for xb, yb in test_loader:
            X_samples.append(xb)
            y_samples.append(yb)
            if len(X_samples) * xb.size(0) >= n_samples:
                break
        X_test = torch.cat(X_samples, dim=0)[:n_samples].to(device)
        
        # 3. 基准值:使用训练集均值(符合无泄漏原则)
        ig = IntegratedGradients(model)
        
        # 准备训练集baseline
        X_train_samples = []
        for xb, _ in train_loader:
            X_train_samples.append(xb)
            if len(X_train_samples) * xb.size(0) >= n_samples:
                break
        X_train_subset = torch.cat(X_train_samples, dim=0)[:n_samples].to(device)
        baseline = X_train_subset.mean(dim=0, keepdim=True).expand_as(X_test)
        
        attributions_list = []
        batch_size = 50
        print(f"  计算Integrated Gradients (cuDNN已暂时禁用)...")
        
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            X_batch = X_test[i:end_idx]
            baseline_batch = baseline[i:end_idx]
            
            preds = model(X_batch).argmax(dim=1)
            attr_batch = ig.attribute(X_batch, baseline_batch, target=preds, n_steps=50)
            attributions_list.append(attr_batch.cpu().detach())
        
        attributions = torch.cat(attributions_list, dim=0)

        # 4. 修改归因逻辑:全时间步平均绝对值
        mean_attr = attributions.abs().mean(dim=[0, 1]).numpy()
        
        ig_importance = pd.DataFrame({
            'Feature': feature_names,
            'IG_Importance': mean_attr
        }).sort_values('IG_Importance', ascending=False)

        # 添加
        print("\n=== IG 完整结果 ===")
        for i, row in ig_importance.iterrows():
            print(f"{row['Feature']:25s} {row['IG_Importance']:.6f}")
        # 添加结束
        
        return ig_importance

    finally:
        # 5. 无论成功与否,恢复模型状态和 cuDNN 全局设置
        torch.backends.cudnn.enabled = orig_cudnn_state
        model.eval()



def shap_deepexplainer_analysis(model, train_loader, test_loader, device, feature_names,
                                 dataset_name, model_type, feat_desc, seed, save_dir,
                                 n_background=100, n_test=500):
    """
    改进版 SHAP 分析：自动识别特征维度，修复长度不匹配报错
    """
    import torch.nn as nn
    import shap
    import numpy as np
    import pandas as pd
    
    print("\n" + "=" * 80)
    print(f"【SHAP分析 - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}】")
    print("=" * 80)
    
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    
    orig_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    
    try:
        # 1. 准备数据
        bg_samples = []
        for xb, _ in train_loader:
            bg_samples.append(xb)
            if len(bg_samples) * xb.size(0) >= n_background: break
        background = torch.cat(bg_samples, dim=0)[:n_background].to(device)
        
        test_samples = []
        for xb, _ in test_loader:
            test_samples.append(xb)
            if len(test_samples) * xb.size(0) >= n_test: break
        test_data = torch.cat(test_samples, dim=0)[:n_test].to(device)
        
        # 2. 创建 Explainer 并计算
        print("  创建 SHAP GradientExplainer (cuDNN已暂时禁用)...")
        explainer = shap.GradientExplainer(model, background)
        
        print("  计算 SHAP 值...")
        shap_values = explainer.shap_values(test_data)
        
        # 3. 核心修复：自动对齐特征维度
        # 将列表或数组统一转为绝对值张量
        shap_array = np.abs(np.array(shap_values))
        print(f"  SHAP 原始形状: {shap_array.shape}")
        
        target_len = len(feature_names)
        
        # 寻找哪一个维度的大小等于特征数量 (13)
        # 根据日志 (500, 60, 13, 3)，特征维度的索引应该是 2
        feat_dim_idx = -1
        for i, dim in enumerate(shap_array.shape):
            if dim == target_len:
                feat_dim_idx = i
                break
        
        if feat_dim_idx != -1:
            # 压缩除了特征维度以外的所有维度
            axes_to_reduce = tuple(i for i in range(shap_array.ndim) if i != feat_dim_idx)
            final_importance = shap_array.mean(axis=axes_to_reduce)
            print(f"  已自动识别特征维度: Index {feat_dim_idx}, 结果长度: {len(final_importance)}")
        else:
            # 兜底：如果没找到13，尝试按最后的维度压缩
            print("  ⚠️ 未找到匹配的特征维度，执行强制重塑")
            final_importance = np.mean(shap_array.reshape(-1, target_len), axis=0)

        # 4. 构建 DataFrame
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': final_importance.flatten()
        }).sort_values('SHAP_Importance', ascending=False)
        
        print(f"\nSHAP GradientExplainer - Top 10 特征:")
        for i, row in shap_importance.head(10).iterrows():
            print(f"  {i+1:2d}. {row['Feature']:25s} {row['SHAP_Importance']:.6f}")

        # 添加
        print("\n=== SHAP 完整结果 ===")
        for i, row in shap_importance.iterrows():
            print(f"{row['Feature']:25s} {row['SHAP_Importance']:.6f}")
        # 添加结束
        
        filename = f"shap_importance_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
        shap_importance.to_csv(save_dir / filename, index=False)
        return shap_importance
        
    except Exception as e:
        print(f"\n❌ SHAP分析失败: {e}")
        return None
    finally:
        torch.backends.cudnn.enabled = orig_cudnn_state
        model.eval()



    
def temporal_attention_visualization(model, test_loader, device, feature_names,
                                     dataset_name, model_type, feat_desc, seed, save_dir, n_samples=100):
    """
    时间步重要性可视化 (修复 cuDNN RNN 报错)
    """
    import torch.nn as nn
    print("\n" + "=" * 80)
    print(f"【时间步重要性分析 - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}】")
    print("=" * 80)
    
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    orig_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    
    try:
        # 采样数据
        X_samples = []
        for xb, _ in test_loader:
            X_samples.append(xb)
            if len(X_samples) * xb.size(0) >= n_samples: break
        X_test = torch.cat(X_samples, dim=0)[:n_samples].to(device)
        
        timestep_importance = []
        print(f"  分析 {X_test.size(1)} 个时间步 (cuDNN已暂时禁用)...")
        
        for t in range(X_test.size(1)):
            X_test.requires_grad = True
            model.zero_grad()
            
            logits = model(X_test)
            pred_class = logits.argmax(dim=1)
            
            grads_at_t = []
            for i in range(len(X_test)):
                if X_test.grad is not None: X_test.grad.zero_()
                
                # 核心修正：backward 处于强制 train 状态
                logits[i, pred_class[i]].backward(retain_graph=True)
                grad_t = X_test.grad[i, t, :].abs().mean().item()
                grads_at_t.append(grad_t)
            
            timestep_importance.append(np.mean(grads_at_t))
            if (t + 1) % 20 == 0: print(f"    进度: 时间步 {t+1}/60")
        
        timestep_df = pd.DataFrame({
            'Timestep': range(1, len(timestep_importance) + 1),
            'Importance': timestep_importance
        })
        # 添加
        print("\n=== 时间步完整结果 ===")
        for i, row in timestep_df.iterrows():
            print(f"Step {row['Timestep']:0f}: {row['Importance']:.6f}")
        # 添加结束
        
        
        filename = f"timestep_importance_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
        timestep_df.to_csv(save_dir / filename, index=False)
        return timestep_df

    finally:
        torch.backends.cudnn.enabled = orig_cudnn_state
        model.eval()


def compare_xai_methods_dl(gradient_imp, ig_imp, shap_imp, 
                           dataset_name, model_type, feat_desc, seed, save_dir):
    """对比DL模型的不同XAI方法"""
    print("\n" + "=" * 80)
    print(f"【XAI方法一致性对比 - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}】")
    print("=" * 80)

    print("\n=== 调试信息 ===")
    print("IG前5特征值:", ig_imp['IG_Importance'].head().values)
    print("SHAP前5特征值:", shap_imp['SHAP_Importance'].head().values)
    print("数值是否完全相同:", np.allclose(
        ig_imp.sort_values('Feature')['IG_Importance'].values,
        shap_imp.sort_values('Feature')['SHAP_Importance'].values
    ))
    
    if shap_imp is None:
        print("⚠️ SHAP结果不可用,仅对比Gradient和IG方法")
        
        comparison_df = gradient_imp[['Feature', 'Gradient_Importance']].merge(
            ig_imp[['Feature', 'IG_Importance']], 
            on='Feature', 
            how='inner'
        )
        
        for col in ['Gradient_Importance', 'IG_Importance']:
            comparison_df[col] = (comparison_df[col] - comparison_df[col].min()) / \
                                (comparison_df[col].max() - comparison_df[col].min() + 1e-8)
        
        comparison_df['Gradient_Rank'] = comparison_df['Gradient_Importance'].rank(ascending=False)
        comparison_df['IG_Rank'] = comparison_df['IG_Importance'].rank(ascending=False)
        
        from scipy.stats import spearmanr
        corr_grad_ig, p_grad_ig = spearmanr(comparison_df['Gradient_Rank'], comparison_df['IG_Rank'])
        
        print(f"\n特征重要性方法一致性 (Spearman相关系数 + p值):")
        print(f"  Gradient vs IG: ρ={corr_grad_ig:.4f}, p={p_grad_ig:.6f}")
        
        correlations = {
            'gradient_ig': corr_grad_ig,
            'gradient_shap': None,
            'ig_shap': None
        }
        
    else:
        comparison_df = gradient_imp[['Feature', 'Gradient_Importance']].copy()
        comparison_df = comparison_df.merge(
            ig_imp[['Feature', 'IG_Importance']], 
            on='Feature', 
            how='outer'
        )
        
        if shap_imp is not None and len(shap_imp) > 0:
            comparison_df = comparison_df.merge(
                shap_imp[['Feature', 'SHAP_Importance']], 
                on='Feature', 
                how='outer'
            )
        else:
            comparison_df['SHAP_Importance'] = 0.0
            
        comparison_df = comparison_df.fillna(0.0)
        
        for col in ['Gradient_Importance', 'IG_Importance', 'SHAP_Importance']:
            comparison_df[col] = (comparison_df[col] - comparison_df[col].min()) / \
                                (comparison_df[col].max() - comparison_df[col].min() + 1e-8)
        
        comparison_df['Gradient_Rank'] = comparison_df['Gradient_Importance'].rank(ascending=False)
        comparison_df['IG_Rank'] = comparison_df['IG_Importance'].rank(ascending=False)
        comparison_df['SHAP_Rank'] = comparison_df['SHAP_Importance'].rank(ascending=False)
        
        from scipy.stats import spearmanr
        
        corr_grad_ig, p_grad_ig = spearmanr(comparison_df['Gradient_Rank'], comparison_df['IG_Rank'])
        
        if (comparison_df['SHAP_Importance'] > 0).sum() >= 3:
            corr_grad_shap, p_grad_shap = spearmanr(comparison_df['Gradient_Rank'], comparison_df['SHAP_Rank'])
            corr_ig_shap, p_ig_shap = spearmanr(comparison_df['IG_Rank'], comparison_df['SHAP_Rank'])
        else:
            corr_grad_shap, p_grad_shap = float('nan'), float('nan')
            corr_ig_shap, p_ig_shap = float('nan'), float('nan')
        
        print(f"\n特征重要性方法一致性 (Spearman相关系数 + p值):")
        print(f"  Gradient vs IG:   ρ={corr_grad_ig:.4f}, p={p_grad_ig:.6f}")
        print(f"  Gradient vs SHAP: ρ={corr_grad_shap:.4f}, p={p_grad_shap:.6f}")
        print(f"  IG vs SHAP:       ρ={corr_ig_shap:.4f}, p={p_ig_shap:.6f}")
        
        correlations = {
            'gradient_ig': corr_grad_ig,
            'gradient_shap': corr_grad_shap,
            'ig_shap': corr_ig_shap
        }
    
    filename = f"xai_comparison_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
    comparison_df.to_csv(save_dir / filename, index=False)
    print(f"\n✓ 对比结果已保存到 {filename}")
    
    return comparison_df, correlations
        



# ==================== 5. 主函数 ====================

def run_single_experiment(ds_key, ds_dir_name, feature_cols, feat_desc, model_type, seed, 
                         script_dir, results_dir, models_dir, device):
    """运行单次实验（一个模型+一个种子）"""
    print(f"\n{'='*80}")
    print(f"🔥 {model_type} | {ds_key} | {feat_desc} | Seed = {seed}")
    print(f"{'='*80}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 1. 加载数据
    data_dir = script_dir / 'datasets/processed' / ds_dir_name
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data_raw(data_dir)
    # ============ 新增：对dataset2的训练数据进行时间过滤 ============
    if ds_key == 'dataset2':
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr, start_date='2010-01-01')
    # ============================================================
    
    q33, q67 = calculate_thresholds(y_tr)
    
    print(f"📊 动态阈值: 33%={q33:.4f}%, 67%={q67:.4f}%")
    
    # 2. 构建序列
    print("构建序列...")
    X_train, y_train, _ = build_sequence_data_with_meta(X_tr, y_tr, feature_cols, q33, q67)
    X_val, y_val, _ = build_sequence_data_with_meta(X_val, y_val, feature_cols, q33, q67)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, feature_cols, q33, q67)
    
    print(f"📊 训练: {X_train.shape}, 验证: {X_val.shape}, 测试: {X_test.shape}")
    
    # 类别分布
    print(f"\n📊 测试集类别分布:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        names = {0: '下跌', 1: '持平', 2: '上涨'}
        pct = cnt/len(y_test)*100
        print(f"  {names[cls]}: {cnt:,} ({pct:.1f}%)")
    
    # 3. DataLoader
    batch_size = 256 if ds_key == 'dataset1' else 512 
    
    tr_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=512)
    te_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=512)
    

    # 4. 训练
    input_dim = X_train.shape[2]
    
    if ds_key == 'dataset1':
        # dataset1专用：添加类别权重
        class_weights = torch.FloatTensor([1.0, 1.0, 1.2]).to(device)
        print(f"   📊 类别权重（手动设置）: {class_weights.cpu().numpy()}")
        
        if model_type == 'GRU':
            model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
        else:  # bilstm
            model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
        
        model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, epochs=400, lr=5e-5)
    else:
        # dataset2专用：添加类别权重
        # class_weights = torch.FloatTensor([1.1, 1.0, 1.2]).to(device)
        # print(f"   📊 类别权重（手动设置）: {class_weights.cpu().numpy()}")
        if model_type == 'GRU':
            class_weights = torch.FloatTensor([1.1, 1.0, 1.1]).to(device)
            print(f"   📊 类别权重（手动设置）: {class_weights.cpu().numpy()}")
            model = GRUClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0).to(device)
            # lr=5e-5
            lr=3e-4
        else:  # bilstm
            class_weights = torch.FloatTensor([1.15, 1.0, 1.15]).to(device)
            print(f"   📊 类别权重（手动设置）: {class_weights.cpu().numpy()}")
            model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.1).to(device)
            lr=1e-4
        
        model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, epochs=400, lr=lr)
    
    # # 5. 训练
    # model, history = train_model(model, tr_loader, val_loader, device, epochs=400, lr=1e-4)
    
    # 6. 最终评估
    acc, f1, y_pred, y_prob = evaluate_model(model, te_loader, device, return_probs=True)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n🏆 最终结果: F1={f1:.4f} | Acc={acc:.4f}")
    print(f"   各类别F1: 下跌={f1_per_class[0]:.4f}, 持平={f1_per_class[1]:.4f}, 上涨={f1_per_class[2]:.4f}")
    
    # 7. 保存所有产物
    
    # A. 保存模型
    model_path = models_dir / f"{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"   💾 模型已保存: {model_path.name}")
    
    # B. 保存训练历史
    hist_path = results_dir / f"history_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   💾 训练历史已保存: {hist_path.name}")
    
    # C. 保存详细预测
    pred_df = meta_test.copy()
    pred_df['pred_label'] = y_pred
    pred_df['prob_0'] = y_prob[:, 0]
    pred_df['prob_1'] = y_prob[:, 1]
    pred_df['prob_2'] = y_prob[:, 2]
    pred_df['model'] = model_type
    pred_df['seed'] = seed
    
    pred_path = results_dir / f"preds_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"   💾 预测结果已保存: {pred_path.name}")
    
    # D. 保存混淆矩阵
    cm_df = pd.DataFrame(cm, 
                         index=['True_Down', 'True_Hold', 'True_Up'],
                         columns=['Pred_Down', 'Pred_Hold', 'Pred_Up'])
    cm_path = results_dir / f"confusion_matrix_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.csv"
    cm_df.to_csv(cm_path)
    print(f"   💾 混淆矩阵已保存: {cm_path.name}")
    
    # E. Bootstrap验证（仅第一个种子）
    if seed == 42:
        bootstrap_df = bootstrap_validation(
            y_test, y_pred, ds_key, model_type, feat_desc, seed, results_dir, n_iterations=100
        )

    # F. XAI分析（仅第一个种子）
    if seed == 42:
        print("\n" + "="*80)
        print("🔍 开始XAI深度分析...")
        print("="*80)
        
        # 获取特征名称
        if feature_cols == 'ALL':
            exclude = ['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']
            feat_names = [c for c in X_tr.columns if c not in exclude]
        else:
            feat_names = feature_cols
        
        # 1. 梯度重要性
        print("\n📊 方法1: 基于梯度的特征重要性")
        gradient_importance = attention_based_importance(model, te_loader, device, feat_names)
        grad_filename = f"gradient_importance_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.csv"
        gradient_importance.to_csv(results_dir / grad_filename, index=False)
        print(f"\nGradient - Top 10 特征:")
        for i, row in gradient_importance.head(10).iterrows():
            print(f"  {i+1:2d}. {row['Feature']:25s} {row['Gradient_Importance']:.6f}")
        print(f"✓ 已保存到 {grad_filename}")
        
        # 2. Integrated Gradients
        print("\n📊 方法2: Integrated Gradients")
        ig_importance = integrated_gradients_analysis(
        # 修改为
            model, tr_loader, te_loader, device, feat_names,  # 新增 tr_loader
            ds_key, model_type, feat_desc, seed, results_dir, n_samples=500
        )
          # 修改结束
        #     model, te_loader, device, feat_names,
        #     ds_key, model_type, feat_desc, seed, results_dir, n_samples=500
        # )
        
        # 3. SHAP DeepExplainer
        print("\n📊 方法3: SHAP DeepExplainer")
        shap_importance = shap_deepexplainer_analysis(
            model, tr_loader, te_loader, device, feat_names,
            ds_key, model_type, feat_desc, seed, results_dir,
            n_background=100, n_test=500
        )
        
        # 4. 时间步重要性
        print("\n📊 方法4: 时间步重要性分析")
        timestep_importance = temporal_attention_visualization(
            model, te_loader, device, feat_names,
            ds_key, model_type, feat_desc, seed, results_dir, n_samples=100
        )
        
        # 5. XAI方法对比
        print("\n📊 方法5: XAI方法一致性对比")
        comparison_df, correlations = compare_xai_methods_dl(
            gradient_importance, ig_importance, shap_importance,
            ds_key, model_type, feat_desc, seed, results_dir
        )
        
        print("\n✅ XAI分析完成！")
    
    return {
        'dataset': ds_key,
        'model': model_type,
        'feature_set': feat_desc,
        'seed': seed,
        'test_f1': f1,
        'test_acc': acc,
        'f1_down': f1_per_class[0],
        'f1_hold': f1_per_class[1],
        'f1_up': f1_per_class[2]
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    try: 
        script_dir = Path(__file__).resolve().parent
    except NameError: 
        script_dir = Path.cwd()
    
    # 目录设置
    results_dir = script_dir / 'results_XAI' / 'rnn_enhanced_results_dataset2'
    models_dir = script_dir / 'models_XAI' / 'rnn_enhanced_dataset2'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🚀 GRU & BiLSTM Enhanced Benchmark (修正版)")
    print("   - Dataset1: 5个种子")
    print("   - Dataset2: 5个种子")
    print("   - 停牌阈值: 100天 (从110天收紧)")
    print("=" * 80)
    
    # Dataset 2 的精选特征
    # SELECTED_FEATURES_13 = [
    #     'dayofyear', 'day', 'volume_ratio', 'close_to_ma5', 'vol', 'low', 'close_open_range', 'macd_signal', 'close_to_ma20', 
    #     'roc_20','high_low_range', 'turnover_rate', 'amount'
    # ]
    SELECTED_FEATURES_14 = [
        'high_low_range', 'turnover_rate', 'amount', 'bb_width', 'roc_10', 'roc_5', 'close_to_ma20', 'close_to_ma5', 
        'close_open_range', 'atr', 'volume_ratio', 'roc_20', 'dayofyear_cos', 'year'
    ]
    
    
    # 配置任务
    tasks = [
        # ('dataset1', 'dataset1_A_grade', 'ALL', 'ALL_18', 10),  # 10个种子
        # ('dataset1', 'dataset1_A_grade', SELECTED_FEATURES_13, 'SELECTED_13', 5),  # 5个种子
        ('dataset2', 'dataset2_AB_grade', SELECTED_FEATURES_14, 'SELECTED_14', 5),  # 5个种子
        # ('dataset2', 'dataset2_AB_grade', 'ALL', 'FULL_22', 5)  # 5个种子
    ]
    
    # 模型列表
    # models_to_run = ['GRU', 'BiLSTM']
    models_to_run = ['BiLSTM']
    
    # 定义种子池
    # seeds_10 = [10, 20, 30, 42, 100, 123, 456, 777, 888, 2024]
    # seeds_5 = [42, 123, 777, 0, 2024]
    seeds_5 = [42]
    
    all_results = []
    
    for ds_key, ds_dir_name, feature_cols, feat_desc, num_seeds in tasks:
        for model_type in models_to_run:
            print(f"\n\n{'='*80}")
            print(f"📊 开始实验: {model_type} - {ds_key} - {feat_desc} ({num_seeds}个种子)")
            print(f"{'='*80}")
            
            # 根据num_seeds选择种子列表
            seeds = seeds_10 if num_seeds == 10 else seeds_5
            
            for seed in seeds:
                result = run_single_experiment(
                    ds_key, ds_dir_name, feature_cols, feat_desc, model_type, seed,
                    script_dir, results_dir, models_dir, device
                )
                all_results.append(result)
            
            # 汇总该模型+配置的结果
            df = pd.DataFrame(all_results)
            df_task = df[(df['dataset'] == ds_key) & (df['model'] == model_type) & (df['feature_set'] == feat_desc)]
            
            print(f"\n{'='*80}")
            print(f"📊 {model_type} - {ds_key} - {feat_desc} 汇总（{len(seeds)}个种子）")
            print(f"{'='*80}")
            print(f"F1:  {df_task['test_f1'].mean():.4f} ± {df_task['test_f1'].std():.4f}")
            print(f"Acc: {df_task['test_acc'].mean():.4f} ± {df_task['test_acc'].std():.4f}")
    
    # 保存最终汇总
    final_df = pd.DataFrame(all_results)
    summary_path = results_dir / 'rnn_enhanced_summary_xai_dataset2_.csv'
    final_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  模型: {models_dir}")
    print(f"  结果: {results_dir}")
    print(f"  总汇总: {summary_path}")
    print(f"\n主要修改:")
    print(f"  ✓ Dataset1: 5个种子 (稳定性)")
    print(f"  ✓ Dataset2: 5个种子 (已经很稳定)")
    print(f"  ✓ 停牌阈值: 100天 (过滤更多停牌数据)")
    print("=" * 80)

if __name__ == "__main__":
    main()
