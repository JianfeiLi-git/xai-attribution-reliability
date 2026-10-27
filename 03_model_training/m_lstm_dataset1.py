"""
LSTM 静态基准模型 (完整增强版 - 修正版)
修改：
1. Dataset1 使用5个随机种子
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# ==================== 2. LSTM 模型定义 ====================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
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
        # x = torch.nan_to_num(x)
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
#dataset1
def train_model(model, train_loader, val_loader, device, epochs=400, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=10, verbose=False
    )
    
    best_f1 = 0
    patience = 50
    counter = 0
    best_state = None

#dataset2
# def train_model(model, train_loader, val_loader, device, epochs=400, lr=1e-4):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
#     criterion = nn.CrossEntropyLoss()
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=10, verbose=False
#     )
    
#     best_f1 = 0
#     patience = 60
#     counter = 0
#     best_state = None
    
    
    # 历史记录容器
    history = []
    
    print(f"🚀 Start LSTM Training (Max {epochs} epochs)...")
    
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

def train_model_with_weight(model, train_loader, val_loader, device, class_weights, epochs=400, lr=5e-5):
    """带类别权重的训练--dataset1"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 使用类别权重
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15, verbose=False
    )
    
    best_f1 = 0
    patience = 50
    counter = 0
    best_state = None
    history = []
# def train_model_with_weight(model, train_loader, val_loader, device, class_weights, epochs=400, lr=5e-5):
#     """带类别权重的训练  dataset2 """
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)  # 使用类别权重
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=15, verbose=False
#     )
    
#     best_f1 = 0
#     patience = 60
#     counter = 0
#     best_state = None
#     history = []
    
    print(f"🚀 Start LSTM Training with Class Weight (Max {epochs} epochs)...")
    
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

def bootstrap_validation(y_test, y_pred, dataset_name, feat_desc, seed, save_dir, n_iterations=100):
    """Bootstrap验证"""
    print("\n" + "=" * 80)
    print(f"【Bootstrap验证 - {dataset_name} - {feat_desc} - Seed {seed}】")
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
    filename = f"bootstrap_{dataset_name}_{feat_desc}_seed{seed}.csv"
    bootstrap_df.to_csv(save_dir / filename, index=False)
    print(f"\n✓ Bootstrap结果已保存到 {filename}")
    
    return bootstrap_df

# ==================== 6. XAI分析 (LSTM专用) ====================

def attention_based_importance_lstm(model, test_loader, device, feature_names):
    """基于梯度的特征重要性分析 (适用于LSTM)"""
    model.train() 
    for module in model.modules():
        if isinstance(module, nn.Dropout): module.eval()
            
    model.zero_grad()
    all_grads = []
    print("  计算 LSTM 梯度重要性...")
    for xb, yb in test_loader:
        xb = xb.to(device)
        xb.requires_grad = True
        with torch.set_grad_enabled(True):
            logits = model(xb)
            pred_class = logits.argmax(dim=1)
            for i in range(len(xb)):
                if xb.grad is not None: xb.grad.zero_()
                model.zero_grad()
                logits[i, pred_class[i]].backward(retain_graph=True)
                # 取全时间步平均梯度，避免局部噪声
                grad = xb.grad[i].abs().mean(dim=0).cpu().detach().numpy()
                all_grads.append(grad)
    
    mean_grads = np.mean(all_grads, axis=0)
    # 添加
    df = pd.DataFrame({'Feature': feature_names, 'Gradient_Importance': mean_grads}).sort_values('Gradient_Importance', ascending=False)
    print("\n=== Gradient 完整结果 ===")
    for i, row in df.iterrows():
        print(f"{row['Feature']:25s} {row['Gradient_Importance']:.6f}")
    return df
    
    # return pd.DataFrame({'Feature': feature_names, 'Gradient_Importance': mean_grads}).sort_values('Gradient_Importance', ascending=False)

def ig_analysis_lstm(model, train_loader, test_loader, device, feature_names, save_dir, ds_key, feat_desc, seed, n_samples=500):
    """Integrated Gradients 分析 (针对 LSTM 优化)"""
    from captum.attr import IntegratedGradients
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.eval()
    
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        X_samples = []
        for xb, _ in test_loader:
            X_samples.append(xb)
            if len(X_samples) * xb.size(0) >= n_samples: break
        X_test = torch.cat(X_samples, dim=0)[:n_samples].to(device)
        ig = IntegratedGradients(model)
        
# 开始修改
# 删除这行
        # baseline = X_test.mean(dim=0, keepdim=True).expand_as(X_test)
        
# 添加训练集baseline
        X_train_samples = []
        for xb, _ in train_loader:
            X_train_samples.append(xb)
            if len(X_train_samples) * xb.size(0) >= n_samples:
                break
        X_train_subset = torch.cat(X_train_samples, dim=0)[:n_samples].to(device)
        baseline = X_train_subset.mean(dim=0, keepdim=True).expand_as(X_test)




        
        attributions = []
        for i in range(0, len(X_test), 50):
            X_batch = X_test[i:i+50]
            preds = model(X_batch).argmax(dim=1)
            attr = ig.attribute(X_batch, baseline[i:i+50], target=preds, n_steps=50)
            attributions.append(attr.cpu().detach())
        
        attr_final = torch.cat(attributions, dim=0).abs().mean(dim=(0, 1)).numpy()
        df = pd.DataFrame({'Feature': feature_names, 'IG_Importance': attr_final}).sort_values('IG_Importance', ascending=False)
        df.to_csv(save_dir / f"ig_lstm_{ds_key}_{feat_desc}_seed{seed}.csv", index=False)
        # 添加
        print("\n=== IG 完整结果 ===")
        for i, row in df.iterrows():
            print(f"{row['Feature']:25s} {row['IG_Importance']:.6f}")
        # 添加结束
        return df
    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()

def shap_analysis_lstm(model, train_loader, test_loader, device, feature_names, save_dir, ds_key, feat_desc, seed):
    """SHAP GradientExplainer 分析 (支持维度自动对齐)"""
    import shap
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.eval()
    
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        bg = next(iter(train_loader))[0][:100].to(device)
        te = next(iter(test_loader))[0][:200].to(device)
        explainer = shap.GradientExplainer(model, bg)
        shap_values = explainer.shap_values(te)
        shap_array = np.abs(np.array(shap_values)) # (类别, 样本, 时间步, 特征)
        
        # 自动锁定特征维度 (13或14)
        target_len = len(feature_names)
        feat_dim = [i for i, dim in enumerate(shap_array.shape) if dim == target_len][0]
        axes = tuple(i for i in range(shap_array.ndim) if i != feat_dim)
        final_imp = shap_array.mean(axis=axes)
        
        df = pd.DataFrame({'Feature': feature_names, 'SHAP_Importance': final_imp}).sort_values('SHAP_Importance', ascending=False)
        df.to_csv(save_dir / f"shap_lstm_{ds_key}_{feat_desc}_seed{seed}.csv", index=False)
        # 添加
        print("\n=== SHAP 完整结果 ===")
        for i, row in df.iterrows():
            print(f"{row['Feature']:25s} {row['SHAP_Importance']:.6f}")
        # 添加结束
        return df
    except Exception as e:
        print(f"  ❌ SHAP 失败: {e}"); return None
    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()

def temporal_importance_lstm(model, test_loader, device, feature_names, save_dir, ds_key, feat_desc, seed, n_samples=100):
    """时间步重要性分析"""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.eval()
    
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    
    try:
        X_samples = []
        for xb, _ in test_loader:
            X_samples.append(xb)
            if len(X_samples) * xb.size(0) >= n_samples: break
        X_test = torch.cat(X_samples, dim=0)[:n_samples].to(device)
        
        timestep_importance = []
        print(f"  分析 {X_test.size(1)} 个时间步...")
        
        for t in range(X_test.size(1)):
            X_test.requires_grad = True
            model.zero_grad()
            
            logits = model(X_test)
            pred_class = logits.argmax(dim=1)
            
            grads_at_t = []
            for i in range(len(X_test)):
                if X_test.grad is not None: X_test.grad.zero_()
                logits[i, pred_class[i]].backward(retain_graph=True)
                grad_t = X_test.grad[i, t, :].abs().mean().item()
                grads_at_t.append(grad_t)
            
            timestep_importance.append(np.mean(grads_at_t))
        
        df = pd.DataFrame({
            'Timestep': range(1, len(timestep_importance) + 1),
            'Importance': timestep_importance
        })
        
        print("\n=== 时间步完整结果 ===")
        for i, row in df.iterrows():
            print(f"Step {row['Timestep']:0f}: {row['Importance']:.6f}")
        
        return df
    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()


# ==================== 5. 主函数 ====================

def run_single_experiment(ds_key, ds_dir_name, feature_cols, feat_desc, seed, 
                         script_dir, results_dir, models_dir, device):
    """运行单次实验（一个种子）"""
    print(f"\n{'='*80}")
    print(f"🔥 {ds_key} | {feat_desc} | Seed = {seed}")
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
    # batch_size = 256
    batch_size = 256 if ds_key == 'dataset1' else 512 
    tr_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=512)
    te_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=512)
    

    # 4. 训练
    input_dim = X_train.shape[2]
    
    if ds_key == 'dataset1':
        # 手动设置类别权重，加大上涨类权重
        class_weights = torch.FloatTensor([1.0, 1.0, 1.15]).to(device)  # 上涨类权重×2.5
        print(f"   📊 类别权重（手动设置）: {class_weights.cpu().numpy()}")
        
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.35).to(device)
        model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, 
                                                   epochs=400, lr=1e-4)
    else:
        # dataset2 手动设置类别权重，加大上涨类权重
        class_weights = torch.FloatTensor([1.3, 1.0, 1.3]).to(device)  
        print(f"   📊 类别权重（手动设置）: {class_weights.cpu().numpy()}")
        
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0).to(device)
        model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, 
                                                   epochs=400, lr=1e-4)


    
    # 5. 最终评估
    acc, f1, y_pred, y_prob = evaluate_model(model, te_loader, device, return_probs=True)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n🏆 最终结果: F1={f1:.4f} | Acc={acc:.4f}")
    print(f"   各类别F1: 下跌={f1_per_class[0]:.4f}, 持平={f1_per_class[1]:.4f}, 上涨={f1_per_class[2]:.4f}")

    # === 新增：XAI 深度分析模块 ===
    if seed == 42:
        print("\n" + "="*80)
        print("🔍 开始 LSTM 深度 XAI 分析 (仅针对 Seed 42)...")
        print("="*80)
        
        # 动态获取特征名称
        if feature_cols == 'ALL':
            exclude = ['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']
            feat_names = [c for c in X_tr.columns if c not in exclude]
        else:
            feat_names = feature_cols
        
        # 1. 梯度分析
        g_imp = attention_based_importance_lstm(model, te_loader, device, feat_names)
        print(f"  ✓ Gradient 分析完成")
        
        # 2. IG 分析
        # ig_imp = ig_analysis_lstm(model, te_loader, device, feat_names, results_dir, ds_key, feat_desc, seed)
        ig_imp = ig_analysis_lstm(model, tr_loader, te_loader, device, feat_names, results_dir, ds_key, feat_desc, seed)
        print(f"  ✓ IG 分析完成")
        
        # 3. SHAP 分析
        s_imp = shap_analysis_lstm(model, tr_loader, te_loader, device, feat_names, results_dir, ds_key, feat_desc, seed)
        print(f"  ✓ SHAP 分析完成")

        # 在shap_analysis_lstm调用之后添加
        timestep_df = temporal_importance_lstm(model, te_loader, device, feat_names, results_dir, ds_key, feat_desc, seed)


        # 4. 一致性对比 - 【完整版本，与GRU/BiLSTM一致】
        if g_imp is not None and ig_imp is not None:
            comparison_df = g_imp[['Feature', 'Gradient_Importance']].merge(
                ig_imp[['Feature', 'IG_Importance']], 
                on='Feature', 
                how='inner'
            )
            
            if s_imp is not None:
                comparison_df = comparison_df.merge(
                    s_imp[['Feature', 'SHAP_Importance']], 
                    on='Feature', 
                    how='inner'
                )
            
            # 归一化到[0,1]
            for col in ['Gradient_Importance', 'IG_Importance']:
                comparison_df[col] = (comparison_df[col] - comparison_df[col].min()) / \
                                    (comparison_df[col].max() - comparison_df[col].min() + 1e-8)
            
            if s_imp is not None:
                comparison_df['SHAP_Importance'] = (comparison_df['SHAP_Importance'] - comparison_df['SHAP_Importance'].min()) / \
                                                   (comparison_df['SHAP_Importance'].max() - comparison_df['SHAP_Importance'].min() + 1e-8)
            
            # 计算排名
            comparison_df['Gradient_Rank'] = comparison_df['Gradient_Importance'].rank(ascending=False)
            comparison_df['IG_Rank'] = comparison_df['IG_Importance'].rank(ascending=False)
            if s_imp is not None:
                comparison_df['SHAP_Rank'] = comparison_df['SHAP_Importance'].rank(ascending=False)
            
            # 🔧 核心：使用Rank计算相关性，并保存p值
            from scipy.stats import spearmanr
            
            corr_gi, p_gi = spearmanr(comparison_df['Gradient_Rank'], comparison_df['IG_Rank'])
            
            print(f"\n  📊 一致性 (Spearman相关系数 + p值):")
            print(f"     Gradient vs IG:   ρ={corr_gi:.4f}, p={p_gi:.6f}")
            
            if s_imp is not None and (comparison_df['SHAP_Importance'] > 0).sum() >= 3:
                corr_gs, p_gs = spearmanr(comparison_df['Gradient_Rank'], comparison_df['SHAP_Rank'])
                corr_is, p_is = spearmanr(comparison_df['IG_Rank'], comparison_df['SHAP_Rank'])
                print(f"     Gradient vs SHAP: ρ={corr_gs:.4f}, p={p_gs:.6f}")
                print(f"     IG vs SHAP:       ρ={corr_is:.4f}, p={p_is:.6f}")
            
            comparison_df.to_csv(results_dir / f"xai_comparison_lstm_{ds_key}_{feat_desc}_seed{seed}.csv", index=False)
            print(f"  ✓ XAI对比结果已保存")
    
    # 6. 保存所有产物
    
    # A. 保存模型
    model_path = models_dir / f"lstm_{ds_key}_{feat_desc}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"   💾 模型已保存: {model_path.name}")
    
    # B. 保存训练历史
    hist_path = results_dir / f"history_lstm_{ds_key}_{feat_desc}_seed{seed}.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   💾 训练历史已保存: {hist_path.name}")
    
    # C. 保存详细预测
    pred_df = meta_test.copy()
    pred_df['pred_label'] = y_pred
    pred_df['prob_0'] = y_prob[:, 0]
    pred_df['prob_1'] = y_prob[:, 1]
    pred_df['prob_2'] = y_prob[:, 2]
    pred_df['seed'] = seed
    
    pred_path = results_dir / f"preds_lstm_{ds_key}_{feat_desc}_seed{seed}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"   💾 预测结果已保存: {pred_path.name}")
    
    # D. 保存混淆矩阵
    cm_df = pd.DataFrame(cm, 
                         index=['True_Down', 'True_Hold', 'True_Up'],
                         columns=['Pred_Down', 'Pred_Hold', 'Pred_Up'])
    cm_path = results_dir / f"confusion_matrix_lstm_{ds_key}_{feat_desc}_seed{seed}.csv"
    cm_df.to_csv(cm_path)
    print(f"   💾 混淆矩阵已保存: {cm_path.name}")
    
    # E. Bootstrap验证（仅第一个种子）
    if seed == 42:
        bootstrap_df = bootstrap_validation(
            y_test, y_pred, ds_key, feat_desc, seed, results_dir, n_iterations=100
        )
    
    return {
        'dataset': ds_key,
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
    results_dir = script_dir / 'results_XAI' / 'lstm_enhanced_results'
    models_dir = script_dir / 'models_XAI' / 'lstm_enhanced'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🚀 LSTM Enhanced Benchmark (修正版)")
    print("   - Dataset1: 5个种子")
    print("   - Dataset2: 5个种子")
    print("   - 停牌阈值: 100天 (从110天收紧)")
    print("=" * 80)

    
    # Dataset 2 的精选特征
    SELECTED_FEATURES_13 = [
        'dayofyear', 'day', 'volume_ratio', 'close_to_ma5', 'vol', 'low', 'close_open_range', 'macd_signal', 'close_to_ma20', 
        'roc_20','high_low_range', 'turnover_rate', 'amount'
    ]
    # SELECTED_FEATURES_14 = [
    #     'high_low_range', 'turnover_rate', 'amount', 'bb_width', 'roc_10', 'roc_5', 'close_to_ma20', 'close_to_ma5', 
    #     'close_open_range', 'atr', 'volume_ratio', 'roc_20', 'dayofyear_cos', 'year'
    # ]
    
    # 配置任务
    tasks = [
        # ('dataset1', 'dataset1_A_grade', 'ALL', 'ALL_18', 10),  # 10个种子
        ('dataset1', 'dataset1_A_grade', SELECTED_FEATURES_13, 'SELECTED_13', 5),  # 5个种子
        # ('dataset2', 'dataset2_AB_grade', SELECTED_FEATURES_14, 'SELECTED_14', 5),  # 5个种子
        # ('dataset2', 'dataset2_AB_grade', 'ALL', 'FULL_22', 5)  # 5个种子
    ]
    
    # 定义种子池
    # seeds_10 = [10, 20, 30, 42, 100, 123, 456, 777, 888, 2024]
    # seeds_5 = [42, 123, 777, 0, 2024]
    seeds_5 = [42]
    
    all_results = []
    
    for ds_key, ds_dir_name, feature_cols, feat_desc, num_seeds in tasks:
        print(f"\n\n{'='*80}")
        print(f"📊 开始实验: {ds_key} - {feat_desc} ({num_seeds}个种子)")
        print(f"{'='*80}")
        
        # 根据num_seeds选择种子列表
        seeds = seeds_10 if num_seeds == 10 else seeds_5
        
        for seed in seeds:
            result = run_single_experiment(
                ds_key, ds_dir_name, feature_cols, feat_desc, seed,
                script_dir, results_dir, models_dir, device
            )
            all_results.append(result)
        
        # 汇总该配置的结果
        df = pd.DataFrame(all_results)
        df_task = df[(df['dataset'] == ds_key) & (df['feature_set'] == feat_desc)]
        
        print(f"\n{'='*80}")
        print(f"📊 {ds_key} - {feat_desc} 汇总（{len(seeds)}个种子）")
        print(f"{'='*80}")
        print(f"F1:  {df_task['test_f1'].mean():.4f} ± {df_task['test_f1'].std():.4f}")
        print(f"Acc: {df_task['test_acc'].mean():.4f} ± {df_task['test_acc'].std():.4f}")
    
    # 保存最终汇总
    final_df = pd.DataFrame(all_results)
    summary_path = results_dir / 'lstm_enhanced_xai_summary.csv'
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
