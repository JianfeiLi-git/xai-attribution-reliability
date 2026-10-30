
"""
BiLSTM Static baseline models (full enhanced version - corrected version)
Change:
1. Dataset1 uses 5 random seeds, Dataset2 uses 5 random seeds
2. Suspension filter threshold changed from 110 to 100 days (stricter)
3. Bootstrap Validation
4. Save all training process data and result data (for later visualization)
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
# XAI Related libraries
import shap
from captum.attr import IntegratedGradients, DeepLift, GradientShap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==================== Set random seeds ====================

def set_seed(seed):
    """Set all random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 1. Data loading and sequence construction ====================

def load_data_raw(data_dir: Path):
    print(f"  [IO] Reading data from: {data_dir.name}")
    X_train = pd.read_csv(data_dir / "train_X.csv")
    X_val = pd.read_csv(data_dir / "val_X.csv")
    X_test = pd.read_csv(data_dir / "test_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv")
    y_val = pd.read_csv(data_dir / "val_y.csv")
    y_test = pd.read_csv(data_dir / "test_y.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



# ============ New: time filter function ============
def filter_data_by_date(X_df, y_df, start_date='2010-01-01'):
    """
    Filter the training data and keep only data after the given date
    start_date: Start date, format 'YYYY-MM-DD'
    """
    X_df['trade_date'] = pd.to_datetime(X_df['trade_date'])
    y_df['trade_date'] = pd.to_datetime(y_df['trade_date'])

    start_datetime = pd.to_datetime(start_date)

    X_filtered = X_df[X_df['trade_date'] >= start_datetime].copy()
    y_filtered = y_df[y_df['trade_date'] >= start_datetime].copy()

    print(f"  ⏰ Time filter: {start_date} after")
    print(f"     Original samples: {len(X_df):,} → After filtering: {len(X_filtered):,}")

    return X_filtered, y_filtered
# ==========================================

def calculate_thresholds(y_df: pd.DataFrame, target_col="pct_chg"):
    train_returns = y_df[target_col].dropna()
    q33 = np.percentile(train_returns, 33.33)
    q67 = np.percentile(train_returns, 66.67)
    return q33, q67

def build_sequence_data_with_meta(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
    """Build sequence data and also return metadata"""
    # 1. Label processing
    y_work = y_df.copy()
    y_work['trade_date'] = pd.to_datetime(y_work['trade_date'])
    y_work['future_change'] = y_work.groupby('stock_code')['pct_chg'].shift(-1)

    def get_label(x):
        if pd.isna(x): return -999
        if x < q33: return 0
        elif x > q67: return 2
        else: return 1
    y_work['label'] = y_work['future_change'].apply(get_label)

    # 2. Feature processing
    X_work = X_df.copy()
    X_work['trade_date'] = pd.to_datetime(X_work['trade_date'])

    # 3. Merge and align
    merged = pd.merge(X_work, y_work[['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']],
                      on=['stock_code', 'trade_date'], how='inner')

    # Determine the feature columns
    if feature_cols == 'ALL':
        exclude = ['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']
        valid_feats = [c for c in X_work.columns if c not in exclude]
    else:
        valid_feats = [f for f in feature_cols if f in X_work.columns]

    print(f"  Feature Count: {len(valid_feats)}")

    all_seqs = []
    all_labels = []
    all_meta = []

    print(f"  → Scan {merged['stock_code'].nunique()} stock sequences...")

    for stock, g in merged.groupby('stock_code'):
        if len(g) < max_seq_len: continue

        g = g.sort_values('trade_date')
        feats = g[valid_feats].values
        dates = g['trade_date'].values
        labels = g['label'].values

        # Extract metadata
        meta_data = g[['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']].values

        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len

            # [Corrected suspension filter logic]
            # Suspension data filter (checked by calendar-day span)
            days_diff = (dates[end-1] - dates[start_idx]) / np.timedelta64(1, 'D')

            # max_seq_len=60 is the number of trading days
            # about 84 days normally (60 ÷ 5 × 7, including weekends)
            # Allow a 16-day buffer (long holidays + a few suspensions)
            # Total threshold: 100 days
            # The original threshold of 110 days was too loose; changed to 100 days to filter more suspension data
            if days_diff > 100:
                continue

            target = labels[end-1]
            if target == -999: continue

            # Feature sequence
            seq_x = feats[start_idx : end]

            all_seqs.append(seq_x)
            all_labels.append(target)
            all_meta.append(meta_data[end-1])

    if not all_seqs:
        return np.empty((0, max_seq_len, len(valid_feats))), np.array([]), pd.DataFrame()

    # Refactoring Meta DataFrame
    meta_df = pd.DataFrame(all_meta, columns=['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change'])

    return np.stack(all_seqs), np.array(all_labels, dtype=np.int64), meta_df

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==================== 2. Model definition (GRU & BiLSTM) ====================

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
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        # Bidirectional, so the input dimension is hidden_dim * 2
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

# ==================== 3. Training and evaluation ====================

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

    # History containers
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

        # Validation
        val_acc, val_f1 = evaluate_model(model, val_loader, device)
        train_loss = np.mean(losses)
        curr_lr = optimizer.param_groups[0]['lr']

        # Records
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
#Suitable for dataset1
# def train_model_with_weight(model, train_loader, val_loader, device, class_weights, epochs=400, lr=5e-5):
# """Training with class weights"""
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
#Suitable for dataset2
def train_model_with_weight(model, train_loader, val_loader, device, class_weights, epochs=400, lr=5e-5):
    """Training with class weights"""
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


# ==================== 4. Bootstrap Validation ====================

def bootstrap_validation(y_test, y_pred, dataset_name, model_type, feat_desc, seed, save_dir, n_iterations=100):
    """Bootstrap Validation"""
    print("\n" + "=" * 80)
    print(f"[Bootstrap check - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}]")
    print("=" * 80)
    print(f"Run {n_iterations} bootstrap resamples...")

    bootstrap_results = {'accuracy': [], 'f1': []}

    n_samples = len(y_test)

    for i in range(n_iterations):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_iterations}")

        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_test_boot = y_test[indices]
        y_pred_boot = y_pred[indices]

        acc = accuracy_score(y_test_boot, y_pred_boot)
        f1 = f1_score(y_test_boot, y_pred_boot, average='macro')

        bootstrap_results['accuracy'].append(acc)
        bootstrap_results['f1'].append(f1)

    print(f"\nBootstrap results (95% confidence interval):")
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
    print(f"\n✓ Bootstrap Results saved to {filename}")

    return bootstrap_df

# ==================== 6. XAI analysis (DL only) ====================

def attention_based_importance(model, test_loader, device, feature_names):
    """
    Gradient-based feature importance analysis (cuDNN error and reference error fixed)
    """
    import torch.nn as nn # Make sure the nn keyword is recognized inside the function

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

                # [cite_start]--- Core change: remove the [cite: 39] tag ---
                logits[i, pred_class[i]].backward(retain_graph=True)

                # Take this sample's mean gradient over all time steps
                grad = xb.grad[i].abs().mean(dim=0).cpu().detach().numpy()
                all_grads.append(grad)

    mean_grads = np.mean(all_grads, axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Gradient_Importance': mean_grads
    }).sort_values('Gradient_Importance', ascending=False)

    # Add
    print("\n=== Gradient Full results ===")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']:25s} {row['Gradient_Importance']:.6f}")
    # End of addition

    model.eval()
    return importance_df



def integrated_gradients_analysis(model, train_loader, test_loader, device, feature_names,
                                  dataset_name, model_type, feat_desc, seed, save_dir, n_samples=500):
    """
    Integrated Gradients analysis (fully fixes the cuDNN RNN state error)
    """
    import torch.nn as nn

    # 1. Force train mode to allow backward, but lock Dropout manually to remove randomness
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    # 2. Disable cuDNN locally; this is the most reliable safeguard against this error
    # It forces PyTorch to compute gradients with ordinary operators that do not check the training state
    orig_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        # Prepare data (unchanged)
        X_samples, y_samples = [], []
        for xb, yb in test_loader:
            X_samples.append(xb)
            y_samples.append(yb)
            if len(X_samples) * xb.size(0) >= n_samples:
                break
        X_test = torch.cat(X_samples, dim=0)[:n_samples].to(device)

        # 3. Baseline: training-set mean (consistent with the no-leakage principle)
        ig = IntegratedGradients(model)

        # Prepare the training set baseline
        X_train_samples = []
        for xb, _ in train_loader:
            X_train_samples.append(xb)
            if len(X_train_samples) * xb.size(0) >= n_samples:
                break
        X_train_subset = torch.cat(X_train_samples, dim=0)[:n_samples].to(device)
        baseline = X_train_subset.mean(dim=0, keepdim=True).expand_as(X_test)

        attributions_list = []
        batch_size = 50
        print(f"  Compute Integrated Gradients (cuDNN temporarily disabled)...")

        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            X_batch = X_test[i:end_idx]
            baseline_batch = baseline[i:end_idx]

            preds = model(X_batch).argmax(dim=1)
            attr_batch = ig.attribute(X_batch, baseline_batch, target=preds, n_steps=50)
            attributions_list.append(attr_batch.cpu().detach())

        attributions = torch.cat(attributions_list, dim=0)

        # 4. Changed attribution logic: mean absolute value over all time steps
        mean_attr = attributions.abs().mean(dim=[0, 1]).numpy()

        ig_importance = pd.DataFrame({
            'Feature': feature_names,
            'IG_Importance': mean_attr
        }).sort_values('IG_Importance', ascending=False)

        # Add
        print("\n=== IG Full results ===")
        for i, row in ig_importance.iterrows():
            print(f"{row['Feature']:25s} {row['IG_Importance']:.6f}")
        # End of addition

        return ig_importance

    finally:
        # 5. Whether successful or not, restore the model state and the global cuDNN setting
        torch.backends.cudnn.enabled = orig_cudnn_state
        model.eval()



def shap_deepexplainer_analysis(model, train_loader, test_loader, device, feature_names,
                                 dataset_name, model_type, feat_desc, seed, save_dir,
                                 n_background=100, n_test=500):
    """
    Improved SHAP analysis: identify the feature dimension automatically and fix the length mismatch error
    """
    import torch.nn as nn
    import shap
    import numpy as np
    import pandas as pd

    print("\n" + "=" * 80)
    print(f"[SHAP analysis - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}]")
    print("=" * 80)

    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    orig_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        # 1. Prepare data
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

        # 2. Create the Explainer and compute
        print("  Create SHAP GradientExplainer (cuDNN temporarily disabled)...")
        explainer = shap.GradientExplainer(model, background)

        print("  Compute SHAP values...")
        shap_values = explainer.shap_values(test_data)

        # 3. Core fix: align the feature dimension automatically
        # Convert lists or arrays to absolute-value tensors
        shap_array = np.abs(np.array(shap_values))
        print(f"  SHAP Original shape: {shap_array.shape}")

        # print(f" SHAP Original shape: {shap_array.shape}")
        print(f"  SHAP Maximum: {shap_array.max():.6f}")
        print(f"  SHAP Minimum: {shap_array.min():.6f}")
        print(f"  SHAP Mean: {shap_array.mean():.6f}")

        target_len = len(feature_names)

        # Find which dimension has a size equal to the number of features (13)
        # According to the log (500, 60, 13, 3), the index of the feature dimension should be 2
        feat_dim_idx = -1
        for i, dim in enumerate(shap_array.shape):
            if dim == target_len:
                feat_dim_idx = i
                break

        if feat_dim_idx != -1:
            # Compress all dimensions except the feature dimension
            axes_to_reduce = tuple(i for i in range(shap_array.ndim) if i != feat_dim_idx)
            final_importance = shap_array.mean(axis=axes_to_reduce)
            print(f"  Feature dimension identified automatically: Index {feat_dim_idx}, Result length: {len(final_importance)}")
        else:
            # Fallback: if 13 is not found, try compressing along the last dimension
            print("  ⚠️ No matching feature dimension found; forcing a reshape")
            final_importance = np.mean(shap_array.reshape(-1, target_len), axis=0)

        # 4. Build DataFrame
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': final_importance.flatten()
        }).sort_values('SHAP_Importance', ascending=False)

        print(f"\nSHAP GradientExplainer - Top 10 Features:")
        for i, row in shap_importance.head(10).iterrows():
            print(f"  {i+1:2d}. {row['Feature']:25s} {row['SHAP_Importance']:.6f}")

        # Add
        print("\n=== SHAP Full results ===")
        for i, row in shap_importance.iterrows():
            print(f"{row['Feature']:25s} {row['SHAP_Importance']:.6f}")
        # End of addition

        filename = f"shap_importance_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
        shap_importance.to_csv(save_dir / filename, index=False)
        return shap_importance

    except Exception as e:
        print(f"\n❌ SHAP Analysis failed: {e}")
        return None
    finally:
        torch.backends.cudnn.enabled = orig_cudnn_state
        model.eval()




def temporal_attention_visualization(model, test_loader, device, feature_names,
                                     dataset_name, model_type, feat_desc, seed, save_dir, n_samples=100):
    """
    Timestep importance visualization (fixes the cuDNN RNN error)
    """
    import torch.nn as nn
    print("\n" + "=" * 80)
    print(f"[Timestep importance analysis - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}]")
    print("=" * 80)

    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    orig_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        # Sample data
        X_samples = []
        for xb, _ in test_loader:
            X_samples.append(xb)
            if len(X_samples) * xb.size(0) >= n_samples: break
        X_test = torch.cat(X_samples, dim=0)[:n_samples].to(device)

        timestep_importance = []
        print(f"  Analysis {X_test.size(1)} time steps (cuDNN temporarily disabled)...")

        for t in range(X_test.size(1)):
            X_test.requires_grad = True
            model.zero_grad()

            logits = model(X_test)
            pred_class = logits.argmax(dim=1)

            grads_at_t = []
            for i in range(len(X_test)):
                if X_test.grad is not None: X_test.grad.zero_()

                # Core fix: backward runs in forced train mode
                logits[i, pred_class[i]].backward(retain_graph=True)
                grad_t = X_test.grad[i, t, :].abs().mean().item()
                grads_at_t.append(grad_t)

            timestep_importance.append(np.mean(grads_at_t))
            if (t + 1) % 20 == 0: print(f"    Progress: timestep {t+1}/60")

        timestep_df = pd.DataFrame({
            'Timestep': range(1, len(timestep_importance) + 1),
            'Importance': timestep_importance
        })
        # Add
        print("\n=== Full timestep results ===")
        for i, row in timestep_df.iterrows():
            print(f"Step {row['Timestep']:0f}: {row['Importance']:.6f}")
        # End of addition


        filename = f"timestep_importance_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
        timestep_df.to_csv(save_dir / filename, index=False)
        return timestep_df

    finally:
        torch.backends.cudnn.enabled = orig_cudnn_state
        model.eval()


def compare_xai_methods_dl(gradient_imp, ig_imp, shap_imp,
                           dataset_name, model_type, feat_desc, seed, save_dir):
    """Compare different XAI methods of the DL model"""
    print("\n" + "=" * 80)
    print(f"[XAI method consistency comparison - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}]")
    print("=" * 80)

    print("\n=== Debug information ===")
    print("IG Top 5 feature values:", ig_imp['IG_Importance'].head().values)
    print("SHAP Top 5 feature values:", shap_imp['SHAP_Importance'].head().values)
    print("whether the values are exactly identical:", np.allclose(
        ig_imp.sort_values('Feature')['IG_Importance'].values,
        shap_imp.sort_values('Feature')['SHAP_Importance'].values
    ))

    if shap_imp is None:
        print("⚠️ SHAP results unavailable, comparing only the Gradient and IG methods")

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

        print(f"\n Consistency of feature importance methods (Spearman correlation + p-value):")
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

        print(f"\n Consistency of feature importance methods (Spearman correlation + p-value):")
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
    print(f"\n✓ Comparison results saved to {filename}")

    return comparison_df, correlations




# def compare_xai_methods_dl(gradient_imp, ig_imp, shap_imp,
#                            dataset_name, model_type, feat_desc, seed, save_dir):
#     """
# Compare different XAI methods of the DL model
#     """
#     print("\n" + "=" * 80)
# print(f"[XAI method consistency comparison - {dataset_name} - {model_type} - {feat_desc} - Seed {seed}]")
#     print("=" * 80)

# print("\n=== Debug information ===")
# print("IG Top 5 feature values:", ig_imp['IG_Importance'].head().values)
# print("SHAP Top 5 feature values:", shap_imp['SHAP_Importance'].head().values)
# print("whether the values are exactly identical:", np.allclose(
#         ig_imp.sort_values('Feature')['IG_Importance'].values,
#         shap_imp.sort_values('Feature')['SHAP_Importance'].values
#     ))

# # ===== Handle the case where SHAP may be None =====
#     if shap_imp is None:
# print("⚠️ SHAP results unavailable, comparing only the Gradient and IG methods")

# # Correct way to change it: use merge to keep features aligned
#         comparison_df = gradient_imp[['Feature', 'Gradient_Importance']].merge(
#             ig_imp[['Feature', 'IG_Importance']],
#             on='Feature',
# how='inner' # Make sure only features present in both models are compared
#         )

    # # before normalization and correlation calculation

    # # Normalization
    #     for col in ['Gradient_Importance', 'IG_Importance']:
    #         comparison_df[col] = (comparison_df[col] - comparison_df[col].min()) / \
    #                             (comparison_df[col].max() - comparison_df[col].min() + 1e-8)

    # # Compute ranks
    #     comparison_df['Gradient_Rank'] = comparison_df['Gradient_Importance'].rank(ascending=False)
    #     comparison_df['IG_Rank'] = comparison_df['IG_Importance'].rank(ascending=False)

    # # Spearman Correlation coefficient
    #     from scipy.stats import spearmanr
    #     corr_grad_ig = spearmanr(comparison_df['Gradient_Rank'], comparison_df['IG_Rank'])[0]

    # print(f"\n Consistency of feature importance methods (Spearman correlation):")
    #     print(f"  Gradient vs IG: {corr_grad_ig:.4f}")

    #     correlations = {
    #         'gradient_ig': corr_grad_ig,
    #         'gradient_shap': None,
    #         'ig_shap': None
    #     }

    # else:
    # # All three methods available

    #     comparison_df = gradient_imp[['Feature', 'Gradient_Importance']].copy()
    #     comparison_df = comparison_df.merge(
    #         ig_imp[['Feature', 'IG_Importance']],
    #         on='Feature',
    #         how='outer'
    #     )


        # # Handle SHAP (may be None)
        # if shap_imp is not None and len(shap_imp) > 0:
        #     comparison_df = comparison_df.merge(
        #         shap_imp[['Feature', 'SHAP_Importance']],
        #         on='Feature',
        #         how='outer'
        #     )
        # else:
        #     comparison_df['SHAP_Importance'] = 0.0

        # # Fill NaN
        # comparison_df = comparison_df.fillna(0.0)

        # # Normalize to [0,1] for comparison
        # for col in ['Gradient_Importance', 'IG_Importance', 'SHAP_Importance']:
        #     comparison_df[col] = (comparison_df[col] - comparison_df[col].min()) / \
        #                         (comparison_df[col].max() - comparison_df[col].min() + 1e-8)

        # # Compute ranks
        # comparison_df['Gradient_Rank'] = comparison_df['Gradient_Importance'].rank(ascending=False)
        # comparison_df['IG_Rank'] = comparison_df['IG_Importance'].rank(ascending=False)
        # comparison_df['SHAP_Rank'] = comparison_df['SHAP_Importance'].rank(ascending=False)

        # # Spearman Correlation coefficient
        # from scipy.stats import spearmanr

        # # Compute correlation for non-zero values only
        # valid_mask = (comparison_df['Gradient_Importance'] > 0) & (comparison_df['IG_Importance'] > 0)
        # if valid_mask.sum() >= 3:
        #     corr_grad_ig = spearmanr(
        #         comparison_df.loc[valid_mask, 'Gradient_Importance'],
        #         comparison_df.loc[valid_mask, 'IG_Importance']
        #     )[0]
        # else:
        #     corr_grad_ig = float('nan')

    # # SHAP Correlation
    #     if (comparison_df['SHAP_Importance'] > 0).sum() >= 3:
    #         valid_mask_gs = (comparison_df['Gradient_Importance'] > 0) & (comparison_df['SHAP_Importance'] > 0)
    #         valid_mask_is = (comparison_df['IG_Importance'] > 0) & (comparison_df['SHAP_Importance'] > 0)

    #         corr_grad_shap = spearmanr(
    #             comparison_df.loc[valid_mask_gs, 'Gradient_Importance'],
    #             comparison_df.loc[valid_mask_gs, 'SHAP_Importance']
    #         )[0] if valid_mask_gs.sum() >= 3 else float('nan')

    #         corr_ig_shap = spearmanr(
    #             comparison_df.loc[valid_mask_is, 'IG_Importance'],
    #             comparison_df.loc[valid_mask_is, 'SHAP_Importance']
    #         )[0] if valid_mask_is.sum() >= 3 else float('nan')
    #     else:
    #         corr_grad_shap = float('nan')
    #         corr_ig_shap = float('nan')

    # print(f"\n Consistency of feature importance methods (Spearman correlation):")
    #     print(f"  Gradient vs IG:   {corr_grad_ig:.4f}")
    #     print(f"  Gradient vs SHAP: {corr_grad_shap:.4f}")
    #     print(f"  IG vs SHAP:       {corr_ig_shap:.4f}")


    #     correlations = {
    #         'gradient_ig': corr_grad_ig,
    #         'gradient_shap': corr_grad_shap,
    #         'ig_shap': corr_ig_shap
    #     }

    # # Save
    # filename = f"xai_comparison_{model_type.lower()}_{dataset_name}_{feat_desc}_seed{seed}.csv"
    # comparison_df.to_csv(save_dir / filename, index=False)
    # print(f"\n✓ Comparison results saved to {filename}")

    # return comparison_df, correlations




# ==================== 5. Main function ====================

def run_single_experiment(ds_key, ds_dir_name, feature_cols, feat_desc, model_type, seed,
                         script_dir, results_dir, models_dir, device):
    """Run a single experiment (one model + one seed)"""
    print(f"\n{'='*80}")
    print(f"🔥 {model_type} | {ds_key} | {feat_desc} | Seed = {seed}")
    print(f"{'='*80}")

    # Set random seeds
    set_seed(seed)

    # 1. Load data
    data_dir = script_dir / 'datasets/processed' / ds_dir_name
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data_raw(data_dir)
    # ============ New: time filtering of the dataset2 training data ============
    if ds_key == 'dataset2':
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr, start_date='2010-01-01')
    # ============================================================

    q33, q67 = calculate_thresholds(y_tr)

    print(f"📊 Dynamic threshold: 33%={q33:.4f}%, 67%={q67:.4f}%")

    # 2. Build sequences
    print("Build sequences...")
    X_train, y_train, _ = build_sequence_data_with_meta(X_tr, y_tr, feature_cols, q33, q67)
    X_val, y_val, _ = build_sequence_data_with_meta(X_val, y_val, feature_cols, q33, q67)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, feature_cols, q33, q67)

    print(f"📊 Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Class distribution
    print(f"\n📊 Test-set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        names = {0: 'Down', 1: 'Flat', 2: 'Up'}
        pct = cnt/len(y_test)*100
        print(f"  {names[cls]}: {cnt:,} ({pct:.1f}%)")

    # 3. DataLoader
    batch_size = 256 if ds_key == 'dataset1' else 512

    tr_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=512)
    te_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=512)

        # 4. Training
    input_dim = X_train.shape[2]

    if ds_key == 'dataset1':
        # dataset1 only: add class weights
        class_weights = torch.FloatTensor([1.0, 1.0, 1.2]).to(device)
        print(f"   📊 Class weights (set manually): {class_weights.cpu().numpy()}")

        if model_type == 'GRU':
            model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
        else:  # bilstm
            model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.1).to(device)

        model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, epochs=400, lr=5e-5)
    # else:
    # # dataset2 only: add class weights
    #     # class_weights = torch.FloatTensor([1.1, 1.0, 1.2]).to(device)
    # # print(f" 📊 Class weights (set manually): {class_weights.cpu().numpy()}")
    #     if model_type == 'GRU':
    #         class_weights = torch.FloatTensor([1.1, 0.9, 1.2]).to(device)
    # print(f" 📊 Class weights (set manually): {class_weights.cpu().numpy()}")
    #         model = GRUClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    #         # lr=5e-5
    #         lr=5e-5
    #     else:  # bilstm
    #         class_weights = torch.FloatTensor([1.1, 1.0, 1.25]).to(device)
    # print(f" 📊 Class weights (set manually): {class_weights.cpu().numpy()}")
    #         model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.35).to(device)
    #         lr=5e-5

    #     model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, epochs=400, lr=lr)
    else:
        # dataset2 only: add class weights
        # class_weights = torch.FloatTensor([1.1, 1.0, 1.2]).to(device)
        # print(f" 📊 Class weights (set manually): {class_weights.cpu().numpy()}")
        if model_type == 'GRU':
            class_weights = torch.FloatTensor([1.1, 0.9, 1.2]).to(device)
            print(f"   📊 Class weights (set manually): {class_weights.cpu().numpy()}")
            model = GRUClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
            # lr=5e-5
            lr=5e-5
        else:  # bilstm
            class_weights = torch.FloatTensor([1.0, 1.0, 1.0]).to(device)
            print(f"   📊 Class weights (set manually): {class_weights.cpu().numpy()}")
            model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=3, dropout=0.1).to(device)
            lr=5e-5

        model, history = train_model_with_weight(model, tr_loader, val_loader, device, class_weights, epochs=400, lr=lr)




    # # 5. Training
    # model, history = train_model(model, tr_loader, val_loader, device, epochs=400, lr=1e-4)

    # 6. Final evaluation
    acc, f1, y_pred, y_prob = evaluate_model(model, te_loader, device, return_probs=True)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n🏆 Final results: F1={f1:.4f} | Acc={acc:.4f}")
    print(f"   Per-class F1: down={f1_per_class[0]:.4f}, Flat={f1_per_class[1]:.4f}, Up={f1_per_class[2]:.4f}")

    # 7. Save all outputs

    # A. Save model
    model_path = models_dir / f"{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"   💾 Model saved: {model_path.name}")

    # B. Save training history
    hist_path = results_dir / f"history_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   💾 Training history saved: {hist_path.name}")

    # C. Save detailed predictions
    pred_df = meta_test.copy()
    pred_df['pred_label'] = y_pred
    pred_df['prob_0'] = y_prob[:, 0]
    pred_df['prob_1'] = y_prob[:, 1]
    pred_df['prob_2'] = y_prob[:, 2]
    pred_df['model'] = model_type
    pred_df['seed'] = seed

    pred_path = results_dir / f"preds_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"   💾 Predictions saved: {pred_path.name}")

    # D. Save the confusion matrix
    cm_df = pd.DataFrame(cm,
                         index=['True_Down', 'True_Hold', 'True_Up'],
                         columns=['Pred_Down', 'Pred_Hold', 'Pred_Up'])
    cm_path = results_dir / f"confusion_matrix_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.csv"
    cm_df.to_csv(cm_path)
    print(f"   💾 Confusion matrix saved: {cm_path.name}")

    # E. Bootstrap check (first seed only)
    if seed == 42:
        bootstrap_df = bootstrap_validation(
            y_test, y_pred, ds_key, model_type, feat_desc, seed, results_dir, n_iterations=100
        )

    # F. XAI analysis (first seed only)
    if seed == 42:
        print("\n" + "="*80)
        print("🔍 Start in-depth XAI analysis...")
        print("="*80)

        # Get feature names
        if feature_cols == 'ALL':
            exclude = ['stock_code', 'trade_date', 'label', 'pct_chg', 'future_change']
            feat_names = [c for c in X_tr.columns if c not in exclude]
        else:
            feat_names = feature_cols

        # 1. Gradient importance
        print("\n📊 Method 1: gradient-based feature importance")
        gradient_importance = attention_based_importance(model, te_loader, device, feat_names)
        grad_filename = f"gradient_importance_{model_type.lower()}_{ds_key}_{feat_desc}_seed{seed}.csv"
        gradient_importance.to_csv(results_dir / grad_filename, index=False)
        print(f"\nGradient - Top 10 Features:")
        for i, row in gradient_importance.head(10).iterrows():
            print(f"  {i+1:2d}. {row['Feature']:25s} {row['Gradient_Importance']:.6f}")
        print(f"✓ Saved to {grad_filename}")

        # 2. Integrated Gradients
        print("\n📊 Method 2: Integrated Gradients")
        ig_importance = integrated_gradients_analysis(
        # changed to
            model, tr_loader, te_loader, device, feat_names,  # New tr_loader
            ds_key, model_type, feat_desc, seed, results_dir, n_samples=500
        )
          # End of change
        #     model, te_loader, device, feat_names,
        #     ds_key, model_type, feat_desc, seed, results_dir, n_samples=500
        # )

        # 3. SHAP DeepExplainer
        print("\n📊 Method 3: SHAP DeepExplainer")
        shap_importance = shap_deepexplainer_analysis(
            model, tr_loader, te_loader, device, feat_names,
            ds_key, model_type, feat_desc, seed, results_dir,
            n_background=100, n_test=500
        )

        # 4. Timestep importance
        print("\n📊 Method 4: timestep importance analysis")
        timestep_importance = temporal_attention_visualization(
            model, te_loader, device, feat_names,
            ds_key, model_type, feat_desc, seed, results_dir, n_samples=100
        )

        # 5. XAI Method comparison
        print("\n📊 Method 5: comparison of XAI method consistency")
        comparison_df, correlations = compare_xai_methods_dl(
            gradient_importance, ig_importance, shap_importance,
            ds_key, model_type, feat_desc, seed, results_dir
        )

        print("\n✅ XAI Analysis complete!")

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

    # Folder settings
    results_dir = script_dir / 'results_XAI' / 'rnn_bilstm_us_60'
    models_dir = script_dir / 'models_XAI' / 'rnn_bilstm_us_60'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 GRU & BiLSTM Enhanced Benchmark (corrected version)")
    print("   - Dataset1: 5 seeds")
    print("   - Dataset2: 5 seeds")
    print("   - Suspension threshold: 100 days (tightened from 110 days)")
    print("=" * 80)


    SELECTED_FEATURES_13 = [
        'roc_5', 'low', 'close_to_ma20', 'roc_20', 'high_low_range',
        'rsi', 'dayofweek', 'macd', 'dayofyear_sin', 'bb_width',
        'bb_position', 'macd_signal', 'dayofyear'
    ]
    SELECTED_FEATURES_12 = [
        'roc_5', 'dayofyear_sin', 'macd_signal', 'ma10', 'macd',
        'dayofyear', 'close', 'high_low_range', 'low', 'close_to_ma20',
        'roc_20', 'pre_close'
    ]

    # SELECTED_FEATURES_20_M = [
    #     'roc_5', 'low', 'close_to_ma20', 'roc_20', 'high_low_range',
    #     'rsi', 'dayofweek', 'macd', 'dayofyear_sin', 'bb_width',
    #     'bb_position', 'ma60', 'close', 'macd_signal', 'dayofyear',
    #     'open', 'roc_10', 'high', 'pre_close', 'momentum_5'
    # ]

    # SELECTED_FEATURES_20_P = [
    #     'roc_5', 'dayofyear_sin', 'macd_signal', 'ma10', 'macd',
    #     'dayofyear', 'close', 'high_low_range', 'low', 'close_to_ma20',
    #     'high', 'roc_20', 'pre_close', 'open', 'momentum_5',
    #     'ma5', 'dayofweek', 'day', 'rsi', 'bb_upper'
    # ]

    tasks = [
        ('dataset4_m', 'dataset4_m', SELECTED_FEATURES_13, 'SELECTED_13', 1),
        ('dataset5_p', 'dataset5_p', SELECTED_FEATURES_12, 'SELECTED_12', 1),
    ]





    # # Dataset 2 selected features

    # SELECTED_FEATURES_12 = [
    #     'dayofyear', 'atr', 'roc_5', 'high_low_range', 'turnover_rate',
    #     'macd_signal', 'close', 'macd', 'month', 'close_to_ma5', 'roc_10', 'amount'
    # ]

    # # Configure tasks
    # tasks = [
    # # ('dataset1', 'dataset1_A_grade', 'ALL', 'ALL_18', 10), # 10 seeds
    # # ('dataset1', 'dataset1_A_grade', SELECTED_FEATURES_13, 'SELECTED_13', 5), # 5 seeds
    # # ('dataset2', 'dataset2_AB_grade', SELECTED_FEATURES_14, 'SELECTED_14', 5), # 5 seeds
    # ('dataset2', 'dataset2_AB_grade', SELECTED_FEATURES_12, 'SELECTED_12', 5), # 5 seeds
    # # ('dataset2', 'dataset2_AB_grade', 'ALL', 'FULL_22', 5) # 5 seeds
    # ]

    # Model list
    # models_to_run = ['GRU', 'BiLSTM']
    models_to_run = ['BiLSTM']

    # Define the seed pool
    # seeds_10 = [10, 20, 30, 42, 100, 123, 456, 777, 888, 2024]
    # seeds_5 = [42, 123, 777, 0, 2024]
    seeds_5 = [42]

    all_results = []

    for ds_key, ds_dir_name, feature_cols, feat_desc, num_seeds in tasks:
        for model_type in models_to_run:
            print(f"\n\n{'='*80}")
            print(f"📊 Start experiment: {model_type} - {ds_key} - {feat_desc} ({num_seeds} seeds)")
            print(f"{'='*80}")

            # Choose the seed list according to num_seeds
            seeds = seeds_10 if num_seeds == 10 else seeds_5

            for seed in seeds:
                result = run_single_experiment(
                    ds_key, ds_dir_name, feature_cols, feat_desc, model_type, seed,
                    script_dir, results_dir, models_dir, device
                )
                all_results.append(result)

            # Summarize the results of this model + configuration
            df = pd.DataFrame(all_results)
            df_task = df[(df['dataset'] == ds_key) & (df['model'] == model_type) & (df['feature_set'] == feat_desc)]

            print(f"\n{'='*80}")
            print(f"📊 {model_type} - {ds_key} - {feat_desc} Summary ({len(seeds)} seeds)")
            print(f"{'='*80}")
            print(f"F1:  {df_task['test_f1'].mean():.4f} ± {df_task['test_f1'].std():.4f}")
            print(f"Acc: {df_task['test_acc'].mean():.4f} ± {df_task['test_acc'].std():.4f}")

    # Save the final summary
    final_df = pd.DataFrame(all_results)
    summary_path = results_dir / 'rnn_us_bilstm_.csv'
    final_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("✅ All done!")
    print("=" * 80)
    print(f"\n Created files:")
    print(f"  Model: {models_dir}")
    print(f"  Results: {results_dir}")
    print(f"  Overall summary: {summary_path}")
    print(f"\n Main changes:")
    print(f"  ✓ Dataset1: 5 seeds (stability)")
    print(f"  ✓ Dataset2: 5 seeds (already very stable)")
    print(f"  ✓ Suspension threshold: 100 days (filters more suspension data)")
    print("=" * 80)

if __name__ == "__main__":
    main()
