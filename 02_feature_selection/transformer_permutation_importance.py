import math
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== 1. Data loading ====================

def load_data_dataset2_original(data_dir: Path):
    print(f"  Reading data from: {data_dir.name}")
    X_train = pd.read_csv(data_dir / "train_X.csv")
    X_val = pd.read_csv(data_dir / "val_X.csv")
    X_test = pd.read_csv(data_dir / "test_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv")
    y_val = pd.read_csv(data_dir / "val_y.csv")
    y_test = pd.read_csv(data_dir / "test_y.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def calculate_thresholds(y_df: pd.DataFrame, target_col: str = "pct_chg"):
    returns = y_df[target_col].dropna()
    q33 = np.percentile(returns, 33.33)
    q67 = np.percentile(returns, 66.67)
    return q33, q67

def build_sequence_aligned(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
    # Label processing
    y_work = y_df.copy()
    y_work['trade_date'] = pd.to_datetime(y_work['trade_date'])
    y_work['future_change'] = y_work.groupby('stock_code')['pct_chg'].shift(-1)

    def get_label(x):
        if pd.isna(x): return -999
        if x < q33: return 0
        elif x > q67: return 2
        else: return 1
    y_work['label'] = y_work['future_change'].apply(get_label)

    # Feature processing
    X_work = X_df.copy()
    X_work['trade_date'] = pd.to_datetime(X_work['trade_date'])

    # Make sure only existing features are used
    available_cols = [c for c in feature_cols if c in X_work.columns]
    if len(available_cols) < len(feature_cols):
        print(f"⚠️ Warning: Missing features. Requested {len(feature_cols)}, Found {len(available_cols)}")

    # Align
    merged = pd.merge(X_work, y_work[['stock_code', 'trade_date', 'label']],
                      on=['stock_code', 'trade_date'], how='inner')

    all_seqs = []
    all_labels = []

    # Convert to NumPy arrays for faster processing
    # The column order must be fixed
    feats_matrix = merged[available_cols].values
    dates = merged['trade_date'].values
    labels = merged['label'].values
    codes = merged['stock_code'].values

    # Build sequences quickly (Vectorized approach is hard due to groups, sticking to loop but optimized)
    # For code simplicity and stability, still use GroupBy

    for stock, g in merged.groupby('stock_code'):
        if len(g) < max_seq_len: continue

        g = g.sort_values('trade_date')
        g_feats = g[available_cols].values
        g_dates = g['trade_date'].values
        g_labels = g['label'].values

        # Sliding window
        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len

            # Temporal continuity check (Numpy timedelta check)
            dt = (g_dates[end-1] - g_dates[start_idx]) / np.timedelta64(1, 'D')
            if dt > (max_seq_len * 1.5 + 20): continue

            target = g_labels[end-1]
            if target == -999: continue

            all_seqs.append(g_feats[start_idx:end])
            all_labels.append(target)

    if not all_seqs: return np.empty((0, max_seq_len, len(available_cols))), np.array([])
    return np.stack(all_seqs), np.array(all_labels, dtype=np.int64), available_cols

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==================== 2. Model definition (Classic) ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])

class ClassicTransformer(nn.Module):
    def __init__(self, input_dim, d_model=80, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.2, max_seq_len=60, num_classes=3):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc_out(x)

# ==================== 3. Feature importance analysis tools ====================

def calculate_permutation_importance(model, val_loader, feature_names, device):
    """
    Compute the permutation importance of features (Permutation Importance)
    Principle: permute one feature column and measure how much F1 drops. The larger the drop, the more important the feature.
    """
    model.eval()

    # 1. Collect all validation data in memory (to make it easy to Shuffle)
    all_X = []
    all_y = []
    with torch.no_grad():
        for x, y in val_loader:
            all_X.append(x)
            all_y.append(y)

    X_full = torch.cat(all_X).to(device)
    y_full = torch.cat(all_y).to(device)

    # 2. Compute the baseline score (Baseline Score)
    def get_score(X_in, y_in):
        with torch.no_grad():
            logits = model(X_in)
            preds = logits.argmax(dim=-1).cpu().numpy()
        return f1_score(y_in.cpu().numpy(), preds, average="macro")

    baseline_f1 = get_score(X_full, y_full)
    print(f"\n📊 Baseline Validation F1: {baseline_f1:.4f}")

    importances = []

    # 3. Permute and test the features one at a time
    print("🔍 Analyzing feature importance...")
    for i, feat_name in enumerate(feature_names):
        # Create a copy
        X_permuted = X_full.clone()

        # Permute feature i along the batch dimension
        # X shape: [Batch, Seq_Len, Features] -> Shuffle dim 0
        perm_idx = torch.randperm(X_full.size(0))
        X_permuted[:, :, i] = X_full[perm_idx, :, i]

        # Compute the new score
        new_f1 = get_score(X_permuted, y_full)

        # importance = baseline score - score after permutation
        # A positive value means the feature is important; a negative value means the feature may be noise
        importance = baseline_f1 - new_f1
        importances.append({
            "Feature": feat_name,
            "Importance": importance,
            "New_F1": new_f1
        })
        # print(f"   Feature: {feat_name:<20} | Drop: {importance:+.4f}")

    return pd.DataFrame(importances).sort_values("Importance", ascending=False)

# ==================== 4. Training pipeline ====================

def run_probe_training(model, train_loader, val_loader, num_epochs, lr, device):
    """Train a'Probe'model; it does not need to be perfect, only to converge"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    # Use Focal Loss so the model learns quickly
    criterion = lambda p, t: focal_loss(p, t)

    print(f"  🚀 Training Probe Model ({num_epochs} epochs)...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if epoch % 10 == 0:
            # Quick check
            model.eval()
            all_preds, all_y = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    all_preds.append(model(xb).argmax(-1).cpu().numpy())
                    all_y.append(yb.cpu().numpy())
            f1 = f1_score(np.concatenate(all_y), np.concatenate(all_preds), average='macro')
            print(f"    Epoch {epoch}: Val F1 = {f1:.4f}")

    return model

def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()

# ==================== 5. Main entry ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    try: script_dir = Path(__file__).resolve().parent
    except NameError: script_dir = Path.cwd()

    config_path = script_dir / 'datasets/feature_selection/feature_config.json'
    with open(config_path, 'r') as f: config = json.load(f)

    ds_key = 'dataset2'
    ds_dir_name = 'dataset2_AB_grade'

    print(f"\n{'='*60}\n🚀 Running FEATURE DIAGNOSIS on {ds_key}\n{'='*60}")

    data_dir = script_dir / 'datasets/processed' / ds_dir_name
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data_dataset2_original(data_dir)

    # ✅ 1. Use all original features of Dataset 2 (21)
    feature_cols = config[ds_key]['selected_features'] + config[ds_key]['temporal_features']
    print(f"Original Feature Count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    q33, q67 = calculate_thresholds(y_tr)

    print("\nBuilding sequences (using ALL features)...")
    X_tr_seq, y_tr_seq, actual_cols = build_sequence_aligned(X_tr, y_tr, feature_cols, q33, q67)
    X_val_seq, y_val_seq, _ = build_sequence_aligned(X_val, y_val, feature_cols, q33, q67)

    # Build Loader
    BATCH_SIZE = 256
    tr_loader = DataLoader(SequenceDataset(X_tr_seq, y_tr_seq), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val_seq, y_val_seq), batch_size=BATCH_SIZE)

    # ✅ 2. Train the probe model
    # Use the classic architecture and train quickly 30 epoch
    model = ClassicTransformer(input_dim=len(actual_cols), d_model=80, num_layers=2, dropout=0.2).to(device)
    model = run_probe_training(model, tr_loader, val_loader, num_epochs=30, lr=2e-4, device=device)

    # ✅ 3. Compute feature importance
    print("\n🧐 Diagnosing Features...")
    importance_df = calculate_permutation_importance(model, val_loader, actual_cols, device)

    print("\n" + "="*40)
    print("FEATURE IMPORTANCE REPORT (Deep Learning)")
    print("="*40)
    print(importance_df)

    # Save report
    out_path = script_dir / 'results' / 'dl_feature_importance.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_path, index=False)

    # ✅ 4. Give recommendations
    print("\n💡 Optimization Suggestion:")
    bad_features = importance_df[importance_df['Importance'] <= 0]['Feature'].tolist()
    if len(bad_features) > 0:
        print(f"❌ Detected {len(bad_features)} harmful/useless features (Importance <= 0):")
        print(f"   {bad_features}")
        print("👉 Please remove these features and retrain the model.")
    else:
        print("✅ All features seem to contribute positively.")

    print(f"\nReport saved to: {out_path}")

if __name__ == "__main__":
    main()
