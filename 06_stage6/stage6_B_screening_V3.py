# ==============================================================================
# stage6_cell_B_data_screening_V3
# Diagnostic version: in V1 the us_med probe model learned nothing (loss stuck at 0.488, i.e., guessing 1/3 for each class),
# and all 22 feature importances were 0. This version uses settings identical to V1 and to the Chinese script, for diagnosis only:
#   1. Run the same code on the Chinese Power data (12 features finally kept originally) to check that the code and pipeline work
#   2. Run Chinese Power, US-Med and US-Pow once each with seeds 42, 0 and 123 to check whether the random seed is the cause
# Each run prints: per-epoch loss and validation F1, the predicted-class distribution on the validation set (to detect single-class prediction), and per-feature permutation importance
# Input: ./stage6/stage6_cell_A_data_preprocessing_V1/ (us_med/, us_pow/, feature_config.json)
#       Original Chinese Power directory XAI_cn/e3/datasets/processed/dataset2_AB_grade/ (read directly, not copied; first-stage features = all columns of train_X except IDs)
# V3 changes: Chinese Power is read from its original directory; all input files are checked before running, and all missing files are listed at once
# Output: ./stage6/stage6_cell_B_data_screening_V3/
# ==============================================================================

# ------------------------------ Settings (editable) ------------------------------
US_INPUT_DIR = "stage6/stage6_cell_A_data_preprocessing_V1"
CN_POWER_DIR = "/home/i/2026_0712/stock_i/XAI_cn/e3/datasets/processed/dataset2_AB_grade"
OUTPUT_DIR = "stage6/stage6_cell_B_data_screening_V3"
RUNS_DATASETS = ["cn_power", "us_med", "us_pow"]
SEEDS = [42, 0, 123]
# The 12 features finally used for Chinese Power (SELECTED_FEATURES_12 in p_*_dataset2.py), for comparison
CN_POWER_FINAL_12 = ["dayofyear", "atr", "roc_5", "high_low_range", "turnover_rate",
                     "macd_signal", "close", "macd", "month", "close_to_ma5", "roc_10", "amount"]
TEMPORAL_NUMERIC = ["year", "month", "day", "dayofweek", "quarter", "dayofyear", "weekofyear"]

MAX_SEQ_LEN = 60
BATCH_SIZE = 256          # same as China
NUM_EPOCHS = 30           # same as China
LR = 2e-4                 # same as China
WEIGHT_DECAY = 1e-4       # same as China
D_MODEL = 80              # same as China
NUM_LAYERS = 2            # same as China
NHEAD = 8                 # same as China
DROPOUT = 0.2             # same as China

# ------------------------------ Imports ------------------------------
import os
import math
import json
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

T0 = time.time()
# Rule: the first two lines print the input and output locations (relative paths only)


def short_path(p):
    # Show only the part starting from the project folder, not the full absolute path
    p = p.replace("\\", "/")
    if "XAI_cn/" in p:
        return p[p.index("XAI_cn/"):]
    return "./" + p.lstrip("./")


print(f"Input directories: {short_path(US_INPUT_DIR)}/ (us_med/, us_pow/, feature_config.json), {short_path(CN_POWER_DIR)}/")
print(f"Output directory: ./{OUTPUT_DIR}/")

# Check all input files before running
need = [os.path.join(US_INPUT_DIR, "feature_config.json")]
for d in ["us_med", "us_pow"]:
    need += [os.path.join(US_INPUT_DIR, d, f) for f in ["train_X.csv", "val_X.csv", "train_y.csv", "val_y.csv"]]
need += [os.path.join(CN_POWER_DIR, f) for f in ["train_X.csv", "val_X.csv", "train_y.csv", "val_y.csv"]]
missing = [p for p in need if not os.path.isfile(p)]
if missing:
    print("\nThe following input files do not exist; check US_INPUT_DIR and CN_POWER_DIR at the top of the code:")
    for p in missing:
        print("  " + short_path(p))
    raise FileNotFoundError(f"{len(missing)} input files are missing")
print(f"Input file check passed ({len(need)} files)")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def elapsed():
    # Return the elapsed time as a string
    s = int(time.time() - T0)
    return f"{s // 60}m{s % 60:02d}s"


def set_seed(seed):
    # Fix all random sources
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== 1. Data loading and sequence construction (same as the Chinese script) ====================

def load_split(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "train_X.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "val_X.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_y.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "val_y.csv"))
    return (X_train, y_train), (X_val, y_val)


def calculate_thresholds(y_df, target_col="pct_chg"):
    returns = y_df[target_col].dropna()
    q33 = np.percentile(returns, 33.33)
    q67 = np.percentile(returns, 66.67)
    return q33, q67


def build_sequence_aligned(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
    # Labels: pct_chg shifted forward by 1 day within each stock (same as China)
    y_work = y_df.copy()
    y_work["trade_date"] = pd.to_datetime(y_work["trade_date"])
    y_work["future_change"] = y_work.groupby("stock_code")["pct_chg"].shift(-1)

    def get_label(x):
        if pd.isna(x):
            return -999
        if x < q33:
            return 0
        elif x > q67:
            return 2
        else:
            return 1

    y_work["label"] = y_work["future_change"].apply(get_label)

    X_work = X_df.copy()
    X_work["trade_date"] = pd.to_datetime(X_work["trade_date"])
    available_cols = [c for c in feature_cols if c in X_work.columns]
    if len(available_cols) < len(feature_cols):
        print(f"  ⚠️ Missing features: {len(feature_cols)} required, {len(available_cols)} found")

    merged = pd.merge(X_work, y_work[["stock_code", "trade_date", "label"]],
                      on=["stock_code", "trade_date"], how="inner")

    all_seqs, all_labels = [], []
    for stock, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue
        g = g.sort_values("trade_date")
        g_feats = g[available_cols].values
        g_dates = g["trade_date"].values
        g_labels = g["label"].values
        # Sliding windows (same as China, including the time-continuity check and the range start and end)
        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len
            dt = (g_dates[end - 1] - g_dates[start_idx]) / np.timedelta64(1, "D")
            if dt > (max_seq_len * 1.5 + 20):
                continue
            target = g_labels[end - 1]
            if target == -999:
                continue
            all_seqs.append(g_feats[start_idx:end])
            all_labels.append(target)

    return np.stack(all_seqs), np.array(all_labels, dtype=np.int64), available_cols


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 2. Probe model (same as China) ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ClassicTransformer(nn.Module):
    def __init__(self, input_dim, d_model=80, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.2, max_seq_len=60, num_classes=3):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc_out(x)


def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


# ==================== 3. Training and permutation importance ====================

def eval_f1(model, loader, device):
    # Compute validation Macro-F1 (evaluation only; does not affect training)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(-1).cpu().numpy())
            ys.append(yb.numpy())
    return f1_score(np.concatenate(ys), np.concatenate(preds), average="macro")


def run_probe_training(model, train_loader, val_loader, num_epochs, lr, device, tag):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    print(f"  Training probe model ({num_epochs} epochs, same as China)")
    t_train = time.time()
    best_f1, best_ep = -1.0, 0
    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum, n_batch = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = focal_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            loss_sum += loss.item()
            n_batch += 1
        val_f1 = eval_f1(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1, best_ep = val_f1, epoch
        per_ep = (time.time() - t_train) / epoch
        remain = per_ep * (num_epochs - epoch)
        print(f"    [{tag}] Epoch {epoch:2d}/{num_epochs} | train loss {loss_sum / n_batch:.4f} | "
              f"val Macro-F1 {val_f1:.4f} | best so far {best_f1:.4f} (epoch {best_ep}) | "
              f"this epoch {time.time() - t_ep:.1f}s | est. remaining {remain / 60:.1f}m | total {elapsed()}")
    print("  Note: as in the Chinese script, permutation importance uses the model at the end of epoch 30 (no rollback to the best epoch)")
    return model


def calculate_permutation_importance(model, val_loader, feature_names, device, tag):
    model.eval()
    all_X, all_y = [], []
    for x, y in val_loader:
        all_X.append(x)
        all_y.append(y)
    X_full = torch.cat(all_X).to(device)
    y_full = torch.cat(all_y).to(device)

    def get_score(X_in, y_in):
        with torch.no_grad():
            preds = model(X_in).argmax(dim=-1).cpu().numpy()
        return f1_score(y_in.cpu().numpy(), preds, average="macro")

    baseline_f1 = get_score(X_full, y_full)
    print(f"\n  [{tag}] Permutation importance: baseline validation Macro-F1 = {baseline_f1:.4f} ({X_full.size(0):,} validation sequences)")
    importances = []
    for i, feat_name in enumerate(feature_names):
        X_permuted = X_full.clone()
        perm_idx = torch.randperm(X_full.size(0))
        X_permuted[:, :, i] = X_full[perm_idx, :, i]
        new_f1 = get_score(X_permuted, y_full)
        importance = baseline_f1 - new_f1
        importances.append({"Feature": feat_name, "Importance": importance, "New_F1": new_f1})
        print(f"    Feature {i + 1:2d}/{len(feature_names)} {feat_name:<18} F1 after shuffling {new_f1:.4f} | "
              f"importance {importance:+.4f} | {'kept' if importance > 0 else 'removed'}")
    return pd.DataFrame(importances).sort_values("Importance", ascending=False), baseline_f1


# ==================== 4. Main ====================

def pred_distribution(model, loader, device):
    # Predicted-class distribution on the validation set, used to detect single-class prediction
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(-1).cpu().numpy())
    cnt = np.bincount(np.concatenate(preds), minlength=3)
    return cnt / cnt.sum()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
with open(os.path.join(US_INPUT_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
    config_a = json.load(f)

# Load data and build sequences (once per dataset; sequence construction involves no randomness)
prepared = {}
for ds in RUNS_DATASETS:
    if ds == "cn_power":
        data_dir = CN_POWER_DIR
        X_tr0 = pd.read_csv(os.path.join(data_dir, "train_X.csv"), nrows=1)
        feature_cols = [c for c in X_tr0.columns if c not in ["stock_code", "trade_date"]]
    else:
        data_dir = os.path.join(US_INPUT_DIR, ds)
        feature_cols = config_a[ds]["selected_features"] + config_a[ds]["temporal_features"]
    (X_tr, y_tr), (X_val, y_val) = load_split(data_dir)
    q33, q67 = calculate_thresholds(y_tr)
    t = time.time()
    X_tr_seq, y_tr_seq, actual_cols = build_sequence_aligned(X_tr, y_tr, feature_cols, q33, q67, MAX_SEQ_LEN)
    X_val_seq, y_val_seq, _ = build_sequence_aligned(X_val, y_val, feature_cols, q33, q67, MAX_SEQ_LEN)
    print(f"\n[{ds}] {len(actual_cols)} first-stage features: {actual_cols}")
    print(f"  Thresholds q33 = {q33:.4f}, q67 = {q67:.4f}; sequence construction {time.time() - t:.1f}s: train {X_tr_seq.shape}, validation {X_val_seq.shape}")
    cnt = np.bincount(y_val_seq, minlength=3)
    print(f"  Validation label distribution: down {cnt[0] / cnt.sum():.1%} | hold {cnt[1] / cnt.sum():.1%} | up {cnt[2] / cnt.sum():.1%}")
    prepared[ds] = (X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, actual_cols)

summary_rows = []
n_runs = len(RUNS_DATASETS) * len(SEEDS)
run_i = 0
for ds in RUNS_DATASETS:
    X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, actual_cols = prepared[ds]
    for seed in SEEDS:
        run_i += 1
        t_run = time.time()
        print(f"\n{'#' * 90}")
        print(f"Run {run_i}/{n_runs}: dataset {ds}, seed {seed}, total {elapsed()}")
        print(f"{'#' * 90}")
        set_seed(seed)
        tr_loader = DataLoader(SequenceDataset(X_tr_seq, y_tr_seq), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_val_seq, y_val_seq), batch_size=BATCH_SIZE)
        model = ClassicTransformer(input_dim=len(actual_cols), d_model=D_MODEL, nhead=NHEAD,
                                   num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        model = run_probe_training(model, tr_loader, val_loader, NUM_EPOCHS, LR, device, f"{ds}-seed{seed}")
        dist = pred_distribution(model, val_loader, device)
        collapsed = dist.max() > 0.95
        print(f"  Validation predicted-class distribution of the epoch-30 model: down {dist[0]:.1%} | hold {dist[1]:.1%} | up {dist[2]:.1%}"
              f" -> {'single class only (nothing learned)' if collapsed else 'multiple classes'}")

        importance_df, base_f1 = calculate_permutation_importance(model, val_loader, actual_cols, device,
                                                                  f"{ds}-seed{seed}")
        importance_df.to_csv(os.path.join(OUTPUT_DIR, f"permutation_importance_{ds}_seed{seed}.csv"), index=False)
        kept = [c for c in actual_cols if c in importance_df[importance_df["Importance"] > 0]["Feature"].tolist()]
        print(f"\n  [{ds}-seed{seed}] {len(kept)} kept: {kept}")
        overlap = ""
        if ds == "cn_power":
            common = [c for c in kept if c in CN_POWER_FINAL_12]
            overlap = f"{len(common)}/12"
            print(f"  Overlap with the 12 features finally used for Chinese Power {overlap}: {common}")
        print(f"  Time for this run {(time.time() - t_run) / 60:.1f}m, total {elapsed()}")
        summary_rows.append({"dataset": ds, "seed": seed, "baseline_val_macro_f1": round(base_f1, 4),
                             "pred_down": round(dist[0], 3), "pred_hold": round(dist[1], 3),
                             "pred_up": round(dist[2], 3), "collapsed": collapsed,
                             "n_stage1": len(actual_cols), "n_kept": len(kept),
                             "overlap_with_cn_final12": overlap, "kept_features": ", ".join(kept)})

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUTPUT_DIR, "diagnosis_summary.csv"), index=False)
print("\n" + "=" * 90)
print("stage6_cell_B_V2 diagnostic summary, total time " + elapsed())
print("=" * 90)
print(summary.drop(columns=["kept_features"]).to_string(index=False))
for _, r in summary.iterrows():
    print(f"  {r['dataset']}-seed{r['seed']} kept: [{r['kept_features']}]")
print(f"\nOutput files: ./{OUTPUT_DIR}/diagnosis_summary.csv, permutation_importance_<dataset>_seed<seed>.csv")
