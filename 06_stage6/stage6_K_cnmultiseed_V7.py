"""
stage6_cell_K_data_cnmultiseed_V7

Aim: correct the IG computation in the five-seed analysis of the Chinese datasets, reading existing models only, without retraining.
1. The original five-seed code built models with the training-configuration dropout and switched to training mode for IG,
   so between-layer dropout of two-layer RNNs with dropout > 0 was active and IG contained randomness;
2. This cell reads the existing seed 42, 123, 777, 0 and 2024 models (weights unchanged, no reselection),
   and recomputes IG with an inference copy with dropout=0 and the same architecture (exactly as in U.S. stage6_cell_J);
3. Original settings are kept: 500 fixed test samples, 64 IG steps, training-set mean trajectory baseline, target fixed to the seed-42 predicted class;
4. Feature-ranking correlation, temporal-ranking correlation, top-5 feature overlap and recency are recomputed;
5. Test Macro-F1, accuracy and prediction agreement are recomputed and checked item by item against the original results;
6. Scope: all 8 Chinese combinations in the original five-seed analysis are recomputed (including those with dropout=0, for uniformity).
   Medical-2 GRU was trained by a separate script and its model dropout is already 0, so it is unaffected and not recomputed in this cell.
7. Sources of the new-seed models are the same as in the paper:
   - Medical-2 BiLSTM and LSTM: models retrained by stage5_cell_B_retrain_med2_V2 (the version used in the paper);
   - the other 6 combinations: models from xai_cell_J_V2.
   The original results are checked against the same sources.
8. Prediction checks have three parts, none with relaxed tolerance:
   A. Reading check of original results: from the stage5 results only Medical-2 BiLSTM and LSTM are taken;
      after merging, each of the 40 targets (dataset + model + seed) must appear exactly once with all required fields; otherwise stop before computation.
   B. "Near-boundary sample count check" against the original results (a plausibility check, not proof of cause):
      - the original results saved only summary metrics, not per-sample predictions or logits, and cuDNN cannot be enabled on the server,
        so per-sample comparison with the original results and recomputation with cuDNN are impossible;
      - logit margin = largest logit minus second-largest logit; samples with margin below MARGIN_EPS are called "near-boundary samples";
      - rules: Macro-F1 difference <= 0.0005; change in the number of correct samples <= the model's near-boundary sample count;
        change in the number of samples agreeing with seed 42 <= the sum of the near-boundary counts of the two models;
      - passing only shows that the size of the differences is consistent with the range floating-point error could affect; it cannot identify which samples changed.
   C. Per-sample cross-backend check of seed 42 (supporting evidence):
      - the same seed-42 model is computed with GPU native kernels and on CPU, and logits and predicted classes are compared per sample;
      - a class flip depends on the difference between two class logits; if a single logit deviates by ε, the between-class difference changes by at most 2ε,
        so the maximum change in all pairwise class differences of each sample between the two backends is measured directly and must be below MARGIN_EPS;
      - this only shows the numerical sensitivity between non-cuDNN backends and does not directly represent the difference between the original cuDNN and native kernels;
      - the predictions are also compared per sample with the seed-42 per-sample predictions (CPU) saved by stage5_cell_A_bootstrap_V4:
        first the sample order is verified to be identical per sample using stock code + trade date, then the true labels are compared;
        then separately: this cell's CPU predictions vs stage5 (whether CPU results can be reproduced),
        this cell's GPU predictions vs stage5 (cross-environment difference), and this cell's GPU vs CPU;
      - per-sample rule: in each comparison, every sample with a different predicted class must have a logit margin below MARGIN_EPS
        (CPU margin for CPU comparisons, GPU margin for GPU comparisons, the smaller of the two for GPU vs CPU),
        i.e., all disagreeing samples must lie in the near-boundary region, not merely the total counts.
   If any check fails, results are still saved, but the cell stops with an error at the end and does not print "finished".
   If all pass, this is described only as "count check and seed-42 cross-backend check passed", not as "strict reproduction of the original results".
"""

# ------------------------------ Settings (editable) ------------------------------
PROJECT_ROOT = "/home/i/2026_0712/stock_i"          # project root containing the Chinese data and the original five-seed results
J_V2_DIR_CANDIDATES = ["xai_cell_J_V2", "0924/xai_cell_J_V2"]   # original five-seed output directory (relative to the project root, checked in order)
# Output directory of stage5_cell_B_retrain_med2_V2 (checked first in the current Notebook directory, then in the project root)
STAGE5_DIR_CANDIDATE = "stage5/cell_B_retrain_med2_V2"
STAGE5_COMBOS = [("Medical-2", "BiLSTM"), ("Medical-2", "LSTM")]  # combinations whose new-seed models come from stage5
# Output directory of stage5_cell_A_bootstrap_V4 (seed-42 per-sample predictions; checked first in the current directory, then in the project root)
STAGE5A_DIR_CANDIDATE = "stage5/cell_A_bootstrap_V4"
OUTPUT_ROOT = "stage6/stage6_cell_K_data_cnmultiseed_V7"         # relative to the current Notebook directory
COMBOS = [("Medical-1", "GRU"), ("Medical-1", "BiLSTM"), ("Medical-1", "LSTM"),
          ("Medical-2", "BiLSTM"), ("Medical-2", "LSTM"),
          ("Power", "GRU"), ("Power", "BiLSTM"), ("Power", "LSTM")]
SEEDS = [42, 123, 777, 0, 2024]
PRIMARY_SEED = 42
SEQ_LEN = 60
IG_SAMPLES = 500
IG_STEPS = 64
IG_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
CHECK_TOL = 5e-4      # tolerance for checking Macro-F1 / accuracy against the original results
MARGIN_EPS = 1e-3     # near-boundary threshold; the measured maximum change in between-class logit differences between GPU native and CPU in this cell must be below this value

# Data settings identical to the original five-seed code
DATASET_CFG = {
    "Medical-1": {"data_dir": "XAI_cn/datasets/processed/dataset1_A_grade", "filter_train_2010": False,
                  "features": ["dayofyear", "day", "volume_ratio", "close_to_ma5", "vol", "low",
                               "close_open_range", "macd_signal", "close_to_ma20", "roc_20",
                               "high_low_range", "turnover_rate", "amount"]},
    "Medical-2": {"data_dir": "XAI_cn/datasets/processed/dataset2_AB_grade", "filter_train_2010": True,
                  "features": ["high_low_range", "turnover_rate", "amount", "bb_width", "roc_10",
                               "roc_5", "close_to_ma20", "close_to_ma5", "close_open_range", "atr",
                               "volume_ratio", "roc_20", "dayofyear_cos", "year"]},
    "Power": {"data_dir": "XAI_cn/e3/datasets/processed/dataset2_AB_grade", "filter_train_2010": True,
              "features": ["dayofyear", "atr", "roc_5", "high_low_range", "turnover_rate",
                           "macd_signal", "close", "macd", "month", "close_to_ma5",
                           "roc_10", "amount"]},
}
# Seed-42 model paths identical to the original five-seed code (relative to the project root)
PRIMARY_MODEL_PATH = {
    ("Medical-1", "GRU"): "XAI_cn/models_XAI/rnn_enhanced_dataset2/gru_dataset1_SELECTED_13_seed42.pt",
    ("Medical-1", "BiLSTM"): "XAI_cn/models_XAI/rnn_enhanced_dataset2/bilstm_dataset1_SELECTED_13_seed42.pt",
    ("Medical-1", "LSTM"): "XAI_cn/models_XAI/lstm_enhanced/lstm_dataset1_SELECTED_13_seed42.pt",
    ("Medical-2", "BiLSTM"): "XAI_cn/models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_14_seed42.pt",
    ("Medical-2", "LSTM"): "XAI_cn/models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_14_seed42.pt",
    ("Power", "GRU"): "XAI_cn/e3/models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_12_seed42.pt",
    ("Power", "BiLSTM"): "XAI_cn/e3/models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_12_seed42.pt",
    ("Power", "LSTM"): "XAI_cn/e3/models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_12_seed42.pt",
}
# Architectures in the original five-seed code (used to check the model files)
EXPECTED_ARCH = {
    ("Medical-1", "GRU"): (64, 2), ("Medical-1", "BiLSTM"): (64, 2), ("Medical-1", "LSTM"): (64, 2),
    ("Medical-2", "BiLSTM"): (128, 2), ("Medical-2", "LSTM"): (128, 2),
    ("Power", "GRU"): (128, 2), ("Power", "BiLSTM"): (128, 2), ("Power", "LSTM"): (96, 2),
}

# ------------------------------ Imports ------------------------------
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(PROJECT_ROOT)
OUTPUT_DIR = Path(OUTPUT_ROOT)


def rel(p):
    # Show only the path starting from the project root, not the full absolute path
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return "./" + p.as_posix()


def elapsed():
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


# Find the original five-seed output directory (candidates checked one by one in order, no scanning)
J_V2_DIR = None
for _c in J_V2_DIR_CANDIDATES:
    if (ROOT / _c / "multiseed_detail.csv").exists():
        J_V2_DIR = ROOT / _c
        break

# Find the stage5 output directories (only two explicit locations are checked)
STAGE5_DIR = None
for _base in [Path("."), ROOT]:
    if (_base / STAGE5_DIR_CANDIDATE / "multiseed_detail.csv").exists():
        STAGE5_DIR = _base / STAGE5_DIR_CANDIDATE
        break
STAGE5A_DIR = None
for _base in [Path("."), ROOT]:
    if (_base / STAGE5A_DIR_CANDIDATE / "A1_predictions_Medical-1.csv").exists():
        STAGE5A_DIR = _base / STAGE5A_DIR_CANDIDATE
        break


def seed_model_dir(dataset_name, model_type):
    # Directory of the new-seed models: stage5 for Medical-2 BiLSTM/LSTM, J_V2 for the others
    return (STAGE5_DIR if (dataset_name, model_type) in STAGE5_COMBOS else J_V2_DIR) / "models"

# Rule: the first two lines print the input and output locations
print(f"Input directories: XAI_cn/ (Chinese data and seed-42 models), "
      f"{rel(J_V2_DIR) if J_V2_DIR else 'original five-seed directory (not found)'}/ (new-seed models and original results for 6 combinations), "
      f"{rel(STAGE5_DIR) if STAGE5_DIR else STAGE5_DIR_CANDIDATE + ' (not found)'}/ (new-seed models and original results for Medical-2 BiLSTM and LSTM), "
      f"{rel(STAGE5A_DIR) if STAGE5A_DIR else STAGE5A_DIR_CANDIDATE + ' (not found)'}/ (seed-42 per-sample predictions)")
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")

torch.backends.cudnn.enabled = False
print("cuDNN disabled (cuDNN cannot be initialized on the server); using PyTorch native CUDA RNN kernels")

if J_V2_DIR is None:
    print("The original five-seed results multiseed_detail.csv were not found in any of the following locations; modify J_V2_DIR_CANDIDATES:")
    for _c in J_V2_DIR_CANDIDATES:
        print("  " + _c + "/multiseed_detail.csv")
    raise FileNotFoundError("Original five-seed output directory not found")
if STAGE5_DIR is None:
    print("The stage5 multiseed_detail.csv was not found in either of the following locations; modify STAGE5_DIR_CANDIDATE:")
    print("  ./" + STAGE5_DIR_CANDIDATE + "/multiseed_detail.csv")
    print("  " + STAGE5_DIR_CANDIDATE + "/multiseed_detail.csv (under the project root)")
    raise FileNotFoundError("stage5_cell_B_retrain_med2_V2 output directory not found")
if STAGE5A_DIR is None:
    print("The stage5 A1_predictions_Medical-1.csv was not found in either of the following locations; modify STAGE5A_DIR_CANDIDATE:")
    print("  ./" + STAGE5A_DIR_CANDIDATE + "/A1_predictions_Medical-1.csv")
    print("  " + STAGE5A_DIR_CANDIDATE + "/A1_predictions_Medical-1.csv (under the project root)")
    raise FileNotFoundError("stage5_cell_A_bootstrap_V4 output directory not found")

# Check all input files before running
_need = []
for _ds in sorted({d for d, _ in COMBOS}):
    _need += [ROOT / DATASET_CFG[_ds]["data_dir"] / f"{_s}_{_xy}.csv" for _s in ["train", "val", "test"] for _xy in ["X", "y"]]
    _need.append(STAGE5A_DIR / f"A1_predictions_{_ds}.csv")
for _ds, _m in COMBOS:
    _need.append(ROOT / PRIMARY_MODEL_PATH[(_ds, _m)])
    _need += [seed_model_dir(_ds, _m) / f"{_ds}_{_m}_seed{_s}.pt" for _s in SEEDS if _s != PRIMARY_SEED]
_missing = [p for p in _need if not p.exists()]
if _missing:
    print("The following input files do not exist:")
    for _p in _missing:
        print("  " + rel(_p))
    raise FileNotFoundError(f"{len(_missing)} input files are missing")
print(f"Input file check passed ({len(_need)} files)")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS}; IG samples {IG_SAMPLES}, IG integration steps {IG_STEPS}")
# Original results: stage5 for Medical-2 BiLSTM/LSTM, J_V2 for the others (consistent with the paper)
REQUIRED_COLS = ["Dataset", "Model", "Seed", "MacroF1", "Accuracy", "Prediction_Agreement_vs_42",
                 "Feature_Spearman_vs_42", "Temporal_Spearman_vs_42", "IG_Last10_to_First10"]
_old_j = pd.read_csv(J_V2_DIR / "multiseed_detail.csv")
_old_s5 = pd.read_csv(STAGE5_DIR / "multiseed_detail.csv")
for _name, _df in [("J_V2", _old_j), ("stage5", _old_s5)]:
    _lack = [c for c in REQUIRED_COLS if c not in _df.columns]
    if _lack:
        raise ValueError(f"{_name} original results are missing fields: {_lack}")
_key_j = pd.Series(list(zip(_old_j["Dataset"], _old_j["Model"])))
_key_s5 = pd.Series(list(zip(_old_s5["Dataset"], _old_s5["Model"])))
_j_keep = _old_j[(~_key_j.isin(STAGE5_COMBOS) & _key_j.isin(COMBOS)).values]
_s5_keep = _old_s5[_key_s5.isin(STAGE5_COMBOS).values]
OLD = pd.concat([_j_keep[REQUIRED_COLS], _s5_keep[REQUIRED_COLS]], ignore_index=True)
_target = pd.DataFrame([(d, m, sd) for d, m in COMBOS for sd in SEEDS], columns=["Dataset", "Model", "Seed"])
_cnt = _target.merge(OLD.groupby(["Dataset", "Model", "Seed"]).size().rename("N").reset_index(),
                     on=["Dataset", "Model", "Seed"], how="left").fillna({"N": 0})
_bad = _cnt[_cnt["N"] != 1]
print(f"Original results read: J_V2 {len(_old_j)} rows, of which {len(_j_keep)} kept (6 combinations); stage5 {len(_old_s5)} rows, of which {len(_s5_keep)} kept"
      f" (Medical-2 BiLSTM, LSTM); {len(OLD)} rows after merging")
if len(_bad) > 0 or len(OLD) != len(_target):
    print("Error: the following targets are missing or duplicated in the original results (N = number of occurrences):")
    print(_bad.to_string(index=False))
    raise ValueError("Merge check of original results failed")
print(f"Merge check of original results passed: {len(_target)} targets (dataset + model + seed) each appear once, all required fields present")
print(f"Check rules: Macro-F1 difference <= {CHECK_TOL}; changes in correct and agreeing sample counts <= near-boundary sample count (logit margin < {MARGIN_EPS})")


# ============================== Computation code below ==============================

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def filter_data_by_date(X_df, y_df, start_date="2010-01-01"):
    X_df = X_df.copy()
    y_df = y_df.copy()
    X_df["trade_date"] = pd.to_datetime(X_df["trade_date"])
    y_df["trade_date"] = pd.to_datetime(y_df["trade_date"])
    start = pd.to_datetime(start_date)
    return X_df[X_df["trade_date"] >= start].copy(), y_df[y_df["trade_date"] >= start].copy()


def load_data_raw(data_dir):
    X_train = pd.read_csv(data_dir / "train_X.csv")
    X_val = pd.read_csv(data_dir / "val_X.csv")
    X_test = pd.read_csv(data_dir / "test_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv")
    y_val = pd.read_csv(data_dir / "val_y.csv")
    y_test = pd.read_csv(data_dir / "test_y.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def calculate_thresholds(y_df):
    arr = y_df["pct_chg"].dropna()
    return np.percentile(arr, 33.33), np.percentile(arr, 66.67)


def build_sequence_data(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
    # Labels: pct_chg shifted forward by 1 day within each stock, split into down/hold/up at the training-set tertiles
    y_work = y_df.copy()
    y_work["trade_date"] = pd.to_datetime(y_work["trade_date"])
    y_work["future_change"] = y_work.groupby("stock_code")["pct_chg"].shift(-1)

    def get_label(x):
        if pd.isna(x):
            return -999
        if x < q33:
            return 0
        if x > q67:
            return 2
        return 1

    y_work["label"] = y_work["future_change"].apply(get_label)
    X_work = X_df.copy()
    X_work["trade_date"] = pd.to_datetime(X_work["trade_date"])
    missing = [f for f in feature_cols if f not in X_work.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    merged = pd.merge(
        X_work,
        y_work[["stock_code", "trade_date", "label", "pct_chg", "future_change"]],
        on=["stock_code", "trade_date"],
        how="inner",
    )
    all_seqs = []
    all_labels = []
    all_meta = []
    for code, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue
        g = g.sort_values("trade_date")
        feats = g[feature_cols].values
        dates = g["trade_date"].values
        labels = g["label"].values
        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len
            # Suspension filter: skip if 60 trading days span more than 100 calendar days
            days_diff = (dates[end - 1] - dates[start_idx]) / np.timedelta64(1, "D")
            if days_diff > 100:
                continue
            target = labels[end - 1]
            if target == -999:
                continue
            all_seqs.append(feats[start_idx:end])
            all_labels.append(target)
            all_meta.append((code, pd.Timestamp(dates[end - 1]).strftime("%Y-%m-%d")))
    if not all_seqs:
        raise RuntimeError("No samples after sequence construction")
    meta = pd.DataFrame(all_meta, columns=["stock_code", "trade_date"])
    return np.stack(all_seqs).astype(np.float32), np.asarray(all_labels, dtype=np.int64), meta


def prepare_dataset(dataset_name):
    cfg = DATASET_CFG[dataset_name]
    data_dir = ROOT / cfg["data_dir"]
    features = cfg["features"]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)
    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)
    q33, q67 = calculate_thresholds(y_tr)
    X_train, y_train, _ = build_sequence_data(X_tr, y_tr, features, q33, q67, SEQ_LEN)
    X_test, y_test, meta_test = build_sequence_data(X_te, y_te, features, q33, q67, SEQ_LEN)
    print(f"{dataset_name}: data directory {rel(data_dir)}; train {X_train.shape}, test {X_test.shape}")
    for name, arr in [("Train", X_train), ("Test", X_test)]:
        n_bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        if n_bad > 0:
            raise ValueError(f"{dataset_name}-{name} set contains {n_bad} NaN/Inf")
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "features": features,
            "meta_test": meta_test}


# ------------------------------ Models (same architecture as in training) ------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        x = torch.nan_to_num(x)
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = torch.nan_to_num(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_inference_model(model_type, path):
    # Infer the architecture from the model file, build a dropout=0 inference model and load the weights strictly
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    prefix = "gru" if model_type == "GRU" else "lstm"
    input_dim = state[f"{prefix}.weight_ih_l0"].shape[1]
    hidden = state[f"{prefix}.weight_hh_l0"].shape[1]
    layers = max(int(k.split("_l")[1].split("_")[0]) for k in state
                 if k.startswith(f"{prefix}.weight_ih_l") and "_reverse" not in k) + 1
    cls = {"GRU": GRUClassifier, "LSTM": LSTMClassifier, "BiLSTM": BiLSTMClassifier}[model_type]
    model = cls(input_dim, hidden, layers, 0.0)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval(), input_dim, hidden, layers


def predict_arrays(model, X, batch_size=512, return_margin=False, device=None, return_logits=False):
    device = DEVICE if device is None else device
    model.eval()
    preds, probs, margins, all_logits = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            top2 = torch.topk(logits, 2, dim=1).values
            margins.append((top2[:, 0] - top2[:, 1]).cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    if return_logits:
        return np.concatenate(preds), np.concatenate(all_logits), np.concatenate(margins)
    if return_margin:
        return np.concatenate(preds), np.concatenate(probs), np.concatenate(margins)
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_arrays(model, X, y):
    pred, prob, margin = predict_arrays(model, X, EVAL_BATCH_SIZE, return_margin=True)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "pred": pred, "prob": prob, "margin": margin,
            "n_near": int((margin < MARGIN_EPS).sum()), "min_margin": float(margin.min())}


# ------------------------------ IG: fixed samples + fixed seed-42 target ------------------------------
def choose_fixed_samples(X_test, n_samples=500):
    rng = np.random.default_rng(PRIMARY_SEED)
    n = min(n_samples, len(X_test))
    return np.sort(rng.choice(len(X_test), size=n, replace=False))


def ig_attribution_fixed_targets(model, X_test, X_train, sample_idx, fixed_targets, n_steps=64, batch_size=64):
    # The input is the dropout=0 inference copy; with cuDNN disabled, the RNN can backpropagate in training mode
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    try:
        ig = IntegratedGradients(model)
        baseline = torch.from_numpy(X_train.mean(axis=0, keepdims=True).astype(np.float32)).to(DEVICE)
        out = []
        for s in range(0, len(sample_idx), batch_size):
            idx = sample_idx[s:s + batch_size]
            xb = torch.from_numpy(X_test[idx].astype(np.float32)).to(DEVICE)
            targets = torch.from_numpy(np.asarray(fixed_targets[s:s + len(idx)], dtype=np.int64)).to(DEVICE)
            base = baseline.expand(xb.size(0), -1, -1)
            attr = ig.attribute(xb, baselines=base, target=targets, n_steps=n_steps,
                                internal_batch_size=min(256, xb.size(0) * 4))
            out.append(attr.detach().abs().cpu().numpy())
            del xb, targets, base, attr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)
    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()


def summarize_ig(attr, feature_names):
    timestep = attr.mean(axis=(0, 2))
    feature = attr.mean(axis=(0, 1))
    time_df = pd.DataFrame({"Timestep": np.arange(1, SEQ_LEN + 1), "IG_Importance": timestep,
                            "IG_Share": timestep / (timestep.sum() + 1e-12)})
    feat_df = pd.DataFrame({"Feature": feature_names, "IG_Importance": feature,
                            "IG_Share": feature / (feature.sum() + 1e-12)})
    return time_df, feat_df


def safe_spearman(x, y):
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def topk_overlap(a, b, k=5):
    aa = set(a.nlargest(k, "IG_Importance")["Feature"].tolist())
    bb = set(b.nlargest(k, "IG_Importance")["Feature"].tolist())
    return len(aa & bb) / float(k)


# ============================== Main ==============================
all_rows, check_rows, feature_rows, timestep_rows, backend_rows = [], [], [], [], []
data_cache = {}
for c_i, (dataset_name, model_type) in enumerate(COMBOS, 1):
    print("\n" + "=" * 100)
    print(f"Combination {c_i}/{len(COMBOS)}: {dataset_name} / {model_type}, total {elapsed()}")
    print("=" * 100)
    if dataset_name not in data_cache:
        print("Step 1: data preparation and checks")
        data_cache[dataset_name] = prepare_dataset(dataset_name)
    data = data_cache[dataset_name]
    sample_idx = choose_fixed_samples(data["X_test"], IG_SAMPLES)

    print("Step 2: load the seed-42 model (dropout=0 inference copy)")
    p_path = ROOT / PRIMARY_MODEL_PATH[(dataset_name, model_type)]
    primary, in_dim, hid, lay = load_inference_model(model_type, p_path)
    if (hid, lay) != EXPECTED_ARCH[(dataset_name, model_type)] or in_dim != len(data["features"]):
        raise AssertionError(f"Architecture mismatch: input {in_dim}, hidden {hid}, layers {lay}")
    primary_eval = evaluate_arrays(primary, data["X_test"], data["y_test"])
    fixed_targets, _ = predict_arrays(primary, data["X_test"][sample_idx], EVAL_BATCH_SIZE)
    print(f"Seed-42 model: {rel(p_path)} | input {in_dim}, hidden {hid}, layers {lay}")
    print(f"Fixed target-class distribution: {pd.Series(fixed_targets).value_counts().sort_index().to_dict()}")

    # Check C: per-sample cross-backend check of seed 42
    print("Step 2b: per-sample cross-backend check of seed 42 (GPU native kernels vs CPU; then vs the per-sample predictions saved by stage5)")
    t = time.time()
    _, gpu_logits, _ = predict_arrays(primary, data["X_test"], EVAL_BATCH_SIZE, return_logits=True)
    cpu_model = copy.deepcopy(primary).to(torch.device("cpu")).eval()
    cpu_pred, cpu_logits, cpu_margin = predict_arrays(cpu_model, data["X_test"], EVAL_BATCH_SIZE,
                                                      device=torch.device("cpu"), return_logits=True)
    del cpu_model
    max_logit_diff = float(np.abs(gpu_logits - cpu_logits).max())
    _pairs = [(i, j) for i in range(gpu_logits.shape[1]) for j in range(i + 1, gpu_logits.shape[1])]
    _pairdiff = np.stack([np.abs((gpu_logits[:, i] - gpu_logits[:, j]) - (cpu_logits[:, i] - cpu_logits[:, j]))
                          for i, j in _pairs], axis=1)
    max_pair_diff = float(_pairdiff.max())
    q999_pair_diff = float(np.quantile(_pairdiff, 0.999))
    ok_eps = max_pair_diff < MARGIN_EPS
    print(f"  GPU native vs CPU: max single-logit difference {max_logit_diff:.2e}; max change in between-class logit differences {max_pair_diff:.2e}"
          f" (99.9th percentile {q999_pair_diff:.2e}), must be below MARGIN_EPS {MARGIN_EPS} -> {'passed' if ok_eps else 'failed'}"
          f" (time {time.time() - t:.1f}s)")
    print("  Note: this is the difference between two non-cuDNN backends, supporting evidence only; it does not directly represent the difference between the original cuDNN and native kernels")
    a1 = pd.read_csv(STAGE5A_DIR / f"A1_predictions_{dataset_name}.csv")
    pred_col = f"pred_{model_type}"
    if pred_col not in a1.columns or len(a1) != len(data["y_test"]):
        raise ValueError(f"stage5 per-sample predictions do not match: column {pred_col} present {pred_col in a1.columns}, "
                         f"rows {len(a1)}, expected {len(data['y_test'])}")
    _meta = data["meta_test"]
    _a1_code = a1["stock_code"].astype(str).values
    _a1_date = pd.to_datetime(a1["trade_date"]).dt.strftime("%Y-%m-%d").values
    n_id_mismatch = int(((_a1_code != _meta["stock_code"].astype(str).values)
                         | (_a1_date != _meta["trade_date"].values)).sum())
    id_ok = n_id_mismatch == 0
    label_ok = bool(np.array_equal(a1["y_true"].values.astype(np.int64), data["y_test"]))
    order_ok = id_ok and label_ok
    gpu_pred, gpu_margin = primary_eval["pred"], primary_eval["margin"]
    a1_pred = a1[pred_col].values.astype(np.int64)

    def flip_audit(pred_a, pred_b, margin):
        # Returns: number of disagreeing samples, number of those with margin not below MARGIN_EPS, and the maximum margin among disagreeing samples
        diff = pred_a != pred_b
        n = int(diff.sum())
        n_out = int((margin[diff] >= MARGIN_EPS).sum())
        mx = float(margin[diff].max()) if n > 0 else 0.0
        return n, n_out, mx

    n_cpu_a1, out_cpu_a1, mx_cpu_a1 = flip_audit(cpu_pred, a1_pred, cpu_margin)
    n_gpu_a1, out_gpu_a1, mx_gpu_a1 = flip_audit(gpu_pred, a1_pred, gpu_margin)
    n_gpu_cpu, out_gpu_cpu, mx_gpu_cpu = flip_audit(gpu_pred, cpu_pred, np.minimum(gpu_margin, cpu_margin))
    ok_a1 = order_ok and out_cpu_a1 == 0 and out_gpu_a1 == 0 and out_gpu_cpu == 0
    print(f"  Sample IDs vs stage5: {n_id_mismatch} samples differ in stock code + trade date; true labels identical per sample {label_ok}")
    for _name, _n, _out, _mx in [("this cell CPU vs stage5 (CPU reproduction)", n_cpu_a1, out_cpu_a1, mx_cpu_a1),
                                 ("this cell GPU vs stage5 (cross-environment)", n_gpu_a1, out_gpu_a1, mx_gpu_a1),
                                 ("this cell GPU vs this cell CPU", n_gpu_cpu, out_gpu_cpu, mx_gpu_cpu)]:
        print(f"  {_name}: {_n} predictions differ; outside the near-boundary region (margin >= {MARGIN_EPS}): {_out}, "
              f"max margin among differing samples {_mx:.2e} -> {'passed' if _out == 0 else 'failed'}")
    print(f"  Seed-42 per-sample cross-backend check: {'passed' if ok_a1 else 'failed'}")
    backend_rows.append({"Dataset": dataset_name, "Model": model_type, "N_Test": len(data["y_test"]),
                         "Max_Logit_Diff_GPU_vs_CPU": max_logit_diff,
                         "Max_Pairwise_Diff_GPU_vs_CPU": max_pair_diff, "Q999_Pairwise_Diff_GPU_vs_CPU": q999_pair_diff,
                         "N_ID_Mismatch_Stage5A": n_id_mismatch, "Label_Match_Stage5A": label_ok,
                         "N_Diff_CPU_vs_Stage5A": n_cpu_a1, "N_Diff_CPU_vs_Stage5A_Outside": out_cpu_a1,
                         "N_Diff_GPU_vs_Stage5A": n_gpu_a1, "N_Diff_GPU_vs_Stage5A_Outside": out_gpu_a1,
                         "N_Diff_GPU_vs_CPU": n_gpu_cpu, "N_Diff_GPU_vs_CPU_Outside": out_gpu_cpu,
                         "Near_Boundary_N_Seed42": primary_eval["n_near"],
                         "Pass_Eps": ok_eps, "Pass_Stage5A": ok_a1, "Match": ok_eps and ok_a1})
    del gpu_logits, cpu_logits, _pairdiff, cpu_margin

    ref_time = ref_feat = None
    for s_i, seed in enumerate(SEEDS, 1):
        print("-" * 60)
        print(f"{dataset_name} / {model_type} / seed {seed} ({s_i}/{len(SEEDS)}), total {elapsed()}")
        if seed == PRIMARY_SEED:
            model, eval_result, m_path = primary, primary_eval, p_path
        else:
            m_path = seed_model_dir(dataset_name, model_type) / f"{dataset_name}_{model_type}_seed{seed}.pt"
            model, in_dim2, hid2, lay2 = load_inference_model(model_type, m_path)
            if (hid2, lay2, in_dim2) != (hid, lay, in_dim):
                raise AssertionError(f"Seed {seed} architecture differs from seed 42: {in_dim2}/{hid2}/{lay2}")
            eval_result = evaluate_arrays(model, data["X_test"], data["y_test"])
        print(f"Model: {rel(m_path)}")

        # Check prediction metrics against the original results
        old = OLD[(OLD["Dataset"] == dataset_name) & (OLD["Model"] == model_type) & (OLD["Seed"] == seed)]
        if len(old) != 1:
            raise ValueError(f"{dataset_name}-{model_type}-seed{seed} not found in the original results")
        old = old.iloc[0]
        n_test = len(data["y_test"])
        d_f1 = float(eval_result["macro_f1"]) - float(old["MacroF1"])
        d_acc = float(eval_result["accuracy"]) - float(old["Accuracy"])
        agree_new = float(np.mean(primary_eval["pred"] == eval_result["pred"]))
        d_agree = agree_new - float(old["Prediction_Agreement_vs_42"])
        # Convert to changes in sample counts
        d_correct_n = int(round(d_acc * n_test))
        d_agree_n = int(round(d_agree * n_test))
        near_model = eval_result["n_near"]
        near_pair = near_model + (primary_eval["n_near"] if seed != PRIMARY_SEED else 0)
        ok_f1 = abs(d_f1) <= CHECK_TOL
        ok_acc = abs(d_correct_n) <= near_model
        ok_agree = abs(d_agree_n) <= near_pair
        ok = ok_f1 and ok_acc and ok_agree
        print(f"Count check: Macro-F1 {eval_result['macro_f1']:.6f} (original {old['MacroF1']:.6f}, difference {d_f1:+.6f})"
              f" -> {'passed' if ok_f1 else 'failed'}")
        print(f"  Accuracy {eval_result['accuracy']:.6f} (original {old['Accuracy']:.6f}), change in correct samples {d_correct_n:+d}, "
              f"near-boundary samples of this model {near_model} (min logit margin {eval_result['min_margin']:.2e})"
              f" -> {'passed' if ok_acc else 'failed'}")
        print(f"  Prediction agreement with seed 42 {agree_new:.6f} (original {old['Prediction_Agreement_vs_42']:.6f}), "
              f"change in agreeing samples {d_agree_n:+d}, near-boundary samples of the two models combined {near_pair}"
              f" -> {'passed' if ok_agree else 'failed'}; count check for this model {'passed' if ok else 'failed'}")
        if seed == PRIMARY_SEED:
            print(f"  Note: seed-42 weights and data are identical; the difference from the original results (change in correct samples {d_correct_n:+d})"
                  f" may come from the computing environment (cuDNN vs native kernels, software versions); the specific samples cannot be identified")
        check_rows.append({"Dataset": dataset_name, "Model": model_type, "Seed": seed, "N_Test": n_test,
                           "MacroF1_New": eval_result["macro_f1"], "MacroF1_Old": old["MacroF1"],
                           "Accuracy_New": eval_result["accuracy"], "Accuracy_Old": old["Accuracy"],
                           "Agreement_New": agree_new, "Agreement_Old": old["Prediction_Agreement_vs_42"],
                           "Delta_Correct_N": d_correct_n, "Delta_Agree_N": d_agree_n,
                           "Near_Boundary_N_Model": near_model, "Near_Boundary_N_Pair": near_pair,
                           "Min_Logit_Margin": eval_result["min_margin"],
                           "Pass_F1": ok_f1, "Pass_Correct": ok_acc, "Pass_Agree": ok_agree, "Match": ok})

        t = time.time()
        attr = ig_attribution_fixed_targets(model, data["X_test"], data["X_train"], sample_idx,
                                            fixed_targets, IG_STEPS, IG_BATCH_SIZE)
        time_df, feat_df = summarize_ig(attr, data["features"])
        print(f"IG done ({time.time() - t:.1f}s)")
        if seed == PRIMARY_SEED:
            ref_time, ref_feat = time_df.copy(), feat_df.copy()

        feat_rho, feat_p = safe_spearman(ref_feat["IG_Importance"].values, feat_df["IG_Importance"].values)
        time_rho, time_p = safe_spearman(ref_time["IG_Importance"].values, time_df["IG_Importance"].values)
        pred_agreement = float(np.mean(primary_eval["pred"] == eval_result["pred"]))
        overlap5 = topk_overlap(ref_feat, feat_df, k=min(5, len(feat_df)))
        last10 = float(time_df.loc[time_df["Timestep"].between(51, 60), "IG_Share"].sum())
        first10 = float(time_df.loc[time_df["Timestep"].between(1, 10), "IG_Share"].sum())
        ratio = last10 / max(first10, 1e-12)
        all_rows.append({
            "Dataset": dataset_name, "Model": model_type, "Seed": seed,
            "MacroF1": float(eval_result["macro_f1"]), "Accuracy": float(eval_result["accuracy"]),
            "Prediction_Agreement_vs_42": pred_agreement,
            "Feature_Spearman_vs_42": feat_rho, "Feature_p_vs_42": feat_p,
            "Temporal_Spearman_vs_42": time_rho, "Temporal_p_vs_42": time_p,
            "Top5_Overlap_vs_42": overlap5, "IG_Last10_Mass": last10, "IG_First10_Mass": first10,
            "IG_Last10_to_First10": ratio,
            "Old_Feature_Spearman_vs_42": float(old["Feature_Spearman_vs_42"]),
            "Old_Temporal_Spearman_vs_42": float(old["Temporal_Spearman_vs_42"]),
            "Old_IG_Last10_to_First10": float(old["IG_Last10_to_First10"])})
        for rows, df in [(feature_rows, feat_df), (timestep_rows, time_df)]:
            tmp = df.copy()
            tmp.insert(0, "Seed", seed)
            tmp.insert(0, "Model", model_type)
            tmp.insert(0, "Dataset", dataset_name)
            rows.append(tmp)
        print(f"Stability: prediction agreement {pred_agreement:.4f}; feature ρ {feat_rho:.4f} (original {old['Feature_Spearman_vs_42']:.4f}); "
              f"temporal ρ {time_rho:.4f} (original {old['Temporal_Spearman_vs_42']:.4f})")
        print(f"Recency: last 10 days {last10:.4f}, first 10 days {first10:.4f}, ratio {ratio:.4f} (original {old['IG_Last10_to_First10']:.4f}), "
              f"top-5 feature overlap {overlap5:.2f}")
        if seed != PRIMARY_SEED:
            del model
        del attr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del primary

# ------------------------------ Summary ------------------------------
print("\n" + "=" * 100)
print("Step 3: summary")
detail_df = pd.DataFrame(all_rows)
check_df = pd.DataFrame(check_rows)
backend_df = pd.DataFrame(backend_rows)
summary_rows = []
for (dataset_name, model_type), g in detail_df.groupby(["Dataset", "Model"], sort=False):
    alt = g[g["Seed"] != PRIMARY_SEED]
    summary_rows.append({
        "Dataset": dataset_name, "Model": model_type, "N_Seeds": int(len(g)),
        "MacroF1_Mean": g["MacroF1"].mean(), "MacroF1_SD": g["MacroF1"].std(ddof=1),
        "Prediction_Agreement_vs_42_Mean_4Seeds": alt["Prediction_Agreement_vs_42"].mean(),
        "Prediction_Agreement_vs_42_Min_4Seeds": alt["Prediction_Agreement_vs_42"].min(),
        "Feature_Spearman_vs_42_Mean_4Seeds": alt["Feature_Spearman_vs_42"].mean(),
        "Feature_Spearman_vs_42_Min_4Seeds": alt["Feature_Spearman_vs_42"].min(),
        "Temporal_Spearman_vs_42_Mean_4Seeds": alt["Temporal_Spearman_vs_42"].mean(),
        "Temporal_Spearman_vs_42_Min_4Seeds": alt["Temporal_Spearman_vs_42"].min(),
        "Top5_Overlap_vs_42_Mean_4Seeds": alt["Top5_Overlap_vs_42"].mean(),
        "IG_Last10_Mass_Mean": g["IG_Last10_Mass"].mean(),
        "All_Seeds_Last10_gt_First10": bool((g["IG_Last10_Mass"] > g["IG_First10_Mass"]).all()),
        "Old_Feature_Spearman_Mean_4Seeds": alt["Old_Feature_Spearman_vs_42"].mean(),
        "Old_Temporal_Spearman_Mean_4Seeds": alt["Old_Temporal_Spearman_vs_42"].mean()})
summary_df = pd.DataFrame(summary_rows)

detail_df.to_csv(OUTPUT_DIR / "multiseed_detail_corrected.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "multiseed_summary_corrected.csv", index=False)
check_df.to_csv(OUTPUT_DIR / "prediction_check_vs_original.csv", index=False)
backend_df.to_csv(OUTPUT_DIR / "seed42_backend_check.csv", index=False)
pd.concat(feature_rows, ignore_index=True).to_csv(OUTPUT_DIR / "ig_feature_all_seeds_corrected.csv", index=False)
pd.concat(timestep_rows, ignore_index=True).to_csv(OUTPUT_DIR / "ig_timestep_all_seeds_corrected.csv", index=False)

n_exp = len(COMBOS) * len(SEEDS)
print(f"Detail rows: {len(detail_df)} (expected {n_exp}); summary rows: {len(summary_df)} (expected {len(COMBOS)})")
if len(detail_df) != n_exp:
    raise AssertionError("Number of result rows differs from expected")
n_match = int(check_df["Match"].sum())
print(f"Check B count check (Macro-F1, correct samples, agreeing samples) passed: {n_match}/{len(check_df)}")
print("\nCheck B per-model details:")
print(check_df[["Dataset", "Model", "Seed", "MacroF1_New", "MacroF1_Old", "Delta_Correct_N", "Delta_Agree_N",
                "Near_Boundary_N_Model", "Near_Boundary_N_Pair", "Min_Logit_Margin", "Match"]].to_string(
    index=False, float_format=lambda x: f"{x:.6f}"))
_s42 = check_df[check_df["Seed"] == PRIMARY_SEED]
print(f"Seed-42 models (identical weights and data; differences may come from the computing environment) whose Macro-F1 differs from the original results: {int((_s42['MacroF1_New'] != _s42['MacroF1_Old']).sum())}/{len(_s42)}; "
      f"with a change in correct samples: {int((_s42['Delta_Correct_N'] != 0).sum())}/{len(_s42)}")
n_backend = int(backend_df["Match"].sum())
print(f"\nCheck C per-sample cross-backend check of seed 42 passed: {n_backend}/{len(backend_df)}")
print(backend_df.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
print(f"Max change in between-class logit differences between GPU native and CPU (all seed-42 models): {backend_df['Max_Pairwise_Diff_GPU_vs_CPU'].max():.2e}, "
      f"MARGIN_EPS {MARGIN_EPS}")

print("\nCorrected summary (new seeds relative to seed 42; columns without prefix are corrected, Old_ columns are original results):")
cols = ["Dataset", "Model", "MacroF1_Mean", "MacroF1_SD",
        "Feature_Spearman_vs_42_Mean_4Seeds", "Old_Feature_Spearman_Mean_4Seeds", "Feature_Spearman_vs_42_Min_4Seeds",
        "Temporal_Spearman_vs_42_Mean_4Seeds", "Old_Temporal_Spearman_Mean_4Seeds", "Temporal_Spearman_vs_42_Min_4Seeds",
        "Prediction_Agreement_vs_42_Mean_4Seeds", "Top5_Overlap_vs_42_Mean_4Seeds", "All_Seeds_Last10_gt_First10"]
print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPer-seed details:")
print(detail_df[["Dataset", "Model", "Seed", "MacroF1", "Prediction_Agreement_vs_42",
                 "Feature_Spearman_vs_42", "Old_Feature_Spearman_vs_42",
                 "Temporal_Spearman_vs_42", "Old_Temporal_Spearman_vs_42",
                 "IG_Last10_to_First10", "Old_IG_Last10_to_First10"]].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nCombinations with last 10 days > first 10 days for all seeds: {int(summary_df['All_Seeds_Last10_gt_First10'].sum())}/{len(summary_df)}")
print(f"Output files: ./{OUTPUT_DIR.as_posix()}/multiseed_detail_corrected.csv, multiseed_summary_corrected.csv, "
      f"prediction_check_vs_original.csv, seed42_backend_check.csv, ig_feature_all_seeds_corrected.csv, "
      f"ig_timestep_all_seeds_corrected.csv")
if not check_df["Match"].all() or not backend_df["Match"].all():
    print("\nError: the following checks failed; this batch cannot be used as the formal corrected results (files saved for troubleshooting):")
    if not check_df["Match"].all():
        print("Check B:")
        print(check_df[~check_df["Match"]].to_string(index=False))
    if not backend_df["Match"].all():
        print("Check C:")
        print(backend_df[~backend_df["Match"]].to_string(index=False))
    raise AssertionError(f"Checks failed: B {len(check_df) - n_match}/{len(check_df)}, C {len(backend_df) - n_backend}/{len(backend_df)}")
print("Checks A, B and C all passed (count check and seed-42 cross-backend check passed; the original results did not save per-sample predictions, so this is not described as strict reproduction)")
print(f"stage6_cell_K_data_cnmultiseed_V7 finished, total time {elapsed()}")
