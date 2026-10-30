# -*- coding: utf-8 -*-
"""
stage5_cell_A_bootstrap_V4

Goals:
1. For those in Table 2"Best RNN vs best conventional baseline"Paired bootstrap of the Macro-F1 / Accuracy difference;
2. Main analysis: moving-block bootstrap by trading day (all stock samples of the same trading day are drawn together,
   one block = 20 consecutive trading days), preserving cross-sectional correlation and temporal autocorrelation;
3. Sensitivity analysis: ordinary sample-level i.i.d. bootstrap;
4. Supplement: each of the 3 RNNs of each dataset is compared with the best baseline;
5. BH-FDR correction of the 5 one-sided p-values of the main comparison.

Principles:
- No RNN is retrained; only the existing seed=42 checkpoints are read;
- Refit only, for each dataset, the"Best conventional baseline"(configuration identical to C_V3), and save per-sample predictions;
- RNN Uses the same X_test built once with the baseline, so the sample order is naturally aligned;
- First check that point estimates match the values reported in A_V9 / C_V3, then run the bootstrap;
- No folder scanning; all paths are given explicitly;
- Console messages were originally in Chinese; output fields are in English.
"""

import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# =========================
# 0. Global configuration
# =========================

OUTPUT_DIR = Path("./stage5/cell_A_bootstrap_V4")
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 60
# Inference only; CPU by default, to avoid cuDNN initialization errors when the GPU is used by other processes
USE_GPU = False
DEVICE = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.enabled = False
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
EVAL_BATCH_SIZE = 512

N_BOOT = 2000          # bootstrap Number of repeats
BLOCK_LEN = 20         # moving-block Length (trading days)
ALPHA = 0.05
POINT_TOL = 5e-4       # Tolerance for the point-estimate check

RUN_DATASETS = ["Medical-1", "Medical-2", "Power", "US-Med", "US-Pow"]
MODELS = ["GRU", "BiLSTM", "LSTM"]

# Automatic root folder search: look for xai_cn / XAI_cn and xai_us
# If still not found, change the two variables below to full absolute paths (for example "/home/i/xxx/xai_cn")
CN_ROOT_MANUAL = None
US_ROOT_MANUAL = None

CWD = Path.cwd().resolve()
print(f"Current working folder: {CWD}")
SEARCH_BASES = [CWD, *CWD.parents,
                Path("/home/i/2026_0712/stock_i"), Path("/home/i/stock_i"),
                Path("/home/i/XAI"), Path("/home/i")]


def resolve_root(names, marker, label, manual):
    """Search the candidate base folders for a folder named names (case-insensitive) and confirm it with the marker subfolder."""
    if manual is not None:
        p = Path(manual)
        if (p / marker).is_dir():
            print(f"{label} root folder (manual): {p}")
            return p
        raise FileNotFoundError(f"{label} Invalid manual path: {p / marker}")
    names_low = {n.lower() for n in names}
    checked = []
    for base in SEARCH_BASES:
        cands = [base] if base.name.lower() in names_low else []
        if base.is_dir():
            try:
                cands += [c for c in base.iterdir() if c.is_dir() and c.name.lower() in names_low]
            except PermissionError:
                pass
        for c in cands:
            checked.append(str(c))
            if (c / marker).is_dir():
                print(f"{label} Root folder: {c}")
                return c
    msg = "\n".join(checked) if checked else "(no folder with this name found)"
    raise FileNotFoundError(
        f"{label} root folder not found. Checked:\n{msg}\n"
        f"Run in a terminal find /home/i -maxdepth 4 -type d -iname '{names[0]}', "
        f"then put the result into {label} corresponding *_ROOT_MANUAL")


CN_ROOT = resolve_root(["xai_cn"], "datasets/processed/dataset1_A_grade", "Chinese data", CN_ROOT_MANUAL)
US_ROOT = resolve_root(["xai_us"], "datasets/processed/dataset4_m", "U.S. data", US_ROOT_MANUAL)

DATA_DIR = {
    "Medical-1": Path(CN_ROOT, "datasets/processed/dataset1_A_grade"),
    "Medical-2": Path(CN_ROOT, "datasets/processed/dataset2_AB_grade"),
    "Power": Path(CN_ROOT, "e3/datasets/processed/dataset2_AB_grade"),
    "US-Med": Path(US_ROOT, "datasets/processed/dataset4_m"),
    "US-Pow": Path(US_ROOT, "datasets/processed/dataset5_p"),
}

MODEL_PATH = {
    ("Medical-1", "GRU"): Path(CN_ROOT, "models_XAI/rnn_enhanced_dataset2/gru_dataset1_SELECTED_13_seed42.pt"),
    ("Medical-1", "BiLSTM"): Path(CN_ROOT, "models_XAI/rnn_enhanced_dataset2/bilstm_dataset1_SELECTED_13_seed42.pt"),
    ("Medical-1", "LSTM"): Path(CN_ROOT, "models_XAI/lstm_enhanced/lstm_dataset1_SELECTED_13_seed42.pt"),

    ("Medical-2", "GRU"): Path(CN_ROOT, "models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_14_seed42.pt"),
    ("Medical-2", "BiLSTM"): Path(CN_ROOT, "models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_14_seed42.pt"),
    ("Medical-2", "LSTM"): Path(CN_ROOT, "models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_14_seed42.pt"),

    ("Power", "GRU"): Path(CN_ROOT, "e3/models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_12_seed42.pt"),
    ("Power", "BiLSTM"): Path(CN_ROOT, "e3/models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_12_seed42.pt"),
    ("Power", "LSTM"): Path(CN_ROOT, "e3/models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_12_seed42.pt"),

    ("US-Med", "GRU"): Path(US_ROOT, "models_XAI/rnn_models_us/gru_dataset4_m_SELECTED_13_seed42.pt"),
    ("US-Med", "BiLSTM"): Path(US_ROOT, "models_XAI/rnn_bilstm_us_60/bilstm_dataset4_m_SELECTED_13_seed42.pt"),
    ("US-Med", "LSTM"): Path(US_ROOT, "models_XAI/lstm_us_60/lstm_dataset4_m_SELECTED_13_seed42.pt"),

    ("US-Pow", "GRU"): Path(US_ROOT, "models_XAI/rnn_models_us/gru_dataset5_p_SELECTED_12_seed42.pt"),
    ("US-Pow", "BiLSTM"): Path(US_ROOT, "models_XAI/rnn_bilstm_us_60/bilstm_dataset5_p_SELECTED_12_seed42.pt"),
    ("US-Pow", "LSTM"): Path(US_ROOT, "models_XAI/lstm_us_60/lstm_dataset5_p_SELECTED_12_seed42.pt"),
}

DATASETS = {
    "Medical-1": {
        "filter_train_2010": False,
        "features": [
            "dayofyear", "day", "volume_ratio", "close_to_ma5", "vol", "low",
            "close_open_range", "macd_signal", "close_to_ma20", "roc_20",
            "high_low_range", "turnover_rate", "amount"
        ],
    },
    "Medical-2": {
        "filter_train_2010": True,
        "features": [
            "high_low_range", "turnover_rate", "amount", "bb_width", "roc_10",
            "roc_5", "close_to_ma20", "close_to_ma5", "close_open_range", "atr",
            "volume_ratio", "roc_20", "dayofyear_cos", "year"
        ],
    },
    "Power": {
        "filter_train_2010": True,
        "features": [
            "dayofyear", "atr", "roc_5", "high_low_range", "turnover_rate",
            "macd_signal", "close", "macd", "month", "close_to_ma5",
            "roc_10", "amount"
        ],
    },
    "US-Med": {
        "filter_train_2010": False,
        "features": [
            "roc_5", "low", "close_to_ma20", "roc_20", "high_low_range",
            "rsi", "dayofweek", "macd", "dayofyear_sin", "bb_width",
            "bb_position", "macd_signal", "dayofyear"
        ],
    },
    "US-Pow": {
        "filter_train_2010": False,
        "features": [
            "roc_5", "dayofyear_sin", "macd_signal", "ma10", "macd",
            "dayofyear", "close", "high_low_range", "low", "close_to_ma20",
            "roc_20", "pre_close"
        ],
    },
}

# Best conventional baseline of each dataset (from C_V3 / Table 2)
BEST_BASELINE = {
    "Medical-1": ("Extra Trees", "balanced"),
    "Medical-2": ("Logistic Regression", "balanced"),
    "Power": ("Extra Trees", None),
    "US-Med": ("Extra Trees", "balanced"),
    "US-Pow": ("Logistic Regression", None),
}

# Reported point estimates (for the hard check)
EXPECTED_RNN_F1 = {
    ("Medical-1", "GRU"): 0.413231, ("Medical-1", "BiLSTM"): 0.403506, ("Medical-1", "LSTM"): 0.412939,
    ("Medical-2", "GRU"): 0.412268, ("Medical-2", "BiLSTM"): 0.419882, ("Medical-2", "LSTM"): 0.412224,
    ("Power", "GRU"): 0.410313, ("Power", "BiLSTM"): 0.412120, ("Power", "LSTM"): 0.402199,
    ("US-Med", "GRU"): 0.356312, ("US-Med", "BiLSTM"): 0.358022, ("US-Med", "LSTM"): 0.363922,
    ("US-Pow", "GRU"): 0.327094, ("US-Pow", "BiLSTM"): 0.365106, ("US-Pow", "LSTM"): 0.332670,
}
EXPECTED_BASELINE_F1 = {
    "Medical-1": 0.390459,
    "Medical-2": 0.377716,
    "Power": 0.398032,
    "US-Med": 0.359013,
    "US-Pow": 0.356558,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# 1. data (consistent with A_V9 / C_V3)
# =========================

def load_data_raw(data_dir):
    required = ["train_X.csv", "val_X.csv", "test_X.csv",
                "train_y.csv", "val_y.csv", "test_y.csv"]
    missing = [x for x in required if not (data_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Data file does not exist: {data_dir} | Missing: {missing}")
    return (
        (pd.read_csv(data_dir / "train_X.csv"), pd.read_csv(data_dir / "train_y.csv")),
        (pd.read_csv(data_dir / "val_X.csv"), pd.read_csv(data_dir / "val_y.csv")),
        (pd.read_csv(data_dir / "test_X.csv"), pd.read_csv(data_dir / "test_y.csv")),
    )


def filter_data_by_date(X_df, y_df, start_date="2010-01-01"):
    X_df, y_df = X_df.copy(), y_df.copy()
    X_df["trade_date"] = pd.to_datetime(X_df["trade_date"])
    y_df["trade_date"] = pd.to_datetime(y_df["trade_date"])
    start = pd.to_datetime(start_date)
    return X_df[X_df["trade_date"] >= start].copy(), y_df[y_df["trade_date"] >= start].copy()


def calculate_thresholds(y_df):
    arr = y_df["pct_chg"].dropna()
    return np.percentile(arr, 33.33), np.percentile(arr, 66.67)


def build_sequence_data_with_meta(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
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
        raise ValueError(f"Missing feature: {missing}")

    merged = pd.merge(
        X_work,
        y_work[["stock_code", "trade_date", "label", "pct_chg", "future_change"]],
        on=["stock_code", "trade_date"], how="inner",
    )

    all_seqs, all_labels, all_meta = [], [], []
    for _, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue
        g = g.sort_values("trade_date")
        feats = g[feature_cols].values
        dates = g["trade_date"].values
        labels = g["label"].values
        meta = g[["stock_code", "trade_date", "label", "pct_chg", "future_change"]].values

        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len
            days_diff = (dates[end - 1] - dates[start_idx]) / np.timedelta64(1, "D")
            if days_diff > 100:
                continue
            target = labels[end - 1]
            if target == -999:
                continue
            all_seqs.append(feats[start_idx:end])
            all_labels.append(target)
            all_meta.append(meta[end - 1])

    meta_df = pd.DataFrame(all_meta, columns=["stock_code", "trade_date", "label", "pct_chg", "future_change"])
    return np.stack(all_seqs).astype(np.float32), np.asarray(all_labels, dtype=np.int64), meta_df


def prepare_dataset(dataset_name):
    cfg = DATASETS[dataset_name]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(DATA_DIR[dataset_name])
    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)
    q33, q67 = calculate_thresholds(y_tr)
    X_train, y_train, _ = build_sequence_data_with_meta(X_tr, y_tr, cfg["features"], q33, q67, SEQ_LEN)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, cfg["features"], q33, q67, SEQ_LEN)
    print(f"{dataset_name}: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test, meta_test


# =========================
# 2. RNN models (consistent with A_V9)
# =========================

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        x = torch.nan_to_num(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = torch.nan_to_num(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def load_rnn(dataset_name, model_type):
    path = MODEL_PATH[(dataset_name, model_type)]
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name}-{model_type}: Model file does not exist: {path}")
    state = clean_state_dict(torch.load(path, map_location="cpu"))
    prefix = "gru" if model_type == "GRU" else "lstm"
    input_dim = state[f"{prefix}.weight_ih_l0"].shape[1]
    hidden_dim = state[f"{prefix}.weight_hh_l0"].shape[1]
    layers = [int(k.split("_l")[1].split("_")[0]) for k in state
              if k.startswith(f"{prefix}.weight_ih_l") and "_reverse" not in k]
    num_layers = max(layers) + 1
    cls = {"GRU": GRUClassifier, "LSTM": LSTMClassifier, "BiLSTM": BiLSTMClassifier}[model_type]
    model = cls(input_dim, hidden_dim, num_layers)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def predict_rnn(model, X):
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), EVAL_BATCH_SIZE):
            xb = torch.from_numpy(X[s:s + EVAL_BATCH_SIZE]).to(DEVICE)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# =========================
# 3. Best conventional baseline (identical to C_V3)
# =========================

def fit_predict_baseline(dataset_name, X_train, y_train, X_test):
    name, cw = BEST_BASELINE[dataset_name]
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)

    if name == "Logistic Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto",
                                       class_weight=cw, n_jobs=None, random_state=SEED)),
        ])
    else:
        model = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                     class_weight=cw, n_jobs=-1, random_state=SEED)
    model.fit(Xtr, y_train)
    weighting = "Balanced" if cw == "balanced" else "Unweighted"
    return model.predict(Xte).astype(np.int64), name, weighting


# =========================
# 4. Quick metrics and bootstrap
# =========================

def macro_f1_fast(y, p):
    """and sklearn f1_score(average='macro', zero_division=0) equivalent fast version."""
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)
    true_sum = cm.sum(axis=1)
    prec = np.divide(tp, pred_sum, out=np.zeros(3), where=pred_sum > 0)
    rec = np.divide(tp, true_sum, out=np.zeros(3), where=true_sum > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros(3), where=denom > 0)
    return f1.mean()


def build_date_groups(meta_test):
    dates = pd.to_datetime(meta_test["trade_date"]).values
    uniq, inv = np.unique(dates, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    return groups, len(uniq)


def block_resample(rng, groups, n_dates):
    """Moving-block bootstrap: contiguous blocks drawn by trading day, with all stocks of the same day kept together."""
    L = min(BLOCK_LEN, n_dates)
    n_blocks = int(math.ceil(n_dates / L))
    starts = rng.integers(0, n_dates - L + 1, size=n_blocks)
    sel = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n_dates]
    return np.concatenate([groups[d] for d in sel])


def iid_resample(rng, n):
    return rng.integers(0, n, size=n)


def run_bootstrap(y_test, pred_dict, groups, n_dates, scheme):
    """
    pred_dict: {name: pred}. Each resample uses the same indices for all models (paired).
    Return {name: {"f1": array, "acc": array}}.
    """
    rng = np.random.default_rng(SEED + (0 if scheme == "block" else 1))
    out = {k: {"f1": np.empty(N_BOOT), "acc": np.empty(N_BOOT)} for k in pred_dict}
    n = len(y_test)

    for b in range(N_BOOT):
        idx = block_resample(rng, groups, n_dates) if scheme == "block" else iid_resample(rng, n)
        yb = y_test[idx]
        for k, p in pred_dict.items():
            pb = p[idx]
            out[k]["f1"][b] = macro_f1_fast(yb, pb)
            out[k]["acc"][b] = (pb == yb).mean()

        if (b + 1) % 500 == 0:
            print(f"  {scheme} bootstrap Progress: {b + 1}/{N_BOOT}")
    return out


def summarize_diff(diff):
    """95% percentile interval and one-sided p-value of the difference (H1: RNN > baseline)."""
    return {
        "CI_Lower": float(np.percentile(diff, 2.5)),
        "CI_Upper": float(np.percentile(diff, 97.5)),
        "p_one_sided": float((np.sum(diff <= 0) + 1) / (len(diff) + 1)),
    }


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


# =========================
# 5. Main flow
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Task: best RNN vs best conventional baseline, paired bootstrap")
    print(f"Main analysis: trading day moving-block bootstrap, block={BLOCK_LEN}, Repeat={N_BOOT}")
    print(f"Sensitivity analysis: sample-level i.i.d. bootstrap, repetitions={N_BOOT}")

    check_rows, primary_rows, all_rows, error_rows = [], [], [], []

    for dataset_name in RUN_DATASETS:
        print("-" * 20)
        print(f"Dataset: {dataset_name}")

        try:
            print("Step 1: data preparation")
            X_train, y_train, X_test, y_test, meta_test = prepare_dataset(dataset_name)
            groups, n_dates = build_date_groups(meta_test)
            print(f"Test samples: {len(y_test)}, Test trading days: {n_dates}")

            print("Step 2: RNN predictions and point-estimate check")
            preds = {}
            for model_type in MODELS:
                model = load_rnn(dataset_name, model_type)
                p = predict_rnn(model, X_test)
                f1 = f1_score(y_test, p, average="macro")
                exp = EXPECTED_RNN_F1[(dataset_name, model_type)]
                ok = abs(f1 - exp) <= POINT_TOL
                print(f"{model_type}: Macro-F1={f1:.6f}, Reported={exp:.6f}, Consistent={ok}")
                check_rows.append({"Dataset": dataset_name, "Model": model_type,
                                   "Recomputed_MacroF1": f1, "Reported_MacroF1": exp, "Match": ok})
                preds[model_type] = p
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print("Step 3: refit and re-check the best baseline")
            p_base, base_name, base_w = fit_predict_baseline(dataset_name, X_train, y_train, X_test)
            f1_base = f1_score(y_test, p_base, average="macro")
            exp_b = EXPECTED_BASELINE_F1[dataset_name]
            ok_b = abs(f1_base - exp_b) <= POINT_TOL
            print(f"{base_name}/{base_w}: Macro-F1={f1_base:.6f}, Reported={exp_b:.6f}, Consistent={ok_b}")
            check_rows.append({"Dataset": dataset_name, "Model": f"{base_name} ({base_w})",
                               "Recomputed_MacroF1": f1_base, "Reported_MacroF1": exp_b, "Match": ok_b})
            preds["Baseline"] = p_base

            # Save per-sample predictions for reproduction from the code repository
            pred_df = meta_test.copy()
            pred_df["y_true"] = y_test
            for k, v in preds.items():
                pred_df[f"pred_{k}"] = v
            pred_df.to_csv(OUTPUT_DIR / f"A1_predictions_{dataset_name}.csv", index=False)

            if not all(r["Match"] for r in check_rows if r["Dataset"] == dataset_name):
                print("Warning: point estimate does not match the reported value; bootstrap still runs, please check the A2 file item by item")

            print("Step 4: paired bootstrap")
            boot = {s: run_bootstrap(y_test, preds, groups, n_dates, s) for s in ["block", "iid"]}

            point_f1 = {k: f1_score(y_test, v, average="macro") for k, v in preds.items()}
            point_acc = {k: accuracy_score(y_test, v) for k, v in preds.items()}
            best_rnn = max(MODELS, key=lambda m: point_f1[m])

            for model_type in MODELS:
                row = {
                    "Dataset": dataset_name,
                    "RNN": model_type,
                    "Baseline": base_name,
                    "Baseline_Weighting": base_w,
                    "N_Test": len(y_test),
                    "N_Test_Dates": n_dates,
                    "RNN_MacroF1": point_f1[model_type],
                    "Baseline_MacroF1": point_f1["Baseline"],
                    "Delta_MacroF1": point_f1[model_type] - point_f1["Baseline"],
                    "RNN_Accuracy": point_acc[model_type],
                    "Baseline_Accuracy": point_acc["Baseline"],
                    "Delta_Accuracy": point_acc[model_type] - point_acc["Baseline"],
                }
                for scheme in ["block", "iid"]:
                    d_f1 = boot[scheme][model_type]["f1"] - boot[scheme]["Baseline"]["f1"]
                    d_acc = boot[scheme][model_type]["acc"] - boot[scheme]["Baseline"]["acc"]
                    s_f1 = summarize_diff(d_f1)
                    s_acc = summarize_diff(d_acc)
                    tag = "Block" if scheme == "block" else "IID"
                    row[f"{tag}_F1_CI_Lower"] = s_f1["CI_Lower"]
                    row[f"{tag}_F1_CI_Upper"] = s_f1["CI_Upper"]
                    row[f"{tag}_F1_p_one_sided"] = s_f1["p_one_sided"]
                    row[f"{tag}_Acc_CI_Lower"] = s_acc["CI_Lower"]
                    row[f"{tag}_Acc_CI_Upper"] = s_acc["CI_Upper"]
                    row[f"{tag}_Acc_p_one_sided"] = s_acc["p_one_sided"]
                all_rows.append(row)
                if model_type == best_rnn:
                    primary_rows.append(row.copy())

            pr = primary_rows[-1]
            print(
                f"Main comparison {best_rnn} vs {base_name}: ΔF1={pr['Delta_MacroF1']:.4f}, "
                f"block 95%CI=[{pr['Block_F1_CI_Lower']:.4f}, {pr['Block_F1_CI_Upper']:.4f}], "
                f"block p={pr['Block_F1_p_one_sided']:.4f}"
            )

        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            error_rows.append({"Dataset": dataset_name, "Error": f"{type(e).__name__}: {e}"})

    print("-" * 20)
    print("Step 5: save and summarize")

    df_check = pd.DataFrame(check_rows)
    df_primary = pd.DataFrame(primary_rows)
    df_all = pd.DataFrame(all_rows)

    if not df_primary.empty:
        df_primary["Block_F1_p_BH_FDR"] = bh_fdr(df_primary["Block_F1_p_one_sided"])
        df_primary["Block_F1_Significant_BH_0.05"] = df_primary["Block_F1_p_BH_FDR"] <= ALPHA
        df_primary["IID_F1_p_BH_FDR"] = bh_fdr(df_primary["IID_F1_p_one_sided"])

    df_check.to_csv(OUTPUT_DIR / "A2_point_estimate_check.csv", index=False)
    df_primary.to_csv(OUTPUT_DIR / "A3_bootstrap_best_rnn_vs_best_baseline.csv", index=False)
    df_all.to_csv(OUTPUT_DIR / "A4_bootstrap_all_rnn_vs_best_baseline.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(OUTPUT_DIR / "A9_errors.csv", index=False)

    print("Final summary 1: point-estimate check")
    print(df_check.to_string(index=False))
    print(f"All consistent: {bool(df_check['Match'].all()) if not df_check.empty else False}")

    print("-" * 20)
    print("Final summary 2: best RNN vs best baseline (Macro-F1 difference)")
    if not df_primary.empty:
        cols = ["Dataset", "RNN", "Baseline", "Delta_MacroF1",
                "Block_F1_CI_Lower", "Block_F1_CI_Upper", "Block_F1_p_one_sided", "Block_F1_p_BH_FDR",
                "IID_F1_CI_Lower", "IID_F1_CI_Upper", "IID_F1_p_one_sided"]
        print(df_primary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        n_sig = int(df_primary["Block_F1_Significant_BH_0.05"].sum())
        print(f"Block bootstrap BH-FDR Significant: {n_sig}/{len(df_primary)}")

    print("-" * 20)
    print("Final summary 3: errors")
    print("Error: 0" if not error_rows else pd.DataFrame(error_rows).to_string(index=False))

    print("-" * 20)
    print("stage5_cell_A_bootstrap_V4 Done")
    print(f"Results folder: {OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
