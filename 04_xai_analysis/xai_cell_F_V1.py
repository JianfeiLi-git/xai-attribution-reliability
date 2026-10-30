# -*- coding: utf-8 -*-
"""
xai_cell_F_V1

Goals:
1. Add a temporal SHAP check that does not depend on the gradient path;
2. Treat each of the 60 historical timesteps as one of 60 KernelSHAP players;
3. Missing timesteps are replaced by the full feature vector at the same position of a real training window;
4. Compare Temporal KernelSHAP with IG on the same test samples and the same fixed predicted targets of the original model;
5. and also summarize the recency pattern at the level of 60 timesteps and 6×10-day block level.

Principles:
- No model retraining;
- Only the existing seed=42 checkpoints are read;
- Does not read any A/B/C/D/E result files;
- IG baseline Only the training-set mean trajectory is used;
- KernelSHAP target is the original model's predicted-class probability;
- All backgrounds come from the training set only;
- No folder scanning.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients
import shap

warnings.filterwarnings("ignore")

SEED = 42
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("./0924/xai_cell_F_V1")
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Temporal KernelSHAP Configuration
N_EXPLAIN = 50
N_BACKGROUNDS = 10
KERNEL_NSAMPLES = 512
KERNEL_BATCH = 512

# IG configuration on the same samples
IG_STEPS = 64
IG_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 512

# Explicit paths: folder scanning is not allowed
# Current notebook location:
# Healthcare: China ./XAI_cn/*.ipynb
# Power: China ./XAI_cn/e3/*.ipynb
# U.S.: ./xai_us/*.ipynb

DATA_DIR = {
    "Medical-1": Path("./XAI_cn/datasets/processed/dataset1_A_grade"),
    "Medical-2": Path("./XAI_cn/datasets/processed/dataset2_AB_grade"),
    "Power": Path("./XAI_cn/e3/datasets/processed/dataset2_AB_grade"),
    "US-Med": Path("./xai_us/datasets/processed/dataset4_m"),
    "US-Pow": Path("./xai_us/datasets/processed/dataset5_p"),
}

MODEL_PATH = {
    ("Medical-1", "GRU"): Path("./XAI_cn/models_XAI/rnn_enhanced_dataset2/gru_dataset1_SELECTED_13_seed42.pt"),
    ("Medical-1", "BiLSTM"): Path("./XAI_cn/models_XAI/rnn_enhanced_dataset2/bilstm_dataset1_SELECTED_13_seed42.pt"),
    ("Medical-1", "LSTM"): Path("./XAI_cn/models_XAI/lstm_enhanced/lstm_dataset1_SELECTED_13_seed42.pt"),

    ("Medical-2", "GRU"): Path("./XAI_cn/models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_14_seed42.pt"),
    ("Medical-2", "BiLSTM"): Path("./XAI_cn/models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_14_seed42.pt"),
    ("Medical-2", "LSTM"): Path("./XAI_cn/models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_14_seed42.pt"),

    ("Power", "GRU"): Path("./XAI_cn/e3/models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_12_seed42.pt"),
    ("Power", "BiLSTM"): Path("./XAI_cn/e3/models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_12_seed42.pt"),
    ("Power", "LSTM"): Path("./XAI_cn/e3/models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_12_seed42.pt"),

    ("US-Med", "GRU"): Path("./xai_us/models_XAI/rnn_models_us/gru_dataset4_m_SELECTED_13_seed42.pt"),
    ("US-Med", "BiLSTM"): Path("./xai_us/models_XAI/rnn_bilstm_us_60/bilstm_dataset4_m_SELECTED_13_seed42.pt"),
    ("US-Med", "LSTM"): Path("./xai_us/models_XAI/lstm_us_60/lstm_dataset4_m_SELECTED_13_seed42.pt"),

    ("US-Pow", "GRU"): Path("./xai_us/models_XAI/rnn_models_us/gru_dataset5_p_SELECTED_12_seed42.pt"),
    ("US-Pow", "BiLSTM"): Path("./xai_us/models_XAI/rnn_bilstm_us_60/bilstm_dataset5_p_SELECTED_12_seed42.pt"),
    ("US-Pow", "LSTM"): Path("./xai_us/models_XAI/lstm_us_60/lstm_dataset5_p_SELECTED_12_seed42.pt"),
}

DATASETS = {
    "Medical-1": {
        "expected_features": 13,
        "filter_train_2010": False,
        "features": [
            "dayofyear", "day", "volume_ratio", "close_to_ma5", "vol", "low",
            "close_open_range", "macd_signal", "close_to_ma20", "roc_20",
            "high_low_range", "turnover_rate", "amount"
        ],
    },
    "Medical-2": {
        "expected_features": 14,
        "filter_train_2010": True,
        "features": [
            "high_low_range", "turnover_rate", "amount", "bb_width", "roc_10",
            "roc_5", "close_to_ma20", "close_to_ma5", "close_open_range", "atr",
            "volume_ratio", "roc_20", "dayofyear_cos", "year"
        ],
    },
    "Power": {
        "expected_features": 12,
        "filter_train_2010": True,
        "features": [
            "dayofyear", "atr", "roc_5", "high_low_range", "turnover_rate",
            "macd_signal", "close", "macd", "month", "close_to_ma5",
            "roc_10", "amount"
        ],
    },
    "US-Med": {
        "expected_features": 13,
        "filter_train_2010": False,
        "features": [
            "roc_5", "low", "close_to_ma20", "roc_20", "high_low_range",
            "rsi", "dayofweek", "macd", "dayofyear_sin", "bb_width",
            "bb_position", "macd_signal", "dayofyear"
        ],
    },
    "US-Pow": {
        "expected_features": 12,
        "filter_train_2010": False,
        "features": [
            "roc_5", "dayofyear_sin", "macd_signal", "ma10", "macd",
            "dayofyear", "close", "high_low_range", "low", "close_to_ma20",
            "roc_20", "pre_close"
        ],
    },
}

RUN_DATASETS = ["Medical-1", "Medical-2", "Power", "US-Med", "US-Pow"]

MODELS = ["GRU", "BiLSTM", "LSTM"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)



def load_data_raw(data_dir):
    X_train = pd.read_csv(data_dir / "train_X.csv")
    X_val = pd.read_csv(data_dir / "val_X.csv")
    X_test = pd.read_csv(data_dir / "test_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv")
    y_val = pd.read_csv(data_dir / "val_y.csv")
    y_test = pd.read_csv(data_dir / "test_y.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


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
        on=["stock_code", "trade_date"], how="inner"
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
            days_diff = (dates[end-1] - dates[start_idx]) / np.timedelta64(1, "D")
            if days_diff > 100:
                continue
            target = labels[end-1]
            if target == -999:
                continue
            all_seqs.append(feats[start_idx:end])
            all_labels.append(target)
            all_meta.append(meta[end-1])

    meta_df = pd.DataFrame(all_meta, columns=["stock_code", "trade_date", "label", "pct_chg", "future_change"])
    return np.stack(all_seqs).astype(np.float32), np.asarray(all_labels, dtype=np.int64), meta_df


def get_data_dir(dataset_name):
    data_dir = DATA_DIR[dataset_name]
    required = [
        "train_X.csv", "val_X.csv", "test_X.csv",
        "train_y.csv", "val_y.csv", "test_y.csv",
    ]
    missing = [x for x in required if not (data_dir / x).exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_name}: Data file does not exist: {data_dir} | Missing: {missing}"
        )
    return data_dir


def prepare_dataset(dataset_name, cfg):
    data_dir = get_data_dir(dataset_name)
    print(f"{dataset_name}: Data folder={data_dir}")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)
    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)
    q33, q67 = calculate_thresholds(y_tr)
    X_train, y_train, _ = build_sequence_data_with_meta(X_tr, y_tr, cfg["features"], q33, q67, SEQ_LEN)
    X_val, y_val, _ = build_sequence_data_with_meta(X_va, y_va, cfg["features"], q33, q67, SEQ_LEN)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, cfg["features"], q33, q67, SEQ_LEN)
    print(f"{dataset_name}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


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


def infer_model_from_state_dict(model_type, state):
    state = clean_state_dict(state)
    mt = model_type.upper()
    prefix = "gru" if mt == "GRU" else "lstm"
    w_ih = state[f"{prefix}.weight_ih_l0"]
    w_hh = state[f"{prefix}.weight_hh_l0"]
    input_dim = w_ih.shape[1]
    hidden_dim = w_hh.shape[1]
    layers = [int(k.split("_l")[1].split("_")[0]) for k in state
              if k.startswith(f"{prefix}.weight_ih_l") and "_reverse" not in k]
    num_layers = max(layers) + 1
    if mt == "GRU":
        model = GRUClassifier(input_dim, hidden_dim, num_layers)
    elif mt == "LSTM":
        model = LSTMClassifier(input_dim, hidden_dim, num_layers)
    else:
        model = BiLSTMClassifier(input_dim, hidden_dim, num_layers)
    model.load_state_dict(state, strict=True)
    return model, input_dim, hidden_dim, num_layers


def get_model_path(dataset_name, model_type):
    model_path = MODEL_PATH[(dataset_name, model_type)]
    if not model_path.exists():
        raise FileNotFoundError(
            f"{dataset_name}-{model_type}: Model file does not exist: {model_path}"
        )
    return model_path


def load_model(dataset_name, model_type, cfg):
    path = get_model_path(dataset_name, model_type)
    state = torch.load(path, map_location="cpu")
    model, input_dim, hidden_dim, num_layers = infer_model_from_state_dict(model_type, state)
    model = model.to(DEVICE).eval()
    print(f"{dataset_name}-{model_type}: {path} | input={input_dim}, hidden={hidden_dim}, layers={num_layers}")
    return model, path


def predict_arrays(model, X, batch_size=512):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s+batch_size]).to(DEVICE)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(prob.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_arrays(model, X, y):
    pred, prob = predict_arrays(model, X, EVAL_BATCH_SIZE)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "mean_true_class_prob": prob[np.arange(len(y)), y].mean(),
        "pred": pred,
        "prob": prob,
    }


def choose_fixed_samples(X_test, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X_test))
    return np.sort(rng.choice(len(X_test), size=n, replace=False))
# =========================
# 5. F Tools specific to this pipeline
# =========================

def predict_prob(model, X, batch_size=512):
    """Returns class probabilities."""
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(
                X[s:s + batch_size].astype(np.float32)
            ).to(DEVICE)
            prob = torch.softmax(model(xb), dim=1)
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs, axis=0)


def choose_explain_samples(n_test, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    n = min(n_samples, n_test)
    return np.sort(rng.choice(n_test, size=n, replace=False))


def ig_temporal_fixed_targets(
    model, X_test, X_train, sample_idx, fixed_targets,
    n_steps=64, batch_size=32
):
    """
    IG:
    - baseline = training-set mean trajectory;
    - target fixed to the class predicted by the original model on clean test samples;
    - Return absolute attribution: N × 60 × F.
    """
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    try:
        ig = IntegratedGradients(model)
        baseline = torch.from_numpy(
            X_train.mean(axis=0, keepdims=True).astype(np.float32)
        ).to(DEVICE)

        out = []
        for s in range(0, len(sample_idx), batch_size):
            idx = sample_idx[s:s + batch_size]
            xb = torch.from_numpy(
                X_test[idx].astype(np.float32)
            ).to(DEVICE)

            targets = torch.from_numpy(
                np.asarray(
                    fixed_targets[s:s + len(idx)],
                    dtype=np.int64
                )
            ).to(DEVICE)

            base = baseline.expand(xb.size(0), -1, -1)

            attr = ig.attribute(
                xb,
                baselines=base,
                target=targets,
                n_steps=n_steps,
                internal_batch_size=min(256, xb.size(0) * 4),
            )

            out.append(attr.detach().abs().cpu().numpy())

            del xb, targets, base, attr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(out, axis=0)

    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()


def make_temporal_value_function(
    model, x_sample, target_class, bg_windows
):
    """
    60 timesteps treated as 60 KernelSHAP players.

    mask[t]=1:
        keep all features of time step t of the test sample;
    mask[t]=0:
        Replaced with all features at the same position of a real training background window.

    For each coalition, the target-class probability is averaged over multiple training backgrounds.
    """
    n_bg, T, F = bg_windows.shape

    if T != SEQ_LEN:
        raise ValueError(f"Abnormal sequence length: {T}")

    def value_fn(Z):
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)

        if Z.shape[1] != T:
            raise ValueError(
                f"Temporal mask Dimension error: {Z.shape[1]} != {T}"
            )

        Z = np.clip(Z, 0.0, 1.0)
        values = []

        for s in range(0, len(Z), KERNEL_BATCH):
            z_batch = Z[s:s + KERNEL_BATCH]
            M = len(z_batch)

            # M × B × T × F
            z = z_batch[:, None, :, None]
            bg = bg_windows[None, :, :, :]
            x = x_sample[None, None, :, :]

            hybrid = z * x + (1.0 - z) * bg
            hybrid = hybrid.reshape(
                M * n_bg, T, F
            ).astype(np.float32)

            prob = predict_prob(
                model, hybrid, EVAL_BATCH_SIZE
            )
            p_target = prob[:, target_class].reshape(
                M, n_bg
            ).mean(axis=1)
            values.append(p_target)

        return np.concatenate(values, axis=0)

    return value_fn


def temporal_kernelshap(
    dataset_name, model_type, model,
    X_train, X_test, sample_idx, fixed_targets
):
    """
    Run 60-player temporal KernelSHAP on the fixed test samples.
    """
    all_rows = []

    # Use deterministic but independent background random streams for different models
    model_offset = {"GRU": 11, "BiLSTM": 23, "LSTM": 37}[model_type]
    dataset_offset = {
        "Medical-1": 101,
        "Medical-2": 211,
        "Power": 307,
        "US-Med": 401,
        "US-Pow": 503,
    }[dataset_name]
    rng = np.random.default_rng(
        SEED + model_offset + dataset_offset
    )

    zero_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
    full_mask = np.ones((1, SEQ_LEN), dtype=np.float32)

    for ii, idx in enumerate(sample_idx, start=1):
        x = X_test[idx]

        bg_idx = rng.choice(
            len(X_train),
            size=min(N_BACKGROUNDS, len(X_train)),
            replace=False,
        )
        bg_windows = X_train[bg_idx]

        target_class = int(fixed_targets[ii - 1])

        value_fn = make_temporal_value_function(
            model, x, target_class, bg_windows
        )

        explainer = shap.KernelExplainer(
            value_fn,
            zero_mask,
            link="identity",
        )

        shap_values = explainer.shap_values(
            full_mask,
            nsamples=KERNEL_NSAMPLES,
            silent=True,
        )

        arr = np.asarray(shap_values)

        # Compatible with different shap versions
        if arr.ndim == 1:
            phi = arr
        elif arr.ndim == 2:
            phi = arr[0]
        elif arr.ndim == 3:
            phi = arr.reshape(-1, arr.shape[-1])[0]
        else:
            raise ValueError(
                f"Cannot recognize the Temporal KernelSHAP output shape: {arr.shape}"
            )

        if len(phi) != SEQ_LEN:
            raise ValueError(
                f"Temporal SHAP Length error: {len(phi)} != {SEQ_LEN}"
            )

        for t in range(SEQ_LEN):
            all_rows.append({
                "Dataset": dataset_name,
                "Model": model_type,
                "Sample_Index": int(idx),
                "Target_Class": target_class,
                "Timestep": t + 1,
                "KernelSHAP_Value": float(phi[t]),
                "Abs_KernelSHAP": float(abs(phi[t])),
            })

        if ii % 10 == 0 or ii == len(sample_idx):
            print(
                f"  Temporal KernelSHAP Progress: "
                f"{ii}/{len(sample_idx)}"
            )

    detail = pd.DataFrame(all_rows)

    summary = (
        detail.groupby(
            ["Dataset", "Model", "Timestep"],
            as_index=False
        )
        .agg(
            KernelSHAP_Importance=(
                "Abs_KernelSHAP", "mean"
            ),
            KernelSHAP_SignedMean=(
                "KernelSHAP_Value", "mean"
            ),
            KernelSHAP_SD=(
                "Abs_KernelSHAP", "std"
            ),
            N_Samples=(
                "Sample_Index", "nunique"
            ),
        )
    )

    total = summary["KernelSHAP_Importance"].sum()
    summary["KernelSHAP_Share"] = (
        summary["KernelSHAP_Importance"]
        / (total + 1e-12)
    )

    return detail, summary


def summarize_ig_temporal(
    dataset_name, model_type, attr
):
    importance = attr.mean(axis=(0, 2))
    share = importance / (importance.sum() + 1e-12)

    return pd.DataFrame({
        "Dataset": dataset_name,
        "Model": model_type,
        "Timestep": np.arange(1, SEQ_LEN + 1),
        "IG_Importance": importance,
        "IG_Share": share,
    })


def build_block_summary(
    dataset_name, model_type,
    ig_time, shap_time
):
    merged = pd.merge(
        ig_time,
        shap_time[
            [
                "Dataset", "Model", "Timestep",
                "KernelSHAP_Importance",
                "KernelSHAP_Share",
            ]
        ],
        on=["Dataset", "Model", "Timestep"],
        how="inner",
        validate="one_to_one",
    )

    merged["Block"] = (
        (merged["Timestep"] - 1) // 10 + 1
    )

    block = (
        merged.groupby(
            ["Dataset", "Model", "Block"],
            as_index=False
        )
        .agg(
            Timestep_Start=("Timestep", "min"),
            Timestep_End=("Timestep", "max"),
            IG_Mass=("IG_Share", "sum"),
            KernelSHAP_Mass=(
                "KernelSHAP_Share", "sum"
            ),
        )
    )

    return merged, block


def method_comparison(
    dataset_name, model_type,
    merged_time, block
):
    rho60, p60 = spearmanr(
        merged_time["IG_Importance"],
        merged_time["KernelSHAP_Importance"],
    )

    rho6, p6 = spearmanr(
        block["IG_Mass"],
        block["KernelSHAP_Mass"],
    )

    ig_first = float(
        merged_time.loc[
            merged_time["Timestep"] <= 10,
            "IG_Share"
        ].sum()
    )
    ig_last = float(
        merged_time.loc[
            merged_time["Timestep"] >= 51,
            "IG_Share"
        ].sum()
    )

    sh_first = float(
        merged_time.loc[
            merged_time["Timestep"] <= 10,
            "KernelSHAP_Share"
        ].sum()
    )
    sh_last = float(
        merged_time.loc[
            merged_time["Timestep"] >= 51,
            "KernelSHAP_Share"
        ].sum()
    )

    ig_peak = int(
        block.loc[
            block["IG_Mass"].idxmax(),
            "Block"
        ]
    )
    sh_peak = int(
        block.loc[
            block["KernelSHAP_Mass"].idxmax(),
            "Block"
        ]
    )

    return {
        "Dataset": dataset_name,
        "Model": model_type,
        "Timestep_Spearman": float(rho60),
        "Timestep_p_value": float(p60),
        "Block_Spearman": float(rho6),
        "Block_p_value": float(p6),
        "IG_First10_Mass": ig_first,
        "IG_Last10_Mass": ig_last,
        "IG_Last10_to_First10": float(
            ig_last / (ig_first + 1e-12)
        ),
        "KernelSHAP_First10_Mass": sh_first,
        "KernelSHAP_Last10_Mass": sh_last,
        "KernelSHAP_Last10_to_First10": float(
            sh_last / (sh_first + 1e-12)
        ),
        "IG_Peak_Block": ig_peak,
        "KernelSHAP_Peak_Block": sh_peak,
        "Same_Peak_Block": bool(
            ig_peak == sh_peak
        ),
        "N_Explain": int(N_EXPLAIN),
        "N_Backgrounds": int(N_BACKGROUNDS),
        "KernelSHAP_NSamples": int(KERNEL_NSAMPLES),
        "IG_Steps": int(IG_STEPS),
    }


# =========================
# 6. Main flow
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Task: F pipeline, temporal non-path SHAP check")
    print(
        "Temporal KernelSHAP: "
        "60 timesteps = 60 independent player"
    )
    print(
        f"Number of explained samples: {N_EXPLAIN}, "
        f"training backgrounds: {N_BACKGROUNDS}"
    )
    print(
        f"KernelSHAP nsamples: {KERNEL_NSAMPLES}, "
        f"IG steps: {IG_STEPS}"
    )
    print(
        "Main comparison: same samples, same fixed targets of the original model "
        "IG vs Temporal KernelSHAP"
    )

    all_model_check = []
    all_shap_detail = []
    all_shap_time = []
    all_ig_time = []
    all_merged_time = []
    all_blocks = []
    all_compare = []
    errors = []

    for dataset_name in RUN_DATASETS:
        cfg = DATASETS[dataset_name]

        print("-" * 20)
        print(f"Dataset: {dataset_name}")
        print("Step 1: data preparation and audit")

        try:
            ds = prepare_dataset(
                dataset_name, cfg
            )

            print(f"Number of fields: {len(cfg['features'])}")
            print(
                f"Training samples: {len(ds['X_train'])}, "
                f"Validation samples: {len(ds['X_val'])}, "
                f"Test samples: {len(ds['X_test'])}"
            )
            print(
                "Training missing values: "
                f"{int(np.isnan(ds['X_train']).sum())}, "
                "Validation missing values: "
                f"{int(np.isnan(ds['X_val']).sum())}, "
                "Test missing values: "
                f"{int(np.isnan(ds['X_test']).sum())}"
            )

            if ds["X_train"].shape[1] != SEQ_LEN:
                raise ValueError(
                    "Training sequence length is not 60"
                )
            if ds["X_test"].shape[1] != SEQ_LEN:
                raise ValueError(
                    "Test sequence length is not 60"
                )
            if ds["X_train"].shape[2] != cfg[
                "expected_features"
            ]:
                raise ValueError(
                    "Number of training features differs from the expected number"
                )

            sample_idx = choose_explain_samples(
                len(ds["X_test"]),
                N_EXPLAIN,
                SEED,
            )
            print(
                f"Number of fixed explained samples: {len(sample_idx)}"
            )
            print("Data audit: passed")

        except Exception as e:
            print(
                f"Error: data preparation failed: "
                f"{type(e).__name__}: {e}"
            )
            errors.append({
                "Dataset": dataset_name,
                "Model": "",
                "Step": "Data",
                "Error": f"{type(e).__name__}: {e}",
            })
            continue

        for model_type in MODELS:
            print("-" * 20)
            print(
                f"{dataset_name} / {model_type}"
            )

            try:
                print("Step 2: load and check the original model")
                model, model_path = load_model(
                    dataset_name,
                    model_type,
                    cfg,
                )

                base = evaluate_arrays(
                    model,
                    ds["X_test"],
                    ds["y_test"],
                )

                print(
                    "Original model check: "
                    f"Macro-F1={base['macro_f1']:.6f}, "
                    f"Accuracy={base['accuracy']:.6f}"
                )

                all_model_check.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Model_Path": str(model_path),
                    "Test_N": len(ds["y_test"]),
                    "Macro_F1": base["macro_f1"],
                    "Accuracy": base["accuracy"],
                })

                fixed_targets = base["pred"][
                    sample_idx
                ].astype(np.int64)

                counts = {
                    int(k): int(v)
                    for k, v in zip(
                        *np.unique(
                            fixed_targets,
                            return_counts=True
                        )
                    )
                }
                print(
                    f"Fixed target distribution: {counts}"
                )
                print("Model check: passed")

            except Exception as e:
                print(
                    f"Error: model check failed: "
                    f"{type(e).__name__}: {e}"
                )
                errors.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Model",
                    "Error": (
                        f"{type(e).__name__}: {e}"
                    ),
                })
                continue

            try:
                print(
                    "Step 3: Temporal KernelSHAP "
                    "(60 timestep players)"
                )

                sh_detail, sh_time = (
                    temporal_kernelshap(
                        dataset_name,
                        model_type,
                        model,
                        ds["X_train"],
                        ds["X_test"],
                        sample_idx,
                        fixed_targets,
                    )
                )

                all_shap_detail.append(sh_detail)
                all_shap_time.append(sh_time)

                top10 = (
                    sh_time.sort_values(
                        "KernelSHAP_Importance",
                        ascending=False
                    )
                    .head(10)[
                        [
                            "Timestep",
                            "KernelSHAP_Importance",
                            "KernelSHAP_Share",
                        ]
                    ]
                )
                print(
                    "Temporal KernelSHAP "
                    "Top10 timesteps:"
                )
                print(
                    top10.to_string(
                        index=False
                    )
                )
                print(
                    "Temporal KernelSHAP: Done"
                )

            except Exception as e:
                print(
                    "Error: Temporal KernelSHAP failed: "
                    f"{type(e).__name__}: {e}"
                )
                errors.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Temporal KernelSHAP",
                    "Error": (
                        f"{type(e).__name__}: {e}"
                    ),
                })
                continue

            try:
                print(
                    "Step 4: same samples IG temporal "
                    "attribution"
                )

                ig_attr = ig_temporal_fixed_targets(
                    model,
                    ds["X_test"],
                    ds["X_train"],
                    sample_idx,
                    fixed_targets,
                    n_steps=IG_STEPS,
                    batch_size=IG_BATCH_SIZE,
                )

                ig_time = summarize_ig_temporal(
                    dataset_name,
                    model_type,
                    ig_attr,
                )
                all_ig_time.append(ig_time)

                print(
                    f"IG timestep Rows: "
                    f"{len(ig_time)}"
                )
                print("IG temporal: Done")

            except Exception as e:
                print(
                    f"Error: IG failed: "
                    f"{type(e).__name__}: {e}"
                )
                errors.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "IG",
                    "Error": (
                        f"{type(e).__name__}: {e}"
                    ),
                })
                continue

            try:
                print(
                    "Step 5: IG vs Temporal "
                    "KernelSHAP Comparison"
                )

                merged_time, block = (
                    build_block_summary(
                        dataset_name,
                        model_type,
                        ig_time,
                        sh_time,
                    )
                )

                comp = method_comparison(
                    dataset_name,
                    model_type,
                    merged_time,
                    block,
                )

                all_merged_time.append(
                    merged_time
                )
                all_blocks.append(block)
                all_compare.append(comp)

                print(
                    "60-timestep Spearman: "
                    f"rho={comp['Timestep_Spearman']:.6f}, "
                    f"p={comp['Timestep_p_value']:.6g}"
                )
                print(
                    "6-block Spearman: "
                    f"rho={comp['Block_Spearman']:.6f}, "
                    f"p={comp['Block_p_value']:.6g}"
                )
                print(
                    "IG Recency: "
                    f"Last10={comp['IG_Last10_Mass']:.6f}, "
                    f"First10={comp['IG_First10_Mass']:.6f}, "
                    f"Ratio={comp['IG_Last10_to_First10']:.6f}"
                )
                print(
                    "KernelSHAP Recency: "
                    f"Last10={comp['KernelSHAP_Last10_Mass']:.6f}, "
                    f"First10={comp['KernelSHAP_First10_Mass']:.6f}, "
                    f"Ratio={comp['KernelSHAP_Last10_to_First10']:.6f}"
                )
                print(
                    "Peak block: "
                    f"IG={comp['IG_Peak_Block']}, "
                    f"KernelSHAP={comp['KernelSHAP_Peak_Block']}, "
                    f"Consistent={comp['Same_Peak_Block']}"
                )

                print("6×10-day block quality:")
                print(
                    block[
                        [
                            "Block",
                            "Timestep_Start",
                            "Timestep_End",
                            "IG_Mass",
                            "KernelSHAP_Mass",
                        ]
                    ].to_string(
                        index=False
                    )
                )

                print("Temporal Method comparison: complete")

            except Exception as e:
                print(
                    f"Error: method comparison failed: "
                    f"{type(e).__name__}: {e}"
                )
                errors.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Compare",
                    "Error": (
                        f"{type(e).__name__}: {e}"
                    ),
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("Step 6: save results and create the final summary")

    def cat_or_empty(items):
        return (
            pd.concat(
                items,
                ignore_index=True
            )
            if items else pd.DataFrame()
        )

    df_model = pd.DataFrame(
        all_model_check
    )
    df_shap_detail = cat_or_empty(
        all_shap_detail
    )
    df_shap_time = cat_or_empty(
        all_shap_time
    )
    df_ig_time = cat_or_empty(
        all_ig_time
    )
    df_merged = cat_or_empty(
        all_merged_time
    )
    df_block = cat_or_empty(
        all_blocks
    )
    df_compare = pd.DataFrame(
        all_compare
    )
    df_errors = pd.DataFrame(
        errors
    )

    df_model.to_csv(
        OUTPUT_DIR / "F0_model_reproduction_check.csv",
        index=False,
    )
    df_shap_detail.to_csv(
        OUTPUT_DIR / "F1_temporal_kernelshap_detail.csv",
        index=False,
    )
    df_shap_time.to_csv(
        OUTPUT_DIR / "F1_temporal_kernelshap_timestep.csv",
        index=False,
    )
    df_ig_time.to_csv(
        OUTPUT_DIR / "F2_ig_same_samples_timestep.csv",
        index=False,
    )
    df_merged.to_csv(
        OUTPUT_DIR / "F3_temporal_method_timestep_comparison.csv",
        index=False,
    )
    df_block.to_csv(
        OUTPUT_DIR / "F4_temporal_block_mass.csv",
        index=False,
    )
    df_compare.to_csv(
        OUTPUT_DIR / "F5_temporal_method_summary.csv",
        index=False,
    )

    if not df_errors.empty:
        df_errors.to_csv(
            OUTPUT_DIR / "F9_errors.csv",
            index=False,
        )

    print(
        f"Rows of model check: {len(df_model)}"
    )
    print(
        f"Temporal KernelSHAP Number of detail rows: "
        f"{len(df_shap_detail)}"
    )
    print(
        f"Temporal KernelSHAP timestep Rows: "
        f"{len(df_shap_time)}"
    )
    print(
        f"IG timestep Rows: {len(df_ig_time)}"
    )
    print(
        f"Block mass Rows: {len(df_block)}"
    )
    print(
        f"Rows of method comparison: {len(df_compare)}"
    )
    print(
        f"Number of abnormal records: {len(df_errors)}"
    )

    print("-" * 20)
    print(
        "Final summary 1: IG vs Temporal KernelSHAP"
    )

    if not df_compare.empty:
        show_cols = [
            "Dataset",
            "Model",
            "Timestep_Spearman",
            "Timestep_p_value",
            "Block_Spearman",
            "Block_p_value",
            "IG_Last10_Mass",
            "KernelSHAP_Last10_Mass",
            "IG_Last10_to_First10",
            "KernelSHAP_Last10_to_First10",
            "IG_Peak_Block",
            "KernelSHAP_Peak_Block",
            "Same_Peak_Block",
        ]

        print(
            df_compare[
                show_cols
            ].sort_values(
                ["Dataset", "Model"]
            ).to_string(
                index=False
            )
        )

        rho = df_compare[
            "Timestep_Spearman"
        ]
        print(
            "60-timestep rho Range: "
            f"{rho.min():.6f} - "
            f"{rho.max():.6f}"
        )

        sig_pos = (
            (df_compare["Timestep_Spearman"] > 0)
            & (
                df_compare[
                    "Timestep_p_value"
                ] < 0.05
            )
        ).sum()
        print(
            "60-timestep positively correlated and p<0.05: "
            f"{int(sig_pos)}/{len(df_compare)}"
        )

        shap_recency = (
            df_compare[
                "KernelSHAP_Last10_Mass"
            ]
            >
            df_compare[
                "KernelSHAP_First10_Mass"
            ]
        ).sum()
        print(
            "KernelSHAP Last10 > First10: "
            f"{int(shap_recency)}/"
            f"{len(df_compare)}"
        )

        same_peak = df_compare[
            "Same_Peak_Block"
        ].sum()
        print(
            "IG matches the KernelSHAP peak block: "
            f"{int(same_peak)}/"
            f"{len(df_compare)}"
        )

        shap_peak6 = (
            df_compare[
                "KernelSHAP_Peak_Block"
            ] == 6
        ).sum()
        print(
            "KernelSHAP peak at days 51-60: "
            f"{int(shap_peak6)}/"
            f"{len(df_compare)}"
        )

    print("-" * 20)
    print("Final summary 2: 6×10 day block mass")

    if not df_block.empty:
        print(
            df_block[
                [
                    "Dataset",
                    "Model",
                    "Block",
                    "Timestep_Start",
                    "Timestep_End",
                    "IG_Mass",
                    "KernelSHAP_Mass",
                ]
            ].sort_values(
                ["Dataset", "Model", "Block"]
            ).to_string(
                index=False
            )
        )

    print("-" * 20)
    print("Final summary 3: errors")

    if df_errors.empty:
        print(
            "Errors: 0, all steps completed"
        )
    else:
        print(
            df_errors.to_string(
                index=False
            )
        )

    print("-" * 20)
    print("xai_cell_F_V1 Done")
    print(
        f"Results folder: "
        f"{OUTPUT_DIR.as_posix()}/"
    )


if __name__ == "__main__":
    main()
