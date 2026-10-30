# -*- coding: utf-8 -*-
"""
xai_cell_A_V9

Goals:
1. Original model and data audit;
2. Corrected cascading parameter-randomization sanity check;
3. IG feature / timestep attribution;
4. IG recency summary;
5. 6×10-day temporal occlusion as an independent check of recency.

Principles:
- No retraining; only the existing seed=42 models are read;
- IG baseline Training set only 60×F mean trajectory;
- At each parameter-randomization stage, the original model's predicted target is kept fixed; true cascading;
- temporal occlusion Replace with the time block at the same position of a real training window;
- temporal occlusion The primary metric is the original-predicted-class probability drop;
- No folder scanning; no dependence on results of deleted old versions.
"""

import copy
import math
import random
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

SEED = 42
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("./0924/xai_cell_A_V9")
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IG_TIMESTEP_SAMPLES = 500
IG_SANITY_SAMPLES = 200
IG_STEPS_TIMESTEP = 64
IG_STEPS_SANITY = 32
IG_BATCH_SIZE = 64
OCCLUSION_REPEATS = 5
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


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def ig_attribution(model, X_test, X_train, sample_idx, n_steps=32, batch_size=64):
    """
    IG attribution.
    PyTorch cuDNN RNN does not allow backward in eval mode, therefore:
    1) Temporarily disable cuDNN;
    2) Switch the RNN to train mode;
    3) Keep Dropout in eval mode separately to avoid randomness;
    4) Restore cuDNN and model.eval() afterwards.
    """
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    try:
        ig = IntegratedGradients(model)

        # Common reference: training set 60×F Mean trajectory
        baseline = torch.from_numpy(
            X_train.mean(axis=0, keepdims=True).astype(np.float32)
        ).to(DEVICE)

        out = []

        for s in range(0, len(sample_idx), batch_size):
            idx = sample_idx[s:s + batch_size]
            xb = torch.from_numpy(
                X_test[idx].astype(np.float32)
            ).to(DEVICE)

            # predicted class as the attribution target
            with torch.no_grad():
                targets = model(xb).argmax(dim=1)

            base = baseline.expand(xb.size(0), -1, -1)

            attr = ig.attribute(
                xb,
                baselines=base,
                target=targets,
                n_steps=n_steps,
                internal_batch_size=min(256, xb.size(0) * 4),
            )

            out.append(attr.detach().abs().cpu().numpy())

            del xb, base, attr, targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(out, axis=0)

    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()



def ig_attribution_fixed_targets(model, X_test, X_train, sample_idx, fixed_targets,
                                 n_steps=32, batch_size=64):
    """The fixed predicted class of the original model is used as the IG target at all randomization stages."""
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
            xb = torch.from_numpy(X_test[idx].astype(np.float32)).to(DEVICE)
            targets = torch.from_numpy(
                np.asarray(fixed_targets[s:s + len(idx)], dtype=np.int64)
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


def summarize_ig(attr, feature_names):
    timestep = attr.mean(axis=(0, 2))
    feature = attr.mean(axis=(0, 1))
    time_share = timestep / (timestep.sum() + 1e-12)
    feat_share = feature / (feature.sum() + 1e-12)
    df_time = pd.DataFrame({"Timestep": np.arange(1, 61), "IG_Importance": timestep, "IG_Share": time_share})
    df_feat = pd.DataFrame({"Feature": feature_names, "IG_Importance": feature, "IG_Share": feat_share})
    return df_time, df_feat


def randomize_recurrent_layer_(model, model_type, layer_idx):
    module = model.gru if model_type == "GRU" else model.lstm
    bound = 1.0 / math.sqrt(module.hidden_size)
    names = [f"weight_ih_l{layer_idx}", f"weight_hh_l{layer_idx}",
             f"bias_ih_l{layer_idx}", f"bias_hh_l{layer_idx}"]
    if isinstance(module, nn.LSTM) and module.bidirectional:
        names += [f"weight_ih_l{layer_idx}_reverse", f"weight_hh_l{layer_idx}_reverse",
                  f"bias_ih_l{layer_idx}_reverse", f"bias_hh_l{layer_idx}_reverse"]
    with torch.no_grad():
        for name in names:
            if hasattr(module, name):
                getattr(module, name).uniform_(-bound, bound)


def randomize_fc_(model):
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()


def randomize_all_(model):
    randomize_fc_(model)
    module = model.gru if hasattr(model, "gru") else model.lstm
    bound = 1.0 / math.sqrt(module.hidden_size)
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(-bound, bound)


def parameter_randomization_sanity(dataset_name, model_type, model, X_train, X_test, feature_names):
    """
    Corrected version:
    1. fixed predicted targets of the original model;
    2. true cascading: Original -> FC -> FC+RNN_top -> ... -> FC+all RNN layers;
    3. Also report T×F map and feature aggregate Spearman;
    4. Each stage randomizes only one more layer and does not resample layers already randomized.
    """
    idx = choose_fixed_samples(X_test, IG_SANITY_SAMPLES, SEED)

    # Fixed predicted class of the original model
    fixed_targets, _ = predict_arrays(model, X_test[idx], EVAL_BATCH_SIZE)
    fixed_targets = np.asarray(fixed_targets, dtype=np.int64)

    orig_attr = ig_attribution_fixed_targets(
        model, X_test, X_train, idx, fixed_targets,
        IG_STEPS_SANITY, IG_BATCH_SIZE
    )
    orig_map = orig_attr.mean(axis=0)  # T×F
    orig_feat = orig_attr.mean(axis=(0, 1))

    n_layers = model.gru.num_layers if model_type == "GRU" else model.lstm.num_layers

    # True cascading
    stages = [("Original", copy.deepcopy(model))]
    current = copy.deepcopy(model)

    randomize_fc_(current)
    stages.append(("Randomize_FC", copy.deepcopy(current)))

    # Add randomization layer by layer from the top to the bottom
    for layer_idx in range(n_layers - 1, -1, -1):
        randomize_recurrent_layer_(current, model_type, layer_idx)
        stage_name = f"Randomize_FC_RNN_L{layer_idx}_to_L{n_layers-1}"
        stages.append((stage_name, copy.deepcopy(current)))

    rows = []
    for stage, m in stages:
        m = m.to(DEVICE)

        if stage == "Original":
            attr = orig_attr
            map_rho, map_p = 1.0, 0.0
            feat_rho, feat_p = 1.0, 0.0
        else:
            attr = ig_attribution_fixed_targets(
                m, X_test, X_train, idx, fixed_targets,
                IG_STEPS_SANITY, IG_BATCH_SIZE
            )
            cur_map = attr.mean(axis=0)
            cur_feat = attr.mean(axis=(0, 1))
            map_rho, map_p = spearmanr(orig_map.ravel(), cur_map.ravel())
            feat_rho, feat_p = spearmanr(orig_feat, cur_feat)

        rows.append({
            "Dataset": dataset_name,
            "Model": model_type,
            "Stage": stage,
            "Map_Spearman_vs_Original": map_rho,
            "Map_p_value": map_p,
            "Feature_Spearman_vs_Original": feat_rho,
            "Feature_p_value": feat_p,
            "N_IG_Samples": len(idx),
            "IG_Steps": IG_STEPS_SANITY,
            "Fixed_Target_0": int(np.sum(fixed_targets == 0)),
            "Fixed_Target_1": int(np.sum(fixed_targets == 1)),
            "Fixed_Target_2": int(np.sum(fixed_targets == 2)),
        })

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def run_ig_timestep(dataset_name, model_type, model, X_train, X_test, feature_names):
    idx = choose_fixed_samples(X_test, IG_TIMESTEP_SAMPLES, SEED)
    attr = ig_attribution(model, X_test, X_train, idx, IG_STEPS_TIMESTEP, IG_BATCH_SIZE)
    df_time, df_feat = summarize_ig(attr, feature_names)
    df_time.insert(0, "Model", model_type); df_time.insert(0, "Dataset", dataset_name)
    df_feat.insert(0, "Model", model_type); df_feat.insert(0, "Dataset", dataset_name)
    last10 = df_time.loc[df_time["Timestep"].between(51, 60), "IG_Share"].sum()
    first10 = df_time.loc[df_time["Timestep"].between(1, 10), "IG_Share"].sum()
    summary = pd.DataFrame([{"Dataset": dataset_name, "Model": model_type,
                             "IG_Last10_Mass": last10, "IG_First10_Mass": first10,
                             "IG_Last10_to_First10": last10 / max(first10, 1e-12),
                             "N_IG_Samples": len(idx), "IG_Steps": IG_STEPS_TIMESTEP}])
    return df_time, df_feat, summary


def temporal_occlusion(dataset_name, model_type, model, X_train, X_test, y_test):
    base = evaluate_arrays(model, X_test, y_test)
    base_pred = base["pred"]
    base_prob = base["prob"]
    base_orig_pred_prob = base_prob[np.arange(len(y_test)), base_pred].mean()
    rng = np.random.default_rng(SEED)
    rows = []

    for b in range(6):
        t0, t1 = b * 10, b * 10 + 10
        reps = []
        for rep in range(OCCLUSION_REPEATS):
            bg_idx = rng.integers(0, len(X_train), size=len(X_test))
            X_occ = X_test.copy()
            X_occ[:, t0:t1, :] = X_train[bg_idx, t0:t1, :]
            pred, prob = predict_arrays(model, X_occ, EVAL_BATCH_SIZE)
            reps.append({
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "true_prob": prob[np.arange(len(y_test)), y_test].mean(),
                "orig_pred_prob": prob[np.arange(len(y_test)), base_pred].mean(),
            })
        tmp = pd.DataFrame(reps)
        rows.append({
            "Dataset": dataset_name, "Model": model_type, "Block": b+1,
            "Timestep_Start": t0+1, "Timestep_End": t1,
            "Baseline_Accuracy": base["accuracy"], "Occluded_Accuracy": tmp["accuracy"].mean(),
            "Delta_Accuracy": base["accuracy"] - tmp["accuracy"].mean(),
            "Baseline_MacroF1": base["macro_f1"], "Occluded_MacroF1": tmp["macro_f1"].mean(),
            "Delta_MacroF1": base["macro_f1"] - tmp["macro_f1"].mean(),
            "Baseline_TrueProb": base["mean_true_class_prob"], "Occluded_TrueProb": tmp["true_prob"].mean(),
            "Delta_TrueProb": base["mean_true_class_prob"] - tmp["true_prob"].mean(),
            "Baseline_OriginalPredProb": base_orig_pred_prob,
            "Occluded_OriginalPredProb": tmp["orig_pred_prob"].mean(),
            "Delta_OriginalPredProb": base_orig_pred_prob - tmp["orig_pred_prob"].mean(),
            "Repeat_SD_MacroF1": tmp["macro_f1"].std(ddof=1),
            "Repeat_SD_TrueProb": tmp["true_prob"].std(ddof=1),
            "Occlusion_Repeats": OCCLUSION_REPEATS,
        })
    return pd.DataFrame(rows)


def main():
    print(f"Device: {DEVICE}")
    print("Task: A pipeline, complete final version")
    print("Parameter randomization: fixed target + true cascading")
    print("IG baseline: training-set mean trajectory")
    print("Temporal occlusion Primary metric: original-predicted-class probability drop")

    all_checks = []
    all_sanity = []
    all_time = []
    all_feat = []
    all_recency = []
    all_occ = []
    error_rows = []

    for dataset_name in RUN_DATASETS:
        cfg = DATASETS[dataset_name]

        print("-" * 20)
        print(f"Dataset: {dataset_name}")
        print("Step 1: data preparation and audit")

        try:
            ds = prepare_dataset(dataset_name, cfg)
            print(f"Number of fields: {len(cfg['features'])}")
            print(f"Field: {', '.join(cfg['features'])}")
            print(
                f"Training samples: {len(ds['X_train'])}, "
                f"Validation samples: {len(ds['X_val'])}, "
                f"Test samples: {len(ds['X_test'])}"
            )
            print(
                f"Training missing values: {int(np.isnan(ds['X_train']).sum())}, "
                f"Validation missing values: {int(np.isnan(ds['X_val']).sum())}, "
                f"Test missing values: {int(np.isnan(ds['X_test']).sum())}"
            )
            print("Data audit: passed")
        except Exception as e:
            print(f"Error: data preparation failed: {type(e).__name__}: {e}")
            error_rows.append({
                "Dataset": dataset_name,
                "Model": "",
                "Step": "Data preparation",
                "Error": f"{type(e).__name__}: {e}",
            })
            continue

        for model_type in MODELS:
            print("-" * 20)
            print(f"{dataset_name} / {model_type}")

            print("Step 2: model loading and original model check")
            try:
                model, _ = load_model(dataset_name, model_type, cfg)
                base = evaluate_arrays(
                    model, ds["X_test"], ds["y_test"]
                )
                n_layers = (
                    model.gru.num_layers
                    if model_type == "GRU"
                    else model.lstm.num_layers
                )
                print(f"RNN Number of layers: {n_layers}")
                print(
                    f"Original model check: Macro-F1={base['macro_f1']:.6f}, "
                    f"Accuracy={base['accuracy']:.6f}"
                )
                print("Model check: passed")
                all_checks.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Macro_F1": base["macro_f1"],
                    "Accuracy": base["accuracy"],
                    "Test_N": len(ds["y_test"]),
                    "RNN_Layers": n_layers,
                })
            except Exception as e:
                print(f"Error: model loading failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Model loading",
                    "Error": f"{type(e).__name__}: {e}",
                })
                continue

            print("Step 3: cascading parameter randomization")
            try:
                sanity = parameter_randomization_sanity(
                    dataset_name, model_type, model,
                    ds["X_train"], ds["X_test"], cfg["features"]
                )
                all_sanity.append(sanity)
                print(f"Number of result rows: {len(sanity)}")
                print(
                    sanity[[
                        "Stage",
                        "Map_Spearman_vs_Original",
                        "Feature_Spearman_vs_Original",
                        "Fixed_Target_0",
                        "Fixed_Target_1",
                        "Fixed_Target_2",
                    ]].to_string(index=False)
                )
                print("Parameter randomization: complete")
            except Exception as e:
                print(f"Error: parameter randomization failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Parameter randomization",
                    "Error": f"{type(e).__name__}: {e}",
                })

            print("Step 4: IG feature/timestep attribution")
            try:
                df_time, df_feat, df_sum = run_ig_timestep(
                    dataset_name, model_type, model,
                    ds["X_train"], ds["X_test"], cfg["features"]
                )
                all_time.append(df_time)
                all_feat.append(df_feat)
                all_recency.append(df_sum)

                print("IG Recency:")
                print(df_sum.to_string(index=False))
                print("IG Features Top5:")
                print(
                    df_feat.sort_values(
                        "IG_Importance", ascending=False
                    )[["Feature", "IG_Importance", "IG_Share"]]
                    .head(5)
                    .to_string(index=False)
                )
                print("IG Attribution: complete")
            except Exception as e:
                print(f"Error: IG attribution failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "IG Attribution",
                    "Error": f"{type(e).__name__}: {e}",
                })

            print("Step 5: 6×10 day temporal occlusion")
            try:
                occ = temporal_occlusion(
                    dataset_name, model_type, model,
                    ds["X_train"], ds["X_test"], ds["y_test"]
                )
                all_occ.append(occ)

                print(
                    occ[[
                        "Block", "Timestep_Start", "Timestep_End",
                        "Delta_OriginalPredProb",
                        "Delta_MacroF1", "Delta_TrueProb",
                    ]].to_string(index=False)
                )

                peak_i = occ["Delta_OriginalPredProb"].idxmax()
                peak = occ.loc[peak_i]
                print(
                    f"Peak time block of the primary metric: Block={int(peak['Block'])}, "
                    f"Timestep={int(peak['Timestep_Start'])}-"
                    f"{int(peak['Timestep_End'])}, "
                    f"Delta_OriginalPredProb="
                    f"{peak['Delta_OriginalPredProb']:.6f}"
                )
                print("Temporal occlusion: Done")
            except Exception as e:
                print(f"Error: temporal occlusion failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Temporal occlusion",
                    "Error": f"{type(e).__name__}: {e}",
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("Step 6: save results and create the final summary")

    def concat_or_empty(items):
        return (
            pd.concat(items, ignore_index=True)
            if items else pd.DataFrame()
        )

    df_checks = pd.DataFrame(all_checks)
    df_sanity = concat_or_empty(all_sanity)
    df_time = concat_or_empty(all_time)
    df_feat = concat_or_empty(all_feat)
    df_recency = concat_or_empty(all_recency)
    df_occ = concat_or_empty(all_occ)
    df_errors = pd.DataFrame(error_rows)

    if not df_checks.empty:
        df_checks.to_csv(
            OUTPUT_DIR / "A0_model_reproduction_check.csv",
            index=False
        )
    if not df_sanity.empty:
        df_sanity.to_csv(
            OUTPUT_DIR / "A1_parameter_randomization_sanity.csv",
            index=False
        )
    if not df_time.empty:
        df_time.to_csv(
            OUTPUT_DIR / "A2_ig_timestep_importance.csv",
            index=False
        )
    if not df_feat.empty:
        df_feat.to_csv(
            OUTPUT_DIR / "A2_ig_feature_importance.csv",
            index=False
        )
    if not df_recency.empty:
        df_recency.to_csv(
            OUTPUT_DIR / "A2_ig_recency_summary.csv",
            index=False
        )

    if not df_occ.empty:
        df_occ.to_csv(
            OUTPUT_DIR / "A3_temporal_occlusion.csv",
            index=False
        )
        peak_idx = (
            df_occ.groupby(["Dataset", "Model"])
            ["Delta_OriginalPredProb"]
            .idxmax()
        )
        df_peak = (
            df_occ.loc[peak_idx]
            .sort_values(["Dataset", "Model"])
            .reset_index(drop=True)
        )
        df_peak.to_csv(
            OUTPUT_DIR / "A3_temporal_occlusion_peak_blocks.csv",
            index=False
        )
    else:
        df_peak = pd.DataFrame()

    if not df_errors.empty:
        df_errors.to_csv(
            OUTPUT_DIR / "A9_errors.csv",
            index=False
        )

    print(f"Rows of model check: {len(df_checks)}")
    print(f"Sanity Number of result rows: {len(df_sanity)}")
    print(f"IG timestep Rows: {len(df_time)}")
    print(f"IG feature Rows: {len(df_feat)}")
    print(f"IG recency Rows: {len(df_recency)}")
    print(f"Temporal occlusion Rows: {len(df_occ)}")
    print(f"Number of abnormal records: {len(df_errors)}")

    print("-" * 20)
    print("Final summary 1: final stage of parameter randomization")
    if not df_sanity.empty:
        last = (
            df_sanity.groupby(
                ["Dataset", "Model"], sort=False
            )
            .tail(1)
        )
        print(
            last[[
                "Dataset", "Model", "Stage",
                "Map_Spearman_vs_Original",
                "Feature_Spearman_vs_Original",
            ]].to_string(index=False)
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 2: IG recency")
    if not df_recency.empty:
        print(
            df_recency.sort_values(
                ["Dataset", "Model"]
            ).to_string(index=False)
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 3: temporal occlusion peak")
    if not df_peak.empty:
        print(
            df_peak[[
                "Dataset", "Model", "Block",
                "Timestep_Start", "Timestep_End",
                "Delta_OriginalPredProb",
                "Delta_MacroF1", "Delta_TrueProb",
            ]].to_string(index=False)
        )
        n_last = int(
            (
                (df_peak["Timestep_Start"] == 51)
                & (df_peak["Timestep_End"] == 60)
            ).sum()
        )
        print(
            f"The primary metric peaks at days 51-60: "
            f"{n_last}/{len(df_peak)}"
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 4: errors")
    if df_errors.empty:
        print("Errors: 0, all steps completed")
    else:
        print(df_errors.to_string(index=False))

    print("-" * 20)
    print("xai_cell_A_V9 Done")
    print(f"Results folder: {OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
