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


def summarize_ig(attr, feature_names):
    timestep = attr.mean(axis=(0, 2))
    feature = attr.mean(axis=(0, 1))
    time_share = timestep / (timestep.sum() + 1e-12)
    feat_share = feature / (feature.sum() + 1e-12)
    df_time = pd.DataFrame({"Timestep": np.arange(1, 61), "IG_Importance": timestep, "IG_Share": time_share})
    df_feat = pd.DataFrame({"Feature": feature_names, "IG_Importance": feature, "IG_Share": feat_share})
    return df_time, df_feat




# =========================
# Cell D Dedicated configuration
# =========================

OUTPUT_DIR = Path("./0924/xai_cell_D_V1")
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [0.005, 0.01, 0.02]   # 0.5%, 1%, 2% of training feature SD
NOISE_REPEATS = 3
NOISE_IG_SAMPLES = 200
NOISE_IG_STEPS = 32

# =========================
# 6. Noise robustness function
# =========================

def ig_attribution_fixed_targets(model, X_input, X_train, sample_idx, fixed_targets,
                                 n_steps=32, batch_size=64):
    """
    Compute IG with fixed targets:
    - clean/noisy all use the classes predicted by the original model on clean samples;
    - reference fixed to the training-set mean trajectory;
    - so that target changes do not enter the attribution stability comparison.
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
            xb = torch.from_numpy(X_input[idx].astype(np.float32)).to(DEVICE)
            targets = torch.from_numpy(
                fixed_targets[s:s + len(idx)].astype(np.int64)
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

            del xb, base, attr, targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(out, axis=0)

    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()


def get_feature_sd(X_train):
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0, ddof=0).astype(np.float32)
    sd[~np.isfinite(sd)] = 0.0
    sd[sd < 1e-12] = 0.0
    return sd


def add_feature_scaled_noise(X, feature_sd, level, seed):
    rng = np.random.default_rng(seed)
    scale = feature_sd.reshape(1, 1, -1) * float(level)
    noise = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32) * scale
    return (X.astype(np.float32) + noise).astype(np.float32)


def spearman_safe(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def summarize_attr_arrays(attr):
    """
    Returns:
    - sample-averaged T×F map
    - feature aggregate F
    - temporal aggregate T
    """
    map_tf = attr.mean(axis=0)
    feat = attr.mean(axis=(0, 1))
    time = attr.mean(axis=(0, 2))
    return map_tf, feat, time


def prediction_noise_metrics(model, X_test, y_test, feature_sd, level, repeat):
    seed = SEED + int(level * 100000) + repeat * 1009
    X_noisy = add_feature_scaled_noise(X_test, feature_sd, level, seed)

    clean = evaluate_arrays(model, X_test, y_test)
    noisy = evaluate_arrays(model, X_noisy, y_test)

    agreement = np.mean(clean["pred"] == noisy["pred"])

    return {
        "Pred_Agreement": float(agreement),
        "Clean_Accuracy": float(clean["accuracy"]),
        "Noisy_Accuracy": float(noisy["accuracy"]),
        "Delta_Accuracy": float(noisy["accuracy"] - clean["accuracy"]),
        "Clean_MacroF1": float(clean["macro_f1"]),
        "Noisy_MacroF1": float(noisy["macro_f1"]),
        "Delta_MacroF1": float(noisy["macro_f1"] - clean["macro_f1"]),
        "Clean_TrueProb": float(clean["mean_true_class_prob"]),
        "Noisy_TrueProb": float(noisy["mean_true_class_prob"]),
        "Delta_TrueProb": float(noisy["mean_true_class_prob"] - clean["mean_true_class_prob"]),
    }


def run_noise_robustness(dataset_name, model_type, model, X_train, X_test, y_test, feature_names):
    print("Step 3: build fixed samples and the clean IG reference")

    sample_idx = choose_fixed_samples(X_test, NOISE_IG_SAMPLES, SEED)
    clean_pred, _ = predict_arrays(model, X_test[sample_idx], EVAL_BATCH_SIZE)
    fixed_targets = clean_pred.astype(np.int64)

    print(f"IG Number of fixed samples: {len(sample_idx)}")
    print(f"IG steps: {NOISE_IG_STEPS}")
    print(f"Fixed target distribution: {pd.Series(fixed_targets).value_counts().sort_index().to_dict()}")

    clean_attr = ig_attribution_fixed_targets(
        model, X_test, X_train, sample_idx, fixed_targets,
        n_steps=NOISE_IG_STEPS
    )

    clean_map, clean_feat, clean_time = summarize_attr_arrays(clean_attr)

    print(f"Clean attribution shape: {clean_attr.shape}")
    print(f"Clean attribution Missing values: {int(np.isnan(clean_attr).sum())}")
    print("Clean IG Features Top5:")
    top_idx = np.argsort(-clean_feat)[:5]
    for rank, j in enumerate(top_idx, start=1):
        print(f"{rank}. {feature_names[j]} | importance={clean_feat[j]:.8f}")

    feature_sd = get_feature_sd(X_train)
    zero_sd = [feature_names[i] for i, x in enumerate(feature_sd) if x == 0]
    print(f"Minimum SD of training features: {feature_sd.min():.8f}, Maximum: {feature_sd.max():.8f}")
    print(f"Number of zero-variance features: {len(zero_sd)}")
    if zero_sd:
        print(f"Zero-variance features: {', '.join(zero_sd)}")

    rows = []

    for level in NOISE_LEVELS:
        print("-" * 20)
        print(f"Step 4: input noise {level*100:.1f}% SD")

        for repeat in range(1, NOISE_REPEATS + 1):
            seed = SEED + int(level * 100000) + repeat * 1009
            print(f"Repeat {repeat}/{NOISE_REPEATS} | seed={seed}")

            try:
                X_noisy = add_feature_scaled_noise(X_test, feature_sd, level, seed)

                # Model prediction stability: full test set
                clean_eval = evaluate_arrays(model, X_test, y_test)
                noisy_eval = evaluate_arrays(model, X_noisy, y_test)
                pred_agreement = np.mean(clean_eval["pred"] == noisy_eval["pred"])

                # XAI Stability: 200 fixed samples, fixed original target
                noisy_attr = ig_attribution_fixed_targets(
                    model, X_noisy, X_train, sample_idx, fixed_targets,
                    n_steps=NOISE_IG_STEPS
                )
                noisy_map, noisy_feat, noisy_time = summarize_attr_arrays(noisy_attr)

                map_rho, map_p = spearman_safe(clean_map, noisy_map)
                feat_rho, feat_p = spearman_safe(clean_feat, noisy_feat)
                time_rho, time_p = spearman_safe(clean_time, noisy_time)

                mean_abs_noise = np.mean(np.abs(X_noisy - X_test))
                max_abs_noise = np.max(np.abs(X_noisy - X_test))

                row = {
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Noise_Level_SD": level,
                    "Noise_Percent_SD": level * 100,
                    "Repeat": repeat,
                    "N_IG_Samples": len(sample_idx),
                    "IG_Steps": NOISE_IG_STEPS,
                    "Map_Spearman": map_rho,
                    "Map_p": map_p,
                    "Feature_Spearman": feat_rho,
                    "Feature_p": feat_p,
                    "Temporal_Spearman": time_rho,
                    "Temporal_p": time_p,
                    "Pred_Agreement": float(pred_agreement),
                    "Clean_Accuracy": float(clean_eval["accuracy"]),
                    "Noisy_Accuracy": float(noisy_eval["accuracy"]),
                    "Delta_Accuracy": float(noisy_eval["accuracy"] - clean_eval["accuracy"]),
                    "Clean_MacroF1": float(clean_eval["macro_f1"]),
                    "Noisy_MacroF1": float(noisy_eval["macro_f1"]),
                    "Delta_MacroF1": float(noisy_eval["macro_f1"] - clean_eval["macro_f1"]),
                    "Clean_TrueProb": float(clean_eval["mean_true_class_prob"]),
                    "Noisy_TrueProb": float(noisy_eval["mean_true_class_prob"]),
                    "Delta_TrueProb": float(noisy_eval["mean_true_class_prob"] - clean_eval["mean_true_class_prob"]),
                    "Mean_Abs_Noise": float(mean_abs_noise),
                    "Max_Abs_Noise": float(max_abs_noise),
                }
                rows.append(row)

                print(
                    f"Results: Mapρ={map_rho:.6f}, Featureρ={feat_rho:.6f}, "
                    f"Temporalρ={time_rho:.6f}, PredAgreement={pred_agreement:.6f}, "
                    f"ΔF1={row['Delta_MacroF1']:.6f}, ΔAcc={row['Delta_Accuracy']:.6f}"
                )

            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


# =========================
# 7. Main flow
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Task: model and IG robustness under small input noise")
    print("Noise: 0.5%, 1%, 2% × training feature SD")
    print(f"Repeats per noise level: {NOISE_REPEATS}")
    print("XAI Comparison: sample-averaged T×F map / feature ranking / temporal ranking")
    print("Prediction comparison: prediction agreement / Macro-F1 / Accuracy / true-class probability")

    all_results = []
    all_checks = []
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
            print(f"Training samples: {len(ds['X_train'])}, Validation samples: {len(ds['X_val'])}, Test samples: {len(ds['X_test'])}")
            print(f"Training missing values: {int(np.isnan(ds['X_train']).sum())}, Validation missing values: {int(np.isnan(ds['X_val']).sum())}, Test missing values: {int(np.isnan(ds['X_test']).sum())}")
            print(f"Training label distribution: {pd.Series(ds['y_train']).value_counts().sort_index().to_dict()}")
            print(f"Test label distribution: {pd.Series(ds['y_test']).value_counts().sort_index().to_dict()}")
            print("Data audit: passed")
        except Exception as e:
            print(f"Error: data preparation failed: {type(e).__name__}: {e}")
            print("Data audit: failed")
            error_rows.append({
                "Dataset": dataset_name, "Model": "", "Step": "Data preparation",
                "Error": f"{type(e).__name__}: {e}"
            })
            continue

        for model_type in MODELS:
            print("-" * 20)
            print(f"{dataset_name} / {model_type}")
            print("Step 2: model loading and original model check")

            try:
                model, _ = load_model(dataset_name, model_type, cfg)
                base = evaluate_arrays(model, ds["X_test"], ds["y_test"])

                print(f"Test samples: {len(ds['y_test'])}")
                print(f"Original model check: Macro-F1={base['macro_f1']:.6f}, Accuracy={base['accuracy']:.6f}, TrueProb={base['mean_true_class_prob']:.6f}")
                print("Model check: passed")

                all_checks.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Test_N": len(ds["y_test"]),
                    "Accuracy": base["accuracy"],
                    "Macro_F1": base["macro_f1"],
                    "Mean_True_Class_Prob": base["mean_true_class_prob"],
                })

            except Exception as e:
                print(f"Error: model loading/check failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name, "Model": model_type, "Step": "Model check",
                    "Error": f"{type(e).__name__}: {e}"
                })
                continue

            try:
                result = run_noise_robustness(
                    dataset_name, model_type, model,
                    ds["X_train"], ds["X_test"], ds["y_test"], cfg["features"]
                )
                if not result.empty:
                    all_results.append(result)
                    print(f"Rows of noise results for the current model: {len(result)}")
                    print("Current model summary:")
                    current_summary = result.groupby(
                        ["Noise_Percent_SD"], as_index=False
                    ).agg(
                        Map_Spearman_Mean=("Map_Spearman", "mean"),
                        Feature_Spearman_Mean=("Feature_Spearman", "mean"),
                        Temporal_Spearman_Mean=("Temporal_Spearman", "mean"),
                        Pred_Agreement_Mean=("Pred_Agreement", "mean"),
                        Delta_MacroF1_Mean=("Delta_MacroF1", "mean"),
                        Delta_Accuracy_Mean=("Delta_Accuracy", "mean"),
                    )
                    print(current_summary.to_string(index=False))
                else:
                    print("Error: the current model produced no noise results")
                    error_rows.append({
                        "Dataset": dataset_name, "Model": model_type, "Step": "Noise robustness",
                        "Error": "No results produced"
                    })
            except Exception as e:
                print(f"Error: noise robustness failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name, "Model": model_type, "Step": "Noise robustness",
                    "Error": f"{type(e).__name__}: {e}"
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("Step 5: save results and create the final summary")

    df_checks = pd.DataFrame(all_checks)
    df_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    df_errors = pd.DataFrame(error_rows)

    df_checks.to_csv(OUTPUT_DIR / "D0_model_reproduction_check.csv", index=False)
    if not df_results.empty:
        df_results.to_csv(OUTPUT_DIR / "D1_input_noise_robustness_detail.csv", index=False)
    if not df_errors.empty:
        df_errors.to_csv(OUTPUT_DIR / "D9_errors.csv", index=False)

    if not df_results.empty:
        df_summary = df_results.groupby(
            ["Dataset", "Model", "Noise_Percent_SD"], as_index=False
        ).agg(
            N_Repeats=("Repeat", "count"),
            Map_Spearman_Mean=("Map_Spearman", "mean"),
            Map_Spearman_SD=("Map_Spearman", "std"),
            Feature_Spearman_Mean=("Feature_Spearman", "mean"),
            Feature_Spearman_SD=("Feature_Spearman", "std"),
            Temporal_Spearman_Mean=("Temporal_Spearman", "mean"),
            Temporal_Spearman_SD=("Temporal_Spearman", "std"),
            Pred_Agreement_Mean=("Pred_Agreement", "mean"),
            Pred_Agreement_SD=("Pred_Agreement", "std"),
            Delta_MacroF1_Mean=("Delta_MacroF1", "mean"),
            Delta_MacroF1_SD=("Delta_MacroF1", "std"),
            Delta_Accuracy_Mean=("Delta_Accuracy", "mean"),
            Delta_Accuracy_SD=("Delta_Accuracy", "std"),
            Delta_TrueProb_Mean=("Delta_TrueProb", "mean"),
            Delta_TrueProb_SD=("Delta_TrueProb", "std"),
        )
        df_summary.to_csv(OUTPUT_DIR / "D2_input_noise_robustness_summary.csv", index=False)
    else:
        df_summary = pd.DataFrame()

    print(f"Rows of model check: {len(df_checks)}")
    print(f"Rows of detailed noise results: {len(df_results)}")
    print(f"Rows of summarized noise results: {len(df_summary)}")
    print(f"Number of abnormal records: {len(df_errors)}")

    print("-" * 20)
    print("Final summary 1: original model check")
    if not df_checks.empty:
        print(df_checks.sort_values(["Dataset", "Model"]).to_string(index=False))
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 2: input noise robustness")
    if not df_summary.empty:
        show_cols = [
            "Dataset", "Model", "Noise_Percent_SD",
            "Map_Spearman_Mean", "Feature_Spearman_Mean",
            "Temporal_Spearman_Mean", "Pred_Agreement_Mean",
            "Delta_MacroF1_Mean", "Delta_Accuracy_Mean",
            "Delta_TrueProb_Mean"
        ]
        print(df_summary[show_cols].sort_values(
            ["Dataset", "Model", "Noise_Percent_SD"]
        ).to_string(index=False))
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 3: overall cross-model statistics for each noise level")
    if not df_results.empty:
        overall = df_results.groupby("Noise_Percent_SD", as_index=False).agg(
            N=("Feature_Spearman", "count"),
            Map_Spearman_Mean=("Map_Spearman", "mean"),
            Map_Spearman_Min=("Map_Spearman", "min"),
            Feature_Spearman_Mean=("Feature_Spearman", "mean"),
            Feature_Spearman_Min=("Feature_Spearman", "min"),
            Temporal_Spearman_Mean=("Temporal_Spearman", "mean"),
            Temporal_Spearman_Min=("Temporal_Spearman", "min"),
            Pred_Agreement_Mean=("Pred_Agreement", "mean"),
            Pred_Agreement_Min=("Pred_Agreement", "min"),
            Delta_MacroF1_Mean=("Delta_MacroF1", "mean"),
            Delta_Accuracy_Mean=("Delta_Accuracy", "mean"),
        )
        print(overall.to_string(index=False))
        overall.to_csv(OUTPUT_DIR / "D3_input_noise_overall_summary.csv", index=False)
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 4: errors")
    if df_errors.empty:
        print("Errors: 0, all steps completed")
    else:
        print(df_errors.to_string(index=False))

    print("-" * 20)
    print("xai_cell_D_V1 Done")
    print(f"Results folder: {OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
