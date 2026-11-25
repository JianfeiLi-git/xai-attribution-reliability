"""
stage6_cell_H_data_robustness_V4

Aims:
1. Under small Gaussian input noise (0.5%, 1% and 2% of the training-set feature standard deviation, 3 times per level),
   test the stability of model predictions and IG attribution structure;
2. IG uses 200 fixed test samples, with the target fixed to the original model's predicted class on clean samples;
3. Prediction stability is computed on the full test set.

Principles:
- No retraining; only the seed-42 models selected in stage6_cell_C are loaded;
- The IG baseline is the training-set mean trajectory only;
- No folder scanning; all paths are given explicitly.
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

IG_TIMESTEP_SAMPLES = 500
IG_SANITY_SAMPLES = 200
IG_STEPS_TIMESTEP = 64
IG_STEPS_SANITY = 32
IG_BATCH_SIZE = 64
OCCLUSION_REPEATS = 5
EVAL_BATCH_SIZE = 512


# ==============================================================================
# stage6_cell_H_data_robustness_V4
# New U.S. models: IG and prediction stability under small Gaussian input noise (0.5%, 1% and 2% of the training-set SD, 3 times each)
# Methods are identical to the original paper pipeline (parameters, sample sizes and steps unchanged); only the following configuration changes:
#   - Only the two U.S. datasets (US-Med, US-Pow) are run; data from stage6_cell_A, features from its feature_config.json
#   - seed-42 models are those selected in stage6_cell_C_data_training_V2 (*_seed42_selected.pth)
#   - cuDNN cannot be initialized on the server, so cuDNN is disabled throughout and PyTorch native CUDA RNN kernels are used
# V4 changes: for each model and noise level, check that all 3 repetitions succeeded; incomplete combinations are written to the error table,
#             their details are saved separately to D8_incomplete_repeats_detail.csv and excluded from the formal summary (the computation itself is unchanged)
# Field names and values written to CSV are in English so that they can be merged with the original result tables
# ==============================================================================
US_DATA_ROOT = "stage6/stage6_cell_A_data_preprocessing_V1"
US_MODEL_ROOT = "stage6/stage6_cell_C_data_training_V2"
OUTPUT_DIR = Path("stage6/stage6_cell_H_data_robustness_V4")
# Rule: the first two lines print the input and output locations (relative paths only)
print(f"Input directories: ./{US_DATA_ROOT}/, ./{US_MODEL_ROOT}/")
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")

import json as _json
torch.backends.cudnn.enabled = False
print("cuDNN disabled (cuDNN cannot be initialized on the server); using PyTorch native CUDA RNN kernels")

_US_KEYS = {"US-Med": "us_med", "US-Pow": "us_pow"}
_cfg_path = Path(US_DATA_ROOT) / "feature_config.json"
_need = [_cfg_path]
for _k in _US_KEYS.values():
    _need += [Path(US_DATA_ROOT) / _k / f"{_s}_{_xy}.csv" for _s in ["train", "val", "test"] for _xy in ["X", "y"]]
    _need += [Path(US_MODEL_ROOT) / _k / f"{_m}_seed42_selected.pth" for _m in ["gru", "bilstm", "lstm"]]
_missing = [p for p in _need if not p.exists()]
if _missing:
    print("The following input files do not exist; check US_DATA_ROOT / US_MODEL_ROOT:")
    for _p in _missing:
        print("  ./" + _p.as_posix())
    raise FileNotFoundError(f"{len(_missing)} input files are missing")
print(f"Input file check passed ({len(_need)} files)")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(_cfg_path, "r", encoding="utf-8") as _f:
    _feat_cfg = _json.load(_f)

DATA_DIR = {name: Path(US_DATA_ROOT) / key for name, key in _US_KEYS.items()}
MODEL_PATH = {
    (name, m): Path(US_MODEL_ROOT) / key / f"{m.lower()}_seed42_selected.pth"
    for name, key in _US_KEYS.items() for m in ["GRU", "BiLSTM", "LSTM"]
}
DATASETS = {}
for _name, _key in _US_KEYS.items():
    _feats = _feat_cfg[_key]["selected_features"] + _feat_cfg[_key]["temporal_features"]
    DATASETS[_name] = {"expected_features": len(_feats), "filter_train_2010": False, "features": _feats}
    print(f"{_name}: {len(_feats)} features: {_feats}")
RUN_DATASETS = ["US-Med", "US-Pow"]
# ============================== Computation code below ==============================





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
        raise ValueError(f"Missing features: {missing}")

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
            f"{dataset_name}: data files not found: ./{data_dir.as_posix()} | missing: {missing}"
        )
    return data_dir


def prepare_dataset(dataset_name, cfg):
    data_dir = get_data_dir(dataset_name)
    print(f"{dataset_name}: data directory ./{data_dir.as_posix()}")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)
    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)
    q33, q67 = calculate_thresholds(y_tr)
    X_train, y_train, _ = build_sequence_data_with_meta(X_tr, y_tr, cfg["features"], q33, q67, SEQ_LEN)
    X_val, y_val, _ = build_sequence_data_with_meta(X_va, y_va, cfg["features"], q33, q67, SEQ_LEN)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, cfg["features"], q33, q67, SEQ_LEN)
    print(f"{dataset_name}: train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
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
            f"{dataset_name}-{model_type}: model file not found: ./{model_path.as_posix()}"
        )
    return model_path


def load_model(dataset_name, model_type, cfg):
    path = get_model_path(dataset_name, model_type)
    state = torch.load(path, map_location="cpu")
    model, input_dim, hidden_dim, num_layers = infer_model_from_state_dict(model_type, state)
    model = model.to(DEVICE).eval()
    print(f"{dataset_name}-{model_type}: ./{path.as_posix()} | input dim {input_dim}, hidden dim {hidden_dim}, layers {num_layers}")
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
    PyTorch's cuDNN RNN does not allow backpropagation in evaluation mode, therefore:
    1) cuDNN is temporarily disabled;
    2) the RNN is switched to training mode;
    3) Dropout is kept in evaluation mode separately to avoid randomness (the within-RNN dropout of reconstructed models is 0);
    4) the cuDNN setting and model.eval() are restored afterwards.
    """
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    try:
        ig = IntegratedGradients(model)

        # Common baseline: 60xF mean trajectory of the training set
        baseline = torch.from_numpy(
            X_train.mean(axis=0, keepdims=True).astype(np.float32)
        ).to(DEVICE)

        out = []

        for s in range(0, len(sample_idx), batch_size):
            idx = sample_idx[s:s + batch_size]
            xb = torch.from_numpy(
                X_test[idx].astype(np.float32)
            ).to(DEVICE)

            # Use the predicted class as the attribution target
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
# Settings specific to this cell
# =========================


NOISE_LEVELS = [0.005, 0.01, 0.02]   # 0.5%, 1% and 2% of the training-set feature SD
NOISE_REPEATS = 3
NOISE_IG_SAMPLES = 200
NOISE_IG_STEPS = 32

# =========================
# 6. Noise-robustness functions
# =========================

def ig_attribution_fixed_targets(model, X_input, X_train, sample_idx, fixed_targets,
                                 n_steps=32, batch_size=64):
    """
    Compute IG with a fixed target:
    - both clean and noisy inputs use the original model's predicted class on the clean sample;
    - the baseline is fixed to the training-set mean trajectory;
    - so changes in the target class do not enter the attribution-stability comparison.
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
    - sample-averaged TxF attribution map
    - feature summary (length F)
    - temporal summary (length T)
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
    print("Step 3: build fixed samples and the clean-input IG reference")

    sample_idx = choose_fixed_samples(X_test, NOISE_IG_SAMPLES, SEED)
    clean_pred, _ = predict_arrays(model, X_test[sample_idx], EVAL_BATCH_SIZE)
    fixed_targets = clean_pred.astype(np.int64)

    print(f"IG fixed samples: {len(sample_idx)}")
    print(f"IG integration steps: {NOISE_IG_STEPS}")
    print(f"Fixed target-class distribution: {pd.Series(fixed_targets).value_counts().sort_index().to_dict()}")

    clean_attr = ig_attribution_fixed_targets(
        model, X_test, X_train, sample_idx, fixed_targets,
        n_steps=NOISE_IG_STEPS
    )

    clean_map, clean_feat, clean_time = summarize_attr_arrays(clean_attr)

    print(f"Clean-input attribution shape: {clean_attr.shape}")
    print(f"Clean-input attribution missing values: {int(np.isnan(clean_attr).sum())}")
    print("Clean-input IG feature importance top 5:")
    top_idx = np.argsort(-clean_feat)[:5]
    for rank, j in enumerate(top_idx, start=1):
        print(f"{rank}. {feature_names[j]} | importance {clean_feat[j]:.8f}")

    feature_sd = get_feature_sd(X_train)
    zero_sd = [feature_names[i] for i, x in enumerate(feature_sd) if x == 0]
    print(f"Training feature SD: min {feature_sd.min():.8f}, max {feature_sd.max():.8f}")
    print(f"Zero-variance features: {len(zero_sd)}")
    if zero_sd:
        print(f"Zero-variance features: {', '.join(zero_sd)}")

    rows = []

    for level in NOISE_LEVELS:
        print("-" * 20)
        print(f"Step 4: input noise = {level*100:.1f}% of the training-set SD")

        for repeat in range(1, NOISE_REPEATS + 1):
            seed = SEED + int(level * 100000) + repeat * 1009
            print(f"Repetition {repeat}/{NOISE_REPEATS} | random seed {seed}")

            try:
                X_noisy = add_feature_scaled_noise(X_test, feature_sd, level, seed)

                # Prediction stability: full test set
                clean_eval = evaluate_arrays(model, X_test, y_test)
                noisy_eval = evaluate_arrays(model, X_noisy, y_test)
                pred_agreement = np.mean(clean_eval["pred"] == noisy_eval["pred"])

                # Attribution stability: 200 fixed samples, fixed original target class
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
                    f"Result: map ρ {map_rho:.6f}, feature ρ {feat_rho:.6f}, "
                    f"temporal ρ {time_rho:.6f}, prediction agreement {pred_agreement:.6f}, "
                    f"ΔF1 {row['Delta_MacroF1']:.6f}, Δaccuracy {row['Delta_Accuracy']:.6f}"
                )

            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


# =========================
# 7. Main
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Task: model prediction and IG stability under small input noise")
    print("Noise: training-set feature SD x 0.5%, 1%, 2%")
    print(f"Repetitions per noise level: {NOISE_REPEATS}")
    print("Attribution comparison: sample-averaged TxF attribution map / feature ranking / temporal ranking")
    print("Prediction comparison: prediction agreement / Macro-F1 / accuracy / true-class probability")

    all_results = []
    all_incomplete = []
    all_checks = []
    error_rows = []

    for dataset_name in RUN_DATASETS:
        cfg = DATASETS[dataset_name]

        print("-" * 20)
        print(f"Dataset: {dataset_name}")
        print("Step 1: data preparation and checks")

        try:
            ds = prepare_dataset(dataset_name, cfg)
            print(f"Number of features: {len(cfg['features'])}")
            print(f"Features: {', '.join(cfg['features'])}")
            print(f"Training samples: {len(ds['X_train'])}, validation samples: {len(ds['X_val'])}, test samples: {len(ds['X_test'])}")
            print(f"Training missing values: {int(np.isnan(ds['X_train']).sum())}, validation missing values: {int(np.isnan(ds['X_val']).sum())}, test missing values: {int(np.isnan(ds['X_test']).sum())}")
            print(f"Training label distribution: {pd.Series(ds['y_train']).value_counts().sort_index().to_dict()}")
            print(f"Test label distribution: {pd.Series(ds['y_test']).value_counts().sort_index().to_dict()}")
            print("Data check: passed")
        except Exception as e:
            print(f"Error: data preparation failed: {type(e).__name__}: {e}")
            print("Data check: failed")
            error_rows.append({
                "Dataset": dataset_name, "Model": "", "Step": "Data preparation",
                "Error": f"{type(e).__name__}: {e}"
            })
            continue

        for model_type in MODELS:
            print("-" * 20)
            print(f"{dataset_name} / {model_type}")
            print("Step 2: load and check models")

            try:
                model, _ = load_model(dataset_name, model_type, cfg)
                base = evaluate_arrays(model, ds["X_test"], ds["y_test"])

                print(f"Test samples: {len(ds['y_test'])}")
                print(f"Model check: test Macro-F1 {base['macro_f1']:.6f}, accuracy {base['accuracy']:.6f}, mean true-class probability {base['mean_true_class_prob']:.6f}")
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
                print(f"Error: model loading or check failed: {type(e).__name__}: {e}")
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
                    # Completeness check: each noise level must have NOISE_REPEATS successful repetitions
                    rep_counts = result.groupby("Noise_Percent_SD").size()
                    complete_levels = []
                    for level in NOISE_LEVELS:
                        n_ok = int(rep_counts.get(level * 100, 0))
                        if n_ok == NOISE_REPEATS:
                            complete_levels.append(level * 100)
                            print(f"Completeness check: noise {level * 100:.1f}%, all {NOISE_REPEATS} repetitions succeeded")
                        else:
                            print(f"Error: noise {level * 100:.1f}%, only {n_ok}/{NOISE_REPEATS} repetitions succeeded; this combination is excluded from the formal summary")
                            error_rows.append({
                                "Dataset": dataset_name, "Model": model_type, "Step": "Noise robustness",
                                "Error": f"Incomplete repeats at {level * 100:.1f}% SD: {n_ok}/{NOISE_REPEATS}"
                            })
                    incomplete = result[~result["Noise_Percent_SD"].isin(complete_levels)]
                    if not incomplete.empty:
                        all_incomplete.append(incomplete)
                    result = result[result["Noise_Percent_SD"].isin(complete_levels)].copy()
                if not result.empty:
                    all_results.append(result)
                    print(f"Noise result rows for the current model (complete combinations only): {len(result)}")
                    print("Summary for the current model:")
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
                    print("Error: the current model has no complete noise results")
                    error_rows.append({
                        "Dataset": dataset_name, "Model": model_type, "Step": "Noise robustness",
                        "Error": "No results produced"
                    })
            except Exception as e:
                print(f"Error: noise-robustness test failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name, "Model": model_type, "Step": "Noise robustness",
                    "Error": f"{type(e).__name__}: {e}"
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("Step 5: save results and produce the final summary")

    df_checks = pd.DataFrame(all_checks)
    df_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    df_errors = pd.DataFrame(error_rows)

    df_checks.to_csv(OUTPUT_DIR / "D0_model_reproduction_check.csv", index=False)
    if not df_results.empty:
        df_results.to_csv(OUTPUT_DIR / "D1_input_noise_robustness_detail.csv", index=False)
    if not df_errors.empty:
        df_errors.to_csv(OUTPUT_DIR / "D9_errors.csv", index=False)
    if all_incomplete:
        pd.concat(all_incomplete, ignore_index=True).to_csv(
            OUTPUT_DIR / "D8_incomplete_repeats_detail.csv", index=False)
        print(f"Details of incomplete combinations saved separately: ./{OUTPUT_DIR.as_posix()}/D8_incomplete_repeats_detail.csv")
    else:
        print("Completeness check: all repetitions succeeded for all models and noise levels")

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

    print(f"Model check rows: {len(df_checks)}")
    print(f"Noise result detail rows: {len(df_results)}")
    print(f"Noise result summary rows: {len(df_summary)}")
    print(f"Error records: {len(df_errors)}")

    print("-" * 20)
    print("Final summary 1: model checks")
    if not df_checks.empty:
        print(df_checks.sort_values(["Dataset", "Model"]).to_string(index=False))
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 2: input-noise robustness")
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
    print("Final summary 3: overall statistics across models for each noise level")
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
    print("stage6_cell_H_data_robustness_V4 finished")
    print(f"Results directory: ./{OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
