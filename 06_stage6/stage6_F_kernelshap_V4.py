# -*- coding: utf-8 -*-
"""
stage6_cell_F_data_kernelshap_V4

Aims:
1. Model and data checks;
2. Grouped KernelSHAP: each feature group covers all 60 timesteps;
3. Missing feature groups are replaced using real training windows as background;
4. Independent-method comparison with the IG feature importance from stage6_cell_E_data_attribution_V2;
5. Top-k paired perturbation test: k = 1/3/5, 50 repetitions;
6. Within one repetition, Top-k and Random-k use exactly the same training-background indices.

Principles:
- No folder scanning;
- No retraining; only the seed-42 models selected in stage6_cell_C are used;
- KernelSHAP explains the probability of the original model's predicted class;
- The main perturbation-test metric is the drop in the original predicted-class probability.
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
import shap

warnings.filterwarnings("ignore")

# =========================
# 0. Global settings
# =========================

SEED = 42
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# KernelSHAP settings
N_EXPLAIN = 50
N_BACKGROUNDS = 10
KERNEL_NSAMPLES = 256
KERNEL_BATCH = 1024

# Perturbation-test settings
FAITHFULNESS_K = [1, 3, 5]
FAITHFULNESS_REPEATS = 50
EVAL_BATCH = 512



# =========================
# 1. Explicit paths, no folder scanning
# =========================

# ==============================================================================
# stage6_cell_F_data_kernelshap_V4
# New U.S. models: grouped KernelSHAP, comparison with the IG feature importance from stage6_cell_E, Top-k paired perturbation test (k=1/3/5, 50 repetitions)
# Methods are identical to the original paper pipeline (parameters, sample sizes and steps unchanged); only the following configuration changes:
#   - Only the two U.S. datasets (US-Med, US-Pow) are run; data from stage6_cell_A, features from its feature_config.json
#   - seed-42 models are those selected in stage6_cell_C_data_training_V2 (*_seed42_selected.pth)
#   - cuDNN cannot be initialized on the server, so cuDNN is disabled throughout and PyTorch native CUDA RNN kernels are used
# V4 changes: KernelSHAP output is parsed after removing all length-1 dimensions (compatible with (F,), (1,F), (1,F,1), (1,1,F) and list forms),
#             then its length is checked against the number of features, with an error if they differ (the computation itself is unchanged)
# Field names and values written to CSV are in English so that they can be merged with the original result tables
# ==============================================================================
US_DATA_ROOT = "stage6/stage6_cell_A_data_preprocessing_V1"
US_MODEL_ROOT = "stage6/stage6_cell_C_data_training_V2"
OUTPUT_DIR = Path("stage6/stage6_cell_F_data_kernelshap_V4")
# Rule: the first two lines print the input and output locations (relative paths only)
print(f"Input directories: ./{US_DATA_ROOT}/, ./{US_MODEL_ROOT}/, ./stage6/stage6_cell_E_data_attribution_V2/")
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
IG_RESULT_PATH = Path("stage6/stage6_cell_E_data_attribution_V2/A2_ig_feature_importance.csv")
_need.append(IG_RESULT_PATH)
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


# =========================
# 2. Random seed
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# 3. Data functions
# =========================

def load_data_raw(data_dir):
    X_train = pd.read_csv(data_dir / "train_X.csv")
    X_val = pd.read_csv(data_dir / "val_X.csv")
    X_test = pd.read_csv(data_dir / "test_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv")
    y_val = pd.read_csv(data_dir / "val_y.csv")
    y_test = pd.read_csv(data_dir / "test_y.csv")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def filter_data_by_date(X_df, y_df, start_date="2010-01-01"):
    X_df = X_df.copy()
    y_df = y_df.copy()
    X_df["trade_date"] = pd.to_datetime(X_df["trade_date"])
    y_df["trade_date"] = pd.to_datetime(y_df["trade_date"])
    start = pd.to_datetime(start_date)
    return (
        X_df[X_df["trade_date"] >= start].copy(),
        y_df[y_df["trade_date"] >= start].copy(),
    )


def calculate_thresholds(y_df, target_col="pct_chg"):
    arr = y_df[target_col].dropna()
    return (
        np.percentile(arr, 33.33),
        np.percentile(arr, 66.67),
    )


def build_sequence_data_with_meta(
    X_df, y_df, feature_cols, q33, q67, max_seq_len=60
):
    y_work = y_df.copy()
    y_work["trade_date"] = pd.to_datetime(y_work["trade_date"])
    y_work["future_change"] = (
        y_work.groupby("stock_code")["pct_chg"].shift(-1)
    )

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

    merged = pd.merge(
        X_work,
        y_work[
            ["stock_code", "trade_date", "label", "pct_chg", "future_change"]
        ],
        on=["stock_code", "trade_date"],
        how="inner",
    )

    all_seqs, all_labels, all_meta = [], [], []

    for _, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue

        g = g.sort_values("trade_date")
        feats = g[feature_cols].values
        dates = g["trade_date"].values
        labels = g["label"].values
        meta = g[
            ["stock_code", "trade_date", "label", "pct_chg", "future_change"]
        ].values

        for end in range(max_seq_len, len(g)):
            start = end - max_seq_len
            days_diff = (
                (dates[end - 1] - dates[start])
                / np.timedelta64(1, "D")
            )
            if days_diff > 100:
                continue

            target = labels[end - 1]
            if target == -999:
                continue

            all_seqs.append(feats[start:end])
            all_labels.append(target)
            all_meta.append(meta[end - 1])

    meta_df = pd.DataFrame(
        all_meta,
        columns=[
            "stock_code", "trade_date", "label",
            "pct_chg", "future_change"
        ],
    )

    return (
        np.stack(all_seqs).astype(np.float32),
        np.asarray(all_labels, dtype=np.int64),
        meta_df,
    )


def prepare_dataset(dataset_name, cfg):
    data_dir = DATA_DIR[dataset_name]
    print(f"{dataset_name}: data directory ./{data_dir.as_posix()}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)

    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)

    q33, q67 = calculate_thresholds(y_tr)

    X_train, y_train, _ = build_sequence_data_with_meta(
        X_tr, y_tr, cfg["features"], q33, q67, SEQ_LEN
    )
    X_val, y_val, _ = build_sequence_data_with_meta(
        X_va, y_va, cfg["features"], q33, q67, SEQ_LEN
    )
    X_test, y_test, meta_test = build_sequence_data_with_meta(
        X_te, y_te, cfg["features"], q33, q67, SEQ_LEN
    )

    print(
        f"{dataset_name}: train {X_train.shape}, "
        f"validation {X_val.shape}, test {X_test.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


# =========================
# 4. Model definition and loading
# =========================

class GRUClassifier(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=128,
        num_layers=2, dropout=0.0, num_classes=3
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = torch.nan_to_num(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=128,
        num_layers=2, dropout=0.0, num_classes=3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMClassifier(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=128,
        num_layers=2, dropout=0.0, num_classes=3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = torch.nan_to_num(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def clean_state_dict(state):
    if "state_dict" in state:
        state = state["state_dict"]
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state.items()
    }


def infer_model(model_type, state):
    state = clean_state_dict(state)

    if model_type == "GRU":
        w_ih = state["gru.weight_ih_l0"]
        w_hh = state["gru.weight_hh_l0"]
        input_dim = w_ih.shape[1]
        hidden_dim = w_hh.shape[1]
        layers = [
            int(k.split("_l")[1].split("_")[0])
            for k in state
            if k.startswith("gru.weight_ih_l") and "_reverse" not in k
        ]
        model = GRUClassifier(
            input_dim, hidden_dim, max(layers) + 1, dropout=0.0
        )

    elif model_type == "LSTM":
        w_ih = state["lstm.weight_ih_l0"]
        w_hh = state["lstm.weight_hh_l0"]
        input_dim = w_ih.shape[1]
        hidden_dim = w_hh.shape[1]
        layers = [
            int(k.split("_l")[1].split("_")[0])
            for k in state
            if k.startswith("lstm.weight_ih_l") and "_reverse" not in k
        ]
        model = LSTMClassifier(
            input_dim, hidden_dim, max(layers) + 1, dropout=0.0
        )

    else:
        w_ih = state["lstm.weight_ih_l0"]
        w_hh = state["lstm.weight_hh_l0"]
        input_dim = w_ih.shape[1]
        hidden_dim = w_hh.shape[1]
        layers = [
            int(k.split("_l")[1].split("_")[0])
            for k in state
            if k.startswith("lstm.weight_ih_l") and "_reverse" not in k
        ]
        model = BiLSTMClassifier(
            input_dim, hidden_dim, max(layers) + 1, dropout=0.0
        )

    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def load_model(dataset_name, model_type):
    path = MODEL_PATH[(dataset_name, model_type)]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: ./{Path(path).as_posix()}")

    state = torch.load(path, map_location="cpu")
    model = infer_model(model_type, state)

    print(f"{dataset_name}-{model_type}: ./{path.as_posix()}")
    return model


# =========================
# 5. General prediction
# =========================

def predict_prob(model, X, batch_size=512):
    model.eval()
    out = []

    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(
                X[s:s + batch_size].astype(np.float32)
            ).to(DEVICE)

            prob = torch.softmax(model(xb), dim=1)
            out.append(prob.cpu().numpy())

    return np.concatenate(out, axis=0)


def base_metrics(model, X, y):
    prob = predict_prob(model, X, EVAL_BATCH)
    pred = prob.argmax(axis=1)

    return {
        "prob": prob,
        "pred": pred,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }


# =========================
# 6. Grouped KernelSHAP
# =========================

def make_grouped_value_function(
    model, x_sample, target_class, bg_windows
):
    """
    Input Z: n_masks x F
    Z[j,f]=1: keep the original sample's trajectory of feature f over all 60 timesteps
    Z[j,f]=0: use the trajectory of that feature from the same real training background window

    The coalition value is averaged over multiple background windows.
    """

    n_bg, T, F = bg_windows.shape

    def value_fn(Z):
        Z = np.asarray(Z, dtype=np.float32)

        if Z.ndim == 1:
            Z = Z.reshape(1, -1)

        # KernelExplainer should in theory give 0/1 masks;
        # to be safe, clip to [0,1]
        Z = np.clip(Z, 0.0, 1.0)

        values = []

        for s in range(0, len(Z), KERNEL_BATCH):
            z_batch = Z[s:s + KERNEL_BATCH]
            M = len(z_batch)

            # M × B × T × F
            z = z_batch[:, None, None, :]
            bg = bg_windows[None, :, :, :]
            x = x_sample[None, None, :, :]

            hybrid = z * x + (1.0 - z) * bg
            hybrid = hybrid.reshape(M * n_bg, T, F).astype(np.float32)

            prob = predict_prob(model, hybrid, EVAL_BATCH)
            p_target = prob[:, target_class].reshape(M, n_bg).mean(axis=1)
            values.append(p_target)

        return np.concatenate(values, axis=0)

    return value_fn


def grouped_kernelshap(
    dataset_name, model_type, model,
    X_train, X_test, feature_names
):
    rng = np.random.default_rng(SEED)

    n_exp = min(N_EXPLAIN, len(X_test))
    explain_idx = np.sort(
        rng.choice(len(X_test), size=n_exp, replace=False)
    )

    all_rows = []

    zero_bg = np.zeros((1, len(feature_names)), dtype=np.float32)
    full_mask = np.ones((1, len(feature_names)), dtype=np.float32)

    for ii, idx in enumerate(explain_idx, start=1):
        x = X_test[idx]

        # Fix the set of background windows for this sample
        bg_idx = rng.choice(
            len(X_train),
            size=min(N_BACKGROUNDS, len(X_train)),
            replace=False,
        )
        bg_windows = X_train[bg_idx]

        # Explain the original model's predicted class
        p0 = predict_prob(
            model, x[None, :, :], batch_size=1
        )[0]
        target_class = int(np.argmax(p0))

        value_fn = make_grouped_value_function(
            model, x, target_class, bg_windows
        )

        explainer = shap.KernelExplainer(
            value_fn,
            zero_bg,
            link="identity",
        )

        shap_values = explainer.shap_values(
            full_mask,
            nsamples=KERNEL_NSAMPLES,
            silent=True,
        )

        arr = np.asarray(shap_values)

        # Compatible with different shap versions: one sample and one output are explained, so after removing all length-1 dimensions the result should be 1-D
        phi = np.squeeze(arr)
        if phi.ndim != 1:
            raise ValueError(
                f"Cannot recognize the KernelSHAP output shape: {arr.shape}"
            )

        # Length check: the attribution vector length must equal the number of features; otherwise stop with an error and write no wrong results
        if len(phi) != len(feature_names):
            raise ValueError(
                f"Wrong KernelSHAP output length: {len(phi)} != {len(feature_names)} (original shape {arr.shape})"
            )

        for f, name in enumerate(feature_names):
            all_rows.append({
                "Dataset": dataset_name,
                "Model": model_type,
                "Sample_Index": int(idx),
                "Target_Class": target_class,
                "Feature": name,
                "KernelSHAP_Value": float(phi[f]),
                "Abs_KernelSHAP": float(abs(phi[f])),
            })

        if ii % 10 == 0 or ii == n_exp:
            print(
                f"  KernelSHAP progress: {ii}/{n_exp}"
            )

    detail = pd.DataFrame(all_rows)

    summary = (
        detail.groupby(
            ["Dataset", "Model", "Feature"],
            as_index=False
        )
        .agg(
            KernelSHAP_Importance=("Abs_KernelSHAP", "mean"),
            KernelSHAP_SignedMean=("KernelSHAP_Value", "mean"),
            KernelSHAP_SD=("Abs_KernelSHAP", "std"),
            N_Samples=("Sample_Index", "nunique"),
        )
    )

    summary["KernelSHAP_Rank"] = (
        summary.groupby(["Dataset", "Model"])["KernelSHAP_Importance"]
        .rank(method="average", ascending=False)
    )

    return detail, summary


# =========================
# 7. Comparison with IG
# =========================

def compare_with_ig(kernel_summary, ig_df):
    rows = []

    for (dataset, model), g in kernel_summary.groupby(
        ["Dataset", "Model"]
    ):
        ig = ig_df[
            (ig_df["Dataset"] == dataset)
            & (ig_df["Model"] == model)
        ][["Feature", "IG_Importance"]].copy()

        m = g.merge(ig, on="Feature", how="inner")

        if len(m) >= 3:
            rho, p = spearmanr(
                m["KernelSHAP_Importance"],
                m["IG_Importance"],
            )
        else:
            rho, p = np.nan, np.nan

        rows.append({
            "Dataset": dataset,
            "Model": model,
            "N_Features": len(m),
            "IG_vs_KernelSHAP_rho": rho,
            "p_value": p,
        })

    return pd.DataFrame(rows)


# =========================
# 8. Feature perturbation test
# =========================

def mask_feature_groups_with_bg_idx(
    X, X_train, feature_indices, bg_idx
):
    """
    Replace feature groups using externally fixed background indices.
    Within one repetition, Top-k and Random-k use exactly the same bg_idx,
    to remove additional noise caused by different background sampling.
    """
    X_mask = X.copy()
    X_mask[:, :, feature_indices] = X_train[
        bg_idx, :, :
    ][:, :, feature_indices]
    return X_mask


def _paired_interval(values):
    """Monte Carlo repetitions: mean, standard deviation and 2.5%/97.5% percentiles."""
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "q025": float(np.quantile(a, 0.025)),
        "q975": float(np.quantile(a, 0.975)),
    }


def run_faithfulness_paired(
    dataset_name, model_type, model,
    X_train, X_test, y_test,
    feature_names, kernel_summary
):
    rng = np.random.default_rng(SEED)

    base = base_metrics(model, X_test, y_test)
    base_prob = base["prob"]
    base_pred = base["pred"]
    base_pred_prob = base_prob[
        np.arange(len(X_test)), base_pred
    ].mean()

    rank_df = (
        kernel_summary[
            (kernel_summary["Dataset"] == dataset_name)
            & (kernel_summary["Model"] == model_type)
        ]
        .sort_values("KernelSHAP_Importance", ascending=False)
    )

    if len(rank_df) != len(feature_names):
        raise ValueError(
            f"KernelSHAP feature count mismatch: {len(rank_df)} in summary, {len(feature_names)} expected"
        )

    ordered_features = rank_df["Feature"].tolist()
    name_to_idx = {
        name: i for i, name in enumerate(feature_names)
    }

    rows = []
    detail_rows = []

    for k in FAITHFULNESS_K:
        k = min(k, len(feature_names))
        top_names = ordered_features[:k]
        top_idx = np.asarray(
            [name_to_idx[x] for x in top_names],
            dtype=int
        )

        paired_extra_f1 = []
        paired_extra_acc = []
        paired_extra_prob = []

        top_delta_f1 = []
        rand_delta_f1 = []
        top_delta_acc = []
        rand_delta_acc = []
        top_delta_prob = []
        rand_delta_prob = []

        for rep in range(FAITHFULNESS_REPEATS):
            # Within one repetition, Top-k and Random-k use the same set of real training background windows
            bg_idx = rng.integers(
                0, len(X_train), size=len(X_test)
            )

            # The Random-k feature set is still drawn independently at random;
            # overlap with Top-k is allowed, making the comparison more conservative.
            rand_idx = np.sort(
                rng.choice(
                    len(feature_names),
                    size=k,
                    replace=False,
                )
            )

            X_top = mask_feature_groups_with_bg_idx(
                X_test, X_train, top_idx, bg_idx
            )
            X_rand = mask_feature_groups_with_bg_idx(
                X_test, X_train, rand_idx, bg_idx
            )

            top_prob = predict_prob(
                model, X_top, EVAL_BATCH
            )
            rand_prob = predict_prob(
                model, X_rand, EVAL_BATCH
            )

            top_pred = top_prob.argmax(axis=1)
            rand_pred = rand_prob.argmax(axis=1)

            top_f1 = f1_score(
                y_test, top_pred, average="macro"
            )
            rand_f1 = f1_score(
                y_test, rand_pred, average="macro"
            )

            top_acc = accuracy_score(
                y_test, top_pred
            )
            rand_acc = accuracy_score(
                y_test, rand_pred
            )

            top_orig_prob = top_prob[
                np.arange(len(X_test)), base_pred
            ].mean()
            rand_orig_prob = rand_prob[
                np.arange(len(X_test)), base_pred
            ].mean()

            d_top_f1 = base["macro_f1"] - top_f1
            d_rand_f1 = base["macro_f1"] - rand_f1
            d_top_acc = base["accuracy"] - top_acc
            d_rand_acc = base["accuracy"] - rand_acc
            d_top_prob = base_pred_prob - top_orig_prob
            d_rand_prob = base_pred_prob - rand_orig_prob

            extra_f1 = d_top_f1 - d_rand_f1
            extra_acc = d_top_acc - d_rand_acc
            extra_prob = d_top_prob - d_rand_prob

            top_delta_f1.append(d_top_f1)
            rand_delta_f1.append(d_rand_f1)
            top_delta_acc.append(d_top_acc)
            rand_delta_acc.append(d_rand_acc)
            top_delta_prob.append(d_top_prob)
            rand_delta_prob.append(d_rand_prob)

            paired_extra_f1.append(extra_f1)
            paired_extra_acc.append(extra_acc)
            paired_extra_prob.append(extra_prob)

            detail_rows.append({
                "Dataset": dataset_name,
                "Model": model_type,
                "K": k,
                "Repeat": rep + 1,
                "TopK_Features": "|".join(top_names),
                "RandomK_Features": "|".join(
                    [feature_names[i] for i in rand_idx]
                ),
                "TopK_Delta_MacroF1": d_top_f1,
                "RandomK_Delta_MacroF1": d_rand_f1,
                "Extra_Delta_MacroF1": extra_f1,
                "TopK_Delta_Accuracy": d_top_acc,
                "RandomK_Delta_Accuracy": d_rand_acc,
                "Extra_Delta_Accuracy": extra_acc,
                "TopK_Delta_OriginalPredProb": d_top_prob,
                "RandomK_Delta_OriginalPredProb": d_rand_prob,
                "Extra_Delta_OriginalPredProb": extra_prob,
            })

            del X_top, X_rand, top_prob, rand_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        f1_s = _paired_interval(paired_extra_f1)
        acc_s = _paired_interval(paired_extra_acc)
        prob_s = _paired_interval(paired_extra_prob)

        rows.append({
            "Dataset": dataset_name,
            "Model": model_type,
            "K": k,
            "TopK_Features": "|".join(top_names),
            "Repeats": FAITHFULNESS_REPEATS,

            "Baseline_MacroF1": base["macro_f1"],
            "TopK_Delta_MacroF1_Mean": np.mean(top_delta_f1),
            "RandomK_Delta_MacroF1_Mean": np.mean(rand_delta_f1),
            "Extra_Delta_MacroF1_Mean": f1_s["mean"],
            "Extra_Delta_MacroF1_SD": f1_s["sd"],
            "Extra_Delta_MacroF1_Q025": f1_s["q025"],
            "Extra_Delta_MacroF1_Q975": f1_s["q975"],

            "Baseline_Accuracy": base["accuracy"],
            "TopK_Delta_Accuracy_Mean": np.mean(top_delta_acc),
            "RandomK_Delta_Accuracy_Mean": np.mean(rand_delta_acc),
            "Extra_Delta_Accuracy_Mean": acc_s["mean"],
            "Extra_Delta_Accuracy_SD": acc_s["sd"],
            "Extra_Delta_Accuracy_Q025": acc_s["q025"],
            "Extra_Delta_Accuracy_Q975": acc_s["q975"],

            "Baseline_OriginalPredProb": base_pred_prob,
            "TopK_Delta_OriginalPredProb_Mean": np.mean(top_delta_prob),
            "RandomK_Delta_OriginalPredProb_Mean": np.mean(rand_delta_prob),
            "Extra_Delta_OriginalPredProb_Mean": prob_s["mean"],
            "Extra_Delta_OriginalPredProb_SD": prob_s["sd"],
            "Extra_Delta_OriginalPredProb_Q025": prob_s["q025"],
            "Extra_Delta_OriginalPredProb_Q975": prob_s["q975"],

            # Describes only the directional stability across Monte Carlo repetitions; not a formal significance test
            "Positive_ExtraProb_Repeats": int(
                np.sum(np.asarray(paired_extra_prob) > 0)
            ),
            "Positive_ExtraProb_Rate": float(
                np.mean(np.asarray(paired_extra_prob) > 0)
            ),
        })

    return pd.DataFrame(rows), pd.DataFrame(detail_rows)

# =========================
# 9. Main
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Tasks: grouped KernelSHAP, IG-KernelSHAP agreement, Top-k paired perturbation test")
    print("Grouped KernelSHAP: recomputed")
    print(f"KernelSHAP explained samples: {N_EXPLAIN}")
    print(f"Training backgrounds per explained sample: {N_BACKGROUNDS}")
    print(f"Perturbation-test repetitions: {FAITHFULNESS_REPEATS}")
    print("Main metric: drop in the original predicted-class probability")

    if not IG_RESULT_PATH.exists():
        print(f"Error: IG results from stage6_cell_E not found: ./{IG_RESULT_PATH.as_posix()}")
        print("stage6_cell_F_data_kernelshap_V4 stopped")
        return

    try:
        ig_df = pd.read_csv(IG_RESULT_PATH)
        required_ig = {
            "Dataset", "Model", "Feature", "IG_Importance"
        }
        missing_ig = required_ig - set(ig_df.columns)
        if missing_ig:
            raise ValueError(
                f"IG results are missing fields: {sorted(missing_ig)}"
            )
        print(
            f"IG results from stage6_cell_E loaded: {len(ig_df)} rows, "
            f"missing values {int(ig_df.isna().sum().sum())}"
        )
    except Exception as e:
        print(f"Error: failed to read IG results from stage6_cell_E: {type(e).__name__}: {e}")
        print("stage6_cell_F_data_kernelshap_V4 stopped")
        return

    all_checks = []
    all_kernel_detail = []
    all_kernel_summary = []
    all_compare = []
    all_faith_summary = []
    all_faith_detail = []
    error_rows = []

    for dataset_name in RUN_DATASETS:
        print("-" * 20)
        print(f"Dataset: {dataset_name}")
        print("Step 1: data preparation and checks")

        cfg = DATASETS[dataset_name]

        try:
            ds = prepare_dataset(dataset_name, cfg)
            print(f"Number of features: {len(cfg['features'])}")
            print(f"Features: {', '.join(cfg['features'])}")
            print(
                f"Training samples: {len(ds['X_train'])}, "
                f"validation samples: {len(ds['X_val'])}, "
                f"test samples: {len(ds['X_test'])}"
            )
            print(
                f"Training missing values: {int(np.isnan(ds['X_train']).sum())}, "
                f"validation missing values: {int(np.isnan(ds['X_val']).sum())}, "
                f"test missing values: {int(np.isnan(ds['X_test']).sum())}"
            )
            print("Data check: passed")
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

            print("Step 2: load and check models")
            try:
                model = load_model(dataset_name, model_type)
                base = base_metrics(
                    model, ds["X_test"], ds["y_test"]
                )
                print(
                    f"Model check: test Macro-F1 {base['macro_f1']:.6f}, "
                    f"accuracy {base['accuracy']:.6f}"
                )
                all_checks.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Macro_F1": base["macro_f1"],
                    "Accuracy": base["accuracy"],
                    "Test_N": len(ds["y_test"]),
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

            print("Step 3: grouped KernelSHAP")
            try:
                detail, summary = grouped_kernelshap(
                    dataset_name, model_type, model,
                    ds["X_train"], ds["X_test"], cfg["features"]
                )
                all_kernel_detail.append(detail)
                all_kernel_summary.append(summary)

                print(
                    f"Detail rows: {len(detail)}, "
                    f"summary rows: {len(summary)}, "
                    f"missing values: {int(summary.isna().sum().sum())}"
                )
                print("KernelSHAP top 5:")
                print(
                    summary.sort_values(
                        "KernelSHAP_Importance",
                        ascending=False
                    )[[
                        "Feature",
                        "KernelSHAP_Importance",
                        "KernelSHAP_SignedMean",
                        "KernelSHAP_Rank",
                    ]]
                    .head(5)
                    .to_string(index=False)
                )

                cmp_one = compare_with_ig(
                    summary, ig_df
                )
                all_compare.append(cmp_one)
                print("IG vs KernelSHAP:")
                print(cmp_one.to_string(index=False))
                print("Grouped KernelSHAP: done")
            except Exception as e:
                print(f"Error: grouped KernelSHAP failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Grouped KernelSHAP",
                    "Error": f"{type(e).__name__}: {e}",
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            print("Step 4: Top-k paired perturbation test")
            try:
                faith_summary, faith_detail = (
                    run_faithfulness_paired(
                        dataset_name, model_type, model,
                        ds["X_train"], ds["X_test"], ds["y_test"],
                        cfg["features"], summary
                    )
                )
                all_faith_summary.append(faith_summary)
                all_faith_detail.append(faith_detail)

                show = faith_summary[[
                    "K",
                    "TopK_Delta_OriginalPredProb_Mean",
                    "RandomK_Delta_OriginalPredProb_Mean",
                    "Extra_Delta_OriginalPredProb_Mean",
                    "Extra_Delta_OriginalPredProb_SD",
                    "Extra_Delta_OriginalPredProb_Q025",
                    "Extra_Delta_OriginalPredProb_Q975",
                    "Positive_ExtraProb_Rate",
                    "Extra_Delta_MacroF1_Mean",
                    "TopK_Features",
                ]]
                print(show.to_string(index=False))
                print("Paired perturbation test: done")
            except Exception as e:
                print(f"Error: perturbation test failed: {type(e).__name__}: {e}")
                error_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Step": "Paired faithfulness",
                    "Error": f"{type(e).__name__}: {e}",
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("Step 5: save results and produce the final summary")

    def concat_or_empty(items):
        return (
            pd.concat(items, ignore_index=True)
            if items else pd.DataFrame()
        )

    df_checks = pd.DataFrame(all_checks)
    df_kernel_detail = concat_or_empty(all_kernel_detail)
    df_kernel_summary = concat_or_empty(all_kernel_summary)
    df_compare = concat_or_empty(all_compare)
    df_faith_summary = concat_or_empty(all_faith_summary)
    df_faith_detail = concat_or_empty(all_faith_detail)
    df_errors = pd.DataFrame(error_rows)

    if not df_checks.empty:
        df_checks.to_csv(
            OUTPUT_DIR / "B0_model_reproduction_check.csv",
            index=False
        )
    if not df_kernel_detail.empty:
        df_kernel_detail.to_csv(
            OUTPUT_DIR / "B1_grouped_kernelshap_detail.csv",
            index=False
        )
    if not df_kernel_summary.empty:
        df_kernel_summary.to_csv(
            OUTPUT_DIR / "B1_grouped_kernelshap_summary.csv",
            index=False
        )
    if not df_compare.empty:
        df_compare.to_csv(
            OUTPUT_DIR / "B2_ig_vs_kernelshap.csv",
            index=False
        )
    if not df_faith_summary.empty:
        df_faith_summary.to_csv(
            OUTPUT_DIR / "B3_paired_faithfulness_summary.csv",
            index=False
        )
    if not df_faith_detail.empty:
        df_faith_detail.to_csv(
            OUTPUT_DIR / "B3_paired_faithfulness_detail.csv",
            index=False
        )
    if not df_errors.empty:
        df_errors.to_csv(
            OUTPUT_DIR / "B9_errors.csv",
            index=False
        )

    print(f"Model check rows: {len(df_checks)}")
    print(f"KernelSHAP detail rows: {len(df_kernel_detail)}")
    print(f"KernelSHAP summary rows: {len(df_kernel_summary)}")
    print(f"IG vs KernelSHAP rows: {len(df_compare)}")
    print(f"Perturbation-test summary rows: {len(df_faith_summary)}")
    print(f"Perturbation-test detail rows: {len(df_faith_detail)}")
    print(f"Error records: {len(df_errors)}")

    print("-" * 20)
    print("Final summary 1: IG vs KernelSHAP")
    if not df_compare.empty:
        print(
            df_compare.sort_values(
                ["Dataset", "Model"]
            ).to_string(index=False)
        )
        rho = df_compare["IG_vs_KernelSHAP_rho"]
        p = df_compare["p_value"]
        print(
            f"rho range: {rho.min():.6f} - {rho.max():.6f}"
        )
        print(
            f"Combinations with positive correlation and p<0.05: "
            f"{int(((rho > 0) & (p < 0.05)).sum())}/"
            f"{len(df_compare)}"
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 2: paired perturbation test (drop in original predicted-class probability)")
    if not df_faith_summary.empty:
        cols = [
            "Dataset", "Model", "K",
            "TopK_Delta_OriginalPredProb_Mean",
            "RandomK_Delta_OriginalPredProb_Mean",
            "Extra_Delta_OriginalPredProb_Mean",
            "Extra_Delta_OriginalPredProb_SD",
            "Extra_Delta_OriginalPredProb_Q025",
            "Extra_Delta_OriginalPredProb_Q975",
            "Positive_ExtraProb_Rate",
        ]
        print(
            df_faith_summary[cols]
            .sort_values(["Dataset", "Model", "K"])
            .to_string(index=False)
        )

        n_total = len(df_faith_summary)
        n_positive = int(
            (
                df_faith_summary[
                    "Extra_Delta_OriginalPredProb_Mean"
                ] > 0
            ).sum()
        )
        n_q025 = int(
            (
                df_faith_summary[
                    "Extra_Delta_OriginalPredProb_Q025"
                ] > 0
            ).sum()
        )
        print(
            f"Comparisons with mean extra probability drop > 0: "
            f"{n_positive}/{n_total}"
        )
        print(
            f"Comparisons with Monte Carlo 2.5th percentile > 0: "
            f"{n_q025}/{n_total}"
        )
        print(
            "Note: the percentiles describe the Monte Carlo stability over background and random-k "
            "repetitions, not confidence intervals based on independent samples."
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 3: secondary metric Macro-F1")
    if not df_faith_summary.empty:
        print(
            df_faith_summary[[
                "Dataset", "Model", "K",
                "TopK_Delta_MacroF1_Mean",
                "RandomK_Delta_MacroF1_Mean",
                "Extra_Delta_MacroF1_Mean",
                "Extra_Delta_MacroF1_Q025",
                "Extra_Delta_MacroF1_Q975",
            ]]
            .sort_values(["Dataset", "Model", "K"])
            .to_string(index=False)
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
    print("stage6_cell_F_data_kernelshap_V4 finished")
    print(f"Results directory: ./{OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
