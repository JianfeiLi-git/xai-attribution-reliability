# -*- coding: utf-8 -*-
"""
stage6_cell_I_data_labelrandom_V2

Aims:
1. Label randomization test;
2. For 2 U.S. datasets x 3 RNNs: retrain models with randomized labels;
3. Training and validation labels are shuffled separately; the test set keeps the true labels;
4. Compare the test-set performance of the original and random-label models;
5. Compare their IG attribution structures on the same samples, with the same training-set mean baseline and the same fixed target of the original model.

Principles:
- No folder scanning;
- Result files from other cells are not read;
- Original models are read only from the model files selected in stage6_cell_C;
- Random-label models use a common short training protocol (the same as the test on the Chinese datasets in the original paper), not the original optimization hyperparameters of each model.
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
# stage6_cell_I_data_labelrandom_V2
# New U.S. models: label randomization test (retrain with a common short protocol after shuffling training/validation labels, and compare IG structure with the original model)
# Methods are identical to the original paper pipeline (parameters, sample sizes and steps unchanged); only the following configuration changes:
#   - Only the two U.S. datasets (US-Med, US-Pow) are run; data from stage6_cell_A, features from its feature_config.json
#   - seed-42 models are those selected in stage6_cell_C_data_training_V2 (*_seed42_selected.pth)
#   - cuDNN cannot be initialized on the server, so cuDNN is disabled throughout and PyTorch native CUDA RNN kernels are used
# Field names and values written to CSV are in English so that they can be merged with the original result tables
# ==============================================================================
US_DATA_ROOT = "stage6/stage6_cell_A_data_preprocessing_V1"
US_MODEL_ROOT = "stage6/stage6_cell_C_data_training_V2"
OUTPUT_DIR = Path("stage6/stage6_cell_I_data_labelrandom_V2")
# Rule: the first two lines print the input and output locations (relative paths only)
print(f"Input directories: ./{US_DATA_ROOT}/, ./{US_MODEL_ROOT}/")
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")

import json as _json
import time
_T0 = time.time()
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


# =========================
# Settings specific to this cell
# =========================

from torch.utils.data import TensorDataset, DataLoader


# Label-randomization retraining protocol
# The model architecture is inferred strictly from the original model file; the test retraining uses a common fixed training protocol
# and does not reuse the original optimization hyperparameters of each model.
E_BATCH_SIZE = 256
E_LR = 1e-3
E_WEIGHT_DECAY = 1e-5
E_MAX_EPOCHS = 20
E_PATIENCE = 5
E_IG_SAMPLES = 200
E_IG_STEPS = 32

# =========================
# 6. Label randomization
# =========================

def clone_architecture_from_original(model_type, original_model):
    """Read input dim, hidden dim and number of layers from the original model and build a new model with the same architecture."""
    mt = model_type.upper()

    if mt == "GRU":
        rnn = original_model.gru
        model = GRUClassifier(
            input_dim=rnn.input_size,
            hidden_dim=rnn.hidden_size,
            num_layers=rnn.num_layers,
            dropout=0.0,
            num_classes=3,
        )
    elif mt == "LSTM":
        rnn = original_model.lstm
        model = LSTMClassifier(
            input_dim=rnn.input_size,
            hidden_dim=rnn.hidden_size,
            num_layers=rnn.num_layers,
            dropout=0.0,
            num_classes=3,
        )
    elif mt == "BILSTM":
        rnn = original_model.lstm
        model = BiLSTMClassifier(
            input_dim=rnn.input_size,
            hidden_dim=rnn.hidden_size,
            num_layers=rnn.num_layers,
            dropout=0.0,
            num_classes=3,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(DEVICE)


def permute_labels(y, seed):
    rng = np.random.default_rng(seed)
    yp = np.asarray(y).copy()
    rng.shuffle(yp)
    return yp


def train_randomized_label_model(model, X_train, y_train_perm, X_val, y_val_perm):
    """
    Train a model of the same architecture with random labels.
    Early stopping is based on the randomized validation loss, only to avoid pointlessly long training.
    """
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train_perm.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val_perm.astype(np.int64)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=E_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=E_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=E_LR,
        weight_decay=E_WEIGHT_DECAY,
    )

    best_state = None
    best_val = np.inf
    bad_epochs = 0
    history = []

    for epoch in range(1, E_MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(yb)
            train_n += len(yb)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        correct = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_sum += float(loss.item()) * len(yb)
                val_n += len(yb)
                correct += int((logits.argmax(dim=1) == yb).sum().item())

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = correct / max(val_n, 1)

        history.append({
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Permuted_Val_Loss": val_loss,
            "Permuted_Val_Accuracy": val_acc,
        })

        print(
            f"Epoch {epoch:02d}/{E_MAX_EPOCHS}: "
            f"train loss {train_loss:.6f}, "
            f"randomized val loss {val_loss:.6f}, "
            f"randomized val accuracy {val_acc:.6f}, "
            f"total {(time.time() - _T0) / 60:.1f}m"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= E_PATIENCE:
            print(f"Early stopping: no improvement for {E_PATIENCE} consecutive epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, pd.DataFrame(history)


def ig_fixed_target(model, X_input, X_train, sample_idx, fixed_targets,
                    n_steps=32, batch_size=64):
    """
    The original and random-label models use the same samples, the same training-set mean baseline
    and the same fixed target, so that changes in the target class do not interfere with the comparison.
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

        attrs = []

        for s in range(0, len(sample_idx), batch_size):
            idx = sample_idx[s:s + batch_size]
            xb = torch.from_numpy(X_input[idx].astype(np.float32)).to(DEVICE)
            target = torch.from_numpy(
                fixed_targets[s:s + len(idx)].astype(np.int64)
            ).to(DEVICE)

            base = baseline.expand(xb.size(0), -1, -1)

            attr = ig.attribute(
                xb,
                baselines=base,
                target=target,
                n_steps=n_steps,
                internal_batch_size=min(256, xb.size(0) * 4),
            )

            attrs.append(attr.detach().abs().cpu().numpy())

            del xb, target, base, attr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(attrs, axis=0)

    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        model.eval()


def safe_spearman(a, b):
    rho, p = spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel())
    return float(rho), float(p)


def summarize_attr(attr):
    return (
        attr.mean(axis=0),      # T×F
        attr.mean(axis=(0, 1)), # F
        attr.mean(axis=(0, 2)), # T
    )


def evaluate_true_test(model, X_test, y_test):
    res = evaluate_arrays(model, X_test, y_test)
    return {
        "Accuracy": float(res["accuracy"]),
        "Macro_F1": float(res["macro_f1"]),
        "Mean_TrueProb": float(res["mean_true_class_prob"]),
    }


def run_label_randomization_one(dataset_name, model_type, original_model, ds, feature_names):
    print("Step 3: generate random labels")

    y_train_perm = permute_labels(ds["y_train"], SEED + 100)
    y_val_perm = permute_labels(ds["y_val"], SEED + 200)

    print(f"Original training label distribution: {pd.Series(ds['y_train']).value_counts().sort_index().to_dict()}")
    print(f"Shuffled training label distribution: {pd.Series(y_train_perm).value_counts().sort_index().to_dict()}")
    print(f"Original validation label distribution: {pd.Series(ds['y_val']).value_counts().sort_index().to_dict()}")
    print(f"Shuffled validation label distribution: {pd.Series(y_val_perm).value_counts().sort_index().to_dict()}")

    train_changed = float(np.mean(y_train_perm != ds["y_train"]))
    val_changed = float(np.mean(y_val_perm != ds["y_val"]))

    print(f"Share of training labels whose position changed: {train_changed:.6f}")
    print(f"Share of validation labels whose position changed: {val_changed:.6f}")

    print("Step 4: retrain the model with random labels")
    rand_model = clone_architecture_from_original(model_type, original_model)

    print(
        f"Training protocol: batch {E_BATCH_SIZE}, learning rate {E_LR}, "
        f"weight decay {E_WEIGHT_DECAY}, at most {E_MAX_EPOCHS} epochs, early-stopping patience {E_PATIENCE}"
    )

    rand_model, history = train_randomized_label_model(
        rand_model,
        ds["X_train"], y_train_perm,
        ds["X_val"], y_val_perm,
    )

    print(f"Epochs actually trained: {len(history)}")
    print(f"Lowest randomized val loss: {history['Permuted_Val_Loss'].min():.6f}")

    print("Step 5: check predictive ability on the true test labels")
    original_perf = evaluate_true_test(original_model, ds["X_test"], ds["y_test"])
    random_perf = evaluate_true_test(rand_model, ds["X_test"], ds["y_test"])

    print(
        f"Original model: Macro-F1 {original_perf['Macro_F1']:.6f}, "
        f"accuracy {original_perf['Accuracy']:.6f}, "
        f"mean true-class probability {original_perf['Mean_TrueProb']:.6f}"
    )
    print(
        f"Random-label model: Macro-F1 {random_perf['Macro_F1']:.6f}, "
        f"accuracy {random_perf['Accuracy']:.6f}, "
        f"mean true-class probability {random_perf['Mean_TrueProb']:.6f}"
    )
    print(
        f"Change: ΔF1 {random_perf['Macro_F1']-original_perf['Macro_F1']:.6f}, "
        f"Δaccuracy {random_perf['Accuracy']-original_perf['Accuracy']:.6f}"
    )

    print("Step 6: IG comparison between the original and random-label models")
    sample_idx = choose_fixed_samples(ds["X_test"], E_IG_SAMPLES, SEED)

    original_pred, _ = predict_arrays(
        original_model, ds["X_test"][sample_idx], EVAL_BATCH_SIZE
    )
    fixed_targets = original_pred.astype(np.int64)

    print(f"IG fixed samples: {len(sample_idx)}")
    print(f"IG integration steps: {E_IG_STEPS}")
    print(f"Fixed target-class distribution: {pd.Series(fixed_targets).value_counts().sort_index().to_dict()}")

    original_attr = ig_fixed_target(
        original_model, ds["X_test"], ds["X_train"],
        sample_idx, fixed_targets, E_IG_STEPS
    )
    random_attr = ig_fixed_target(
        rand_model, ds["X_test"], ds["X_train"],
        sample_idx, fixed_targets, E_IG_STEPS
    )

    orig_map, orig_feat, orig_time = summarize_attr(original_attr)
    rand_map, rand_feat, rand_time = summarize_attr(random_attr)

    map_rho, map_p = safe_spearman(orig_map, rand_map)
    feat_rho, feat_p = safe_spearman(orig_feat, rand_feat)
    time_rho, time_p = safe_spearman(orig_time, rand_time)

    print(
        f"Attribution correlations: map ρ {map_rho:.6f} (p={map_p:.6g}), "
        f"feature ρ {feat_rho:.6f} (p={feat_p:.6g}), "
        f"temporal ρ {time_rho:.6f} (p={time_p:.6g})"
    )

    print("Original model IG features top 5:")
    for rank, j in enumerate(np.argsort(-orig_feat)[:5], 1):
        print(f"{rank}. {feature_names[j]} | {orig_feat[j]:.8f}")

    print("Random-label model IG features top 5:")
    for rank, j in enumerate(np.argsort(-rand_feat)[:5], 1):
        print(f"{rank}. {feature_names[j]} | {rand_feat[j]:.8f}")

    row = {
        "Dataset": dataset_name,
        "Model": model_type,
        "Train_Label_Changed_Rate": train_changed,
        "Val_Label_Changed_Rate": val_changed,
        "Epochs_Trained": len(history),
        "Best_Permuted_Val_Loss": float(history["Permuted_Val_Loss"].min()),
        "Original_MacroF1": original_perf["Macro_F1"],
        "RandomLabel_MacroF1": random_perf["Macro_F1"],
        "Delta_MacroF1": random_perf["Macro_F1"] - original_perf["Macro_F1"],
        "Original_Accuracy": original_perf["Accuracy"],
        "RandomLabel_Accuracy": random_perf["Accuracy"],
        "Delta_Accuracy": random_perf["Accuracy"] - original_perf["Accuracy"],
        "Original_TrueProb": original_perf["Mean_TrueProb"],
        "RandomLabel_TrueProb": random_perf["Mean_TrueProb"],
        "Map_Spearman": map_rho,
        "Map_p": map_p,
        "Feature_Spearman": feat_rho,
        "Feature_p": feat_p,
        "Temporal_Spearman": time_rho,
        "Temporal_p": time_p,
        "N_IG_Samples": len(sample_idx),
        "IG_Steps": E_IG_STEPS,
    }

    return row, history


# =========================
# 7. Main
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Task: label randomization test")
    print("Dependencies: raw data + fixed model files; results of other cells are not read")
    print("Scope: 2 U.S. datasets x 3 RNNs = 6 models")
    print("Principles: same architecture as the original model; training and validation labels shuffled separately; test set keeps true labels")
    print("IG comparison: same samples, same training-set mean baseline, same fixed target")
    print("Note: this cell uses a common test retraining protocol and does not reuse the original optimization hyperparameters")

    all_rows = []
    all_hist = []
    errors = []

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
            print(f"Validation label distribution: {pd.Series(ds['y_val']).value_counts().sort_index().to_dict()}")
            print(f"Test label distribution: {pd.Series(ds['y_test']).value_counts().sort_index().to_dict()}")
            print("Data check: passed")
        except Exception as e:
            print(f"Error: data preparation failed: {type(e).__name__}: {e}")
            errors.append({
                "Dataset": dataset_name, "Model": "",
                "Step": "Data preparation", "Error": f"{type(e).__name__}: {e}"
            })
            continue

        for model_type in MODELS:
            print("-" * 20)
            print(f"{dataset_name} / {model_type}")
            print("Step 2: load the original model and check its architecture")

            try:
                original_model, _ = load_model(
                    dataset_name, model_type, cfg
                )

                if model_type == "GRU":
                    rnn = original_model.gru
                    direction = "unidirectional"
                else:
                    rnn = original_model.lstm
                    direction = "bidirectional" if model_type == "BiLSTM" else "unidirectional"

                print(
                    f"Architecture: input dim {rnn.input_size}, hidden dim {rnn.hidden_size}, "
                    f"layers {rnn.num_layers}, directions {direction}"
                )

                perf = evaluate_true_test(
                    original_model, ds["X_test"], ds["y_test"]
                )
                print(
                    f"Model check: test Macro-F1 {perf['Macro_F1']:.6f}, "
                    f"accuracy {perf['Accuracy']:.6f}"
                )
                print("Model check: passed")

            except Exception as e:
                print(f"Error: failed to read the original model: {type(e).__name__}: {e}")
                errors.append({
                    "Dataset": dataset_name, "Model": model_type,
                    "Step": "Original model loading", "Error": f"{type(e).__name__}: {e}"
                })
                continue

            try:
                row, hist = run_label_randomization_one(
                    dataset_name, model_type, original_model,
                    ds, cfg["features"]
                )

                all_rows.append(row)

                hist = hist.copy()
                hist.insert(0, "Model", model_type)
                hist.insert(0, "Dataset", dataset_name)
                all_hist.append(hist)

                print("Final results for the current model:")
                print(pd.DataFrame([row])[
                    [
                        "Dataset", "Model",
                        "RandomLabel_MacroF1", "RandomLabel_Accuracy",
                        "Map_Spearman", "Feature_Spearman", "Temporal_Spearman"
                    ]
                ].to_string(index=False))
                print("Label randomization: done")

            except Exception as e:
                print(f"Error: label randomization failed: {type(e).__name__}: {e}")
                errors.append({
                    "Dataset": dataset_name, "Model": model_type,
                    "Step": "Label randomization",
                    "Error": f"{type(e).__name__}: {e}"
                })

            del original_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("Step 7: save results and produce the final summary")

    df = pd.DataFrame(all_rows)
    hist_df = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    err_df = pd.DataFrame(errors)

    if not df.empty:
        df.to_csv(
            OUTPUT_DIR / "E1_label_randomization_sanity.csv",
            index=False
        )
    if not hist_df.empty:
        hist_df.to_csv(
            OUTPUT_DIR / "E2_label_randomization_training_history.csv",
            index=False
        )
    if not err_df.empty:
        err_df.to_csv(
            OUTPUT_DIR / "E9_errors.csv",
            index=False
        )

    print(f"Label randomization result rows: {len(df)}")
    print(f"Training history rows: {len(hist_df)}")
    print(f"Error records: {len(err_df)}")

    print("-" * 20)
    print("Final summary 1: performance of random-label models on the true test labels")
    if not df.empty:
        print(
            df[
                [
                    "Dataset", "Model",
                    "Original_MacroF1", "RandomLabel_MacroF1", "Delta_MacroF1",
                    "Original_Accuracy", "RandomLabel_Accuracy", "Delta_Accuracy"
                ]
            ].sort_values(["Dataset", "Model"]).to_string(index=False)
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 2: attribution correlations between original and random-label models")
    if not df.empty:
        print(
            df[
                [
                    "Dataset", "Model",
                    "Map_Spearman", "Map_p",
                    "Feature_Spearman", "Feature_p",
                    "Temporal_Spearman", "Temporal_p"
                ]
            ].sort_values(["Dataset", "Model"]).to_string(index=False)
        )
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 3: cross-model statistics")
    if not df.empty:
        summary = pd.DataFrame({
            "Metric": [
                "RandomLabel_MacroF1",
                "RandomLabel_Accuracy",
                "Map_Spearman",
                "Feature_Spearman",
                "Temporal_Spearman",
            ],
            "Mean": [
                df["RandomLabel_MacroF1"].mean(),
                df["RandomLabel_Accuracy"].mean(),
                df["Map_Spearman"].mean(),
                df["Feature_Spearman"].mean(),
                df["Temporal_Spearman"].mean(),
            ],
            "Min": [
                df["RandomLabel_MacroF1"].min(),
                df["RandomLabel_Accuracy"].min(),
                df["Map_Spearman"].min(),
                df["Feature_Spearman"].min(),
                df["Temporal_Spearman"].min(),
            ],
            "Max": [
                df["RandomLabel_MacroF1"].max(),
                df["RandomLabel_Accuracy"].max(),
                df["Map_Spearman"].max(),
                df["Feature_Spearman"].max(),
                df["Temporal_Spearman"].max(),
            ],
        })
        print(summary.to_string(index=False))
    else:
        print("No results")

    print("-" * 20)
    print("Final summary 4: errors")
    if err_df.empty:
        print("Errors: 0, all steps completed")
    else:
        print(err_df.to_string(index=False))

    print("-" * 20)
    print("stage6_cell_I_data_labelrandom_V2 finished")
    print(f"Results directory: ./{OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
