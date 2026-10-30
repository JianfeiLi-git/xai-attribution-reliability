# -*- coding: utf-8 -*-
"""
stage5_cell_B_retrain_med2_V2

Goals:
1. Use the actual configuration of the original training scripts (m_bilstm_dataset2 / m_lstm_dataset2),
   Retrain the 4 new seeds of Medical-2 / BiLSTM and Medical-2 / LSTM (8 models in total);
   J_V2 wrongly used factor=0.7 / sched_patience=15 / early_patience=50; this cell corrects it:
   - BiLSTM: factor=0.5, sched_patience=10, early_patience=60
   - LSTM:   factor=0.5, sched_patience=15, early_patience=60
2. seed=42 Read the original model directly (no retraining);
3. Recompute Macro-F1, Accuracy, Prediction Agreement and IG stability;
4. IG Settings are identical to J_V2: the same 500 test samples, targets fixed to the seed42 predictions,
   baseline is the training-set mean trajectory, 64 steps;
5. If the J_V2 summary file is found, a comparison table of J_V2 and this cell is written.

Notes:
- New seeds save the weights of the best validation F1 (explicit clone);
- The original seed42 checkpoints hold the early-stopping-epoch weights (state_dict reference issue in the original code); they are not changed here;
- New models and results are written only to ./stage5/cell_B_retrain_med2_V2/.
"""

import copy
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients

warnings.filterwarnings("ignore")

# --------------------
# 0. Global settings
# --------------------
OUTPUT_DIR = Path("./stage5/cell_B_retrain_med2_V2")
MODEL_OUT_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type != "cuda":
    print("Warning: no GPU detected; training 8 models on CPU will be very slow")


def gpu_self_test():
    """Try cuDNN first; if cuDNN cannot be initialized, disable cuDNN and continue GPU training with PyTorch's native CUDA RNN kernels."""
    if DEVICE.type != "cuda":
        return
    print(f"CUDA Version: {torch.version.cuda}, cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        m = nn.LSTM(4, 8, batch_first=True).to(DEVICE)
        m(torch.zeros(2, 5, 4, device=DEVICE))
        del m
        torch.cuda.empty_cache()
        print("GPU self-check: cuDNN available")
        return
    except RuntimeError as e:
        print(f"cuDNN unavailable: {e}")
        print("Disabling cuDNN and using the native GPU RNN kernels instead (slower, results valid)")
    torch.backends.cudnn.enabled = False
    try:
        m = nn.LSTM(4, 8, batch_first=True, bidirectional=True).to(DEVICE)
        m(torch.zeros(2, 5, 4, device=DEVICE))
        del m
        torch.cuda.empty_cache()
        print("GPU self-check: native CUDA kernels available")
    except RuntimeError as e:
        raise RuntimeError(f"GPU still unavailable after disabling cuDNN; please check the CUDA environment. Original error: {e}")


SEEDS = [42, 123, 777, 0, 2024]
PRIMARY_SEED = 42

SEQ_LEN = 60
IG_SAMPLES = 500
IG_STEPS = 64
IG_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
MAX_EPOCHS = 400

# If models already trained in this cell's folder exist, reuse them (allows resuming after an interruption); J_V2 models are not reused
REUSE_EXISTING_NEW_MODELS = True

RUN_COMBOS = [("Medical-2", "BiLSTM"), ("Medical-2", "LSTM")]

print(f"Random seed: {SEEDS}")
print(f"Combinations to run: {RUN_COMBOS}")
print(f"IG Number of samples: {IG_SAMPLES}, IG Steps: {IG_STEPS}")
print("--------------------")

# --------------------
# 1. Paths: locate automatically from the current working folder and its parent folders XAI_cn
# --------------------
CN_ROOT_MANUAL = None   # If not found, fill in the full path, for example "/home/i/2026_0712/stock_i/XAI_cn"

CWD = Path.cwd().resolve()
print(f"Current working folder: {CWD}")
SEARCH_BASES = [CWD, *CWD.parents, Path("/home/i/2026_0712/stock_i"), Path("/home/i")]


def find_child_dir(names, marker=None):
    names_low = {n.lower() for n in names}
    for base in SEARCH_BASES:
        cands = [base] if base.name.lower() in names_low else []
        if base.is_dir():
            try:
                cands += [c for c in base.iterdir() if c.is_dir() and c.name.lower() in names_low]
            except PermissionError:
                pass
        for c in cands:
            if marker is None or (c / marker).exists():
                return c
    return None


if CN_ROOT_MANUAL is not None:
    CN_ROOT = Path(CN_ROOT_MANUAL)
else:
    CN_ROOT = find_child_dir(["xai_cn"], "datasets/processed/dataset2_AB_grade")
if CN_ROOT is None or not (CN_ROOT / "datasets/processed/dataset2_AB_grade").is_dir():
    raise FileNotFoundError("Chinese data root folder not found; please fill in CN_ROOT_MANUAL")
print(f"Chinese data root folder: {CN_ROOT}")

# J_V2 results (for comparison only, may be missing)
J_V2_DIR = find_child_dir(["xai_cell_J_V2"], "multiseed_summary.csv")
print(f"J_V2 Results folder: {J_V2_DIR if J_V2_DIR else 'not found (comparison table skipped)'}")

DATASET_CFG = {
    "Medical-2": {
        "data_dir": CN_ROOT / "datasets/processed/dataset2_AB_grade",
        "filter_train_2010": True,
        "features": [
            "high_low_range", "turnover_rate", "amount", "bb_width", "roc_10",
            "roc_5", "close_to_ma20", "close_to_ma5", "close_open_range", "atr",
            "volume_ratio", "roc_20", "dayofyear_cos", "year"
        ],
    },
}

PRIMARY_MODEL_PATH = {
    ("Medical-2", "BiLSTM"): CN_ROOT / "models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_14_seed42.pt",
    ("Medical-2", "LSTM"): CN_ROOT / "models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_14_seed42.pt",
}

# --------------------
# 2. Original training configuration (checked item by item against train_model_with_weight and main in the original scripts)
# --------------------
TRAIN_CFG = {
    ("Medical-2", "BiLSTM"): {
        # m_bilstm_dataset2: hidden=128, layers=2, dropout=0.1, lr=1e-4,
        # weights=[1.15,1.0,1.15], batch=512, AdamW wd=3e-4,
        # ReduceLROnPlateau(factor=0.5, patience=10), early patience=60
        "hidden": 128, "layers": 2, "dropout": 0.10, "lr": 1e-4,
        "weights": [1.15, 1.0, 1.15], "batch": 512,
        "weight_decay": 3e-4, "sched_factor": 0.5, "sched_patience": 10, "early_patience": 60,
    },
    ("Medical-2", "LSTM"): {
        # m_lstm_dataset2: hidden=128, layers=2, dropout=0.0, lr=1e-4,
        # weights=[1.3,1.0,1.3], batch=512, AdamW wd=3e-4,
        # ReduceLROnPlateau(factor=0.5, patience=15), early patience=60
        "hidden": 128, "layers": 2, "dropout": 0.00, "lr": 1e-4,
        "weights": [1.3, 1.0, 1.3], "batch": 512,
        "weight_decay": 3e-4, "sched_factor": 0.5, "sched_patience": 15, "early_patience": 60,
    },
}


# --------------------
# 3. Random seed
# --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------
# 4. Data processing
# --------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data_raw(data_dir):
    required = [
        "train_X.csv", "val_X.csv", "test_X.csv",
        "train_y.csv", "val_y.csv", "test_y.csv",
    ]
    missing = [x for x in required if not (data_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Data missing: {data_dir} | {missing}")

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


def calculate_thresholds(y_df):
    arr = y_df["pct_chg"].dropna()
    return np.percentile(arr, 33.33), np.percentile(arr, 66.67)


def build_sequence_data(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
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
        on=["stock_code", "trade_date"],
        how="inner",
    )

    all_seqs = []
    all_labels = []

    for _, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue

        g = g.sort_values("trade_date")
        feats = g[feature_cols].values
        dates = g["trade_date"].values
        labels = g["label"].values

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

    if not all_seqs:
        raise RuntimeError("No samples after sequence construction")

    return (
        np.stack(all_seqs).astype(np.float32),
        np.asarray(all_labels, dtype=np.int64),
    )


def prepare_dataset(dataset_name):
    cfg = DATASET_CFG[dataset_name]
    data_dir = cfg["data_dir"]
    features = cfg["features"]

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)

    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)

    q33, q67 = calculate_thresholds(y_tr)

    X_train, y_train = build_sequence_data(X_tr, y_tr, features, q33, q67, SEQ_LEN)
    X_val, y_val = build_sequence_data(X_va, y_va, features, q33, q67, SEQ_LEN)
    X_test, y_test = build_sequence_data(X_te, y_te, features, q33, q67, SEQ_LEN)

    print(
        f"{dataset_name}: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )

    for name, arr in [
        ("train", X_train), ("val", X_val), ("test", X_test)
    ]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"{name}: NaN={n_nan}, Inf={n_inf}")
        if n_nan > 0 or n_inf > 0:
            raise ValueError(f"{dataset_name}-{name}: exists NaN/Inf")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
    }


# --------------------
# 5. Model
# --------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
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
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
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
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
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


def build_model(model_type, input_dim, cfg):
    if model_type == "GRU":
        return GRUClassifier(
            input_dim, cfg["hidden"], cfg["layers"], cfg["dropout"]
        )
    if model_type == "LSTM":
        return LSTMClassifier(
            input_dim, cfg["hidden"], cfg["layers"], cfg["dropout"]
        )
    if model_type == "BiLSTM":
        return BiLSTMClassifier(
            input_dim, cfg["hidden"], cfg["layers"], cfg["dropout"]
        )
    raise ValueError(model_type)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state.items()
    }


def load_state_strict(model, path):
    if not path.exists():
        raise FileNotFoundError(f"Model does not exist: {path}")
    state = torch.load(path, map_location="cpu")
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    return model


# --------------------
# 7. Training and evaluation
# --------------------
def predict_arrays(model, X, batch_size=512):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(
                X[s:s + batch_size].astype(np.float32)
            ).to(DEVICE)

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
        "pred": pred,
        "prob": prob,
    }


def train_one_seed(
    dataset_name,
    model_type,
    seed,
    data,
    cfg,
):
    set_seed(seed)

    train_ds = SequenceDataset(data["X_train"], data["y_train"])
    val_ds = SequenceDataset(data["X_val"], data["y_val"])

    # An independent generator makes the shuffle controlled by the seed
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch"],
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        model_type,
        data["X_train"].shape[2],
        cfg,
    ).to(DEVICE)

    class_weights = torch.tensor(
        cfg["weights"], dtype=torch.float32, device=DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["sched_factor"],
        patience=cfg["sched_patience"],
    )

    best_f1 = 0.0
    best_state = None
    counter = 0
    history = []

    print(
        f"Training configuration: hidden={cfg['hidden']}, layers={cfg['layers']}, "
        f"dropout={cfg['dropout']}, lr={cfg['lr']:.1e}, "
        f"batch={cfg['batch']}, weights={cfg['weights']}"
    )

    for epoch in range(MAX_EPOCHS):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))

        val_result = evaluate_arrays(
            model,
            data["X_val"],
            data["y_val"],
        )
        val_f1 = float(val_result["macro_f1"])
        val_acc = float(val_result["accuracy"])
        train_loss = float(np.mean(losses))
        curr_lr = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "lr": curr_lr,
        })

        scheduler.step(val_f1)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1:03d}: "
                f"Loss={train_loss:.4f}, "
                f"ValF1={val_f1:.4f}, "
                f"LR={curr_lr:.2e}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1

            # Keeps the intention of the original training logic to "save by best validation F1";
            # Clone explicitly here so that the state_dict reference does not keep changing in later epochs.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            counter = 0
        else:
            counter += 1

        if counter >= cfg["early_patience"]:
            print(f"Early stopping: epoch={epoch+1}")
            break

    if best_state is None:
        raise RuntimeError(
            f"{dataset_name}-{model_type}-seed{seed}: not saved best_state"
        )

    model.load_state_dict(best_state, strict=True)
    model = model.to(DEVICE).eval()

    return model, history, best_f1


# --------------------
# 8. IG: fixed samples + fixed seed42 target
# --------------------
def choose_fixed_samples(X_test, n_samples=500):
    rng = np.random.default_rng(PRIMARY_SEED)
    n = min(n_samples, len(X_test))
    return np.sort(rng.choice(len(X_test), size=n, replace=False))


def ig_attribution_fixed_targets(
    model,
    X_test,
    X_train,
    sample_idx,
    fixed_targets,
    n_steps=64,
    batch_size=64,
):
    orig_cudnn = torch.backends.cudnn.enabled  # Restore the state from before the call afterwards
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
                    dtype=np.int64,
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


def summarize_ig(attr, feature_names):
    timestep = attr.mean(axis=(0, 2))
    feature = attr.mean(axis=(0, 1))

    timestep_share = timestep / (timestep.sum() + 1e-12)
    feature_share = feature / (feature.sum() + 1e-12)

    time_df = pd.DataFrame({
        "Timestep": np.arange(1, SEQ_LEN + 1),
        "IG_Importance": timestep,
        "IG_Share": timestep_share,
    })

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "IG_Importance": feature,
        "IG_Share": feature_share,
    })

    return time_df, feat_df


def safe_spearman(x, y):
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def topk_overlap(a, b, k=5):
    aa = set(a.nlargest(k, "IG_Importance")["Feature"].tolist())
    bb = set(b.nlargest(k, "IG_Importance")["Feature"].tolist())
    return len(aa & bb) / float(k)


# --------------------
# 9. Output helpers
# --------------------
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def audit_primary_model_architecture(model, cfg, model_type):
    if model_type == "GRU":
        rnn = model.gru
    else:
        rnn = model.lstm

    if rnn.hidden_size != cfg["hidden"]:
        raise AssertionError(
            f"hidden mismatch: checkpoint={rnn.hidden_size}, cfg={cfg['hidden']}"
        )
    if rnn.num_layers != cfg["layers"]:
        raise AssertionError(
            f"layers mismatch: checkpoint={rnn.num_layers}, cfg={cfg['layers']}"
        )


# --------------------
# 10. Main flow
# --------------------
def main():
    print("Task: Medical-2 BiLSTM/LSTM retraining with the original configuration + IG stability")
    print("--------------------")

    gpu_self_test()

    all_rows = []
    feature_rows = []
    timestep_rows = []
    reference_cache = {}

    datasets = sorted({d for d, _ in RUN_COMBOS}, key=[d for d, _ in RUN_COMBOS].index)

    for dataset_name in datasets:
        print("--------------------")
        print(f"Dataset: {dataset_name}")
        print("Step 1: data preparation and audit")

        data = prepare_dataset(dataset_name)
        sample_idx = choose_fixed_samples(data["X_test"], IG_SAMPLES)

        print(f"Number of fixed IG samples: {len(sample_idx)}")
        print(f"Number of fields: {len(data['features'])}")
        print("Data audit: passed")

        for model_type in [m for d, m in RUN_COMBOS if d == dataset_name]:

            print("--------------------")
            print(f"{dataset_name} / {model_type}")

            cfg = TRAIN_CFG[(dataset_name, model_type)]

            # Load seed42 first to get the fixed targets and the reference IG
            print("Step 2: re-check the original seed42 model")

            primary_model = build_model(
                model_type,
                data["X_train"].shape[2],
                cfg,
            )
            primary_path = PRIMARY_MODEL_PATH[(dataset_name, model_type)]
            primary_model = load_state_strict(
                primary_model,
                primary_path,
            ).to(DEVICE).eval()

            audit_primary_model_architecture(
                primary_model,
                cfg,
                model_type,
            )

            primary_eval = evaluate_arrays(
                primary_model,
                data["X_test"],
                data["y_test"],
            )

            fixed_targets, _ = predict_arrays(
                primary_model,
                data["X_test"][sample_idx],
                EVAL_BATCH_SIZE,
            )

            print(
                f"seed42 Check: Macro-F1={primary_eval['macro_f1']:.6f}, "
                f"Accuracy={primary_eval['accuracy']:.6f}"
            )
            print(f"seed42 Model: {primary_path}")

            print("Step 3: seed42 reference IG")
            primary_attr = ig_attribution_fixed_targets(
                primary_model,
                data["X_test"],
                data["X_train"],
                sample_idx,
                fixed_targets,
                IG_STEPS,
                IG_BATCH_SIZE,
            )

            primary_time_df, primary_feat_df = summarize_ig(
                primary_attr,
                data["features"],
            )

            reference_cache[(dataset_name, model_type)] = {
                "pred": primary_eval["pred"],
                "time": primary_time_df.copy(),
                "feat": primary_feat_df.copy(),
                "fixed_targets": fixed_targets.copy(),
            }

            # Process all 5 together seed
            for seed in SEEDS:
                print("--------------------")
                print(f"Seed={seed}")

                if seed == PRIMARY_SEED:
                    model = primary_model
                    history = []
                    best_val_f1 = np.nan
                    eval_result = primary_eval
                    time_df = primary_time_df.copy()
                    feat_df = primary_feat_df.copy()
                    epochs_run = np.nan
                    source = "original_seed42"
                else:
                    model_file = (
                        MODEL_OUT_DIR
                        / f"{dataset_name}_{model_type}_seed{seed}.pt"
                    )
                    history_file = (
                        OUTPUT_DIR
                        / f"history_{dataset_name}_{model_type}_seed{seed}.json"
                    )

                    # Priority 1: models already trained by J_V2 itself
                    reuse_model_file = None
                    reuse_history_file = None
                    reuse_source = None

                    if REUSE_EXISTING_NEW_MODELS and model_file.exists():
                        reuse_model_file = model_file
                        reuse_history_file = history_file
                        reuse_source = "reused_this_cell"

                    if reuse_model_file is not None:
                        print(f"Reusing existing model: {reuse_model_file}")

                        model = build_model(
                            model_type,
                            data["X_train"].shape[2],
                            cfg,
                        )
                        model = load_state_strict(
                            model,
                            reuse_model_file,
                        ).to(DEVICE).eval()

                        if (
                            reuse_history_file is not None
                            and reuse_history_file.exists()
                        ):
                            with open(
                                reuse_history_file,
                                "r",
                                encoding="utf-8",
                            ) as f:
                                history = json.load(f)
                        else:
                            history = []

                        best_val_f1 = (
                            max(x["val_f1"] for x in history)
                            if history else np.nan
                        )
                        epochs_run = (
                            len(history) if history else np.nan
                        )
                        source = reuse_source

                    else:
                        print("Step 4: train new-seed model")

                        model, history, best_val_f1 = train_one_seed(
                            dataset_name,
                            model_type,
                            seed,
                            data,
                            cfg,
                        )
                        epochs_run = len(history)

                        torch.save(
                            model.state_dict(),
                            model_file,
                        )
                        save_json(
                            history,
                            history_file,
                        )

                        print(f"Model saving: {model_file}")
                        source = "retrained_original_config"

                    eval_result = evaluate_arrays(
                        model,
                        data["X_test"],
                        data["y_test"],
                    )

                    print("Step 5: fixed target IG")
                    attr = ig_attribution_fixed_targets(
                        model,
                        data["X_test"],
                        data["X_train"],
                        sample_idx,
                        fixed_targets,
                        IG_STEPS,
                        IG_BATCH_SIZE,
                    )

                    time_df, feat_df = summarize_ig(
                        attr,
                        data["features"],
                    )

                # compared with seed42
                ref = reference_cache[(dataset_name, model_type)]

                feat_rho, feat_p = safe_spearman(
                    ref["feat"]["IG_Importance"].values,
                    feat_df["IG_Importance"].values,
                )
                time_rho, time_p = safe_spearman(
                    ref["time"]["IG_Importance"].values,
                    time_df["IG_Importance"].values,
                )

                pred_agreement = float(
                    np.mean(
                        ref["pred"] == eval_result["pred"]
                    )
                )

                overlap5 = topk_overlap(
                    ref["feat"],
                    feat_df,
                    k=min(5, len(feat_df)),
                )

                last10 = float(
                    time_df.loc[
                        time_df["Timestep"].between(51, 60),
                        "IG_Share",
                    ].sum()
                )
                first10 = float(
                    time_df.loc[
                        time_df["Timestep"].between(1, 10),
                        "IG_Share",
                    ].sum()
                )
                ratio = last10 / max(first10, 1e-12)

                all_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_type,
                    "Seed": seed,
                    "Source": source,
                    "MacroF1": float(eval_result["macro_f1"]),
                    "Accuracy": float(eval_result["accuracy"]),
                    "Prediction_Agreement_vs_42": pred_agreement,
                    "Feature_Spearman_vs_42": feat_rho,
                    "Feature_p_vs_42": feat_p,
                    "Temporal_Spearman_vs_42": time_rho,
                    "Temporal_p_vs_42": time_p,
                    "Top5_Overlap_vs_42": overlap5,
                    "IG_Last10_Mass": last10,
                    "IG_First10_Mass": first10,
                    "IG_Last10_to_First10": ratio,
                    "Epochs_Run": epochs_run,
                    "Best_Val_MacroF1": best_val_f1,
                    "Hidden": cfg["hidden"],
                    "Layers": cfg["layers"],
                    "Dropout": cfg["dropout"],
                    "Learning_Rate": cfg["lr"],
                    "Sched_Factor": cfg["sched_factor"],
                    "Sched_Patience": cfg["sched_patience"],
                    "Early_Patience": cfg["early_patience"],
                    "cuDNN_Enabled": bool(torch.backends.cudnn.enabled),
                })

                tmp_feat = feat_df.copy()
                tmp_feat.insert(0, "Seed", seed)
                tmp_feat.insert(0, "Model", model_type)
                tmp_feat.insert(0, "Dataset", dataset_name)
                feature_rows.append(tmp_feat)

                tmp_time = time_df.copy()
                tmp_time.insert(0, "Seed", seed)
                tmp_time.insert(0, "Model", model_type)
                tmp_time.insert(0, "Dataset", dataset_name)
                timestep_rows.append(tmp_time)

                print(
                    f"Results: F1={eval_result['macro_f1']:.6f}, "
                    f"Acc={eval_result['accuracy']:.6f}"
                )
                print(
                    f"Stability: PredAgree={pred_agreement:.4f}, "
                    f"Featureρ={feat_rho:.4f}, "
                    f"Temporalρ={time_rho:.4f}"
                )
                print(
                    f"Recency: Last10={last10:.4f}, "
                    f"Last10/First10={ratio:.4f}, "
                    f"Top5Overlap={overlap5:.2f}"
                )

                if seed != PRIMARY_SEED:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del primary_model, primary_attr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------------------
    # 11. Summary
    # --------------------
    print("--------------------")
    print("Step 6: global summary")

    detail_df = pd.DataFrame(all_rows)
    feature_df = pd.concat(feature_rows, ignore_index=True)
    timestep_df = pd.concat(timestep_rows, ignore_index=True)

    summary_rows = []

    for (dataset_name, model_type), g in detail_df.groupby(
        ["Dataset", "Model"],
        sort=False,
    ):
        alt = g[g["Seed"] != PRIMARY_SEED]

        summary_rows.append({
            "Dataset": dataset_name,
            "Model": model_type,
            "N_Seeds": int(len(g)),
            "MacroF1_Mean": g["MacroF1"].mean(),
            "MacroF1_SD": g["MacroF1"].std(ddof=1),
            "Accuracy_Mean": g["Accuracy"].mean(),
            "Accuracy_SD": g["Accuracy"].std(ddof=1),
            "Prediction_Agreement_vs_42_Mean_4Seeds":
                alt["Prediction_Agreement_vs_42"].mean(),
            "Prediction_Agreement_vs_42_Min_4Seeds":
                alt["Prediction_Agreement_vs_42"].min(),
            "Feature_Spearman_vs_42_Mean_4Seeds":
                alt["Feature_Spearman_vs_42"].mean(),
            "Feature_Spearman_vs_42_Min_4Seeds":
                alt["Feature_Spearman_vs_42"].min(),
            "Temporal_Spearman_vs_42_Mean_4Seeds":
                alt["Temporal_Spearman_vs_42"].mean(),
            "Temporal_Spearman_vs_42_Min_4Seeds":
                alt["Temporal_Spearman_vs_42"].min(),
            "Top5_Overlap_vs_42_Mean_4Seeds":
                alt["Top5_Overlap_vs_42"].mean(),
            "IG_Last10_Mass_Mean":
                g["IG_Last10_Mass"].mean(),
            "IG_Last10_Mass_SD":
                g["IG_Last10_Mass"].std(ddof=1),
            "All_Seeds_Last10_gt_First10":
                bool((g["IG_Last10_Mass"] > g["IG_First10_Mass"]).all()),
        })

    summary_df = pd.DataFrame(summary_rows)

    detail_path = OUTPUT_DIR / "multiseed_detail.csv"
    summary_path = OUTPUT_DIR / "multiseed_summary.csv"
    feature_path = OUTPUT_DIR / "ig_feature_all_seeds.csv"
    timestep_path = OUTPUT_DIR / "ig_timestep_all_seeds.csv"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    timestep_df.to_csv(timestep_path, index=False)

    print(f"Number of detail rows: {len(detail_df)}")
    print(f"Expected number of detail rows: {len(RUN_COMBOS) * len(SEEDS)}")
    print(f"Number of summary rows: {len(summary_df)}")
    print(f"Expected number of summary rows: {len(RUN_COMBOS)}")

    if len(detail_df) != len(RUN_COMBOS) * len(SEEDS):
        raise AssertionError(f"Abnormal number of detail rows: {len(detail_df)}")
    if len(summary_df) != len(RUN_COMBOS):
        raise AssertionError(f"Abnormal number of summary rows: {len(summary_df)}")

    print("--------------------")
    print("Core summary")
    show_cols = [
        "Dataset",
        "Model",
        "MacroF1_Mean",
        "MacroF1_SD",
        "Feature_Spearman_vs_42_Mean_4Seeds",
        "Feature_Spearman_vs_42_Min_4Seeds",
        "Temporal_Spearman_vs_42_Mean_4Seeds",
        "Temporal_Spearman_vs_42_Min_4Seeds",
        "IG_Last10_Mass_Mean",
        "All_Seeds_Last10_gt_First10",
    ]
    print(
        summary_df[show_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("--------------------")
    print("Audit")
    print(
        f"5-seed Full: "
        f"{bool((summary_df['N_Seeds'] == 5).all())}"
    )
    print(
        f"All combinations Last10>First10: "
        f"{bool(summary_df['All_Seeds_Last10_gt_First10'].all())}"
    )

    print("--------------------")
    print("J_V2(wrong configuration) vs this cell (original configuration)")
    compare_path = OUTPUT_DIR / "compare_J_V2_vs_B_V2.csv"
    if J_V2_DIR is not None:
        old = pd.read_csv(J_V2_DIR / "multiseed_summary.csv")
        keys = ["Dataset", "Model"]
        cmp_cols = [
            "MacroF1_Mean", "MacroF1_SD",
            "Prediction_Agreement_vs_42_Mean_4Seeds",
            "Feature_Spearman_vs_42_Mean_4Seeds", "Feature_Spearman_vs_42_Min_4Seeds",
            "Temporal_Spearman_vs_42_Mean_4Seeds", "Temporal_Spearman_vs_42_Min_4Seeds",
            "Top5_Overlap_vs_42_Mean_4Seeds", "IG_Last10_Mass_Mean",
            "All_Seeds_Last10_gt_First10",
        ]
        cmp_cols = [c for c in cmp_cols if c in old.columns]
        merged = summary_df[keys + cmp_cols].merge(
            old[keys + cmp_cols], on=keys, how="left", suffixes=("_B_V2", "_J_V2")
        )
        merged.to_csv(compare_path, index=False)
        for _, r in merged.iterrows():
            print(f"{r['Dataset']} / {r['Model']}")
            for c in cmp_cols:
                print(f"  {c}: J_V2={r[c + '_J_V2']}, B_V2={r[c + '_B_V2']}")
    else:
        print("J_V2 summary not found; skipped")

    print("--------------------")
    print("Output files")
    print(detail_path)
    print(summary_path)
    print(feature_path)
    print(timestep_path)
    if J_V2_DIR is not None:
        print(compare_path)
    print(MODEL_OUT_DIR)

    print("--------------------")
    print("All done")


if __name__ == "__main__":
    main()
