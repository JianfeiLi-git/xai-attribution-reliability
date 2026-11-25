"""
stage6_cell_J_data_multiseed_V1

Aims:
1. Five-seed stability analysis for 2 U.S. datasets x 3 RNNs: seeds 42, 123, 777, 0 and 2024;
2. Seed 42 uses the model selected in stage6_cell_C directly; the other 4 seeds are retrained with exactly the same training configuration,
   and, as in the original paper pipeline, new-seed models are saved at the epoch with the highest validation Macro-F1;
3. Report the test Macro-F1 and accuracy of each seed, and its prediction agreement with seed 42;
4. Compute IG on 500 fixed test samples, with the target class fixed to the seed-42 model's predicted class and the training-set mean trajectory as baseline;
5. Compare each seed's feature and temporal rankings with seed 42 using Spearman correlation, and report last-10-day attribution mass and top-5 feature overlap.

Methods are the same as the five-seed analysis in the original paper (seeds, IG sample size, steps and comparison metrics unchanged), with only the following handling:
- The training configuration is taken from the candidate selected in selected_models.csv of stage6_cell_C, using the full corresponding configuration rather than manual entry;
- cuDNN cannot be initialized on the server, so cuDNN is disabled throughout and PyTorch native CUDA RNN kernels are used;
- IG uses an inference copy with dropout=0 and the same architecture (dropout has no parameters, so the weights are identical),
  ensuring that between-layer RNN dropout is never active during IG, so that IG results are deterministic and reproducible;
- Progress and time are printed for every epoch; new-seed models already trained by this cell can be reused directly, so an interrupted run can be resumed.
"""

# ------------------------------ Settings (editable) ------------------------------
US_DATA_ROOT = "stage6/stage6_cell_A_data_preprocessing_V1"
US_MODEL_ROOT = "stage6/stage6_cell_C_data_training_V2"
OUTPUT_ROOT = "stage6/stage6_cell_J_data_multiseed_V1"
DATASETS = {"US-Med": "us_med", "US-Pow": "us_pow"}
MODELS = ["GRU", "BiLSTM", "LSTM"]
SEEDS = [42, 123, 777, 0, 2024]
PRIMARY_SEED = 42
SEQ_LEN = 60
IG_SAMPLES = 500
IG_STEPS = 64
IG_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
MAX_EPOCHS = 400
REUSE_EXISTING_NEW_MODELS = True   # reuse new-seed models already present in this cell's output directory

# All candidate configurations of stage6_cell_C (identical to cell C), selected by the source column in selected_models.csv
CANDIDATE_CONFIGS = {
    "Medical-1 GRU": {"hidden": 64, "layers": 2, "dropout": 0.1, "lr": 5e-5, "weights": [1.0, 1.0, 1.2],
                      "batch": 256, "sched_factor": 0.7, "sched_patience": 15, "early_patience": 50},
    "Power GRU": {"hidden": 128, "layers": 2, "dropout": 0.3, "lr": 5e-5, "weights": [1.1, 0.9, 1.2],
                  "batch": 512, "sched_factor": 0.5, "sched_patience": 10, "early_patience": 60},
    "Medical-1 LSTM": {"hidden": 64, "layers": 2, "dropout": 0.35, "lr": 1e-4, "weights": [1.0, 1.0, 1.15],
                       "batch": 256, "sched_factor": 0.7, "sched_patience": 15, "early_patience": 50},
    "Medical-2 LSTM": {"hidden": 128, "layers": 2, "dropout": 0.0, "lr": 1e-4, "weights": [1.3, 1.0, 1.3],
                       "batch": 512, "sched_factor": 0.5, "sched_patience": 15, "early_patience": 60},
    "Power LSTM": {"hidden": 96, "layers": 2, "dropout": 0.25, "lr": 1e-4, "weights": [1.2, 1.0, 1.35],
                   "batch": 512, "sched_factor": 0.5, "sched_patience": 15, "early_patience": 60},
    "Medical-1 BiLSTM": {"hidden": 64, "layers": 2, "dropout": 0.05, "lr": 1e-4, "weights": [1.1, 0.95, 1.25],
                         "batch": 256, "sched_factor": 0.7, "sched_patience": 15, "early_patience": 50},
    "Medical-2 BiLSTM": {"hidden": 128, "layers": 2, "dropout": 0.1, "lr": 1e-4, "weights": [1.15, 1.0, 1.15],
                         "batch": 512, "sched_factor": 0.5, "sched_patience": 10, "early_patience": 60},
    "Power BiLSTM": {"hidden": 128, "layers": 2, "dropout": 0.35, "lr": 5e-5, "weights": [1.1, 1.0, 1.25],
                     "batch": 512, "sched_factor": 0.5, "sched_patience": 10, "early_patience": 60},
}
WEIGHT_DECAY = 3e-4   # same as China and cell C training

# ------------------------------ Imports ------------------------------
import copy
import json
import time
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
T0 = time.time()

OUTPUT_DIR = Path(OUTPUT_ROOT)
MODEL_OUT_DIR = OUTPUT_DIR / "models"
SELECTED_CSV = Path(US_MODEL_ROOT) / "selected_models.csv"

# Rule: the first two lines print the input and output locations (relative paths only)
print(f"Input directories: ./{US_DATA_ROOT}/, ./{US_MODEL_ROOT}/")
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")

torch.backends.cudnn.enabled = False
print("cuDNN disabled (cuDNN cannot be initialized on the server); using PyTorch native CUDA RNN kernels")

# Check all input files before running
_need = [Path(US_DATA_ROOT) / "feature_config.json", SELECTED_CSV]
for _k in DATASETS.values():
    _need += [Path(US_DATA_ROOT) / _k / f"{_s}_{_xy}.csv" for _s in ["train", "val", "test"] for _xy in ["X", "y"]]
    _need += [Path(US_MODEL_ROOT) / _k / f"{_m.lower()}_seed42_selected.pth" for _m in MODELS]
_missing = [p for p in _need if not p.exists()]
if _missing:
    print("The following input files do not exist; check US_DATA_ROOT / US_MODEL_ROOT at the top of the code:")
    for _p in _missing:
        print("  ./" + _p.as_posix())
    raise FileNotFoundError(f"{len(_missing)} input files are missing")
print(f"Input file check passed ({len(_need)} files)")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Random seeds: {SEEDS}")
print(f"IG samples: {IG_SAMPLES}, IG integration steps: {IG_STEPS}")


def elapsed():
    # Return the elapsed time as a string
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


# Feature configuration
with open(Path(US_DATA_ROOT) / "feature_config.json", "r", encoding="utf-8") as _f:
    _feat_cfg = json.load(_f)
DATASET_CFG = {}
for _name, _key in DATASETS.items():
    _feats = _feat_cfg[_key]["selected_features"] + _feat_cfg[_key]["temporal_features"]
    DATASET_CFG[_name] = {"data_dir": Path(US_DATA_ROOT) / _key, "features": _feats}
    print(f"{_name}: {len(_feats)} features: {_feats}")

# Training configuration: read from the selection results of cell C
_sel = pd.read_csv(SELECTED_CSV)
_key_to_name = {v: k for k, v in DATASETS.items()}
TRAIN_CFG = {}
print("Training configurations (from the selection results of stage6_cell_C):")
for _, _r in _sel.iterrows():
    _name = _key_to_name[_r["dataset"]]
    _cfg = dict(CANDIDATE_CONFIGS[_r["source"]])
    _cfg["weight_decay"] = WEIGHT_DECAY
    # Check that the parameters recorded in selected_models.csv match the candidate configuration
    for _col in ["hidden", "layers", "dropout", "lr", "batch"]:
        if not np.isclose(float(_r[_col]), float(_cfg[_col])):
            raise ValueError(f"{_name}-{_r['arch']}: {_col} does not match the cell C record ({_r[_col]} vs {_cfg[_col]})")
    TRAIN_CFG[(_name, _r["arch"])] = _cfg
    print(f"  {_name}-{_r['arch']}: from {_r['source']}, hidden {_cfg['hidden']}, layers {_cfg['layers']}, "
          f"dropout {_cfg['dropout']}, lr {_cfg['lr']}, class weights {_cfg['weights']}, batch {_cfg['batch']}, "
          f"scheduler {_cfg['sched_factor']}/{_cfg['sched_patience']}, early-stopping patience {_cfg['early_patience']}")
for _name in DATASETS:
    for _m in MODELS:
        if (_name, _m) not in TRAIN_CFG:
            raise ValueError(f"{_name}-{_m} is missing from selected_models.csv")

PRIMARY_MODEL_PATH = {
    (name, m): Path(US_MODEL_ROOT) / key / f"{m.lower()}_seed42_selected.pth"
    for name, key in DATASETS.items() for m in MODELS
}


# ============================== Computation code below ==============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    for _, g in merged.groupby("stock_code"):
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
    if not all_seqs:
        raise RuntimeError("No samples after sequence construction")
    return np.stack(all_seqs).astype(np.float32), np.asarray(all_labels, dtype=np.int64)


def prepare_dataset(dataset_name):
    cfg = DATASET_CFG[dataset_name]
    data_dir = cfg["data_dir"]
    features = cfg["features"]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)
    q33, q67 = calculate_thresholds(y_tr)
    X_train, y_train = build_sequence_data(X_tr, y_tr, features, q33, q67, SEQ_LEN)
    X_val, y_val = build_sequence_data(X_va, y_va, features, q33, q67, SEQ_LEN)
    X_test, y_test = build_sequence_data(X_te, y_te, features, q33, q67, SEQ_LEN)
    print(f"{dataset_name}: train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
    for name, arr in [("Train", X_train), ("Validation", X_val), ("Test", X_test)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"{name} set: NaN {n_nan}, Inf {n_inf}")
        if n_nan > 0 or n_inf > 0:
            raise ValueError(f"{dataset_name}-{name} set contains NaN/Inf")
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test, "features": features}


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


def build_model(model_type, input_dim, cfg, dropout=None):
    # dropout=None uses the training configuration; dropout=0.0 is used to build the inference copy (same weights)
    d = cfg["dropout"] if dropout is None else dropout
    cls = {"GRU": GRUClassifier, "LSTM": LSTMClassifier, "BiLSTM": BiLSTMClassifier}[model_type]
    return cls(input_dim, cfg["hidden"], cfg["layers"], d)


def inference_copy(model, model_type, input_dim, cfg):
    # Build a copy with dropout=0 and the same architecture and copy the weights, for evaluation and IG, ensuring no randomness
    m = build_model(model_type, input_dim, cfg, dropout=0.0)
    m.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    return m.to(DEVICE).eval()


def load_state_strict(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def audit_architecture(model, cfg, model_type):
    rnn = model.gru if model_type == "GRU" else model.lstm
    if rnn.hidden_size != cfg["hidden"]:
        raise AssertionError(f"hidden mismatch: model {rnn.hidden_size}, config {cfg['hidden']}")
    if rnn.num_layers != cfg["layers"]:
        raise AssertionError(f"layers mismatch: model {rnn.num_layers}, config {cfg['layers']}")


def predict_arrays(model, X, batch_size=512):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size].astype(np.float32)).to(DEVICE)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_arrays(model, X, y):
    pred, prob = predict_arrays(model, X, EVAL_BATCH_SIZE)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "pred": pred, "prob": prob}


# ------------------------------ New-seed training (same as the five-seed analysis in the original paper; only per-epoch printing added) ------------------------------
def train_one_seed(dataset_name, model_type, seed, data, cfg):
    set_seed(seed)
    train_ds = SequenceDataset(data["X_train"], data["y_train"])
    # Independent random generator, so that data shuffling depends only on the seed
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, generator=g, num_workers=0)
    model = build_model(model_type, data["X_train"].shape[2], cfg).to(DEVICE)
    class_weights = torch.tensor(cfg["weights"], dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["sched_factor"], patience=cfg["sched_patience"])

    best_f1 = 0.0
    best_epoch = 0
    best_state = None
    counter = 0
    history = []
    t_train = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_result = evaluate_arrays(model, data["X_val"], data["y_val"])
        val_f1 = float(val_result["macro_f1"])
        val_acc = float(val_result["accuracy"])
        train_loss = float(np.mean(losses))
        curr_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_f1": val_f1,
                        "val_acc": val_acc, "lr": curr_lr})
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            # New-seed models are saved at the epoch with the highest validation Macro-F1 (explicit copy, unaffected by later training)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        per_ep = (time.time() - t_train) / (epoch + 1)
        print(f"    [{dataset_name}-{model_type}-seed{seed}] Epoch {epoch + 1:3d} | loss {train_loss:.4f} | "
              f"val F1 {val_f1:.4f} acc {val_acc:.4f} | lr {curr_lr:.2e} | "
              f"best F1 {best_f1:.4f} (epoch {best_epoch}) | early-stop counter {counter}/{cfg['early_patience']} | "
              f"{per_ep:.1f}s per epoch | early stopping at most {per_ep * (cfg['early_patience'] - counter) / 60:.1f}m more | total {elapsed()}")
        if counter >= cfg["early_patience"]:
            print(f"    Early stopping triggered at epoch {epoch + 1}; loading the weights of epoch {best_epoch} (highest val F1)")
            break
    if best_state is None:
        raise RuntimeError(f"{dataset_name}-{model_type}-seed{seed}: no best weights were saved")
    model.load_state_dict(best_state, strict=True)
    return model.to(DEVICE).eval(), history, best_f1


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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================== Main ==============================
all_rows, feature_rows, timestep_rows = [], [], []
n_combo = len(DATASETS) * len(MODELS)
combo_i = 0
for dataset_name in DATASETS:
    print("\n" + "#" * 100)
    print(f"Dataset: {dataset_name}, total {elapsed()}")
    print("#" * 100)
    print("Step 1: data preparation and checks")
    data = prepare_dataset(dataset_name)
    sample_idx = choose_fixed_samples(data["X_test"], IG_SAMPLES)
    input_dim = data["X_train"].shape[2]
    print(f"IG fixed samples: {len(sample_idx)}, features: {len(data['features'])}")
    print("Data check: passed")

    for model_type in MODELS:
        combo_i += 1
        cfg = TRAIN_CFG[(dataset_name, model_type)]
        print("\n" + "=" * 100)
        print(f"Combination {combo_i}/{n_combo}: {dataset_name} / {model_type}, total {elapsed()}")
        print("=" * 100)

        print("Step 2: load and check the seed-42 model")
        primary_raw = build_model(model_type, input_dim, cfg)
        primary_raw = load_state_strict(primary_raw, PRIMARY_MODEL_PATH[(dataset_name, model_type)])
        audit_architecture(primary_raw, cfg, model_type)
        primary_model = inference_copy(primary_raw, model_type, input_dim, cfg)
        primary_eval = evaluate_arrays(primary_model, data["X_test"], data["y_test"])
        fixed_targets, _ = predict_arrays(primary_model, data["X_test"][sample_idx], EVAL_BATCH_SIZE)
        print(f"Seed-42 model: ./{PRIMARY_MODEL_PATH[(dataset_name, model_type)].as_posix()}")
        print(f"Seed-42 check: test Macro-F1 {primary_eval['macro_f1']:.6f}, accuracy {primary_eval['accuracy']:.6f}")
        print(f"Fixed target-class distribution: {pd.Series(fixed_targets).value_counts().sort_index().to_dict()}")

        print("Step 3: seed-42 reference IG")
        t = time.time()
        primary_attr = ig_attribution_fixed_targets(primary_model, data["X_test"], data["X_train"], sample_idx,
                                                    fixed_targets, IG_STEPS, IG_BATCH_SIZE)
        ref_time, ref_feat = summarize_ig(primary_attr, data["features"])
        print(f"Seed-42 IG done ({time.time() - t:.1f}s); top 5 features: "
              f"{ref_feat.nlargest(5, 'IG_Importance')['Feature'].tolist()}")

        for s_i, seed in enumerate(SEEDS, 1):
            print("-" * 60)
            print(f"{dataset_name} / {model_type} / seed {seed} ({s_i}/{len(SEEDS)}), total {elapsed()}")
            if seed == PRIMARY_SEED:
                eval_result, time_df, feat_df = primary_eval, ref_time.copy(), ref_feat.copy()
                history, best_val_f1, epochs_run, source = [], np.nan, np.nan, "stage6_cell_C_selected_seed42"
            else:
                model_file = MODEL_OUT_DIR / f"{dataset_name}_{model_type}_seed{seed}.pt"
                history_file = OUTPUT_DIR / f"history_{dataset_name}_{model_type}_seed{seed}.json"
                if REUSE_EXISTING_NEW_MODELS and model_file.exists():
                    print(f"Reusing the model already trained by this cell: ./{model_file.as_posix()}")
                    raw = load_state_strict(build_model(model_type, input_dim, cfg), model_file)
                    history = json.load(open(history_file, "r", encoding="utf-8")) if history_file.exists() else []
                    best_val_f1 = max(x["val_f1"] for x in history) if history else np.nan
                    epochs_run = len(history) if history else np.nan
                    source = "reused_this_cell"
                else:
                    print("Step 4: train the new-seed model")
                    raw, history, best_val_f1 = train_one_seed(dataset_name, model_type, seed, data, cfg)
                    epochs_run = len(history)
                    torch.save(raw.state_dict(), model_file)
                    save_json(history, history_file)
                    print(f"Model saved: ./{model_file.as_posix()}; {epochs_run} epochs trained, best val F1 {best_val_f1:.4f}")
                    source = "trained_new_seed"
                model = inference_copy(raw, model_type, input_dim, cfg)
                eval_result = evaluate_arrays(model, data["X_test"], data["y_test"])
                print("Step 5: fixed-target IG")
                t = time.time()
                attr = ig_attribution_fixed_targets(model, data["X_test"], data["X_train"], sample_idx,
                                                    fixed_targets, IG_STEPS, IG_BATCH_SIZE)
                time_df, feat_df = summarize_ig(attr, data["features"])
                print(f"IG done ({time.time() - t:.1f}s)")
                del model, raw, attr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            feat_rho, feat_p = safe_spearman(ref_feat["IG_Importance"].values, feat_df["IG_Importance"].values)
            time_rho, time_p = safe_spearman(ref_time["IG_Importance"].values, time_df["IG_Importance"].values)
            pred_agreement = float(np.mean(primary_eval["pred"] == eval_result["pred"]))
            overlap5 = topk_overlap(ref_feat, feat_df, k=min(5, len(feat_df)))
            last10 = float(time_df.loc[time_df["Timestep"].between(51, 60), "IG_Share"].sum())
            first10 = float(time_df.loc[time_df["Timestep"].between(1, 10), "IG_Share"].sum())
            ratio = last10 / max(first10, 1e-12)
            all_rows.append({
                "Dataset": dataset_name, "Model": model_type, "Seed": seed, "Source": source,
                "MacroF1": float(eval_result["macro_f1"]), "Accuracy": float(eval_result["accuracy"]),
                "Prediction_Agreement_vs_42": pred_agreement,
                "Feature_Spearman_vs_42": feat_rho, "Feature_p_vs_42": feat_p,
                "Temporal_Spearman_vs_42": time_rho, "Temporal_p_vs_42": time_p,
                "Top5_Overlap_vs_42": overlap5, "IG_Last10_Mass": last10, "IG_First10_Mass": first10,
                "IG_Last10_to_First10": ratio, "Epochs_Run": epochs_run, "Best_Val_MacroF1": best_val_f1,
                "Hidden": cfg["hidden"], "Layers": cfg["layers"], "Dropout": cfg["dropout"],
                "Learning_Rate": cfg["lr"]})
            for rows, df in [(feature_rows, feat_df), (timestep_rows, time_df)]:
                tmp = df.copy()
                tmp.insert(0, "Seed", seed)
                tmp.insert(0, "Model", model_type)
                tmp.insert(0, "Dataset", dataset_name)
                rows.append(tmp)
            print(f"Result: test Macro-F1 {eval_result['macro_f1']:.6f}, accuracy {eval_result['accuracy']:.6f}")
            print(f"Stability: prediction agreement {pred_agreement:.4f}, feature ρ {feat_rho:.4f}, temporal ρ {time_rho:.4f}")
            print(f"Recency: last 10 days {last10:.4f}, first 10 days {first10:.4f}, ratio {ratio:.4f}, top-5 feature overlap {overlap5:.2f}")

        del primary_model, primary_raw, primary_attr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ------------------------------ Summary ------------------------------
print("\n" + "=" * 100)
print("Step 6: overall summary")
detail_df = pd.DataFrame(all_rows)
feature_df = pd.concat(feature_rows, ignore_index=True)
timestep_df = pd.concat(timestep_rows, ignore_index=True)
summary_rows = []
for (dataset_name, model_type), g in detail_df.groupby(["Dataset", "Model"], sort=False):
    alt = g[g["Seed"] != PRIMARY_SEED]
    summary_rows.append({
        "Dataset": dataset_name, "Model": model_type, "N_Seeds": int(len(g)),
        "MacroF1_Mean": g["MacroF1"].mean(), "MacroF1_SD": g["MacroF1"].std(ddof=1),
        "Accuracy_Mean": g["Accuracy"].mean(), "Accuracy_SD": g["Accuracy"].std(ddof=1),
        "Prediction_Agreement_vs_42_Mean_4Seeds": alt["Prediction_Agreement_vs_42"].mean(),
        "Prediction_Agreement_vs_42_Min_4Seeds": alt["Prediction_Agreement_vs_42"].min(),
        "Feature_Spearman_vs_42_Mean_4Seeds": alt["Feature_Spearman_vs_42"].mean(),
        "Feature_Spearman_vs_42_Min_4Seeds": alt["Feature_Spearman_vs_42"].min(),
        "Temporal_Spearman_vs_42_Mean_4Seeds": alt["Temporal_Spearman_vs_42"].mean(),
        "Temporal_Spearman_vs_42_Min_4Seeds": alt["Temporal_Spearman_vs_42"].min(),
        "Top5_Overlap_vs_42_Mean_4Seeds": alt["Top5_Overlap_vs_42"].mean(),
        "IG_Last10_Mass_Mean": g["IG_Last10_Mass"].mean(), "IG_Last10_Mass_SD": g["IG_Last10_Mass"].std(ddof=1),
        "All_Seeds_Last10_gt_First10": bool((g["IG_Last10_Mass"] > g["IG_First10_Mass"]).all())})
summary_df = pd.DataFrame(summary_rows)

detail_df.to_csv(OUTPUT_DIR / "multiseed_detail.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "multiseed_summary.csv", index=False)
feature_df.to_csv(OUTPUT_DIR / "ig_feature_all_seeds.csv", index=False)
timestep_df.to_csv(OUTPUT_DIR / "ig_timestep_all_seeds.csv", index=False)

n_expect_detail = len(DATASETS) * len(MODELS) * len(SEEDS)
n_expect_summary = len(DATASETS) * len(MODELS)
print(f"Detail rows: {len(detail_df)} (expected {n_expect_detail}); summary rows: {len(summary_df)} (expected {n_expect_summary})")
if len(detail_df) != n_expect_detail or len(summary_df) != n_expect_summary:
    raise AssertionError("Number of result rows differs from expected")

print("\nCore summary (new seeds relative to seed 42):")
show_cols = ["Dataset", "Model", "MacroF1_Mean", "MacroF1_SD",
             "Feature_Spearman_vs_42_Mean_4Seeds", "Feature_Spearman_vs_42_Min_4Seeds",
             "Temporal_Spearman_vs_42_Mean_4Seeds", "Temporal_Spearman_vs_42_Min_4Seeds",
             "Prediction_Agreement_vs_42_Mean_4Seeds", "Top5_Overlap_vs_42_Mean_4Seeds",
             "IG_Last10_Mass_Mean", "All_Seeds_Last10_gt_First10"]
print(summary_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPer-seed details:")
print(detail_df[["Dataset", "Model", "Seed", "MacroF1", "Prediction_Agreement_vs_42", "Feature_Spearman_vs_42",
                 "Temporal_Spearman_vs_42", "IG_Last10_to_First10", "Epochs_Run"]].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
print("\nChecks:")
print(f"All 5 seeds present: {bool((summary_df['N_Seeds'] == len(SEEDS)).all())}")
print(f"Combinations with last 10 days > first 10 days for all seeds: {int(summary_df['All_Seeds_Last10_gt_First10'].sum())}/{len(summary_df)}")
print(f"\nOutput files: ./{OUTPUT_DIR.as_posix()}/multiseed_detail.csv, multiseed_summary.csv, ig_feature_all_seeds.csv, "
      f"ig_timestep_all_seeds.csv; models in ./{MODEL_OUT_DIR.as_posix()}/")
print(f"stage6_cell_J_data_multiseed_V1 finished, total time {elapsed()}")
