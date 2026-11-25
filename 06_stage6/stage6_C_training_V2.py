# ==============================================================================
# stage6_cell_C_data_training_V2
# Retrain the seed-42 primary models for the U.S. datasets (US-Med, US-Pow): GRU, LSTM, BiLSTM
# Parts identical to China (code copied line by line from the Chinese training scripts m_*/p_*):
#   - Sequence construction: 60-day window, 100-day suspension filter, label = pct_chg shifted forward by 1 day, three classes at the training-set 33.33/66.67 percentiles
#   - Model architectures: GRUClassifier, LSTMClassifier, BiLSTMClassifier
#   - Training function: AdamW (weight_decay=3e-4), weighted cross-entropy, gradient clipping 1.0, ReduceLROnPlateau, early stopping, at most 400 epochs
#   - Saving rule: same as the Chinese seed-42 models (best_state is stored by reference, so the weights actually saved are those of the epoch at which early stopping was triggered)
#   - Seed: 42 (set_seed identical to China, including cudnn.deterministic)
# Tuning: for each architecture, only configurations actually used by the same Chinese architecture are considered, and the one with the highest validation Macro-F1 is selected;
#         the test set is evaluated only for the selected configuration and is not used for selection
# Input: ./stage6/stage6_cell_A_data_preprocessing_V1/ (us_med/, us_pow/, feature_config.json)
#        Features = first-stage features (the second-stage probe model learned no signal on the U.S. data; see stage6_cell_B_V3)
# Output: ./stage6/stage6_cell_C_data_training_V2/
# V2 changes: check before running whether cuDNN can be initialized (V1 reported CUDNN_STATUS_NOT_INITIALIZED);
#             if not, switch automatically to PyTorch native CUDA RNN kernels (as in stage5_cell_B) and state this clearly in the console
# ==============================================================================

# ------------------------------ Settings (editable) ------------------------------
INPUT_DIR = "stage6/stage6_cell_A_data_preprocessing_V1"
OUTPUT_DIR = "stage6/stage6_cell_C_data_training_V2"
DATASETS = ["us_med", "us_pow"]          # datasets to train
ARCHS = ["GRU", "LSTM", "BiLSTM"]        # models to train
SEED = 42
MAX_EPOCHS = 400                         # same as China

# Configurations actually used by the Chinese seed-42 models (from the branches run at the time in the scripts in 03_model_training)
#   batch: 256 for Medical-1 (dataset1), 512 for Medical-2 and Power (dataset2)
#   sched_factor / sched_patience: ReduceLROnPlateau parameters; es_patience: early-stopping patience
#   The original training script for Chinese Medical-2 GRU is missing, so GRU has only 2 candidate configurations
CANDIDATE_CONFIGS = {
    "GRU": [
        {"source": "Medical-1 GRU", "hidden": 64, "layers": 2, "dropout": 0.1, "lr": 5e-5,
         "class_weights": [1.0, 1.0, 1.2], "batch": 256, "sched_factor": 0.7, "sched_patience": 15, "es_patience": 50},
        {"source": "Power GRU", "hidden": 128, "layers": 2, "dropout": 0.3, "lr": 5e-5,
         "class_weights": [1.1, 0.9, 1.2], "batch": 512, "sched_factor": 0.5, "sched_patience": 10, "es_patience": 60},
    ],
    "LSTM": [
        {"source": "Medical-1 LSTM", "hidden": 64, "layers": 2, "dropout": 0.35, "lr": 1e-4,
         "class_weights": [1.0, 1.0, 1.15], "batch": 256, "sched_factor": 0.7, "sched_patience": 15, "es_patience": 50},
        {"source": "Medical-2 LSTM", "hidden": 128, "layers": 2, "dropout": 0.0, "lr": 1e-4,
         "class_weights": [1.3, 1.0, 1.3], "batch": 512, "sched_factor": 0.5, "sched_patience": 15, "es_patience": 60},
        {"source": "Power LSTM", "hidden": 96, "layers": 2, "dropout": 0.25, "lr": 1e-4,
         "class_weights": [1.2, 1.0, 1.35], "batch": 512, "sched_factor": 0.5, "sched_patience": 15, "es_patience": 60},
    ],
    "BiLSTM": [
        {"source": "Medical-1 BiLSTM", "hidden": 64, "layers": 2, "dropout": 0.05, "lr": 1e-4,
         "class_weights": [1.1, 0.95, 1.25], "batch": 256, "sched_factor": 0.7, "sched_patience": 15, "es_patience": 50},
        {"source": "Medical-2 BiLSTM", "hidden": 128, "layers": 2, "dropout": 0.1, "lr": 1e-4,
         "class_weights": [1.15, 1.0, 1.15], "batch": 512, "sched_factor": 0.5, "sched_patience": 10, "es_patience": 60},
        {"source": "Power BiLSTM", "hidden": 128, "layers": 2, "dropout": 0.35, "lr": 5e-5,
         "class_weights": [1.1, 1.0, 1.25], "batch": 512, "sched_factor": 0.5, "sched_patience": 10, "es_patience": 60},
    ],
}

# ------------------------------ Imports ------------------------------
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")
T0 = time.time()
# Rule: the first two lines print the input and output locations (relative paths only)
print(f"Input directory: ./{INPUT_DIR}/ (" + ", ".join(f"{d}/" for d in DATASETS) + ", feature_config.json)")
print(f"Output directory: ./{OUTPUT_DIR}/")

# Check all input files before running
need = [os.path.join(INPUT_DIR, "feature_config.json")]
for d in DATASETS:
    need += [os.path.join(INPUT_DIR, d, f"{s}_{xy}.csv") for s in ["train", "val", "test"] for xy in ["X", "y"]]
missing = [p for p in need if not os.path.isfile(p)]
if missing:
    print("\nThe following input files do not exist; check INPUT_DIR at the top of the code:")
    for p in missing:
        print("  ./" + p)
    raise FileNotFoundError(f"{len(missing)} input files are missing")
print(f"Input file check passed ({len(need)} files)")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def elapsed():
    # Return the elapsed time as a string
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


# ==================== 1. Random seed (same as China) ====================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 2. Data and sequence construction (same as China) ====================

def load_data_raw(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "train_X.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "val_X.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "test_X.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "train_y.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "val_y.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "test_y.csv"))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def calculate_thresholds(y_df, target_col="pct_chg"):
    train_returns = y_df[target_col].dropna()
    q33 = np.percentile(train_returns, 33.33)
    q67 = np.percentile(train_returns, 66.67)
    return q33, q67


def build_sequence_data_with_meta(X_df, y_df, feature_cols, q33, q67, max_seq_len=60):
    # Labels: pct_chg shifted forward by 1 day within each stock
    y_work = y_df.copy()
    y_work["trade_date"] = pd.to_datetime(y_work["trade_date"])
    y_work["future_change"] = y_work.groupby("stock_code")["pct_chg"].shift(-1)

    def get_label(x):
        if pd.isna(x):
            return -999
        if x < q33:
            return 0
        elif x > q67:
            return 2
        else:
            return 1

    y_work["label"] = y_work["future_change"].apply(get_label)
    X_work = X_df.copy()
    X_work["trade_date"] = pd.to_datetime(X_work["trade_date"])
    merged = pd.merge(X_work, y_work[["stock_code", "trade_date", "label", "pct_chg", "future_change"]],
                      on=["stock_code", "trade_date"], how="inner")
    valid_feats = [f for f in feature_cols if f in X_work.columns]

    all_seqs, all_labels, all_meta = [], [], []
    for stock, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue
        g = g.sort_values("trade_date")
        feats = g[valid_feats].values
        dates = g["trade_date"].values
        labels = g["label"].values
        meta_data = g[["stock_code", "trade_date", "label", "pct_chg", "future_change"]].values
        for end in range(max_seq_len, len(g)):
            start_idx = end - max_seq_len
            # Suspension filter: skip if 60 trading days span more than 100 calendar days (same as China)
            days_diff = (dates[end - 1] - dates[start_idx]) / np.timedelta64(1, "D")
            if days_diff > 100:
                continue
            target = labels[end - 1]
            if target == -999:
                continue
            all_seqs.append(feats[start_idx:end])
            all_labels.append(target)
            all_meta.append(meta_data[end - 1])
    meta_df = pd.DataFrame(all_meta, columns=["stock_code", "trade_date", "label", "pct_chg", "future_change"])
    return np.stack(all_seqs), np.array(all_labels, dtype=np.int64), meta_df


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 3. Models (same as China) ====================

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


MODEL_CLASSES = {"GRU": GRUClassifier, "LSTM": LSTMClassifier, "BiLSTM": BiLSTMClassifier}


# ==================== 4. Training and evaluation (same as China; printing added only) ====================

def evaluate_model(model, loader, device, return_probs=False):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            preds.append(logits.argmax(-1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    y_p, y_t, y_prob = np.concatenate(preds), np.concatenate(trues), np.concatenate(probs, axis=0)
    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average="macro")
    if return_probs:
        return acc, f1, y_p, y_prob
    return acc, f1


def train_model_with_weight(model, train_loader, val_loader, device, class_weights, cfg, tag, epochs=400):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["sched_factor"], patience=cfg["sched_patience"])
    best_f1 = 0
    best_ep = 0
    patience = cfg["es_patience"]
    counter = 0
    best_state = None
    history = []
    t_train = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_acc, val_f1 = evaluate_model(model, val_loader, device)
        train_loss = np.mean(losses)
        curr_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "train_loss": float(train_loss), "val_acc": float(val_acc),
                        "val_f1": float(val_f1), "lr": float(curr_lr)})
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_ep = epoch + 1
            # Same as China: stored by reference, so the weights finally loaded are those at the end of training (early stopping)
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        scheduler.step(val_f1)
        per_ep = (time.time() - t_train) / (epoch + 1)
        print(f"    [{tag}] Epoch {epoch + 1:3d} | loss {train_loss:.4f} | val F1 {val_f1:.4f} acc {val_acc:.4f} | "
              f"lr {curr_lr:.2e} | best F1 {best_f1:.4f} (epoch {best_ep}) | early-stop counter {counter}/{patience} | "
              f"{per_ep:.1f}s per epoch | at most {per_ep * (epochs - epoch - 1) / 60:.1f}m remaining (early stopping at most "
              f"{per_ep * (patience - counter) / 60:.1f}m more) | total {elapsed()}")
        if counter >= patience:
            print(f"    [{tag}] Early stopping triggered at epoch {epoch + 1}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_f1, best_ep, len(history)


# ==================== 5. Main ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
CUDNN_USED = False
if device.type == "cuda":
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU memory: {free_b / 1024 ** 3:.1f} GB free / {total_b / 1024 ** 3:.1f} GB total")
    # Check whether cuDNN can be initialized: trial run with a very small GRU (does not affect later random seeds; set_seed is called again before training)
    for attempt in range(2):
        try:
            torch.backends.cudnn.enabled = True
            _probe = nn.GRU(4, 4, batch_first=True).to(device)
            _probe(torch.zeros(2, 3, 4, device=device))
            del _probe
            CUDNN_USED = True
            break
        except RuntimeError as e:
            print(f"cuDNN initialization failed (attempt {attempt + 1}): {str(e).splitlines()[0]}")
            torch.cuda.empty_cache()
    if CUDNN_USED:
        print(f"cuDNN OK (version {torch.backends.cudnn.version()}), same as the original Chinese experiments")
    else:
        torch.backends.cudnn.enabled = False
        print("⚠️ cuDNN could not be initialized; switched to PyTorch native CUDA RNN kernels (same as stage5_cell_B). "
              "Models and training procedure are unchanged, but values differ slightly from cuDNN; this must be stated in the paper")
with open(os.path.join(INPUT_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
    config_a = json.load(f)

n_total = sum(len(CANDIDATE_CONFIGS[a]) for a in ARCHS) * len(DATASETS)
print(f"{len(DATASETS)} datasets x {len(ARCHS)} models, {n_total} candidate-configuration training runs in total")
print("Candidate configurations:")
for a in ARCHS:
    for c in CANDIDATE_CONFIGS[a]:
        print(f"  {a:<7} from {c['source']:<17} hidden={c['hidden']} layers={c['layers']} dropout={c['dropout']} "
              f"lr={c['lr']} class_weights={c['class_weights']} batch={c['batch']} "
              f"scheduler={c['sched_factor']}/{c['sched_patience']} es_patience={c['es_patience']}")

all_rows = []
selected_rows = []
run_i = 0
for ds in DATASETS:
    print(f"\n{'#' * 100}")
    print(f"Dataset {ds}, total {elapsed()}")
    print(f"{'#' * 100}")
    feature_cols = config_a[ds]["selected_features"] + config_a[ds]["temporal_features"]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(os.path.join(INPUT_DIR, ds))
    q33, q67 = calculate_thresholds(y_tr)
    t = time.time()
    X_train, y_train, _ = build_sequence_data_with_meta(X_tr, y_tr, feature_cols, q33, q67)
    X_val, y_val, _ = build_sequence_data_with_meta(X_va, y_va, feature_cols, q33, q67)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, feature_cols, q33, q67)
    print(f"  {len(feature_cols)} features: {feature_cols}")
    print(f"  Thresholds q33 = {q33:.4f}, q67 = {q67:.4f}; sequence construction {time.time() - t:.1f}s")
    print(f"  Train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
    for nm, yy in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        cnt = np.bincount(yy, minlength=3)
        print(f"    {nm} labels: down {cnt[0] / cnt.sum():.1%} | hold {cnt[1] / cnt.sum():.1%} | up {cnt[2] / cnt.sum():.1%}")
    input_dim = X_train.shape[2]
    ds_dir = os.path.join(OUTPUT_DIR, ds)
    os.makedirs(ds_dir, exist_ok=True)

    for arch in ARCHS:
        arch_rows = []
        for c_idx, cfg in enumerate(CANDIDATE_CONFIGS[arch], 1):
            run_i += 1
            tag = f"{ds}-{arch}-cand{c_idx}"
            print(f"\n  {'=' * 90}")
            print(f"  Run {run_i}/{n_total}: {ds} {arch} candidate {c_idx} (from {cfg['source']}), total {elapsed()}")
            print(f"  {'=' * 90}")
            t_run = time.time()
            set_seed(SEED)
            tr_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=cfg["batch"], shuffle=True)
            val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=512)
            class_weights = torch.FloatTensor(cfg["class_weights"]).to(device)
            model = MODEL_CLASSES[arch](input_dim=input_dim, hidden_dim=cfg["hidden"],
                                        num_layers=cfg["layers"], dropout=cfg["dropout"]).to(device)
            model, history, best_f1, best_ep, n_ep = train_model_with_weight(
                model, tr_loader, val_loader, device, class_weights, cfg, tag, epochs=MAX_EPOCHS)
            # Validation performance of the saved model (early-stopping weights), used for selection
            saved_val_acc, saved_val_f1 = evaluate_model(model, val_loader, device)
            fname = f"{arch.lower()}_seed{SEED}_cand{c_idx}"
            torch.save(model.state_dict(), os.path.join(ds_dir, f"{fname}.pth"))
            pd.DataFrame(history).to_csv(os.path.join(ds_dir, f"{fname}_history.csv"), index=False)
            row = {"dataset": ds, "arch": arch, "candidate": c_idx, "source": cfg["source"],
                   "epochs_run": n_ep, "best_val_f1": round(best_f1, 4), "best_val_epoch": best_ep,
                   "saved_model_val_f1": round(saved_val_f1, 4), "saved_model_val_acc": round(saved_val_acc, 4),
                   "minutes": round((time.time() - t_run) / 60, 1)}
            arch_rows.append(row)
            all_rows.append(row)
            print(f"  [{tag}] {n_ep} epochs trained; best val F1 {best_f1:.4f} (epoch {best_ep}); "
                  f"saved model (epoch-{n_ep} weights) val F1 {saved_val_f1:.4f}, acc {saved_val_acc:.4f}; "
                  f"time {row['minutes']}m, total {elapsed()}")

        # Select the configuration by the validation F1 of the saved model
        best = max(arch_rows, key=lambda r: r["saved_model_val_f1"])
        c_idx = best["candidate"]
        cfg = CANDIDATE_CONFIGS[arch][c_idx - 1]
        print(f"\n  >>> {ds} {arch} selected candidate {c_idx} (from {cfg['source']}), saved-model val F1 = {best['saved_model_val_f1']:.4f}")
        for r in arch_rows:
            print(f"      Candidate {r['candidate']} ({r['source']}): saved-model val F1 {r['saved_model_val_f1']:.4f}, "
                  f"{r['epochs_run']} epochs")

        # Evaluate the test set only for the selected configuration
        model = MODEL_CLASSES[arch](input_dim=input_dim, hidden_dim=cfg["hidden"],
                                    num_layers=cfg["layers"], dropout=cfg["dropout"]).to(device)
        fname = f"{arch.lower()}_seed{SEED}_cand{c_idx}"
        model.load_state_dict(torch.load(os.path.join(ds_dir, f"{fname}.pth"), map_location=device))
        te_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=512)
        acc, f1, y_pred, y_prob = evaluate_model(model, te_loader, device, return_probs=True)
        f1_pc = f1_score(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)
        torch.save(model.state_dict(), os.path.join(ds_dir, f"{arch.lower()}_seed{SEED}_selected.pth"))
        pred_df = meta_test.copy()
        pred_df["pred_label"] = y_pred
        pred_df[["prob_down", "prob_hold", "prob_up"]] = y_prob
        pred_df.to_csv(os.path.join(ds_dir, f"{arch.lower()}_seed{SEED}_test_predictions.csv"), index=False)
        print(f"  >>> {ds} {arch} test set: Macro-F1 {f1:.4f}, accuracy {acc:.4f}; "
              f"per-class F1 down {f1_pc[0]:.4f} / hold {f1_pc[1]:.4f} / up {f1_pc[2]:.4f}")
        print(f"      Confusion matrix (rows = true down/hold/up, columns = predicted):\n{cm}")
        selected_rows.append({"dataset": ds, "arch": arch, "cudnn": CUDNN_USED, "selected_candidate": c_idx, "source": cfg["source"],
                              "hidden": cfg["hidden"], "layers": cfg["layers"], "dropout": cfg["dropout"],
                              "lr": cfg["lr"], "class_weights": str(cfg["class_weights"]), "batch": cfg["batch"],
                              "epochs_run": best["epochs_run"], "val_f1": best["saved_model_val_f1"],
                              "test_macro_f1": round(f1, 4), "test_acc": round(acc, 4),
                              "test_f1_down": round(f1_pc[0], 4), "test_f1_hold": round(f1_pc[1], 4),
                              "test_f1_up": round(f1_pc[2], 4)})
        pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "all_candidates.csv"), index=False)
        pd.DataFrame(selected_rows).to_csv(os.path.join(OUTPUT_DIR, "selected_models.csv"), index=False)

print("\n" + "=" * 100)
print("stage6_cell_C finished, total time " + elapsed())
print("=" * 100)
print("All candidates (selected on the validation set):")
print(pd.DataFrame(all_rows).to_string(index=False))
print("\nSelected models and test results:")
print(pd.DataFrame(selected_rows).drop(columns=["class_weights"]).to_string(index=False))
print(f"\nOutput files: ./{OUTPUT_DIR}/all_candidates.csv, selected_models.csv; "
      f"in each dataset folder: *_selected.pth (selected model), *_test_predictions.csv, and .pth and _history.csv for each candidate")
