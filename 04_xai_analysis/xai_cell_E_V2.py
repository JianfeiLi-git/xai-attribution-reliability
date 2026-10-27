# -*- coding: utf-8 -*-
"""
xai_cell_E_V2

目标：
1. Label-randomization sanity check；
2. 对5个数据集 × 3种RNN重新训练随机标签模型；
3. train/val标签分别随机置换，真实test标签保持不变；
4. 比较原模型与随机标签模型的测试性能；
5. 在相同样本、相同training-mean baseline、相同原模型固定target下比较IG归因结构。

原则：
- 不扫描目录；
- 不依赖任何A/B/C/D流程结果文件；
- 原模型仅从固定checkpoint路径读取；
- 随机标签模型采用统一sanity retraining protocol；
- 不宣称复制原论文的optimizer超参数；
- 本流程唯一有效版本为E_V2。
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

# 明确路径：禁止扫描目录
# 当前Notebook位置：
# 医疗：中国 ./XAI_cn/*.ipynb
# 电力：中国 ./XAI_cn/e3/*.ipynb
# 美国：   ./xai_us/*.ipynb

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
        raise ValueError(f"缺少特征: {missing}")

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
            f"{dataset_name}: 数据文件不存在: {data_dir} | 缺少: {missing}"
        )
    return data_dir


def prepare_dataset(dataset_name, cfg):
    data_dir = get_data_dir(dataset_name)
    print(f"{dataset_name}: 数据目录={data_dir}")
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
            f"{dataset_name}-{model_type}: 模型文件不存在: {model_path}"
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
# Cell E 专用配置
# =========================

from torch.utils.data import TensorDataset, DataLoader

OUTPUT_DIR = Path("./0924/xai_cell_E_V2")
print(f"输出目录: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 标签随机化重训练协议
# 架构严格从原checkpoint推断；训练协议统一固定，用于sanity retraining，
# 不宣称复制原论文的优化超参数。
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
    """从原模型结构读取input/hidden/layers，构造同架构新模型。"""
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
        raise ValueError(f"未知模型类型: {model_type}")

    return model.to(DEVICE)


def permute_labels(y, seed):
    rng = np.random.default_rng(seed)
    yp = np.asarray(y).copy()
    rng.shuffle(yp)
    return yp


def train_randomized_label_model(model, X_train, y_train_perm, X_val, y_val_perm):
    """
    使用随机标签训练同架构模型。
    早停依据随机化validation loss，仅用于避免无意义过长训练。
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
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.6f}, "
            f"perm_val_loss={val_loss:.6f}, "
            f"perm_val_acc={val_acc:.6f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= E_PATIENCE:
            print(f"早停: 连续{E_PATIENCE}轮无改善")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, pd.DataFrame(history)


def ig_fixed_target(model, X_input, X_train, sample_idx, fixed_targets,
                    n_steps=32, batch_size=64):
    """
    原模型与随机标签模型使用同一批样本、同一训练均值baseline、
    同一固定target，避免target变化造成比较混淆。
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
    print("步骤3：构造随机标签")

    y_train_perm = permute_labels(ds["y_train"], SEED + 100)
    y_val_perm = permute_labels(ds["y_val"], SEED + 200)

    print(f"训练标签原分布: {pd.Series(ds['y_train']).value_counts().sort_index().to_dict()}")
    print(f"训练标签随机化后分布: {pd.Series(y_train_perm).value_counts().sort_index().to_dict()}")
    print(f"验证标签原分布: {pd.Series(ds['y_val']).value_counts().sort_index().to_dict()}")
    print(f"验证标签随机化后分布: {pd.Series(y_val_perm).value_counts().sort_index().to_dict()}")

    train_changed = float(np.mean(y_train_perm != ds["y_train"]))
    val_changed = float(np.mean(y_val_perm != ds["y_val"]))

    print(f"训练标签位置改变比例: {train_changed:.6f}")
    print(f"验证标签位置改变比例: {val_changed:.6f}")

    print("步骤4：随机标签模型重训练")
    rand_model = clone_architecture_from_original(model_type, original_model)

    print(
        f"训练协议: batch={E_BATCH_SIZE}, lr={E_LR}, "
        f"weight_decay={E_WEIGHT_DECAY}, max_epochs={E_MAX_EPOCHS}, patience={E_PATIENCE}"
    )

    rand_model, history = train_randomized_label_model(
        rand_model,
        ds["X_train"], y_train_perm,
        ds["X_val"], y_val_perm,
    )

    print(f"实际训练轮数: {len(history)}")
    print(f"最佳随机验证loss: {history['Permuted_Val_Loss'].min():.6f}")

    print("步骤5：真实测试集预测能力检查")
    original_perf = evaluate_true_test(original_model, ds["X_test"], ds["y_test"])
    random_perf = evaluate_true_test(rand_model, ds["X_test"], ds["y_test"])

    print(
        f"原模型: Macro-F1={original_perf['Macro_F1']:.6f}, "
        f"Accuracy={original_perf['Accuracy']:.6f}, "
        f"TrueProb={original_perf['Mean_TrueProb']:.6f}"
    )
    print(
        f"随机标签模型: Macro-F1={random_perf['Macro_F1']:.6f}, "
        f"Accuracy={random_perf['Accuracy']:.6f}, "
        f"TrueProb={random_perf['Mean_TrueProb']:.6f}"
    )
    print(
        f"变化: ΔF1={random_perf['Macro_F1']-original_perf['Macro_F1']:.6f}, "
        f"ΔAcc={random_perf['Accuracy']-original_perf['Accuracy']:.6f}"
    )

    print("步骤6：原模型 vs 随机标签模型 IG")
    sample_idx = choose_fixed_samples(ds["X_test"], E_IG_SAMPLES, SEED)

    original_pred, _ = predict_arrays(
        original_model, ds["X_test"][sample_idx], EVAL_BATCH_SIZE
    )
    fixed_targets = original_pred.astype(np.int64)

    print(f"固定IG样本数: {len(sample_idx)}")
    print(f"IG steps: {E_IG_STEPS}")
    print(f"固定target分布: {pd.Series(fixed_targets).value_counts().sort_index().to_dict()}")

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
        f"归因相关: Mapρ={map_rho:.6f} (p={map_p:.6g}), "
        f"Featureρ={feat_rho:.6f} (p={feat_p:.6g}), "
        f"Temporalρ={time_rho:.6f} (p={time_p:.6g})"
    )

    print("原模型IG Top5:")
    for rank, j in enumerate(np.argsort(-orig_feat)[:5], 1):
        print(f"{rank}. {feature_names[j]} | {orig_feat[j]:.8f}")

    print("随机标签模型IG Top5:")
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
# 7. 主流程
# =========================

def main():
    print(f"设备: {DEVICE}")
    print("任务: Label randomization sanity check（独立流程）")
    print("依赖: 原始数据 + 固定checkpoint；不读取A/B/C/D结果文件")
    print("范围: 5个数据集 × 3种RNN = 15个模型")
    print("原则: 架构匹配原checkpoint；train/val标签分别随机置换；真实test标签保持不变")
    print("IG比较: 同一样本、同一training-mean baseline、同一原模型固定target")
    print("注意: 本Cell采用统一sanity重训练协议，不宣称复制原训练optimizer超参数")

    all_rows = []
    all_hist = []
    errors = []

    for dataset_name in RUN_DATASETS:
        cfg = DATASETS[dataset_name]

        print("-" * 20)
        print(f"数据集: {dataset_name}")
        print("步骤1：数据准备与审计")

        try:
            ds = prepare_dataset(dataset_name, cfg)
            print(f"字段数: {len(cfg['features'])}")
            print(f"字段: {', '.join(cfg['features'])}")
            print(f"训练样本: {len(ds['X_train'])}, 验证样本: {len(ds['X_val'])}, 测试样本: {len(ds['X_test'])}")
            print(f"训练缺失值: {int(np.isnan(ds['X_train']).sum())}, 验证缺失值: {int(np.isnan(ds['X_val']).sum())}, 测试缺失值: {int(np.isnan(ds['X_test']).sum())}")
            print(f"训练标签分布: {pd.Series(ds['y_train']).value_counts().sort_index().to_dict()}")
            print(f"验证标签分布: {pd.Series(ds['y_val']).value_counts().sort_index().to_dict()}")
            print(f"测试标签分布: {pd.Series(ds['y_test']).value_counts().sort_index().to_dict()}")
            print("数据审计: 通过")
        except Exception as e:
            print(f"异常: 数据准备失败: {type(e).__name__}: {e}")
            errors.append({
                "Dataset": dataset_name, "Model": "",
                "Step": "数据准备", "Error": f"{type(e).__name__}: {e}"
            })
            continue

        for model_type in MODELS:
            print("-" * 20)
            print(f"{dataset_name} / {model_type}")
            print("步骤2：读取原模型并复核架构")

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
                    f"架构: input={rnn.input_size}, hidden={rnn.hidden_size}, "
                    f"layers={rnn.num_layers}, direction={direction}"
                )

                perf = evaluate_true_test(
                    original_model, ds["X_test"], ds["y_test"]
                )
                print(
                    f"原模型复核: Macro-F1={perf['Macro_F1']:.6f}, "
                    f"Accuracy={perf['Accuracy']:.6f}"
                )
                print("原模型审计: 通过")

            except Exception as e:
                print(f"异常: 原模型读取失败: {type(e).__name__}: {e}")
                errors.append({
                    "Dataset": dataset_name, "Model": model_type,
                    "Step": "原模型读取", "Error": f"{type(e).__name__}: {e}"
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

                print("当前模型最终结果:")
                print(pd.DataFrame([row])[
                    [
                        "Dataset", "Model",
                        "RandomLabel_MacroF1", "RandomLabel_Accuracy",
                        "Map_Spearman", "Feature_Spearman", "Temporal_Spearman"
                    ]
                ].to_string(index=False))
                print("Label randomization: 完成")

            except Exception as e:
                print(f"异常: Label randomization失败: {type(e).__name__}: {e}")
                errors.append({
                    "Dataset": dataset_name, "Model": model_type,
                    "Step": "Label randomization",
                    "Error": f"{type(e).__name__}: {e}"
                })

            del original_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("-" * 20)
    print("步骤7：保存结果并最终汇总")

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

    print(f"Sanity结果行数: {len(df)}")
    print(f"训练历史行数: {len(hist_df)}")
    print(f"异常记录数: {len(err_df)}")

    print("-" * 20)
    print("最终汇总1：随机标签模型真实测试表现")
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
        print("无结果")

    print("-" * 20)
    print("最终汇总2：原模型 vs 随机标签模型归因相关")
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
        print("无结果")

    print("-" * 20)
    print("最终汇总3：跨模型统计")
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
        print("无结果")

    print("-" * 20)
    print("最终汇总4：异常")
    if err_df.empty:
        print("异常: 0，全部步骤运行完成")
    else:
        print(err_df.to_string(index=False))

    print("-" * 20)
    print("xai_cell_E_V2 完成")
    print(f"结果目录: {OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
