# -*- coding: utf-8 -*-
"""
xai_cell_K_V1

专门处理 Medical-2 / GRU 的 5-seed 稳定性实验。

背景：
- 原 seed=42 checkpoint 可确认：input=14, hidden=256, layers=2；
- 但生成该 checkpoint 的完整训练脚本已丢失；
- 因此本 Cell 不声称“完全复现原 seed42 训练”；
- 这里定义一个透明、固定、可重复的补充实验配置，独立训练5个seed；
- 5-seed统计只基于本Cell新训练的5个模型，不把旧seed42混入；
- 同时额外复核旧seed42 checkpoint，作为 legacy reference，不参与5-seed均值/SD。

固定补充配置：
hidden=256, layers=2, dropout=0.0
lr=3e-4, class_weights=[1.1, 1.0, 1.1]
batch=512
AdamW weight_decay=3e-4
ReduceLROnPlateau factor=0.7, patience=15
early stopping patience=50
max_epochs=400

其中：
- hidden=256、layers=2由正式checkpoint直接确认；
- 其余训练参数采用现存Medical GRU训练流程中最接近的固定设置，
  仅作为新的补充稳定性实验设置，不冒充丢失脚本的原始超参数。

输出目录：
./xai_cell_K_V1/
"""

import copy
import json
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
# 0. 全局设置
# --------------------
OUTPUT_DIR = Path("./xai_cell_K_V1")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"输出目录: {OUTPUT_DIR.as_posix()}/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")

SEEDS = [42, 123, 777, 0, 2024]
PRIMARY_SEED = 42

SEQ_LEN = 60
IG_SAMPLES = 500
IG_STEPS = 64
IG_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
MAX_EPOCHS = 400

REUSE_EXISTING_MODELS = True

DATA_DIR = Path("./XAI_cn/datasets/processed/dataset2_AB_grade")
LEGACY_MODEL = Path(
    "./XAI_cn/models_XAI/rnn_enhanced_dataset2/"
    "gru_dataset2_SELECTED_14_seed42.pt"
)

FEATURES = [
    "high_low_range", "turnover_rate", "amount", "bb_width", "roc_10",
    "roc_5", "close_to_ma20", "close_to_ma5", "close_open_range", "atr",
    "volume_ratio", "roc_20", "dayofyear_cos", "year"
]

CFG = {
    "hidden": 256,
    "layers": 2,
    "dropout": 0.0,
    "lr": 3e-4,
    "weights": [1.1, 1.0, 1.1],
    "batch": 512,
    "weight_decay": 3e-4,
    "sched_factor": 0.7,
    "sched_patience": 15,
    "early_patience": 50,
}

print(f"随机种子: {SEEDS}")
print(f"IG样本数: {IG_SAMPLES}")
print(f"IG步数: {IG_STEPS}")
print("--------------------")
print("实验性质: 固定配置的补充5-seed实验")
print("注意: 不声称完全复现丢失的原Medical-2/GRU训练脚本")
print(
    f"配置: hidden={CFG['hidden']}, layers={CFG['layers']}, "
    f"dropout={CFG['dropout']}, lr={CFG['lr']:.1e}, "
    f"weights={CFG['weights']}, batch={CFG['batch']}"
)
print("--------------------")


# --------------------
# 1. 随机种子
# --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------
# 2. 数据
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
        raise FileNotFoundError(f"数据缺失: {missing}")

    return (
        (pd.read_csv(data_dir / "train_X.csv"),
         pd.read_csv(data_dir / "train_y.csv")),
        (pd.read_csv(data_dir / "val_X.csv"),
         pd.read_csv(data_dir / "val_y.csv")),
        (pd.read_csv(data_dir / "test_X.csv"),
         pd.read_csv(data_dir / "test_y.csv")),
    )


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


def build_sequence_data(X_df, y_df, q33, q67):
    y_work = y_df.copy()
    y_work["trade_date"] = pd.to_datetime(y_work["trade_date"])
    y_work["future_change"] = (
        y_work.groupby("stock_code")["pct_chg"].shift(-1)
    )

    def label_func(x):
        if pd.isna(x):
            return -999
        if x < q33:
            return 0
        if x > q67:
            return 2
        return 1

    y_work["label"] = y_work["future_change"].apply(label_func)

    X_work = X_df.copy()
    X_work["trade_date"] = pd.to_datetime(X_work["trade_date"])

    missing = [f for f in FEATURES if f not in X_work.columns]
    if missing:
        raise ValueError(f"缺少特征: {missing}")

    merged = pd.merge(
        X_work,
        y_work[[
            "stock_code", "trade_date", "label",
            "pct_chg", "future_change"
        ]],
        on=["stock_code", "trade_date"],
        how="inner",
    )

    seqs = []
    labels = []

    for _, g in merged.groupby("stock_code"):
        if len(g) < SEQ_LEN:
            continue

        g = g.sort_values("trade_date")
        feats = g[FEATURES].values
        dates = g["trade_date"].values
        y = g["label"].values

        for end in range(SEQ_LEN, len(g)):
            start = end - SEQ_LEN
            days_diff = (
                dates[end - 1] - dates[start]
            ) / np.timedelta64(1, "D")

            if days_diff > 100:
                continue

            target = y[end - 1]
            if target == -999:
                continue

            seqs.append(feats[start:end])
            labels.append(target)

    if not seqs:
        raise RuntimeError("序列构建后无样本")

    return (
        np.stack(seqs).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def prepare_data():
    print("步骤1：数据准备与审计")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(DATA_DIR)

    X_tr, y_tr = filter_data_by_date(X_tr, y_tr)

    q33, q67 = calculate_thresholds(y_tr)

    X_train, y_train = build_sequence_data(X_tr, y_tr, q33, q67)
    X_val, y_val = build_sequence_data(X_va, y_va, q33, q67)
    X_test, y_test = build_sequence_data(X_te, y_te, q33, q67)

    print(
        f"Medical-2: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )
    print(f"动态阈值: q33={q33:.6f}, q67={q67:.6f}")

    for name, arr in [
        ("train", X_train), ("val", X_val), ("test", X_test)
    ]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"{name}: NaN={n_nan}, Inf={n_inf}")
        if n_nan or n_inf:
            raise ValueError(f"{name}存在NaN/Inf")

    print("数据审计: 通过")
    print("--------------------")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# --------------------
# 3. 模型
# --------------------
class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
        num_classes=3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
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


def new_model():
    return GRUClassifier(
        input_dim=len(FEATURES),
        hidden_dim=CFG["hidden"],
        num_layers=CFG["layers"],
        dropout=CFG["dropout"],
    )


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state.items()
    }


def load_model(path):
    model = new_model()
    state = torch.load(path, map_location="cpu")
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


# --------------------
# 4. 评估
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


# --------------------
# 5. 训练
# --------------------
def train_one_seed(seed, data):
    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        SequenceDataset(data["X_train"], data["y_train"]),
        batch_size=CFG["batch"],
        shuffle=True,
        generator=g,
        num_workers=0,
    )

    model = new_model().to(DEVICE)

    class_weights = torch.tensor(
        CFG["weights"],
        dtype=torch.float32,
        device=DEVICE,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CFG["sched_factor"],
        patience=CFG["sched_patience"],
    )

    best_f1 = -np.inf
    best_state = None
    counter = 0
    history = []

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
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            counter = 0
        else:
            counter += 1

        if counter >= CFG["early_patience"]:
            print(f"Early stopping: epoch={epoch+1}")
            break

    if best_state is None:
        raise RuntimeError("best_state为空")

    model.load_state_dict(best_state, strict=True)
    model = model.to(DEVICE).eval()

    return model, history, best_f1


# --------------------
# 6. IG
# --------------------
def choose_fixed_samples(X_test):
    rng = np.random.default_rng(PRIMARY_SEED)
    n = min(IG_SAMPLES, len(X_test))
    return np.sort(
        rng.choice(len(X_test), size=n, replace=False)
    )


def ig_attribution_fixed_targets(
    model,
    X_test,
    X_train,
    sample_idx,
    fixed_targets,
):
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

        for s in range(0, len(sample_idx), IG_BATCH_SIZE):
            idx = sample_idx[s:s + IG_BATCH_SIZE]

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
                n_steps=IG_STEPS,
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


def summarize_ig(attr):
    timestep = attr.mean(axis=(0, 2))
    feature = attr.mean(axis=(0, 1))

    timestep_share = timestep / (timestep.sum() + 1e-12)
    feature_share = feature / (feature.sum() + 1e-12)

    time_df = pd.DataFrame({
        "Timestep": np.arange(1, 61),
        "IG_Importance": timestep,
        "IG_Share": timestep_share,
    })

    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "IG_Importance": feature,
        "IG_Share": feature_share,
    })

    return time_df, feat_df


def top5_overlap(a, b):
    aa = set(a.nlargest(5, "IG_Importance")["Feature"])
    bb = set(b.nlargest(5, "IG_Importance")["Feature"])
    return len(aa & bb) / 5.0


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------
# 7. 主流程
# --------------------
def main():
    data = prepare_data()
    sample_idx = choose_fixed_samples(data["X_test"])

    print(f"固定IG样本数: {len(sample_idx)}")
    print("--------------------")

    # 先复核legacy checkpoint，只作旁证
    print("步骤2：旧seed42 checkpoint复核")
    if not LEGACY_MODEL.exists():
        raise FileNotFoundError(f"旧模型不存在: {LEGACY_MODEL}")

    legacy_model = load_model(LEGACY_MODEL)
    legacy_eval = evaluate_arrays(
        legacy_model,
        data["X_test"],
        data["y_test"],
    )

    print(
        f"Legacy seed42: "
        f"Macro-F1={legacy_eval['macro_f1']:.6f}, "
        f"Accuracy={legacy_eval['accuracy']:.6f}"
    )
    print("Legacy模型仅作参考，不进入新5-seed均值/SD")
    print("--------------------")

    rows = []
    feat_rows = []
    time_rows = []

    seed_models = {}
    seed_evals = {}
    seed_histories = {}

    # 训练或复用5个新模型
    for seed in SEEDS:
        print(f"Seed={seed}")

        model_path = MODEL_DIR / f"Medical-2_GRU_seed{seed}.pt"
        hist_path = OUTPUT_DIR / f"history_Medical-2_GRU_seed{seed}.json"

        if REUSE_EXISTING_MODELS and model_path.exists():
            print(f"复用模型: {model_path}")
            model = load_model(model_path)

            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []

            best_val = (
                max(x["val_f1"] for x in history)
                if history else np.nan
            )
        else:
            print("训练新模型")
            model, history, best_val = train_one_seed(seed, data)
            torch.save(model.state_dict(), model_path)
            save_json(history, hist_path)
            print(f"模型保存: {model_path}")

        ev = evaluate_arrays(
            model,
            data["X_test"],
            data["y_test"],
        )

        print(
            f"测试结果: F1={ev['macro_f1']:.6f}, "
            f"Acc={ev['accuracy']:.6f}"
        )

        seed_models[seed] = model
        seed_evals[seed] = ev
        seed_histories[seed] = history

        print("--------------------")

    # 以本Cell新训练的seed42作为5-seed参考
    ref_model = seed_models[PRIMARY_SEED]
    ref_eval = seed_evals[PRIMARY_SEED]

    fixed_targets, _ = predict_arrays(
        ref_model,
        data["X_test"][sample_idx],
        EVAL_BATCH_SIZE,
    )

    ref_attr = ig_attribution_fixed_targets(
        ref_model,
        data["X_test"],
        data["X_train"],
        sample_idx,
        fixed_targets,
    )
    ref_time, ref_feat = summarize_ig(ref_attr)

    # legacy IG：同一批样本、同一fixed target
    legacy_attr = ig_attribution_fixed_targets(
        legacy_model,
        data["X_test"],
        data["X_train"],
        sample_idx,
        fixed_targets,
    )
    legacy_time, legacy_feat = summarize_ig(legacy_attr)

    legacy_feat_rho, _ = spearmanr(
        ref_feat["IG_Importance"].values,
        legacy_feat["IG_Importance"].values,
    )
    legacy_time_rho, _ = spearmanr(
        ref_time["IG_Importance"].values,
        legacy_time["IG_Importance"].values,
    )

    print("Legacy vs 新seed42")
    print(
        f"Featureρ={legacy_feat_rho:.4f}, "
        f"Temporalρ={legacy_time_rho:.4f}"
    )
    print("--------------------")

    # 5-seed IG
    for seed in SEEDS:
        print(f"IG稳定性: Seed={seed}")

        model = seed_models[seed]
        ev = seed_evals[seed]

        if seed == PRIMARY_SEED:
            time_df = ref_time.copy()
            feat_df = ref_feat.copy()
        else:
            attr = ig_attribution_fixed_targets(
                model,
                data["X_test"],
                data["X_train"],
                sample_idx,
                fixed_targets,
            )
            time_df, feat_df = summarize_ig(attr)

        feat_rho, feat_p = spearmanr(
            ref_feat["IG_Importance"].values,
            feat_df["IG_Importance"].values,
        )
        time_rho, time_p = spearmanr(
            ref_time["IG_Importance"].values,
            time_df["IG_Importance"].values,
        )

        pred_agreement = float(
            np.mean(ref_eval["pred"] == ev["pred"])
        )

        overlap = top5_overlap(ref_feat, feat_df)

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

        hist = seed_histories[seed]
        best_val = (
            max(x["val_f1"] for x in hist)
            if hist else np.nan
        )

        rows.append({
            "Dataset": "Medical-2",
            "Model": "GRU",
            "Seed": seed,
            "MacroF1": ev["macro_f1"],
            "Accuracy": ev["accuracy"],
            "Prediction_Agreement_vs_NewSeed42": pred_agreement,
            "Feature_Spearman_vs_NewSeed42": feat_rho,
            "Feature_p_vs_NewSeed42": feat_p,
            "Temporal_Spearman_vs_NewSeed42": time_rho,
            "Temporal_p_vs_NewSeed42": time_p,
            "Top5_Overlap_vs_NewSeed42": overlap,
            "IG_Last10_Mass": last10,
            "IG_First10_Mass": first10,
            "IG_Last10_to_First10": last10 / max(first10, 1e-12),
            "Epochs_Run": len(hist) if hist else np.nan,
            "Best_Val_MacroF1": best_val,
        })

        f = feat_df.copy()
        f.insert(0, "Seed", seed)
        f.insert(0, "Model", "GRU")
        f.insert(0, "Dataset", "Medical-2")
        feat_rows.append(f)

        t = time_df.copy()
        t.insert(0, "Seed", seed)
        t.insert(0, "Model", "GRU")
        t.insert(0, "Dataset", "Medical-2")
        time_rows.append(t)

        print(
            f"F1={ev['macro_f1']:.6f}, "
            f"Featureρ={feat_rho:.4f}, "
            f"Temporalρ={time_rho:.4f}, "
            f"Last10={last10:.4f}"
        )
        print("--------------------")

    detail_df = pd.DataFrame(rows)
    feature_df = pd.concat(feat_rows, ignore_index=True)
    timestep_df = pd.concat(time_rows, ignore_index=True)

    alt = detail_df[detail_df["Seed"] != PRIMARY_SEED]

    summary_df = pd.DataFrame([{
        "Dataset": "Medical-2",
        "Model": "GRU",
        "N_Seeds": 5,
        "MacroF1_Mean": detail_df["MacroF1"].mean(),
        "MacroF1_SD": detail_df["MacroF1"].std(ddof=1),
        "Accuracy_Mean": detail_df["Accuracy"].mean(),
        "Accuracy_SD": detail_df["Accuracy"].std(ddof=1),
        "Prediction_Agreement_Mean_4Seeds":
            alt["Prediction_Agreement_vs_NewSeed42"].mean(),
        "Feature_Spearman_Mean_4Seeds":
            alt["Feature_Spearman_vs_NewSeed42"].mean(),
        "Feature_Spearman_Min_4Seeds":
            alt["Feature_Spearman_vs_NewSeed42"].min(),
        "Temporal_Spearman_Mean_4Seeds":
            alt["Temporal_Spearman_vs_NewSeed42"].mean(),
        "Temporal_Spearman_Min_4Seeds":
            alt["Temporal_Spearman_vs_NewSeed42"].min(),
        "Top5_Overlap_Mean_4Seeds":
            alt["Top5_Overlap_vs_NewSeed42"].mean(),
        "IG_Last10_Mass_Mean":
            detail_df["IG_Last10_Mass"].mean(),
        "IG_Last10_Mass_SD":
            detail_df["IG_Last10_Mass"].std(ddof=1),
        "All_Seeds_Last10_gt_First10":
            bool(
                (
                    detail_df["IG_Last10_Mass"]
                    > detail_df["IG_First10_Mass"]
                ).all()
            ),
        "Legacy_MacroF1": legacy_eval["macro_f1"],
        "Legacy_Accuracy": legacy_eval["accuracy"],
        "Legacy_vs_NewSeed42_Feature_Rho": legacy_feat_rho,
        "Legacy_vs_NewSeed42_Temporal_Rho": legacy_time_rho,
    }])

    detail_path = OUTPUT_DIR / "multiseed_detail.csv"
    summary_path = OUTPUT_DIR / "multiseed_summary.csv"
    feature_path = OUTPUT_DIR / "ig_feature_all_seeds.csv"
    timestep_path = OUTPUT_DIR / "ig_timestep_all_seeds.csv"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    timestep_df.to_csv(timestep_path, index=False)

    print("步骤3：最终汇总")
    print(f"明细行数: {len(detail_df)}")
    print("预期明细行数: 5")

    if len(detail_df) != 5:
        raise AssertionError("明细行数不是5")

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("--------------------")
    print("输出文件")
    print(detail_path)
    print(summary_path)
    print(feature_path)
    print(timestep_path)
    print(MODEL_DIR)
    print("--------------------")
    print("全部完成")


if __name__ == "__main__":
    main()
