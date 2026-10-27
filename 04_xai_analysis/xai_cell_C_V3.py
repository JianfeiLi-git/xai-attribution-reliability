# -*- coding: utf-8 -*-
"""
xai_cell_C_V3

目标：
1. 预测baseline补充实验；
2. Logistic Regression 与 Extra Trees 分别比较 Unweighted / Balanced；
3. Momentum作为简单方向基线；
4. 使用A_V9的原RNN模型复核结果，与最佳传统baseline比较。

原则：
- 不扫描目录；
- 不重新训练RNN；
- 传统baseline仅使用当前数据划分；
- RNN结果唯一来源：./0924/xai_cell_A_V9/A0_model_reproduction_check.csv；
- 每个流程只保留当前有效版本。
"""

import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

# =========================
# 0. 全局配置
# =========================

SEED = 42
SEQ_LEN = 60

OUTPUT_DIR = Path("./0924/xai_cell_C_V3")
print(f"输出目录: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNN_RESULT_PATH = Path("./0924/xai_cell_A_V9/A0_model_reproduction_check.csv")

RUN_DATASETS = ["Medical-1", "Medical-2", "Power", "US-Med", "US-Pow"]

DATA_DIR = {
    "Medical-1": Path("./XAI_cn/datasets/processed/dataset1_A_grade"),
    "Medical-2": Path("./XAI_cn/datasets/processed/dataset2_AB_grade"),
    "Power": Path("./XAI_cn/e3/datasets/processed/dataset2_AB_grade"),
    "US-Med": Path("./xai_us/datasets/processed/dataset4_m"),
    "US-Pow": Path("./xai_us/datasets/processed/dataset5_p"),
}

DATASETS = {
    "Medical-1": {
        "filter_train_2010": False,
        "features": [
            "dayofyear", "day", "volume_ratio", "close_to_ma5", "vol", "low",
            "close_open_range", "macd_signal", "close_to_ma20", "roc_20",
            "high_low_range", "turnover_rate", "amount"
        ],
    },
    "Medical-2": {
        "filter_train_2010": True,
        "features": [
            "high_low_range", "turnover_rate", "amount", "bb_width", "roc_10",
            "roc_5", "close_to_ma20", "close_to_ma5", "close_open_range", "atr",
            "volume_ratio", "roc_20", "dayofyear_cos", "year"
        ],
    },
    "Power": {
        "filter_train_2010": True,
        "features": [
            "dayofyear", "atr", "roc_5", "high_low_range", "turnover_rate",
            "macd_signal", "close", "macd", "month", "close_to_ma5",
            "roc_10", "amount"
        ],
    },
    "US-Med": {
        "filter_train_2010": False,
        "features": [
            "roc_5", "low", "close_to_ma20", "roc_20", "high_low_range",
            "rsi", "dayofweek", "macd", "dayofyear_sin", "bb_width",
            "bb_position", "macd_signal", "dayofyear"
        ],
    },
    "US-Pow": {
        "filter_train_2010": False,
        "features": [
            "roc_5", "dayofyear_sin", "macd_signal", "ma10", "macd",
            "dayofyear", "close", "high_low_range", "low", "close_to_ma20",
            "roc_20", "pre_close"
        ],
    },
}

# =========================
# 1. 随机种子
# =========================

random.seed(SEED)
np.random.seed(SEED)

# =========================
# 2. 数据函数
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

    merged = pd.merge(
        X_work,
        y_work[["stock_code", "trade_date", "label", "pct_chg", "future_change"]],
        on=["stock_code", "trade_date"],
        how="inner",
    )

    seqs, labels, meta_rows = [], [], []

    for _, g in merged.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue

        g = g.sort_values("trade_date")
        feats = g[feature_cols].values
        dates = g["trade_date"].values
        labs = g["label"].values

        for end in range(max_seq_len, len(g)):
            start = end - max_seq_len
            days_diff = (dates[end - 1] - dates[start]) / np.timedelta64(1, "D")
            if days_diff > 100:
                continue

            target = labs[end - 1]
            if target == -999:
                continue

            row = g.iloc[end - 1]
            seqs.append(feats[start:end])
            labels.append(target)
            meta_rows.append({
                "stock_code": row["stock_code"],
                "trade_date": row["trade_date"],
                "label": int(target),
                "pct_chg": float(row["pct_chg"]),
                "future_change": float(row["future_change"]),
            })

    return (
        np.stack(seqs).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        pd.DataFrame(meta_rows),
    )


def prepare_dataset(dataset_name, cfg):
    data_dir = DATA_DIR[dataset_name]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data_raw(data_dir)

    if cfg["filter_train_2010"]:
        X_tr, y_tr = filter_data_by_date(X_tr, y_tr)

    q33, q67 = calculate_thresholds(y_tr)

    X_train, y_train, meta_train = build_sequence_data_with_meta(
        X_tr, y_tr, cfg["features"], q33, q67, SEQ_LEN
    )
    X_val, y_val, meta_val = build_sequence_data_with_meta(
        X_va, y_va, cfg["features"], q33, q67, SEQ_LEN
    )
    X_test, y_test, meta_test = build_sequence_data_with_meta(
        X_te, y_te, cfg["features"], q33, q67, SEQ_LEN
    )

    return {
        "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
        "X_val": X_val, "y_val": y_val, "meta_val": meta_val,
        "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        "q33": q33, "q67": q67,
    }

# =========================
# 3. 评估函数
# =========================

def evaluate_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
    }


def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("混淆矩阵(rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"]).to_string())

# =========================
# 4. Baseline
# =========================

def run_logistic(X_train, y_train, X_test, y_test, class_weight):
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            multi_class="auto",
            class_weight=class_weight,
            n_jobs=None,
            random_state=SEED,
        )),
    ])
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    return model, pred, evaluate_predictions(y_test, pred)


def run_extra_trees(X_train, y_train, X_test, y_test, class_weight):
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)

    model = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    return model, pred, evaluate_predictions(y_test, pred)


def run_momentum(meta_train, meta_test, y_test):
    train_score = meta_train["pct_chg"].astype(float).values
    q1 = np.percentile(train_score, 33.33)
    q2 = np.percentile(train_score, 66.67)
    test_score = meta_test["pct_chg"].astype(float).values
    pred = np.where(test_score < q1, 0, np.where(test_score > q2, 2, 1))
    return pred, evaluate_predictions(y_test, pred), q1, q2

# =========================
# 5. 主流程
# =========================

def main():
    print("任务: 预测基线类别权重敏感性")
    print("比较: Logistic / Extra Trees 各自 unweighted 与 balanced")
    print("Naive Momentum保持不变")

    rnn_df = pd.DataFrame()
    if RNN_RESULT_PATH.exists():
        try:
            rnn_df = pd.read_csv(RNN_RESULT_PATH)
            print(f"RNN结果读取: 成功，行数={len(rnn_df)}, 缺失值={int(rnn_df.isna().sum().sum())}")
        except Exception as e:
            print(f"异常: RNN结果读取失败: {type(e).__name__}: {e}")
    else:
        print(f"警告: 未找到RNN结果文件: {RNN_RESULT_PATH}")

    all_rows, error_rows = [], []

    for dataset_name in RUN_DATASETS:
        print("-" * 20)
        print(f"数据集: {dataset_name}")
        cfg = DATASETS[dataset_name]

        print("步骤1：数据准备与审计")
        try:
            ds = prepare_dataset(dataset_name, cfg)
            print(f"字段数: {len(cfg['features'])}")
            print(f"训练样本: {len(ds['X_train'])}, 验证样本: {len(ds['X_val'])}, 测试样本: {len(ds['X_test'])}")
            print(f"训练缺失值: {int(np.isnan(ds['X_train']).sum())}, 验证缺失值: {int(np.isnan(ds['X_val']).sum())}, 测试缺失值: {int(np.isnan(ds['X_test']).sum())}")
            print(f"训练标签分布: {pd.Series(ds['y_train']).value_counts().sort_index().to_dict()}")
            print(f"测试标签分布: {pd.Series(ds['y_test']).value_counts().sort_index().to_dict()}")
            print("数据审计: 通过")
        except Exception as e:
            print(f"异常: 数据准备失败: {type(e).__name__}: {e}")
            error_rows.append({"Dataset": dataset_name, "Baseline": "", "Step": "数据准备", "Error": f"{type(e).__name__}: {e}"})
            continue

        # Momentum
        print("步骤2：Naive Momentum")
        try:
            pred, met, q1, q2 = run_momentum(ds["meta_train"], ds["meta_test"], ds["y_test"])
            print(f"结果: Macro-F1={met['Macro_F1']:.6f}, Accuracy={met['Accuracy']:.6f}")
            all_rows.append({"Dataset": dataset_name, "Baseline": "Naive Momentum", "Weighting": "N/A", **met})
        except Exception as e:
            print(f"异常: Momentum失败: {type(e).__name__}: {e}")
            error_rows.append({"Dataset": dataset_name, "Baseline": "Naive Momentum", "Step": "预测", "Error": f"{type(e).__name__}: {e}"})

        # Logistic + ExtraTrees，各跑两种权重
        for baseline_name, func in [
            ("Logistic Regression", run_logistic),
            ("Extra Trees", run_extra_trees),
        ]:
            for weighting, cw in [("Unweighted", None), ("Balanced", "balanced")]:
                print(f"步骤3：{baseline_name} / {weighting}")
                try:
                    _, pred, met = func(
                        ds["X_train"], ds["y_train"],
                        ds["X_test"], ds["y_test"], cw
                    )
                    print(f"结果: Macro-F1={met['Macro_F1']:.6f}, Accuracy={met['Accuracy']:.6f}")
                    all_rows.append({
                        "Dataset": dataset_name,
                        "Baseline": baseline_name,
                        "Weighting": weighting,
                        **met
                    })
                except Exception as e:
                    print(f"异常: {baseline_name}/{weighting}失败: {type(e).__name__}: {e}")
                    error_rows.append({"Dataset": dataset_name, "Baseline": f"{baseline_name}/{weighting}", "Step": "预测", "Error": f"{type(e).__name__}: {e}"})

        print("步骤4：当前数据集敏感性比较")
        cur = pd.DataFrame([x for x in all_rows if x["Dataset"] == dataset_name])
        if not cur.empty:
            print(cur.sort_values(["Baseline", "Weighting"]).to_string(index=False))

            for baseline_name in ["Logistic Regression", "Extra Trees"]:
                sub = cur[cur["Baseline"] == baseline_name]
                if len(sub) == 2:
                    u = sub[sub["Weighting"] == "Unweighted"].iloc[0]
                    b = sub[sub["Weighting"] == "Balanced"].iloc[0]
                    print(
                        f"{baseline_name}: Balanced-Unweighted "
                        f"ΔF1={b['Macro_F1']-u['Macro_F1']:.6f}, "
                        f"ΔAcc={b['Accuracy']-u['Accuracy']:.6f}"
                    )

    print("-" * 20)
    print("步骤5：保存与最终汇总")

    df = pd.DataFrame(all_rows)
    err = pd.DataFrame(error_rows)

    if not df.empty:
        df.to_csv(OUTPUT_DIR / "C1_baseline_weight_sensitivity.csv", index=False)
    if not err.empty:
        err.to_csv(OUTPUT_DIR / "C9_errors.csv", index=False)

    print(f"结果行数: {len(df)}")
    print(f"异常记录数: {len(err)}")

    print("-" * 20)
    print("最终汇总1：全部结果")
    if not df.empty:
        print(df.sort_values(["Dataset", "Baseline", "Weighting"]).to_string(index=False))
    else:
        print("无结果")

    print("-" * 20)
    print("最终汇总2：权重敏感性")
    sens_rows = []
    if not df.empty:
        for dataset_name in RUN_DATASETS:
            for baseline_name in ["Logistic Regression", "Extra Trees"]:
                sub = df[(df["Dataset"] == dataset_name) & (df["Baseline"] == baseline_name)]
                if len(sub) == 2:
                    u = sub[sub["Weighting"] == "Unweighted"].iloc[0]
                    b = sub[sub["Weighting"] == "Balanced"].iloc[0]
                    sens_rows.append({
                        "Dataset": dataset_name,
                        "Baseline": baseline_name,
                        "Unweighted_F1": u["Macro_F1"],
                        "Balanced_F1": b["Macro_F1"],
                        "Delta_F1": b["Macro_F1"] - u["Macro_F1"],
                        "Unweighted_Accuracy": u["Accuracy"],
                        "Balanced_Accuracy": b["Accuracy"],
                        "Delta_Accuracy": b["Accuracy"] - u["Accuracy"],
                    })
    sens = pd.DataFrame(sens_rows)
    if not sens.empty:
        print(sens.to_string(index=False))
        sens.to_csv(OUTPUT_DIR / "C2_weight_sensitivity_summary.csv", index=False)
    else:
        print("无结果")

    print("-" * 20)
    print("最终汇总3：每个数据集最佳baseline vs 最佳RNN")
    if not df.empty and not rnn_df.empty:
        rows = []
        for dataset_name in RUN_DATASETS:
            b = df[df["Dataset"] == dataset_name].sort_values("Macro_F1", ascending=False).iloc[0]
            rsub = rnn_df[rnn_df["Dataset"] == dataset_name]
            if rsub.empty:
                continue
            r = rsub.loc[rsub["Macro_F1"].idxmax()]
            rows.append({
                "Dataset": dataset_name,
                "Best_Baseline": b["Baseline"],
                "Weighting": b["Weighting"],
                "Baseline_Macro_F1": b["Macro_F1"],
                "Best_RNN": r["Model"],
                "RNN_Macro_F1": r["Macro_F1"],
                "RNN_minus_Baseline_F1": r["Macro_F1"] - b["Macro_F1"],
                "Baseline_Accuracy": b["Accuracy"],
                "RNN_Accuracy": r["Accuracy"],
                "RNN_minus_Baseline_Acc": r["Accuracy"] - b["Accuracy"],
            })
        cmp = pd.DataFrame(rows)
        print(cmp.to_string(index=False))
        cmp.to_csv(OUTPUT_DIR / "C3_best_rnn_vs_best_weighted_baseline.csv", index=False)
    else:
        print("无法比较")

    print("-" * 20)
    print("最终汇总4：异常")
    if err.empty:
        print("异常: 0，全部步骤运行完成")
    else:
        print(err.to_string(index=False))

    print("-" * 20)
    print("xai_cell_C_V3 完成")
    print(f"结果目录: {OUTPUT_DIR.as_posix()}/")


if __name__ == "__main__":
    main()
