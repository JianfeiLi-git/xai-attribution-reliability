# ==============================================================================
# stage6_cell_D_data_baseline_V1
# For the U.S. datasets (US-Med, US-Pow) on the new data (new labels, new features):
#   1. Recompute the conventional baselines (identical to xai_cell_C_V3):
#      Naive Momentum; Logistic Regression and Extra Trees, each Unweighted / Balanced
#   2. Compare with the seed-42 RNNs selected in cell_C using a paired bootstrap (identical to stage5_cell_A_bootstrap_V4):
#      Main analysis: moving-block bootstrap over trading days (block = 20 trading days, 2000 replicates)
#      Sensitivity analysis: sample-level i.i.d. bootstrap (2000 replicates)
#      One-sided test H1: RNN > baseline; BH-FDR correction across 5 datasets (the 3 Chinese datasets use the original stage5 p-values, the 2 U.S. datasets use the new values)
# RNN predictions are read directly from the test-set predictions saved by cell_C (no re-inference, avoiding cuDNN issues), and the sample order is checked row by row
# Input: ./stage6/stage6_cell_A_data_preprocessing_V1/, ./stage6/stage6_cell_C_data_training_V2/
# Output: ./stage6/stage6_cell_D_data_baseline_V1/
# ==============================================================================

# ------------------------------ Settings (editable) ------------------------------
DATA_INPUT_DIR = "stage6/stage6_cell_A_data_preprocessing_V1"
RNN_INPUT_DIR = "stage6/stage6_cell_C_data_training_V2"
OUTPUT_DIR = "stage6/stage6_cell_D_data_baseline_V1"
DATASETS = {"us_med": "US-Med", "us_pow": "US-Pow"}   # folder name -> name in the paper
MODELS = ["GRU", "BiLSTM", "LSTM"]
SEED = 42
SEQ_LEN = 60
N_BOOT = 2000          # same as stage5
BLOCK_LEN = 20         # same as stage5
ALPHA = 0.05

# One-sided p-values of the 3 Chinese datasets from the original stage5_cell_A_bootstrap_V4 (table A3), used for BH correction across all 5 datasets
CN_P_VALUES = {
    "Medical-1": {"block": 0.001000, "iid": 0.000500},
    "Medical-2": {"block": 0.000500, "iid": 0.000500},
    "Power": {"block": 0.202399, "iid": 0.005497},
}
# Original U.S. results (stage5 table A3), printed for old-versus-new comparison only
OLD_US = {
    "US-Med": {"rnn": "LSTM", "rnn_f1": 0.364083, "base": "Extra Trees/Balanced", "base_f1": 0.359013,
               "block_p": 0.340330, "block_p_bh": 0.425412},
    "US-Pow": {"rnn": "BiLSTM", "rnn_f1": 0.364745, "base": "Logistic Regression/Unweighted", "base_f1": 0.357997,
               "block_p": 0.491254, "block_p_bh": 0.491254},
}

# ------------------------------ Imports ------------------------------
import os
import json
import math
import time
import random
import inspect
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
T0 = time.time()
# Rule: the first two lines print the input and output locations (relative paths only)
print(f"Input directories: ./{DATA_INPUT_DIR}/, ./{RNN_INPUT_DIR}/")
print(f"Output directory: ./{OUTPUT_DIR}/")

# Check all input files before running
need = [os.path.join(DATA_INPUT_DIR, "feature_config.json")]
for d in DATASETS:
    need += [os.path.join(DATA_INPUT_DIR, d, f"{s}_{xy}.csv") for s in ["train", "test"] for xy in ["X", "y"]]
    need += [os.path.join(RNN_INPUT_DIR, d, f"{m.lower()}_seed{SEED}_test_predictions.csv") for m in MODELS]
missing = [p for p in need if not os.path.isfile(p)]
if missing:
    print("\nThe following input files do not exist; check DATA_INPUT_DIR and RNN_INPUT_DIR at the top of the code:")
    for p in missing:
        print("  ./" + p)
    raise FileNotFoundError(f"{len(missing)} input files are missing")
print(f"Input file check passed ({len(need)} files)")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def elapsed():
    # Return the elapsed time as a string
    s = int(time.time() - T0)
    return f"{s // 60}m{s % 60:02d}s"


# ==================== 1. Data and sequence construction (same as xai_cell_C_V3) ====================

def load_split(data_dir, split):
    return (pd.read_csv(os.path.join(data_dir, f"{split}_X.csv")),
            pd.read_csv(os.path.join(data_dir, f"{split}_y.csv")))


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
    merged = pd.merge(X_work, y_work[["stock_code", "trade_date", "label", "pct_chg", "future_change"]],
                      on=["stock_code", "trade_date"], how="inner")
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
            meta_rows.append({"stock_code": row["stock_code"], "trade_date": row["trade_date"],
                              "label": int(target), "pct_chg": float(row["pct_chg"]),
                              "future_change": float(row["future_change"])})
    return np.stack(seqs).astype(np.float32), np.asarray(labels, dtype=np.int64), pd.DataFrame(meta_rows)


# ==================== 2. Baselines (same as xai_cell_C_V3) ====================

def evaluate_predictions(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred), "Macro_F1": f1_score(y_true, y_pred, average="macro")}


def run_logistic(X_train, y_train, X_test, class_weight):
    lr_kwargs = dict(max_iter=500, solver="lbfgs", class_weight=class_weight, n_jobs=None, random_state=SEED)
    # The original code used multi_class="auto"; newer sklearn versions removed this parameter (same default behavior), older versions receive it unchanged
    if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
        lr_kwargs["multi_class"] = "auto"
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**lr_kwargs)),
    ])
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model.predict(X_test.reshape(len(X_test), -1)).astype(np.int64)


def run_extra_trees(X_train, y_train, X_test, class_weight):
    model = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                 class_weight=class_weight, n_jobs=-1, random_state=SEED)
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model.predict(X_test.reshape(len(X_test), -1)).astype(np.int64)


def run_momentum(meta_train, meta_test):
    train_score = meta_train["pct_chg"].astype(float).values
    q1 = np.percentile(train_score, 33.33)
    q2 = np.percentile(train_score, 66.67)
    test_score = meta_test["pct_chg"].astype(float).values
    return np.where(test_score < q1, 0, np.where(test_score > q2, 2, 1)).astype(np.int64)


# ==================== 3. Bootstrap (same as stage5_cell_A_bootstrap_V4) ====================

def macro_f1_fast(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)
    true_sum = cm.sum(axis=1)
    prec = np.divide(tp, pred_sum, out=np.zeros(3), where=pred_sum > 0)
    rec = np.divide(tp, true_sum, out=np.zeros(3), where=true_sum > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros(3), where=denom > 0)
    return f1.mean()


def build_date_groups(meta_test):
    dates = pd.to_datetime(meta_test["trade_date"]).values
    uniq, inv = np.unique(dates, return_inverse=True)
    return [np.where(inv == k)[0] for k in range(len(uniq))], len(uniq)


def block_resample(rng, groups, n_dates):
    L = min(BLOCK_LEN, n_dates)
    n_blocks = int(math.ceil(n_dates / L))
    starts = rng.integers(0, n_dates - L + 1, size=n_blocks)
    sel = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n_dates]
    return np.concatenate([groups[d] for d in sel])


def iid_resample(rng, n):
    return rng.integers(0, n, size=n)


def run_bootstrap(y_test, pred_dict, groups, n_dates, scheme, tag):
    rng = np.random.default_rng(SEED + (0 if scheme == "block" else 1))
    out = {k: {"f1": np.empty(N_BOOT), "acc": np.empty(N_BOOT)} for k in pred_dict}
    n = len(y_test)
    t = time.time()
    for b in range(N_BOOT):
        idx = block_resample(rng, groups, n_dates) if scheme == "block" else iid_resample(rng, n)
        yb = y_test[idx]
        for k, p in pred_dict.items():
            pb = p[idx]
            out[k]["f1"][b] = macro_f1_fast(yb, pb)
            out[k]["acc"][b] = (pb == yb).mean()
        if (b + 1) % 250 == 0:
            per = (time.time() - t) / (b + 1)
            print(f"    [{tag}] {scheme} bootstrap {b + 1}/{N_BOOT} | est. remaining {per * (N_BOOT - b - 1):.0f}s | total {elapsed()}")
    return out


def summarize_diff(diff):
    return {"CI_Lower": float(np.percentile(diff, 2.5)), "CI_Upper": float(np.percentile(diff, 97.5)),
            "p_one_sided": float((np.sum(diff <= 0) + 1) / (len(diff) + 1))}


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


# ==================== 4. Main ====================

with open(os.path.join(DATA_INPUT_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
    config_a = json.load(f)

baseline_rows, all_rnn_rows, primary_rows = [], [], []
for d_idx, (ds, name) in enumerate(DATASETS.items(), 1):
    print(f"\n{'#' * 100}")
    print(f"Dataset {d_idx}/{len(DATASETS)}: {name} (./{DATA_INPUT_DIR}/{ds}/), total {elapsed()}")
    print(f"{'#' * 100}")
    feature_cols = config_a[ds]["selected_features"] + config_a[ds]["temporal_features"]
    X_tr, y_tr = load_split(os.path.join(DATA_INPUT_DIR, ds), "train")
    X_te, y_te = load_split(os.path.join(DATA_INPUT_DIR, ds), "test")
    q33, q67 = calculate_thresholds(y_tr)
    X_train, y_train, meta_train = build_sequence_data_with_meta(X_tr, y_tr, feature_cols, q33, q67, SEQ_LEN)
    X_test, y_test, meta_test = build_sequence_data_with_meta(X_te, y_te, feature_cols, q33, q67, SEQ_LEN)
    print(f"[Step 1] Data preparation: {len(feature_cols)} features; train {X_train.shape}, test {X_test.shape}")
    print(f"  Missing values: train {int(np.isnan(X_train).sum())}, test {int(np.isnan(X_test).sum())}")
    print(f"  Training label distribution {pd.Series(y_train).value_counts().sort_index().to_dict()}; "
          f"test label distribution {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    # Read the cell_C RNN test predictions and check sample order and labels
    preds = {}
    rnn_f1 = {}
    for m in MODELS:
        p_df = pd.read_csv(os.path.join(RNN_INPUT_DIR, ds, f"{m.lower()}_seed{SEED}_test_predictions.csv"))
        same_len = len(p_df) == len(meta_test)
        same_order = same_len and (p_df["stock_code"].astype(str).values == meta_test["stock_code"].astype(str).values).all() \
            and (pd.to_datetime(p_df["trade_date"]).values == pd.to_datetime(meta_test["trade_date"]).values).all()
        same_label = same_len and (p_df["label"].astype(int).values == y_test).all()
        if not (same_order and same_label):
            raise ValueError(f"{name} {m}: cell_C test predictions do not match the test samples rebuilt in this cell"
                             f" (same length {same_len}, same order {same_order}, same labels {same_label})")
        preds[m] = p_df["pred_label"].astype(np.int64).values
        rnn_f1[m] = f1_score(y_test, preds[m], average="macro")
        print(f"  RNN {m:<6}: sample order and labels check passed; test Macro-F1 {rnn_f1[m]:.6f}, "
              f"accuracy {accuracy_score(y_test, preds[m]):.6f}")

    # Baselines
    print(f"\n[Step 2] Conventional baselines (same as xai_cell_C_V3)")
    base_preds = {}
    t = time.time()
    p = run_momentum(meta_train, meta_test)
    base_preds[("Naive Momentum", "N/A")] = p
    print(f"  Naive Momentum: Macro-F1 {f1_score(y_test, p, average='macro'):.6f}, "
          f"accuracy {accuracy_score(y_test, p):.6f} ({time.time() - t:.1f}s)")
    for b_name, func in [("Logistic Regression", run_logistic), ("Extra Trees", run_extra_trees)]:
        for weighting, cw in [("Unweighted", None), ("Balanced", "balanced")]:
            t = time.time()
            p = func(X_train, y_train, X_test, cw)
            base_preds[(b_name, weighting)] = p
            print(f"  {b_name} / {weighting}: Macro-F1 {f1_score(y_test, p, average='macro'):.6f}, "
                  f"accuracy {accuracy_score(y_test, p):.6f} ({time.time() - t:.1f}s, total {elapsed()})")
    for (b_name, weighting), p in base_preds.items():
        met = evaluate_predictions(y_test, p)
        baseline_rows.append({"Dataset": name, "Baseline": b_name, "Weighting": weighting, **met})

    # Strongest baseline (same as the original pipeline: selected by test Macro-F1)
    best_key = max(base_preds, key=lambda k: f1_score(y_test, base_preds[k], average="macro"))
    preds["Baseline"] = base_preds[best_key]
    base_f1 = f1_score(y_test, preds["Baseline"], average="macro")
    print(f"  >>> Strongest baseline: {best_key[0]} / {best_key[1]}, Macro-F1 {base_f1:.6f}")
    cm = confusion_matrix(y_test, preds["Baseline"], labels=[0, 1, 2])
    print(f"      Confusion matrix of the strongest baseline (rows = true down/hold/up, columns = predicted):\n{cm}")

    pred_df = meta_test.copy()
    pred_df["y_true"] = y_test
    for k, v in preds.items():
        pred_df[f"pred_{k}"] = v
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"D1_predictions_{name}.csv"), index=False)

    # Bootstrap
    print(f"\n[Step 3] Paired bootstrap (block={BLOCK_LEN} trading days, {N_BOOT} replicates; i.i.d. {N_BOOT} replicates)")
    groups, n_dates = build_date_groups(meta_test)
    print(f"  {len(y_test)} test samples, {n_dates} test trading days")
    boot = {s: run_bootstrap(y_test, preds, groups, n_dates, s, name) for s in ["block", "iid"]}
    point_f1 = {k: f1_score(y_test, v, average="macro") for k, v in preds.items()}
    point_acc = {k: accuracy_score(y_test, v) for k, v in preds.items()}
    best_rnn = max(MODELS, key=lambda m: point_f1[m])

    for m in MODELS:
        row = {"Dataset": name, "RNN": m, "Baseline": best_key[0], "Baseline_Weighting": best_key[1],
               "N_Test": len(y_test), "N_Test_Dates": n_dates,
               "RNN_MacroF1": point_f1[m], "Baseline_MacroF1": point_f1["Baseline"],
               "Delta_MacroF1": point_f1[m] - point_f1["Baseline"],
               "RNN_Accuracy": point_acc[m], "Baseline_Accuracy": point_acc["Baseline"],
               "Delta_Accuracy": point_acc[m] - point_acc["Baseline"]}
        for scheme in ["block", "iid"]:
            tag = "Block" if scheme == "block" else "IID"
            s_f1 = summarize_diff(boot[scheme][m]["f1"] - boot[scheme]["Baseline"]["f1"])
            s_acc = summarize_diff(boot[scheme][m]["acc"] - boot[scheme]["Baseline"]["acc"])
            row[f"{tag}_F1_CI_Lower"], row[f"{tag}_F1_CI_Upper"] = s_f1["CI_Lower"], s_f1["CI_Upper"]
            row[f"{tag}_F1_p_one_sided"] = s_f1["p_one_sided"]
            row[f"{tag}_Acc_CI_Lower"], row[f"{tag}_Acc_CI_Upper"] = s_acc["CI_Lower"], s_acc["CI_Upper"]
            row[f"{tag}_Acc_p_one_sided"] = s_acc["p_one_sided"]
        all_rnn_rows.append(row)
        print(f"  {m:<6} vs strongest baseline: ΔF1 {row['Delta_MacroF1']:+.4f} | block 95%CI [{row['Block_F1_CI_Lower']:+.4f}, "
              f"{row['Block_F1_CI_Upper']:+.4f}] p={row['Block_F1_p_one_sided']:.4f} | i.i.d. 95%CI "
              f"[{row['IID_F1_CI_Lower']:+.4f}, {row['IID_F1_CI_Upper']:+.4f}] p={row['IID_F1_p_one_sided']:.4f}")
        if m == best_rnn:
            primary_rows.append(row)
    print(f"  >>> Main comparison: best RNN = {best_rnn} (Macro-F1 {point_f1[best_rnn]:.4f}) vs {best_key[0]}/{best_key[1]}"
          f" ({point_f1['Baseline']:.4f}), total time so far {elapsed()}")

# BH-FDR: across 5 datasets (China uses the original p-values)
df_primary = pd.DataFrame(primary_rows)
names_all = list(CN_P_VALUES.keys()) + df_primary["Dataset"].tolist()
for scheme, tag in [("block", "Block"), ("iid", "IID")]:
    p_all = [CN_P_VALUES[n][scheme] for n in CN_P_VALUES] + df_primary[f"{tag}_F1_p_one_sided"].tolist()
    adj = dict(zip(names_all, bh_fdr(p_all)))
    df_primary[f"{tag}_F1_p_BH_FDR_5datasets"] = df_primary["Dataset"].map(adj)
    if scheme == "block":
        print("\n[Step 4] BH-FDR correction (5 datasets, main block analysis):")
        for n in names_all:
            src = "original stage5" if n in CN_P_VALUES else "new value"
            raw = CN_P_VALUES[n]["block"] if n in CN_P_VALUES else \
                float(df_primary.loc[df_primary["Dataset"] == n, "Block_F1_p_one_sided"].iloc[0])
            print(f"  {n:<10} one-sided p {raw:.4f} ({src}) -> BH-adjusted {adj[n]:.4f} -> "
                  f"{'significant' if adj[n] <= ALPHA else 'not significant'}")
df_primary["Block_F1_Significant_BH_0.05"] = df_primary["Block_F1_p_BH_FDR_5datasets"] <= ALPHA

pd.DataFrame(baseline_rows).to_csv(os.path.join(OUTPUT_DIR, "D2_baselines.csv"), index=False)
df_primary.to_csv(os.path.join(OUTPUT_DIR, "D3_bootstrap_best_rnn_vs_best_baseline.csv"), index=False)
pd.DataFrame(all_rnn_rows).to_csv(os.path.join(OUTPUT_DIR, "D4_bootstrap_all_rnn_vs_best_baseline.csv"), index=False)

print("\n" + "=" * 100)
print("stage6_cell_D finished, total time " + elapsed())
print("=" * 100)
print("Summary 1: all baselines")
print(pd.DataFrame(baseline_rows).to_string(index=False))
print("\nSummary 2: best RNN vs strongest baseline (new results)")
cols = ["Dataset", "RNN", "Baseline", "Baseline_Weighting", "RNN_MacroF1", "Baseline_MacroF1", "Delta_MacroF1",
        "Block_F1_CI_Lower", "Block_F1_CI_Upper", "Block_F1_p_one_sided", "Block_F1_p_BH_FDR_5datasets",
        "Block_F1_Significant_BH_0.05", "IID_F1_p_one_sided"]
print(df_primary[cols].to_string(index=False))
print("\nSummary 3: old versus new")
for _, r in df_primary.iterrows():
    o = OLD_US[r["Dataset"]]
    print(f"  {r['Dataset']}: old {o['rnn']} {o['rnn_f1']:.4f} vs {o['base']} {o['base_f1']:.4f}, "
          f"ΔF1 {o['rnn_f1'] - o['base_f1']:+.4f}, block p {o['block_p']:.4f} (BH {o['block_p_bh']:.4f})")
    print(f"  {' ' * len(r['Dataset'])}  new {r['RNN']} {r['RNN_MacroF1']:.4f} vs {r['Baseline']}/{r['Baseline_Weighting']} "
          f"{r['Baseline_MacroF1']:.4f}, ΔF1 {r['Delta_MacroF1']:+.4f}, block p {r['Block_F1_p_one_sided']:.4f}"
          f" (BH {r['Block_F1_p_BH_FDR_5datasets']:.4f})")
print(f"\nOutput files: ./{OUTPUT_DIR}/D1_predictions_US-Med.csv, D1_predictions_US-Pow.csv, D2_baselines.csv, "
      f"D3_bootstrap_best_rnn_vs_best_baseline.csv, D4_bootstrap_all_rnn_vs_best_baseline.csv")
