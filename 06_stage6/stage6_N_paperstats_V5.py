"""
stage6_cell_N_data_paperstats_V5

Aim: final read-only check of the paper's statistics. Only frozen results are summarized and verified; no retraining, no recomputation of attributions,
and no statistical test is rerun (BH correction is only a deterministic transformation of saved raw p-values, cross-checked against the saved adjusted values).

I. Output following the existing table structure of the paper (fields consistent with the tables in 89.pdf):
   Table 1 dataset statistics; Table 2 strongest baseline vs best RNN; Table 3 feature agreement; Table 4 temporal agreement and recency;
   Table 5 label randomization; Table 6 input noise; Supplementary Table S1 all baselines; Supplementary Table S2 five-seed stability.
II. Numbers needed in the text: parameter-randomization ranges, recency proportions (per-combination counts and ratio ranges), occlusion peaks, key numbers of the two new analyses.
III. Time-alignment checks, in two parts: formal verification and supporting diagnostics:
   Formal verification (based on the actual rules of the training code, checking each file read during training):
   - build_sequence_data in the training code computes future_change = pct_chg.shift(-1) by stock on the original row order of the y file,
     then sorts by date after merging and takes windows; hence correct labels require that each y file is sorted by trade date in ascending order within each stock with no duplicate dates;
     this cell checks this condition for the X and y files of 5 datasets x 3 splits;
   - using the same rule as the training code, the first 3 eligible test windows of each dataset are taken, and the first day, the last day d and the label date of each window are printed,
     checking that the label date is the next trading day of the same stock after d;
   - Timing of features: the line-by-line audit of the preprocessing source code in the GitHub repository is cited (see SOURCE_AUDIT),
     and this cell checks that every feature finally entering the model in each dataset belongs to the "37 price/technical features with shift(1)"
     or the "calendar features" (calendar features are taken from the date of day d, known in advance, and are not future information);
     this cell runs independently and reads no source files: the location of shift(1) is cited from the line-by-line audit record of the V1.2 preprocessing source code in the GitHub repository
     (both the line of the statement and the calls of its function were checked; see SOURCE_AUDIT, SOURCE_CALLS), printed only and not used as a gate check.
   - The final Chinese feature lists (names, order, count) are the same as in the original five-seed code; the predictions of the 40 models recomputed by K V7 with these lists
     passed the item-by-item check against the original results (seed 42 identical per sample under three computation modes); with a wrong list or order the predictions could not agree;
     this cell additionally checks that the list lengths equal the feature counts in Table 1.
   - Label dates are checked over all eligible windows (not only examples): for every window counted in Table 1, the label date must be the next trading day of the same stock
     after d; the number of checked windows and anomalies is reported.
   Supporting diagnostics (not conclusive): correlations of roc_n features with n-day returns compounded from pct_chg at lags 0, 1 and 2 days.
IV. Sample counts in Table 1 follow exactly the rules of the training code and the original row order of the files; the three Chinese rows are compared one by one with the original paper, and the U.S. test sample counts with
   N_Test saved by the stage6_cell_D training pipeline, to verify the counting rules independently.
V. Cross-table consistency checks: the same number must agree across tables and sources (tolerances stated per item).
   If any gate check (Gate=True) fails, output files are still saved, but the cell stops with an error at the end and does not print "finished";
   reference checks (Gate=False, e.g., the comparison of F5 and A2 recency definitions) only list differences and do not stop the run.
All numbers are printed in the console with their source files, valid denominators and missing values.
"""

# ------------------------------ Settings (editable) ------------------------------
PROJECT_ROOT = "/home/i/2026_0712/stock_i"
CN_RESULT_DIRS = {"A": "xai_cell_A_V9", "B": "xai_cell_B_V6", "D": "xai_cell_D_V1", "E": "xai_cell_E_V2",
                  "F": "xai_cell_F_V1", "I": "xai_cell_I_V2", "K1": "xai_cell_K_V1"}
CN_SEARCH_SUBDIRS = ["", "0924"]          # checked in order: project root, project root/0924, then the current directory
STAGE5A_CANDIDATE = "stage5/cell_A_bootstrap_V4"   # checked first in the current directory, then in the project root
CN_DATA = {   # data settings identical to the original five-seed code (relative to the project root)
    "Medical-1": {"dir": "XAI_cn/datasets/processed/dataset1_A_grade", "filter_train_2010": False, "n_features": 13,
                  "market": "A-share", "sector": "Medical"},
    "Medical-2": {"dir": "XAI_cn/datasets/processed/dataset2_AB_grade", "filter_train_2010": True, "n_features": 14,
                  "market": "A-share", "sector": "Medical"},
    "Power": {"dir": "XAI_cn/e3/datasets/processed/dataset2_AB_grade", "filter_train_2010": True, "n_features": 12,
              "market": "A-share", "sector": "Utility"},
}
US_A_DIR = "stage6/stage6_cell_A_data_preprocessing_V1"
US_KEYS = {"US-Med": "us_med", "US-Pow": "us_pow"}
US_INFO = {"US-Med": {"market": "U.S.", "sector": "Healthcare"}, "US-Pow": {"market": "U.S.", "sector": "Utility"}}
US_DIRS = {"C": "stage6/stage6_cell_C_data_training_V2", "D": "stage6/stage6_cell_D_data_baseline_V1",
           "E": "stage6/stage6_cell_E_data_attribution_V2", "F": "stage6/stage6_cell_F_data_kernelshap_V4",
           "G": "stage6/stage6_cell_G_data_temporal_V2", "H": "stage6/stage6_cell_H_data_robustness_V4",
           "I": "stage6/stage6_cell_I_data_labelrandom_V2", "J": "stage6/stage6_cell_J_data_multiseed_V1"}
K7_DIR = "stage6/stage6_cell_K_data_cnmultiseed_V7"
L_DIR = "stage6/stage6_cell_L_data_crossmarket_V5"
OUTPUT_ROOT = "stage6/stage6_cell_N_data_paperstats_V5"

CN_DATASETS = ["Medical-1", "Medical-2", "Power"]
US_DATASETS = ["US-Med", "US-Pow"]
DATASETS = CN_DATASETS + US_DATASETS
MODELS = ["GRU", "BiLSTM", "LSTM"]
SEEDS = [42, 123, 777, 0, 2024]
SEQ_LEN = 60
ALPHA = 0.05
SINGLE_CLASS_TOL = 1e-6
TOL_F1 = 5e-4            # tolerance for the same model's Macro-F1 across sources (numerical difference between cuDNN and native kernels)
SEPARATE_REF = [("Medical-2", "GRU")]

# Audit record of the preprocessing source code (GitHub repository xai-attribution-reliability V1.2, checked line by line)
SOURCE_AUDIT = [
    ("china_medical/medical_05_features_selection_split.py", 379, "group[all_feature_columns] = group[all_feature_columns].shift(1)"),
    ("china_power/power_05_features_selection_split.py", 379, "group[all_feature_columns] = group[all_feature_columns].shift(1)"),
    ("us/us_02_feature_engineering.py", 215, "group[ALL_FEATURE_COLS] = group[ALL_FEATURE_COLS].shift(1)"),
    ("us/us_02_feature_engineering.py", 212, "group['pct_chg_next'] = group['pct_chg_raw'].shift(-1)"),
]
# Calls of the functions containing the statements above (showing that the statements are executed)
SOURCE_CALLS = [
    ("china_medical/medical_05_features_selection_split.py", "df = df.groupby('stock_code', group_keys=False).apply(calculate_technical_indicators_corrected)"),
    ("china_medical/medical_05_features_selection_split.py", "df_dataset1 = clean_and_engineer_features(df_raw_ds1"),
    ("china_medical/medical_05_features_selection_split.py", "df_dataset2 = clean_and_engineer_features(df_raw_ds2"),
    ("china_power/power_05_features_selection_split.py", "df = df.groupby('stock_code', group_keys=False).apply(calculate_technical_indicators_corrected)"),
    ("china_power/power_05_features_selection_split.py", "df_dataset2 = clean_and_engineer_features(df_raw_ds2"),
    ("us/us_02_feature_engineering.py", "result = calculate_features_one_stock(g)"),
]
# Columns affected by the shift(1) above (same in all three scripts: 8 raw price features + 29 technical indicators)
SHIFTED_FEATURES = ["open", "close", "high", "low", "vol", "amount", "turnover_rate", "pre_close",
                    "ma5", "ma10", "ma20", "ma60", "close_to_ma5", "close_to_ma20", "volume_ma5", "volume_ma10", "volume_ratio",
                    "roc_5", "roc_10", "roc_20", "volatility_5", "volatility_10", "volatility_20", "high_low_range", "close_open_range",
                    "rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_position",
                    "atr", "momentum_5", "momentum_10"]
CALENDAR_FEATURES = ["year", "month", "day", "dayofweek", "quarter", "dayofyear", "weekofyear",
                     "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos"]
CN_FEATURES = {   # final features identical to the original five-seed code
    "Medical-1": ["dayofyear", "day", "volume_ratio", "close_to_ma5", "vol", "low", "close_open_range", "macd_signal",
                  "close_to_ma20", "roc_20", "high_low_range", "turnover_rate", "amount"],
    "Medical-2": ["high_low_range", "turnover_rate", "amount", "bb_width", "roc_10", "roc_5", "close_to_ma20", "close_to_ma5",
                  "close_open_range", "atr", "volume_ratio", "roc_20", "dayofyear_cos", "year"],
    "Power": ["dayofyear", "atr", "roc_5", "high_low_range", "turnover_rate", "macd_signal", "close", "macd", "month",
              "close_to_ma5", "roc_10", "amount"],
}

# ------------------------------ Imports ------------------------------
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(PROJECT_ROOT)
OUTPUT_DIR = Path(OUTPUT_ROOT)
ALL_COMBOS = [(d, m) for d in DATASETS for m in MODELS]
CHECKS, TEXT = [], []


def rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return "./" + p.as_posix()


def elapsed():
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def find_cn_dir(name):
    for sub in CN_SEARCH_SUBDIRS:
        cand = ROOT / sub / name if sub else ROOT / name
        if cand.is_dir():
            return cand
    return Path(name) if Path(name).is_dir() else None


CN_DIR = {k: find_cn_dir(v) for k, v in CN_RESULT_DIRS.items()}
S5A = next((b / STAGE5A_CANDIDATE for b in [Path("."), ROOT] if (b / STAGE5A_CANDIDATE).is_dir()), None)

# Rule: the first two lines print the input and output locations
_in = [rel(CN_DIR[k]) + "/" if CN_DIR[k] else CN_RESULT_DIRS[k] + " (not found)" for k in CN_RESULT_DIRS]
_in += [(rel(S5A) + "/") if S5A else STAGE5A_CANDIDATE + " (not found)"]
_in += ["XAI_cn/ (Chinese data)", "./" + US_A_DIR + "/", "./" + K7_DIR + "/"] + ["./" + v + "/" for v in US_DIRS.values()] + ["./" + L_DIR + "/"]
print("Input directories: " + ", ".join(_in))
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")


def P(d, f):
    return None if d is None else Path(d) / f


FILES = {
    "cn_B2": P(CN_DIR["B"], "B2_ig_vs_kernelshap.csv"), "cn_B3": P(CN_DIR["B"], "B3_paired_faithfulness_summary.csv"),
    "cn_F5": P(CN_DIR["F"], "F5_temporal_method_summary.csv"), "cn_A1": P(CN_DIR["A"], "A1_parameter_randomization_sanity.csv"),
    "cn_A2": P(CN_DIR["A"], "A2_ig_recency_summary.csv"), "cn_A3": P(CN_DIR["A"], "A3_temporal_occlusion.csv"),
    "cn_E1": P(CN_DIR["E"], "E1_label_randomization_sanity.csv"), "cn_D1": P(CN_DIR["D"], "D1_input_noise_robustness_detail.csv"),
    "cn_D2": P(CN_DIR["D"], "D2_input_noise_robustness_summary.csv"), "cn_S1": P(CN_DIR["I"], "Supplementary_Table_S1_full_baselines.csv"),
    "cn_K1": P(CN_DIR["K1"], "multiseed_detail.csv"), "cn_T2": P(S5A, "A3_bootstrap_best_rnn_vs_best_baseline.csv"),
    "cn_K7": Path(K7_DIR) / "multiseed_detail_corrected.csv",
    "us_cfg": Path(US_A_DIR) / "feature_config.json", "us_sel": Path(US_DIRS["C"]) / "selected_models.csv",
    "us_D2": Path(US_DIRS["D"]) / "D2_baselines.csv", "us_T2": Path(US_DIRS["D"]) / "D3_bootstrap_best_rnn_vs_best_baseline.csv",
    "us_A1": Path(US_DIRS["E"]) / "A1_parameter_randomization_sanity.csv", "us_A2": Path(US_DIRS["E"]) / "A2_ig_recency_summary.csv",
    "us_A3": Path(US_DIRS["E"]) / "A3_temporal_occlusion.csv", "us_B2": Path(US_DIRS["F"]) / "B2_ig_vs_kernelshap.csv",
    "us_B3": Path(US_DIRS["F"]) / "B3_paired_faithfulness_summary.csv", "us_F5": Path(US_DIRS["G"]) / "F5_temporal_method_summary.csv",
    "us_D1": Path(US_DIRS["H"]) / "D1_input_noise_robustness_detail.csv", "us_D2n": Path(US_DIRS["H"]) / "D2_input_noise_robustness_summary.csv",
    "us_E1": Path(US_DIRS["I"]) / "E1_label_randomization_sanity.csv", "us_J": Path(US_DIRS["J"]) / "multiseed_detail.csv",
    "L2": Path(L_DIR) / "L2_bh_fdr.csv", "L3c": Path(L_DIR) / "L3_agreement_faithfulness_crosstab.csv",
    "L3a": Path(L_DIR) / "L3_agreement_faithfulness_association.csv", "L4s": Path(L_DIR) / "L4_stability_gap_summary.csv",
    "L4": Path(L_DIR) / "L4_stability_gap_by_combo.csv",
}
for ds, c in CN_DATA.items():
    for s in ["train", "val", "test"]:
        for xy in ["X", "y"]:
            FILES[f"cn_data_{ds}_{s}_{xy}"] = ROOT / c["dir"] / f"{s}_{xy}.csv"
for ds, k in US_KEYS.items():
    for s in ["train", "val", "test"]:
        for xy in ["X", "y"]:
            FILES[f"us_data_{ds}_{s}_{xy}"] = Path(US_A_DIR) / k / f"{s}_{xy}.csv"
_miss = [f"{k}: {'directory not found' if v is None else rel(v)}" for k, v in FILES.items() if v is None or not v.exists()]
if _miss:
    print("The following input files do not exist:")
    for m in _miss:
        print("  " + m)
    raise FileNotFoundError(f"{len(_miss)} input files are missing")
print(f"Input file check passed ({len(FILES)} files)")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read(key, **kw):
    df = pd.read_csv(FILES[key], encoding="utf-8-sig", **kw)
    return df


def pick(cn_key, us_key, cols=None, extra_keys=None, key_check=True):
    # Concatenate Chinese and U.S. results and check duplicates, missing and extra records according to the table structure (gate check)
    cn, us = read(cn_key), read(us_key)
    out = pd.concat([cn[cn["Dataset"].isin(CN_DATASETS)], us[us["Dataset"].isin(US_DATASETS)]], ignore_index=True)
    if cols:
        lack = [c for c in cols if c not in out.columns]
        if lack:
            raise KeyError(f"{cn_key}/{us_key} is missing fields: {lack}")
    if not key_check:
        return out
    keys = ["Dataset", "Model"] + [k for k, _ in (extra_keys or [])]
    expected = set()
    for d, m in ALL_COMBOS:
        rest = [[]]
        for _, vals in (extra_keys or []):
            rest = [r + [v] for r in rest for v in vals]
        for r in rest:
            expected.add(tuple([d, m] + r))
    kk = out[keys].copy()
    for k, vals in (extra_keys or []):
        if isinstance(vals[0], float):
            kk[k] = kk[k].astype(float).round(3)
    have = list(map(tuple, kk.values.tolist()))
    dup = len(have) - len(set(have))
    lack = sorted(expected - set(have), key=str)
    extra = sorted(set(have) - expected, key=str)
    check(f"{cn_key}+{us_key} records complete (by {' x '.join(keys)}, {len(expected)} expected)", dup == 0 and not lack and not extra,
          f"present {len(have)}, duplicates {dup}, missing {len(lack)}, extra {len(extra)}" + (f"; missing examples {lack[:3]}" if lack else "") +
          (f"; extra examples {extra[:3]}" if extra else ""))
    return out


def check(name, ok, detail, gate=True):
    CHECKS.append({"Check": name, "Pass": bool(ok), "Gate": gate, "Detail": detail})
    print(f"  {'Gate' if gate else 'Reference'} check {'passed' if ok else 'failed'}: {name}; {detail}")


def note(section, key, value, source):
    TEXT.append({"Section": section, "Item": key, "Value": value, "Source": source})


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(r, 1)
    return out


def show(df, title, fmt="{:.4f}"):
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda v: fmt.format(v)))


def block_label(b):
    b = int(b)
    return f"{(b - 1) * 10 + 1}–{b * 10}"


# ============================== Step 1: Table 1 and time alignment ==============================
print("\n" + "=" * 100)
print(f"Step 1: Table 1 (dataset statistics) and time-alignment checks, total {elapsed()}")
print("=" * 100)


def count_sequences(X_df, y_df, max_seq_len=SEQ_LEN):
    # Same filtering rules as build_sequence_data in the training code; counting only, no arrays built
    y_w = y_df[["stock_code", "trade_date", "pct_chg"]].copy()
    y_w["trade_date"] = pd.to_datetime(y_w["trade_date"])
    y_w["future_change"] = y_w.groupby("stock_code")["pct_chg"].shift(-1)
    y_w["label_date"] = y_w.groupby("stock_code")["trade_date"].shift(-1)        # label date obtained by the training code on the file row order
    _srt = y_w[["stock_code", "trade_date"]].sort_values(["stock_code", "trade_date"])
    _srt["true_next"] = _srt.groupby("stock_code")["trade_date"].shift(-1)        # true next trading day after sorting by date
    y_w = y_w.merge(_srt, on=["stock_code", "trade_date"], how="left")
    X_w = X_df[["stock_code", "trade_date"]].copy()
    X_w["trade_date"] = pd.to_datetime(X_w["trade_date"])
    mg = pd.merge(X_w, y_w, on=["stock_code", "trade_date"], how="inner")
    n, n_bad = 0, 0
    for _, g in mg.groupby("stock_code"):
        if len(g) < max_seq_len:
            continue
        g = g.sort_values("trade_date")
        dates = g["trade_date"].values
        valid = g["future_change"].notna().values
        same = (g["label_date"].values == g["true_next"].values)
        for end in range(max_seq_len, len(g)):
            if (dates[end - 1] - dates[end - max_seq_len]) / np.timedelta64(1, "D") > 100:
                continue
            if not valid[end - 1]:
                continue
            n += 1
            n_bad += int(not same[end - 1])
    return n, n_bad


us_cfg = json.load(open(FILES["us_cfg"], encoding="utf-8"))
t1_rows, align_rows, order_rows, window_rows, label_rows = [], [], [], [], []
PAPER_T1 = {"Medical-1": (24, 13, 38820, 7168, 6583), "Medical-2": (79, 14, 138900, 23608, 21647),
            "Power": (30, 12, 74093, 8969, 8543)}
for ds in DATASETS:
    t = time.time()
    pre = f"cn_data_{ds}" if ds in CN_DATASETS else f"us_data_{ds}"
    counts, codes = {}, set()
    for s in ["train", "val", "test"]:
        X = read(f"{pre}_{s}_X", usecols=["stock_code", "trade_date"])
        y = read(f"{pre}_{s}_y", usecols=["stock_code", "trade_date", "pct_chg"])
        # Formal verification: the training code relies on the original file row order; check stock by stock that dates are strictly ascending with no duplicates
        for nm_, df_ in [("y", y), ("X", X)]:
            dd = pd.to_datetime(df_["trade_date"], errors="coerce")
            n_bad_date = int(dd.isna().sum())
            n_bad_code = int(df_["stock_code"].isna().sum() + (df_["stock_code"].astype(str).str.strip() == "").sum())
            n_dup = int(pd.DataFrame({"c": df_["stock_code"], "d": dd}).duplicated().sum())
            dif = dd.groupby(df_["stock_code"]).diff()
            n_bad = int((dif <= pd.Timedelta(0)).sum())
            order_rows.append({"Dataset": ds, "Split": s, "File": nm_, "Rows": len(df_), "Stocks": df_["stock_code"].nunique(),
                               "Invalid_Dates": n_bad_date, "Invalid_Codes": n_bad_code, "Duplicate_Stock_Date": n_dup,
                               "Non_Increasing_Dates": n_bad})
            if n_bad or n_bad_date or n_bad_code or n_dup:
                print(f"    File problem: {ds} {s}_{nm_}: invalid dates {n_bad_date}, invalid stock codes {n_bad_code}, duplicate stock x date {n_dup}, "
                      f"dates not strictly ascending within a stock {n_bad}")
        if ds in CN_DATASETS and s == "train" and CN_DATA[ds]["filter_train_2010"]:
            X = X[pd.to_datetime(X["trade_date"]) >= pd.Timestamp("2010-01-01")]
            y = y[pd.to_datetime(y["trade_date"]) >= pd.Timestamp("2010-01-01")]
        counts[s], _nb = count_sequences(X, y)
        label_rows.append({"Dataset": ds, "Split": s, "Windows_Checked": counts[s], "Label_Date_Not_Next_Trading_Day": _nb})
        codes |= set(X["stock_code"].astype(str))
    if ds in CN_DATASETS:
        nf, mk, sec = CN_DATA[ds]["n_features"], CN_DATA[ds]["market"], CN_DATA[ds]["sector"]
    else:
        k = US_KEYS[ds]
        nf = len(us_cfg[k]["selected_features"]) + len(us_cfg[k]["temporal_features"])
        mk, sec = US_INFO[ds]["market"], US_INFO[ds]["sector"]
    t1_rows.append({"Dataset": ds, "Market": mk, "Sector": sec, "Companies": len(codes), "Features": nf,
                    "Train": counts["train"], "Val": counts["val"], "Test": counts["test"]})
    print(f"  {ds}: companies {len(codes)}, features {nf}, train {counts['train']:,}, validation {counts['val']:,}, test {counts['test']:,} ({time.time() - t:.1f}s)")
    if ds in PAPER_T1:
        check(f"Table 1 {ds} matches the original paper", tuple(t1_rows[-1][c] for c in ["Companies", "Features", "Train", "Val", "Test"]) == PAPER_T1[ds],
              f"this run {tuple(t1_rows[-1][c] for c in ['Companies', 'Features', 'Train', 'Val', 'Test'])}, original {PAPER_T1[ds]}")
    else:
        print(f"    U.S. features {nf} = first-stage selected {len(us_cfg[US_KEYS[ds]]['selected_features'])} + discrete time features "
              f"{len(us_cfg[US_KEYS[ds]]['temporal_features'])}: {us_cfg[US_KEYS[ds]]['selected_features']}")

    # Time-alignment data check: lagged correlations between roc_n and returns compounded from pct_chg
    X = read(f"{pre}_test_X")
    y = read(f"{pre}_test_y", usecols=["stock_code", "trade_date", "pct_chg"])
    mg = pd.merge(X, y, on=["stock_code", "trade_date"], how="inner").sort_values(["stock_code", "trade_date"])
    rocs = [c for c in ["roc_5", "roc_10", "roc_20"] if c in mg.columns]
    if not rocs:
        align_rows.append({"Dataset": ds, "Feature": "no roc feature", "Corr_Lag0": np.nan, "Corr_Lag1": np.nan,
                           "Corr_Lag2": np.nan, "Best_Lag": np.nan, "N": 0})
        print(f"    Time-alignment data check: the test set has no roc_5/10/20 features, so no data check is possible; relying on the code record only")
    for rc in rocs:
        n = int(rc.split("_")[1])
        gr = 1 + mg["pct_chg"].astype(float) / 100.0
        logr = np.log(gr.clip(lower=1e-6))
        cum = logr.groupby(mg["stock_code"]).transform(lambda s: s.rolling(n).sum())
        corrs = {}
        for lag in [0, 1, 2]:
            ref = np.exp(cum.groupby(mg["stock_code"]).shift(lag)) - 1
            ok = ref.notna() & mg[rc].notna()
            corrs[lag] = float(np.corrcoef(mg.loc[ok, rc].astype(float), ref[ok])[0, 1]) if ok.sum() > 100 else np.nan
        best = max(corrs, key=lambda k_: -np.inf if np.isnan(corrs[k_]) else corrs[k_])
        align_rows.append({"Dataset": ds, "Feature": rc, "Corr_Lag0": corrs[0], "Corr_Lag1": corrs[1], "Corr_Lag2": corrs[2],
                           "Best_Lag": best, "N": int(ok.sum())})
        print(f"    Time-alignment data check {rc}: correlation with the {n}-day return up to day d {corrs[0]:.4f}, up to d−1 {corrs[1]:.4f}, "
              f"up to d−2 {corrs[2]:.4f} -> highest at lag {best}")
    # Formal verification: take the first 3 eligible windows by the training-code rule (shift(-1) on the original file row order, windows taken after merging and sorting by date)
    yy = y.copy()
    yy["future_change"] = yy.groupby("stock_code")["pct_chg"].shift(-1)
    yy["label_date"] = yy.groupby("stock_code")["trade_date"].shift(-1)
    xx = X[["stock_code", "trade_date"]].copy()
    mg2 = pd.merge(xx, yy, on=["stock_code", "trade_date"], how="inner")
    got = 0
    for code, g in mg2.groupby("stock_code"):
        g = g.assign(_d=pd.to_datetime(g["trade_date"])).sort_values("_d")
        if len(g) < SEQ_LEN:
            continue
        dts, valid = g["_d"].values, g["future_change"].notna().values
        for end in range(SEQ_LEN, len(g)):
            if (dts[end - 1] - dts[end - SEQ_LEN]) / np.timedelta64(1, "D") > 100 or not valid[end - 1]:
                continue
            last = g.iloc[end - 1]
            nxt = pd.to_datetime(last["label_date"])
            later = g[g["_d"] > pd.to_datetime(last["trade_date"])]["_d"]
            ok_next = len(later) > 0 and later.min() == nxt
            window_rows.append({"Dataset": ds, "Stock": code, "Window_First": str(g.iloc[end - SEQ_LEN]["trade_date"])[:10],
                                "Window_Last_d": str(last["trade_date"])[:10], "Label_Date": str(nxt)[:10],
                                "Label_Is_Next_Trading_Day": bool(ok_next), "Label_Return": float(last["future_change"])})
            print(f"    Window under the training rule: stock {code}, first day {str(g.iloc[end - SEQ_LEN]['trade_date'])[:10]}, last day d = "
                  f"{str(last['trade_date'])[:10]} -> label date {str(nxt)[:10]} (next trading day after d: {'yes' if ok_next else 'no'}), "
                  f"return {last['future_change']:.4f}%")
            got += 1
            if got >= 3:
                break
        if got >= 3:
            break
t1 = pd.DataFrame(t1_rows)
align = pd.DataFrame(align_rows)
order_df, win_df = pd.DataFrame(order_rows), pd.DataFrame(window_rows)
order_df.to_csv(OUTPUT_DIR / "N_file_order_check.csv", index=False)
win_df.to_csv(OUTPUT_DIR / "N_label_window_check.csv", index=False)
_probs = order_df[["Invalid_Dates", "Invalid_Codes", "Duplicate_Stock_Date", "Non_Increasing_Dates"]].sum()
check("Formal verification: in the X and y files of all 15 splits, dates and stock codes are valid, stock x date is unique, and dates are strictly ascending within each stock (prerequisite for correct training labels)",
      len(order_df) == 30 and int(_probs.sum()) == 0,
      f"{len(order_df)} files checked; invalid dates {int(_probs['Invalid_Dates'])}, invalid codes {int(_probs['Invalid_Codes'])}, "
      f"duplicates {int(_probs['Duplicate_Stock_Date'])}, not ascending {int(_probs['Non_Increasing_Dates'])}")
# Timing of features: all final features must be features with shift(1) or calendar features
_frows = []
for ds in DATASETS:
    feats = CN_FEATURES[ds] if ds in CN_DATASETS else us_cfg[US_KEYS[ds]]["selected_features"] + us_cfg[US_KEYS[ds]]["temporal_features"]
    for f_ in feats:
        _frows.append({"Dataset": ds, "Feature": f_, "Type": "shift(1)" if f_ in SHIFTED_FEATURES else
                       ("calendar" if f_ in CALENDAR_FEATURES else "UNKNOWN")})
feat_df = pd.DataFrame(_frows)
_nf = feat_df.groupby("Dataset").size()
check("Final feature list lengths equal the feature counts in Table 1 (the Chinese lists are identical to the original five-seed code; predictions recomputed with them by K V7 passed the check against the original results)",
      all(int(_nf[d]) == int(t1.loc[t1["Dataset"] == d, "Features"].iloc[0]) for d in DATASETS),
      ", ".join(f"{d} {int(_nf[d])}" for d in DATASETS))
feat_df.to_csv(OUTPUT_DIR / "N_feature_timing_check.csv", index=False)
_unk = feat_df[feat_df["Type"] == "UNKNOWN"]
check("Formal verification: all features finally entering the model in the 5 datasets belong to the 37 price/technical features with shift(1) or to the calendar features",
      len(_unk) == 0, f"{len(feat_df)} (dataset x feature) in total; shift(1) {int((feat_df['Type'] == 'shift(1)').sum())}, "
      f"calendar {int((feat_df['Type'] == 'calendar').sum())}, unknown {len(_unk)}" + (f": {_unk[['Dataset', 'Feature']].values.tolist()}" if len(_unk) else ""))
print("  Preprocessing source code (citing the line-by-line audit record of GitHub repository V1.2; this cell does not read the source code):")
for f_, ln, code in SOURCE_AUDIT:
    print(f"    {f_} line {ln}: {code}")
for f_, code in SOURCE_CALLS:
    print(f"    {f_} call: {code}")
note("2.3", "Evidence on feature timing", "Line-by-line audit of the GitHub repository source code (shift(1) statements and calls of their functions) + this cell's check of the final feature classification", "SOURCE_AUDIT")
lab_df = pd.DataFrame(label_rows)
lab_df.to_csv(OUTPUT_DIR / "N_label_all_windows_check.csv", index=False)
check("Formal verification: for all eligible windows (all training, validation and test windows counted in Table 1), the label date is the next trading day of the same stock after d",
      len(lab_df) == 15 and int(lab_df["Label_Date_Not_Next_Trading_Day"].sum()) == 0,
      f"windows checked {int(lab_df['Windows_Checked'].sum()):,}, anomalies {int(lab_df['Label_Date_Not_Next_Trading_Day'].sum())}")
check("Formal verification (printed examples): label dates of the first 3 windows of each dataset under the training rule are the next trading day after d",
      len(win_df) == 15 and win_df["Label_Is_Next_Trading_Day"].all(),
      f"{len(win_df)} windows checked, {int(win_df['Label_Is_Next_Trading_Day'].sum())} passed")
t1.to_csv(OUTPUT_DIR / "N_Table1_dataset_statistics.csv", index=False)
align.to_csv(OUTPUT_DIR / "N_time_alignment_check.csv", index=False)
show(t1, "Table 1:", "{:,.0f}")
_al = align.dropna(subset=["Best_Lag"])
check("Supporting diagnostic: for each dataset, the correlation between roc features and returns is highest at lag 1 (not conclusive)", len(_al) > 0 and (_al["Best_Lag"] == 1).all(),
      f"{_al['Dataset'].nunique()}/5 datasets and {len(_al)} features checkable; highest lag: " +
      ", ".join(f"{r['Dataset']} {r['Feature']}={int(r['Best_Lag'])}" for _, r in _al.iterrows()), gate=False)
print("  Basis of the time-alignment conclusion: verified = date order and uniqueness of X and y files, label dates of windows under the training rule, classification of final features; "
      "location of feature shift(1) = source code (see the source check or audit record above); roc lagged correlation = supporting diagnostic")

# ============================== Step 2: Table 2 and Supplementary Table S1 ==============================
print("\n" + "=" * 100)
print(f"Step 2: Table 2 (strongest baseline vs best RNN) and Supplementary Table S1, total {elapsed()}")
print("=" * 100)
cn_t2, us_t2 = read("cn_T2"), read("us_T2")
cn_t2 = cn_t2[cn_t2["Dataset"].isin(CN_DATASETS)]
t2 = pd.concat([cn_t2, us_t2[us_t2["Dataset"].isin(US_DATASETS)]], ignore_index=True)
if sorted(t2["Dataset"]) != sorted(DATASETS) or t2["Dataset"].duplicated().any():
    raise ValueError(f"Table 2 datasets incomplete or duplicated: {t2['Dataset'].tolist()}")
t2["BH_p_5datasets"] = bh(t2["Block_F1_p_one_sided"].values)
_us_saved = us_t2.set_index("Dataset")["Block_F1_p_BH_FDR_5datasets"]
for ds in US_DATASETS:
    v = float(t2.loc[t2["Dataset"] == ds, "BH_p_5datasets"].iloc[0])
    check(f"Table 2 {ds} BH-adjusted p matches the value saved by stage6_cell_D", abs(v - float(_us_saved[ds])) < 1e-9, f"this run {v:.6f}, saved {float(_us_saved[ds]):.6f}")
T2 = t2[["Dataset", "Baseline", "Baseline_Weighting", "Baseline_MacroF1", "RNN", "RNN_MacroF1", "Delta_MacroF1",
         "Block_F1_CI_Lower", "Block_F1_CI_Upper", "Block_F1_p_one_sided", "BH_p_5datasets",
         "Baseline_Accuracy", "RNN_Accuracy", "Delta_Accuracy", "N_Test"]].copy()
T2["Dataset"] = pd.Categorical(T2["Dataset"], DATASETS, ordered=True)
T2 = T2.sort_values("Dataset")
T2.to_csv(OUTPUT_DIR / "N_Table2_baseline_vs_rnn.csv", index=False)
show(T2, "Table 2 (BH-adjusted p of the three Chinese rows recomputed with the new U.S. p-values):")
for _, r in T2.iterrows():
    note("3.1", f"{r['Dataset']} best RNN vs strongest baseline", f"{r['RNN']} {r['RNN_MacroF1']:.4f} vs {r['Baseline']}/{r['Baseline_Weighting']} "
         f"{r['Baseline_MacroF1']:.4f}, ΔF1 {r['Delta_MacroF1']:+.4f}, 95%CI [{r['Block_F1_CI_Lower']:+.4f}, {r['Block_F1_CI_Upper']:+.4f}], "
         f"BH p {r['BH_p_5datasets']:.4f}", "stage5 A3 / stage6 D3")
cn_s1 = read("cn_S1")
cn_s1 = cn_s1[cn_s1["Dataset"].isin(CN_DATASETS)]
us_s1 = read("us_D2")
us_s1 = us_s1[us_s1["Dataset"].isin(US_DATASETS)]
S1 = pd.concat([cn_s1[["Dataset", "Baseline", "Weighting", "Macro_F1", "Accuracy"]],
                us_s1[["Dataset", "Baseline", "Weighting", "Macro_F1", "Accuracy"]]], ignore_index=True)
S1["Best_MacroF1_in_Dataset"] = S1.groupby("Dataset")["Macro_F1"].transform("max") == S1["Macro_F1"]
S1["Best_Accuracy_in_Dataset"] = S1.groupby("Dataset")["Accuracy"].transform("max") == S1["Accuracy"]
S1.to_csv(OUTPUT_DIR / "N_TableS1_all_baselines.csv", index=False)
show(S1, "Supplementary Table S1:")
check("Supplementary Table S1 has 5 baselines per dataset", (S1.groupby("Dataset").size() == 5).all() and S1["Dataset"].nunique() == 5,
      str(S1.groupby("Dataset").size().to_dict()))
for ds in US_DATASETS:
    b = T2[T2["Dataset"] == ds].iloc[0]
    s = S1[(S1["Dataset"] == ds) & S1["Best_MacroF1_in_Dataset"]].iloc[0]
    check(f"{ds} strongest baseline in Table 2 matches the highest in Supplementary Table S1 (name, weighting, Macro-F1)",
          (s["Baseline"] == b["Baseline"]) and (s["Weighting"] == b["Baseline_Weighting"]) and abs(s["Macro_F1"] - b["Baseline_MacroF1"]) < 1e-9,
          f"Table 2 {b['Baseline']}/{b['Baseline_Weighting']} {b['Baseline_MacroF1']:.4f}, S1 {s['Baseline']}/{s['Weighting']} {s['Macro_F1']:.4f}")

# ============================== Step 3: Table 3 and perturbation test ==============================
print("\n" + "=" * 100)
print(f"Step 3: Table 3 (feature agreement) and perturbation-test numbers, total {elapsed()}")
print("=" * 100)
b2 = pick("cn_B2", "us_B2", ["Dataset", "Model", "N_Features", "IG_vs_KernelSHAP_rho", "p_value"])
L2 = read("L2")
lf = L2[L2["Family"] == "Feature_IG_vs_KernelSHAP"]
T3 = b2.merge(lf[["Dataset", "Model", "rho", "p_BH_within_family"]], on=["Dataset", "Model"], how="left")
check("Table 3 ρ matches L2", (T3["IG_vs_KernelSHAP_rho"] - T3["rho"]).abs().max() < 1e-9 and len(T3) == 15,
      f"max difference {(T3['IG_vs_KernelSHAP_rho'] - T3['rho']).abs().max():.2e}, rows {len(T3)}")
T3 = T3.rename(columns={"N_Features": "Features", "IG_vs_KernelSHAP_rho": "Spearman_rho", "p_BH_within_family": "BH_FDR_adjusted_p"})
T3["BH_p_display"] = T3["BH_FDR_adjusted_p"].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
T3 = T3[["Dataset", "Model", "Features", "Spearman_rho", "BH_FDR_adjusted_p", "BH_p_display"]]
T3.to_csv(OUTPUT_DIR / "N_Table3_feature_agreement.csv", index=False)
show(T3, "Table 3:")
cn3, us3 = T3[T3["Dataset"].isin(CN_DATASETS)], T3[T3["Dataset"].isin(US_DATASETS)]
note("3.2", "Feature ρ range (15)", f"{T3['Spearman_rho'].min():.4f}–{T3['Spearman_rho'].max():.4f}; significant after BH {int((T3['BH_FDR_adjusted_p'] < ALPHA).sum())}/15; "
     f"max adjusted p {T3['BH_FDR_adjusted_p'].max():.4f}", "L2")
note("3.3", "China / U.S. feature ρ range", f"China {cn3['Spearman_rho'].min():.4f}–{cn3['Spearman_rho'].max():.4f}; U.S. {us3['Spearman_rho'].min():.4f}–{us3['Spearman_rho'].max():.4f}", "L2")
_mx, _mn = T3.loc[T3["Spearman_rho"].idxmax()], T3.loc[T3["Spearman_rho"].idxmin()]
note("3.3", "Strongest / weakest feature agreement", f"strongest {_mx['Dataset']} {_mx['Model']} {_mx['Spearman_rho']:.4f}; weakest {_mn['Dataset']} {_mn['Model']} {_mn['Spearman_rho']:.4f}; "
     f"strongest U.S. {us3.loc[us3['Spearman_rho'].idxmax(), 'Dataset']} {us3.loc[us3['Spearman_rho'].idxmax(), 'Model']} {us3['Spearman_rho'].max():.4f}", "L2")
b3 = pick("cn_B3", "us_B3", ["Dataset", "Model", "K", "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_Q025"],
          [("K", [1, 3, 5])])
_neg = b3[b3["Extra_Delta_OriginalPredProb_Mean"] <= 0]
note("3.2", "Perturbation test (45 comparisons)", f"mean extra drop > 0: {int((b3['Extra_Delta_OriginalPredProb_Mean'] > 0).sum())}/45; "
     f"2.5th percentile > 0: {int((b3['Extra_Delta_OriginalPredProb_Q025'] > 0).sum())}/45; mean <= 0: " +
     (", ".join(f"{r['Dataset']} {r['Model']} k={r['K']} ({r['Extra_Delta_OriginalPredProb_Mean']:+.4f})" for _, r in _neg.iterrows()) or "none"), "B3")
for k in [1, 3, 5]:
    g = b3[b3["K"] == k]
    note("3.2", f"k={k} extra-drop range", f"China {g[g['Dataset'].isin(CN_DATASETS)]['Extra_Delta_OriginalPredProb_Mean'].min():.4f}–"
         f"{g[g['Dataset'].isin(CN_DATASETS)]['Extra_Delta_OriginalPredProb_Mean'].max():.4f}; U.S. "
         f"{g[g['Dataset'].isin(US_DATASETS)]['Extra_Delta_OriginalPredProb_Mean'].min():.4f}–{g[g['Dataset'].isin(US_DATASETS)]['Extra_Delta_OriginalPredProb_Mean'].max():.4f}; "
         f"2.5th percentile > 0: {int((g['Extra_Delta_OriginalPredProb_Q025'] > 0).sum())}/15", "B3")

# ============================== Step 4: Table 4 and recency ==============================
print("\n" + "=" * 100)
print(f"Step 4: Table 4 (temporal agreement and recency) and recency proportions, total {elapsed()}")
print("=" * 100)
f5 = pick("cn_F5", "us_F5", ["Dataset", "Model", "IG_Last10_Mass", "IG_First10_Mass", "IG_Last10_to_First10", "KernelSHAP_Last10_Mass",
                             "KernelSHAP_First10_Mass", "Timestep_Spearman", "IG_Peak_Block", "KernelSHAP_Peak_Block", "Same_Peak_Block"])
a3 = pick("cn_A3", "us_A3", ["Dataset", "Model", "Block", "Delta_OriginalPredProb"], [("Block", [1, 2, 3, 4, 5, 6])])
occ = a3.loc[a3.groupby(["Dataset", "Model"])["Delta_OriginalPredProb"].idxmax(), ["Dataset", "Model", "Block"]].rename(columns={"Block": "Occlusion_Peak_Block"})
T4 = f5.merge(occ, on=["Dataset", "Model"], how="left")
lt = L2[L2["Family"] == "Temporal_IG_vs_KernelSHAP"]
_chk4 = T4.merge(lt[["Dataset", "Model", "rho", "p_BH_within_family"]], on=["Dataset", "Model"])
check("Table 4 temporal ρ matches L2", (_chk4["Timestep_Spearman"] - _chk4["rho"]).abs().max() < 1e-9 and len(_chk4) == 15,
      f"max difference {(_chk4['Timestep_Spearman'] - _chk4['rho']).abs().max():.2e}")
T4["IG_Last10_gt_First10"] = T4["IG_Last10_Mass"] > T4["IG_First10_Mass"]
T4["KS_Last10_gt_First10"] = T4["KernelSHAP_Last10_Mass"] > T4["KernelSHAP_First10_Mass"]
for c in ["IG_Peak_Block", "KernelSHAP_Peak_Block", "Occlusion_Peak_Block"]:
    T4[c.replace("_Block", "_Days")] = T4[c].map(block_label)
T4o = T4[["Dataset", "Model", "IG_Last10_Mass", "KernelSHAP_Last10_Mass", "Timestep_Spearman", "IG_Peak_Days", "KernelSHAP_Peak_Days",
          "Occlusion_Peak_Days", "IG_Last10_to_First10", "IG_Last10_gt_First10", "KS_Last10_gt_First10"]]
T4o.to_csv(OUTPUT_DIR / "N_Table4_temporal_recency.csv", index=False)
show(T4o, "Table 4 (the original table has only one Peak Block column; IG, KernelSHAP and occlusion peaks are given here, to choose the definition when revising the paper):")
_ig_no = T4[~T4["IG_Last10_gt_First10"]]
note("3.4", "IG last 10 days > first 10 days (50 samples, same definition as the original text and Table 4)", f"{int(T4['IG_Last10_gt_First10'].sum())}/15" +
     (f"; not satisfied: {', '.join(_ig_no['Dataset'] + ' ' + _ig_no['Model'])}" if len(_ig_no) else ""), "F5")
_r = T4[T4["IG_Last10_gt_First10"]]
note("3.4", "IG ratio range (satisfying combinations)", f"{_r['IG_Last10_to_First10'].min():.2f} ({_r.loc[_r['IG_Last10_to_First10'].idxmin(), 'Dataset']} "
     f"{_r.loc[_r['IG_Last10_to_First10'].idxmin(), 'Model']})–{_r['IG_Last10_to_First10'].max():.2f} ({_r.loc[_r['IG_Last10_to_First10'].idxmax(), 'Dataset']} "
     f"{_r.loc[_r['IG_Last10_to_First10'].idxmax(), 'Model']}); range over all 15 combinations {T4['IG_Last10_to_First10'].min():.2f}–{T4['IG_Last10_to_First10'].max():.2f}", "F5")
_ks_no = T4[~T4["KS_Last10_gt_First10"]]
note("3.4", "KernelSHAP last 10 days > first 10 days", f"{int(T4['KS_Last10_gt_First10'].sum())}/15" +
     (f"; not satisfied: {', '.join(_ks_no['Dataset'] + ' ' + _ks_no['Model'])}" if len(_ks_no) else ""), "F5")
for c, nm in [("IG_Peak_Block", "IG peak"), ("KernelSHAP_Peak_Block", "KernelSHAP peak"), ("Occlusion_Peak_Block", "Occlusion peak")]:
    no = T4[T4[c] != 6]
    note("3.4", f"{nm} in days 51–60", f"{int((T4[c] == 6).sum())}/15" + (f"; exceptions: " + ", ".join(f"{r['Dataset']} {r['Model']} ({block_label(r[c])})" for _, r in no.iterrows()) if len(no) else ""),
         "F5 / A3")
note("3.4", "Same IG and KernelSHAP peak block", f"{int(T4['Same_Peak_Block'].astype(bool).sum())}/15", "F5")
note("3.4", "Temporal ρ range and max adjusted p", f"{T4['Timestep_Spearman'].min():.4f}–{T4['Timestep_Spearman'].max():.4f}; significant after BH "
     f"{int((_chk4['p_BH_within_family'] < ALPHA).sum())}/15; max adjusted p {_chk4['p_BH_within_family'].max():.3g}", "L2")
a2 = pick("cn_A2", "us_A2", ["Dataset", "Model", "IG_Last10_Mass", "IG_First10_Mass", "IG_Last10_to_First10"])
_fa = f5[["Dataset", "Model", "IG_Last10_Mass", "IG_First10_Mass", "IG_Last10_to_First10"]].merge(
    a2[["Dataset", "Model", "IG_Last10_Mass", "IG_First10_Mass", "IG_Last10_to_First10"]], on=["Dataset", "Model"], suffixes=("_F5", "_A2"))
_fa["Gt_F5"] = _fa["IG_Last10_Mass_F5"] > _fa["IG_First10_Mass_F5"]
_fa["Gt_A2"] = _fa["IG_Last10_Mass_A2"] > _fa["IG_First10_Mass_A2"]
from scipy.stats import spearmanr as _sp
_rr = _sp(_fa["IG_Last10_to_First10_F5"], _fa["IG_Last10_to_First10_A2"])[0]
_dif = _fa[_fa["Gt_F5"] != _fa["Gt_A2"]]
_fa.to_csv(OUTPUT_DIR / "N_recency_F5_vs_A2.csv", index=False)
check("Reference: IG recency judgments of F5 (50 samples) and A2 (500 samples) agree", len(_fa) == 15 and len(_dif) == 0,
      f"combinations judged differently {len(_dif)}" + (f" ({', '.join(_dif['Dataset'] + ' ' + _dif['Model'])})" if len(_dif) else "") +
      f"; combination-level Spearman ρ of the two ratios = {_rr:.3f}; max Last10 mass difference {(_fa['IG_Last10_Mass_F5'] - _fa['IG_Last10_Mass_A2']).abs().max():.3f}", gate=False)
note("3.4", "Cross-check: IG recency on 500 samples (A2)", f"last 10 days > first 10 days {int((a2['IG_Last10_Mass'] > a2['IG_First10_Mass']).sum())}/15; "
     f"ratio range {a2['IG_Last10_to_First10'].min():.2f}–{a2['IG_Last10_to_First10'].max():.2f}", "A2")

# ============================== Step 5: randomization, Table 5, Table 6 ==============================
print("\n" + "=" * 100)
print(f"Step 5: parameter randomization, Table 5 (label randomization), Table 6 (input noise), total {elapsed()}")
print("=" * 100)
a1 = pick("cn_A1", "us_A1", ["Dataset", "Model", "Stage", "Map_Spearman_vs_Original", "Feature_Spearman_vs_Original"],
          key_check=False)   # multi-stage table, checked by the dedicated check below
fin = []
_dupst = int(a1.duplicated(["Dataset", "Model", "Stage"]).sum())
_combo_a1 = set(zip(a1["Dataset"], a1["Model"]))
_bad_a1 = []
for (d, m), g in a1.groupby(["Dataset", "Model"]):
    stages = g["Stage"].astype(str).tolist()
    f_ = [s_ for s_ in stages if re.fullmatch(r"Randomize_FC_RNN_L0_to_L\d+", s_)]
    layers = [int(x) for s_ in stages for x in re.findall(r"_L(\d+)", s_)]
    if "Original" not in stages or len(f_) != 1 or not layers or int(re.findall(r"_to_L(\d+)$", f_[0])[0]) != max(layers):
        _bad_a1.append(f"{d} {m}")
        continue
    fin.append(g[g["Stage"] == f_[0]].iloc[0])
fin = pd.DataFrame(fin)
check("Parameter randomization: each of the 15 combinations has Original and a unique final stage, the final-stage layer number equals the largest layer number, and (combination x stage) has no duplicates (same standard as M V2)",
      _combo_a1 == set(ALL_COMBOS) and _dupst == 0 and not _bad_a1 and len(fin) == 15,
      f"combinations {len(_combo_a1)}, duplicates {_dupst}, invalid {_bad_a1 or 'none'}")
note("3.5", "Final parameter-randomization stage (15)", f"feature ρ {fin['Feature_Spearman_vs_Original'].min():.3f}–{fin['Feature_Spearman_vs_Original'].max():.3f}; "
     f"map ρ {fin['Map_Spearman_vs_Original'].min():.3f}–{fin['Map_Spearman_vs_Original'].max():.3f}", "A1")
e1 = pick("cn_E1", "us_E1", ["Dataset", "Model", "RandomLabel_MacroF1", "Original_MacroF1", "Map_Spearman", "Feature_Spearman", "Temporal_Spearman"])
T5 = e1[["Dataset", "Model", "RandomLabel_MacroF1", "Map_Spearman", "Feature_Spearman", "Temporal_Spearman", "Original_MacroF1"]].copy()
T5["Dataset"] = pd.Categorical(T5["Dataset"], DATASETS, ordered=True)
T5["Model"] = pd.Categorical(T5["Model"], MODELS, ordered=True)
T5 = T5.sort_values(["Dataset", "Model"])
T5.to_csv(OUTPUT_DIR / "N_Table5_label_randomization.csv", index=False)
show(T5, "Table 5 (cells with missing correlations are written as n/a in the paper):")
_val = T5.dropna(subset=["Map_Spearman", "Feature_Spearman", "Temporal_Spearman"])
_na = T5[T5[["Map_Spearman", "Feature_Spearman", "Temporal_Spearman"]].isna().any(axis=1)]
note("3.5", "Label randomization", f"random-label Macro-F1 {T5['RandomLabel_MacroF1'].min():.3f}–{T5['RandomLabel_MacroF1'].max():.3f}; original models "
     f"{T5['Original_MacroF1'].min():.3f}–{T5['Original_MacroF1'].max():.3f}; mean ρ ({len(_val)} combinations) map {_val['Map_Spearman'].mean():.3f}, "
     f"feature {_val['Feature_Spearman'].mean():.3f}, temporal {_val['Temporal_Spearman'].mean():.3f}; highest temporal ρ {_val['Temporal_Spearman'].max():.3f}"
     f" ({_val.loc[_val['Temporal_Spearman'].idxmax(), 'Dataset']} {_val.loc[_val['Temporal_Spearman'].idxmax(), 'Model']}); not applicable: "
     + (", ".join(_na["Dataset"].astype(str) + " " + _na["Model"].astype(str)) or "none"), "E1")
_usv = _val[_val["Dataset"].isin(US_DATASETS)]
note("3.5", "U.S. temporal ρ after label randomization", f"{_usv['Temporal_Spearman'].min():.3f}–{_usv['Temporal_Spearman'].max():.3f} ({len(_usv)} combinations)", "E1")
d2 = pick("cn_D2", "us_D2n", ["Dataset", "Model", "Noise_Percent_SD", "Map_Spearman_Mean", "Feature_Spearman_Mean", "Temporal_Spearman_Mean",
                              "Pred_Agreement_Mean", "Delta_MacroF1_Mean", "Delta_Accuracy_Mean"], [("Noise_Percent_SD", [0.5, 1.0, 2.0])])
T6 = d2.groupby("Noise_Percent_SD", as_index=False).agg(Map=("Map_Spearman_Mean", "mean"), Feature=("Feature_Spearman_Mean", "mean"),
                                                        Temporal=("Temporal_Spearman_Mean", "mean"), Prediction_Agreement=("Pred_Agreement_Mean", "mean"),
                                                        Delta_MacroF1=("Delta_MacroF1_Mean", "mean"), Delta_Accuracy=("Delta_Accuracy_Mean", "mean"),
                                                        N_Combos=("Model", "size"))
d1 = pick("cn_D1", "us_D1", ["Dataset", "Model", "Noise_Percent_SD", "Repeat", "Temporal_Spearman", "Pred_Agreement"],
          [("Noise_Percent_SD", [0.5, 1.0, 2.0]), ("Repeat", sorted(read("cn_D1")["Repeat"].unique().tolist()))])
mins = d1.groupby("Noise_Percent_SD", as_index=False).agg(Min_Temporal=("Temporal_Spearman", "min"), Min_Pred_Agreement=("Pred_Agreement", "min"),
                                                          N_Repeats=("Repeat", "size"))
T6 = T6.merge(mins, on="Noise_Percent_SD")
T6.to_csv(OUTPUT_DIR / "N_Table6_input_noise.csv", index=False)
show(T6, "Table 6 (the two Min_ columns are minima over all combinations x repetitions, used in the text):", "{:.6f}")
check("Table 6 has 15 combinations and 45 repetitions per noise level", (T6["N_Combos"] == 15).all() and (T6["N_Repeats"] == 45).all(),
      f"combinations {T6['N_Combos'].tolist()}, repetitions {T6['N_Repeats'].tolist()}")
for _, r in T6.iterrows():
    note("3.5", f"Noise {r['Noise_Percent_SD']:g}%", f"map {r['Map']:.5f}, feature {r['Feature']:.5f}, temporal {r['Temporal']:.5f}, prediction agreement {r['Prediction_Agreement']:.4f}; "
         f"min temporal ρ {r['Min_Temporal']:.4f}, min prediction agreement {r['Min_Pred_Agreement']:.4f}", "D2 / D1")
_md = d1.loc[d1.groupby("Noise_Percent_SD")["Temporal_Spearman"].idxmin(), ["Noise_Percent_SD", "Dataset", "Model", "Temporal_Spearman"]]
print("  Combinations with the minimum temporal ρ at each noise level:")
print(_md.to_string(index=False))

# ============================== Step 6: Supplementary Table S2 ==============================
print("\n" + "=" * 100)
print(f"Step 6: Supplementary Table S2 (five-seed stability), total {elapsed()}")
print("=" * 100)
SC = ["Dataset", "Model", "Seed", "MacroF1", "Accuracy", "Prediction_Agreement_vs_42", "Feature_Spearman_vs_42",
      "Temporal_Spearman_vs_42", "IG_Last10_Mass", "IG_First10_Mass"]
k7 = read("cn_K7")
k1 = read("cn_K1").rename(columns={"Prediction_Agreement_vs_NewSeed42": "Prediction_Agreement_vs_42",
                                   "Feature_Spearman_vs_NewSeed42": "Feature_Spearman_vs_42",
                                   "Temporal_Spearman_vs_NewSeed42": "Temporal_Spearman_vs_42"})
jj = read("us_J")
seed = pd.concat([k7[k7["Dataset"].isin(CN_DATASETS)][SC].assign(Source="stage6_cell_K_data_cnmultiseed_V7"),
                  k1[(k1["Dataset"] == "Medical-2") & (k1["Model"] == "GRU")][SC].assign(Source="xai_cell_K_V1"),
                  jj[jj["Dataset"].isin(US_DATASETS)][SC].assign(Source="stage6_cell_J_data_multiseed_V1")], ignore_index=True)
_cnt = seed.groupby(["Dataset", "Model"])["Seed"].apply(lambda s: sorted(s.tolist()))
check("Five-seed: 5 seeds for each of the 15 combinations", len(_cnt) == 15 and all(v == sorted(SEEDS) for v in _cnt.values), f"combinations {len(_cnt)}")
seed["SingleClass"] = (seed["MacroF1"] - 2 * seed["Accuracy"] / (3 * (1 + seed["Accuracy"]))).abs() < SINGLE_CLASS_TOL
srows = []
for d, m in ALL_COMBOS:
    g = seed[(seed["Dataset"] == d) & (seed["Model"] == m)]
    alt = g[g["Seed"] != 42]
    status = ("Reconstructed configuration" if (d, m) in SEPARATE_REF else
              "Fully recoverable" if d in CN_DATASETS else "Selected from candidate configurations")
    for tag, gg, aa in [("All seeds", g, alt), ("Excluding flagged seeds", g[~g["SingleClass"]], alt[~alt["SingleClass"]])]:
        if tag != "All seeds" and not g["SingleClass"].any():
            continue
        srows.append({"Dataset": d, "Model": m, "Status": status, "Seed_Set": tag, "N_Seeds": len(gg),
                      "N_Flagged_Other_Seeds": int(alt["SingleClass"].sum()),
                      "MacroF1_Mean": gg["MacroF1"].mean(), "MacroF1_SD": gg["MacroF1"].std(ddof=1) if len(gg) > 1 else np.nan,
                      "Feature_rho_Mean": aa["Feature_Spearman_vs_42"].mean() if len(aa) else np.nan,
                      "Feature_rho_Min": aa["Feature_Spearman_vs_42"].min() if len(aa) else np.nan,
                      "Temporal_rho_Mean": aa["Temporal_Spearman_vs_42"].mean() if len(aa) else np.nan,
                      "Temporal_rho_Min": aa["Temporal_Spearman_vs_42"].min() if len(aa) else np.nan,
                      "Agreement_Mean": aa["Prediction_Agreement_vs_42"].mean() if len(aa) else np.nan,
                      "Last10_Mass_Mean": gg["IG_Last10_Mass"].mean(),
                      "Last10_gt_First10": "All seeds" if (gg["IG_Last10_Mass"] > gg["IG_First10_Mass"]).all() else
                      f"{int((gg['IG_Last10_Mass'] > gg['IG_First10_Mass']).sum())}/{len(gg)} seeds"})
S2 = pd.DataFrame(srows)
S2.to_csv(OUTPUT_DIR / "N_TableS2_multiseed.csv", index=False)
show(S2, "Supplementary Table S2 (combinations containing seeds consistent with single-class prediction have an extra row \"excluding these seeds\"):")
_main = S2[(S2["Seed_Set"] == "All seeds") & (S2["Status"] != "Reconstructed configuration")]
for mk, ds_ in [("China", CN_DATASETS), ("U.S.", US_DATASETS)]:
    g = _main[_main["Dataset"].isin(ds_)]
    note("3.5", f"{mk} five-seed ({len(g)} combinations, all seeds)", f"Macro-F1 SD {g['MacroF1_SD'].min():.4f}–{g['MacroF1_SD'].max():.4f}; mean feature ρ "
         f"{g['Feature_rho_Mean'].min():.4f}–{g['Feature_rho_Mean'].max():.4f}; mean temporal ρ {g['Temporal_rho_Mean'].min():.4f}–{g['Temporal_rho_Mean'].max():.4f}; "
         f"mean prediction agreement {g['Agreement_Mean'].min():.4f}–{g['Agreement_Mean'].max():.4f}; last 10 days > first 10 days for all seeds "
         f"{int((g['Last10_gt_First10'] == 'All seeds').sum())}/{len(g)}", "K V7 / J V1")
_fl = seed[seed["SingleClass"]]
note("3.5", "Seeds consistent with single-class prediction", (", ".join(f"{r['Dataset']} {r['Model']} seed{r['Seed']}" for _, r in _fl.iterrows()) or "none") +
     f" ({len(_fl)} in total)", "K V7 / K V1 / J V1")

# ============================== Step 7: numbers of the new analyses ==============================
print("\n" + "=" * 100)
print(f"Step 7: key numbers of the two new analyses (from L V5), total {elapsed()}")
print("=" * 100)
L3c, L3a, L4s = read("L3c"), read("L3a"), read("L4s")
for k in [1, 3, 5]:
    c = L3c[(L3c["K"] == k) & (L3c["Scope"] == "All")].iloc[0]
    a = L3a[(L3a["K"] == k) & (L3a["Scope"] == "All")].iloc[0]
    note("3.6", f"k={k} agreement significant and lower-tail criterion positive / non-positive", f"{int(c['Agreement_Sig_and_Q025_Positive'])} / {int(c['Agreement_Sig_and_Q025_NotPositive'])}"
         f" (determinable {int(c['N_Complete'])}); after removing identical draws {int(c['Sens_Agreement_Sig_and_Q025NI_Positive'])} / {int(c['Sens_Agreement_Sig_and_Q025NI_NotPositive'])}; "
         f"combination-level Spearman r = {a['Spearman_FeatureAgreement_vs_ExtraDrop']:.3f} (p = {a['p_FeatureAgreement']:.3f}, n = {int(a['N_Combos'])})", "L3")
for ss in ["Main_Include_All", "Sensitivity_Exclude_Flagged"]:
    for mt in ["Feature", "Temporal", "Prediction"]:
        for sc in ["All", "China", "US"]:
            r = L4s[(L4s["Seed_Set"] == ss) & (L4s["Metric"] == mt) & (L4s["Scope"] == sc)]
            if len(r):
                r = r.iloc[0]
                note("3.7", f"{ss} {sc} {mt}", f"positive gap {int(r['N_Gap_Positive'])}/{int(r['N_Combos'])}; median {r['Gap_Median']:.3f} ({r['Gap_Min']:.3f}–{r['Gap_Max']:.3f}); "
                     f"noise mean {r['Noise_Mean']:.4f}, seed mean {r['Seed_Mean_of_Means']:.4f}; descriptive Wilcoxon p {r['Wilcoxon_p_Descriptive']:.4g}", "L4")

# ============================== Step 8: cross-table consistency ==============================
print("\n" + "=" * 100)
print(f"Step 8: cross-table consistency checks, total {elapsed()}")
print("=" * 100)
s42 = seed[seed["Seed"] == 42].set_index(["Dataset", "Model"])["MacroF1"]
for _, r in T2.iterrows():
    v = float(s42[(str(r["Dataset"]), r["RNN"])])
    check(f"{r['Dataset']} Table 2 RNN Macro-F1 matches five-seed seed 42 (tolerance {TOL_F1})", abs(v - r["RNN_MacroF1"]) <= TOL_F1,
          f"Table 2 {r['RNN_MacroF1']:.6f}, five-seed {v:.6f}")
sel = read("us_sel")
for _, r in sel.iterrows():
    ds = {v: k for k, v in US_KEYS.items()}[r["dataset"]]
    v = float(s42[(ds, r["arch"])])
    check(f"{ds} {r['arch']} cell C test Macro-F1 matches five-seed seed 42 (cell C keeps 4 decimals)", abs(v - r["test_macro_f1"]) <= 6e-5,
          f"cell C {r['test_macro_f1']:.4f} (from candidate {r['source']}), five-seed {v:.6f}")
for _, r in e1.iterrows():
    key = (r["Dataset"], r["Model"])
    if key in SEPARATE_REF:
        continue
    v = float(s42[key])
    check(f"{key[0]} {key[1]} Table 5 original-model Macro-F1 matches five-seed seed 42 (tolerance {TOL_F1})", abs(v - r["Original_MacroF1"]) <= TOL_F1,
          f"Table 5 {r['Original_MacroF1']:.6f}, five-seed {v:.6f}")
for ds in DATASETS:
    n1 = int(t1.loc[t1["Dataset"] == ds, "Test"].iloc[0])
    n2 = int(T2.loc[T2["Dataset"] == ds, "N_Test"].iloc[0])
    check(f"{ds} Table 1 test sample count matches Table 2", n1 == n2, f"Table 1 {n1}, Table 2 {n2}")

# ============================== Summary ==============================
print("\n" + "=" * 100)
print(f"Summary, total {elapsed()}")
print("=" * 100)
text_df, chk_df = pd.DataFrame(TEXT), pd.DataFrame(CHECKS)
text_df.to_csv(OUTPUT_DIR / "N_text_numbers.csv", index=False)
chk_df.to_csv(OUTPUT_DIR / "N_consistency_checks.csv", index=False)
print("Numbers for the text (by section):")
for sec, g in text_df.groupby("Section", sort=False):
    print(f"\n[{sec}]")
    for _, r in g.iterrows():
        print(f"  {r['Item']}: {r['Value']}  [source {r['Source']}]")
n_fail = int((~chk_df["Pass"]).sum())
n_gate_fail = int((~chk_df["Pass"] & chk_df["Gate"]).sum())
print(f"\n{len(chk_df)} checks in total ({int(chk_df['Gate'].sum())} gate), {len(chk_df) - n_fail} passed, "
      f"{n_fail} failed (of which {n_gate_fail} gate)")
if n_fail:
    print(chk_df[~chk_df["Pass"]].to_string(index=False))
print(f"Output files under ./{OUTPUT_DIR.as_posix()}/: N_Table1–6, N_TableS1, N_TableS2, N_text_numbers.csv, N_consistency_checks.csv, "
      f"N_time_alignment_check.csv, N_file_order_check.csv, N_label_window_check.csv, N_recency_F5_vs_A2.csv")
if n_gate_fail:
    print("Error: some gate checks failed; the numbers above must not be written into the paper (output files saved for troubleshooting)")
    raise AssertionError(f"Audit failed: {n_gate_fail} gate checks failed")
print(f"All gate checks passed; stage6_cell_N_data_paperstats_V5 finished, total time {elapsed()}")
