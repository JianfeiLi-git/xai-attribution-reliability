# ==============================================================================
# cell_A_data_preprocessing_V1
# Re-preprocess the U.S. datasets (US-Med, US-Pow) following the Chinese-dataset pipeline:
#   1. Label alignment: convert the U.S. pct_chg_next (return on day t+1) to pct_chg (return on day t), as in the Chinese data;
#      shift(-1) is applied during training, so that, as in China, features up to day t-1 predict day t+1
#   2. First-stage feature selection: identical to feature_selection_pipeline_corrected in Chinese medical_05
#      (43 candidate features, adaptive Top-K, redundancy removal at |r|>0.95, 7 discrete time features kept directly)
#   3. Chronological split + RobustScaler standardization: identical to Chinese process_and_save_dataset (only the 7 discrete time features are skipped)
# Input files in the current Jupyter directory: ./us_medical_features_full.csv, ./us_power_features_full.csv
# Output directory: ./cell_A_data_preprocessing_V1/
# ==============================================================================

# ------------------------------ Settings (editable) ------------------------------
INPUT_FILES = {
    "us_med": "us_medical_features_full.csv",   # U.S. healthcare (US-Med in the paper)
    "us_pow": "us_power_features_full.csv",     # U.S. utilities (US-Pow in the paper)
}
OUTPUT_DIR = "stage6_cell_A_data_preprocessing_V1"

TRAIN_END_DATE = "2022-12-31"                   # same as China
VAL_START = "2023-01-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-11-28"                         # same as the Chinese code (U.S. data actually end on 2025-11-14)
REDUNDANCY_THRESHOLD = 0.95                     # same as China
PREDICT_HORIZON = 1                             # same as China
RANDOM_STATE = 42                               # same as China

# ------------------------------ Imports ------------------------------
import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
T0 = time.time()

# Rule: the first two lines print the input and output locations (relative paths only)
print("Input files: " + ", ".join(f"./{f}" for f in INPUT_FILES.values()))
print(f"Output directory: ./{OUTPUT_DIR}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def elapsed():
    # Return the elapsed time as a string
    s = int(time.time() - T0)
    return f"{s // 60}m{s % 60:02d}s"


# ------------------------------ Feature categories (identical to Chinese medical_05) ------------------------------
original_features = ["open", "close", "high", "low", "vol", "amount", "turnover_rate", "pre_close"]
time_features_numeric = ["year", "month", "day", "dayofweek", "quarter", "dayofyear", "weekofyear"]
time_features_cyclic = ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
                        "dayofyear_sin", "dayofyear_cos"]
technical_features = [
    "ma5", "ma10", "ma20", "ma60", "close_to_ma5", "close_to_ma20",
    "volume_ma5", "volume_ma10", "volume_ratio",
    "roc_5", "roc_10", "roc_20",
    "volatility_5", "volatility_10", "volatility_20",
    "high_low_range", "close_open_range", "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr", "momentum_5", "momentum_10",
]
# China: candidate features = raw + cyclic time encodings + technical indicators (43); the 7 discrete time features are not screened
candidate_features = original_features + time_features_cyclic + technical_features

print("\n" + "=" * 90)
print("Feature composition (same as the Chinese datasets)")
print("=" * 90)
print(f"  Raw market features: {len(original_features)}")
print(f"  Cyclic time-encoding features: {len(time_features_cyclic)}")
print(f"  Technical indicator features: {len(technical_features)}")
print(f"  Total candidate features for first-stage screening: {len(candidate_features)}")
print(f"  Discrete time features (not screened, kept directly): {len(time_features_numeric)}")


# ------------------------------ Step 1: label alignment ------------------------------
def align_target_to_china(df, name):
    """
    Chinese data: each row holds features from day t-1 (shift(1)); pct_chg is the return on day t (not shifted).
    Old U.S. data: pct_chg_next is the return on day t+1 (already shifted forward by one day).
    Processing: pct_chg(t) = pct_chg_next(t-1), shifted back by one row within each stock, giving pct_chg with the same meaning as in China.
    """
    print(f"\n[Step 1] Label alignment: {name}")
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    df["pct_chg"] = df.groupby("stock_code")["pct_chg_next"].shift(1)
    n_nan = int(df["pct_chg"].isna().sum())
    print(f"  Created pct_chg (return on day t); the first row of each stock has no previous row, {n_nan} missing (skipped automatically during training)")

    # Check: pct_chg in the next row should equal old pct_chg_next in the current row
    check = df.groupby("stock_code")["pct_chg"].shift(-1)
    mask = check.notna()
    max_diff = float((check[mask] - df.loc[mask, "pct_chg_next"]).abs().max())
    print(f"  Max difference between pct_chg(t+1) and old pct_chg_next(t): {max_diff:.2e} (should be 0)")

    # Print an example to check the time correspondence
    sample_code = df["stock_code"].iloc[0]
    ex = df[df["stock_code"] == sample_code][["stock_code", "trade_date", "close", "pct_chg", "pct_chg_next"]].iloc[1:6]
    print(f"  Example (stock {sample_code}; close is the previous trading day's close, pct_chg is the same-day return):")
    print(ex.to_string(index=False))
    df = df.drop(columns=["pct_chg_next"])
    return df


# ------------------------------ Step 2: first-stage feature selection (identical to China) ------------------------------
def feature_selection_pipeline_corrected(df, dataset_name, feature_cols, target_col="pct_chg",
                                         train_end_date=TRAIN_END_DATE, predict_horizon=PREDICT_HORIZON,
                                         top_k_method="adaptive",
                                         redundancy_threshold=REDUNDANCY_THRESHOLD):
    t_start = time.time()
    print(f"\n{'=' * 90}")
    print(f"[Step 2] First-stage feature selection: {dataset_name}")
    print(f"{'=' * 90}")

    # Step 0: target leakage check
    leakage_cols = [c for c in feature_cols if c in [target_col, "pct_chg", "change", "pct_chg_next"]]
    if leakage_cols:
        raise ValueError(f"Target leakage detected: {leakage_cols}")
    print("  [Step 0] Target leakage check: passed")

    # Step 1: use training-period data only
    df_train = df[df["trade_date"] <= pd.to_datetime(train_end_date)].copy()
    print(f"  [Step 1] Training period: {df_train['trade_date'].min().date()} to {df_train['trade_date'].max().date()}, "
          f"{len(df_train):,} rows, {df_train['stock_code'].nunique()} stocks")

    # Step 2: target = pct_chg shifted forward by 1 day within each stock (features in row T -> return on day T+1)
    df_train = df_train.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    df_train["target_future"] = df_train.groupby("stock_code")[target_col].shift(-predict_horizon)
    df_train_valid = df_train[~df_train["target_future"].isna()].copy()
    print(f"  [Step 2] Prediction horizon T+{predict_horizon}; {len(df_train_valid):,} valid samples"
          f" ({len(df_train) - len(df_train_valid):,} rows removed)")

    X = df_train_valid[feature_cols].copy()
    y = df_train_valid["target_future"].copy()
    stock_codes = df_train_valid["stock_code"].copy()

    # Step 3: forward-fill missing values within each stock, fill the rest with 0 (same as China)
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_before = int(X.isna().sum().sum())
    print(f"  [Step 3] Missing values before processing: {missing_before:,}")
    if missing_before > 0:
        X_filled = []
        for stock in stock_codes.unique():
            X_stock = X[stock_codes == stock].copy().ffill().fillna(0)
            X_filled.append(X_stock)
        X = pd.concat(X_filled, axis=0)
        print(f"  Missing values after processing: {int(X.isna().sum().sum())}")

    X_values = X.values
    y_values = y.values

    # Step 4: importance from six methods (same parameters as China)
    print("  [Step 4] Multi-method feature importance")
    t = time.time()
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                  random_state=RANDOM_STATE, verbose=-1, force_col_wise=True, n_jobs=-1)
    lgb_model.fit(X_values, y_values)
    lgb_importance = lgb_model.feature_importances_
    print(f"    [1/6] LightGBM done ({time.time() - t:.1f}s)")

    t = time.time()
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                 random_state=RANDOM_STATE, tree_method="hist", verbosity=0, n_jobs=-1)
    xgb_model.fit(X_values, y_values)
    xgb_importance = xgb_model.feature_importances_
    print(f"    [2/6] XGBoost done ({time.time() - t:.1f}s)")

    t = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_values, y_values)
    rf_importance = rf_model.feature_importances_
    print(f"    [3/6] Random Forest done ({time.time() - t:.1f}s)")

    t = time.time()
    f_scores, _ = f_regression(X_values, y_values)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    [4/6] F statistic done ({time.time() - t:.1f}s)")

    t = time.time()
    mi_scores = mutual_info_regression(X_values, y_values, random_state=RANDOM_STATE, n_neighbors=3)
    print(f"    [5/6] Mutual information done ({time.time() - t:.1f}s)")

    t = time.time()
    target_corr = np.array([np.corrcoef(X_values[:, i], y_values)[0, 1] for i in range(X_values.shape[1])])
    target_corr = np.nan_to_num(target_corr, nan=0.0)
    print(f"    [6/6] Pearson correlation done ({time.time() - t:.1f}s)")

    # Step 5: correlations between features
    corr_matrix = np.nan_to_num(np.corrcoef(X_values.T), nan=0.0)
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append({"Feature1": feature_cols[i], "Feature2": feature_cols[j],
                                        "Correlation": corr_matrix[i, j]})
    print(f"  [Step 5] Highly correlated feature pairs (|r|>0.9): {len(high_corr_pairs)}")

    # Step 6: ensemble score (same normalization as China)
    def normalize_scores(scores):
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-10:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "LightGBM": normalize_scores(lgb_importance),
        "XGBoost": normalize_scores(xgb_importance),
        "RandomForest": normalize_scores(rf_importance),
        "F_Statistic": normalize_scores(f_scores),
        "Mutual_Info": normalize_scores(mi_scores),
        "Target_Corr": normalize_scores(np.abs(target_corr)),
    })
    importance_df["Ensemble_Score"] = importance_df[["LightGBM", "XGBoost", "RandomForest",
                                                     "F_Statistic", "Mutual_Info", "Target_Corr"]].mean(axis=1)
    importance_df["Rank"] = importance_df["Ensemble_Score"].rank(ascending=False)
    importance_df = importance_df.sort_values("Ensemble_Score", ascending=False)

    # Step 7: adaptive Top-K (same sample-size thresholds as China)
    n_samples = len(X_values)
    if top_k_method == "adaptive":
        if n_samples < 50000:
            top_k = 15
        elif n_samples < 100000:
            top_k = 20
        elif n_samples < 200000:
            top_k = 25
        else:
            top_k = 30
        print(f"  [Step 7] Adaptive Top-K = {top_k} ({n_samples:,} training samples)")
    else:
        top_k = int(top_k_method)
        print(f"  [Step 7] Manual Top-K = {top_k}")

    top_features = importance_df.head(top_k)["Feature"].tolist()
    redundant_features = set()
    for pair in high_corr_pairs:
        if abs(pair["Correlation"]) > redundancy_threshold:
            f1, f2 = pair["Feature1"], pair["Feature2"]
            if f1 in top_features and f2 in top_features:
                s1 = importance_df[importance_df["Feature"] == f1]["Ensemble_Score"].values[0]
                s2 = importance_df[importance_df["Feature"] == f2]["Ensemble_Score"].values[0]
                redundant_features.add(f2 if s1 > s2 else f1)
    selected_features = [f for f in top_features if f not in redundant_features]

    print(f"  Redundancy removal (|r|>{redundancy_threshold}): {len(redundant_features)} -> {sorted(redundant_features)}")
    print(f"  Finally selected: {len(selected_features)}")
    print(f"\n  Top 25 features:")
    print(f"  {'Rank':<6}{'Status':<8}{'Feature':<22}{'Ensemble':<10}")
    for _, row in importance_df.head(25).iterrows():
        if row["Feature"] in selected_features:
            status = "Kept"
        elif row["Feature"] in redundant_features:
            status = "Redundant"
        else:
            status = "Not Top-K"
        print(f"  {int(row['Rank']):<6}{status:<8}{row['Feature']:<22}{row['Ensemble_Score']:.4f}")
    print(f"  First-stage time for this dataset {time.time() - t_start:.1f}s, total {elapsed()}")

    metadata = {"dataset_name": dataset_name, "train_end_date": train_end_date,
                "predict_horizon": predict_horizon, "n_train_samples": int(n_samples),
                "n_candidate_features": len(feature_cols), "top_k": top_k,
                "redundancy_threshold": redundancy_threshold,
                "n_redundant_removed": len(redundant_features),
                "n_final_selected": len(selected_features)}
    return selected_features, importance_df, high_corr_pairs, metadata


# ------------------------------ Step 3: split and standardization (identical to China) ------------------------------
def process_and_save_dataset(df, dataset_key, feature_cols, temporal_cols, output_dir):
    print(f"\n[Step 3] Split and standardization: {dataset_key}")
    target_col = "pct_chg"
    id_cols = ["stock_code", "trade_date"]
    X_df = df[id_cols + feature_cols].copy()
    y_df = df[id_cols + [target_col]].copy()

    def split_time(data):
        train = data[data["trade_date"] <= pd.to_datetime(TRAIN_END_DATE)].copy()
        val = data[(data["trade_date"] >= pd.to_datetime(VAL_START)) &
                   (data["trade_date"] <= pd.to_datetime(VAL_END))].copy()
        test = data[(data["trade_date"] >= pd.to_datetime(TEST_START)) &
                    (data["trade_date"] <= pd.to_datetime(TEST_END))].copy()
        return train, val, test

    X_train, X_val, X_test = split_time(X_df)
    y_train, y_val, y_test = split_time(y_df)
    for nm, d in [("Train", X_train), ("Validation", X_val), ("Test", X_test)]:
        print(f"  {nm}: {len(d):>7,} rows  {d['trade_date'].min().date()} to {d['trade_date'].max().date()}")

    # Only the 7 discrete time features are skipped (Chinese temporal_features contains only discrete time features)
    cols_to_scale = [c for c in feature_cols if c not in temporal_cols]
    print(f"  Standardized {len(cols_to_scale)} features, skipped {len(feature_cols) - len(cols_to_scale)} discrete time features")

    def safe_fill(d):
        d[cols_to_scale] = d.groupby("stock_code")[cols_to_scale].transform(lambda x: x.ffill().fillna(0))
        return d

    X_train, X_val, X_test = safe_fill(X_train), safe_fill(X_val), safe_fill(X_test)
    scaler = RobustScaler()
    scaler.fit(X_train[cols_to_scale])
    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    os.makedirs(output_dir, exist_ok=True)
    for nm, d in [("train_X", X_train), ("val_X", X_val), ("test_X", X_test),
                  ("train_y", y_train), ("val_y", y_val), ("test_y", y_test)]:
        d.to_csv(os.path.join(output_dir, f"{nm}.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, f"scaler_{dataset_key}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved: ./{output_dir}/train_X.csv and 5 other files, ./{OUTPUT_DIR}/scaler_{dataset_key}.pkl")
    return {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)}


# ------------------------------ Main ------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


feature_config = {}
summary = {}
for idx, (key, fname) in enumerate(INPUT_FILES.items(), 1):
    print(f"\n{'#' * 90}")
    print(f"Dataset {idx}/{len(INPUT_FILES)}: {key} (./{fname}), total {elapsed()}")
    print(f"{'#' * 90}")
    df = pd.read_csv(fname)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["stock_code"] = df["stock_code"].astype(str)
    print(f"  Read {df.shape[0]:,} rows x {df.shape[1]} columns, {df['stock_code'].nunique()} stocks, "
          f"{df['trade_date'].min().date()} to {df['trade_date'].max().date()}")
    miss_cols = [c for c in candidate_features + time_features_numeric + ["pct_chg_next"] if c not in df.columns]
    if miss_cols:
        raise ValueError(f"Input file is missing columns: {miss_cols}")

    df = align_target_to_china(df, key)
    selected, importance_df, corr_pairs, meta = feature_selection_pipeline_corrected(
        df, key, candidate_features)

    importance_df.to_csv(os.path.join(OUTPUT_DIR, f"importance_{key}.csv"), index=False)
    if corr_pairs:
        pd.DataFrame(corr_pairs).to_csv(os.path.join(OUTPUT_DIR, f"high_corr_pairs_{key}.csv"), index=False)

    # Same as China: all features = selected features + 7 discrete time features
    all_features = selected + time_features_numeric
    feature_config[key] = {"all_features": all_features, "selected_features": selected,
                           "temporal_features": time_features_numeric, "n_total": len(all_features),
                           "metadata": meta}
    print(f"\n  First-stage output features: {len(all_features)} ({len(selected)} selected + {len(time_features_numeric)} discrete time)")
    print(f"  Feature list: {all_features}")

    summary[key] = process_and_save_dataset(df, key, all_features, time_features_numeric,
                                            os.path.join(OUTPUT_DIR, key))
    summary[key]["n_features"] = len(all_features)
    print(f"  Dataset {key} done, total {elapsed()}")

feature_config["train_end_date"] = TRAIN_END_DATE
with open(os.path.join(OUTPUT_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

print("\n" + "=" * 90)
print("cell_A finished, total time " + elapsed())
print("=" * 90)
for key, res in summary.items():
    print(f"  {key}: {res['n_features']} features; train {res['n_train']:,} rows, validation {res['n_val']:,} rows, test {res['n_test']:,} rows")
    print(f"    {feature_config[key]['all_features']}")
print(f"\nOutput files: ./{OUTPUT_DIR}/feature_config.json, importance_*.csv, high_corr_pairs_*.csv, scaler_*.pkl, "
      f"train/val/test X and y under us_med/ and us_pow/")
print("Next: cell_B uses these data for second-stage Transformer permutation-importance screening.")
