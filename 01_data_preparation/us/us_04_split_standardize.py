# -*- coding: utf-8 -*-
"""
Step 4: chronological split and standardization.
"""

"""
U.S. dataset split and standardization
Aligned with the A-share logic (Step 7-8), physical X/Y separation

Input:
  stock_data_tushare/us_medical_features_full.csv
  stock_data_tushare/us_power_features_full.csv
  stock_data_tushare/feature_selection/feature_config.json

Output:
  stock_data_tushare/processed/us_medical/train_X.csv etc.
  stock_data_tushare/processed/us_power/train_X.csv etc.

Install dependencies: pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("U.S. dataset split and standardization (physical X/Y separation)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
INPUT_DIR    = "./stock_data_tushare"
CONFIG_PATH  = "./stock_data_tushare/feature_selection/feature_config.json"
OUTPUT_BASE  = "./stock_data_tushare/processed"
SCALER_DIR   = "./stock_data_tushare/scalers"

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(SCALER_DIR,  exist_ok=True)

# Time split (identical to the A-shares)
TRAIN_END  = pd.to_datetime('2022-12-31')
VAL_START  = pd.to_datetime('2023-01-01')
VAL_END    = pd.to_datetime('2024-06-30')
TEST_START = pd.to_datetime('2024-07-01')
TEST_END   = pd.to_datetime('2025-11-14')

TARGET_COL = 'pct_chg_next'   # U.S. target column name

# Dataset configuration
DATASETS = {
    "us_medical": "us_medical_features_full.csv",
    "us_power":   "us_power_features_full.csv",
}

# ============================================================
# Read the feature configuration (from the Step 6 feature selection results)
# ============================================================
print(f"\n[1] Read the feature configuration: {CONFIG_PATH}")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Feature configuration file not found; please run first feature_selection_us.py")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    feature_config = json.load(f)

print(f"  Datasets in the configuration: {[k for k in feature_config.keys() if k != 'train_end_date']}")

# ============================================================
# Core processing function
# ============================================================
def process_and_save(df, label, config_key, output_dir):
    print(f"\n{'='*70}")
    print(f"Process: {label}")
    print(f"{'='*70}")

    # ------ Get the feature list ------
    cfg = feature_config[config_key]
    selected  = cfg['selected_features']
    temporal  = cfg['temporal_features']          # ['year', 'month']
    feature_cols = cfg['all_features']            # selected + temporal

    print(f"  Total number of features: {len(feature_cols)}")
    print(f"  of which time features: {temporal}")
    print(f"  Target column: {TARGET_COL}")

    # ------ Check whether features exist ------
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️  The following features are not in the data and were skipped: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
        selected     = [c for c in selected     if c in df.columns]

    # Check the target column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column missing from the data {TARGET_COL}; please check the feature engineering output")

    # ------ Build X and y ------
    id_cols = ['stock_code', 'trade_date']
    X_df    = df[id_cols + feature_cols].copy()
    y_df    = df[id_cols + [TARGET_COL]].copy()

    X_df['trade_date'] = pd.to_datetime(X_df['trade_date'])
    y_df['trade_date'] = pd.to_datetime(y_df['trade_date'])

    # ------ Time split ------
    def split_by_time(data):
        train = data[data['trade_date'] <= TRAIN_END].copy()
        val   = data[(data['trade_date'] >= VAL_START) & (data['trade_date'] <= VAL_END)].copy()
        test  = data[(data['trade_date'] >= TEST_START) & (data['trade_date'] <= TEST_END)].copy()
        return train, val, test

    X_train, X_val, X_test = split_by_time(X_df)
    y_train, y_val, y_test = split_by_time(y_df)

    print(f"\n  Split results:")
    print(f"    Train: {len(X_train):>6,} rows  "
          f"({X_train['trade_date'].min().date()} ~ {X_train['trade_date'].max().date()})")
    print(f"    Val:   {len(X_val):>6,} rows  "
          f"({X_val['trade_date'].min().date()} ~ {X_val['trade_date'].max().date()})")
    print(f"    Test:  {len(X_test):>6,} rows  "
          f"({X_test['trade_date'].min().date()} ~ {X_test['trade_date'].max().date()})")

    # ------ Standardization (time features are not scaled)------
    cols_to_scale = [c for c in feature_cols if c not in temporal]
    print(f"\n  Standardization {len(cols_to_scale)} features (time features skipped)...")

    # Missing-value handling (forward fill within each stock)
    def safe_fill(d):
        d[cols_to_scale] = (
            d.groupby('stock_code')[cols_to_scale]
             .transform(lambda x: x.ffill().fillna(0))
        )
        return d

    X_train = safe_fill(X_train)
    X_val   = safe_fill(X_val)
    X_test  = safe_fill(X_test)

    # Training set only fit scaler
    scaler = RobustScaler()
    scaler.fit(X_train[cols_to_scale])

    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale]   = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    print(f"  ✅ Standardization complete (fit on the training set only, no future information leakage)")

    # ------ Save files ------
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f'{output_dir}/train_X.csv', index=False)
    X_val.to_csv(  f'{output_dir}/val_X.csv',   index=False)
    X_test.to_csv( f'{output_dir}/test_X.csv',  index=False)

    y_train.to_csv(f'{output_dir}/train_y.csv', index=False)
    y_val.to_csv(  f'{output_dir}/val_y.csv',   index=False)
    y_test.to_csv( f'{output_dir}/test_y.csv',  index=False)

    # Save scaler
    scaler_path = os.path.join(SCALER_DIR, f'scaler_{label}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n  💾 Saved to: {output_dir}")
    print(f"     train_X.csv ({len(X_train):,} rows) / train_y.csv")
    print(f"     val_X.csv   ({len(X_val):,} rows) / val_y.csv")
    print(f"     test_X.csv  ({len(X_test):,} rows) / test_y.csv")
    print(f"     scaler: {scaler_path}")

    # ------ Check: leakage check ------
    print(f"\n  Check (leakage prevention check):")
    print(f"    X_train contains the target column: {TARGET_COL in X_train.columns}")
    print(f"    y_train contains the feature:   {any(c in y_train.columns for c in cols_to_scale)}")
    print(f"    Train/Val Date overlap:     "
          f"{X_train['trade_date'].max() >= X_val['trade_date'].min()}")

    return {
        'n_train': len(X_train),
        'n_val':   len(X_val),
        'n_test':  len(X_test),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
    }

# ============================================================
# Main flow
# ============================================================
print(f"\n[2] Start processing dataset...")

summary = {}
for label, fname in DATASETS.items():
    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['stock_code'] = df['stock_code'].astype(str)

    output_dir = os.path.join(OUTPUT_BASE, label)
    result = process_and_save(
        df=df,
        label=label,
        config_key=label,
        output_dir=output_dir,
    )
    summary[label] = result

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("✅ All done")
print("=" * 80)
for label, res in summary.items():
    print(f"\n  {label}:")
    print(f"    Number of features:  {res['n_features']}")
    print(f"    Train:   {res['n_train']:,} rows")
    print(f"    Val:     {res['n_val']:,} rows")
    print(f"    Test:    {res['n_test']:,} rows")

print(f"\n  Output folder structure:")
print(f"  stock_data_tushare/processed/")
for label in DATASETS:
    print(f"    {label}/")
    print(f"      train_X.csv / train_y.csv")
    print(f"      val_X.csv   / val_y.csv")
    print(f"      test_X.csv  / test_y.csv")
