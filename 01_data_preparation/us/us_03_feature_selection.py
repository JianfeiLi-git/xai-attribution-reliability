# -*- coding: utf-8 -*-
"""
Step 3: first-stage feature ranking.
"""

"""
U.S. feature selection script
Based on the A-share feature selection code, oil features removed, aligned with the U.S. data format

Input: stock_data_tushare/us_medical_features_full.csv
      stock_data_tushare/us_power_features_full.csv
Output: stock_data_tushare/feature_selection/

Install dependencies: pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "=" * 80)
print("U.S. feature selection (aligned with the A-share logic, no oil features)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
INPUT_DIR  = "./stock_data_tushare"
OUTPUT_DIR = "./stock_data_tushare/feature_selection"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)

TRAIN_END_DATE = '2022-12-31'
TARGET_COL     = 'pct_chg_next'

FILES = {
    "us_medical": "us_medical_features_full.csv",
    "us_power":   "us_power_features_full.csv",
}

# ============================================================
# Feature category definitions (oil features removed, aligned with the U.S. data)
# ============================================================
original_features = [
    'open', 'close', 'high', 'low', 'vol', 'amount',
    'turnover_rate', 'pre_close'
]

time_features_numeric = [
    'year', 'month', 'day', 'dayofweek', 'quarter',
    'dayofyear', 'weekofyear'
]

time_features_cyclic = [
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
]

technical_features = [
    'ma5', 'ma10', 'ma20', 'ma60',
    'close_to_ma5', 'close_to_ma20',
    'volume_ma5', 'volume_ma10', 'volume_ratio',
    'roc_5', 'roc_10', 'roc_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'high_low_range', 'close_open_range',
    'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'atr',
    'momentum_5', 'momentum_10',
]

# Candidate features (no oil features)
candidate_features = original_features + time_features_numeric + time_features_cyclic + technical_features

print(f"\n Feature composition:")
print(f"  Raw market features:          {len(original_features)}")
print(f"  Time features (cyclic encoding): {len(time_features_cyclic)}")
print(f"  Technical indicator features:          {len(technical_features)}")
print(f"  Total number of candidate features:          {len(candidate_features)}")

# ============================================================
# JSON encoder (handles numpy types)
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)

# ============================================================
# Main feature selection function
# ============================================================
def feature_selection_pipeline(df, dataset_name, feature_cols,
                                target_col=TARGET_COL,
                                train_end_date=TRAIN_END_DATE,
                                # redundancy_threshold=0.95,
                               redundancy_threshold=1.0,
                                top_k=50,
                                verbose=True):

    if verbose:
        print(f"\n{'='*70}")
        print(f"Feature selection: {dataset_name}")
        print(f"{'='*70}")

    # ------ Step 0: target leakage check ------
    leakage = [c for c in feature_cols if c in [target_col, 'pct_chg', 'change']]
    if leakage:
        raise ValueError(f"Target leakage detected: {leakage}")
    if verbose:
        print(f"\n[Step 0] Target leakage check: ✅ Passed")

    # ------ Step 1: temporal isolation, training set only ------
    df_train = df[df['trade_date'] <= pd.to_datetime(train_end_date)].copy()
    if verbose:
        print(f"\n[Step 1] Training set: {len(df_train):,} rows | "
              f"{df_train['trade_date'].min().date()} ~ {df_train['trade_date'].max().date()} | "
              f"{df_train['stock_code'].nunique()} stocks")

    # ------ Step 2: build the feature matrix and target ------
    df_train = df_train.sort_values(['stock_code', 'trade_date']).reset_index(drop=True)

    # Keep only the columns of feature_cols that exist in df
    valid_cols = [c for c in feature_cols if c in df_train.columns]
    missing_cols = [c for c in feature_cols if c not in df_train.columns]
    if missing_cols and verbose:
        print(f"  ⚠️  The following features are not in the data and were skipped: {missing_cols}")

    valid_mask = df_train[target_col].notna()
    df_train = df_train[valid_mask].copy()

    X = df_train[valid_cols].copy()
    y = df_train[target_col].copy()

    if verbose:
        print(f"\n[Step 2] Feature matrix: {X.shape} | Target: {y.shape}")

    # ------ Step 3: missing-value handling (within each stock) ------
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_before = X.isna().sum().sum()

    if missing_before > 0:
        X_filled = []
        for stock, grp in X.groupby(df_train['stock_code']):
            grp = grp.ffill().fillna(0)
            X_filled.append(grp)
        X = pd.concat(X_filled).loc[X.index]

    if verbose:
        print(f"\n[Step 3] Missing values: {missing_before} → {X.isna().sum().sum()}")

    X_values = X.values
    y_values = y.values

    # ------ Step 4: multi-method feature importance ------
    if verbose:
        print(f"\n[Step 4] Compute feature importance...")

    print("  [1/6] LightGBM...")
    lgb_imp = lgb.LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1, n_jobs=-1
    ).fit(X_values, y_values).feature_importances_

    print("  [2/6] XGBoost...")
    xgb_imp = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, tree_method='hist', verbosity=0, n_jobs=-1
    ).fit(X_values, y_values).feature_importances_

    print("  [3/6] Random Forest...")
    rf_imp = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    ).fit(X_values, y_values).feature_importances_

    print("  [4/6] F-statistic...")
    f_scores, _ = f_regression(X_values, y_values)
    f_scores = np.nan_to_num(f_scores)

    print("  [5/6] Mutual Information...")
    mi_scores = mutual_info_regression(X_values, y_values, random_state=42)

    print("  [6/6] Pearson Correlation...")
    corr_scores = np.array([
        abs(np.corrcoef(X_values[:, i], y_values)[0, 1])
        for i in range(X_values.shape[1])
    ])
    corr_scores = np.nan_to_num(corr_scores)

    # ------ Step 5: correlation analysis ------
    corr_matrix = np.nan_to_num(np.corrcoef(X_values.T))
    high_corr_pairs = []
    for i in range(len(valid_cols)):
        for j in range(i+1, len(valid_cols)):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append({
                    'Feature1': valid_cols[i],
                    'Feature2': valid_cols[j],
                    'Correlation': corr_matrix[i, j]
                })
    if verbose:
        print(f"\n[Step 5] Highly correlated feature pairs (|r|>0.9): {len(high_corr_pairs)}")

    # ------ Step 6: ensemble score ------
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-10)

    importance_df = pd.DataFrame({
        'Feature':      valid_cols,
        'LightGBM':     norm(lgb_imp),
        'XGBoost':      norm(xgb_imp),
        'RandomForest': norm(rf_imp),
        'F_Statistic':  norm(f_scores),
        'Mutual_Info':  norm(mi_scores),
        'Target_Corr':  norm(corr_scores),
    })
    importance_df['Ensemble_Score'] = importance_df[[
        'LightGBM','XGBoost','RandomForest',
        'F_Statistic','Mutual_Info','Target_Corr'
    ]].mean(axis=1)
    importance_df['Rank'] = importance_df['Ensemble_Score'].rank(ascending=False).astype(int)
    importance_df = importance_df.sort_values('Ensemble_Score', ascending=False).reset_index(drop=True)

    # ------ Step 7: feature screening ------
    top_features = importance_df.head(top_k)['Feature'].tolist()

    # Remove redundancy
    redundant = set()
    for pair in high_corr_pairs:
        f1, f2 = pair['Feature1'], pair['Feature2']
        if abs(pair['Correlation']) > redundancy_threshold:
            if f1 in top_features and f2 in top_features:
                s1 = importance_df[importance_df['Feature']==f1]['Ensemble_Score'].values[0]
                s2 = importance_df[importance_df['Feature']==f2]['Ensemble_Score'].values[0]
                redundant.add(f2 if s1 > s2 else f1)

    selected = [f for f in top_features if f not in redundant]

    if verbose:
        print(f"\n[Step 7] Screening results:")
        print(f"  Candidate features: {len(valid_cols)}")
        print(f"  Top-K Initial selection: {top_k}")
        print(f"  Redundancy removal: {len(redundant)}")
        print(f"  Final selection: {len(selected)}")
        print(f"\n  Top-20 Features:")
        print(f"  {'Rank':<5} {'Status':<10} {'Feature name':<25} {'Score':<10}")
        print(f"  {'-'*52}")
        for _, row in importance_df.head(20).iterrows():
            status = "✓ Keep" if row['Feature'] in selected else "✗ Remove"
            print(f"  {row['Rank']:>4}  {status:<10} {row['Feature']:<25} {row['Ensemble_Score']:.4f}")

    metadata = {
        'dataset_name':        dataset_name,
        'train_end_date':      train_end_date,
        'n_train_samples':     len(X_values),
        'n_candidate_features':len(valid_cols),
        'top_k':               top_k,
        'redundancy_threshold':redundancy_threshold,
        'n_redundant_removed': len(redundant),
        'n_final_selected':    len(selected),
        'timestamp':           datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return selected, importance_df, high_corr_pairs, metadata

# ============================================================
# Main flow: run feature selection for each dataset
# ============================================================
results = {}

for label, fname in FILES.items():
    path = os.path.join(INPUT_DIR, fname)
    print(f"\n{'='*70}")
    print(f"Read: {path}")
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['stock_code'] = df['stock_code'].astype(str)
    print(f"shape: {df.shape} | Number of stocks: {df['stock_code'].nunique()}")

    selected, importance_df, corr_pairs, metadata = feature_selection_pipeline(
        df=df,
        dataset_name=label,
        feature_cols=candidate_features,
        top_k=50,
        verbose=True
    )

    # Add the discrete time features
    # time_keep = ['year', 'month']
    # all_features = selected + time_keep

    time_keep = [f for f in time_features_numeric if f not in selected]
    all_features = selected + time_keep

    results[label] = {
        'df':           df,
        'selected':     selected,
        'all_features': all_features,
        'importance':   importance_df,
        'corr_pairs':   corr_pairs,
        'metadata':     metadata,
    }

    # Save importance
    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, f"importance_{label}.csv"), index=False)

    # Save correlated pairs
    if corr_pairs:
        pd.DataFrame(corr_pairs).to_csv(
            os.path.join(OUTPUT_DIR, f"high_corr_pairs_{label}.csv"), index=False)

    # Save the selected feature list
    feature_info = []
    for i, feat in enumerate(all_features, 1):
        if feat in original_features:       cat = 'Original'
        elif feat in time_features_numeric: cat = 'Temporal'
        elif feat in time_features_cyclic:  cat = 'Temporal_Cyclic'
        elif feat in technical_features:    cat = 'Technical'
        else:                               cat = 'Unknown'
        feature_info.append({'Index': i, 'Feature': feat, 'Category': cat})

    pd.DataFrame(feature_info).to_csv(
        os.path.join(OUTPUT_DIR, f"selected_features_{label}.csv"), index=False)

    print(f"\n  Final number of features: {len(all_features)} (Selected {len(selected)} + Time {len(time_keep)})")

# ============================================================
# Save the summary configuration
# ============================================================
feature_config = {}
for label, res in results.items():
    feature_config[label] = {
        'all_features':      res['all_features'],
        'selected_features': res['selected'],
        # 'temporal_features': ['year', 'month'],
        'temporal_features': time_features_numeric + time_features_cyclic,
        'n_total':           len(res['all_features']),
    }
feature_config['train_end_date'] = TRAIN_END_DATE

with open(os.path.join(OUTPUT_DIR, 'feature_config.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print(f"\n✓ feature_config.json Saved")

# ============================================================
# Visualization: feature importance comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Feature Importance Analysis - US Stocks', fontsize=14, fontweight='bold')

for idx, (label, res) in enumerate(results.items()):
    ax = axes[idx]
    imp = res['importance'].head(20).copy()
    imp = imp.sort_values('Ensemble_Score', ascending=True)
    colors = ['#2E86AB' if f in res['selected'] else '#CCCCCC' for f in imp['Feature']]
    ax.barh(range(len(imp)), imp['Ensemble_Score'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['Feature'], fontsize=9)
    ax.set_xlabel('Ensemble Importance Score')
    ax.set_title(f'({chr(97+idx)}) {label}', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.0)
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='black', label='Selected'),
        Patch(facecolor='#CCCCCC', edgecolor='black', label='Not Selected')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
fig_path = "./visualizations/feature_importance_us.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved: {fig_path}")
plt.close()

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 70)
print("✅ Feature selection complete")
print("=" * 70)
for label, res in results.items():
    print(f"\n  {label}:")
    print(f"    Final number of features: {len(res['all_features'])}")
    print(f"    Feature list:   {res['all_features']}")

print(f"\n💾 Output files:")
print(f"  {OUTPUT_DIR}/importance_*.csv")
print(f"  {OUTPUT_DIR}/selected_features_*.csv")
print(f"  {OUTPUT_DIR}/feature_config.json")
print(f"  ./visualizations/feature_importance_us.png")
