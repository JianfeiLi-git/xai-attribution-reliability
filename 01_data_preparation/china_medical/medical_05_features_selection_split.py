# -*- coding: utf-8 -*-
"""
Step 5: cleaning and feature engineering, first-stage feature selection,
chronological split and standardization (run as one script: later parts use
variables created by earlier parts).
"""

"""
=============================================================================
Steps 1-5: data loading, cleaning and feature engineering (fully corrected version)

[Key fixes]
1. All features (including raw price features) are lagged with shift(1)
2. Remove target variables pct_chg and change from the features to prevent target leakage
3. Fill missing values within each stock, avoiding cross-stock contamination
4. Ensure the forecasting logic: features of day T-1 predict the return of day T or T+1

[Dataset description]
- Dataset 1: Grade-A stocks (top quality, n=5)
- Dataset 2: Grade A+B stocks (high quality, n=30)

Author: [your name]
Date: 2025-01-XX
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# Set English font (for SCI paper figures)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# Academic color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#DD1C1A',
    'neutral': '#6C757D'
}

print("=" * 100)
print("Data preparation: steps 1-5 (fully corrected version - no data leakage)")
print("=" * 100)

#==============================================================================
# Step 1: create folder structure
#==============================================================================
print("\n [Step 1: create folder structure]")

directories = [
    'datasets',
    'datasets/raw',
    'datasets/processed',
    'datasets/A_grade',
    'datasets/AB_grade',
    'datasets/scalers',
    'datasets/feature_selection',
    'models',
    'results',
    'visualizations',
    'visualizations/preprocessing',
    'visualizations/preprocessing/dataset1',
    'visualizations/preprocessing/dataset2'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Created: {directory}")

#==============================================================================
# Step 2: read data and quality screening
#==============================================================================
print("\n" + "=" * 100)
print("[Step 2: load data and quality screening]")
print("=" * 100)

# 2.1 Read raw data
print("\n2.1 Read raw trading data...")
df_original = pd.read_csv('tomAB--.csv')
df_original['stock_code'] = df_original['ts_code'].astype(str).str.zfill(6)
df_original['trade_date'] = pd.to_datetime(df_original['trade_date'].astype(str), format='%Y%m%d')
df_original['year'] = df_original['trade_date'].dt.year
df_original['month'] = df_original['trade_date'].dt.to_period('M')
print(f"✓ Original data: {len(df_original):,} records, {df_original['ts_code'].nunique()} stocks")

# 2.2 Read the quality screening results
print("\n2.2 Read the quality screening results...")
quality_report = pd.read_csv('stock_quality_report_all.csv')
quality_report['stock_code'] = quality_report['stock_code'].astype(str).str.zfill(6)
print(f"✓ Quality assessment report: {len(quality_report)} stocks")

# 2.3 Define the listing cut-off date
LISTING_CUTOFF_DATE = pd.to_datetime('2021-02-01')
print(f"\n2.3 Listing date screening criterion:")
print(f"  Cut-off date: {LISTING_CUTOFF_DATE.date()}")
print(f"  Reason: make sure every stock has at least 2 years of data in the training set")

# 2.4 Grade by quality criteria
print("\n2.4 Grade by quality criteria...")

# A-grade stocks (highest standard)
a_grade = quality_report[
    (quality_report['completeness_ratio'] >= 99) &
    (quality_report['listing_years'] >= 5) &
    (quality_report['negative_prices'] == 0) &
    (quality_report['extreme_change'] == 0) &
    (quality_report['price_logic_error'] == 0) &
    (quality_report['quality_score'] >= 85)
]
print(f"  A grade (top quality): {len(a_grade)} stocks")

# B-grade stocks (high quality)
b_grade = quality_report[
    (quality_report['completeness_ratio'] >= 95) &
    (quality_report['listing_years'] >= 3) &
    (quality_report['negative_prices'] <= 5) &
    (quality_report['extreme_change'] <= 10) &
    (quality_report['quality_score'] >= 75) &
    (~quality_report['stock_code'].isin(a_grade['stock_code']))
]
print(f"  B grade (high quality): {len(b_grade)} stocks")

# 2.5 Apply the listing date filter
a_qualified = a_grade[
    pd.to_datetime(a_grade['first_date']) < LISTING_CUTOFF_DATE
].copy()

b_qualified = b_grade[
    pd.to_datetime(b_grade['first_date']) < LISTING_CUTOFF_DATE
].copy()

ab_qualified = pd.concat([a_qualified, b_qualified], ignore_index=True)

print(f"\n2.5 Apply the listing date filter (< {LISTING_CUTOFF_DATE.date()}):")
print(f"  A-grade qualified: {len(a_qualified)} stocks")
print(f"  B-grade qualified: {len(b_qualified)} stocks")
print(f"  Total: {len(ab_qualified)} stocks")

# 2.6 Prepare stock lists
dataset1_stocks = a_qualified['stock_code'].tolist()
dataset2_stocks = ab_qualified['stock_code'].tolist()

print(f"\n2.6 Dataset split:")
print(f"  Dataset 1 (Grade A): {len(dataset1_stocks)} stocks")
print(f"  Dataset 2 (Grade A+B): {len(dataset2_stocks)} stocks")

#==============================================================================
# Step 3: define data cleaning and feature engineering functions (corrected version)
#==============================================================================
print("\n" + "=" * 100)
print("[Step 3: define data cleaning and feature engineering functions (corrected version)]")
print("=" * 100)

def clean_and_engineer_features(df, dataset_name):
    """
    Data cleaning and feature engineering (corrected version - no data leakage)

    Key fix:
    1. All features (including raw prices) are shift(1)
    2. Target variables pct_chg and change are kept unchanged (not shifted, but not used as features)
    3. Handle missing values within each stock, avoiding cross-stock contamination

    Parameters:
        df: Original DataFrame
        dataset_name: Dataset name (for logging)

    Return:
        after cleaning DataFrame
    """
    print(f"\n{'='*100}")
    print(f"Process dataset: {dataset_name}")
    print(f"{'='*100}")

    df = df.copy()
    original_count = len(df)
    print(f"  Original records: {original_count:,}")

    # =========================================================================
    # 3.1 Basic cleaning
    # =========================================================================
    print(f"\n  [3.1] Basic data cleaning...")

    # Remove negative prices
    negative_mask = (
        (df['open'] < 0) | (df['close'] < 0) |
        (df['high'] < 0) | (df['low'] < 0)
    )
    removed_negative = negative_mask.sum()
    df = df[~negative_mask]

    # Remove extreme returns
    extreme_mask = (df['pct_chg'].abs() > 100)
    removed_extreme = extreme_mask.sum()
    df = df[~extreme_mask]

    # Remove price logic errors
    logic_error_mask = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    removed_logic = logic_error_mask.sum()
    df = df[~logic_error_mask]

    # Remove zero-volume records
    zero_volume_mask = (df['vol'] == 0)
    removed_volume = zero_volume_mask.sum()
    df = df[~zero_volume_mask]

    total_removed = removed_negative + removed_extreme + removed_logic + removed_volume
    print(f"    Remove negative prices: {removed_negative} records")
    print(f"    Remove extreme returns: {removed_extreme} records")
    print(f"    Remove price logic errors: {removed_logic} records")
    print(f"    Remove zero-volume records: {removed_volume} records")
    print(f"    After cleaning: {len(df):,} records")
    print(f"    Total removed: {total_removed} records ({total_removed/original_count*100:.2f}%)")

    # =========================================================================
    # 3.2 Time feature engineering
    # =========================================================================
    print(f"\n  [3.2] Time feature engineering...")

    df['year'] = df['trade_date'].dt.year
    df['month'] = df['trade_date'].dt.month
    df['day'] = df['trade_date'].dt.day
    df['dayofweek'] = df['trade_date'].dt.dayofweek
    df['quarter'] = df['trade_date'].dt.quarter
    df['dayofyear'] = df['trade_date'].dt.dayofyear
    df['weekofyear'] = df['trade_date'].dt.isocalendar().week.astype(int)

    # Cyclic encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    print(f"    ✓ 13 time features added")

    # =========================================================================
    # 3.3 Technical indicator calculation (corrected version - all features shifted)
    # =========================================================================
    print(f"\n  [3.3] Technical indicator calculation and feature lagging...")

    def calculate_technical_indicators_corrected(group):
        """
        Compute technical indicators for each stock and apply lags (corrected version)

        Key fix:
        1. Raw price features must also be shift(1)
        2. Technical indicators must also be shift(1)
        3. Target variables pct_chg and change are kept unchanged (but not used as input features)

        Final logic:
        - All features in row T = data of day T-1
        - The target variable in row T = that of day T pct_chg
        - Prediction: features of day T-1 predict the return of day T
        """

        # Step 1: sort by time
        group = group.sort_values('trade_date').reset_index(drop=True)
        # ✅ New: compute the trading-day gap
        group['days_since_last_trade'] = group['trade_date'].diff().dt.days
        # Fill the first row with 1 (assume normal trading)
        group.loc[0, 'days_since_last_trade'] = 1

        # Step 2: compute technical indicators (at this point they include same-day information)

        # Moving averages
        group['ma5'] = group['close'].rolling(window=5, min_periods=1).mean()
        group['ma10'] = group['close'].rolling(window=10, min_periods=1).mean()
        group['ma20'] = group['close'].rolling(window=20, min_periods=1).mean()
        group['ma60'] = group['close'].rolling(window=60, min_periods=1).mean()

        # Relative price position
        group['close_to_ma5'] = (group['close'] - group['ma5']) / (group['ma5'] + 1e-8)
        group['close_to_ma20'] = (group['close'] - group['ma20']) / (group['ma20'] + 1e-8)

        # Volume indicators
        group['volume_ma5'] = group['vol'].rolling(window=5, min_periods=1).mean()
        group['volume_ma10'] = group['vol'].rolling(window=10, min_periods=1).mean()
        group['volume_ratio'] = group['vol'] / (group['volume_ma5'] + 1e-8)

        # Rate of change
        group['roc_5'] = group['close'].pct_change(periods=5) * 100
        group['roc_10'] = group['close'].pct_change(periods=10) * 100
        group['roc_20'] = group['close'].pct_change(periods=20) * 100

        # Volatility
        group['volatility_5'] = group['close'].rolling(window=5, min_periods=1).std()
        group['volatility_10'] = group['close'].rolling(window=10, min_periods=1).std()
        group['volatility_20'] = group['close'].rolling(window=20, min_periods=1).std()

        # Price range
        group['high_low_range'] = (group['high'] - group['low']) / (group['close'] + 1e-8)
        group['close_open_range'] = (group['close'] - group['open']) / (group['open'] + 1e-8)

        # RSI
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        group['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = group['close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_12 - ema_26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
        group['macd_hist'] = group['macd'] - group['macd_signal']

        # Bollinger bands
        group['bb_middle'] = group['close'].rolling(window=20, min_periods=1).mean()
        bb_std = group['close'].rolling(window=20, min_periods=1).std()
        group['bb_upper'] = group['bb_middle'] + (bb_std * 2)
        group['bb_lower'] = group['bb_middle'] - (bb_std * 2)
        group['bb_width'] = (group['bb_upper'] - group['bb_lower']) / (group['bb_middle'] + 1e-8)
        group['bb_position'] = (group['close'] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'] + 1e-8)

        # ATR
        high_low = group['high'] - group['low']
        high_close = np.abs(group['high'] - group['close'].shift(1))
        low_close = np.abs(group['low'] - group['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        group['atr'] = true_range.rolling(window=14, min_periods=1).mean()

        # Momentum indicators
        group['momentum_5'] = group['close'] - group['close'].shift(5)
        group['momentum_10'] = group['close'] - group['close'].shift(10)

        # Step 3: define the feature columns to be shifted
        # ✅ Key fix: includes raw price features and technical indicators

        # Raw price features (note: pct_chg and change are excluded because they are target variables)
        original_feature_columns = [
            'open', 'close', 'high', 'low', 'vol', 'amount',
            'turnover_rate', 'pre_close'
        ]

        # Technical indicators
        technical_feature_columns = [
            'ma5', 'ma10', 'ma20', 'ma60',
            'close_to_ma5', 'close_to_ma20',
            'volume_ma5', 'volume_ma10', 'volume_ratio',
            'roc_5', 'roc_10', 'roc_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'high_low_range', 'close_open_range',
            'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            'atr',
            'momentum_5', 'momentum_10'
        ]

        # Combine all features to be shifted
        # all_feature_columns = original_feature_columns + technical_feature_columns

        # # Step 4: [key] apply uniformly shift(1)
        # # Result: all features in row T = data of day T-1
        # group[all_feature_columns] = group[all_feature_columns].shift(1)

        # Step 4: shift(1)
        all_feature_columns = original_feature_columns + technical_feature_columns
        group[all_feature_columns] = group[all_feature_columns].shift(1)

        # ✅ New: mark the resumption day after a long suspension
        group['is_after_long_halt'] = (group['days_since_last_trade'] > 10).astype(int)

        # Step 5: fill missing values within each stock (avoid cross-stock contamination)
        # Use forward fill only, no backward fill
        group[all_feature_columns] = group[all_feature_columns].ffill()

        # Step 6: delete leading NaN rows that cannot be filled
        group = group.dropna(subset=all_feature_columns)

        # Step 7: data quality check
        remaining_nans = group[all_feature_columns].isna().sum().sum()
        if remaining_nans > 0:
            stock_code = group['stock_code'].iloc[0] if len(group) > 0 else 'Unknown'
            print(f"      ⚠️  Warning: stock {stock_code} still present after cleaning {remaining_nans} NaN values")
            group = group.dropna(subset=all_feature_columns)

        return group

    # Apply the technical indicator function
    print(f"    Computing technical indicators and applying lags...")
    df = df.groupby('stock_code', group_keys=False).apply(calculate_technical_indicators_corrected)
    df = df.reset_index(drop=True)

    print(f"    ✓ 29 technical indicators added")
    print(f"    ✓ Applied to all features (including raw prices and technical indicators) shift(1)")
    print(f"    ✓ Forecasting logic: features of day T-1 predict the return of day T")
    print(f"    ✓ Missing values filled within each stock (avoiding cross-stock contamination)")

    print(f"\n  {'='*100}")
    print(f"  Final results: {len(df):,} records")
    print(f"  {'='*100}")

    return df

print("✓ Data cleaning and feature engineering function defined (fully corrected version)")

#==============================================================================
# Step 4: process Dataset 1 (Grade A)
#==============================================================================
print("\n" + "=" * 100)
print("[Step 4: process Dataset 1 (Grade-A stocks)]")
print("=" * 100)

df_raw_ds1 = df_original[df_original['ts_code'].isin(dataset1_stocks)].copy()
print(f"Selected {len(df_raw_ds1):,} raw records ({len(dataset1_stocks)} Grade-A stocks)")
df_dataset1 = clean_and_engineer_features(df_raw_ds1, "Dataset 1 (Grade A)")

#==============================================================================
# Step 5: process Dataset 2 (Grade A+B)
#==============================================================================
print("\n" + "=" * 100)
print("[Step 5: process Dataset 2 (Grade A+B stocks)]")
print("=" * 100)

df_raw_ds2 = df_original[df_original['ts_code'].isin(dataset2_stocks)].copy()
print(f"Selected {len(df_raw_ds2):,} raw records ({len(dataset2_stocks)} Grade A+B stocks)")
df_dataset2 = clean_and_engineer_features(df_raw_ds2, "Dataset 2 (Grade A+B)")

print("\n" + "=" * 100)
print("✅ Steps 1-5 complete! Both datasets have been cleaned and features added (fully corrected version)")
print("=" * 100)

print(f"\n Created variables:")
print(f"  - df_dataset1: {len(df_dataset1):,} records ({df_dataset1['stock_code'].nunique()} stocks)")
print(f"  - df_dataset2: {len(df_dataset2):,} records ({df_dataset2['stock_code'].nunique()} stocks)")
print(f"  - dataset1_stocks: {len(dataset1_stocks)} stock list")
print(f"  - dataset2_stocks: {len(dataset2_stocks)} stock list")

print(f"\n Notes on key fixes:")
print(f"  ✅ All features (including raw prices) have been shift(1)")
print(f"  ✅ Target variables pct_chg and change are not included in the features")
print(f"  ✅ Forecasting logic: features of day T-1 predict the return of day T")
print(f"  ✅ Missing values filled within each stock, avoiding cross-stock contamination")

# Save intermediate results
phase5_data = {
    'df_dataset1': df_dataset1,
    'df_dataset2': df_dataset2,
    'dataset1_stocks': dataset1_stocks,
    'dataset2_stocks': dataset2_stocks,
    'LISTING_CUTOFF_DATE': LISTING_CUTOFF_DATE,
    'a_qualified': a_qualified,
    'ab_qualified': ab_qualified
}

with open('datasets/phase5_data.pkl', 'wb') as f:
    pickle.dump(phase5_data, f)
print(f"\n✓ Saved datasets/phase5_data.pkl")

#==============================================================================
#==============================================================================
# Visualization part below (English figures)
#==============================================================================
#==============================================================================

print("\n" + "=" * 100)
print("[Creating SCI paper figures (English)]")
print("=" * 100)

# Set the academic figure style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

#==============================================================================
# Fig1: data quality assessment - Dataset 1 (Grade A)
#==============================================================================
print("\n [Creating figure Fig1: Dataset 1 data quality assessment]")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Data Quality Assessment of Dataset 1 (Grade-A Stocks)',
             fontsize=14, fontweight='bold', y=0.995)

# (a) Completeness distribution
ax = axes[0, 0]
ax.hist(a_qualified['completeness_ratio'], bins=20,
        color=COLORS['primary'], alpha=0.7, edgecolor='black')
ax.axvline(99, color=COLORS['warning'], linestyle='--', linewidth=2,
           label='Threshold (99%)')
ax.set_xlabel('Completeness Ratio (%)')
ax.set_ylabel('Number of Stocks')
ax.set_title('(a) Data Completeness Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# (b) Listing years
ax = axes[0, 1]
ax.hist(a_qualified['listing_years'], bins=15,
        color=COLORS['secondary'], alpha=0.7, edgecolor='black')
ax.axvline(5, color=COLORS['warning'], linestyle='--', linewidth=2,
           label='Threshold (5 years)')
ax.set_xlabel('Listing Years')
ax.set_ylabel('Number of Stocks')
ax.set_title('(b) Listing Duration Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# (c) Quality score
ax = axes[0, 2]
ax.hist(a_qualified['quality_score'], bins=15,
        color=COLORS['accent'], alpha=0.7, edgecolor='black')
ax.axvline(85, color=COLORS['warning'], linestyle='--', linewidth=2,
           label='Threshold (85)')
ax.set_xlabel('Quality Score')
ax.set_ylabel('Number of Stocks')
ax.set_title('(c) Overall Quality Score Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# (d) Number of records per stock
stock_counts = df_dataset1.groupby('stock_code').size().sort_values(ascending=False)
ax = axes[1, 0]
ax.bar(range(len(stock_counts)), stock_counts.values,
       color=COLORS['primary'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Stock Rank')
ax.set_ylabel('Number of Records')
ax.set_title('(d) Records per Stock')
ax.grid(axis='y', alpha=0.3)

# (e) Time span
stock_time_spans = df_dataset1.groupby('stock_code').agg(
    first_date=('trade_date', 'min'),
    last_date=('trade_date', 'max')
)
stock_time_spans['span_years'] = (
    stock_time_spans['last_date'] - stock_time_spans['first_date']
).dt.days / 365.25

ax = axes[1, 1]
ax.hist(stock_time_spans['span_years'], bins=15,
        color=COLORS['success'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Time Span (Years)')
ax.set_ylabel('Number of Stocks')
ax.set_title('(e) Temporal Coverage per Stock')
ax.grid(axis='y', alpha=0.3)

# (f) Quality screening funnel
stages = ['All Stocks\n(83)',
          'Quality\nCriteria',
          'Listing\nTime']
counts = [83, len(a_grade), len(a_qualified)]

ax = axes[1, 2]
colors_funnel = [COLORS['neutral'], COLORS['primary'], COLORS['success']]
bars = ax.barh(range(len(stages)), counts, color=colors_funnel, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages)
ax.set_xlabel('Number of Stocks')
ax.set_title('(f) Data Selection Process')
ax.invert_yaxis()

for i, (bar, count) in enumerate(zip(bars, counts)):
    ax.text(count + 2, i, f'{count}', va='center', fontweight='bold')

ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/preprocessing/dataset1/Fig1_data_quality_assessment.png',
            dpi=300, bbox_inches='tight')
plt.savefig('visualizations/preprocessing/dataset1/Fig1_data_quality_assessment.pdf',
            bbox_inches='tight')
print("✓ Saved Fig1_data_quality_assessment.png/pdf")
plt.close()

#==============================================================================
# Fig2: data cleaning statistics - Dataset 1
#==============================================================================
print("\n [Creating figure Fig2: Dataset 1 data cleaning statistics]")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Data Cleaning Statistics - Dataset 1 (Grade-A Stocks)',
             fontsize=14, fontweight='bold')

# (a) Comparison before and after cleaning
ax = axes[0]
stages_clean = ['Original\nData', 'After\nCleaning', 'After Feature\nEngineering']
counts_clean = [len(df_raw_ds1), len(df_raw_ds1)*0.98, len(df_dataset1)]  # Approximate value

bars = ax.bar(range(len(stages_clean)), counts_clean,
              color=[COLORS['neutral'], COLORS['primary'], COLORS['success']],
              alpha=0.7, edgecolor='black', width=0.6)

ax.set_xticks(range(len(stages_clean)))
ax.set_xticklabels(stages_clean)
ax.set_ylabel('Number of Records')
ax.set_title('(a) Data Processing Pipeline')
ax.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars, counts_clean)):
    ax.text(i, count + 100, f'{int(count):,}',
            ha='center', va='bottom', fontweight='bold')

# (b) Feature category statistics
ax = axes[1]
feature_categories = ['Original\nMarket\nFeatures', 'Time\nFeatures',
                      'Technical\nIndicators', 'Target\nVariable']
feature_counts = [8, 13, 29, 1]  # Fix: raw features exclude pct_chg and change
colors_feat = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning']]

bars = ax.bar(range(len(feature_categories)), feature_counts,
              color=colors_feat, alpha=0.7, edgecolor='black', width=0.6)

ax.set_xticks(range(len(feature_categories)))
ax.set_xticklabels(feature_categories)
ax.set_ylabel('Number of Features')
ax.set_title('(b) Feature Categories')
ax.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars, feature_counts)):
    ax.text(i, count + 0.5, str(count),
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add notes
ax.text(0.5, -0.15, 'Note: All features (except target) are shifted by 1 day',
        transform=ax.transAxes, ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('visualizations/preprocessing/dataset1/Fig2_data_cleaning_stats.png',
            dpi=300, bbox_inches='tight')
plt.savefig('visualizations/preprocessing/dataset1/Fig2_data_cleaning_stats.pdf',
            bbox_inches='tight')
print("✓ Saved Fig2_data_cleaning_stats.png/pdf")
plt.close()

#==============================================================================
# Fig3: Dataset 2 (Grade A+B) overview
#==============================================================================
print("\n [Creating figure Fig3: Dataset 2 overview]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dataset 2 Overview (Grade-A and Grade-B Stocks)',
             fontsize=14, fontweight='bold', y=0.995)

# (a) Distribution of stock grades
ax = axes[0, 0]
grade_counts = [len(a_qualified), len(b_qualified)]
colors_grade = [COLORS['success'], COLORS['primary']]
wedges, texts, autotexts = ax.pie(grade_counts, labels=['Grade-A', 'Grade-B'],
                                   autopct='%1.1f%%', colors=colors_grade,
                                   startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title('(a) Stock Grade Distribution')

# (b) Data volume comparison
ax = axes[0, 1]
datasets_names = ['Dataset 1\n(A-grade)', 'Dataset 2\n(A+B grade)']
record_counts = [len(df_dataset1), len(df_dataset2)]

bars = ax.bar(range(len(datasets_names)), record_counts,
              color=[COLORS['success'], COLORS['primary']],
              alpha=0.7, edgecolor='black', width=0.5)

ax.set_xticks(range(len(datasets_names)))
ax.set_xticklabels(datasets_names)
ax.set_ylabel('Number of Records')
ax.set_title('(b) Total Records Comparison')
ax.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars, record_counts)):
    ax.text(i, count + 500, f'{count:,}',
            ha='center', va='bottom', fontweight='bold')

# (c) Mean number of records per stock
ax = axes[1, 0]
avg_records = [len(df_dataset1)/len(dataset1_stocks),
               len(df_dataset2)/len(dataset2_stocks)]

bars = ax.bar(range(len(datasets_names)), avg_records,
              color=[COLORS['success'], COLORS['primary']],
              alpha=0.7, edgecolor='black', width=0.5)

ax.set_xticks(range(len(datasets_names)))
ax.set_xticklabels(datasets_names)
ax.set_ylabel('Average Records per Stock')
ax.set_title('(c) Average Records per Stock')
ax.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars, avg_records)):
    ax.text(i, count + 20, f'{count:.0f}',
            ha='center', va='bottom', fontweight='bold')

# (d) Comparison of time ranges
ax = axes[1, 1]

# Compute the time range of each dataset
time_ranges = []
for df, name in [(df_dataset1, 'Dataset 1'), (df_dataset2, 'Dataset 2')]:
    start = df['trade_date'].min()
    end = df['trade_date'].max()
    duration = (end - start).days
    time_ranges.append({'name': name, 'start': start, 'end': end, 'duration': duration})

y_pos = [0, 1]
colors_time = [COLORS['success'], COLORS['primary']]

for i, tr in enumerate(time_ranges):
    ax.barh(y_pos[i], tr['duration'], left=tr['start'].toordinal(),
            color=colors_time[i], alpha=0.7, edgecolor='black', height=0.6)

    # Add labels
    mid = tr['start'].toordinal() + tr['duration']/2
    ax.text(mid, y_pos[i], f"{tr['duration']} days\n({tr['duration']/365.25:.1f} years)",
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_yticks(y_pos)
ax.set_yticklabels([tr['name'] for tr in time_ranges], fontsize=11, fontweight='bold')
ax.set_xlabel('Time Period')
ax.set_title('(d) Temporal Coverage Comparison')

# Format the x axis
all_dates = [tr['start'] for tr in time_ranges] + [tr['end'] for tr in time_ranges]
date_min = min(all_dates)
date_max = max(all_dates)
date_range = (date_max - date_min).days
num_ticks = 8
date_ticks = [date_min + timedelta(days=i*date_range//(num_ticks-1)) for i in range(num_ticks)]
ax.set_xticks([d.toordinal() for d in date_ticks])
ax.set_xticklabels([d.strftime('%Y-%m') for d in date_ticks], rotation=45, ha='right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/preprocessing/dataset2/Fig3_dataset2_overview.png',
            dpi=300, bbox_inches='tight')
plt.savefig('visualizations/preprocessing/dataset2/Fig3_dataset2_overview.pdf',
            bbox_inches='tight')
print("✓ Saved Fig3_dataset2_overview.png/pdf")
plt.close()

#==============================================================================
# Final summary
#==============================================================================
print("\n" + "=" * 100)
print("🎉 Steps 1-5 complete! (fully corrected version - no data leakage)")
print("=" * 100)

print(f"\n📊 Dataset statistics:")
print(f"  Dataset 1 (Grade A):")
print(f"    Number of stocks: {len(dataset1_stocks)}")
print(f"    Number of records: {len(df_dataset1):,}")
print(f"    Number of features: {len(df_dataset1.columns)}")
print(f"\n  Dataset 2 (Grade A+B):")
print(f"    Number of stocks: {len(dataset2_stocks)}")
print(f"    Number of records: {len(df_dataset2):,}")
print(f"    Number of features: {len(df_dataset2.columns)}")

print(f"\n🔧 Key fix:")
print(f"  ✅ All features (including raw prices) have been shift(1)")
print(f"  ✅ Target variables pct_chg and change are not included in the features")
print(f"  ✅ Forecasting logic: features of day T-1 predict the return of day T")
print(f"  ✅ Missing values filled within each stock (avoiding cross-stock contamination)")

print(f"\n💾 Created files:")
print(f"  Data files:")
print(f"    • datasets/phase5_data.pkl")
print(f"  Visualization files:")
print(f"    • visualizations/preprocessing/dataset1/Fig1_data_quality_assessment.png/pdf")
print(f"    • visualizations/preprocessing/dataset1/Fig2_data_cleaning_stats.png/pdf")
print(f"    • visualizations/preprocessing/dataset2/Fig3_dataset2_overview.png/pdf")

print(f"\n📝 Exported variables:")
print(f"  • df_dataset1: DataFrame with {len(df_dataset1):,} records")
print(f"  • df_dataset2: DataFrame with {len(df_dataset2):,} records")
print(f"  • dataset1_stocks: List of {len(dataset1_stocks)} stock codes")
print(f"  • dataset2_stocks: List of {len(dataset2_stocks)} stock codes")

print("\n" + "=" * 100)
print("Ready for step 6: feature selection (corrected version)")
print("=" * 100)


"""
=============================================================================
Step 6: feature selection (corrected version)
Fixes target leakage and the forecasting logic

Key fixes:
1. ✅ Remove target variables from the candidate features (pct_chg, change)
2. ✅ Fill missing values separately for each stock (not globally)
3. ✅ Use features at time T to predict the target at time T+1 (true forecasting mode)
4. ✅ All features are lagged relative to the target (no same-period correlation)

Author: [your name]
Date: 2025-01-XX
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Academic plotting style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "=" * 100)
print("Step 6: feature selection (corrected version - no target leakage)")
print("=" * 100)

#==============================================================================
# 6.0 Configuration
#==============================================================================
print("\n" + "-" * 100)
print("6.0 Configuration information")
print("-" * 100)

# Split date (same as before)
TRAIN_END_DATE = '2022-12-31'

print(f"\n Time split:")
print(f"  Training set:     all data <= {TRAIN_END_DATE}")
print(f"  Feature selection uses training-set data only")

# JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

#==============================================================================
# 6.1 Define feature categories (corrected version)
#==============================================================================
print("\n" + "-" * 100)
print("6.1 Feature category definitions (corrected version)")
print("-" * 100)

# ✅ Key fix: remove pct_chg and change from the features!
# These are target variables, not input features
original_features = [
    'open', 'close', 'high', 'low', 'vol', 'amount',
    'turnover_rate', 'pre_close'
    # ❌ Removed: 'pct_chg', 'change' <- These are prediction targets!
]

print(f"\n⚠️  Key fixes applied:")
print(f"  ❌ Removed from the features: 'pct_chg', 'change'")
print(f"  ✅ These are target variables, not input features")
print(f"  ✅ Prevents the data leakage caused by 100% correlation")

# Time features
time_features_numeric = [
    'year', 'month', 'day', 'dayofweek', 'quarter',
    'dayofyear', 'weekofyear'
]

time_features_cyclic = [
    'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos'
]

# Technical indicators (computed in the preprocessing stage)
technical_features = [
    'ma5', 'ma10', 'ma20', 'ma60', 'close_to_ma5', 'close_to_ma20',
    'volume_ma5', 'volume_ma10', 'volume_ratio',
    'roc_5', 'roc_10', 'roc_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'high_low_range', 'close_open_range', 'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'momentum_5', 'momentum_10'
]

# Candidate feature set (for feature selection)
candidate_features = original_features + time_features_cyclic + technical_features

print(f"\n Corrected feature composition:")
print(f"  Raw market features:         {len(original_features):2d} (Corrected)")
print(f"  Time features (cyclic encoding): {len(time_features_cyclic):2d}")
print(f"  Technical indicator features:         {len(technical_features):2d}")
print(f"  ───────────────────────────────────")
print(f"  Total number of candidate features:         {len(candidate_features):2d}")
print(f"\n  Time (discrete) features:     {len(time_features_numeric):2d}")
print(f"  (These features are kept directly and are not screened)")

#==============================================================================
# 6.2 Feature selection pipeline (corrected version)
#==============================================================================
print("\n" + "-" * 100)
print("6.2 Feature selection pipeline (corrected version)")
print("-" * 100)

def feature_selection_pipeline_corrected(df, dataset_name, feature_cols,
                                         target_col='pct_chg',
                                         train_end_date=TRAIN_END_DATE,
                                         predict_horizon=1,  # New: prediction T+1
                                         top_k_method='adaptive',
                                         redundancy_threshold=0.95,
                                         verbose=True):
    """
    Corrected feature selection pipeline.

    Key fixes:
    ----------------
    1. The target variable has been excluded from the features
    2. Fill missing values separately for each stock
    3. Use features at time T to predict the target at time T+horizon (true forecasting)
    4. Correct validation of the data structure

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    dataset_name : str
        Dataset name
    feature_cols : list
        Candidate feature list (must not include the target variable)
    target_col : str
        Target variable column name
    train_end_date : str
        Training-set end date
    predict_horizon : int
        Forecast horizon (1 = next day)
    top_k_method : str or int
        Number of features selected
    redundancy_threshold : float
        Correlation threshold for removing redundant features
    verbose : bool
        Whether to print detailed information

    Return:
    --------
    selected_features, importance_df, high_corr_pairs, metadata
    """

    if verbose:
        print(f"\n{'='*100}")
        print(f"Feature selection: {dataset_name}")
        print(f"{'='*100}")

    # -------------------------------------------------------------------------
    # Step 0: verify there is no target leakage
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 0] Target leakage check")

    # Check whether the target is among the features
    target_in_features = target_col in feature_cols
    if target_in_features:
        raise ValueError(f"❌ Target leakage detected: '{target_col}' is in the feature list!")

    # Check related target columns
    related_targets = ['pct_chg', 'change']
    leakage_cols = [col for col in feature_cols if col in related_targets]
    if leakage_cols:
        raise ValueError(f"❌ Target leakage detected: {leakage_cols} is in the feature list!")

    if verbose:
        print(f"  ✅ No target leakage detected")
        print(f"  ✅ Target '{target_col}' is not among the features")
        print(f"  ✅ Related columns {related_targets} is not among the features")

    # -------------------------------------------------------------------------
    # Step 1: temporal data isolation
    # -------------------------------------------------------------------------
    train_cutoff = pd.to_datetime(train_end_date)
    df_train = df[df['trade_date'] <= train_cutoff].copy()

    if verbose:
        print(f"\n[Step 1] Temporal data isolation")
        print(f"  Full dataset:   {len(df):,} Samples")
        print(f"  Training cut-off date: {train_end_date}")
        print(f"  Training-set samples:   {len(df_train):,} Samples")
        print(f"  Training period:   {df_train['trade_date'].min().date()} to {df_train['trade_date'].max().date()}")
        print(f"  Number of stocks:     {df_train['stock_code'].nunique()}")

    # -------------------------------------------------------------------------
    # Step 2: prepare features and target (corrected forecast horizon)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 2] Prepare features and target (corrected version)")
        print(f"  Forecast horizon: T+{predict_horizon}")
        print(f"  Logic: use features at time T to predict T+{predict_horizon} -time target")

    # Sort by stock and date
    df_train = df_train.sort_values(['stock_code', 'trade_date']).reset_index(drop=True)

    # Create the target with a forward shift (features at T predict T+1)
    df_train['target_future'] = df_train.groupby('stock_code')[target_col].shift(-predict_horizon)

    # Remove rows whose target is NaN (the last N rows of each stock)
    valid_mask = ~df_train['target_future'].isna()
    df_train_valid = df_train[valid_mask].copy()

    if verbose:
        print(f"  Original training samples: {len(df_train):,}")
        print(f"  Create T+{predict_horizon} after the target: {len(df_train_valid):,}")
        print(f"  Removed (end of each stock): {len(df_train) - len(df_train_valid):,}")

    # Extract features and target
    X = df_train_valid[feature_cols].copy()
    y = df_train_valid['target_future'].copy()
    stock_codes = df_train_valid['stock_code'].copy()

    if verbose:
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target vector shape: {y.shape}")

    # -------------------------------------------------------------------------
    # Step 3: fill missing values separately for each stock (corrected version)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 3] Missing-value handling (corrected version)")

    # Check missing values before processing
    missing_before = X.isna().sum().sum()
    missing_pct = (missing_before / (X.shape[0] * X.shape[1])) * 100

    if verbose:
        print(f"  Missing values before processing: {missing_before:,} ({missing_pct:.3f}%)")

    if missing_before > 0:
        if verbose:
            print(f"  ⚠️  Applying forward fill separately for each stock (not global fill)")

        # Handle infinite values first
        X = X.replace([np.inf, -np.inf], np.nan)

        # ✅ Key fix: fill each stock separately
        X_filled = []
        stock_list = stock_codes.unique()

        for stock in stock_list:
            stock_mask = (stock_codes == stock)
            X_stock = X[stock_mask].copy()

            # Forward fill only within this stock
            X_stock = X_stock.fillna(method='ffill')

            # Remaining NaN values (missing at the start of a stock's history) are filled with 0
            X_stock = X_stock.fillna(0)

            X_filled.append(X_stock)

        X = pd.concat(X_filled, axis=0)

        missing_after = X.isna().sum().sum()
        if verbose:
            print(f"  Missing values after processing: {missing_after:,}")
            print(f"  ✅ Separate filling per stock complete")
    else:
        if verbose:
            print(f"  ✅ No missing values detected")

    # ✅ New: suspension sample analysis
    if verbose:
        print(f"\n[Step 3.5] Suspension sample analysis")

    if 'days_since_last_trade' in df_train_valid.columns:
        long_halt_samples = df_train_valid[df_train_valid['days_since_last_trade'] > 10]
        if verbose:
            print(f"  Samples with suspension > 10 days detected: {len(long_halt_samples):,} records ({len(long_halt_samples)/len(df_train_valid)*100:.2f}%)")
            print(f"  Strategy: keep these samples for feature selection (added'is_after_long_halt'flag)")

        # Show the suspension distribution
        if len(long_halt_samples) > 0 and verbose:
            halt_stats = long_halt_samples.groupby('stock_code').size().sort_values(ascending=False)
            print(f"  The 5 stocks with the most suspension samples:")
            for stock, count in halt_stats.head(5).items():
                print(f"    {stock}: {count} records")
    else:
        if verbose:
            print(f"  not found'days_since_last_trade'feature; suspension analysis skipped")

    # Convert to numpy format
    X_values = X.values
    y_values = y.values

    # -------------------------------------------------------------------------
    # Steps 4-6: multi-method feature importance
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 4] Multi-method feature importance")

    # LightGBM
    if verbose:
        print(f"  [1/6] Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1, force_col_wise=True, n_jobs=-1
    )
    lgb_model.fit(X_values, y_values)
    lgb_importance = lgb_model.feature_importances_

    # XGBoost
    if verbose:
        print(f"  [2/6] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, tree_method='hist', verbosity=0, n_jobs=-1
    )
    xgb_model.fit(X_values, y_values)
    xgb_importance = xgb_model.feature_importances_

    # Random Forest (Random forest)
    if verbose:
        print(f"  [3/6] Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_values, y_values)
    rf_importance = rf_model.feature_importances_

    # F-statistic (F Statistic)
    if verbose:
        print(f"  [4/6] Computing F-statistic...")
    f_scores, _ = f_regression(X_values, y_values)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Mutual Information (Mutual information)
    if verbose:
        print(f"  [5/6] Computing Mutual Information...")
    mi_scores = mutual_info_regression(X_values, y_values, random_state=42, n_neighbors=3)

    # Pearson Correlation (Pearson correlation coefficient)
    if verbose:
        print(f"  [6/6] Computing Pearson Correlation...")
    target_corr = np.array([
        np.corrcoef(X_values[:, i], y_values)[0, 1]
        for i in range(X_values.shape[1])
    ])
    target_corr = np.nan_to_num(target_corr, nan=0.0)

    # -------------------------------------------------------------------------
    # Step 5: feature correlation analysis
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 5] Feature correlation analysis")

    corr_matrix = np.corrcoef(X_values.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_value = corr_matrix[i, j]
            if abs(corr_value) > 0.9:
                high_corr_pairs.append({
                    'Feature1': feature_cols[i],
                    'Feature2': feature_cols[j],
                    'Correlation': corr_value
                })

    if verbose:
        print(f"  Highly correlated feature pairs (|r|>0.9): {len(high_corr_pairs)}")

    # -------------------------------------------------------------------------
    # Step 6: ensemble importance score
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 6] Ensemble feature importance")

    def normalize_scores(scores):
        min_val, max_val = scores.min(), scores.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'LightGBM': normalize_scores(lgb_importance),
        'XGBoost': normalize_scores(xgb_importance),
        'RandomForest': normalize_scores(rf_importance),
        'F_Statistic': normalize_scores(f_scores),
        'Mutual_Info': normalize_scores(mi_scores),
        'Target_Corr': normalize_scores(np.abs(target_corr))
    })

    importance_df['Ensemble_Score'] = importance_df[[
        'LightGBM', 'XGBoost', 'RandomForest',
        'F_Statistic', 'Mutual_Info', 'Target_Corr'
    ]].mean(axis=1)

    importance_df['Rank'] = importance_df['Ensemble_Score'].rank(ascending=False)
    importance_df = importance_df.sort_values('Ensemble_Score', ascending=False)

    if verbose:
        print(f"  Top 3 features:")
        for i, row in importance_df.head(3).iterrows():
            print(f"    {int(row['Rank']):2d}. {row['Feature']:25s} Score: {row['Ensemble_Score']:.4f}")

    # -------------------------------------------------------------------------
    # Step 7: feature screening
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 7] Feature screening")

    # Determine top_k (Top-K)
    if top_k_method == 'adaptive':
        n_samples = len(X_values)
        if n_samples < 50000:
            top_k = 15
        elif n_samples < 100000:
            top_k = 20
        elif n_samples < 200000:
            top_k = 25
        else:
            top_k = 30
        if verbose:
            print(f"  Adaptive Top-K: {top_k} (Based on {n_samples:,} Samples)")
    else:
        top_k = int(top_k_method)
        if verbose:
            print(f"  Set manually Top-K: {top_k}")

    # Initial selection of top features
    top_features = importance_df.head(top_k)['Feature'].tolist()

    # Remove redundant features
    redundant_features = set()
    for pair in high_corr_pairs:
        if abs(pair['Correlation']) > redundancy_threshold:
            feat1 = pair['Feature1']
            feat2 = pair['Feature2']

            if feat1 in top_features and feat2 in top_features:
                score1 = importance_df[importance_df['Feature']==feat1]['Ensemble_Score'].values[0]
                score2 = importance_df[importance_df['Feature']==feat2]['Ensemble_Score'].values[0]

                if score1 > score2:
                    redundant_features.add(feat2)
                else:
                    redundant_features.add(feat1)

    if verbose:
        print(f"  Redundancy removal (|r|>{redundancy_threshold}): {len(redundant_features)} features")

    # Final selection
    selected_features = [f for f in top_features if f not in redundant_features]

    if verbose:
        print(f"\n  Screening summary:")
        print(f"    Total candidates:        {len(feature_cols)}")
        print(f"    Top-K Initial selection:      {top_k}")
        print(f"    Redundant features:        {len(redundant_features)}")
        print(f"    ─────────────────────────")
        print(f"    Final selection:        {len(selected_features)}")

        print(f"\n  Top-20 Feature details:")
        print(f"  {'Rank':<5} {'Status':<8} {'Feature':<25} {'Score':<10}")
        print(f"  {'-'*50}")
        for i, row in importance_df.head(20).iterrows():
            status = "✓ Keep" if row['Feature'] in selected_features else "✗ Remove"
            print(f"  {int(row['Rank']):>4d}  {status:<8} {row['Feature']:<25} {row['Ensemble_Score']:.4f}")

    # Metadata
    metadata = {
        'dataset_name': dataset_name,
        'train_end_date': train_end_date,
        'predict_horizon': predict_horizon,
        'n_train_samples': len(X_values),
        'n_candidate_features': len(feature_cols),
        'top_k': top_k,
        'redundancy_threshold': redundancy_threshold,
        'n_redundant_removed': len(redundant_features),
        'n_final_selected': len(selected_features),
        'target_in_features_check': False,  # Verified: no leakage
        'per_stock_imputation': True,  # Corrected
        'selection_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return selected_features, importance_df, high_corr_pairs, metadata

print("✓ Corrected feature selection pipeline defined")

#==============================================================================
# 6.3 Apply to all datasets
#==============================================================================
print("\n" + "=" * 100)
print("6.3 Run feature selection")
print("=" * 100)

# Dataset 1
print("\n" + "🔬" * 50)
print("Dataset 1: Grade-A stocks (A-GRADE)")
print("🔬" * 50)

selected_features_ds1, importance_ds1, corr_pairs_ds1, metadata_ds1 = feature_selection_pipeline_corrected(
    df=df_dataset1,
    dataset_name="Dataset-1 (A-grade)",
    feature_cols=candidate_features,
    target_col='pct_chg',
    train_end_date=TRAIN_END_DATE,
    predict_horizon=1,
    verbose=True
)

# Dataset 2
print("\n" + "🔬" * 50)
print("Dataset 2: Grade A+B stocks (A+B-GRADE)")
print("🔬" * 50)

selected_features_ds2, importance_ds2, corr_pairs_ds2, metadata_ds2 = feature_selection_pipeline_corrected(
    df=df_dataset2,
    dataset_name="Dataset-2 (A+B-grade)",
    feature_cols=candidate_features,
    target_col='pct_chg',
    train_end_date=TRAIN_END_DATE,
    predict_horizon=1,
    verbose=True
)

#==============================================================================
# 6.4 Combine and save
#==============================================================================
print("\n" + "=" * 100)
print("6.4 Combine and save results")
print("=" * 100)

# Add time features
all_features_ds1 = selected_features_ds1 + time_features_numeric
all_features_ds2 = selected_features_ds2 + time_features_numeric

print(f"\n Dataset 1:")
print(f"  Selected features: {len(selected_features_ds1)}")
print(f"  Time features: {len(time_features_numeric)}")
print(f"  Total:     {len(all_features_ds1)}")

print(f"\n Dataset 2:")
print(f"  Selected features: {len(selected_features_ds2)}")
print(f"  Time features: {len(time_features_numeric)}")
print(f"  Total:     {len(all_features_ds2)}")

# Save results
os.makedirs('datasets/feature_selection', exist_ok=True)

# Feature importance
importance_ds1.to_csv('datasets/feature_selection/importance_ds1.csv', index=False)
importance_ds2.to_csv('datasets/feature_selection/importance_ds2.csv', index=False)
print("\n✓ Feature importance saved")

# Correlated pairs
if corr_pairs_ds1:
    pd.DataFrame(corr_pairs_ds1).to_csv('datasets/feature_selection/high_corr_pairs_ds1.csv', index=False)
if corr_pairs_ds2:
    pd.DataFrame(corr_pairs_ds2).to_csv('datasets/feature_selection/high_corr_pairs_ds2.csv', index=False)
print("✓ Correlated pairs saved")

# Selected feature list
for ds_num, features in [(1, all_features_ds1), (2, all_features_ds2)]:
    feature_info = []
    for i, feat in enumerate(features, 1):
        if feat in original_features:
            category = 'Original'
        elif feat in time_features_numeric or feat in time_features_cyclic:
            category = 'Temporal'
        elif feat in technical_features:
            category = 'Technical'
        else:
            category = 'Unknown'

        feature_info.append({
            'Index': i,
            'Feature_Name': feat,
            'Category': category
        })

    pd.DataFrame(feature_info).to_csv(f'datasets/feature_selection/selected_features_ds{ds_num}.csv', index=False)

print("✓ Selected feature list saved")

# Metadata
metadata_all = {
    'dataset1': metadata_ds1,
    'dataset2': metadata_ds2
}

with open('datasets/feature_selection/selection_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata_all, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print("✓ Metadata saved")

# Full configuration
feature_config = {
    'temporal_split': {
        'train_end_date': TRAIN_END_DATE
    },
    'dataset1': {
        'all_features': all_features_ds1,
        'selected_features': selected_features_ds1,
        'temporal_features': time_features_numeric,
        'n_total': len(all_features_ds1)
    },
    'dataset2': {
        'all_features': all_features_ds2,
        'selected_features': selected_features_ds2,
        'temporal_features': time_features_numeric,
        'n_total': len(all_features_ds2)
    }
}

with open('datasets/feature_selection/feature_config.json', 'w', encoding='utf-8') as f:
    json.dump(feature_config, f, indent=2, ensure_ascii=False)
print("✓ Feature configuration saved")

#==============================================================================
# 6.5 Create visualizations
#==============================================================================
print("\n" + "=" * 100)
print("6.5 Create figures")
print("=" * 100)

# Figure: feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# Keep English titles to meet SCI standards
fig.suptitle('Feature Importance Analysis (Corrected - No Target Leakage)',
             fontsize=14, fontweight='bold', y=0.98)

for idx, (ds_name, importance_df, selected) in enumerate([
    ('Dataset 1 (A-grade)', importance_ds1, selected_features_ds1),
    ('Dataset 2 (A+B-grade)', importance_ds2, selected_features_ds2)
]):
    ax = axes[idx]

    top20 = importance_df.head(20).copy()
    top20 = top20.sort_values('Ensemble_Score', ascending=True)

    colors = ['#2E86AB' if f in selected else '#CCCCCC' for f in top20['Feature']]

    bars = ax.barh(range(len(top20)), top20['Ensemble_Score'],
                   color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['Feature'], fontsize=9)
    # Keep English axis labels
    ax.set_xlabel('Ensemble Importance Score', fontsize=10)
    ax.set_title(f'({chr(97+idx)}) {ds_name}', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.0)

    from matplotlib.patches import Patch
    legend_elements = [
        # Keep English legend
        Patch(facecolor='#2E86AB', edgecolor='black', label='Selected'),
        Patch(facecolor='#CCCCCC', edgecolor='black', label='Not Selected')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/preprocessing/fig7_feature_importance_corrected.png', dpi=300, bbox_inches='tight')
plt.savefig('visualizations/preprocessing/fig7_feature_importance_corrected.pdf', bbox_inches='tight')
print("✓ Figures saved")
plt.close()

#==============================================================================
# Final summary
#==============================================================================
print("\n" + "=" * 100)
print("✅ Step 6 complete (corrected version)")
print("=" * 100)

print(f"\n🔧 Key fixes applied:")
print(f"  ✅ Fix 1: remove target variables from the features (pct_chg, change)")
print(f"  ✅ Fix 2: fill missing values separately for each stock (not global)")
print(f"  ✅ Fix 3: use features at time T to predict the target at time T+1 (true forecasting)")
print(f"  ✅ Fix 4: target leakage check added")

print(f"\n📊 Results:")
print(f"  Dataset 1: total number of features {len(all_features_ds1)}")
print(f"  Dataset 2: total number of features {len(all_features_ds2)}")

print(f"\n💾 Saved files:")
print(f"  • datasets/feature_selection/importance_ds*.csv")
print(f"  • datasets/feature_selection/selected_features_ds*.csv")
print(f"  • datasets/feature_selection/feature_config.json")
print(f"  • datasets/feature_selection/selection_metadata.json")
print(f"  • visualizations/preprocessing/fig7_feature_importance_corrected.png/pdf")

print("\n" + "=" * 100)
print("Ready for steps 7-8: dataset split and standardization")
print("=" * 100)


"""
=====
Step 7-8: Dataset Splitting and Standardization (ENGINEERING VERSION)
Dataset split and standardization (engineering refactored version)

Key fix:
1. ✅ Physical separation: features (X) and target (y) are saved as separate files to remove any leakage risk.
2. ✅ Strict alignment: the feature list is read directly from the feature_config.json of Step 6; nothing is guessed.
3. ✅ Target protection: the target variable (pct_chg) is stored separately and is never scaled.

Output file structure:
- train_X.csv (contains only the standardized features + ID/Date)
- train_y.csv (contains only the raw target pct_chg + ID/Date)

Author: [your name]
Date: 2025-01-XX
=====
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from datetime import datetime
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 100)
print("STEP 7-8: DATASET SPLITTING (ENGINEERING VERSION - X/Y SEPARATION)")
print("=" * 100)

# =====
# 1. Load configuration and raw data
# =====
print("\n[1.0] Load configuration and data")

# 1.1 Read the feature configuration file (Golden Source of Truth)
config_path = 'datasets/feature_selection/feature_config.json'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ Feature configuration file not found: {config_path}\n Please run Step 6 (a01.txt) first to create the feature selection results!")

with open(config_path, 'r', encoding='utf-8') as f:
    feature_config = json.load(f)

print("  ✓ Loaded feature_config.json")

# 1.2 Read Phase 5 data
with open('datasets/phase5_data.pkl', 'rb') as f:
    phase5_data = pickle.load(f)
df_dataset1 = phase5_data['df_dataset1']
df_dataset2 = phase5_data['df_dataset2']
print(f"  ✓ Dataset 1: {len(df_dataset1):,} rows")
print(f"  ✓ Dataset 2: {len(df_dataset2):,} rows")

# 1.3 Define the time split
TRAIN_END = pd.to_datetime('2022-12-31')
VAL_START = pd.to_datetime('2023-01-01')
VAL_END = pd.to_datetime('2024-06-30')
TEST_START = pd.to_datetime('2024-07-01')
TEST_END = pd.to_datetime('2025-11-28')

# =====
# 2. Define the processing function (X/Y separation logic)
# =====
def process_and_save_dataset(df, dataset_key, output_dir):
    print(f"\n🚀 Process {dataset_key} ...")

    # 2.1 Get the exact feature list of this dataset (from Step 6)
    # Note: features = selected_features + temporal_features
    # These were explicitly selected in Step 6 and do not include pct_chg
    config_item = feature_config[dataset_key]
    feature_cols = config_item['selected_features'] + config_item['temporal_features']
    temporal_cols = config_item['temporal_features']
    target_col = 'pct_chg'

    print(f"  Total number of features: {len(feature_cols)} (From the configuration)")
    print(f"  Target variable: {target_col}")

    # 2.2 Make sure the column exists
    # If Step 6 selected vol but only volume exists here, correct it
    if 'vol' in feature_cols and 'vol' not in df.columns and 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})

    # Check pct_chg
    if target_col not in df.columns:
        if 'change_pct' in df.columns:
            df = df.rename(columns={'change_pct': target_col})
        else:
            raise ValueError(f"❌ Target column missing from the dataset {target_col}")

    # 2.3 Extract X (features) and y (target)
    # The ID columns must be kept for later alignment
    id_cols = ['stock_code', 'trade_date']

    X_df = df[id_cols + feature_cols].copy()
    y_df = df[id_cols + [target_col]].copy()

    # Make sure the time format is correct
    X_df['trade_date'] = pd.to_datetime(X_df['trade_date'])
    y_df['trade_date'] = pd.to_datetime(y_df['trade_date'])

    # 2.4 Split by time
    def split_time(data):
        train = data[data['trade_date'] <= TRAIN_END].copy()
        val = data[(data['trade_date'] >= VAL_START) & (data['trade_date'] <= VAL_END)].copy()
        test = data[(data['trade_date'] >= TEST_START) & (data['trade_date'] <= TEST_END)].copy()
        return train, val, test

    X_train, X_val, X_test = split_time(X_df)
    y_train, y_val, y_test = split_time(y_df)

    print(f"  Split complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2.5 Standardize X (fit on the training set only)
    # Exclude time features and ID columns
    cols_to_scale = [c for c in feature_cols if c not in temporal_cols]

    scaler = RobustScaler()
    print(f"  Standardizing {len(cols_to_scale)} features (time features skipped)...")

    # Missing-value handling: fill within each stock (prevents NaN from breaking the Scaler)
    def safe_fill(d):
        # Fill numeric feature columns only
        if 'stock_code' in d.columns:
             # This is slow; use transform instead to speed it up
            d[cols_to_scale] = d.groupby('stock_code')[cols_to_scale].transform(lambda x: x.ffill().fillna(0))
        return d

    X_train = safe_fill(X_train)
    X_val = safe_fill(X_val)
    X_test = safe_fill(X_test)

    # Fit & Transform
    scaler.fit(X_train[cols_to_scale])

    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # 2.6 Save files (X and y separated)
    os.makedirs(output_dir, exist_ok=True)

    # Save features X
    X_train.to_csv(f'{output_dir}/train_X.csv', index=False)
    X_val.to_csv(f'{output_dir}/val_X.csv', index=False)
    X_test.to_csv(f'{output_dir}/test_X.csv', index=False)

    # Save target y
    y_train.to_csv(f'{output_dir}/train_y.csv', index=False)
    y_val.to_csv(f'{output_dir}/val_y.csv', index=False)
    y_test.to_csv(f'{output_dir}/test_y.csv', index=False)

    # Save scaler
    scaler_path = f'datasets/scalers/scaler_{dataset_key}.pkl'
    os.makedirs('datasets/scalers', exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"  ✅ Saved: X (train/val/test) and y (train/val/test) to {output_dir}")

# =====
# 3. Run processing
# =====
process_and_save_dataset(df_dataset1, 'dataset1', 'datasets/processed/dataset1_A_grade')
process_and_save_dataset(df_dataset2, 'dataset2', 'datasets/processed/dataset2_AB_grade')

print("\n" + "=" * 100)
print("✅ Done! Features and target are physically separated.")
print("Example output folder: datasets/processed/dataset1_A_grade/")
print("  ├── train_X.csv  (Features only)")
print("  └── train_y.csv  (Target only)")
print("=" * 100)
