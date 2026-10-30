# -*- coding: utf-8 -*-
"""
Step 2: feature engineering (same candidate pool as the Chinese datasets).
"""

"""
Full U.S. feature engineering script
The feature pool is fully aligned with the A-shares (based on the A-share feature engineering code)

Input: stock_data_tushare/us_medical_all.csv
      stock_data_tushare/us_power_all.csv
Output: stock_data_tushare/us_medical_features_full.csv
      stock_data_tushare/us_power_features_full.csv

Install dependencies: pip install pandas numpy
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Path configuration
# ============================================================
INPUT_DIR  = "./stock_data_tushare"
OUTPUT_DIR = "./stock_data_tushare"

FILES = {
    "us_medical": "us_medical_all.csv",
    "us_power":   "us_power_all.csv",
}

# ============================================================
# U.S. float shares (used to compute turnover, unit: shares)
# Note: current approximate value; historical periods may differ because of buybacks/issuance
# ============================================================
US_SHARES_OUTSTANDING = {
    # Healthcare sector
    "JNJ":  4.05e9, "PFE":  5.65e9, "UNH":  0.92e9, "ABT":  1.74e9,
    "MRK":  2.53e9, "TMO":  0.38e9, "DHR":  0.74e9, "MDT":  1.28e9,
    "BMY":  2.03e9, "AMGN": 0.54e9, "GILD": 1.25e9, "LLY":  0.95e9,
    "ABBV": 1.77e9, "ISRG": 0.36e9, "SYK":  0.38e9, "BSX":  1.47e9,
    "CI":   0.28e9, "HUM":  0.12e9, "REGN": 0.11e9, "VRTX": 0.26e9,
    # Power sector
    "NEE":  2.06e9, "DUK":  0.77e9, "SO":   1.10e9, "AEP":  0.53e9,
    "EXC":  1.00e9, "SRE":  0.64e9, "XEL":  0.57e9, "ES":   0.36e9,
    "WEC":  0.32e9, "D":    0.84e9, "PEG":  0.50e9, "ED":   0.35e9,
    "EIX":  0.39e9, "FE":   0.58e9, "ETR":  0.21e9, "CMS":  0.30e9,
    "CNP":  0.64e9, "ATO":  0.15e9, "LNT":  0.26e9, "NI":   0.45e9,
}

# ============================================================
# Raw price features (aligned with A-shares; pre_close replaced by close.shift(1))
# ============================================================
ORIGINAL_FEATURE_COLS = [
    'open', 'close', 'high', 'low', 'vol', 'amount',
    'turnover_rate', 'pre_close',
]

# ============================================================
# Technical indicator features (identical to the A-shares)
# ============================================================
TECHNICAL_FEATURE_COLS = [
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
    'momentum_5', 'momentum_10',
]

# ============================================================
# Time features (identical to the A-shares)
# ============================================================
TIME_FEATURE_COLS = [
    'year', 'month', 'day', 'dayofweek', 'quarter',
    'dayofyear', 'weekofyear',
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
]

ALL_FEATURE_COLS = ORIGINAL_FEATURE_COLS + TECHNICAL_FEATURE_COLS

# ============================================================
# Feature construction for one stock (aligned with the A-share logic)
# ============================================================
def calculate_features_one_stock(group):
    """
    Compute all features for one U.S. stock, with logic fully aligned with the A-shares
    Key: all raw price features and technical indicators are shift(1)
    Target variable: next-day return (pct_chg_next)
    """
    group = group.sort_values('trade_date').reset_index(drop=True)

    # ---------- Data cleaning ----------
    group = group[
        (group['open']  > 0) & (group['high'] > 0) &
        (group['low']   > 0) & (group['close'] > 0)
    ]
    group = group[group['high'] >= group['low']]
    group = group[group['vol']  > 0]

    group['pct_chg_raw'] = group['close'].pct_change() * 100
    group = group[group['pct_chg_raw'].abs() < 100].copy()
    group = group.reset_index(drop=True)

    # ---------- Trading-day gap (aligned with the A-shares) ----------
    group['days_since_last_trade'] = group['trade_date'].diff().dt.days
    group.loc[0, 'days_since_last_trade'] = 1

    # ---------- Basic derived fields (aligned with the raw A-share fields) ----------
    group['amount']    = group['close'] * group['vol']
    ticker             = group['stock_code'].iloc[0]
    shares             = US_SHARES_OUTSTANDING.get(ticker, np.nan)
    group['turnover_rate'] = (
        group['vol'] / shares * 100 if (not isinstance(shares, float) or not np.isnan(shares))
        else np.nan
    )
    # pre_close: replaced by the previous day's close (aligned with the A-share fields)
    group['pre_close'] = group['close'].shift(1)

    # ---------- Moving average ----------
    group['ma5']  = group['close'].rolling(window=5,  min_periods=1).mean()
    group['ma10'] = group['close'].rolling(window=10, min_periods=1).mean()
    group['ma20'] = group['close'].rolling(window=20, min_periods=1).mean()
    group['ma60'] = group['close'].rolling(window=60, min_periods=1).mean()

    # ---------- Relative price position ----------
    group['close_to_ma5']  = (group['close'] - group['ma5'])  / (group['ma5']  + 1e-8)
    group['close_to_ma20'] = (group['close'] - group['ma20']) / (group['ma20'] + 1e-8)

    # ---------- Volume indicators ----------
    group['volume_ma5']   = group['vol'].rolling(window=5,  min_periods=1).mean()
    group['volume_ma10']  = group['vol'].rolling(window=10, min_periods=1).mean()
    group['volume_ratio'] = group['vol'] / (group['volume_ma5'] + 1e-8)

    # ---------- Rate of change ----------
    group['roc_5']  = group['close'].pct_change(periods=5)  * 100
    group['roc_10'] = group['close'].pct_change(periods=10) * 100
    group['roc_20'] = group['close'].pct_change(periods=20) * 100

    # ---------- Volatility ----------
    group['volatility_5']  = group['close'].rolling(window=5,  min_periods=1).std()
    group['volatility_10'] = group['close'].rolling(window=10, min_periods=1).std()
    group['volatility_20'] = group['close'].rolling(window=20, min_periods=1).std()

    # ---------- Price range ----------
    group['high_low_range']   = (group['high'] - group['low'])    / (group['close'] + 1e-8)
    group['close_open_range'] = (group['close'] - group['open'])  / (group['open']  + 1e-8)

    # ---------- RSI ----------
    delta = group['close'].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs    = gain / (loss + 1e-8)
    group['rsi'] = 100 - (100 / (1 + rs))

    # ---------- MACD ----------
    ema12              = group['close'].ewm(span=12, adjust=False).mean()
    ema26              = group['close'].ewm(span=26, adjust=False).mean()
    group['macd']      = ema12 - ema26
    group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
    group['macd_hist'] = group['macd'] - group['macd_signal']

    # ---------- Bollinger bands ----------
    bb_std               = group['close'].rolling(window=20, min_periods=1).std()
    group['bb_middle']   = group['close'].rolling(window=20, min_periods=1).mean()
    group['bb_upper']    = group['bb_middle'] + (bb_std * 2)
    group['bb_lower']    = group['bb_middle'] - (bb_std * 2)
    group['bb_width']    = (group['bb_upper'] - group['bb_lower']) / (group['bb_middle'] + 1e-8)
    group['bb_position'] = (
        (group['close'] - group['bb_lower']) /
        (group['bb_upper'] - group['bb_lower'] + 1e-8)
    )

    # ---------- ATR ----------
    high_low    = group['high'] - group['low']
    high_close  = (group['high'] - group['close'].shift(1)).abs()
    low_close   = (group['low']  - group['close'].shift(1)).abs()
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    group['atr'] = true_range.rolling(window=14, min_periods=1).mean()

    # ---------- Momentum ----------
    group['momentum_5']  = group['close'] - group['close'].shift(5)
    group['momentum_10'] = group['close'] - group['close'].shift(10)

    # ---------- Time features (no shift needed, taken directly from the date) ----------
    group['year']       = group['trade_date'].dt.year
    group['month']      = group['trade_date'].dt.month
    group['day']        = group['trade_date'].dt.day
    group['dayofweek']  = group['trade_date'].dt.dayofweek
    group['quarter']    = group['trade_date'].dt.quarter
    group['dayofyear']  = group['trade_date'].dt.dayofyear
    group['weekofyear'] = group['trade_date'].dt.isocalendar().week.astype(int)

    group['month_sin']      = np.sin(2 * np.pi * group['month']     / 12)
    group['month_cos']      = np.cos(2 * np.pi * group['month']     / 12)
    group['dayofweek_sin']  = np.sin(2 * np.pi * group['dayofweek'] / 7)
    group['dayofweek_cos']  = np.cos(2 * np.pi * group['dayofweek'] / 7)
    group['dayofyear_sin']  = np.sin(2 * np.pi * group['dayofyear'] / 365)
    group['dayofyear_cos']  = np.cos(2 * np.pi * group['dayofyear'] / 365)

    # ---------- Target variable: next-day return ----------
    group['pct_chg_next'] = group['pct_chg_raw'].shift(-1)

    # ---------- All price/technical features lagged by one period (prevents data leakage) ----------
    group[ALL_FEATURE_COLS] = group[ALL_FEATURE_COLS].shift(1)

    # ---------- Long suspension flag ----------
    group['is_after_long_halt'] = (group['days_since_last_trade'] > 10).astype(int)

    # ---------- Forward fill within each stock ----------
    group[ALL_FEATURE_COLS] = group[ALL_FEATURE_COLS].ffill()

    # ---------- Remove missing values ----------
    drop_cols = ALL_FEATURE_COLS + TIME_FEATURE_COLS + ['pct_chg_next']
    group = group.dropna(subset=drop_cols).reset_index(drop=True)

    # ---------- Clean up intermediate columns ----------
    group = group.drop(columns=['pct_chg_raw'], errors='ignore')

    # ---------- Keep the column order aligned with the A-shares ----------
    keep = (
        ['stock_code', 'trade_date'] +
        ORIGINAL_FEATURE_COLS +
        TECHNICAL_FEATURE_COLS +
        TIME_FEATURE_COLS +
        ['days_since_last_trade', 'is_after_long_halt', 'pct_chg_next']
    )
    keep = [c for c in keep if c in group.columns]
    return group[keep]

# ============================================================
# Main flow
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, fname in FILES.items():
        path = os.path.join(INPUT_DIR, fname)
        print(f"\n{'='*60}")
        print(f"Process: {label}  ({fname})")
        print(f"{'='*60}")

        df = pd.read_csv(path)

        # Unify column names to match the A-shares
        df = df.rename(columns={
            "date":   "trade_date",
            "ticker": "stock_code",
            "volume": "vol",
        })

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["stock_code"] = df["stock_code"].astype(str)

        # Process stock by stock
        out_list = []
        for ticker, g in df.groupby("stock_code"):
            g = g.copy()
            result = calculate_features_one_stock(g)
            out_list.append(result)
            print(f"  [{ticker}] {len(result)} rows")

        out = pd.concat(out_list, ignore_index=True)

        # Save
        save_path = os.path.join(OUTPUT_DIR, f"{label}_features_full.csv")
        out.to_csv(save_path, index=False, encoding="utf-8-sig")

        # Summary
        print(f"\n saved to: {save_path}")
        print(f"shape:        {out.shape}")
        print(f"Date:         {out['trade_date'].min()} ~ {out['trade_date'].max()}")
        print(f"Number of stocks:       {out['stock_code'].nunique()}")
        print(f"Raw price features: {len(ORIGINAL_FEATURE_COLS)}")
        print(f"Technical indicator features: {len(TECHNICAL_FEATURE_COLS)}")
        print(f"Time features:     {len(TIME_FEATURE_COLS)}")
        print(f"Total number of features:     {len(ORIGINAL_FEATURE_COLS)+len(TECHNICAL_FEATURE_COLS)+len(TIME_FEATURE_COLS)}")
        missing = out[ALL_FEATURE_COLS + TIME_FEATURE_COLS + ['pct_chg_next']].isna().sum().sum()
        print(f"Missing values:       {missing}")
        print(f"\n Full column names:")
        print(list(out.columns))
