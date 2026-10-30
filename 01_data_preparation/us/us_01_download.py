# -*- coding: utf-8 -*-
"""
Step 1: download U.S. daily data from Tushare (2010-01-01 to 2025-11-14).
Set your own Tushare token before running.
"""

"""
Stock data download script v3 - using Tushare (Hong Kong + U.S. stocks)
Keeps the same format as the existing A-share data

Install dependencies: pip install tushare pandas
"""

import tushare as ts
import pandas as pd
import os
import time

# ============================================================
# !!! Enter your token here !!!
# ============================================================
TUSHARE_TOKEN = "YOUR_TUSHARE_TOKEN"

# ============================================================
# Stock pool configuration
# ============================================================

# Hong Kong healthcare
HK_MEDICAL = {
    "01177.HK": "Sino Biopharmaceutical",
    "01093.HK": "CSPC Pharmaceutical Group",
    "02269.HK": "WuXi Biologics",
    "00853.HK": "MicroPort Scientific",
    "00867.HK": "China Medical System",
    "01066.HK": "Weigao Group",
    "00241.HK": "Alibaba Health",
    "02196.HK": "Fudan-Zhangjiang Bio-Pharmaceutical",
    "01548.HK": "GenScript Biotech",
    "06160.HK": "BeiGene",
}

# Hong Kong power
HK_POWER = {
    "02638.HK": "HK Electric",
    "00002.HK": "CLP Holdings",
    "00006.HK": "Power Assets Holdings",
    "00836.HK": "China Resources Power",
    "01816.HK": "CGN Power",
    "00135.HK": "Kunlun Energy",
    "03898.HK": "China Power",
    "00958.HK": "Huaneng Power International",
    "02380.HK": "China Power International Development",
    "00354.HK": "China Gas",
}

# U.S. healthcare
# US_MEDICAL = {
# "AAPL": "Apple", # Spare one; test whether the interface works first
# "JNJ": "Johnson & Johnson",
# "PFE": "Pfizer",
# "UNH": "UnitedHealth",
# "ABT": "Abbott",
# "MRK": "Merck",
# "TMO": "Thermo Fisher",
# "DHR": "Danaher",
# "MDT": "Medtronic",
# "BMY": "Bristol Myers Squibb",
# }

# # U.S. power
# US_POWER = {
# "NEE": "NextEra Energy",
# "DUK": "Duke Energy",
# "SO": "Southern Company",
# "AEP": "American Electric Power",
#     "EXC":  "Exelon",
#     "SRE":  "Sempra",
# "XEL": "Xcel Energy",
#     "ES":   "Eversource",
# "WEC": "WEC Energy",
# "D": "Dominion Energy",
# }

# U.S. healthcare / Healthcare, 20 stocks
US_MEDICAL = {
    "JNJ":  "Johnson & Johnson",
    "PFE":  "Pfizer",
    "UNH":  "UnitedHealth",
    "ABT":  "Abbott",
    "MRK":  "Merck",
    "TMO":  "Thermo Fisher",
    "DHR":  "Danaher",
    "MDT":  "Medtronic",
    "BMY":  "Bristol Myers Squibb",
    "AMGN": "Amgen",
    "GILD": "Gilead Sciences",
    "LLY":  "Eli Lilly",
    "ABBV": "AbbVie",
    "ISRG": "Intuitive Surgical",
    "SYK":  "Stryker",
    "BSX":  "Boston Scientific",
    "CI":   "Cigna",
    "HUM":  "Humana",
    "REGN": "Regeneron",
    "VRTX": "Vertex Pharmaceuticals",
}

# U.S. utilities / Utilities, 20 stocks
US_POWER = {
    "NEE": "NextEra Energy",
    "DUK": "Duke Energy",
    "SO":  "Southern Company",
    "AEP": "American Electric Power",
    "EXC": "Exelon",
    "SRE": "Sempra",
    "XEL": "Xcel Energy",
    "ES":  "Eversource Energy",
    "WEC": "WEC Energy",
    "D":   "Dominion Energy",
    "PEG": "Public Service Enterprise Group",
    "ED":  "Consolidated Edison",
    "EIX": "Edison International",
    "FE":  "FirstEnergy",
    "ETR": "Entergy",
    "CMS": "CMS Energy",
    "CNP": "CenterPoint Energy",
    "ATO": "Atmos Energy",
    "LNT": "Alliant Energy",
    "NI":  "NiSource",
}

# ============================================================
# Time range and quality control
# ============================================================

START_DATE       = "20100101"
END_DATE         = "20251114"
MIN_TRADING_DAYS = 1500    # Hong Kong/U.S. stocks have slightly fewer trading days than A-shares; relaxed slightly
MAX_MISSING_RATE = 0.05
OUTPUT_DIR       = "./stock_data_tushare"

# tushare per-minute request limit; wait after each request
SLEEP_BETWEEN = 0.4   # Increase for ordinary users; if a rate-limit error occurs, change to 1.0

# ============================================================
# Core function
# ============================================================

def fetch_hk_stock(pro, ts_code, name, start, end):
    """Download daily data of one Hong Kong stock"""
    try:
        df = pro.hk_daily(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="trade_date,open,high,low,close,vol"
        )
        if df is None or df.empty:
            print(f"  [Empty data] {ts_code} {name}")
            return None

        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]

        return _quality_check(df, ts_code, name)

    except Exception as e:
        print(f"  [Failed] {ts_code} {name}: {e}")
        return None


def fetch_us_stock(pro, ts_code, name, start, end):
    """Download daily data of one U.S. stock"""
    try:
        df = pro.us_daily(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="trade_date,open,high,low,close,vol"
        )
        if df is None or df.empty:
            print(f"  [Empty data] {ts_code} {name}")
            return None

        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]

        return _quality_check(df, ts_code, name)

    except Exception as e:
        print(f"  [Failed] {ts_code} {name}: {e}")
        return None


def _quality_check(df, ticker, name):
    """Common quality check"""
    total = len(df)
    if total < MIN_TRADING_DAYS:
        print(f"  [Insufficient data] {ticker} {name}: only {total} trading days, skipped")
        return None

    missing = df.isnull().mean().max()
    if missing > MAX_MISSING_RATE:
        print(f"  [Missing rate too high] {ticker} {name}: {missing:.1%}, skipped")
        return None

    df = df.ffill().dropna()
    df["ticker"] = ticker
    df["name"]   = name
    print(f"  [OK] {ticker} {name}: {total} trading days, missing rate {missing:.2%}")
    return df


def fetch_batch(pro, ticker_dict, label, start, end, output_dir, market="hk"):
    """Download and save in batch CSV"""
    sector_dir = os.path.join(output_dir, label)
    os.makedirs(sector_dir, exist_ok=True)

    all_dfs, success, fail = [], 0, 0

    for ticker, name in ticker_dict.items():
        if market == "hk":
            df = fetch_hk_stock(pro, ticker, name, start, end)
        else:
            df = fetch_us_stock(pro, ticker, name, start, end)

        if df is not None:
            safe_name = ticker.replace(".", "_")
            save_path  = os.path.join(sector_dir, f"{safe_name}.csv")
            df.to_csv(save_path)
            all_dfs.append(df)
            success += 1
        else:
            fail += 1

        time.sleep(SLEEP_BETWEEN)

    if all_dfs:
        combined      = pd.concat(all_dfs)
        combined_path = os.path.join(output_dir, f"{label}_all.csv")
        combined.to_csv(combined_path)
        print(f"\n  => Combined file saved: {combined_path}")

    print(f"  => {label}: Success {success} stocks, failed/skipped {fail} stocks\n")
    return all_dfs


def print_summary(output_dir):
    print("\n" + "="*55)
    print("Download complete, file summary:")
    print("="*55)
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith("_all.csv"):
            fpath = os.path.join(output_dir, fname)
            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                n  = df["ticker"].nunique() if "ticker" in df.columns else "?"
                print(f"  {fname:40s} | {len(df):7,} rows | {n} stocks")
                print(f"    Date: {df.index.min().date()} ~ {df.index.max().date()}")
            except Exception as e:
                print(f"  {fname}: Read failed {e}")


# ============================================================
# Main flow
# ============================================================

if __name__ == "__main__":
    # Initialization tushare
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    # Quickly check the token and permissions first
    print("Checking token...")
    try:
        info = pro.query('user', fields='nick_name,account_type,remaining_count')
        print(f"  Account: {info.iloc[0]['nick_name']}")
        print(f"  Type: {info.iloc[0]['account_type']}")
        print(f"  Remaining credits: {info.iloc[0]['remaining_count']}\n")
    except Exception as e:
        print(f"  Token Check failed: {e}")
        print("  Please check that the token is correct and that the network can reach tushare.pro")
        exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Start download, time range: {START_DATE} ~ {END_DATE}\n")

    print("[1/4] Hong Kong healthcare sector...")
    fetch_batch(pro, HK_MEDICAL, "hk_medical", START_DATE, END_DATE, OUTPUT_DIR, market="hk")

    print("[2/4] Hong Kong power sector...")
    fetch_batch(pro, HK_POWER, "hk_power", START_DATE, END_DATE, OUTPUT_DIR, market="hk")

    print("[3/4] U.S. healthcare sector...")
    fetch_batch(pro, US_MEDICAL, "us_medical", START_DATE, END_DATE, OUTPUT_DIR, market="us")

    print("[4/4] U.S. power sector...")
    fetch_batch(pro, US_POWER, "us_power", START_DATE, END_DATE, OUTPUT_DIR, market="us")

    print_summary(OUTPUT_DIR)
    print("All done!")
