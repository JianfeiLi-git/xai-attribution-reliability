# -*- coding: utf-8 -*-
"""
Step 4: data quality scoring and Grade-A / Grade-B classification.
"""

#Data quality step 3 of 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Screening analysis of stable high-quality stocks")
print("=" * 80)

# Read data
df = pd.read_csv('tomAB--.csv')
df['stock_code'] = df['ts_code'].astype(str).str.zfill(6)
# df['trade_date'] = pd.to_datetime(df['trade_date'])
# df['year'] = df['trade_date'].dt.year
df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
df['year'] = df['trade_date'].dt.year
df['month'] = df['trade_date'].dt.to_period('M')
# Get all trading days
all_trade_dates = sorted(df['trade_date'].unique())
total_trade_days = len(all_trade_dates)

print(f"\n Basic dataset information:")
print(f"  Total number of stocks: {df['stock_code'].nunique()} stocks")
print(f"  Total trading days: {total_trade_days} days")
print(f"  Time range: {all_trade_dates[0].strftime('%Y-%m-%d')} to {all_trade_dates[-1].strftime('%Y-%m-%d')}")

# ============================================================================
# Step 1: compute quality indicators for each stock
# ============================================================================
print("\n" + "=" * 80)
print("[Step 1: compute stock quality indicators]")
print("=" * 80)

stock_quality = []

for stock_code in df['stock_code'].unique():
    stock_data = df[df['stock_code'] == stock_code].sort_values('trade_date')

    # Basic statistics
    first_date = stock_data['trade_date'].min()
    last_date = stock_data['trade_date'].max()
    actual_days = len(stock_data)

    # Compute the expected number of trading days
    expected_dates = [d for d in all_trade_dates if first_date <= d <= last_date]
    expected_days = len(expected_dates)

    # Data completeness
    missing_days = expected_days - actual_days
    completeness_ratio = (actual_days / expected_days * 100) if expected_days > 0 else 0

    # Listing duration (years)
    listing_years = (last_date - first_date).days / 365.25

    # Data anomaly detection
    # 1. Number of negative prices
    negative_prices = ((stock_data['open'] < 0) | (stock_data['close'] < 0) |
                      (stock_data['high'] < 0) | (stock_data['low'] < 0)).sum()

    # 2. Days with zero volume
    zero_volume_days = (stock_data['vol'] == 0).sum()

    # 3. Extreme returns (possibly data errors)
    extreme_change = ((stock_data['pct_chg'] < -50) | (stock_data['pct_chg'] > 50)).sum()

    # 4. Price logic errors (high < low, high < close, etc.)
    price_logic_error = ((stock_data['high'] < stock_data['low']) |
                         (stock_data['high'] < stock_data['open']) |
                         (stock_data['high'] < stock_data['close']) |
                         (stock_data['low'] > stock_data['open']) |
                         (stock_data['low'] > stock_data['close'])).sum()

    # 5. Trading continuity (longest run of continuous trading)
    trade_dates = stock_data['trade_date'].tolist()
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(trade_dates)):
        expected_next = all_trade_dates[all_trade_dates.index(trade_dates[i-1]) + 1] if all_trade_dates.index(trade_dates[i-1]) < len(all_trade_dates) - 1 else None
        if expected_next and trade_dates[i] == expected_next:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    # 6. Trading activity (mean turnover rate)
    avg_turnover = stock_data['turnover_rate'].mean()

    # 7. Data stability (standard deviation of price changes)
    price_stability = stock_data['close'].std() / stock_data['close'].mean() if stock_data['close'].mean() > 0 else 999

    # Overall quality score (0-100 points)
    quality_score = 0

    # Completeness score (40 points)
    quality_score += min(completeness_ratio * 0.4, 40)

    # No abnormal data (30 points)
    anomaly_ratio = (negative_prices + extreme_change + price_logic_error) / actual_days
    quality_score += max(0, 30 - anomaly_ratio * 100)

    # Listing duration (15 points) - at least 3 years
    quality_score += min(listing_years / 20 * 15, 15)

    # Trading activity (15 points)
    if avg_turnover >= 1:
        quality_score += min(avg_turnover / 5 * 15, 15)

    stock_quality.append({
        'stock_code': stock_code,
        'first_date': first_date,
        'last_date': last_date,
        'listing_years': round(listing_years, 2),
        'actual_days': actual_days,
        'expected_days': expected_days,
        'missing_days': missing_days,
        'completeness_ratio': round(completeness_ratio, 2),
        'negative_prices': negative_prices,
        'zero_volume_days': zero_volume_days,
        'extreme_change': extreme_change,
        'price_logic_error': price_logic_error,
        'max_consecutive_days': max_consecutive,
        'avg_turnover': round(avg_turnover, 2),
        'price_stability': round(price_stability, 2),
        'quality_score': round(quality_score, 2)
    })

quality_df = pd.DataFrame(stock_quality)
quality_df = quality_df.sort_values('quality_score', ascending=False)

print(f"\n Quality score statistics:")
print(f"  Mean quality score: {quality_df['quality_score'].mean():.2f}")
print(f"  Highest quality score: {quality_df['quality_score'].max():.2f}")
print(f"  Lowest quality score: {quality_df['quality_score'].min():.2f}")

# ============================================================================
# Step 2: define criteria for stable high-quality stocks
# ============================================================================
print("\n" + "=" * 80)
print("[Step 2: define criteria for stable high-quality stocks]")
print("=" * 80)

print("\n We define three quality grades:")
print("\n [Grade A: top-quality stocks] strict criteria")
print("  1. Data completeness rate ≥ 99% (Almost no suspension)")
print("  2. Listing date ≥ 5 years (enough historical data)")
print("  3. No negative prices, no extreme outliers")
print("  4. Price logic correct")
print("  5. Quality score ≥ 85 points")

a_level = quality_df[
    (quality_df['completeness_ratio'] >= 99) &
    (quality_df['listing_years'] >= 5) &
    (quality_df['negative_prices'] == 0) &
    (quality_df['extreme_change'] == 0) &
    (quality_df['price_logic_error'] == 0) &
    (quality_df['quality_score'] >= 85)
]

print(f"\n Stocks meeting the Grade-A criteria: {len(a_level)} stocks")

print("\n [Grade B: high-quality stocks] medium criteria")
print("  1. Data completeness rate ≥ 95% (Slight suspension)")
print("  2. Listing date ≥ 3 years")
print("  3. Negative prices ≤ 5 records")
print("  4. Extreme outliers ≤ 10 records")
print("  5. Quality score ≥ 75 points")

b_level = quality_df[
    (quality_df['completeness_ratio'] >= 95) &
    (quality_df['listing_years'] >= 3) &
    (quality_df['negative_prices'] <= 5) &
    (quality_df['extreme_change'] <= 10) &
    (quality_df['quality_score'] >= 75) &
    (~quality_df['stock_code'].isin(a_level['stock_code']))
]

print(f"Stocks meeting the Grade-B criteria: {len(b_level)} stocks")

print("\n [Grade C: usable stocks] loose criteria")
print("  1. Data completeness rate ≥ 90%")
print("  2. Listing date ≥ 2 years")
print("  3. Quality score ≥ 60 points")

c_level = quality_df[
    (quality_df['completeness_ratio'] >= 90) &
    (quality_df['listing_years'] >= 2) &
    (quality_df['quality_score'] >= 60) &
    (~quality_df['stock_code'].isin(a_level['stock_code'])) &
    (~quality_df['stock_code'].isin(b_level['stock_code']))
]

print(f"Stocks meeting the Grade-C criteria: {len(c_level)} stocks")

print(f"\n [Total]")
print(f"  A grade (recommended for the study): {len(a_level)} stocks")
print(f"  B grade (alternative): {len(b_level)} stocks")
print(f"  C grade (optional): {len(c_level)} stocks")
print(f"  Total high-quality stocks: {len(a_level) + len(b_level) + len(c_level)} stocks")
print(f"  Not recommended: {len(quality_df) - len(a_level) - len(b_level) - len(c_level)} stocks")

# ============================================================================
# Step 3: show Grade-A high-quality stocks in detail
# ============================================================================
print("\n" + "=" * 80)
print("[Step 3: details of Grade-A high-quality stocks]")
print("=" * 80)

if len(a_level) > 0:
    print(f"\n total {len(a_level)} Grade-A stocks (sorted by quality score, descending):\n")

    print(f"{'Rank':<6} {'Stock code':<12} {'Listing years':<10} {'Number of trading days':<10} {'Completeness rate%':<10} "
          f"{'Number of anomalies':<10} {'Quality score':<10} {'First trade':<15} {'Last trade'}")
    print("-" * 120)

    for idx, row in enumerate(a_level.head(50).itertuples(), 1):
        anomaly_count = row.negative_prices + row.extreme_change + row.price_logic_error
        print(f"{idx:<6} {row.stock_code:<12} {row.listing_years:<10.1f} {row.actual_days:<10} "
              f"{row.completeness_ratio:<10.2f} {anomaly_count:<10} {row.quality_score:<10.2f} "
              f"{row.first_date.strftime('%Y-%m-%d'):<15} {row.last_date.strftime('%Y-%m-%d')}")

    if len(a_level) > 50:
        print(f"\n... (Also {len(a_level) - 50} stocks)")
else:
    print("\n⚠️  Warning: no stock meets the Grade-A standard; consider relaxing the criteria")

# ============================================================================
# Step 4: show Grade-B high-quality stocks (partial)
# ============================================================================
print("\n" + "=" * 80)
print("[Step 4: Grade-B high-quality stocks (first 30)]")
print("=" * 80)

if len(b_level) > 0:
    print(f"\n total {len(b_level)} Grade-B stocks (first 30):\n")

    print(f"{'Rank':<6} {'Stock code':<12} {'Listing years':<10} {'Completeness rate%':<10} {'Quality score':<10} {'First trade':<15}")
    print("-" * 80)

    for idx, row in enumerate(b_level.head(30).itertuples(), 1):
        print(f"{idx:<6} {row.stock_code:<12} {row.listing_years:<10.1f} "
              f"{row.completeness_ratio:<10.2f} {row.quality_score:<10.2f} "
              f"{row.first_date.strftime('%Y-%m-%d'):<15}")

# ============================================================================
# Step 5: statistical analysis of the data
# ============================================================================
print("\n" + "=" * 80)
print("[Step 5: data volume of high-quality stocks]")
print("=" * 80)

if len(a_level) > 0:
    a_stocks = a_level['stock_code'].tolist()
    a_data = df[df['stock_code'].isin(a_stocks)]

    print(f"\nA-grade stock data statistics:")
    print(f"  Number of stocks: {len(a_stocks)} stocks")
    print(f"  Total number of records: {len(a_data):,} records")
    print(f"  Mean records per stock: {len(a_data) / len(a_stocks):.0f} records")
    print(f"  Time span: {a_data['trade_date'].min().strftime('%Y-%m-%d')} to {a_data['trade_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Mean listing years: {a_level['listing_years'].mean():.1f} years")
    print(f"  Mean data completeness: {a_level['completeness_ratio'].mean():.2f}%")

if len(b_level) > 0:
    b_stocks = b_level['stock_code'].tolist()
    b_data = df[df['stock_code'].isin(b_stocks)]

    print(f"\nB-grade stock data statistics:")
    print(f"  Number of stocks: {len(b_stocks)} stocks")
    print(f"  Total number of records: {len(b_data):,} records")
    print(f"  Mean records per stock: {len(b_data) / len(b_stocks):.0f} records")
    print(f"  Mean listing years: {b_level['listing_years'].mean():.1f} years")
    print(f"  Mean data completeness: {b_level['completeness_ratio'].mean():.2f}%")

# A+B-grade total
if len(a_level) > 0 or len(b_level) > 0:
    ab_stocks = list(set(a_level['stock_code'].tolist() + b_level['stock_code'].tolist()))
    ab_data = df[df['stock_code'].isin(ab_stocks)]

    print(f"\n [Recommended for the study] Grade A+B total:")
    print(f"  Number of stocks: {len(ab_stocks)} stocks")
    print(f"  Total number of records: {len(ab_data):,} records")
    print(f"  of the original data: {len(ab_data) / len(df) * 100:.1f}%")

# ============================================================================
# Step 6: visualization
# ============================================================================
print("\n" + "=" * 80)
print("[Step 6: create visual analysis]")
print("=" * 80)

# Figure 1: stock quality distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribution of quality scores
ax1 = axes[0, 0]
ax1.hist(quality_df['quality_score'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
ax1.axvline(85, color='red', linestyle='--', linewidth=2, label='A-grade threshold (85 points)')
ax1.axvline(75, color='orange', linestyle='--', linewidth=2, label='B-grade threshold (75 points)')
ax1.set_xlabel('Quality score', fontsize=12)
ax1.set_ylabel('Number of stocks', fontsize=12)
ax1.set_title('Distribution of stock quality scores', fontsize=14, fontweight='bold', pad=15)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Distribution of data completeness
ax2 = axes[0, 1]
ax2.hist(quality_df['completeness_ratio'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
ax2.axvline(99, color='red', linestyle='--', linewidth=2, label='A-grade threshold (99%)')
ax2.axvline(95, color='orange', linestyle='--', linewidth=2, label='B-grade threshold (95%)')
ax2.set_xlabel('Data completeness rate (%)', fontsize=12)
ax2.set_ylabel('Number of stocks', fontsize=12)
ax2.set_title('Distribution of data completeness', fontsize=14, fontweight='bold', pad=15)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Distribution of listing years
ax3 = axes[1, 0]
ax3.hist(quality_df['listing_years'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
ax3.axvline(5, color='red', linestyle='--', linewidth=2, label='A-grade threshold (5 years)')
ax3.axvline(3, color='orange', linestyle='--', linewidth=2, label='B-grade threshold (3 years)')
ax3.set_xlabel('Listing years (years)', fontsize=12)
ax3.set_ylabel('Number of stocks', fontsize=12)
ax3.set_title('Distribution of listing years', fontsize=14, fontweight='bold', pad=15)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Pie chart of stock grades
ax4 = axes[1, 1]
sizes = [len(a_level), len(b_level), len(c_level),
         len(quality_df) - len(a_level) - len(b_level) - len(c_level)]
labels = [f'A-grade\n{len(a_level)} stocks', f'B-grade\n{len(b_level)} stocks',
          f'C-grade\n{len(c_level)} stocks', f'Not recommended\n{sizes[3]} stocks']
colors = ['#2ecc71', '#3498db', '#f39c12', '#95a5a6']
explode = (0.1, 0.05, 0, 0)

ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, textprops={'fontsize': 11})
ax4.set_title('Distribution of stock quality grades', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('stock_quality_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stock_quality_analysis.png")
plt.close()

# Figure 2: Grade-A stock detailed analysis
if len(a_level) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Quality scores of the first 20 Grade-A stocks
    ax1 = axes[0]
    top20 = a_level.head(20)
    bars = ax1.barh(range(len(top20)), top20['quality_score'], color='#2ecc71', alpha=0.8)
    ax1.set_yticks(range(len(top20)))
    ax1.set_yticklabels(top20['stock_code'])
    ax1.set_xlabel('Quality score', fontsize=12)
    ax1.set_title('A-grade stocks: TOP20 quality scores', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Label the scores on the bar chart
    for i, (bar, score) in enumerate(zip(bars, top20['quality_score'])):
        ax1.text(score + 0.5, i, f'{score:.1f}', va='center', fontsize=9)

    # 2. A-grade stocks: distribution of listing years
    ax2 = axes[1]
    ax2.scatter(a_level['listing_years'], a_level['quality_score'],
                s=100, alpha=0.6, color='#3498db', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Listing years (years)', fontsize=12)
    ax2.set_ylabel('Quality score', fontsize=12)
    ax2.set_title('A-grade stocks: listing years vs quality score', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grade_A_stock_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: grade_A_stock_analysis.png")
    plt.close()

# ============================================================================
# Step 7: export results
# ============================================================================
print("\n" + "=" * 80)
print("[Step 7: export screening results]")
print("=" * 80)

# Export the full quality report
quality_df.to_csv('stock_quality_report_all.csv', index=False, encoding='utf-8-sig')
print("✓ Saved: stock_quality_report_all.csv")

# Export the Grade-A stock list
if len(a_level) > 0:
    a_level.to_csv('grade_A_stock_list.csv', index=False, encoding='utf-8-sig')
    print("✓ Saved: grade_A_stock_list.csv")

    # Export the raw data of Grade-A stocks
    a_stocks = a_level['stock_code'].tolist()
    a_data = df[df['stock_code'].isin(a_stocks)]
    a_data.to_csv('grade_A_stock_full_data.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Saved: grade_A_stock_full_data.csv ({len(a_data):,} records)")

# Export the Grade-B stock list
if len(b_level) > 0:
    b_level.to_csv('grade_B_stock_list.csv', index=False, encoding='utf-8-sig')
    print("✓ Saved: grade_B_stock_list.csv")

# Export recommended research stocks (Grade A+B)
if len(a_level) > 0 or len(b_level) > 0:
    recommended = pd.concat([a_level, b_level]).sort_values('quality_score', ascending=False)
    recommended.to_csv('recommended_stock_list_grade_AB.csv', index=False, encoding='utf-8-sig')
    print("✓ Saved: recommended_stock_list_grade_AB.csv")

    # Export the raw data of recommended stocks
    rec_stocks = recommended['stock_code'].tolist()
    rec_data = df[df['stock_code'].isin(rec_stocks)]
    rec_data.to_csv('recommended_stock_full_data_grade_AB.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Saved: recommended_stock_full_data_grade_AB.csv ({len(rec_data):,} records)")

print("\n" + "=" * 80)
print("[Analysis complete]")
print("=" * 80)
print("\n📊 Created files:")
print("  1. stock_quality_analysis.png - overall quality distribution")
if len(a_level) > 0:
    print("  2. grade_A_stock_analysis.png - A-grade stock analysis")
print("  3. stock_quality_report_all.csv - full quality assessment data")
if len(a_level) > 0:
    print("  4. grade_A_stock_list.csv - A-grade stock list")
    print("  5. grade_A_stock_full_data.csv - A-grade stock trading data")
if len(b_level) > 0:
    print("  6. grade_B_stock_list.csv - B-grade stock list")
if len(a_level) > 0 or len(b_level) > 0:
    print("  7. recommended_stock_list_grade_AB.csv - stocks recommended for the study")
    print("  8. recommended_stock_full_data_grade_AB.csv - trading data of recommended stocks")

print("\n💡 Recommendation:")
print(f"  • Prefer Grade-A stocks ({len(a_level)} stocks) for the study")
print(f"  • A+B-grade total ({len(a_level) + len(b_level)} stocks) can provide a larger sample size")
print("  • These stocks have high data quality, little noise and continuous trading, suitable for deep learning research")
print("  • More refined prediction models can be designed for these high-quality stocks")
