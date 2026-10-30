# -*- coding: utf-8 -*-
"""
Step 3 (diagnostic only): data quality checks, suspension and missing-trading-day analysis.
This step writes no files used by later steps.
"""

#Data quality step 1 of 3
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

# 1. Read data
print("=" * 80)
print("Reading data...")
df = pd.read_csv('toAB-.csv')
print("Data reading complete!\n")

# 2. Basic information
print("=" * 80)
print("[1. Basic data information]")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Total rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print(f"\n Column names: {list(df.columns)}")
print(f"\n Data types:\n{df.dtypes}")
print(f"\n Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 3. Preview of the data before and after
print("\n" + "=" * 80)
print("[2. Data preview]")
print("=" * 80)
print("First 10 rows:")
print(df.head(10))
print("\n Last 10 rows:")
print(df.tail(10))

# 4. Descriptive statistics of the data
print("\n" + "=" * 80)
print("[3. Descriptive statistics of numeric columns]")
print("=" * 80)
print(df.describe())

# 5. Missing-value analysis
print("\n" + "=" * 80)
print("[4. Missing-value analysis]")
print("=" * 80)
missing_stats = pd.DataFrame({
    'Number missing': df.isnull().sum(),
    'Share missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_stats)
print(f"\n Total number of missing values: {df.isnull().sum().sum()}")

# 6. Unique-value analysis
print("\n" + "=" * 80)
print("[5. Unique-value statistics]")
print("=" * 80)
unique_stats = pd.DataFrame({
    'Number of unique values': df.nunique(),
    'Share of unique values (%)': (df.nunique() / len(df) * 100).round(2)
})
print(unique_stats)

# 7. Date column analysis (if present)
print("\n" + "=" * 80)
print("[6. Date-related analysis]")
print("=" * 80)
date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col or 'time' in col.lower()]
if date_cols:
    for col in date_cols:
        print(f"\n Column names: {col}")
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"  Earliest date: {df[col].min()}")
            print(f"  Latest date: {df[col].max()}")
            print(f"  Time span: {(df[col].max() - df[col].min()).days} days")
            print(f"  Data types: {df[col].dtype}")
        except:
            print(f"  (cannot be converted to a date format)")
            print(f"  Example values: {df[col].head(3).tolist()}")
else:
    print("No obvious date column detected; please check the column names")
    print("All column names:", list(df.columns))

# 8. Detailed analysis of numeric columns
print("\n" + "=" * 80)
print("[7. Detailed analysis of numeric columns]")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    print(f"\n Column names: {col}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Standard deviation: {df[col].std():.2f}")
    print(f"  Minimum: {df[col].min():.2f}")
    print(f"  Maximum: {df[col].max():.2f}")
    print(f"  Skewness: {df[col].skew():.2f}")
    print(f"  Kurtosis: {df[col].kurtosis():.2f}")
    # Outlier detection (IQR method)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"  Number of outliers (IQR): {outliers} ({outliers/len(df)*100:.2f}%)")

# 9. Categorical column analysis
print("\n" + "=" * 80)
print("[8. Categorical column analysis]")
print("=" * 80)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    for col in categorical_cols:
        if df[col].nunique() <= 20:  # Show only columns with <= 20 categories
            print(f"\n Column names: {col}")
            print(f"  Number of classes: {df[col].nunique()}")
            print("  value distribution:")
            print(df[col].value_counts().head(10))
        else:
            print(f"\n Column names: {col}")
            print(f"  Number of classes: {df[col].nunique()} (Too many categories; showing only the first 10)")
            print(df[col].value_counts().head(10))
else:
    print("No categorical columns detected")

# 10. Data quality check
print("\n" + "=" * 80)
print("[9. Data quality check]")
print("=" * 80)
print(f"Number of duplicate rows: {df.duplicated().sum()}")
print(f"Number of completely empty rows: {df.isnull().all(axis=1).sum()}")

# Check numeric columns for zero and negative values
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    neg_count = (df[col] < 0).sum()
    if zero_count > 0 or neg_count > 0:
        print(f"\n columns '{col}':")
        if zero_count > 0:
            print(f"  Number of zero values: {zero_count} ({zero_count/len(df)*100:.2f}%)")
        if neg_count > 0:
            print(f"  Number of negative values: {neg_count} ({neg_count/len(df)*100:.2f}%)")

# 11. Create visualizations
print("\n" + "=" * 80)
print("[10. Creating figures]")
print("=" * 80)

if numeric_cols:
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3

    # Distribution plot
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution plot')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_distribution.png")
    plt.close()

    # Box plot
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col].dropna(), vert=True)
        axes[idx].set_title(f'{col} Box plot')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True, alpha=0.3)

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: boxplot.png")
    plt.close()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, fmt='.2f')
        plt.title('Feature correlation heatmap', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: correlation_heatmap.png")
        plt.close()

print("\n" + "=" * 80)
print("[Analysis complete]")
print("=" * 80)
print("Created files:")
print("  1. data_distribution.png")
print("  2. boxplot.png")
if len(numeric_cols) > 1:
    print("  3. correlation_heatmap.png")
print("\n Please review the statistics above and the figures created to understand the basic state of the data.")


#Data quality step 2 of 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Reading data...")
df = pd.read_csv('toAB--.csv')

# Fix the stock code format by padding to 6 digits
df['stock_code'] = df['ts_code'].astype(str).str.zfill(6)
print("✓ Stock codes formatted to 6 digits")

# Convert the date format
# df['trade_date'] = pd.to_datetime(df['trade_date'])
# df['year'] = df['trade_date'].dt.year
# df['month'] = df['trade_date'].dt.to_period('M')
df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
df['year'] = df['trade_date'].dt.year
df['month'] = df['trade_date'].dt.to_period('M')

print("\n" + "=" * 80)
print("[1. Stock code statistics]")
print("=" * 80)

# Count the total number of stocks
unique_stocks = df['stock_code'].unique()
print(f"Total number of stocks in the dataset: {len(unique_stocks)} stocks")
print(f"\n First 20 stock codes:")
print(unique_stocks[:20])
print(f"\n Last 20 stock codes:")
print(unique_stocks[-20:])

# Show examples of specific stock codes
print(f"\n Stock codes of the first 10 records after correction:")
print(df[['stock_code', 'trade_date', 'close']].head(10))

print("\n" + "=" * 80)
print("[2. Number of listed stocks traded per year]")
print("=" * 80)

# Count stocks by year
yearly_stocks = df.groupby('year')['stock_code'].nunique().reset_index()
yearly_stocks.columns = ['Year', 'Number of stocks']

print(yearly_stocks.to_string(index=False))

# Count new stocks added each year
print("\n" + "=" * 80)
print("[3. Stocks newly listed each year]")
print("=" * 80)

# Find the first trading date of each stock
first_trade = df.groupby('stock_code')['trade_date'].min().reset_index()
first_trade.columns = ['stock_code', 'first_trade_date']
first_trade['first_year'] = first_trade['first_trade_date'].dt.year

# Count newly listed stocks by year
new_listings = first_trade.groupby('first_year').size().reset_index()
new_listings.columns = ['Year', 'newly listed stocks']

print(new_listings.to_string(index=False))

# Show stock codes newly listed each year (only the first and the most recent years)
print("\n Examples of stock codes newly listed in some years:")
for year in [1992, 1993, 1994, 1995, 2020, 2021, 2022, 2023, 2024, 2025]:
    stocks_in_year = first_trade[first_trade['first_year'] == year]['stock_code'].tolist()
    if stocks_in_year:
        print(f"\n{year} newly listed in year ({len(stocks_in_year)} stocks): {stocks_in_year[:10]}{'...' if len(stocks_in_year) > 10 else ''}")

print("\n" + "=" * 80)
print("[4. Suspension detection analysis]")
print("=" * 80)

# Get the full trading calendar (based on the trading dates of all stocks)
all_trade_dates = sorted(df['trade_date'].unique())
print(f"Total number of trading days covered by the dataset: {len(all_trade_dates)} days")
print(f"Time range: {all_trade_dates[0].strftime('%Y-%m-%d')} to {all_trade_dates[-1].strftime('%Y-%m-%d')}")

# Count trading records for each stock
stock_stats = df.groupby('stock_code').agg({
    'trade_date': ['count', 'min', 'max']
}).reset_index()
stock_stats.columns = ['stock_code', 'trading_days', 'first_date', 'last_date']

# Compute the number of trading days each stock should have (all trading days between its first and last trade)
def calculate_expected_days(row):
    # Find all trading days within this stock's trading period
    expected_dates = [d for d in all_trade_dates if row['first_date'] <= d <= row['last_date']]
    return len(expected_dates)

stock_stats['expected_days'] = stock_stats.apply(calculate_expected_days, axis=1)
stock_stats['missing_days'] = stock_stats['expected_days'] - stock_stats['trading_days']
stock_stats['missing_ratio'] = (stock_stats['missing_days'] / stock_stats['expected_days'] * 100).round(2)

# Compute the number of days listed
stock_stats['duration_days'] = (stock_stats['last_date'] - stock_stats['first_date']).dt.days

print(f"\n Statistics of stock trading completeness:")
print(f"{'Stock code':<12} {'actual trading days':<12} {'expected trading days':<12} {'Missing days':<12} {'Missing rate%':<12} {'First trade':<15} {'Last trade'}")
print("-" * 110)

# Sort by missing rate and show the 20 stocks with the most missing data
top_missing = stock_stats.nlargest(20, 'missing_ratio')
for _, row in top_missing.iterrows():
    print(f"{row['stock_code']:<12} {row['trading_days']:<12} {row['expected_days']:<12} "
          f"{row['missing_days']:<12} {row['missing_ratio']:<12.2f} "
          f"{row['first_date'].strftime('%Y-%m-%d'):<15} {row['last_date'].strftime('%Y-%m-%d')}")

# Suspension statistics summary
print("\n" + "=" * 80)
print("[5. Suspension summary]")
print("=" * 80)

print(f"Number of stocks with missing trading records: {(stock_stats['missing_days'] > 0).sum()} stocks")
print(f"Number of stocks with no missing data at all: {(stock_stats['missing_days'] == 0).sum()} stocks")
print(f"\n Mean actual trading days per stock: {stock_stats['trading_days'].mean():.0f} days")
print(f"Mean expected trading days per stock: {stock_stats['expected_days'].mean():.0f} days")
print(f"Mean missing days: {stock_stats['missing_days'].mean():.0f} days")
print(f"Mean missing rate: {stock_stats['missing_ratio'].mean():.2f}%")

# Statistics grouped by missing rate
print("\n Classify by suspension level:")
bins = [0, 1, 5, 10, 20, 50, 100]
labels = ['No suspension (<1%)', 'Slight suspension (1-5%)', 'Moderate suspension (5-10%)',
          'Frequent suspension (10-20%)', 'Very frequent suspension (20-50%)', 'Long-term suspension (>50%)']
stock_stats['suspension_level'] = pd.cut(stock_stats['missing_ratio'],
                                          bins=bins, labels=labels, include_lowest=True)
suspension_summary = stock_stats['suspension_level'].value_counts().sort_index()
print(suspension_summary)

# Detect consecutive suspensions
print("\n" + "=" * 80)
print("[6. Detection of long consecutive suspensions]")
print("=" * 80)
print("Detect suspensions longer than 30 consecutive trading days...\n")

def detect_long_suspensions(stock_code, min_gap_days=30):
    """Detect long suspensions of one stock"""
    stock_data = df[df['stock_code'] == stock_code].sort_values('trade_date')
    trade_dates = stock_data['trade_date'].tolist()

    suspensions = []
    for i in range(len(trade_dates) - 1):
        # Number of trading days that should exist between two trading days
        gap_start = trade_dates[i]
        gap_end = trade_dates[i + 1]
        expected_dates = [d for d in all_trade_dates if gap_start < d < gap_end]
        gap_days = len(expected_dates)

        if gap_days >= min_gap_days:
            suspensions.append({
                'stock_code': stock_code,
                'suspension_start': gap_start.strftime('%Y-%m-%d'),
                'suspension_end': gap_end.strftime('%Y-%m-%d'),
                'missing_days': gap_days
            })

    return suspensions

# Only check stocks with a suspension rate > 5%, to avoid long computation
stocks_to_check = stock_stats[stock_stats['missing_ratio'] > 5]['stock_code'].tolist()
print(f"Check stocks with suspension rate > 5% {len(stocks_to_check)} stocks...")

all_suspensions = []
for stock in stocks_to_check[:50]:  # Only check the first 50, to avoid long run times
    suspensions = detect_long_suspensions(stock)
    all_suspensions.extend(suspensions)

if all_suspensions:
    suspension_df = pd.DataFrame(all_suspensions)
    suspension_df = suspension_df.nlargest(30, 'missing_days')

    print(f"\n Long consecutive suspension cases found (Top 30):")
    print(f"{'Stock code':<12} {'Suspension start date':<15} {'Suspension end date':<15} {'Suspension days'}")
    print("-" * 65)
    for _, row in suspension_df.iterrows():
        print(f"{row['stock_code']:<12} {row['suspension_start']:<15} "
              f"{row['suspension_end']:<15} {row['missing_days']}")
else:
    print("No suspension longer than 30 days found")

# Visualization
print("\n" + "=" * 80)
print("[7. Creating figures]")
print("=" * 80)

# Figure 1: trend in the number of stocks per year
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(yearly_stocks['Year'], yearly_stocks['Number of stocks'],
         marker='o', linewidth=2, markersize=6, color='#2E86AB')
ax1.fill_between(yearly_stocks['Year'], yearly_stocks['Number of stocks'],
                  alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of stocks', fontsize=12)
ax1.set_title('Trend in the number of listed stocks traded per year', fontsize=14, pad=15, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(yearly_stocks['Year'].min() - 1, yearly_stocks['Year'].max() + 1)

# Mark the maximum and minimum on the figure
max_year = yearly_stocks.loc[yearly_stocks['Number of stocks'].idxmax()]
min_year = yearly_stocks.loc[yearly_stocks['Number of stocks'].idxmin()]
ax1.annotate(f"Highest: {int(max_year['Number of stocks'])} stocks",
             xy=(max_year['Year'], max_year['Number of stocks']),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Figure 2: number of newly listed stocks per year
ax2.bar(new_listings['Year'], new_listings['newly listed stocks'],
        color='#A23B72', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('newly listed stocks', fontsize=12)
ax2.set_title('Number of newly listed stocks per year', fontsize=14, pad=15, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.set_xlim(new_listings['Year'].min() - 1, new_listings['Year'].max() + 1)

plt.tight_layout()
plt.savefig('stock_listing_trend.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stock_listing_trend.png")
plt.close()

# Figure 3: distribution of suspensions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram of missing rates
ax1.hist(stock_stats['missing_ratio'], bins=50, color='#F18F01',
         edgecolor='black', alpha=0.7, linewidth=0.5)
ax1.set_xlabel('Missing rate (%)', fontsize=12)
ax1.set_ylabel('Number of stocks', fontsize=12)
ax1.set_title('Distribution of missing trading-day rates', fontsize=14, pad=15, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
ax1.axvline(stock_stats['missing_ratio'].mean(), color='red',
            linestyle='--', linewidth=2, label=f"Mean: {stock_stats['missing_ratio'].mean():.2f}%")
ax1.legend()

# Pie chart of suspension levels
colors_pie = ['#06A77D', '#C1D82F', '#F4D35E', '#F18F01', '#C73E1D', '#8B0000']
ax2.pie(suspension_summary.values, labels=suspension_summary.index,
        autopct='%1.1f%%', startangle=90, colors=colors_pie,
        textprops={'fontsize': 10})
ax2.set_title('Distribution of stock suspension levels', fontsize=14, pad=15, fontweight='bold')

plt.tight_layout()
plt.savefig('stock_suspension_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stock_suspension_analysis.png")
plt.close()

# Save detailed statistics to CSV
stock_stats_export = stock_stats[['stock_code', 'trading_days', 'expected_days',
                                   'missing_days', 'missing_ratio',
                                   'first_date', 'last_date']].copy()
stock_stats_export = stock_stats_export.sort_values('missing_ratio', ascending=False)
stock_stats_export.to_csv('stock_suspension_statistics.csv', index=False, encoding='utf-8-sig')
print("✓ Saved: stock_suspension_statistics.csv")

print("\n" + "=" * 80)
print("[Analysis complete]")
print("=" * 80)
print("Created files:")
print("  1. stock_listing_trend.png - number of stocks and new listings per year")
print("  2. stock_suspension_analysis.png - distribution of suspensions")
print("  3. stock_suspension_statistics.csv - detailed suspension data for each stock")


"""
Detailed analysis of missing trading days per stock
Analyze the time distribution, position and common patterns of missing days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("[Detailed analysis of missing trading days per stock]")
print("=" * 100)

# Read data
df = pd.read_csv('toAB--.csv')
df['stock_code'] = df['ts_code'].astype(str)
df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
df['year'] = df['trade_date'].dt.year

# Get the full trading calendar
all_trade_dates = sorted(df['trade_date'].unique())
all_trade_dates_set = set(all_trade_dates)

print(f"\n Data time range: {all_trade_dates[0].strftime('%Y-%m-%d')} to {all_trade_dates[-1].strftime('%Y-%m-%d')}")
print(f"Total trading days: {len(all_trade_dates)} days\n")

# Analyze the missing data of each stock
def analyze_missing_dates(stock_code):
    """Detailed analysis of the missing trading days of one stock"""
    stock_data = df[df['stock_code'] == stock_code].sort_values('trade_date')

    if len(stock_data) == 0:
        return None

    first_date = stock_data['trade_date'].min()
    last_date = stock_data['trade_date'].max()
    actual_dates = set(stock_data['trade_date'])

    # All trading days this stock should have in its period
    expected_dates = [d for d in all_trade_dates if first_date <= d <= last_date]
    expected_dates_set = set(expected_dates)

    # Missing trading days
    missing_dates = sorted(expected_dates_set - actual_dates)

    if not missing_dates:
        return None

    # Analyze the time distribution of missing dates
    result = {
        'stock_code': stock_code,
        'first_trade': first_date,
        'last_trade': last_date,
        'trading_days': len(actual_dates),
        'expected_days': len(expected_dates),
        'missing_days': len(missing_dates),
        'missing_ratio': len(missing_dates) / len(expected_dates) * 100,
        'missing_dates_list': missing_dates,
    }

    # Compute the position characteristics of missing dates
    total_period = (last_date - first_date).days

    # Early-period missing (first 20% of the time)
    early_cutoff = first_date + timedelta(days=total_period * 0.2)
    early_missing = [d for d in missing_dates if d <= early_cutoff]

    # Middle-period missing (middle 60% of the time)
    mid_start = first_date + timedelta(days=total_period * 0.2)
    mid_end = first_date + timedelta(days=total_period * 0.8)
    mid_missing = [d for d in missing_dates if mid_start < d <= mid_end]

    # Late-period missing (last 20% of the time)
    late_missing = [d for d in missing_dates if d > mid_end]

    result['early_missing_count'] = len(early_missing)
    result['mid_missing_count'] = len(mid_missing)
    result['late_missing_count'] = len(late_missing)
    result['early_missing_dates'] = early_missing
    result['mid_missing_dates'] = mid_missing
    result['late_missing_dates'] = late_missing

    # Detect consecutive missing segments
    consecutive_segments = []
    if missing_dates:
        segment_start = missing_dates[0]
        segment_end = missing_dates[0]

        for i in range(1, len(missing_dates)):
            # Check continuity (in trading days)
            prev_date = missing_dates[i-1]
            curr_date = missing_dates[i]

            # Find the trading days between two missing dates
            dates_between = [d for d in all_trade_dates if prev_date < d < curr_date]

            if len(dates_between) == 0:  # Consecutive missing
                segment_end = curr_date
            else:  # Not continuous; start a new segment
                segment_length = len([d for d in all_trade_dates if segment_start <= d <= segment_end])
                consecutive_segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'length': segment_length
                })
                segment_start = curr_date
                segment_end = curr_date

        # Add the last segment
        segment_length = len([d for d in all_trade_dates if segment_start <= d <= segment_end])
        consecutive_segments.append({
            'start': segment_start,
            'end': segment_end,
            'length': segment_length
        })

    result['consecutive_segments'] = consecutive_segments
    result['max_consecutive_missing'] = max([seg['length'] for seg in consecutive_segments]) if consecutive_segments else 0
    result['num_segments'] = len(consecutive_segments)

    return result

# Analyze all stocks with missing data
print("Analyzing missing data of all stocks...\n")
all_stocks = df['stock_code'].unique()
missing_analysis = []

for stock in all_stocks:
    analysis = analyze_missing_dates(stock)
    if analysis:
        missing_analysis.append(analysis)

# Sort by missing rate
missing_analysis.sort(key=lambda x: x['missing_ratio'], reverse=True)

# ============================================================================
# 1. Print detailed missing-data information for each stock
# ============================================================================
print("=" * 100)
print("[1. Detailed missing-day statistics per stock]")
print("=" * 100)
print(f"\n{'Stock code':<12} {'Missing days':<10} {'Missing rate%':<10} {'Early':<8} {'Middle':<8} {'Late':<8} "
      f"{'Longest consecutive':<10} {'Number of missing segments':<10}")
print("-" * 100)

for item in missing_analysis[:30]:  # Show the 30 with the most missing data
    print(f"{item['stock_code']:<12} {item['missing_days']:<10} {item['missing_ratio']:<10.2f} "
          f"{item['early_missing_count']:<8} {item['mid_missing_count']:<8} {item['late_missing_count']:<8} "
          f"{item['max_consecutive_missing']:<10} {item['num_segments']:<10}")

# ============================================================================
# 2. Analyze the time distribution of missing periods
# ============================================================================
print("\n" + "=" * 100)
print("[2. Distribution of missing periods]")
print("=" * 100)

early_total = sum([item['early_missing_count'] for item in missing_analysis])
mid_total = sum([item['mid_missing_count'] for item in missing_analysis])
late_total = sum([item['late_missing_count'] for item in missing_analysis])
total_missing = early_total + mid_total + late_total

print(f"\n Overall missing distribution:")
print(f"  Early-period missing (first 20% of the period): {early_total} days ({early_total/total_missing*100:.1f}%)")
print(f"  Middle-period missing (middle 60% of the period): {mid_total} days ({mid_total/total_missing*100:.1f}%)")
print(f"  Late-period missing (last 20% of the period): {late_total} days ({late_total/total_missing*100:.1f}%)")

# ============================================================================
# 3. Analyze dates missing for several stocks at once
# ============================================================================
print("\n" + "=" * 100)
print("[3. Common missing dates analysis]")
print("=" * 100)

# Count how many stocks are missing on each date
all_missing_dates = []
for item in missing_analysis:
    all_missing_dates.extend(item['missing_dates_list'])

missing_date_counter = Counter(all_missing_dates)
common_missing = [(date, count) for date, count in missing_date_counter.most_common(50) if count >= 2]

if common_missing:
    print(f"\n has {len(common_missing)} dates are missing for 2 or more stocks at the same time")
    print(f"\n{'Date':<15} {'Number of stocks missing':<12} {'Share%':<10}")
    print("-" * 40)
    for date, count in common_missing[:30]:
        print(f"{date.strftime('%Y-%m-%d'):<15} {count:<12} {count/len(missing_analysis)*100:<10.1f}")

    # Analyze the time distribution of these common missing dates
    common_dates = [date for date, count in common_missing]
    common_years = Counter([d.year for d in common_dates])
    print(f"\n Distribution of common missing dates by year:")
    for year, count in sorted(common_years.items()):
        print(f"  {year} years: {count} days")
else:
    print("\n No date with missing data shared by 2 or more stocks")

# ============================================================================
# 4. Detailed analysis of consecutive missing segments
# ============================================================================
print("\n" + "=" * 100)
print("[4. Detailed analysis of consecutive missing segments (showing consecutive missing≥10 days)]")
print("=" * 100)

all_segments = []
for item in missing_analysis:
    for seg in item['consecutive_segments']:
        if seg['length'] >= 10:  # Consider only consecutive missing data≥10-day
            all_segments.append({
                'stock_code': item['stock_code'],
                'start': seg['start'],
                'end': seg['end'],
                'length': seg['length']
            })

all_segments.sort(key=lambda x: x['length'], reverse=True)

if all_segments:
    print(f"\n found {len(all_segments)} consecutive missing≥10-day period")
    print(f"\n{'Stock code':<12} {'Start date':<15} {'End date':<15} {'Consecutive days':<10}")
    print("-" * 60)
    for seg in all_segments[:50]:  # Show the first 50
        print(f"{seg['stock_code']:<12} {seg['start'].strftime('%Y-%m-%d'):<15} "
              f"{seg['end'].strftime('%Y-%m-%d'):<15} {seg['length']:<10}")
else:
    print("\n No consecutive missing data found≥10-day cases")

# ============================================================================
# 5. Detailed missing dates of a specific stock
# ============================================================================
print("\n" + "=" * 100)
print("[5. Detailed missing dates of the TOP 5 stocks with the most missing data]")
print("=" * 100)

for item in missing_analysis[:5]:
    print(f"\n{'='*80}")
    print(f"Stock code: {item['stock_code']}")
    print(f"Trading period: {item['first_trade'].strftime('%Y-%m-%d')} to {item['last_trade'].strftime('%Y-%m-%d')}")
    print(f"Missing days: {item['missing_days']} days (missing rate {item['missing_ratio']:.2f}%)")
    print(f"Missing distribution: early period {item['early_missing_count']} days, middle {item['mid_missing_count']} days, late {item['late_missing_count']} days")
    print(f"Longest consecutive missing period: {item['max_consecutive_missing']} days")
    print(f"Number of missing segments: {item['num_segments']} segments")

    print(f"\n Details of consecutive missing segments:")
    for i, seg in enumerate(item['consecutive_segments'][:10], 1):  # Show only the first 10 segments
        print(f"  No. {i} segments: {seg['start'].strftime('%Y-%m-%d')} to {seg['end'].strftime('%Y-%m-%d')} "
              f"(Consecutive {seg['length']} days)")

    if item['num_segments'] > 10:
        print(f"  ... Also {item['num_segments'] - 10} segments not shown")

# ============================================================================
# 6. Visual analysis
# ============================================================================
print("\n" + "=" * 100)
print("[6. Creating figures]")
print("=" * 100)

fig = plt.figure(figsize=(18, 12))

# Subplot 1: position of missing data (early, middle, late period)
ax1 = plt.subplot(2, 3, 1)
position_data = [early_total, mid_total, late_total]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax1.bar(['Early (0-20%)', 'Middle (20-80%)', 'Late (80-100%)'], position_data, color=colors, edgecolor='black')
ax1.set_ylabel('Missing days', fontsize=11)
ax1.set_title('Position in time of the missing trading days', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(position_data):
    ax1.text(i, v + max(position_data)*0.02, f'{v}\n({v/total_missing*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

# Subplot 2: distribution of the longest consecutive missing days
ax2 = plt.subplot(2, 3, 2)
max_consecutive = [item['max_consecutive_missing'] for item in missing_analysis]
ax2.hist(max_consecutive, bins=20, color='#F39C12', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Longest consecutive missing days', fontsize=11)
ax2.set_ylabel('Number of stocks', fontsize=11)
ax2.set_title('Distribution of the longest consecutive missing days per stock', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axvline(np.mean(max_consecutive), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(max_consecutive):.1f} days')
ax2.legend()

# Subplot 3: distribution of the number of missing segments
ax3 = plt.subplot(2, 3, 3)
num_segments = [item['num_segments'] for item in missing_analysis]
ax3.hist(num_segments, bins=20, color='#9B59B6', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Number of missing segments', fontsize=11)
ax3.set_ylabel('Number of stocks', fontsize=11)
ax3.set_title('Distribution of the number of missing segments per stock', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.axvline(np.mean(num_segments), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(num_segments):.1f} segments')
ax3.legend()

# Subplot 4: heatmap of common missing dates (by year and month)
ax4 = plt.subplot(2, 3, 4)
if common_missing:
    common_dates_df = pd.DataFrame([(d, c) for d, c in common_missing],
                                    columns=['date', 'count'])
    common_dates_df['year_month'] = pd.to_datetime(common_dates_df['date']).dt.to_period('M')
    monthly_common = common_dates_df.groupby('year_month')['count'].sum().reset_index()
    monthly_common['year_month'] = monthly_common['year_month'].astype(str)

    if len(monthly_common) > 30:
        monthly_common = monthly_common.nlargest(30, 'count')

    ax4.barh(range(len(monthly_common)), monthly_common['count'], color='#E74C3C', edgecolor='black')
    ax4.set_yticks(range(len(monthly_common)))
    ax4.set_yticklabels(monthly_common['year_month'], fontsize=8)
    ax4.set_xlabel('Total number of common missing occurrences', fontsize=11)
    ax4.set_title('Distribution of common missing dates by month (TOP30)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
else:
    ax4.text(0.5, 0.5, 'No common missing dates', ha='center', va='center', fontsize=14)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

# Subplot 5: missing days of the TOP10 stocks
ax5 = plt.subplot(2, 3, 5)
top10 = missing_analysis[:10]
stock_codes = [item['stock_code'] for item in top10]
early_counts = [item['early_missing_count'] for item in top10]
mid_counts = [item['mid_missing_count'] for item in top10]
late_counts = [item['late_missing_count'] for item in top10]

x = np.arange(len(stock_codes))
width = 0.25

ax5.bar(x - width, early_counts, width, label='Early', color='#FF6B6B', edgecolor='black')
ax5.bar(x, mid_counts, width, label='Middle', color='#4ECDC4', edgecolor='black')
ax5.bar(x + width, late_counts, width, label='Late', color='#45B7D1', edgecolor='black')

ax5.set_xlabel('Stock code', fontsize=11)
ax5.set_ylabel('Missing days', fontsize=11)
ax5.set_title('TOP10 Period distribution of stocks with missing data', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(stock_codes, rotation=45, ha='right', fontsize=9)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Subplot 6: classification of missing patterns
ax6 = plt.subplot(2, 3, 6)
# Classify by the period in which missing data are concentrated
patterns = []
for item in missing_analysis:
    early_pct = item['early_missing_count'] / item['missing_days'] * 100
    mid_pct = item['mid_missing_count'] / item['missing_days'] * 100
    late_pct = item['late_missing_count'] / item['missing_days'] * 100

    if early_pct > 60:
        patterns.append('Early-period concentrated')
    elif late_pct > 60:
        patterns.append('Late-period concentrated')
    elif mid_pct > 60:
        patterns.append('Middle-period concentrated')
    else:
        patterns.append('Dispersed')

pattern_counts = Counter(patterns)
colors_pie = ['#FF6B6B', '#45B7D1', '#4ECDC4', '#F39C12']
ax6.pie(pattern_counts.values(), labels=pattern_counts.keys(), autopct='%1.1f%%',
        startangle=90, colors=colors_pie, textprops={'fontsize': 10})
ax6.set_title('Classification of stock missing patterns', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('missing_trading_days_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: missing_trading_days_analysis.png")

# ============================================================================
# 7. Export detailed data to Excel
# ============================================================================
print("\n Exporting detailed data to Excel...")

# Data to export
export_data = []
for item in missing_analysis:
    export_data.append({
        'Stock code': item['stock_code'],
        'First trade': item['first_trade'].strftime('%Y-%m-%d'),
        'Last trade': item['last_trade'].strftime('%Y-%m-%d'),
        'actual trading days': item['trading_days'],
        'expected trading days': item['expected_days'],
        'Missing days': item['missing_days'],
        'Missing rate (%)': round(item['missing_ratio'], 2),
        'Early-period missing': item['early_missing_count'],
        'Middle-period missing': item['mid_missing_count'],
        'Late-period missing': item['late_missing_count'],
        'Longest consecutive missing period': item['max_consecutive_missing'],
        'Number of missing segments': item['num_segments']
    })

export_df = pd.DataFrame(export_data)

# Use ExcelWriter to create a multi-sheet Excel
with pd.ExcelWriter('stock_missing_data_analysis.xlsx', engine='openpyxl') as writer:
    # Sheet1: Summary statistics
    export_df.to_excel(writer, sheet_name='Summary statistics', index=False)

    # Sheet2: common missing dates
    if common_missing:
        common_df = pd.DataFrame([
            {'Date': date.strftime('%Y-%m-%d'), 'Number of stocks missing': count, 'Share (%)': round(count/len(missing_analysis)*100, 1)}
            for date, count in common_missing
        ])
        common_df.to_excel(writer, sheet_name='common missing dates', index=False)

    # Sheet3-Sheet7: TOP5 Detailed missing dates of stocks
    for i, item in enumerate(missing_analysis[:5], 1):
        missing_dates_detail = pd.DataFrame({
            'Missing date': [d.strftime('%Y-%m-%d') for d in item['missing_dates_list']],
            'Year': [d.year for d in item['missing_dates_list']],
            'Month': [d.month for d in item['missing_dates_list']]
        })
        sheet_name = f'{item["stock_code"]}_missing details'
        missing_dates_detail.to_excel(writer, sheet_name=sheet_name, index=False)

print("✓ Saved: stock_missing_data_analysis.xlsx")

print("\n" + "=" * 100)
print("[Analysis complete]")
print("=" * 100)
print("\n Created files:")
print("  1. missing_trading_days_analysis.png - 6 subplots showing the missing-data pattern")
print("  2. stock_missing_data_analysis.xlsx - includes summary statistics, common missing dates and details of the TOP5 stocks")
print("\n Main findings:")
print(f"  • There are {len(missing_analysis)} stocks have missing trading days")
print(f"  • Missing data mainly concentrated in: {'Early' if early_total > max(mid_total, late_total) else 'Middle' if mid_total > late_total else 'Late'}")
print(f"  • Mean longest consecutive missing period: {np.mean(max_consecutive):.1f} days")
print(f"  • Mean number of missing segments: {np.mean(num_segments):.1f} segments")
if common_missing:
    print(f"  • found {len(common_missing)} common missing dates")
print("=" * 100)
