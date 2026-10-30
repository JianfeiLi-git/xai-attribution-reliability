# -*- coding: utf-8 -*-
"""
Step 2: check for missing values.
"""

import pandas as pd

# 1. Read data (using the file name used in 002.txt)
file_name = 'toAB--.csv'
print(f"Reading {file_name} ...")
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: file not found {file_name}; please check that the file name is correct.")
    exit()

# 2. Check whether each row has missing values (NaN)
# df.isnull().any(axis=1) returns a boolean series marking rows with at least one missing value
missing_rows = df[df.isnull().any(axis=1)].copy()

print("=" * 80)
print(f"[Missing-value location report]")
print(f"Total number of missing values (cells): {df.isnull().sum().sum()}")
print(f"Number of rows with missing values: {len(missing_rows)}")
print("=" * 80)

# 3. Print the details of missing data
if len(missing_rows) > 0:
    print("\n Records with missing values:")
    print(f"{'Row index':<8} | {'Stock code':<10} | {'Trading date':<12} | {'Missing column names'}")
    print("-" * 60)

    for index, row in missing_rows.iterrows():
        # Get the names of the empty columns in this row
        null_columns = row[row.isnull()].index.tolist()

        # Get the identifier columns (try stock_code and trade_date)
        # Use .get() to avoid errors when column names do not match exactly
        stock = row.get('stock_code', row.get('ts_code', 'Unknown'))
        date = row.get('trade_date', row.get('date', 'Unknown'))

        print(f"{index:<8} | {str(stock):<10} | {str(date):<12} | {null_columns}")

    # Optional: save the results to CSV for inspection
    output_file = 'missing_values_detail.csv'
    missing_rows.to_csv(output_file, index=True)
    print("\n" + "=" * 80)
    print(f"Detailed missing-value records saved to: {output_file}")
    print("You can open this file to see the full data.")
else:
    print("No NaN missing values found in the dataset.")
