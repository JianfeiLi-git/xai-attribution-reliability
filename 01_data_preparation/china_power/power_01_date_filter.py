# -*- coding: utf-8 -*-
"""
Step 1: keep records from 2000-01-01 onward (input: toAB-.csv).
"""

import pandas as pd

# 1. Read dataset
file_path = 'toAB-.csv'
print(f"Reading {file_path} ...")
df = pd.read_csv(file_path)

# 2. Data filtering
# According to 001-2.txt, the date column is named 'trade_date'
# According to 001-2.txt, this column is numeric (int64) and can be compared directly
# Keep data from 20000101 (inclusive) onward
original_count = len(df)
df_filtered = df[df['trade_date'] >= 20000101].copy()

# Count deleted records
filtered_count = len(df_filtered)
deleted_count = original_count - filtered_count

# 3. Save the new dataset
output_path = 'toAB--.csv'
df_filtered.to_csv(output_path, index=False)

print("="*40)
print("Processing complete!")
print(f"Original column names for reference: {list(df.columns)}") # corresponds to the column names in 001-2.txt
print(f"Original number of rows: {original_count}")
print(f"Number of rows after filtering: {filtered_count}")
print(f"Number of deleted rows: {deleted_count}")
print(f"File saved as: {output_path}")
print("="*40)
