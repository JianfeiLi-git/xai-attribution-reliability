# -*- coding: utf-8 -*-
"""
Step 5: copy the processed U.S. data into the datasets folder used by the training scripts.
"""

"""
Organizing script: copy the U.S. data into the datasets folder, aligned with the A-share structure
- original files are not changed
- y file: pct_chg_next renamed to pct_chg
- folder and key changed to dataset4_m and dataset5_p
- Create phase5_data_us.pkl
"""

import pandas as pd
import pickle
import os
import shutil

SRC_BASE   = "./stock_data_tushare/processed"
DST_BASE   = "./datasets/processed"
SCALER_SRC = "./stock_data_tushare/scalers"
SCALER_DST = "./datasets/scalers"
PKL_DST    = "./datasets/phase5_data_us.pkl"

DATASETS = {
    "us_medical": "dataset4_m",
    "us_power":   "dataset5_p",
}

SPLITS = ["train", "val", "test"]

os.makedirs(SCALER_DST, exist_ok=True)

phase_data = {}

for src_key, dst_key in DATASETS.items():
    src_dir = os.path.join(SRC_BASE, src_key)
    dst_dir = os.path.join(DST_BASE, dst_key)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"\n Process: {src_key} → {dst_key}")

    for split in SPLITS:
        # Copy the X file
        x_src = os.path.join(src_dir, f"{split}_X.csv")
        x_dst = os.path.join(dst_dir, f"{split}_X.csv")
        shutil.copy2(x_src, x_dst)
        print(f"  ✓ {split}_X.csv Copy complete")

        # Copy the y file and rename the column
        y_src = os.path.join(src_dir, f"{split}_y.csv")
        y_dst = os.path.join(dst_dir, f"{split}_y.csv")
        y_df  = pd.read_csv(y_src)
        y_df  = y_df.rename(columns={"pct_chg_next": "pct_chg"})
        y_df.to_csv(y_dst, index=False)
        print(f"  ✓ {split}_y.csv Copy complete (pct_chg_next→pct_chg)")

    # Copy scaler
    scaler_src = os.path.join(SCALER_SRC, f"scaler_{src_key}.pkl")
    scaler_dst = os.path.join(SCALER_DST, f"scaler_{dst_key}.pkl")
    shutil.copy2(scaler_src, scaler_dst)
    print(f"  ✓ scaler_{dst_key}.pkl Copy complete")

    # Read data to create pkl
    X_train = pd.read_csv(os.path.join(dst_dir, "train_X.csv"))
    y_train = pd.read_csv(os.path.join(dst_dir, "train_y.csv"))
    X_val   = pd.read_csv(os.path.join(dst_dir, "val_X.csv"))
    y_val   = pd.read_csv(os.path.join(dst_dir, "val_y.csv"))
    X_test  = pd.read_csv(os.path.join(dst_dir, "test_X.csv"))
    y_test  = pd.read_csv(os.path.join(dst_dir, "test_y.csv"))

    phase_data[dst_key] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val':   X_val,
        'y_val':   y_val,
        'X_test':  X_test,
        'y_test':  y_test,
    }

# Save pkl
with open(PKL_DST, 'wb') as f:
    pickle.dump(phase_data, f)
print(f"\n✓ Saved: {PKL_DST}")

print("\n" + "="*60)
print("✅ All done")
print("="*60)
print(f"\n Folder structure created:")
for src_key, dst_key in DATASETS.items():
    print(f"  datasets/processed/{dst_key}/")
    for split in SPLITS:
        print(f"    {split}_X.csv / {split}_y.csv")
print(f"  datasets/scalers/scaler_dataset4_m.pkl")
print(f"  datasets/scalers/scaler_dataset5_p.pkl")
print(f"  datasets/phase5_data_us.pkl")
