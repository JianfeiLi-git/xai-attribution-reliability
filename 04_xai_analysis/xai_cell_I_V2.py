# -*- coding: utf-8 -*-
"""
xai_cell_I_V2.txt

Purpose:
1. Read only the frozen results of all 25 conventional baselines from C_V3;
2. No model is retrained or recomputed;
3. Create Supplementary Table S1;
4. Report Logistic Regression, Extra Trees and Naive Momentum in full;
5. Fix the V1 handling of the empty Weighting value of Naive Momentum;
6. Hard checks on input row counts, datasets, baselines and weighting combinations.

Input:
./0924/xai_cell_C_V3/C1_baseline_weight_sensitivity.csv

Output folder:
./0924/xai_cell_I_V2/
"""

from pathlib import Path
import pandas as pd

VERSION = "xai_cell_I_V2"
OUT_DIR = Path("./0924/xai_cell_I_V2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = Path(
    "./0924/xai_cell_C_V3/"
    "C1_baseline_weight_sensitivity.csv"
)

CSV_OUT = OUT_DIR / "Supplementary_Table_S1_full_baselines.csv"
XLSX_OUT = OUT_DIR / "Supplementary_Table_S1_full_baselines.xlsx"

SEP = "-" * 20

print(str(OUT_DIR))
print(SEP)
print("Start creating Supplementary Table S1")


# ====================
# 1. Input hard check
# ====================

if not SOURCE.is_file():
    raise FileNotFoundError(
        f"Fixed input file does not exist: {SOURCE}"
    )

df = pd.read_csv(SOURCE)

required_cols = [
    "Dataset",
    "Baseline",
    "Weighting",
    "Macro_F1",
    "Accuracy",
]

missing_cols = [
    c for c in required_cols
    if c not in df.columns
]
if missing_cols:
    raise RuntimeError(
        f"Required fields missing: {missing_cols}"
    )

df = df[required_cols].copy()

# C_V3 Naive Momentum does not use class weighting,
# CSV After reading, this field appears as NaN; the formal supplementary table shows it as N/A.
df["Weighting"] = (
    df["Weighting"]
    .fillna("N/A")
    .astype(str)
    .str.strip()
)

# Compatible with possible string-type empty values.
df.loc[
    df["Weighting"].str.lower().isin(
        ["nan", "none", ""]
    ),
    "Weighting",
] = "N/A"

expected_datasets = [
    "Medical-1",
    "Medical-2",
    "Power",
    "US-Med",
    "US-Pow",
]

expected_pairs = [
    ("Naive Momentum", "N/A"),
    ("Logistic Regression", "Unweighted"),
    ("Logistic Regression", "Balanced"),
    ("Extra Trees", "Unweighted"),
    ("Extra Trees", "Balanced"),
]

if len(df) != 25:
    raise RuntimeError(
        f"Abnormal number of result rows: actual={len(df)}, expected=25"
    )

actual_datasets = set(df["Dataset"].astype(str))
if actual_datasets != set(expected_datasets):
    raise RuntimeError(
        "Abnormal dataset set:"
        f"Actual={sorted(actual_datasets)}"
    )

for dataset in expected_datasets:
    sub = df[df["Dataset"] == dataset].copy()

    if len(sub) != 5:
        raise RuntimeError(
            f"{dataset} Abnormal number of rows:"
            f"Actual={len(sub)}, expected=5"
        )

    actual_pairs = set(
        zip(
            sub["Baseline"].astype(str),
            sub["Weighting"].astype(str),
        )
    )

    if actual_pairs != set(expected_pairs):
        raise RuntimeError(
            f"{dataset} baseline Abnormal combination:"
            f"{sorted(actual_pairs)}"
        )

if df[["Macro_F1", "Accuracy"]].isna().any().any():
    raise RuntimeError(
        "Macro_F1 or Accuracy has missing values"
    )

print("Input hard check: passed")
print(f"Input result rows: {len(df)}")


# ====================
# 2. Order matches the formal table
# ====================

dataset_order = {
    name: i
    for i, name in enumerate(expected_datasets)
}

baseline_order = {
    "Logistic Regression": 0,
    "Extra Trees": 1,
    "Naive Momentum": 2,
}

weight_order = {
    "Unweighted": 0,
    "Balanced": 1,
    "N/A": 2,
}

df["_dataset_order"] = df["Dataset"].map(dataset_order)
df["_baseline_order"] = df["Baseline"].map(baseline_order)
df["_weight_order"] = df["Weighting"].map(weight_order)

if df[
    [
        "_dataset_order",
        "_baseline_order",
        "_weight_order",
    ]
].isna().any().any():
    raise RuntimeError(
        "Undefined sort category found"
    )

table = (
    df.sort_values(
        [
            "_dataset_order",
            "_baseline_order",
            "_weight_order",
        ]
    )
    .drop(
        columns=[
            "_dataset_order",
            "_baseline_order",
            "_weight_order",
        ]
    )
    .reset_index(drop=True)
)

table.to_csv(
    CSV_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ====================
# 3. Excel Formal layout
# ====================

with pd.ExcelWriter(
    XLSX_OUT,
    engine="openpyxl",
) as writer:
    table.to_excel(
        writer,
        sheet_name="Table S1",
        index=False,
        startrow=1,
    )

    ws = writer.book["Table S1"]

    ws["A1"] = (
        "Supplementary Table S1. "
        "Performance of all conventional predictive baselines"
    )

    ws.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=5,
    )

    for col in ["D", "E"]:
        for row in range(3, 28):
            ws[f"{col}{row}"].number_format = "0.0000"

    widths = {
        "A": 14,
        "B": 22,
        "C": 14,
        "D": 14,
        "E": 14,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    ws.freeze_panes = "A3"


# ====================
# 4. Console check
# ====================

print(SEP)
print("Supplementary Table S1")
print(
    table.to_string(
        index=False,
        formatters={
            "Macro_F1": lambda x: f"{x:.4f}",
            "Accuracy": lambda x: f"{x:.4f}",
        },
    )
)

print(SEP)
print("Check of the best conventional baseline per dataset")

best_rows = (
    table.sort_values(
        ["Dataset", "Macro_F1"],
        ascending=[True, False],
    )
    .groupby(
        "Dataset",
        sort=False,
        as_index=False,
    )
    .first()
)

best_rows["_dataset_order"] = (
    best_rows["Dataset"].map(dataset_order)
)
best_rows = (
    best_rows
    .sort_values("_dataset_order")
    .drop(columns="_dataset_order")
)

print(
    best_rows[
        [
            "Dataset",
            "Baseline",
            "Weighting",
            "Macro_F1",
            "Accuracy",
        ]
    ].to_string(
        index=False,
        formatters={
            "Macro_F1": lambda x: f"{x:.4f}",
            "Accuracy": lambda x: f"{x:.4f}",
        },
    )
)

print(SEP)
print("Output complete")
print(f"CSV: {CSV_OUT}")
print(f"Excel: {XLSX_OUT}")
print("No model was retrained or recomputed")
