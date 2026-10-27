# -*- coding: utf-8 -*-
"""
xai_cell_I_V2.txt

目的：
1. 只读取已冻结的 C_V3 全部25个传统baseline结果；
2. 不重新训练、不重新计算任何模型；
3. 生成 Supplementary Table S1；
4. 完整报告 Logistic Regression、Extra Trees 和 Naive Momentum；
5. 修正 V1 对 Naive Momentum 的 Weighting 空值识别问题；
6. 对输入行数、数据集、baseline和权重组合进行硬校验。

输入：
./0924/xai_cell_C_V3/C1_baseline_weight_sensitivity.csv

输出目录：
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
print("开始生成Supplementary Table S1")


# ====================
# 1. 输入硬校验
# ====================

if not SOURCE.is_file():
    raise FileNotFoundError(
        f"固定输入文件不存在：{SOURCE}"
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
        f"缺少必要字段：{missing_cols}"
    )

df = df[required_cols].copy()

# C_V3中 Naive Momentum 不适用 class weighting，
# CSV读取后该字段表现为 NaN；正式补充表统一显示为 N/A。
df["Weighting"] = (
    df["Weighting"]
    .fillna("N/A")
    .astype(str)
    .str.strip()
)

# 兼容可能存在的字符串型空值。
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
        f"结果行数异常：实际={len(df)}，期望=25"
    )

actual_datasets = set(df["Dataset"].astype(str))
if actual_datasets != set(expected_datasets):
    raise RuntimeError(
        "数据集集合异常："
        f"实际={sorted(actual_datasets)}"
    )

for dataset in expected_datasets:
    sub = df[df["Dataset"] == dataset].copy()

    if len(sub) != 5:
        raise RuntimeError(
            f"{dataset} 行数异常："
            f"实际={len(sub)}，期望=5"
        )

    actual_pairs = set(
        zip(
            sub["Baseline"].astype(str),
            sub["Weighting"].astype(str),
        )
    )

    if actual_pairs != set(expected_pairs):
        raise RuntimeError(
            f"{dataset} baseline组合异常："
            f"{sorted(actual_pairs)}"
        )

if df[["Macro_F1", "Accuracy"]].isna().any().any():
    raise RuntimeError(
        "Macro_F1或Accuracy存在缺失值"
    )

print("输入硬校验：通过")
print(f"输入结果行数：{len(df)}")


# ====================
# 2. 排序与正式表
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
        "发现未定义的排序类别"
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
# 3. Excel正式排版
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
# 4. 控制台核验
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
print("逐数据集最佳传统baseline核验")

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
print("输出完成")
print(f"CSV：{CSV_OUT}")
print(f"Excel：{XLSX_OUT}")
print("未重新训练或重新计算任何模型")
