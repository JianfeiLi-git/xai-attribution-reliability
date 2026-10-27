# -*- coding: utf-8 -*-
"""
xai_cell_H_V1

目标：
1. 对B_V6中15个feature-level IG vs grouped KernelSHAP Spearman检验做BH-FDR校正；
2. 对F_V1中15个timestep-level IG vs temporal KernelSHAP Spearman检验做BH-FDR校正；
3. 两个检验族分别校正，不混合；
4. 不重新计算模型、IG或KernelSHAP，只对已锁定原始p值做统计后处理。

原则：
- 只读取当前有效版本B_V6、F_V1；
- 不扫描目录；
- 缺少文件时明确报出固定路径；
- 控制台最先打印输出目录；
- 控制台说明使用中文；
- 输出字段使用英文；
- 本流程当前唯一有效版本为H_V1。
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("./0924/xai_cell_H_V1")
print(f"输出目录: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

B_PATH = Path("./0924/xai_cell_B_V6/B2_ig_vs_kernelshap.csv")
F_PATH = Path("./0924/xai_cell_F_V1/F5_temporal_method_summary.csv")

DATASET_ORDER = ["Medical-1", "Medical-2", "Power", "US-Med", "US-Pow"]
MODEL_ORDER = ["GRU", "BiLSTM", "LSTM"]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少固定结果文件: {path.as_posix()}")
    return path


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}缺少字段: {missing}")


def validate_family(df, p_col, name):
    if len(df) != 15:
        raise ValueError(f"{name}应为15行，实际为{len(df)}行")
    if df[["Dataset", "Model"]].duplicated().any():
        dup = df.loc[
            df[["Dataset", "Model"]].duplicated(keep=False),
            ["Dataset", "Model"]
        ]
        raise ValueError(f"{name}存在重复组合:\n{dup.to_string(index=False)}")
    p = pd.to_numeric(df[p_col], errors="coerce")
    if p.isna().any():
        raise ValueError(f"{name}的{p_col}存在缺失或非数值")
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"{name}的{p_col}存在超出[0,1]范围的值")


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    ranked_p = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    adjusted_ranked = ranked_p * m / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    reject = adjusted <= ALPHA
    return adjusted, reject


def ordered(df):
    out = df.copy()
    out["Dataset"] = pd.Categorical(
        out["Dataset"], categories=DATASET_ORDER, ordered=True
    )
    out["Model"] = pd.Categorical(
        out["Model"], categories=MODEL_ORDER, ordered=True
    )
    out = out.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    out["Dataset"] = out["Dataset"].astype(str)
    out["Model"] = out["Model"].astype(str)
    return out


print("-" * 20)
print("步骤1：读取固定结果")

b = pd.read_csv(require_file(B_PATH))
f = pd.read_csv(require_file(F_PATH))

print(f"B_V6读取成功: {len(b)}行")
print(f"路径: {B_PATH.as_posix()}")
print(f"F_V1读取成功: {len(f)}行")
print(f"路径: {F_PATH.as_posix()}")

require_columns(
    b,
    ["Dataset", "Model", "IG_vs_KernelSHAP_rho", "p_value"],
    "B_V6"
)
require_columns(
    f,
    ["Dataset", "Model", "Timestep_Spearman", "Timestep_p_value"],
    "F_V1"
)

validate_family(b, "p_value", "Feature-level检验族")
validate_family(f, "Timestep_p_value", "Temporal检验族")

print("-" * 20)
print("步骤2：Feature BH-FDR")

b_out = b[
    ["Dataset", "Model", "IG_vs_KernelSHAP_rho", "p_value"]
].copy()

b_out["BH_FDR_p"], b_out["Reject_BH_FDR_0.05"] = bh_fdr(
    b_out["p_value"].to_numpy()
)
b_out = ordered(b_out)

b_out.to_csv(
    OUTPUT_DIR / "H1_feature_spearman_bh_fdr.csv",
    index=False
)

print(
    b_out[
        [
            "Dataset",
            "Model",
            "IG_vs_KernelSHAP_rho",
            "p_value",
            "BH_FDR_p",
            "Reject_BH_FDR_0.05",
        ]
    ].to_string(index=False)
)

n_b = int(b_out["Reject_BH_FDR_0.05"].sum())
print(f"Feature校正后显著: {n_b}/15")
print(f"最大BH-FDR p值: {b_out['BH_FDR_p'].max():.6g}")

print("-" * 20)
print("步骤3：Temporal BH-FDR")

f_out = f[
    ["Dataset", "Model", "Timestep_Spearman", "Timestep_p_value"]
].copy()

f_out["BH_FDR_p"], f_out["Reject_BH_FDR_0.05"] = bh_fdr(
    f_out["Timestep_p_value"].to_numpy()
)
f_out = ordered(f_out)

f_out.to_csv(
    OUTPUT_DIR / "H2_temporal_spearman_bh_fdr.csv",
    index=False
)

print(
    f_out[
        [
            "Dataset",
            "Model",
            "Timestep_Spearman",
            "Timestep_p_value",
            "BH_FDR_p",
            "Reject_BH_FDR_0.05",
        ]
    ].to_string(index=False)
)

n_f = int(f_out["Reject_BH_FDR_0.05"].sum())
print(f"Temporal校正后显著: {n_f}/15")
print(f"最大BH-FDR p值: {f_out['BH_FDR_p'].max():.6g}")

print("-" * 20)
print("步骤4：汇总")

summary = pd.DataFrame([
    {
        "Family": "Feature-level IG vs grouped KernelSHAP",
        "N_Tests": 15,
        "Raw_p_lt_0.05": int((b_out["p_value"] < ALPHA).sum()),
        "BH_FDR_Significant": n_b,
        "Max_BH_FDR_p": float(b_out["BH_FDR_p"].max()),
    },
    {
        "Family": "Timestep-level IG vs temporal KernelSHAP",
        "N_Tests": 15,
        "Raw_p_lt_0.05": int((f_out["Timestep_p_value"] < ALPHA).sum()),
        "BH_FDR_Significant": n_f,
        "Max_BH_FDR_p": float(f_out["BH_FDR_p"].max()),
    },
])

summary.to_csv(
    OUTPUT_DIR / "H3_bh_fdr_summary.csv",
    index=False
)

print(summary.to_string(index=False))

print("-" * 20)
print("最终结论")
print(f"Feature-level BH-FDR显著: {n_b}/15")
print(f"Temporal BH-FDR显著: {n_f}/15")

if n_b == 15 and n_f == 15:
    print("两组Spearman检验经BH-FDR校正后仍全部显著。")
else:
    print("存在BH-FDR校正后不再显著的检验，请按上表逐项核对。")

print("-" * 20)
print("xai_cell_H_V1 完成")
print(f"结果目录: {OUTPUT_DIR.as_posix()}/")
