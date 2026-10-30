# -*- coding: utf-8 -*-
"""
xai_cell_H_V1

Goals:
1. Apply BH-FDR correction to the 15 feature-level IG vs grouped KernelSHAP Spearman tests from B_V6;
2. Apply BH-FDR correction to the 15 timestep-level IG vs temporal KernelSHAP Spearman tests from F_V1;
3. The two test families are corrected separately, not mixed;
4. Models, IG and KernelSHAP are not recomputed; only statistical post-processing of the locked original p-values is done.

Principles:
- Read only the current valid versions B_V6 and F_V1;
- No folder scanning;
- When a file is missing, report its fixed path;
- The console prints the output folder first;
- console messages were originally in Chinese;
- output fields are in English;
- The only valid version of this pipeline is H_V1.
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("./0924/xai_cell_H_V1")
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

B_PATH = Path("./0924/xai_cell_B_V6/B2_ig_vs_kernelshap.csv")
F_PATH = Path("./0924/xai_cell_F_V1/F5_temporal_method_summary.csv")

DATASET_ORDER = ["Medical-1", "Medical-2", "Power", "US-Med", "US-Pow"]
MODEL_ORDER = ["GRU", "BiLSTM", "LSTM"]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Fixed result file missing: {path.as_posix()}")
    return path


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} Missing field: {missing}")


def validate_family(df, p_col, name):
    if len(df) != 15:
        raise ValueError(f"{name} should have 15 rows, actual {len(df)} rows")
    if df[["Dataset", "Model"]].duplicated().any():
        dup = df.loc[
            df[["Dataset", "Model"]].duplicated(keep=False),
            ["Dataset", "Model"]
        ]
        raise ValueError(f"{name} duplicate combinations exist:\n{dup.to_string(index=False)}")
    p = pd.to_numeric(df[p_col], errors="coerce")
    if p.isna().any():
        raise ValueError(f"{name} {p_col} has missing or non-numeric values")
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"{name} {p_col} contains values outside the [0,1] range")


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
print("Step 1: read the fixed results")

b = pd.read_csv(require_file(B_PATH))
f = pd.read_csv(require_file(F_PATH))

print(f"B_V6 Read successfully: {len(b)} rows")
print(f"Path: {B_PATH.as_posix()}")
print(f"F_V1 Read successfully: {len(f)} rows")
print(f"Path: {F_PATH.as_posix()}")

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

validate_family(b, "p_value", "Feature-level Test family")
validate_family(f, "Timestep_p_value", "Temporal Test family")

print("-" * 20)
print("Step 2: Feature BH-FDR")

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
print(f"Feature significant after correction: {n_b}/15")
print(f"Maximum BH-FDR p-value: {b_out['BH_FDR_p'].max():.6g}")

print("-" * 20)
print("Step 3: Temporal BH-FDR")

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
print(f"Temporal significant after correction: {n_f}/15")
print(f"Maximum BH-FDR p-value: {f_out['BH_FDR_p'].max():.6g}")

print("-" * 20)
print("Step 4: summary")

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
print("Final conclusion")
print(f"Feature-level BH-FDR Significant: {n_b}/15")
print(f"Temporal BH-FDR Significant: {n_f}/15")

if n_b == 15 and n_f == 15:
    print("Both groups of Spearman tests remain significant after BH-FDR correction.")
else:
    print("Some tests are no longer significant after BH-FDR correction; please check them item by item in the table above.")

print("-" * 20)
print("xai_cell_H_V1 Done")
print(f"Results folder: {OUTPUT_DIR.as_posix()}/")
