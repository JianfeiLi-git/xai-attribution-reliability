# -*- coding: utf-8 -*-
"""
xai_cell_G_V8

Goals:
1. Create the main figures of the paper from the current locked results Figure 1–3;
2. Figure 1: Feature-level attribution consistency and perturbation-based behavioral relevance;
3. Figure 2: Temporal attribution and recency validation;
4. Figure 3: Attribution sanity and perturbation robustness;

Version dependencies (read only the current valid versions):
- A_V9: parameter randomization, temporal occlusion
- B_V6: IG vs grouped KernelSHAP, paired behavioral relevance
- D_V1: small-noise robustness
- E_V2: label-randomization sanity
- F_V1: temporal IG vs KernelSHAP, block mass

Changes in this version:
- Figure 1–3 All composite figure panel labels changed from uppercase (A)/(B)/(C) to lowercase (a)/(b)/(c);
- EPS vector format added in addition to the original PNG / JPEG / TIFF output;
- all other data, values, figure structure, axes, legends and layout logic are unchanged from G_V7;
- The formal figure titles are still given by the captions in the paper.

Principles:
- No folder scanning;
- Use fixed paths only;
- When a file is missing, print its exact path;
- Text in figures is in English;
- console messages were originally in Chinese;
- XAI is not recomputed; only the current locked results are organized and plotted;
- The only valid version of this pipeline is G_V8.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================
# 0. Global configuration
# =========================

OUTPUT_DIR = Path("./0924/xai_cell_G_V8")
print(f"Output folder: {OUTPUT_DIR.as_posix()}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = ["Medical-1", "Medical-2", "Power", "US-Med", "US-Pow"]
MODEL_ORDER = ["GRU", "BiLSTM", "LSTM"]

PATHS = {
    "A_SANITY": Path("./0924/xai_cell_A_V9/A1_parameter_randomization_sanity.csv"),
    "A_OCCLUSION": Path("./0924/xai_cell_A_V9/A3_temporal_occlusion.csv"),
    "B_COMPARE": Path("./0924/xai_cell_B_V6/B2_ig_vs_kernelshap.csv"),
    "B_RELEVANCE": Path("./0924/xai_cell_B_V6/B3_paired_faithfulness_summary.csv"),
    "D_NOISE": Path("./0924/xai_cell_D_V1/D2_input_noise_robustness_summary.csv"),
    "E_LABEL": Path("./0924/xai_cell_E_V2/E1_label_randomization_sanity.csv"),
    "F_BLOCK": Path("./0924/xai_cell_F_V1/F4_temporal_block_mass.csv"),
    "F_SUMMARY": Path("./0924/xai_cell_F_V1/F5_temporal_method_summary.csv"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =========================
# 1. Utility functions
# =========================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Fixed result file missing: {path.as_posix()}")
    return path


def read_csv_fixed(key):
    path = require_file(PATHS[key])
    df = pd.read_csv(path)
    print(f"{key}: Read successfully | rows={len(df)} | Path={path.as_posix()}")
    return df


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} Missing field: {missing}")


def ordered_combo_df(df):
    out = df.copy()
    out["Dataset"] = pd.Categorical(out["Dataset"], categories=DATASET_ORDER, ordered=True)
    out["Model"] = pd.Categorical(out["Model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def save_figure(fig, stem):
    formats = [
        ("png", {}),
        ("jpeg", {"quality": 95}),
        ("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}),
        ("eps", {}),
    ]

    for ext, kwargs in formats:
        path = OUTPUT_DIR / f"{stem}.{ext}"
        if ext == "jpeg":
            fig.savefig(path, bbox_inches="tight", dpi=600, format="jpeg", pil_kwargs={"quality": 95})
        elif ext == "tiff":
            fig.savefig(path, bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        elif ext == "eps":
            fig.savefig(path, bbox_inches="tight", format="eps")
        else:
            fig.savefig(path, bbox_inches="tight", dpi=600, format="png")
        print(f"Saved: {path.as_posix()}")

    plt.close(fig)


def draw_heatmap(ax, matrix, row_labels, col_labels, title, vmin=None, vmax=None, fmt=".2f", cmap=None, text_threshold=0.52):
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, loc="left", fontweight="bold")

    norm = im.norm
    cmap_obj = im.get_cmap()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            rgba = cmap_obj(norm(val))
            r, g, b = rgba[:3]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            txt_color = "white" if luminance < text_threshold else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, fontweight="bold", color=txt_color)
    return im


def make_combo_x_map():
    rows = []
    x = 0
    for d_i, dataset in enumerate(DATASET_ORDER):
        if d_i > 0:
            x += 0.7
        for model in MODEL_ORDER:
            rows.append({"Dataset": dataset, "Model": model, "X": x})
            x += 1
    return pd.DataFrame(rows)


def model_short(model):
    return {"GRU": "GRU", "BiLSTM": "BiLSTM", "LSTM": "LSTM"}[model]


# =========================
# 2. Read the current locked results
# =========================

print("-" * 20)
print("Step 1: read the fixed results")

a_sanity = read_csv_fixed("A_SANITY")
a_occ = read_csv_fixed("A_OCCLUSION")
b_cmp = read_csv_fixed("B_COMPARE")
b_rel = read_csv_fixed("B_RELEVANCE")
d_noise = read_csv_fixed("D_NOISE")
e_label = read_csv_fixed("E_LABEL")
f_block = read_csv_fixed("F_BLOCK")
f_sum = read_csv_fixed("F_SUMMARY")

# =========================
# 3. Figure 1
# =========================

print("-" * 20)
print("Step 2: create Figure 1")

require_columns(b_cmp, ["Dataset", "Model", "IG_vs_KernelSHAP_rho", "p_value"], "B2")
require_columns(
    b_rel,
    ["Dataset", "Model", "K", "Extra_Delta_OriginalPredProb_Mean",
     "Extra_Delta_OriginalPredProb_Q025", "Extra_Delta_OriginalPredProb_Q975"],
    "B3"
)

cmp1 = ordered_combo_df(b_cmp)
rel1 = ordered_combo_df(b_rel)

rho_matrix = (
    cmp1.pivot(index="Dataset", columns="Model", values="IG_vs_KernelSHAP_rho")
    .reindex(index=DATASET_ORDER, columns=MODEL_ORDER)
    .astype(float).values
)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.0, 2.2]})

im = draw_heatmap(
    axes[0], rho_matrix, DATASET_ORDER, MODEL_ORDER,
    "(a) IG–KernelSHAP feature-ranking agreement",
    vmin=0.65, vmax=1.00
)

cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label("Spearman rho")

combo_map = make_combo_x_map()
x_lookup = {(r["Dataset"], r["Model"]): r["X"] for _, r in combo_map.iterrows()}
markers = {1: "o", 3: "s", 5: "^"}

for k in [1, 3, 5]:
    g = rel1[rel1["K"] == k].copy()
    xs = [x_lookup[(str(r["Dataset"]), str(r["Model"]))] for _, r in g.iterrows()]
    axes[1].scatter(xs, g["Extra_Delta_OriginalPredProb_Mean"], marker=markers[k], s=38, label=f"k={k}", zorder=3)

axes[1].axhline(0, linewidth=0.8, linestyle="--")
axes[1].set_xticks(combo_map["X"])
axes[1].set_xticklabels([model_short(x) for x in combo_map["Model"]], rotation=45, ha="right")

ymin, ymax = axes[1].get_ylim()
text_y = ymin - (ymax - ymin) * 0.16

for dataset in DATASET_ORDER:
    g = combo_map[combo_map["Dataset"] == dataset]
    axes[1].text(g["X"].mean(), text_y, dataset, ha="center", va="top", fontsize=8, fontweight="bold", clip_on=False)

axes[1].set_ylabel("Extra probability drop: Top-k minus Random-k")
axes[1].set_title("(b) Paired perturbation-based behavioral relevance", loc="left", fontweight="bold")
axes[1].legend(frameon=False, ncol=3, loc="upper right")
axes[1].grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.subplots_adjust(bottom=0.23)
save_figure(fig, "Figure1_feature_consistency_behavioral_relevance")

cmp1.to_csv(OUTPUT_DIR / "G1_feature_consistency_source.csv", index=False)
rel1.to_csv(OUTPUT_DIR / "G1_behavioral_relevance_source.csv", index=False)

print(
    "Figure 1 Summary: "
    f"rho Range={b_cmp['IG_vs_KernelSHAP_rho'].min():.6f}-"
    f"{b_cmp['IG_vs_KernelSHAP_rho'].max():.6f}; "
    f"positively correlated and p<0.05="
    f"{int(((b_cmp['IG_vs_KernelSHAP_rho'] > 0) & (b_cmp['p_value'] < 0.05)).sum())}/"
    f"{len(b_cmp)}"
)

print(
    "Behavioral relevance Summary: "
    f"mean contrast > 0={int((b_rel['Extra_Delta_OriginalPredProb_Mean'] > 0).sum())}/{len(b_rel)}"
)

# =========================
# 4. Figure 2
# =========================

print("-" * 20)
print("Step 3: create Figure 2")

require_columns(
    f_sum,
    ["Dataset", "Model", "Timestep_Spearman", "Timestep_p_value", "IG_Last10_Mass",
     "KernelSHAP_Last10_Mass", "IG_Peak_Block", "KernelSHAP_Peak_Block", "Same_Peak_Block"],
    "F5"
)
require_columns(
    f_block,
    ["Dataset", "Model", "Block", "Timestep_Start", "Timestep_End", "IG_Mass", "KernelSHAP_Mass"],
    "F4"
)
require_columns(
    a_occ,
    ["Dataset", "Model", "Block", "Timestep_Start", "Timestep_End", "Delta_OriginalPredProb"],
    "A3"
)

fs = ordered_combo_df(f_sum)
fb = f_block.copy()
ao = a_occ.copy()

temp_rho_matrix = (
    fs.pivot(index="Dataset", columns="Model", values="Timestep_Spearman")
    .reindex(index=DATASET_ORDER, columns=MODEL_ORDER)
    .astype(float).values
)

block_mean = (
    fb.groupby("Block", as_index=False)
    .agg(
        IG_Mean=("IG_Mass", "mean"),
        IG_SD=("IG_Mass", "std"),
        KernelSHAP_Mean=("KernelSHAP_Mass", "mean"),
        KernelSHAP_SD=("KernelSHAP_Mass", "std"),
    )
    .sort_values("Block")
)

occ_mean = (
    ao.groupby("Block", as_index=False)
    .agg(
        Probability_Drop_Mean=("Delta_OriginalPredProb", "mean"),
        Probability_Drop_SD=("Delta_OriginalPredProb", "std"),
    )
    .sort_values("Block")
)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))

im = draw_heatmap(
    axes[0], temp_rho_matrix, DATASET_ORDER, MODEL_ORDER,
    "(a) Timestep-level IG–KernelSHAP agreement",
    vmin=0.50, vmax=0.90
)

cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label("Spearman rho")

x = block_mean["Block"].to_numpy()

axes[1].errorbar(x, block_mean["IG_Mean"], yerr=block_mean["IG_SD"], marker="o", capsize=3, label="IG")
axes[1].errorbar(x, block_mean["KernelSHAP_Mean"], yerr=block_mean["KernelSHAP_SD"], marker="s", capsize=3, label="KernelSHAP")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["1–10", "11–20", "21–30", "31–40", "41–50", "51–60"])
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Attribution mass")
axes[1].set_title("(b) Temporal attribution mass by 10-day block", loc="left", fontweight="bold")
axes[1].legend(frameon=False)
axes[1].grid(axis="y", alpha=0.25)

axes[2].bar(occ_mean["Block"], occ_mean["Probability_Drop_Mean"], yerr=occ_mean["Probability_Drop_SD"], capsize=3)
axes[2].set_xticks(x)
axes[2].set_xticklabels(["1–10", "11–20", "21–30", "31–40", "41–50", "51–60"])
axes[2].set_xlabel("Days")
axes[2].set_ylabel("Original predicted-class probability drop")
axes[2].set_title("(c) Empirical temporal occlusion", loc="left", fontweight="bold")
axes[2].grid(axis="y", alpha=0.25)

fig.tight_layout()
save_figure(fig, "Figure2_temporal_recency_validation")

fs.to_csv(OUTPUT_DIR / "G2_temporal_summary_source.csv", index=False)
block_mean.to_csv(OUTPUT_DIR / "G2_block_mass_summary.csv", index=False)
occ_mean.to_csv(OUTPUT_DIR / "G2_occlusion_summary.csv", index=False)

print(
    "Figure 2 Summary: "
    f"60-timestep rho Range={f_sum['Timestep_Spearman'].min():.6f}-"
    f"{f_sum['Timestep_Spearman'].max():.6f}; "
    f"positively correlated and p<0.05="
    f"{int(((f_sum['Timestep_Spearman'] > 0) & (f_sum['Timestep_p_value'] < 0.05)).sum())}/"
    f"{len(f_sum)}"
)

print(f"peak block matches: {int(f_sum['Same_Peak_Block'].astype(bool).sum())}/{len(f_sum)}")
print(f"KernelSHAP Peak 51-60: {int((f_sum['KernelSHAP_Peak_Block'] == 6).sum())}/{len(f_sum)}")

# =========================
# 5. Figure 3
# =========================

print("-" * 20)
print("Step 4: create Figure 3")

require_columns(
    a_sanity,
    ["Dataset", "Model", "Stage", "Map_Spearman_vs_Original", "Feature_Spearman_vs_Original"],
    "A1"
)
require_columns(
    e_label,
    ["Dataset", "Model", "Map_Spearman", "Feature_Spearman", "Temporal_Spearman"],
    "E1"
)
require_columns(
    d_noise,
    ["Dataset", "Model", "Noise_Percent_SD", "Map_Spearman_Mean", "Feature_Spearman_Mean",
     "Temporal_Spearman_Mean", "Pred_Agreement_Mean"],
    "D2"
)

final_param = (
    a_sanity.groupby(["Dataset", "Model"], sort=False)
    .tail(1)
    .reset_index(drop=True)
)
final_param = ordered_combo_df(final_param)

param_rows = [f"{dataset} | {model}" for dataset in DATASET_ORDER for model in MODEL_ORDER]
param_map, param_feat = [], []

for dataset in DATASET_ORDER:
    for model in MODEL_ORDER:
        row = final_param[
            (final_param["Dataset"].astype(str) == dataset)
            & (final_param["Model"].astype(str) == model)
        ].iloc[0]
        param_map.append(float(row["Map_Spearman_vs_Original"]))
        param_feat.append(float(row["Feature_Spearman_vs_Original"]))

param_matrix = np.column_stack([param_map, param_feat])

label_plot = ordered_combo_df(e_label)
label_map, label_feat, label_time = [], [], []

for dataset in DATASET_ORDER:
    for model in MODEL_ORDER:
        row = label_plot[
            (label_plot["Dataset"].astype(str) == dataset)
            & (label_plot["Model"].astype(str) == model)
        ].iloc[0]
        label_map.append(float(row["Map_Spearman"]))
        label_feat.append(float(row["Feature_Spearman"]))
        label_time.append(float(row["Temporal_Spearman"]))

label_matrix = np.column_stack([label_map, label_feat, label_time])

noise_plot = (
    d_noise.groupby("Noise_Percent_SD", as_index=False)
    .agg(
        Map=("Map_Spearman_Mean", "mean"),
        Feature=("Feature_Spearman_Mean", "mean"),
        Temporal=("Temporal_Spearman_Mean", "mean"),
        PredictionAgreement=("Pred_Agreement_Mean", "mean"),
    )
    .sort_values("Noise_Percent_SD")
)

noise_matrix = noise_plot[["Map", "Feature", "Temporal", "PredictionAgreement"]].to_numpy(dtype=float)
noise_rows = [f"{x:.1f}%" for x in noise_plot["Noise_Percent_SD"]]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), gridspec_kw={"width_ratios": [1.0, 1.15, 1.05]})

im_a = draw_heatmap(
    axes[0], param_matrix, param_rows, ["Map", "Feature"],
    "(a) Final-stage cascading parameter randomization",
    vmin=-0.05, vmax=0.80, cmap="coolwarm"
)
cb_a = fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)
cb_a.set_label("Spearman rho vs original")

im_b = draw_heatmap(
    axes[1], label_matrix, param_rows, ["Map", "Feature", "Temporal"],
    "(b) Label-randomization attribution similarity",
    vmin=-0.70, vmax=1.00, cmap="coolwarm"
)
cb_b = fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
cb_b.set_label("Spearman rho vs original")

im_c = draw_heatmap(
    axes[2], noise_matrix, noise_rows, ["Map", "Feature", "Temporal", "Pred. agr."],
    "(c) Robustness to small input perturbations",
    vmin=0.975, vmax=1.000, fmt=".4f", cmap="viridis"
)
cb_c = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)
cb_c.set_label("Similarity / agreement")
axes[2].set_ylabel("Noise level (% of training feature SD)")

fig.tight_layout()
save_figure(fig, "Figure3_sanity_robustness")

final_param.to_csv(OUTPUT_DIR / "G3_parameter_randomization_final_stage.csv", index=False)
label_plot.to_csv(OUTPUT_DIR / "G3_label_randomization_source.csv", index=False)
noise_plot.to_csv(OUTPUT_DIR / "G3_noise_overall_summary.csv", index=False)

print(
    "Figure 3A Summary: "
    "final-stage feature rho Range="
    f"{final_param['Feature_Spearman_vs_Original'].min():.6f}-"
    f"{final_param['Feature_Spearman_vs_Original'].max():.6f}"
)

print(
    "Figure 3B Summary: "
    "label-randomization Mean rho | "
    f"Map={e_label['Map_Spearman'].mean():.6f}, "
    f"Feature={e_label['Feature_Spearman'].mean():.6f}, "
    f"Temporal={e_label['Temporal_Spearman'].mean():.6f}"
)

print("Figure 3C Cross-model mean:")
print(noise_plot.to_string(index=False))

# =========================
# 6. Final audit
# =========================

print("-" * 20)
print("Step 5: final audit")

expected = []

for stem in [
    "Figure1_feature_consistency_behavioral_relevance",
    "Figure2_temporal_recency_validation",
    "Figure3_sanity_robustness",
]:
    for ext in ["png", "jpeg", "tiff", "eps"]:
        expected.append(f"{stem}.{ext}")

missing = [name for name in expected if not (OUTPUT_DIR / name).exists()]

if missing:
    print(f"Figure file missing: {missing}")
    raise RuntimeError("Figure Output incomplete")

print("Figure 1–3 All created successfully")
print("Output format: PNG / JPEG / TIFF / EPS")
print("Error: 0")
print("-" * 20)
print("xai_cell_G_V8 Done")
print(f"Results folder: {OUTPUT_DIR.as_posix()}/")
