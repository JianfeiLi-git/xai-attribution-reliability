"""
stage6_cell_M_data_figures_V2

Aim: read frozen results only and draw figures, without recomputing any values.
I. Redraw the original data figures of the paper (U.S. part replaced by the new stage6 results, Chinese part still from the original results):
   - Figure 2: feature-attribution agreement and perturbation test (Figure1 of the original xai_cell_G_V8)
   - Figure 3: temporal attribution and recency validation (original Figure2)
   - Figure 4: randomization tests and input-noise robustness (original Figure3)
   Layout, colors and fonts are the same as xai_cell_G_V8; color-scale ranges are widened automatically when data exceed the original range;
   cells of label randomization whose correlation cannot be computed (US-Med GRU: the IG of the random-label model is all zero) are marked "n/a".
   Figure 1 (study framework diagram) is not a data figure and is not generated in this cell.
II. New figures for the two new analyses (data from stage6_cell_L_data_crossmarket_V5):
   - Figure 5: explanation agreement and perturbation extra drop (only k=3 and k=5 in the main text)
   - Figure 6: input-noise stability and cross-seed stability (main analysis keeps all seeds; combinations containing seeds consistent with single-class prediction
     are marked with † after the label, and the sensitivity mean without these seeds is drawn as an open circle)
   - Figure 7: predictive performance and per-sample prediction agreement (combinations containing seeds consistent with single-class prediction use open markers; individual such seeds use x)
   - Supplementary Figure S1: k=1 results under the original definition and sensitivity results after removing repetitions where "Random-k is identical to Top-k"
All figures are output in PNG, JPEG, TIFF and EPS at 600 dpi; text in the figures is in English.
III. Data checks:
   - The final stage of parameter randomization is determined by stage name (stage names in the original code are Randomize_FC_RNN_L{i}_to_L{n-1},
     the final stage being Randomize_FC_RNN_L0_to_L{n-1}, fully randomized from layer 0), independent of file row order;
     each combination must have an Original stage and exactly one final stage, and n-1 of the final stage must equal the largest layer number appearing for that combination;
   - parameter randomization (combination x stage) and label randomization (combination) are checked for duplicates, missing and extra records;
   - the sensitivity panel of Supplementary Figure S1 recomputes the 2.5%–97.5% range from the per-repetition details, so the two panels show symmetric content.
"""

# ------------------------------ Settings (editable) ------------------------------
PROJECT_ROOT = "/home/i/2026_0712/stock_i"      # project root containing the original Chinese results
CN_RESULT_DIRS = {"A": "xai_cell_A_V9", "B": "xai_cell_B_V6", "D": "xai_cell_D_V1",
                  "E": "xai_cell_E_V2", "F": "xai_cell_F_V1"}
CN_SEARCH_SUBDIRS = ["", "0924"]                 # checked in order: project root, project root/0924, then the current directory
US_DIRS = {"E": "stage6/stage6_cell_E_data_attribution_V2",
           "F": "stage6/stage6_cell_F_data_kernelshap_V4",
           "G": "stage6/stage6_cell_G_data_temporal_V2",
           "H": "stage6/stage6_cell_H_data_robustness_V4",
           "I": "stage6/stage6_cell_I_data_labelrandom_V2"}
L_DIR = "stage6/stage6_cell_L_data_crossmarket_V5"
OUTPUT_ROOT = "stage6/stage6_cell_M_data_figures_V2"

CN_DATASETS = ["Medical-1", "Medical-2", "Power"]
US_DATASETS = ["US-Med", "US-Pow"]
DATASET_ORDER = CN_DATASETS + US_DATASETS
MODEL_ORDER = ["GRU", "BiLSTM", "LSTM"]
MAIN_K = [3, 5]                 # k used for Figure 5 in the main text
PRIMARY_NOISE = 2.0
FIG_FORMATS = ["png", "jpeg", "tiff", "eps"]

# ------------------------------ Imports ------------------------------
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(PROJECT_ROOT)
OUTPUT_DIR = Path(OUTPUT_ROOT)
ALL_COMBOS = [(d, m) for d in DATASET_ORDER for m in MODEL_ORDER]


def rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return "./" + p.as_posix()


def elapsed():
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def find_cn_dir(name):
    for sub in CN_SEARCH_SUBDIRS:
        cand = ROOT / sub / name if sub else ROOT / name
        if cand.is_dir():
            return cand
    if Path(name).is_dir():
        return Path(name)
    return None


CN_DIR = {k: find_cn_dir(v) for k, v in CN_RESULT_DIRS.items()}

# Rule: the first two lines print the input and output locations
_in = [rel(CN_DIR[k]) + "/" if CN_DIR[k] else CN_RESULT_DIRS[k] + " (not found)" for k in CN_RESULT_DIRS]
_in += ["./" + v + "/" for v in US_DIRS.values()] + ["./" + L_DIR + "/"]
print("Input directories: " + ", ".join(_in))
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")

INPUT_FILES = {
    "cn_param": (CN_DIR["A"], "A1_parameter_randomization_sanity.csv"),
    "cn_occl": (CN_DIR["A"], "A3_temporal_occlusion.csv"),
    "cn_agree": (CN_DIR["B"], "B2_ig_vs_kernelshap.csv"),
    "cn_faith": (CN_DIR["B"], "B3_paired_faithfulness_summary.csv"),
    "cn_noise": (CN_DIR["D"], "D2_input_noise_robustness_summary.csv"),
    "cn_label": (CN_DIR["E"], "E1_label_randomization_sanity.csv"),
    "cn_block": (CN_DIR["F"], "F4_temporal_block_mass.csv"),
    "cn_tsum": (CN_DIR["F"], "F5_temporal_method_summary.csv"),
    "us_param": (Path(US_DIRS["E"]), "A1_parameter_randomization_sanity.csv"),
    "us_occl": (Path(US_DIRS["E"]), "A3_temporal_occlusion.csv"),
    "us_agree": (Path(US_DIRS["F"]), "B2_ig_vs_kernelshap.csv"),
    "us_faith": (Path(US_DIRS["F"]), "B3_paired_faithfulness_summary.csv"),
    "us_noise": (Path(US_DIRS["H"]), "D2_input_noise_robustness_summary.csv"),
    "us_label": (Path(US_DIRS["I"]), "E1_label_randomization_sanity.csv"),
    "us_block": (Path(US_DIRS["G"]), "F4_temporal_block_mass.csv"),
    "us_tsum": (Path(US_DIRS["G"]), "F5_temporal_method_summary.csv"),
    "cn_fdet": (CN_DIR["B"], "B3_paired_faithfulness_detail.csv"),
    "us_fdet": (Path(US_DIRS["F"]), "B3_paired_faithfulness_detail.csv"),
    "L3": (Path(L_DIR), "L3_agreement_faithfulness_by_combo.csv"),
    "L3a": (Path(L_DIR), "L3_agreement_faithfulness_association.csv"),
    "L4": (Path(L_DIR), "L4_stability_gap_by_combo.csv"),
    "L4x": (Path(L_DIR), "L4_stability_gap_by_combo_exclude_flagged.csv"),
    "L5": (Path(L_DIR), "L5_seed_performance_vs_agreement.csv"),
}
_missing = [f"{k}: {('directory not found ' + f) if d is None else rel(d / f)}" for k, (d, f) in INPUT_FILES.items()
            if d is None or not (d / f).exists()]
if _missing:
    print("The following input files do not exist:")
    for _m in _missing:
        print("  " + _m)
    raise FileNotFoundError(f"{len(_missing)} input files are missing")
print(f"Input file check passed ({len(INPUT_FILES)} files)")
for _k, (_d, _f) in INPUT_FILES.items():
    print(f"  {_k}: {rel(_d / _f)}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = {k: pd.read_csv(d / f) for k, (d, f) in INPUT_FILES.items()}

# Same layout as xai_cell_G_V8
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 600, "axes.spines.top": False, "axes.spines.right": False,
})
MARKET_COLOR = {"China": "tab:blue", "US": "tab:orange"}
MODEL_MARKER = {"GRU": "o", "BiLSTM": "s", "LSTM": "^"}
SAVED = []


def combine(cn_key, us_key, name, required, combo_keys=None):
    # Chinese rows from the original results, U.S. rows from the stage6 results; check fields, duplicates, missing and extra
    cn, us = RAW[cn_key], RAW[us_key]
    for nm, df in [(cn_key, cn), (us_key, us)]:
        lack = [c for c in required if c not in df.columns]
        if lack:
            raise KeyError(f"{nm} is missing fields: {lack}")
    out = pd.concat([cn[cn["Dataset"].isin(CN_DATASETS)], us[us["Dataset"].isin(US_DATASETS)]], ignore_index=True)
    if combo_keys is not None:
        keys, expected = combo_keys
        have = out.groupby(keys).size()
        if (have > 1).any():
            raise ValueError(f"{name} has duplicates: {have[have > 1].index.tolist()}")
        hs = set(map(tuple, out[keys].values.tolist()))
        lack = [e for e in expected if e not in hs]
        extra = sorted(hs - set(map(tuple, expected)), key=str)
        if lack or extra:
            raise ValueError(f"{name} missing {lack}, extra {extra}")
    else:
        hs = set(zip(out["Dataset"], out["Model"]))
        lack = [c for c in ALL_COMBOS if c not in hs]
        extra = sorted(hs - set(ALL_COMBOS), key=str)
        if lack or extra:
            raise ValueError(f"{name} missing combinations {lack}, extra combinations {extra}")
    print(f"  {name}: China {int(out['Dataset'].isin(CN_DATASETS).sum())} rows + U.S. {int(out['Dataset'].isin(US_DATASETS).sum())} rows, check passed")
    return out


def ordered(df):
    out = df.copy()
    out["Dataset"] = pd.Categorical(out["Dataset"], categories=DATASET_ORDER, ordered=True)
    out["Model"] = pd.Categorical(out["Model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def widen(vmin, vmax, values, step=0.05):
    # When data exceed the original color-scale range, widen it to a multiple of 0.05 and print
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return vmin, vmax
    lo = vmin if v.min() >= vmin else round(np.floor(v.min() / step) * step, 4)
    hi = vmax if v.max() <= vmax else round(np.ceil(v.max() / step) * step, 4)
    if (lo, hi) != (vmin, vmax):
        print(f"    Color scale widened from [{vmin}, {vmax}] to [{lo:.2f}, {hi:.2f}] (data range {v.min():.4f}–{v.max():.4f})")
    return lo, hi


def save_figure(fig, stem):
    for ext in FIG_FORMATS:
        path = OUTPUT_DIR / f"{stem}.{ext}"
        if ext == "jpeg":
            fig.savefig(path, bbox_inches="tight", dpi=600, format="jpeg", pil_kwargs={"quality": 95})
        elif ext == "tiff":
            fig.savefig(path, bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        elif ext == "eps":
            fig.savefig(path, bbox_inches="tight", format="eps")
        else:
            fig.savefig(path, bbox_inches="tight", dpi=600, format="png")
        SAVED.append(path)
    plt.close(fig)
    print(f"  Saved: ./{OUTPUT_DIR.as_posix()}/{stem}." + "/.".join(FIG_FORMATS))


def draw_heatmap(ax, matrix, row_labels, col_labels, title, vmin=None, vmax=None, fmt=".2f", cmap=None, text_threshold=0.52):
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, loc="left", fontweight="bold")
    norm, cmap_obj = im.norm, im.get_cmap()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, fontweight="bold", color="dimgray")
                continue
            r, g, b = cmap_obj(norm(val))[:3]
            txt_color = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < text_threshold else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, fontweight="bold", color=txt_color)
    return im


def make_combo_x_map():
    rows, x = [], 0
    for d_i, dataset in enumerate(DATASET_ORDER):
        if d_i > 0:
            x += 0.7
        for model in MODEL_ORDER:
            rows.append({"Dataset": dataset, "Model": model, "X": x})
            x += 1
    return pd.DataFrame(rows)


def criterion_handles():
    return [Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markeredgecolor="black", markersize=6,
                   label="Filled: 2.5% Monte Carlo\nquantile > 0"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=6,
                   label="Open: 2.5% Monte Carlo\nquantile ≤ 0"),
            Line2D([0], [0], color="grey", linewidth=0.8, label="Bars: 2.5%–97.5%\nMonte Carlo range")]


def market_handles():
    hs = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor=c, markersize=6, label=k)
          for k, c in MARKET_COLOR.items()]
    hs += [Line2D([0], [0], marker=mk, color="w", markerfacecolor="white", markeredgecolor="black", markersize=6, label=k)
           for k, mk in MODEL_MARKER.items()]
    return hs


# ============================== Step 1: read and check ==============================
print("\n" + "=" * 100)
print(f"Step 1: read and check, total {elapsed()}")
print("=" * 100)
agree = combine("cn_agree", "us_agree", "IG vs KernelSHAP feature agreement", ["Dataset", "Model", "IG_vs_KernelSHAP_rho", "p_value"])
faith = combine("cn_faith", "us_faith", "Top-k perturbation test",
                ["Dataset", "Model", "K", "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_Q025",
                 "Extra_Delta_OriginalPredProb_Q975"],
                (["Dataset", "Model", "K"], [(d, m, k) for d, m in ALL_COMBOS for k in [1, 3, 5]]))
tsum = combine("cn_tsum", "us_tsum", "temporal agreement summary", ["Dataset", "Model", "Timestep_Spearman", "Timestep_p_value",
                                                  "KernelSHAP_Peak_Block", "Same_Peak_Block"])
block = combine("cn_block", "us_block", "10-day block attribution mass", ["Dataset", "Model", "Block", "IG_Mass", "KernelSHAP_Mass"],
                (["Dataset", "Model", "Block"], [(d, m, b) for d, m in ALL_COMBOS for b in range(1, 7)]))
occl = combine("cn_occl", "us_occl", "temporal occlusion", ["Dataset", "Model", "Block", "Delta_OriginalPredProb"],
               (["Dataset", "Model", "Block"], [(d, m, b) for d, m in ALL_COMBOS for b in range(1, 7)]))
param = combine("cn_param", "us_param", "parameter randomization", ["Dataset", "Model", "Stage", "Map_Spearman_vs_Original",
                                                  "Feature_Spearman_vs_Original"])
# Parameter randomization: (combination x stage) unique; final stage determined by stage name
import re
_dup = param.duplicated(["Dataset", "Model", "Stage"]).sum()
if _dup:
    raise ValueError(f"Parameter randomization has {_dup} duplicated rows of (combination x stage)")
_final_rows, _stage_log = [], []
for (d, m), g in param.groupby(["Dataset", "Model"], sort=False):
    stages = g["Stage"].astype(str).tolist()
    if "Original" not in stages:
        raise ValueError(f"Parameter randomization {d} {m} is missing the Original stage")
    layers = [int(x) for st_ in stages for x in re.findall(r"_L(\d+)", st_)]
    finals = [st_ for st_ in stages if re.fullmatch(r"Randomize_FC_RNN_L0_to_L\d+", st_)]
    if len(finals) != 1:
        raise ValueError(f"Parameter randomization {d} {m} should have exactly 1 final stage, found {finals}")
    top = int(re.findall(r"_to_L(\d+)$", finals[0])[0])
    if not layers or top != max(layers):
        raise ValueError(f"Parameter randomization {d} {m}: final stage {finals[0]} does not match the largest layer number {max(layers) if layers else 'none'}")
    orig = g[g["Stage"] == "Original"].iloc[0]
    if abs(float(orig["Map_Spearman_vs_Original"]) - 1) > 1e-6 or abs(float(orig["Feature_Spearman_vs_Original"]) - 1) > 1e-6:
        raise ValueError(f"Parameter randomization {d} {m}: correlations of the Original stage are not 1")
    _final_rows.append(g[g["Stage"] == finals[0]].iloc[0])
    _stage_log.append((d, m, len(stages), finals[0], str(g["Stage"].iloc[-1]) == finals[0]))
final_param_df = pd.DataFrame(_final_rows).reset_index(drop=True)
_hs = set(zip(final_param_df["Dataset"], final_param_df["Model"]))
if _hs != set(ALL_COMBOS) or len(final_param_df) != len(ALL_COMBOS):
    raise ValueError("Combinations of the final parameter-randomization stage are incomplete")
print(f"  Final parameter-randomization stage check passed: each of the 15 combinations has Original and a unique final stage; "
      f"final stage is exactly the last row of that combination in the file for {sum(x[4] for x in _stage_log)}/15")
for d, m, n, fs_, last in _stage_log:
    print(f"    {d} {m}: {n} stages, final stage {fs_}" + ("" if last else " (not the last row in the file; taken by stage name)"))
label = combine("cn_label", "us_label", "label randomization", ["Dataset", "Model", "Map_Spearman", "Feature_Spearman", "Temporal_Spearman"],
                (["Dataset", "Model"], ALL_COMBOS))
# Perturbation details: used for the 2.5%–97.5% range in the sensitivity panel of Supplementary Figure S1
_fd = []
for _k, _ds in [("cn_fdet", CN_DATASETS), ("us_fdet", US_DATASETS)]:
    _x = RAW[_k]
    _lack = [c for c in ["Dataset", "Model", "K", "Repeat", "TopK_Features", "RandomK_Features", "Extra_Delta_OriginalPredProb"] if c not in _x.columns]
    if _lack:
        raise KeyError(f"{_k} is missing fields: {_lack}")
    _fd.append(_x[_x["Dataset"].isin(_ds)])
fdet = pd.concat(_fd, ignore_index=True)
if fdet.duplicated(["Dataset", "Model", "K", "Repeat"]).any():
    raise ValueError("Perturbation details contain duplicated (combination x k x repetition)")
fdet1 = fdet[fdet["K"] == 1].copy()
fdet1["Identical_Set"] = [set(str(a).split("|")) == set(str(b).split("|")) for a, b in zip(fdet1["TopK_Features"], fdet1["RandomK_Features"])]
s1_ni = (fdet1[~fdet1["Identical_Set"]].groupby(["Dataset", "Model"])["Extra_Delta_OriginalPredProb"]
         .agg(NI_Mean="mean", NI_Q025=lambda v: v.quantile(0.025), NI_Q975=lambda v: v.quantile(0.975), NI_N="size").reset_index())
if set(zip(s1_ni["Dataset"], s1_ni["Model"])) != set(ALL_COMBOS):
    raise ValueError("k=1 sensitivity detail combinations are incomplete")
print(f"  k=1 perturbation details: {len(fdet1)} rows, Random-1 identical to Top-1 {int(fdet1['Identical_Set'].sum())} times; "
      f"sensitivity ranges recomputed for {len(s1_ni)} combinations")
noise = combine("cn_noise", "us_noise", "input noise", ["Dataset", "Model", "Noise_Percent_SD", "Map_Spearman_Mean",
                                               "Feature_Spearman_Mean", "Temporal_Spearman_Mean", "Pred_Agreement_Mean"],
                (["Dataset", "Model", "Noise_Percent_SD"], [(d, m, lv) for d, m in ALL_COMBOS for lv in [0.5, 1.0, 2.0]]))
for _k in ["L3", "L3a", "L4", "L4x", "L5"]:
    print(f"  {_k}: {len(RAW[_k])} rows")

# ============================== Step 2: Figure 2 ==============================
print("\n" + "=" * 100)
print(f"Step 2: Figure 2 (feature-attribution agreement and perturbation test), total {elapsed()}")
print("=" * 100)
cmp1, rel1 = ordered(agree), ordered(faith)
rho_matrix = cmp1.pivot(index="Dataset", columns="Model", values="IG_vs_KernelSHAP_rho").reindex(
    index=DATASET_ORDER, columns=MODEL_ORDER).astype(float).values
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.0, 2.2]})
vmin, vmax = widen(0.65, 1.00, rho_matrix)
im = draw_heatmap(axes[0], rho_matrix, DATASET_ORDER, MODEL_ORDER, "(a) IG–KernelSHAP feature-ranking agreement", vmin=vmin, vmax=vmax)
cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label("Spearman rho")
combo_map = make_combo_x_map()
x_lookup = {(r["Dataset"], r["Model"]): r["X"] for _, r in combo_map.iterrows()}
for k, mk in {1: "o", 3: "s", 5: "^"}.items():
    g = rel1[rel1["K"] == k]
    xs = [x_lookup[(str(r["Dataset"]), str(r["Model"]))] for _, r in g.iterrows()]
    axes[1].scatter(xs, g["Extra_Delta_OriginalPredProb_Mean"], marker=mk, s=38, label=f"k={k}", zorder=3)
axes[1].axhline(0, linewidth=0.8, linestyle="--")
axes[1].set_xticks(combo_map["X"])
axes[1].set_xticklabels(combo_map["Model"], rotation=45, ha="right")
ymin, ymax = axes[1].get_ylim()
for dataset in DATASET_ORDER:
    g = combo_map[combo_map["Dataset"] == dataset]
    axes[1].text(g["X"].mean(), ymin - (ymax - ymin) * 0.16, dataset, ha="center", va="top", fontsize=8,
                 fontweight="bold", clip_on=False)
axes[1].set_ylabel("Extra probability drop: Top-k minus Random-k")
axes[1].set_title("(b) Paired perturbation-based behavioral relevance", loc="left", fontweight="bold")
axes[1].legend(frameon=False, ncol=3, loc="upper right")
axes[1].grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.subplots_adjust(bottom=0.23)
save_figure(fig, "Figure2_feature_consistency_behavioral_relevance")
cmp1.to_csv(OUTPUT_DIR / "M2_feature_consistency_source.csv", index=False)
rel1.to_csv(OUTPUT_DIR / "M2_behavioral_relevance_source.csv", index=False)
print(f"  Figure 2 data: feature ρ range {agree['IG_vs_KernelSHAP_rho'].min():.4f}–{agree['IG_vs_KernelSHAP_rho'].max():.4f}; "
      f"mean extra drop > 0: {int((faith['Extra_Delta_OriginalPredProb_Mean'] > 0).sum())}/{len(faith)}; "
      f"2.5th percentile > 0: {int((faith['Extra_Delta_OriginalPredProb_Q025'] > 0).sum())}/{len(faith)}")
_neg = faith[faith["Extra_Delta_OriginalPredProb_Mean"] <= 0]
for _, r in _neg.iterrows():
    print(f"    Mean extra drop <= 0: {r['Dataset']} {r['Model']} k={r['K']}, {r['Extra_Delta_OriginalPredProb_Mean']:+.4f}")

# ============================== Step 3: Figure 3 ==============================
print("\n" + "=" * 100)
print(f"Step 3: Figure 3 (temporal attribution and recency validation), total {elapsed()}")
print("=" * 100)
fs = ordered(tsum)
temp_matrix = fs.pivot(index="Dataset", columns="Model", values="Timestep_Spearman").reindex(
    index=DATASET_ORDER, columns=MODEL_ORDER).astype(float).values
block_mean = block.groupby("Block", as_index=False).agg(IG_Mean=("IG_Mass", "mean"), IG_SD=("IG_Mass", "std"),
                                                        KernelSHAP_Mean=("KernelSHAP_Mass", "mean"),
                                                        KernelSHAP_SD=("KernelSHAP_Mass", "std")).sort_values("Block")
occ_mean = occl.groupby("Block", as_index=False).agg(Probability_Drop_Mean=("Delta_OriginalPredProb", "mean"),
                                                     Probability_Drop_SD=("Delta_OriginalPredProb", "std")).sort_values("Block")
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
vmin, vmax = widen(0.50, 0.90, temp_matrix)
im = draw_heatmap(axes[0], temp_matrix, DATASET_ORDER, MODEL_ORDER, "(a) Timestep-level IG–KernelSHAP agreement", vmin=vmin, vmax=vmax)
cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label("Spearman rho")
x = block_mean["Block"].to_numpy()
blabels = ["1–10", "11–20", "21–30", "31–40", "41–50", "51–60"]
axes[1].errorbar(x, block_mean["IG_Mean"], yerr=block_mean["IG_SD"], marker="o", capsize=3, label="IG")
axes[1].errorbar(x, block_mean["KernelSHAP_Mean"], yerr=block_mean["KernelSHAP_SD"], marker="s", capsize=3, label="KernelSHAP")
axes[1].set_xticks(x)
axes[1].set_xticklabels(blabels)
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Attribution mass")
axes[1].set_title("(b) Temporal attribution mass by 10-day block", loc="left", fontweight="bold")
axes[1].legend(frameon=False)
axes[1].grid(axis="y", alpha=0.25)
axes[2].bar(occ_mean["Block"], occ_mean["Probability_Drop_Mean"], yerr=occ_mean["Probability_Drop_SD"], capsize=3)
axes[2].set_xticks(x)
axes[2].set_xticklabels(blabels)
axes[2].set_xlabel("Days")
axes[2].set_ylabel("Original predicted-class probability drop")
axes[2].set_title("(c) Empirical temporal occlusion", loc="left", fontweight="bold")
axes[2].grid(axis="y", alpha=0.25)
fig.tight_layout()
save_figure(fig, "Figure3_temporal_recency_validation")
fs.to_csv(OUTPUT_DIR / "M3_temporal_summary_source.csv", index=False)
block_mean.to_csv(OUTPUT_DIR / "M3_block_mass_summary.csv", index=False)
occ_mean.to_csv(OUTPUT_DIR / "M3_occlusion_summary.csv", index=False)
occ_peak = occl.loc[occl.groupby(["Dataset", "Model"])["Delta_OriginalPredProb"].idxmax(), ["Dataset", "Model", "Block"]]
print(f"  Figure 3 data: temporal ρ range {tsum['Timestep_Spearman'].min():.4f}–{tsum['Timestep_Spearman'].max():.4f}; "
      f"KernelSHAP peak in the last block {int((tsum['KernelSHAP_Peak_Block'] == 6).sum())}/15; same IG and KernelSHAP peak block "
      f"{int(tsum['Same_Peak_Block'].astype(bool).sum())}/15; occlusion peak in the last block {int((occ_peak['Block'] == 6).sum())}/15")
for _, r in occ_peak[occ_peak["Block"] != 6].iterrows():
    print(f"    Occlusion peak not in the last block: {r['Dataset']} {r['Model']}, peak block {r['Block']}")
print("  Mean by block (15 combinations):")
print(block_mean.merge(occ_mean, on="Block").to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ============================== Step 4: Figure 4 ==============================
print("\n" + "=" * 100)
print(f"Step 4: Figure 4 (randomization tests and input-noise robustness), total {elapsed()}")
print("=" * 100)
final_param = ordered(final_param_df)
rows_lab = [f"{d} | {m}" for d in DATASET_ORDER for m in MODEL_ORDER]
param_matrix = np.array([[float(final_param[(final_param["Dataset"].astype(str) == d) & (final_param["Model"].astype(str) == m)][c].iloc[0])
                          for c in ["Map_Spearman_vs_Original", "Feature_Spearman_vs_Original"]] for d, m in ALL_COMBOS])
lab = ordered(label)
label_matrix = np.array([[float(lab[(lab["Dataset"].astype(str) == d) & (lab["Model"].astype(str) == m)][c].iloc[0])
                          for c in ["Map_Spearman", "Feature_Spearman", "Temporal_Spearman"]] for d, m in ALL_COMBOS])
noise_plot = noise.groupby("Noise_Percent_SD", as_index=False).agg(
    Map=("Map_Spearman_Mean", "mean"), Feature=("Feature_Spearman_Mean", "mean"),
    Temporal=("Temporal_Spearman_Mean", "mean"), PredictionAgreement=("Pred_Agreement_Mean", "mean")).sort_values("Noise_Percent_SD")
noise_matrix = noise_plot[["Map", "Feature", "Temporal", "PredictionAgreement"]].to_numpy(dtype=float)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9), gridspec_kw={"width_ratios": [1.0, 1.15, 1.05]})
vmin, vmax = widen(-0.05, 0.80, param_matrix)
im_a = draw_heatmap(axes[0], param_matrix, rows_lab, ["Map", "Feature"], "(a) Final-stage cascading parameter randomization",
                    vmin=vmin, vmax=vmax, cmap="coolwarm")
fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04).set_label("Spearman rho vs original")
vmin, vmax = widen(-0.70, 1.00, label_matrix)
im_b = draw_heatmap(axes[1], label_matrix, rows_lab, ["Map", "Feature", "Temporal"], "(b) Label-randomization attribution similarity",
                    vmin=vmin, vmax=vmax, cmap="coolwarm")
fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04).set_label("Spearman rho vs original")
vmin, vmax = widen(0.975, 1.000, noise_matrix, step=0.005)
im_c = draw_heatmap(axes[2], noise_matrix, [f"{v:.1f}%" for v in noise_plot["Noise_Percent_SD"]],
                    ["Map", "Feature", "Temporal", "Pred. agr."], "(c) Robustness to small input perturbations",
                    vmin=vmin, vmax=vmax, fmt=".4f", cmap="viridis")
fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04).set_label("Similarity / agreement")
axes[2].set_ylabel("Noise level (% of training feature SD)")
fig.tight_layout()
save_figure(fig, "Figure4_sanity_robustness")
final_param.to_csv(OUTPUT_DIR / "M4_parameter_randomization_final_stage.csv", index=False)
lab.to_csv(OUTPUT_DIR / "M4_label_randomization_source.csv", index=False)
noise_plot.to_csv(OUTPUT_DIR / "M4_noise_overall_summary.csv", index=False)
_na = [rows_lab[i] for i in range(len(rows_lab)) if not np.isfinite(label_matrix[i]).all()]
print(f"  Figure 4 data: final-stage parameter-randomization feature ρ range {np.nanmin(param_matrix[:, 1]):.4f}–{np.nanmax(param_matrix[:, 1]):.4f}; "
      f"label-randomization mean ρ (excluding n/a): map {np.nanmean(label_matrix[:, 0]):.4f}, feature {np.nanmean(label_matrix[:, 1]):.4f}, "
      f"temporal {np.nanmean(label_matrix[:, 2]):.4f}; rows marked n/a: {', '.join(_na) if _na else 'none'}")
print("  Input noise (mean over 15 combinations):")
print(noise_plot.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

# ============================== Step 5: Figure 5 ==============================
print("\n" + "=" * 100)
print(f"Step 5: Figure 5 (explanation agreement and perturbation extra drop, k = {MAIN_K}), total {elapsed()}")
print("=" * 100)
L3, L3a = RAW["L3"], RAW["L3a"]


def faith_panel(ax, g, ycol, q1col, q2col, state_col, title):
    n_draw = 0
    for _, r in g.iterrows():
        c = MARKET_COLOR[r["Market"]]
        xv, yv = float(r["Feature_Agreement_rho"]), float(r[ycol])
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        if q1col is not None and np.isfinite(float(r[q1col])) and np.isfinite(float(r[q2col])):
            ax.errorbar(xv, yv, yerr=[[max(yv - float(r[q1col]), 0)], [max(float(r[q2col]) - yv, 0)]],
                        fmt="none", ecolor=c, elinewidth=0.7, capsize=2)
        st = r[state_col]
        if st == "undetermined":
            ax.plot(xv, yv, marker="x", color="grey", markersize=6, linestyle="none")
        else:
            ax.plot(xv, yv, marker=MODEL_MARKER[r["Model"]], markersize=6, linestyle="none", markeredgewidth=1.0,
                    markerfacecolor=c if st == "positive" else "white", markeredgecolor=c)
        n_draw += 1
    ax.axhline(0, linewidth=0.8, linestyle="--", color="grey")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    return n_draw


fig, axes = plt.subplots(1, len(MAIN_K), figsize=(10.5, 4.2), sharey=True)
for i, (ax, k) in enumerate(zip(np.atleast_1d(axes), MAIN_K)):
    g = L3[L3["K"] == k]
    a = L3a[(L3a["K"] == k) & (L3a["Scope"] == "All")].iloc[0]
    n = faith_panel(ax, g, "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_Q025",
                    "Extra_Delta_OriginalPredProb_Q975", "Faith_State",
                    f"({'abc'[i]}) k = {k}")
    ax.text(0.02, 0.97, f"Spearman r = {a['Spearman_FeatureAgreement_vs_ExtraDrop']:.2f} (n = {int(a['N_Combos'])})",
            transform=ax.transAxes, va="top", fontsize=8)
    if i == 0:
        ax.set_ylabel("Extra drop in predicted-class probability\n(Top-k minus Random-k)")
    n_pos = int((g["Faith_State"] == "positive").sum())
    print(f"  k={k}: {n} points drawn; lower-tail criterion positive {n_pos}, non-positive {int((g['Faith_State'] == 'not_positive').sum())}, "
          f"undetermined {int((g['Faith_State'] == 'undetermined').sum())}; Spearman r = {a['Spearman_FeatureAgreement_vs_ExtraDrop']:.3f}")
fig.supxlabel("Feature-ranking agreement between IG and KernelSHAP (Spearman ρ)", fontsize=9)
np.atleast_1d(axes)[-1].legend(handles=market_handles() + criterion_handles(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
fig.tight_layout()
save_figure(fig, "Figure5_agreement_vs_behavioral_relevance")

# Supplementary Figure S1: k = 1
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
g1 = L3[L3["K"] == 1].merge(s1_ni, on=["Dataset", "Model"], how="left")
_dev = (g1["NI_Mean"] - g1["Extra_Mean_NonIdentical"]).abs().max()
if not np.isfinite(_dev) or _dev > 1e-6:
    raise ValueError(f"k=1 sensitivity means do not match the L results (max difference {_dev})")
print(f"  k=1 sensitivity means match the L results (max difference {_dev:.2e})")
faith_panel(axes[0], g1, "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_Q025",
            "Extra_Delta_OriginalPredProb_Q975", "Faith_State", "(a) k = 1, all repeats")
faith_panel(axes[1], g1, "NI_Mean", "NI_Q025", "NI_Q975", "Faith_State_NonIdentical",
            "(b) k = 1, excluding identical Top-1/Random-1 draws")
axes[0].set_ylabel("Extra drop in predicted-class probability\n(Top-k minus Random-k)")
fig.supxlabel("Feature-ranking agreement between IG and KernelSHAP (Spearman ρ)", fontsize=9)
axes[1].legend(handles=market_handles() + criterion_handles(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
fig.tight_layout()
save_figure(fig, "FigureS1_k1_identical_draw_sensitivity")
print(f"  Supplementary Figure S1: lower-tail criterion positive under the original definition {int((g1['Faith_State'] == 'positive').sum())}/15; "
      f"after removing identical repetitions {int((g1['Faith_State_NonIdentical'] == 'positive').sum())}/15; "
      f"identical repetitions in total {int(g1['N_Identical_Set'].sum())}")

# ============================== Step 6: Figure 6 ==============================
print("\n" + "=" * 100)
print(f"Step 6: Figure 6 (input-noise stability and cross-seed stability), total {elapsed()}")
print("=" * 100)
L4, L4x = RAW["L4"], RAW["L4x"]
st = ordered(L4[L4["In_Stability_Main"].astype(bool)]).reset_index(drop=True)
stx = L4x.set_index(["Dataset", "Model"])
flagged = st["N_Other_Seeds_SingleClass"] > 0
labels6 = [f"{r['Dataset']} {r['Model']}" + ("†" if f else "") for (_, r), f in zip(st.iterrows(), flagged)]
ypos = np.arange(len(st))[::-1]
METRICS = [("Feature", "(a) Feature ranking (Spearman ρ)"), ("Temporal", "(b) Temporal ranking (Spearman ρ)"),
           ("Prediction", "(c) Prediction agreement")]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 0.33 * len(st) + 1.6), sharey=True)
for ax, (labm, title) in zip(axes, METRICS):
    for yi, (_, r), f in zip(ypos, st.iterrows(), flagged):
        c = MARKET_COLOR["China" if r["Dataset"] in CN_DATASETS else "US"]
        ax.plot([r[f"Seed_{labm}_Min"], r[f"Seed_{labm}_Max"]], [yi, yi], color=c, linewidth=1.2)
        ax.plot(r[f"Seed_{labm}_Mean"], yi, marker="o", markersize=5, color=c, linestyle="none")
        if f:
            key = (str(r["Dataset"]), str(r["Model"]))
            v = float(stx.loc[key, f"Seed_{labm}_Mean"]) if key in stx.index else np.nan
            if np.isfinite(v):
                ax.plot(v, yi, marker="o", markersize=5, markerfacecolor="white", markeredgecolor=c, linestyle="none")
        ax.plot(r[f"Noise{PRIMARY_NOISE:g}_{labm}"], yi, marker="D", markersize=5, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=0.9, linestyle="none")
    vals = st[[f"Seed_{labm}_Min"]].to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        raise ValueError(f"All {labm} values in Figure 6 are missing")
    ax.set_xlim(min(0.0, np.nanmin(vals) - 0.05), 1.03)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(labels6)
h6 = [Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="black", markersize=6,
             label=f"Input noise ({PRIMARY_NOISE:g}% of training SD)"),
      Line2D([0], [0], marker="o", color=MARKET_COLOR["China"], markersize=5, label="Other seeds vs seed 42, China (mean; min–max)"),
      Line2D([0], [0], marker="o", color=MARKET_COLOR["US"], markersize=5, label="Other seeds vs seed 42, US (mean; min–max)"),
      Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="grey", markersize=5,
             label="† Mean after excluding seeds consistent with single-class prediction")]
fig.legend(handles=h6, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.09, 1, 1))
save_figure(fig, "Figure6_noise_vs_seed_stability")
print(f"  Figure 6: {len(st)} main-analysis combinations (Medical-2 GRU excluded); combinations marked †: "
      f"{', '.join([l for l, f in zip(labels6, flagged) if f]) or 'none'}")

# ============================== Step 7: Figure 7 ==============================
print("\n" + "=" * 100)
print(f"Step 7: Figure 7 (predictive performance and per-sample prediction agreement), total {elapsed()}")
print("=" * 100)
L5 = RAW["L5"]
L5m = L5[L5["In_Stability_Main"].astype(bool)]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
for (_, r), f in zip(st.iterrows(), flagged):
    c = MARKET_COLOR["China" if r["Dataset"] in CN_DATASETS else "US"]
    ax.plot(r["MacroF1_SD_Used"], r["Seed_Prediction_Mean"], marker=MODEL_MARKER[str(r["Model"])], markersize=7,
            linestyle="none", markeredgewidth=1.0, markeredgecolor=c, markerfacecolor="white" if f else c)
ax.set_xlabel("SD of Macro-F1 across five seeds")
ax.set_ylabel("Mean prediction agreement with seed 42")
ax.set_title("(a) Model–dataset combinations", loc="left", fontweight="bold")
ax.grid(alpha=0.25)
ax = axes[1]
for _, r in L5m.iterrows():
    c = MARKET_COLOR[r["Market"]]
    if bool(r["SingleClass"]):
        ax.plot(r["Abs_Delta_MacroF1_vs_42"], r["Disagreement_vs_42"], marker="x", markersize=6, color=c,
                markeredgewidth=1.0, linestyle="none")
    else:
        ax.plot(r["Abs_Delta_MacroF1_vs_42"], r["Disagreement_vs_42"], marker=MODEL_MARKER[r["Model"]], markersize=5,
                markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.9, linestyle="none")
ax.set_xlabel("|Macro-F1 difference| vs seed 42")
ax.set_ylabel("Share of test predictions differing from seed 42")
ax.set_title("(b) Individual seed models", loc="left", fontweight="bold")
ax.grid(alpha=0.25)
h7 = market_handles() + [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="grey", markersize=6,
           label="(a) open: includes seeds consistent\nwith single-class prediction"),
    Line2D([0], [0], marker="x", color="grey", markersize=6, linestyle="none",
           label="(b) seed consistent with\nsingle-class prediction")]
axes[1].legend(handles=h7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
fig.tight_layout()
save_figure(fig, "Figure7_performance_vs_prediction_agreement")
_cn = L5m[L5m["Market"] == "China"]
print(f"  Figure 7: {len(st)} combinations ({int(flagged.sum())} open); {len(L5m)} seed models ({int(L5m['SingleClass'].astype(bool).sum())} x); "
      f"China seed median |ΔMacro-F1| {_cn['Abs_Delta_MacroF1_vs_42'].median():.4f}, prediction agreement "
      f"{_cn['Prediction_Agreement_vs_42'].min():.4f}–{_cn['Prediction_Agreement_vs_42'].max():.4f}")

# ============================== Step 8: checks ==============================
print("\n" + "=" * 100)
print(f"Step 8: output checks, total {elapsed()}")
print("=" * 100)
_stems = ["Figure2_feature_consistency_behavioral_relevance", "Figure3_temporal_recency_validation", "Figure4_sanity_robustness",
          "Figure5_agreement_vs_behavioral_relevance", "Figure6_noise_vs_seed_stability",
          "Figure7_performance_vs_prediction_agreement", "FigureS1_k1_identical_draw_sensitivity"]
_lack = [f"{s}.{e}" for s in _stems for e in FIG_FORMATS if not (OUTPUT_DIR / f"{s}.{e}").exists()]
if _lack:
    raise RuntimeError(f"Figure files missing: {_lack}")
print(f"All {len(_stems) * len(FIG_FORMATS)} figure files generated: " + ", ".join(_stems))
print("Caption notes: in Figure 6, † marks combinations containing seeds consistent with single-class prediction (Macro-F1 = 2·accuracy/(3·(1+accuracy)), "
      "a necessary condition; prediction distributions not saved, not confirmed); error bars in Figure 5 and Supplementary Figure S1 are Monte Carlo 2.5%–97.5% ranges, not confidence intervals")
print(f"stage6_cell_M_data_figures_V2 finished, total time {elapsed()}")
