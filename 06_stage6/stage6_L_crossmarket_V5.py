"""
stage6_cell_L_data_crossmarket_V5

Aim: read existing Chinese and U.S. results only, without retraining or recomputing IG or KernelSHAP, to produce the cross-market summary and the two new analyses.

1. Merge 15 combinations (3 Chinese datasets x 3 models, 2 U.S. datasets x 3 models), recording the source of every metric:
   - Chinese agreement, perturbation test, input noise: original results xai_cell_B_V6, xai_cell_F_V1, xai_cell_D_V1 (Chinese dataset rows only);
   - Chinese five-seed: 8 combinations from the corrected results of stage6_cell_K_data_cnmultiseed_V7;
     Medical-2 GRU from xai_cell_K_V1 (this model's dropout is already 0, so no recomputation is needed),
     but its new seeds are computed relative to "seed 42 retrained under the reconstructed configuration", a reference model different from the noise experiment, so it is flagged separately and excluded from the main analysis;
   - All U.S. metrics: stage6_cell_F_data_kernelshap_V4, stage6_cell_G_data_temporal_V2,
     stage6_cell_H_data_robustness_V4, stage6_cell_J_data_multiseed_V1.
2. BH-FDR correction: uses the p-values of the original tests for each combination (means of correlation coefficients are not corrected).
   - Family 1: IG vs KernelSHAP feature-ranking correlations, 15 tests;
   - Family 2: IG vs KernelSHAP timestep-ranking correlations, 15 tests;
   - Sensitivity analysis: the 30 tests above corrected once more as one family.
3. New analysis 1: relationship between explanation agreement and behavioral faithfulness (paired by combination).
   - Explanation agreement: IG vs KernelSHAP feature-ranking correlation (main) and timestep-ranking correlation;
   - Behavioral faithfulness: drop in the original predicted-class probability after replacing the Top-k feature groups, minus the Random-k drop under the same background (extra drop),
     k = 1, 3, 5, following the direction and definition of the original experiment; "extra drop > 0" means Top-k is more important than random features;
   - "Monte Carlo lower-tail criterion": whether the 2.5th percentile of the extra drop over 50 background and random-feature repetitions is > 0;
     it describes whether the lower tail of the repetition distribution remains positive and is not a formal significance test;
   - each combination has one of three states: positive, non-positive, undetermined (agreement or faithfulness metric missing); undetermined combinations are excluded from the cross-table;
   - evidence: per-combination cross-table (agreement significant or not x lower-tail criterion),
     and the Spearman correlation between agreement and extra drop across 15 combinations (descriptive; few combinations, limited statistical power);
   - sensitivity analysis: the original experiment allows Random-k to overlap with Top-k; for small k (especially k=1), when the random draw is identical to the Top-k feature set
     the extra drop is exactly 0, which pulls the 2.5th percentile down to 0. The main results keep the original definition; in addition, using the per-repetition details,
     the 2.5th percentile is recomputed after removing repetitions where "Random-k is identical to Top-k", as a sensitivity result.
4. New analysis 2: difference in explanation stability between local input perturbations and training seeds (descriptive).
   - Same metrics: feature-ranking ρ, timestep-ranking ρ, prediction agreement; same reference: the seed-42 model on clean inputs;
   - Input noise: fixed model, noise of 0.5%, 1% and 2% of the training-set SD added to the inputs (the strongest, 2%, is used for the main comparison);
   - Training seeds: the other 4 seeds relative to seed 42.
     The five-seed results did not save per-sample predictions or predicted-class distributions, so single-class prediction cannot be confirmed directly and is only flagged:
     if a model predicts only class c, accuracy = the share s of class c, and Macro-F1 = 2·accuracy / (3·(1+accuracy));
     seeds satisfying this equation (difference < SINGLE_CLASS_TOL) are flagged as "consistent with single-class prediction" (a necessary condition, not confirmation);
     seeds with low Macro-F1 that do not satisfy the equation are flagged only as "low performance";
   - both versions are reported: the main analysis keeps all seeds (no unconfirmed filtering);
     the sensitivity analysis removes the flagged seeds;
   - gap = noise stability − cross-seed stability; the two perturbations differ in nature, so the gap is descriptive evidence only;
     a combination-level Wilcoxon signed-rank test is also given, for descriptive reference only, as combinations are not independent.
   - Note: IG in the noise experiment uses 200 fixed samples and the five-seed analysis uses 500; this is stated in the table notes.
5. Output: combination source table, BH details, per-combination tables and association statistics for both analyses, paper summary table,
   and 3 figures (PNG, JPEG, TIFF, EPS, 600 dpi).
All analyses use only combinations and original metrics that actually exist: if a combination required by the main analysis is missing, the cell stops with an error; missing metric values are recorded as undetermined and the actual denominators are printed,
with no zero-filling or fabrication.
"""

# ------------------------------ Settings (editable) ------------------------------
PROJECT_ROOT = "/home/i/2026_0712/stock_i"      # project root containing the original Chinese results
# Directory names of the original Chinese results (checked in order: project root, project root/0924, current Notebook directory)
CN_RESULT_DIRS = {"B": "xai_cell_B_V6", "F": "xai_cell_F_V1", "D": "xai_cell_D_V1", "K1": "xai_cell_K_V1"}
CN_SEARCH_SUBDIRS = ["", "0924"]
# Result directories of this stage (stage6) (relative to the current Notebook directory)
CN_MULTISEED_DIR = "stage6/stage6_cell_K_data_cnmultiseed_V7"
US_DIRS = {"F": "stage6/stage6_cell_F_data_kernelshap_V4",
           "G": "stage6/stage6_cell_G_data_temporal_V2",
           "H": "stage6/stage6_cell_H_data_robustness_V4",
           "J": "stage6/stage6_cell_J_data_multiseed_V1"}
OUTPUT_ROOT = "stage6/stage6_cell_L_data_crossmarket_V5"

CN_DATASETS = ["Medical-1", "Medical-2", "Power"]
US_DATASETS = ["US-Med", "US-Pow"]
MODELS = ["GRU", "BiLSTM", "LSTM"]
SEPARATE_REF_COMBOS = [("Medical-2", "GRU")]   # combinations whose five-seed reference model differs from the noise experiment; excluded from the main stability analysis
PRIMARY_SEED = 42
K_LIST = [1, 3, 5]
PRIMARY_K = 3                  # k used in the main panel of Figure 1 and in the main association result
NOISE_LEVELS = [0.5, 1.0, 2.0]  # percentage of the training-set SD
PRIMARY_NOISE = 2.0            # strongest noise, used for the main comparison
SINGLE_CLASS_TOL = 1e-6        # single-class prediction rule: |Macro-F1 − 2·accuracy/(3·(1+accuracy))| below this value
LOW_F1 = 0.20                  # used only to flag "low performance", not for exclusion
SEED_SETS = [("Main_Include_All", False), ("Sensitivity_Exclude_Flagged", True)]   # two versions of the stability analysis (the main analysis keeps all seeds)
ALPHA = 0.05
FIG_DPI = 600
FIG_FORMATS = ["png", "jpg", "tif", "eps"]

# ------------------------------ Imports ------------------------------
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path(PROJECT_ROOT)
OUTPUT_DIR = Path(OUTPUT_ROOT)
ALL_COMBOS = [(d, m) for d in CN_DATASETS + US_DATASETS for m in MODELS]


def rel(p):
    # Show only the path starting from the project root or current directory, not the full absolute path
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return "./" + p.as_posix()


def elapsed():
    s = int(time.time() - T0)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def find_cn_dir(name):
    # Check the original Chinese result directories in a fixed order, no folder scanning
    for sub in CN_SEARCH_SUBDIRS:
        cand = ROOT / sub / name if sub else ROOT / name
        if cand.is_dir():
            return cand
    if Path(name).is_dir():
        return Path(name)
    return None


CN_DIR = {k: find_cn_dir(v) for k, v in CN_RESULT_DIRS.items()}

# Rule: the first two lines print the input and output locations
_in_list = [rel(CN_DIR[k]) + "/" if CN_DIR[k] else CN_RESULT_DIRS[k] + " (not found)" for k in CN_RESULT_DIRS]
_in_list += ["./" + CN_MULTISEED_DIR + "/"] + ["./" + v + "/" for v in US_DIRS.values()]
print("Input directories: " + ", ".join(_in_list))
print(f"Output directory: ./{OUTPUT_DIR.as_posix()}/")

# ------------------------------ Input file check ------------------------------
INPUT_FILES = {
    "cn_agree_feat": (CN_DIR["B"], "B2_ig_vs_kernelshap.csv"),
    "cn_faith": (CN_DIR["B"], "B3_paired_faithfulness_summary.csv"),
    "cn_faith_detail": (CN_DIR["B"], "B3_paired_faithfulness_detail.csv"),
    "cn_agree_time": (CN_DIR["F"], "F5_temporal_method_summary.csv"),
    "cn_noise": (CN_DIR["D"], "D2_input_noise_robustness_summary.csv"),
    "cn_seed_k1": (CN_DIR["K1"], "multiseed_detail.csv"),
    "cn_seed_v7": (Path(CN_MULTISEED_DIR), "multiseed_detail_corrected.csv"),
    "us_agree_feat": (Path(US_DIRS["F"]), "B2_ig_vs_kernelshap.csv"),
    "us_faith": (Path(US_DIRS["F"]), "B3_paired_faithfulness_summary.csv"),
    "us_faith_detail": (Path(US_DIRS["F"]), "B3_paired_faithfulness_detail.csv"),
    "us_agree_time": (Path(US_DIRS["G"]), "F5_temporal_method_summary.csv"),
    "us_noise": (Path(US_DIRS["H"]), "D2_input_noise_robustness_summary.csv"),
    "us_seed": (Path(US_DIRS["J"]), "multiseed_detail.csv"),
}
_missing = []
for _key, (_d, _f) in INPUT_FILES.items():
    if _d is None:
        _missing.append(f"{_key}: directory not found ({_f})")
    elif not (_d / _f).exists():
        _missing.append(f"{_key}: {rel(_d / _f)}")
if _missing:
    print("The following input files do not exist (original Chinese result directories checked: project root, project root/0924, current directory):")
    for _m in _missing:
        print("  " + _m)
    raise FileNotFoundError(f"{len(_missing)} input files are missing")
print(f"Input file check passed ({len(INPUT_FILES)} files)")
for _key, (_d, _f) in INPUT_FILES.items():
    print(f"  {_key}: {rel(_d / _f)}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = {k: pd.read_csv(d / f) for k, (d, f) in INPUT_FILES.items()}


def market_of(ds):
    return "China" if ds in CN_DATASETS else "US"


def take(df, datasets, name, required):
    # Keep only rows of the specified datasets and check fields
    lack = [c for c in required if c not in df.columns]
    if lack:
        raise ValueError(f"{name} is missing fields: {lack}")
    out = df[df["Dataset"].isin(datasets)].copy()
    print(f"  {name}: {len(df)} rows in the original file, rows of {', '.join(datasets)}: {len(out)}")
    return out


def check_keys(df, keys, expected, name):
    # Check that each target key appears exactly once: duplicates, missing keys and unexpected extra keys all stop the cell
    cnt = df.groupby(keys).size()
    dup = cnt[cnt > 1]
    if len(dup) > 0:
        raise ValueError(f"{name} has duplicate rows: {dup.index.tolist()}")
    have = set(map(tuple, df[keys].values.tolist()))
    lack = [e for e in expected if e not in have]
    extra = sorted(have - set(map(tuple, expected)), key=str)
    print(f"  {name}: expected {len(expected)}, present {len(expected) - len(lack)}, extra {len(extra)}" +
          (f", missing {lack}" if lack else "") + (f", extra {extra}" if extra else ""))
    if extra:
        raise ValueError(f"{name} contains {len(extra)} unexpected items; stopping")
    if lack:
        raise ValueError(f"{name} is missing {len(lack)} items required by the main analysis; stopping (no fabrication)")
    return lack


def bh_fdr(p):
    # Benjamini-Hochberg correction; missing p-values are excluded
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def save_fig(fig, stem):
    for fmt in FIG_FORMATS:
        path = OUTPUT_DIR / f"{stem}.{fmt}"
        kw = {"dpi": FIG_DPI, "bbox_inches": "tight"}
        if fmt == "jpg":
            kw["pil_kwargs"] = {"quality": 95}
        if fmt == "tif":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format={"jpg": "jpeg", "tif": "tiff"}.get(fmt, fmt), **kw)
    plt.close(fig)
    print(f"  Saved: ./{OUTPUT_DIR.as_posix()}/{stem}." + "/.".join(FIG_FORMATS))


# ============================== Step 1: read and merge ==============================
print("\n" + "=" * 100)
print(f"Step 1: read and merge 15 combinations, total {elapsed()}")
print("=" * 100)

# 1a. IG vs KernelSHAP feature-ranking agreement
feat = pd.concat([
    take(RAW["cn_agree_feat"], CN_DATASETS, "China feature agreement B2", ["Dataset", "Model", "IG_vs_KernelSHAP_rho", "p_value"]).assign(Source_Agreement_Feature="xai_cell_B_V6"),
    take(RAW["us_agree_feat"], US_DATASETS, "U.S. feature agreement B2", ["Dataset", "Model", "IG_vs_KernelSHAP_rho", "p_value"]).assign(Source_Agreement_Feature="stage6_cell_F_data_kernelshap_V4"),
], ignore_index=True).rename(columns={"IG_vs_KernelSHAP_rho": "Feature_Agreement_rho", "p_value": "Feature_Agreement_p"})
check_keys(feat, ["Dataset", "Model"], ALL_COMBOS, "feature agreement")

# 1b. IG vs KernelSHAP timestep-ranking agreement
_tcols = ["Dataset", "Model", "Timestep_Spearman", "Timestep_p_value", "Block_Spearman", "Block_p_value",
          "IG_Peak_Block", "KernelSHAP_Peak_Block", "Same_Peak_Block"]
tim = pd.concat([
    take(RAW["cn_agree_time"], CN_DATASETS, "China temporal agreement F5", _tcols).assign(Source_Agreement_Temporal="xai_cell_F_V1"),
    take(RAW["us_agree_time"], US_DATASETS, "U.S. temporal agreement F5", _tcols).assign(Source_Agreement_Temporal="stage6_cell_G_data_temporal_V2"),
], ignore_index=True)[_tcols + ["Source_Agreement_Temporal"]].rename(
    columns={"Timestep_Spearman": "Temporal_Agreement_rho", "Timestep_p_value": "Temporal_Agreement_p"})
check_keys(tim, ["Dataset", "Model"], ALL_COMBOS, "temporal agreement")

# 1c. Top-k paired perturbation test
_fcols = ["Dataset", "Model", "K", "Repeats", "Baseline_OriginalPredProb",
          "TopK_Delta_OriginalPredProb_Mean", "RandomK_Delta_OriginalPredProb_Mean",
          "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_SD",
          "Extra_Delta_OriginalPredProb_Q025", "Extra_Delta_OriginalPredProb_Q975",
          "Positive_ExtraProb_Rate", "Extra_Delta_MacroF1_Mean"]
faith = pd.concat([
    take(RAW["cn_faith"], CN_DATASETS, "China perturbation test B3", _fcols).assign(Source_Faithfulness="xai_cell_B_V6"),
    take(RAW["us_faith"], US_DATASETS, "U.S. perturbation test B3", _fcols).assign(Source_Faithfulness="stage6_cell_F_data_kernelshap_V4"),
], ignore_index=True)[_fcols + ["Source_Faithfulness"]]
check_keys(faith, ["Dataset", "Model", "K"], [(d, m, k) for d, m in ALL_COMBOS for k in K_LIST], "perturbation test")

# 1c-2. Per-repetition perturbation-test details: flag repetitions where Random-k is identical to Top-k (for the sensitivity analysis)
_dcols = ["Dataset", "Model", "K", "Repeat", "TopK_Features", "RandomK_Features", "Extra_Delta_OriginalPredProb"]
fdet = pd.concat([take(RAW["cn_faith_detail"], CN_DATASETS, "China perturbation-test details B3", _dcols),
                  take(RAW["us_faith_detail"], US_DATASETS, "U.S. perturbation-test details B3", _dcols)], ignore_index=True)[_dcols]
# Format check: feature sets are separated by "|"; each set has exactly K distinct features; repetition numbers are 1..Repeats and complete
def _split(x):
    return [t.strip() for t in str(x).split("|") if t.strip() != ""]


_top = fdet["TopK_Features"].map(_split)
_rnd = fdet["RandomK_Features"].map(_split)
_bad_fmt = [(i, len(a), len(b)) for i, (a, b, k) in enumerate(zip(_top, _rnd, fdet["K"]))
            if len(a) != k or len(b) != k or len(set(a)) != k or len(set(b)) != k]
if _bad_fmt:
    print(f"  Error: {len(_bad_fmt)} rows have an invalid feature-set format (row, Top-k count, Random-k count); first 10 rows: {_bad_fmt[:10]}")
    raise ValueError("Feature-set format in the perturbation-test details does not match K")
check_keys(fdet.drop_duplicates(["Dataset", "Model", "K"]), ["Dataset", "Model", "K"],
           [(d, m, k) for d, m in ALL_COMBOS for k in K_LIST], "perturbation-test details (combination x k)")
_dup_rep = fdet.duplicated(["Dataset", "Model", "K", "Repeat"]).sum()
if _dup_rep:
    raise ValueError(f"Perturbation-test details contain {_dup_rep} duplicated rows of (combination x k x repetition)")
_n_na = int(fdet["Extra_Delta_OriginalPredProb"].isna().sum() + np.isinf(fdet["Extra_Delta_OriginalPredProb"].astype(float)).sum())
if _n_na:
    raise ValueError(f"Extra drop in the perturbation-test details has {_n_na} missing or infinite values")
_rep_chk = fdet.groupby(["Dataset", "Model", "K"])["Repeat"].agg(lambda r: sorted(r.tolist()) == list(range(1, len(r) + 1)))
_rep_n = fdet.groupby(["Dataset", "Model", "K"]).size().rename("N").reset_index().merge(
    faith[["Dataset", "Model", "K", "Repeats"]], on=["Dataset", "Model", "K"], how="left")
if (not _rep_chk.all()) or (_rep_n["N"] != _rep_n["Repeats"]).any():
    raise ValueError("Repetition numbers in the perturbation-test details are incomplete, or the number of repetitions differs from the summary table")
print(f"  Perturbation-test detail format check passed: Top-k and Random-k in every row are K distinct features; "
      f"repetition numbers complete for {len(_rep_n)} (combination x k), counts consistent with the summary table")
fdet["Identical_Set"] = [set(a) == set(b) for a, b in zip(_top, _rnd)]
_sens = []
for (d, m, k), g in fdet.groupby(["Dataset", "Model", "K"]):
    gg = g[~g["Identical_Set"]]["Extra_Delta_OriginalPredProb"]
    # Check: the mean over all repetitions in the details should match the summary table
    _sens.append({"Dataset": d, "Model": m, "K": k, "Repeats_Detail": len(g),
                  "Extra_Mean_Detail": g["Extra_Delta_OriginalPredProb"].mean(),
                  "N_Identical_Set": int(g["Identical_Set"].sum()), "N_NonIdentical": int(len(gg)),
                  "Extra_Mean_NonIdentical": gg.mean() if len(gg) else np.nan,
                  "Extra_Q025_NonIdentical": gg.quantile(0.025) if len(gg) else np.nan})
sens = pd.DataFrame(_sens)
_chk = faith.merge(sens, on=["Dataset", "Model", "K"], how="left")
_dev = (_chk["Extra_Mean_Detail"] - _chk["Extra_Delta_OriginalPredProb_Mean"]).abs().max()
print(f"  Perturbation-test detail check: {len(fdet)} detail rows, max difference between detail means and summary table {_dev:.2e}; "
      f"repetitions with Random-k identical to Top-k: {int(fdet['Identical_Set'].sum())}"
      f" (k=1: {int(fdet.loc[fdet['K'] == 1, 'Identical_Set'].sum())}, k=3: {int(fdet.loc[fdet['K'] == 3, 'Identical_Set'].sum())}, "
      f"k=5: {int(fdet.loc[fdet['K'] == 5, 'Identical_Set'].sum())})")
_n_unmatched = int(_chk["Extra_Mean_Detail"].isna().sum())
if _n_unmatched or not np.isfinite(_dev) or _dev > 1e-6:
    raise ValueError(f"Perturbation-test details do not match the summary table ({_n_unmatched} detail combinations not found in the summary table, max mean difference {_dev})")
print(f"  Perturbation-test detail completeness check passed: {len(_chk)}/{len(ALL_COMBOS) * len(K_LIST)} (combination x k) have complete details with means matching the summary table")
faith = faith.merge(sens[["Dataset", "Model", "K", "N_Identical_Set", "N_NonIdentical", "Extra_Mean_NonIdentical",
                          "Extra_Q025_NonIdentical"]], on=["Dataset", "Model", "K"], how="left")

# 1d. Input-noise stability
_ncols = ["Dataset", "Model", "Noise_Percent_SD", "N_Repeats", "Feature_Spearman_Mean", "Feature_Spearman_SD",
          "Temporal_Spearman_Mean", "Temporal_Spearman_SD", "Pred_Agreement_Mean", "Pred_Agreement_SD",
          "Map_Spearman_Mean"]
noise = pd.concat([
    take(RAW["cn_noise"], CN_DATASETS, "China input noise D2", _ncols).assign(Source_Noise="xai_cell_D_V1"),
    take(RAW["us_noise"], US_DATASETS, "U.S. input noise D2", _ncols).assign(Source_Noise="stage6_cell_H_data_robustness_V4"),
], ignore_index=True)[_ncols + ["Source_Noise"]]
noise["Noise_Percent_SD"] = noise["Noise_Percent_SD"].round(3)
check_keys(noise, ["Dataset", "Model", "Noise_Percent_SD"],
           [(d, m, lv) for d, m in ALL_COMBOS for lv in NOISE_LEVELS], "input noise")

# 1e. Five-seed stability
_scols = ["Dataset", "Model", "Seed", "MacroF1", "Accuracy", "Prediction_Agreement_vs_42",
          "Feature_Spearman_vs_42", "Temporal_Spearman_vs_42"]
_k1 = RAW["cn_seed_k1"].rename(columns={"Prediction_Agreement_vs_NewSeed42": "Prediction_Agreement_vs_42",
                                        "Feature_Spearman_vs_NewSeed42": "Feature_Spearman_vs_42",
                                        "Temporal_Spearman_vs_NewSeed42": "Temporal_Spearman_vs_42"})
_k1 = take(_k1, ["Medical-2"], "China Medical-2 GRU five-seed xai_cell_K_V1", _scols)
_k1 = _k1[_k1["Model"] == "GRU"]
seed = pd.concat([
    take(RAW["cn_seed_v7"], CN_DATASETS, "China five-seed corrected K V7", _scols).assign(Source_Seed="stage6_cell_K_data_cnmultiseed_V7", Seed_Reference="original_seed42"),
    _k1.assign(Source_Seed="xai_cell_K_V1", Seed_Reference="reconstructed_new_seed42"),
    take(RAW["us_seed"], US_DATASETS, "U.S. five-seed J V1", _scols).assign(Source_Seed="stage6_cell_J_data_multiseed_V1", Seed_Reference="original_seed42"),
], ignore_index=True)[_scols + ["Source_Seed", "Seed_Reference"]]
check_keys(seed, ["Dataset", "Model", "Seed"], [(d, m, s) for d, m in ALL_COMBOS for s in [42, 123, 777, 0, 2024]], "five-seed")
# Single-class prediction: Macro-F1 exactly equals 2·accuracy/(3·(1+accuracy))
seed["SingleClass_F1_If_One_Class"] = 2 * seed["Accuracy"] / (3 * (1 + seed["Accuracy"]))
seed["SingleClass_Residual"] = (seed["MacroF1"] - seed["SingleClass_F1_If_One_Class"]).abs()
seed["SingleClass"] = seed["SingleClass_Residual"] < SINGLE_CLASS_TOL
seed["Low_Performance"] = (seed["MacroF1"] < LOW_F1) & (~seed["SingleClass"])
print(f"  Consistent with single-class prediction (|Macro-F1 − 2·accuracy/(3·(1+accuracy))| < {SINGLE_CLASS_TOL}; necessary condition; prediction distributions not saved, not confirmed): "
      f"{int(seed['SingleClass'].sum())}/{len(seed)} seed models; additionally low performance (Macro-F1 < {LOW_F1}, but not single-class)"
      f" {int(seed['Low_Performance'].sum())}")
_show_sc = seed[seed["SingleClass"] | seed["Low_Performance"] | (seed["MacroF1"] < 0.26)]
if len(_show_sc):
    print("  Seed models with low Macro-F1 (< 0.26) or consistent with single-class prediction:")
    print(_show_sc[["Dataset", "Model", "Seed", "MacroF1", "Accuracy", "SingleClass_F1_If_One_Class",
                    "SingleClass_Residual", "SingleClass", "Low_Performance"]].to_string(
        index=False, float_format=lambda x: f"{x:.6g}"))
for _c in ["Feature_Spearman_vs_42", "Temporal_Spearman_vs_42", "Prediction_Agreement_vs_42"]:
    _n_bad = int(seed[_c].isna().sum() + np.isinf(seed[_c].astype(float)).sum())
    if _n_bad:
        print(f"  Note: five-seed {_c} has {_n_bad} missing or infinite values, treated as missing")

# Combination source table
src_rows = []
for d, m in ALL_COMBOS:
    s = seed[(seed["Dataset"] == d) & (seed["Model"] == m)]
    alt = s[s["Seed"] != PRIMARY_SEED]
    src_rows.append({
        "Market": market_of(d), "Dataset": d, "Model": m,
        "Source_Agreement_Feature": feat.loc[(feat["Dataset"] == d) & (feat["Model"] == m), "Source_Agreement_Feature"].squeeze() if ((feat["Dataset"] == d) & (feat["Model"] == m)).any() else "missing",
        "Source_Agreement_Temporal": tim.loc[(tim["Dataset"] == d) & (tim["Model"] == m), "Source_Agreement_Temporal"].squeeze() if ((tim["Dataset"] == d) & (tim["Model"] == m)).any() else "missing",
        "Source_Faithfulness": faith.loc[(faith["Dataset"] == d) & (faith["Model"] == m), "Source_Faithfulness"].iloc[0] if ((faith["Dataset"] == d) & (faith["Model"] == m)).any() else "missing",
        "Source_Noise": noise.loc[(noise["Dataset"] == d) & (noise["Model"] == m), "Source_Noise"].iloc[0] if ((noise["Dataset"] == d) & (noise["Model"] == m)).any() else "missing",
        "Source_Seed": s["Source_Seed"].iloc[0] if len(s) else "missing",
        "Seed_Reference": s["Seed_Reference"].iloc[0] if len(s) else "missing",
        "Seed42_MacroF1": float(s.loc[s["Seed"] == PRIMARY_SEED, "MacroF1"].iloc[0]) if (s["Seed"] == PRIMARY_SEED).any() else np.nan,
        "N_SingleClass_Other_Seeds": int(alt["SingleClass"].sum()),
        "In_Stability_Main": (d, m) not in SEPARATE_REF_COMBOS,
    })
src_df = pd.DataFrame(src_rows)
src_df.to_csv(OUTPUT_DIR / "L1_combination_sources.csv", index=False)
print("\nCombination source table:")
print(src_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================== Step 2: BH-FDR ==============================
print("\n" + "=" * 100)
print(f"Step 2: BH-FDR correction (original test p-values), total {elapsed()}")
print("=" * 100)
bh = pd.concat([
    feat[["Dataset", "Model", "Feature_Agreement_rho", "Feature_Agreement_p"]].rename(
        columns={"Feature_Agreement_rho": "rho", "Feature_Agreement_p": "p_raw"}).assign(Family="Feature_IG_vs_KernelSHAP"),
    tim[["Dataset", "Model", "Temporal_Agreement_rho", "Temporal_Agreement_p"]].rename(
        columns={"Temporal_Agreement_rho": "rho", "Temporal_Agreement_p": "p_raw"}).assign(Family="Temporal_IG_vs_KernelSHAP"),
], ignore_index=True)
bh.insert(0, "Market", bh["Dataset"].map(market_of))
# BH input validity: non-missing p-values must be finite and within [0, 1]; ρ must be finite
_p = bh["p_raw"].astype(float)
_bad_p = bh[_p.notna() & (~np.isfinite(_p) | (_p < 0) | (_p > 1))]
_bad_r = bh[bh["rho"].notna() & ~np.isfinite(bh["rho"].astype(float))]
print(f"BH input check: {len(bh)} tests, missing p-values {int(_p.isna().sum())}, invalid (non-finite or outside 0–1) {len(_bad_p)}, non-finite ρ {len(_bad_r)}")
if len(_bad_p) or len(_bad_r):
    print(pd.concat([_bad_p, _bad_r]).to_string(index=False))
    raise ValueError("BH input contains invalid p-values or ρ")
bh["p_BH_within_family"] = np.nan
for fam in bh["Family"].unique():
    idx = bh["Family"] == fam
    bh.loc[idx, "p_BH_within_family"] = bh_fdr(bh.loc[idx, "p_raw"].values)
bh["p_BH_all30"] = bh_fdr(bh["p_raw"].values)
bh["Significant_within_family"] = (bh["p_BH_within_family"] < ALPHA) & (bh["rho"] > 0)
bh["Significant_all30"] = (bh["p_BH_all30"] < ALPHA) & (bh["rho"] > 0)
bh.to_csv(OUTPUT_DIR / "L2_bh_fdr.csv", index=False)
print(bh.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
for fam in bh["Family"].unique():
    g = bh[bh["Family"] == fam]
    print(f"{fam}: valid {int(g['p_raw'].notna().sum())}/15; ρ range {g['rho'].min():.3f}–{g['rho'].max():.3f}; "
          f"positive and significant after within-family correction {int(g['Significant_within_family'].sum())}/15; after correction across all 30 {int(g['Significant_all30'].sum())}/15")

# ============================== Step 3: agreement and faithfulness ==============================
print("\n" + "=" * 100)
print(f"Step 3: new analysis 1: explanation agreement and behavioral faithfulness (paired by combination), total {elapsed()}")
print("=" * 100)
_bh_f = bh[bh["Family"] == "Feature_IG_vs_KernelSHAP"][["Dataset", "Model", "p_BH_within_family", "Significant_within_family"]].rename(
    columns={"p_BH_within_family": "Feature_Agreement_p_BH", "Significant_within_family": "Feature_Agreement_Significant"})
_bh_t = bh[bh["Family"] == "Temporal_IG_vs_KernelSHAP"][["Dataset", "Model", "p_BH_within_family", "Significant_within_family"]].rename(
    columns={"p_BH_within_family": "Temporal_Agreement_p_BH", "Significant_within_family": "Temporal_Agreement_Significant"})
af = faith.merge(feat[["Dataset", "Model", "Feature_Agreement_rho", "Feature_Agreement_p"]], on=["Dataset", "Model"], how="left")
af = af.merge(_bh_f, on=["Dataset", "Model"], how="left")
af = af.merge(tim[["Dataset", "Model", "Temporal_Agreement_rho"]], on=["Dataset", "Model"], how="left").merge(_bh_t, on=["Dataset", "Model"], how="left")
af.insert(0, "Market", af["Dataset"].map(market_of))
af["Extra_Drop_Positive_Mean"] = af["Extra_Delta_OriginalPredProb_Mean"] > 0
def tri_state(v):
    # Three states: positive / non-positive / undetermined (missing or infinite)
    a = pd.to_numeric(pd.Series(v), errors="coerce").astype(float).values
    bad = ~np.isfinite(a)
    return np.where(bad, "undetermined", np.where(a > 0, "positive", "not_positive"))


af["Faith_State"] = tri_state(af["Extra_Delta_OriginalPredProb_Q025"])
af["Faith_State_NonIdentical"] = tri_state(af["Extra_Q025_NonIdentical"])
af["Agreement_State"] = np.where((~np.isfinite(af["Feature_Agreement_p_BH"].astype(float))) |
                                 (~np.isfinite(af["Feature_Agreement_rho"].astype(float))), "undetermined",
                                 np.where(af["Feature_Agreement_Significant"], "significant", "not_significant"))
af["Extra_Drop_Q025_Positive"] = af["Faith_State"] == "positive"
af["Extra_Drop_Q025_Positive_NonIdentical"] = af["Faith_State_NonIdentical"] == "positive"
af["Complete"] = (af["Agreement_State"] != "undetermined") & (af["Faith_State"] != "undetermined")
af["Complete_NonIdentical"] = (af["Agreement_State"] != "undetermined") & (af["Faith_State_NonIdentical"] != "undetermined")
af["Relative_Extra_Drop"] = af["Extra_Delta_OriginalPredProb_Mean"] / af["Baseline_OriginalPredProb"]
# Decoupled: agreement significant, but the Monte Carlo 2.5th percentile of the Top-k extra drop relative to random is not positive
# Decoupled: agreement significant, but the lower-tail criterion is "non-positive"; undetermined combinations are recorded as missing
af["Decoupled"] = np.where(af["Complete"], (af["Agreement_State"] == "significant") & (af["Faith_State"] == "not_positive"), np.nan)
af["Decoupled_NonIdentical"] = np.where(af["Complete_NonIdentical"],
                                        (af["Agreement_State"] == "significant") & (af["Faith_State_NonIdentical"] == "not_positive"), np.nan)
_n_und = int((~af["Complete"]).sum())
print(f"Undetermined (combination x k): {_n_und}/{len(af)}" + (" (excluded from the cross-table)" if _n_und else ""))
af = af.sort_values(["Market", "Dataset", "Model", "K"], ascending=[True, True, True, True]).reset_index(drop=True)
af.to_csv(OUTPUT_DIR / "L3_agreement_faithfulness_by_combo.csv", index=False)
print(af[["Market", "Dataset", "Model", "K", "Feature_Agreement_rho", "Feature_Agreement_Significant",
          "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_Q025", "Positive_ExtraProb_Rate",
          "Relative_Extra_Drop", "Decoupled", "N_Identical_Set", "Extra_Q025_NonIdentical", "Decoupled_NonIdentical"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Cross-table and association
assoc_rows, cross_rows = [], []
for k in K_LIST:
    g = af[af["K"] == k]
    for scope, gg_all in [("All", g), ("China", g[g["Market"] == "China"]), ("US", g[g["Market"] == "US"])]:
        gg = gg_all[gg_all["Complete"]]
        gs = gg_all[gg_all["Complete_NonIdentical"]]
        ok = gg_all[["Feature_Agreement_rho", "Extra_Delta_OriginalPredProb_Mean"]].dropna()
        if len(ok) >= 4 and ok["Feature_Agreement_rho"].nunique() > 1:
            r, p = spearmanr(ok["Feature_Agreement_rho"], ok["Extra_Delta_OriginalPredProb_Mean"])
        else:
            r, p = np.nan, np.nan
        ok2 = gg_all[["Temporal_Agreement_rho", "Extra_Delta_OriginalPredProb_Mean"]].dropna()
        if len(ok2) >= 4 and ok2["Temporal_Agreement_rho"].nunique() > 1:
            r2, p2 = spearmanr(ok2["Temporal_Agreement_rho"], ok2["Extra_Delta_OriginalPredProb_Mean"])
        else:
            r2, p2 = np.nan, np.nan
        assoc_rows.append({"K": k, "Scope": scope, "N_Combos": len(ok),
                           "Spearman_FeatureAgreement_vs_ExtraDrop": r, "p_FeatureAgreement": p,
                           "Spearman_TemporalAgreement_vs_ExtraDrop": r2, "p_TemporalAgreement": p2})
        cross_rows.append({
            "K": k, "Scope": scope, "N_Combos": len(gg_all), "N_Complete": len(gg), "N_Undetermined": len(gg_all) - len(gg),
            "Agreement_Sig_and_Q025_Positive": int((gg["Feature_Agreement_Significant"] & gg["Extra_Drop_Q025_Positive"]).sum()),
            "Agreement_Sig_and_Q025_NotPositive": int((gg["Feature_Agreement_Significant"] & ~gg["Extra_Drop_Q025_Positive"]).sum()),
            "Agreement_NotSig_and_Q025_Positive": int((~gg["Feature_Agreement_Significant"] & gg["Extra_Drop_Q025_Positive"]).sum()),
            "Agreement_NotSig_and_Q025_NotPositive": int((~gg["Feature_Agreement_Significant"] & ~gg["Extra_Drop_Q025_Positive"]).sum()),
            "Mean_Extra_Drop_Positive": int(gg["Extra_Drop_Positive_Mean"].sum()),
            "Sens_N_Complete": len(gs),
            "Sens_Agreement_Sig_and_Q025NI_Positive": int((gs["Feature_Agreement_Significant"] & gs["Extra_Drop_Q025_Positive_NonIdentical"]).sum()),
            "Sens_Agreement_Sig_and_Q025NI_NotPositive": int((gs["Feature_Agreement_Significant"] & ~gs["Extra_Drop_Q025_Positive_NonIdentical"]).sum()),
            "Feature_Agreement_rho_Min": gg["Feature_Agreement_rho"].min(),
            "Feature_Agreement_rho_Max": gg["Feature_Agreement_rho"].max(),
            "Extra_Drop_Min": gg["Extra_Delta_OriginalPredProb_Mean"].min(),
            "Extra_Drop_Max": gg["Extra_Delta_OriginalPredProb_Mean"].max()})
assoc_df = pd.DataFrame(assoc_rows)
cross_df = pd.DataFrame(cross_rows)
assoc_df.to_csv(OUTPUT_DIR / "L3_agreement_faithfulness_association.csv", index=False)
cross_df.to_csv(OUTPUT_DIR / "L3_agreement_faithfulness_crosstab.csv", index=False)
print("\nCross-table (agreement significant after BH correction x Monte Carlo lower-tail criterion: extra-drop 2.5th percentile > 0; determinable combinations only):")
print(cross_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nCombination-level association (Spearman, descriptive):")
print(assoc_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for k in K_LIST:
    c = cross_df[(cross_df["K"] == k) & (cross_df["Scope"] == "All")].iloc[0]
    n_sig = c["Agreement_Sig_and_Q025_Positive"] + c["Agreement_Sig_and_Q025_NotPositive"]
    print(f"k={k} (determinable {c['N_Complete']}/{c['N_Combos']}): of the {n_sig} combinations with significant agreement, lower-tail criterion positive (2.5th percentile > 0) {c['Agreement_Sig_and_Q025_Positive']}, "
          f"non-positive (decoupled) {c['Agreement_Sig_and_Q025_NotPositive']}; mean extra drop > 0 {c['Mean_Extra_Drop_Positive']}/{c['N_Complete']}; "
          f"sensitivity (repetitions with Random-k identical to Top-k removed): > 0 {c['Sens_Agreement_Sig_and_Q025NI_Positive']}, "
          f"non-positive {c['Sens_Agreement_Sig_and_Q025NI_NotPositive']} (determinable {c['Sens_N_Complete']})")

# ============================== Step 4: stability difference ==============================
print("\n" + "=" * 100)
print(f"Step 4: new analysis 2: difference in explanation stability between local input perturbations and training seeds, total {elapsed()}")
print("=" * 100)
METRICS = [("Feature", "Feature_Spearman_Mean", "Feature_Spearman_vs_42"),
           ("Temporal", "Temporal_Spearman_Mean", "Temporal_Spearman_vs_42"),
           ("Prediction", "Pred_Agreement_Mean", "Prediction_Agreement_vs_42")]
def build_stab(exclude_flagged):
    rows = []
    for d, m in ALL_COMBOS:
        nz = noise[(noise["Dataset"] == d) & (noise["Model"] == m)]
        sd = seed[(seed["Dataset"] == d) & (seed["Model"] == m)]
        alt = sd[(sd["Seed"] != PRIMARY_SEED)]
        alt_ok = alt[~alt["SingleClass"]] if exclude_flagged else alt
        ok5 = sd[~sd["SingleClass"]] if exclude_flagged else sd
        row = {"Market": market_of(d), "Dataset": d, "Model": m,
               "Seed_Reference": sd["Seed_Reference"].iloc[0] if len(sd) else "missing",
               "In_Stability_Main": (d, m) not in SEPARATE_REF_COMBOS,
               "Seed42_MacroF1": float(sd.loc[sd["Seed"] == PRIMARY_SEED, "MacroF1"].iloc[0]) if (sd["Seed"] == PRIMARY_SEED).any() else np.nan,
               "N_Other_Seeds": len(alt), "N_Other_Seeds_SingleClass": int(alt["SingleClass"].sum()),
               "N_Other_Seeds_Used": len(alt_ok),
               "MacroF1_SD_Used": ok5["MacroF1"].std(ddof=1) if len(ok5) >= 2 else np.nan,
               "MacroF1_Range_Used": (ok5["MacroF1"].max() - ok5["MacroF1"].min()) if len(ok5) >= 2 else np.nan}
        for lab, ncol, scol in METRICS:
            for lv in NOISE_LEVELS:
                v = nz.loc[nz["Noise_Percent_SD"] == lv, ncol]
                row[f"Noise{lv:g}_{lab}"] = float(v.iloc[0]) if len(v) else np.nan
            vals = alt_ok[scol].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            row[f"Seed_{lab}_Mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"Seed_{lab}_Min"] = float(vals.min()) if len(vals) else np.nan
            row[f"Seed_{lab}_Max"] = float(vals.max()) if len(vals) else np.nan
            row[f"Seed_{lab}_N"] = int(len(vals))
            row[f"Gap_{lab}"] = row[f"Noise{PRIMARY_NOISE:g}_{lab}"] - row[f"Seed_{lab}_Mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def gap_tests(stab_df, seed_set):
    main_df = stab_df[stab_df["In_Stability_Main"] & (stab_df["N_Other_Seeds_Used"] > 0)]
    rows = []
    for scope, g in [("All", main_df), ("China", main_df[main_df["Market"] == "China"]), ("US", main_df[main_df["Market"] == "US"])]:
        for lab, _, _ in METRICS:
            x = g[f"Noise{PRIMARY_NOISE:g}_{lab}"]
            y = g[f"Seed_{lab}_Mean"]
            ok = (~x.isna()) & (~y.isna())
            x, y = x[ok], y[ok]
            diff = x - y
            if len(diff) >= 5 and (diff != 0).any():
                try:
                    w_stat, w_p = wilcoxon(x, y, alternative="two-sided")
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
            else:
                w_stat, w_p = np.nan, np.nan
            rows.append({"Seed_Set": seed_set, "Scope": scope, "Metric": lab, "N_Combos": int(len(diff)),
                         "Noise_Mean": x.mean(), "Noise_Min": x.min(),
                         "Seed_Mean_of_Means": y.mean(), "Seed_Min_of_Means": y.min(),
                         "Gap_Mean": diff.mean(), "Gap_Median": diff.median(), "Gap_Min": diff.min(), "Gap_Max": diff.max(),
                         "N_Gap_Positive": int((diff > 0).sum()),
                         "Wilcoxon_Stat": w_stat, "Wilcoxon_p_Descriptive": w_p})
    return pd.DataFrame(rows)


stab = build_stab(False)          # main analysis: keep all seeds
stab_excl = build_stab(True)      # sensitivity analysis: remove seeds consistent with single-class prediction
stab.to_csv(OUTPUT_DIR / "L4_stability_gap_by_combo.csv", index=False)
stab_excl.to_csv(OUTPUT_DIR / "L4_stability_gap_by_combo_exclude_flagged.csv", index=False)
_show = ["Market", "Dataset", "Model", "In_Stability_Main", "N_Other_Seeds_Used"]
for lab, _, _ in METRICS:
    _show += [f"Noise{PRIMARY_NOISE:g}_{lab}", f"Seed_{lab}_Mean", f"Seed_{lab}_Min", f"Gap_{lab}"]
print(f"Main analysis by combination (noise at {PRIMARY_NOISE:g}% SD; seeds are the mean and minimum of all other 4 seeds relative to seed 42; "
      f"gap = noise − seed mean):")
print(stab[_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
_excl = stab[~stab["In_Stability_Main"]]
print("\nCombinations excluded from the main stability analysis:")
for _, r in _excl.iterrows():
    print(f"  {r['Dataset']} / {r['Model']}: the five-seed reference model is seed 42 retrained under the reconstructed configuration, which differs from the noise-experiment reference")
_flag = stab[stab["N_Other_Seeds_SingleClass"] > 0]
for _, r in _flag.iterrows():
    print(f"  Contains seeds consistent with single-class prediction (kept in the main analysis): {r['Dataset']} / {r['Model']}, {r['N_Other_Seeds_SingleClass']}")
_chg = stab_excl[stab_excl["N_Other_Seeds_SingleClass"] > 0]
if len(_chg):
    print("\nCombinations whose values change in the sensitivity analysis (seeds consistent with single-class prediction removed; combinations with all seeds removed are not counted in the sensitivity analysis):")
    print(_chg[_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

test_df = pd.concat([gap_tests(stab, "Main_Include_All"), gap_tests(stab_excl, "Sensitivity_Exclude_Flagged")], ignore_index=True)
test_df.to_csv(OUTPUT_DIR / "L4_stability_gap_summary.csv", index=False)
print(f"\nStability difference summary (two versions; Wilcoxon for descriptive reference only, combinations are not independent):")
print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# Prediction level: similar performance but different per-sample predictions
perf = seed[(seed["Seed"] != PRIMARY_SEED)].copy()
perf = perf.merge(seed[seed["Seed"] == PRIMARY_SEED][["Dataset", "Model", "MacroF1"]].rename(columns={"MacroF1": "Seed42_MacroF1"}),
                  on=["Dataset", "Model"], how="left")
perf["Abs_Delta_MacroF1_vs_42"] = (perf["MacroF1"] - perf["Seed42_MacroF1"]).abs()
perf["Disagreement_vs_42"] = 1 - perf["Prediction_Agreement_vs_42"]
perf.insert(0, "Market", perf["Dataset"].map(market_of))
perf["In_Stability_Main"] = [(d, m) not in SEPARATE_REF_COMBOS for d, m in zip(perf["Dataset"], perf["Model"])]
perf.to_csv(OUTPUT_DIR / "L5_seed_performance_vs_agreement.csv", index=False)
pm = perf[perf["In_Stability_Main"]]
pm_excl = pm[~pm["SingleClass"]]
print(f"\nSensitivity (seeds consistent with single-class prediction removed, {len(pm_excl)}): median |ΔMacro-F1| {pm_excl['Abs_Delta_MacroF1_vs_42'].median():.4f}, "
      f"median disagreement {pm_excl['Disagreement_vs_42'].median():.4f}")
print(f"Prediction level (main analysis, all {len(pm)} seed models): median |ΔMacro-F1| vs seed 42 {pm['Abs_Delta_MacroF1_vs_42'].median():.4f}, "
      f"max {pm['Abs_Delta_MacroF1_vs_42'].max():.4f}; median disagreement {pm['Disagreement_vs_42'].median():.4f}, "
      f"range {pm['Disagreement_vs_42'].min():.4f}–{pm['Disagreement_vs_42'].max():.4f}")
for mk in ["China", "US"]:
    g = pm[pm["Market"] == mk]
    if len(g):
        print(f"  {mk}: {len(g)} seed models, median |ΔMacro-F1| {g['Abs_Delta_MacroF1_vs_42'].median():.4f}, "
              f"prediction agreement {g['Prediction_Agreement_vs_42'].min():.4f}–{g['Prediction_Agreement_vs_42'].max():.4f}")

# ============================== Step 5: paper summary table ==============================
print("\n" + "=" * 100)
print(f"Step 5: paper summary table, total {elapsed()}")
print("=" * 100)
_fk = af[af["K"] == PRIMARY_K][["Dataset", "Model", "Extra_Delta_OriginalPredProb_Mean", "Extra_Delta_OriginalPredProb_Q025"]].rename(
    columns={"Extra_Delta_OriginalPredProb_Mean": f"ExtraDrop_k{PRIMARY_K}", "Extra_Delta_OriginalPredProb_Q025": f"ExtraDrop_k{PRIMARY_K}_Q025"})
_npos = af.groupby(["Dataset", "Model"]).agg(N_k_Q025_Positive=("Extra_Drop_Q025_Positive", "sum"),
                                             N_k_Valid=("Faith_State", lambda v: int((v != "undetermined").sum()))).reset_index()
summary = (src_df[["Market", "Dataset", "Model", "Seed42_MacroF1"]]
           .merge(feat[["Dataset", "Model", "Feature_Agreement_rho"]], on=["Dataset", "Model"], how="left")
           .merge(_bh_f, on=["Dataset", "Model"], how="left")
           .merge(tim[["Dataset", "Model", "Temporal_Agreement_rho"]], on=["Dataset", "Model"], how="left")
           .merge(_bh_t, on=["Dataset", "Model"], how="left")
           .merge(_fk, on=["Dataset", "Model"], how="left")
           .merge(_npos, on=["Dataset", "Model"], how="left")
           .merge(stab[["Dataset", "Model", "In_Stability_Main", "N_Other_Seeds_Used",
                        f"Noise{PRIMARY_NOISE:g}_Feature", "Seed_Feature_Mean", "Seed_Feature_Min",
                        f"Noise{PRIMARY_NOISE:g}_Temporal", "Seed_Temporal_Mean",
                        f"Noise{PRIMARY_NOISE:g}_Prediction", "Seed_Prediction_Mean", "MacroF1_SD_Used"]],
                  on=["Dataset", "Model"], how="left"))
summary.to_csv(OUTPUT_DIR / "L6_crossmarket_summary_table.csv", index=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"Table note: p-values of IG-KernelSHAP agreement are BH-corrected separately for features and time within the 15 combinations; extra drop is the additional drop in the original predicted-class probability of Top-k relative to Random-k at k={PRIMARY_K}; "
      f"N_k_Q025_Positive / N_k_Valid is the number of k=1/3/5 with a positive Monte Carlo lower-tail criterion / the number determinable; noise is {PRIMARY_NOISE:g}% of the training-set SD"
      f" (IG on 200 fixed samples); seeds are the mean of all other 4 seeds relative to seed 42 (IG on 500 fixed samples; sensitivity results without seeds consistent with single-class prediction are in L4).")

# ============================== Step 6: figures ==============================
print("\n" + "=" * 100)
print(f"Step 6: plotting ({', '.join(FIG_FORMATS)}, {FIG_DPI} dpi), total {elapsed()}")
print("=" * 100)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
MARKET_COLOR = {"China": "#1f5fa8", "US": "#c8501e"}
MODEL_MARKER = {"GRU": "o", "BiLSTM": "s", "LSTM": "^"}


def legend_handles():
    hs = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor=c, markersize=5, label=k)
          for k, c in MARKET_COLOR.items()]
    hs += [Line2D([0], [0], marker=mk, color="w", markerfacecolor="white", markeredgecolor="black", markersize=5, label=k)
           for k, mk in MODEL_MARKER.items()]
    return hs


# Figure 1: agreement and perturbation extra drop
fig1_skipped, fig1_undetermined = [], []
fig, axes = plt.subplots(1, len(K_LIST), figsize=(7.2, 2.6), sharey=False)
for ax, k in zip(axes, K_LIST):
    g = af[af["K"] == k]
    for _, r in g.iterrows():
        c = MARKET_COLOR[r["Market"]]
        x, y = float(r["Feature_Agreement_rho"]), float(r["Extra_Delta_OriginalPredProb_Mean"])
        if not (np.isfinite(x) and np.isfinite(y)):
            fig1_skipped.append(f"k={k} {r['Dataset']} {r['Model']}")
            continue
        q1, q2 = float(r["Extra_Delta_OriginalPredProb_Q025"]), float(r["Extra_Delta_OriginalPredProb_Q975"])
        if np.isfinite(q1) and np.isfinite(q2):
            ax.errorbar(x, y, yerr=[[max(y - q1, 0)], [max(q2 - y, 0)]], fmt="none", ecolor=c, elinewidth=0.6, capsize=1.5)
        if r["Faith_State"] == "undetermined":
            # Undetermined: grey x, distinguished from the open marker for "non-positive"
            fig1_undetermined.append(f"k={k} {r['Dataset']} {r['Model']}")
            ax.plot(x, y, marker="x", markersize=4.5, color="grey", markeredgewidth=0.8, linestyle="none")
        else:
            face = c if r["Faith_State"] == "positive" else "white"
            ax.plot(x, y, marker=MODEL_MARKER[r["Model"]], markersize=4.5, markerfacecolor=face, markeredgecolor=c,
                    markeredgewidth=0.8, linestyle="none")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    a = assoc_df[(assoc_df["K"] == k) & (assoc_df["Scope"] == "All")].iloc[0]
    ax.set_title(f"k = {k}  (Spearman r = {a['Spearman_FeatureAgreement_vs_ExtraDrop']:.2f})", fontsize=8)
    ax.set_xlabel("Feature agreement, IG vs KernelSHAP (ρ)", fontsize=7)
    if k == K_LIST[0]:
        ax.set_ylabel("Extra drop in predicted-class probability\n(Top-k minus Random-k)")
axes[-1].legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7)
_note = ("Filled markers: Monte Carlo lower-tail criterion met (2.5% quantile > 0); open: not met. "
         "Error bars: 2.5%–97.5% Monte Carlo range (not a confidence interval).")
if fig1_undetermined:
    _note += " Grey x: criterion undetermined (missing quantile)."
fig.text(0.0, -0.06, _note, fontsize=6.5)
print(f"  Figure 1: points not drawn because agreement or extra drop is missing: {len(fig1_skipped)}" + (f" ({'; '.join(fig1_skipped)})" if fig1_skipped else "") +
      f"; points with undetermined lower-tail criterion drawn as grey x: {len(fig1_undetermined)}" + (f" ({'; '.join(fig1_undetermined)})" if fig1_undetermined else ""))
fig.tight_layout()
save_fig(fig, "Figure1_agreement_vs_faithfulness")

# Figure 2: noise stability and cross-seed stability
fig2_df = stab[stab["In_Stability_Main"] & (stab["N_Other_Seeds_Used"] > 0)].reset_index(drop=True)
labels = [f"{r['Dataset']} {r['Model']}" for _, r in fig2_df.iterrows()]
ypos = np.arange(len(fig2_df))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.22 * len(fig2_df) + 1.2), sharey=True)
for ax, (lab, _, _) in zip(axes, METRICS):
    for yi, (_, r) in zip(ypos, fig2_df.iterrows()):
        c = MARKET_COLOR[r["Market"]]
        ax.plot([r[f"Seed_{lab}_Min"], r[f"Seed_{lab}_Max"]], [yi, yi], color=c, linewidth=1.0)
        ax.plot(r[f"Seed_{lab}_Mean"], yi, marker="o", markersize=4, markerfacecolor=c, markeredgecolor=c, linestyle="none")
        ax.plot(r[f"Noise{PRIMARY_NOISE:g}_{lab}"], yi, marker="D", markersize=4, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=0.8, linestyle="none")
    ax.set_title({"Feature": "Feature ranking (ρ)", "Temporal": "Temporal ranking (ρ)",
                  "Prediction": "Prediction agreement"}[lab], fontsize=8)
    ax.set_xlim(min(0.0, np.nanmin(fig2_df[f"Seed_{lab}_Min"]) - 0.05), 1.02)
    ax.grid(axis="x", linewidth=0.3, color="#dddddd")
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(labels, fontsize=7)
h = [Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="black", markersize=5,
            label=f"Input noise ({PRIMARY_NOISE:g}% SD)"),
     Line2D([0], [0], marker="o", color=MARKET_COLOR["China"], markersize=4, label="Seeds, China (mean, min–max)"),
     Line2D([0], [0], marker="o", color=MARKET_COLOR["US"], markersize=4, label="Seeds, US (mean, min–max)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_fig(fig, "Figure2_noise_vs_seed_stability")

# Figure 3: similar performance but different per-sample predictions
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
for _, r in fig2_df.iterrows():
    c = MARKET_COLOR[r["Market"]]
    ax.plot(r["MacroF1_SD_Used"], r["Seed_Prediction_Mean"], marker=MODEL_MARKER[r["Model"]], markersize=5,
            markerfacecolor=c, markeredgecolor=c, linestyle="none")
ax.set_xlabel("SD of Macro-F1 across seeds")
ax.set_ylabel("Mean prediction agreement with seed 42")
ax.set_title("(a) Combination level", fontsize=8)
ax = axes[1]
for _, r in pm.iterrows():
    c = MARKET_COLOR[r["Market"]]
    if r["SingleClass"]:
        ax.plot(r["Abs_Delta_MacroF1_vs_42"], r["Disagreement_vs_42"], marker="x", markersize=4,
                color=c, markeredgewidth=0.8, linestyle="none")
    else:
        ax.plot(r["Abs_Delta_MacroF1_vs_42"], r["Disagreement_vs_42"], marker=MODEL_MARKER[r["Model"]], markersize=4,
                markerfacecolor="white", markeredgecolor=c, markeredgewidth=0.8, linestyle="none")
if pm["SingleClass"].any():
    ax.text(0.98, 0.02, "x: consistent with single-class prediction", transform=ax.transAxes, ha="right", fontsize=6)
ax.set_xlabel("|Macro-F1 difference| vs seed 42")
ax.set_ylabel("Share of test predictions that differ")
ax.set_title("(b) Individual seed models", fontsize=8)
axes[1].legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7)
fig.tight_layout()
save_fig(fig, "Figure3_performance_vs_prediction_agreement")

# ============================== Step 7: conclusion check ==============================
print("\n" + "=" * 100)
print(f"Step 7: conclusion check (determined by actual results, not preset), total {elapsed()}")
print("=" * 100)
_f = bh[bh["Family"] == "Feature_IG_vs_KernelSHAP"]
_t = bh[bh["Family"] == "Temporal_IG_vs_KernelSHAP"]
print(f"1. Explanation agreement: features {int(_f['Significant_within_family'].sum())}/15, time {int(_t['Significant_within_family'].sum())}/15 combinations positive and significant after BH correction")
for k in K_LIST:
    c = cross_df[(cross_df["K"] == k) & (cross_df["Scope"] == "All")].iloc[0]
    a = assoc_df[(assoc_df["K"] == k) & (assoc_df["Scope"] == "All")].iloc[0]
    print(f"2. k={k}: combinations with significant agreement but a non-positive lower-tail criterion {c['Agreement_Sig_and_Q025_NotPositive']}"
          f" (sensitivity {c['Sens_Agreement_Sig_and_Q025NI_NotPositive']}); "
          f"combination-level Spearman r between agreement and extra drop = {a['Spearman_FeatureAgreement_vs_ExtraDrop']:.3f} (p = {a['p_FeatureAgreement']:.3f}, n = {a['N_Combos']})")
for lab, _, _ in METRICS:
    t = test_df[(test_df["Seed_Set"] == "Main_Include_All") & (test_df["Scope"] == "All") & (test_df["Metric"] == lab)].iloc[0]
    t2 = test_df[(test_df["Seed_Set"] == "Sensitivity_Exclude_Flagged") & (test_df["Scope"] == "All") & (test_df["Metric"] == lab)].iloc[0]
    print(f"3. {lab}: of {t['N_Combos']} combinations in the main analysis, noise stability higher than cross-seed stability in {t['N_Gap_Positive']}; "
          f"median gap {t['Gap_Median']:.3f} (range {t['Gap_Min']:.3f}–{t['Gap_Max']:.3f}); descriptive Wilcoxon p = {t['Wilcoxon_p_Descriptive']:.4g}")
    print(f"   Sensitivity (seeds consistent with single-class prediction removed) {lab}: of {t2['N_Combos']} combinations, positive gap in {t2['N_Gap_Positive']}, median gap {t2['Gap_Median']:.3f}")
print("Note: item 2 assesses whether agreement and faithfulness are decoupled, item 3 whether the stability difference exists; rely on the actual numbers above.")

print(f"\nOutput files under ./{OUTPUT_DIR.as_posix()}/: L1_combination_sources.csv, L2_bh_fdr.csv, "
      f"L3_agreement_faithfulness_by_combo.csv, L3_agreement_faithfulness_crosstab.csv, L3_agreement_faithfulness_association.csv, "
      f"L4_stability_gap_by_combo.csv, L4_stability_gap_by_combo_exclude_flagged.csv, L4_stability_gap_summary.csv, L5_seed_performance_vs_agreement.csv, "
      f"L6_crossmarket_summary_table.csv, and Figures 1–3 in {len(FIG_FORMATS)} formats each")
print(f"stage6_cell_L_data_crossmarket_V5 finished, total time {elapsed()}")
