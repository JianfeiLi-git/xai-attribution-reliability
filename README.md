# Attribution Reliability in Financial Time-Series Classification

Code, trained models and result tables for the paper:

> *Attribution Reliability in Financial Time-Series Classification: Cross-Market Evidence from Multidimensional XAI Validation* (submitted to PeerJ Computer Science).

Archived release: https://doi.org/10.5281/zenodo.23697967

## Repository structure

| Folder | Content |
|---|---|
| `01_data_preparation/` | Data quality screening, feature engineering, first-stage feature selection, chronological split and standardization (`china_medical/`, `china_power/`, `us/`) |
| `02_feature_selection/` | Second-stage feature screening with a Transformer-based permutation importance model |
| `03_model_training/` | Training of the primary seed-42 GRU, BiLSTM and LSTM models |
| `04_xai_analysis/` | Attribution analyses (IG, KernelSHAP, perturbation tests, sanity checks, robustness, multi-seed stability) and figure/table generation |
| `05_revision_analysis/` | Block bootstrap of predictive performance and corrected Medical-2 multi-seed retraining |
| `models/` | All trained model weights (PyTorch `state_dict`) |
| `results/` | Result tables (CSV) and figures produced by the analysis scripts |
| `data/stock_lists/` | Stock codes and date coverage of the five datasets |
| `environment/` | Python package versions used |
| `tools/` | Helper scripts used to assemble and check this repository |

## Datasets

| Paper name | Market | Sector | Data folder used in scripts |
|---|---|---|---|
| Medical-1 | China A-share | Healthcare (Grade A) | `XAI_cn/datasets/processed/dataset1_A_grade` |
| Medical-2 | China A-share | Healthcare (Grade A + B) | `XAI_cn/datasets/processed/dataset2_AB_grade` |
| Power | China A-share | Utilities (Grade A + B) | `XAI_cn/e3/datasets/processed/dataset2_AB_grade` |
| US-Med | United States | Healthcare | `xai_us/datasets/processed/dataset4_m` |
| US-Pow | United States | Utilities | `xai_us/datasets/processed/dataset5_p` |

All price data were obtained from the [Tushare](https://tushare.pro) financial data platform (2010-01-04 to 2025-11-14). Tushare data are provided under the platform's own terms, so raw price data are not redistributed here. The stock codes and date ranges in `data/stock_lists/` allow the datasets to be downloaded again with a Tushare account. A download script is included for the U.S. data (`01_data_preparation/us/us_01_download.py`); the Chinese data preparation starts from the downloaded daily-quote files (`tomAB-.csv` for healthcare, `toAB-.csv` for utilities).

## Workflow

Run the scripts in the order below. Scripts expect the working directory to contain the `XAI_cn/` and `xai_us/` data folders.

### 1. Data preparation (`01_data_preparation/`)

| Order | Script | Purpose |
|---|---|---|
| 1 | `china_medical/medical_01_date_filter.py` | Keep records from 2000-01-01 onward |
| 2 | `china_medical/medical_02_missing_value_check.py` | Missing-value check |
| 3 | `china_medical/medical_03_quality_diagnostics.py` | Diagnostic only: data quality, suspension and missing-day analysis (no files used later) |
| 4 | `china_medical/medical_04_quality_grading.py` | Quality scoring and Grade-A / Grade-B classification |
| 5 | `china_medical/medical_05_features_selection_split.py` | Feature engineering, first-stage feature selection, split and standardization (Medical-1 and Medical-2) |
| 1–5 | `china_power/power_01` to `power_05` | Same steps for the Power dataset |
| 1 | `us/us_01_download.py` | Download U.S. daily data (set your own Tushare token) |
| 2 | `us/us_02_feature_engineering.py` | Feature engineering |
| 3 | `us/us_03_feature_selection.py` | First-stage feature ranking |
| 4 | `us/us_04_split_standardize.py` | Split and standardization |
| 5 | `us/us_05_copy_to_datasets.py` | Copy processed U.S. data to the folder used for training |

### 2–5. Feature screening, training and analysis

| Step | Script | Produces (paper) |
|---|---|---|
| 2 | `02_feature_selection/transformer_permutation_importance.py` | Second-stage feature screening (Section 2.2) |
| 3 | `03_model_training/m_*.py`, `p_*.py`, `us_*.py` | Primary seed-42 models |
| 4 | `04_xai_analysis/xai_cell_A_V9.py` | Parameter randomization, IG attribution, temporal occlusion (Figure 4a, Table 4) |
| 4 | `04_xai_analysis/xai_cell_B_V6.py` | IG versus grouped KernelSHAP, perturbation-based behavioral relevance (Table 3, Figure 2) |
| 4 | `04_xai_analysis/xai_cell_C_V3.py` | Conventional baselines (Table 2, Supplementary Table S1) |
| 4 | `04_xai_analysis/xai_cell_D_V1.py` | Small input-perturbation robustness (Table 6, Figure 4c) |
| 4 | `04_xai_analysis/xai_cell_E_V2.py` | Label randomization sanity check (Table 5, Figure 4b) |
| 4 | `04_xai_analysis/xai_cell_F_V1.py` | Temporal KernelSHAP and block attribution mass (Table 4, Figure 3) |
| 4 | `04_xai_analysis/xai_cell_H_V1.py` | Benjamini–Hochberg correction of Spearman tests (Tables 3 and 4) |
| 4 | `04_xai_analysis/xai_cell_I_V2.py` | Supplementary Table S1 |
| 4 | `04_xai_analysis/xai_cell_G_V8.py` | Figures 2–4 (named Figure 1–3 in the script output) |
| 4 | `04_xai_analysis/xai_cell_J_V1.py`, `xai_cell_J_V2.py` | Five-seed stability (Supplementary Table S2) |
| 4 | `04_xai_analysis/xai_cell_K_V1.py` | Medical-2 GRU reconstructed-configuration sensitivity analysis (Supplementary Table S2) |
| 5 | `05_revision_analysis/stage5_cell_A_bootstrap_V4.py` | Paired moving-block bootstrap (Table 2 confidence intervals and p-values) |
| 5 | `05_revision_analysis/stage5_cell_B_retrain_med2_V2.py` | Medical-2 BiLSTM and LSTM additional seeds (Supplementary Table S2) |

Figure 1 of the paper is a conceptual diagram and is not produced by code.

## Notes on the code

1. **Origin of the scripts.** The scripts were originally run as Jupyter notebook cells. The data preparation notebooks were converted into the step scripts above, keeping only the cells that were run, in pipeline order. Cells with exploratory notes, font settings, unused market-index downloads and superseded download attempts were left out.
2. **Language.** Comments, docstrings and console messages were translated from Chinese into English for this repository, and Chinese output file names were renamed to English. The program logic was not changed: for every script, the Python syntax tree of the translated version was checked to be identical to the original, apart from the text of strings.
3. **Absolute paths.** Some scripts contain absolute paths of the original server (for example `/home/i/...`); adjust them to your own folders.

## Notes on provenance

1. **Seed-42 checkpoints.** Because of an implementation detail in the original training scripts (the best model state was stored by reference), the saved seed-42 checkpoints contain the weights from the epoch at which early stopping was triggered, not from the epoch with the best validation Macro-F1. All reported seed-42 results use these saved checkpoints. All additional-seed models were saved at the best validation epoch.
2. **Missing original training scripts.** The training scripts that produced the Medical-2 GRU, US-Med GRU and US-Pow GRU seed-42 checkpoints were not kept. `us_gru.py` is the available U.S. GRU script; it does not reproduce the saved U.S. GRU architecture. These three models are treated as reconstructed configurations in Supplementary Table S2.
3. **Second-stage feature screening.** `transformer_permutation_importance.py` was run separately for each Chinese dataset by changing the dataset setting in `main()`. The Transformer-based screening was not applied to the U.S. datasets; their 13 and 12 features were selected manually from the first-stage ranking (Section 2.2).
4. **Superseded multi-seed runs.**
   - `xai_cell_J_V1`: its Medical-1 BiLSTM seeds used an incorrect configuration and were retrained in `xai_cell_J_V2`. Its Medical-1 GRU and LSTM seeds were reused by `xai_cell_J_V2`.
   - `xai_cell_J_V2`: its Medical-2 BiLSTM and LSTM seeds used incorrect scheduler and early-stopping settings and were replaced by `stage5_cell_B_retrain_med2_V2`. Those rows in the J_V2 result files are not used in the paper, and the corresponding models are not included.
5. **cuDNN.** cuDNN could not be initialized on the server when `stage5_cell_B_retrain_med2_V2` was run, so that run used PyTorch's native CUDA RNN kernels.
6. **Logistic Regression baselines.** When refitted in `stage5_cell_A_bootstrap_V4`, the Medical-2 and US-Pow Logistic Regression baselines gave slightly different Macro-F1 values (0.3757 and 0.3580) from those reported in Table 2, because of numerical differences in the iterative solver. This is stated in the Table 2 note.

## Environment

Experiments were run on Ubuntu 22.04 with an NVIDIA GeForce RTX 3090 GPU (CUDA 12.1) and Python 3.10. Package versions are listed in `environment/versions.json` and `environment/requirements_freeze.txt`.

## License

Code is released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper (citation details will be added after publication) and the archived release of this repository (https://doi.org/10.5281/zenodo.23697967).
