# Attribution Reliability in Financial Time-Series Classification: Cross-Market Evidence from Multidimensional Explainable Artificial Intelligence Validation

Code, trained models and result tables for the paper submitted to PeerJ Computer Science.

Archived release: [https://doi.org/10.5281/zenodo.23789348](https://doi.org/10.5281/zenodo.23789348)

## Description

This repository contains the complete code and outputs for a study that evaluates whether attribution-based explanations of recurrent neural networks are reliable in financial time-series classification. Three recurrent architectures (gated recurrent unit (GRU), long short-term memory (LSTM) and bidirectional LSTM (BiLSTM)) were trained to classify next-day stock returns into three classes (down, hold, up) on five equity datasets from the Chinese A-share and U.S. markets. Their explanations were then evaluated along seven dimensions: cross-method agreement, perturbation-based behavioral relevance, parameter dependence, label dependence, temporal consistency, robustness to small input perturbations, and stability across independently trained models. Two cross-dimension analyses examine whether cross-method agreement corresponds to behavioral relevance and whether robustness to input perturbations corresponds to stability across training seeds.

The repository includes:

- data preparation and feature selection scripts;
- model training scripts;
- attribution and validation analysis scripts (Integrated Gradients (IG), grouped interventional KernelSHAP, perturbation tests, sanity checks, robustness and multi-seed stability);
- conventional baselines and the block bootstrap test;
- all trained model weights;
- result tables and figures of the analyses in `04_xai_analysis/` and `05_revision_analysis/`;
- console logs of the analyses in `06_stage6/`;
- the stock code lists and date ranges needed to rebuild the datasets.

## Dataset Information

| Paper name | Market        | Sector                   | Data folder used in scripts                      |
|------------|---------------|--------------------------|--------------------------------------------------|
| Medical-1  | China A-share | Healthcare (Grade A)     | `XAI_cn/datasets/processed/dataset1_A_grade`     |
| Medical-2  | China A-share | Healthcare (Grade A + B) | `XAI_cn/datasets/processed/dataset2_AB_grade`    |
| Power      | China A-share | Utilities (Grade A + B)  | `XAI_cn/e3/datasets/processed/dataset2_AB_grade` |
| US-Med     | United States | Healthcare               | `stage6/stage6_cell_A_data_preprocessing_V1/us_med` |
| US-Pow     | United States | Utilities                | `stage6/stage6_cell_A_data_preprocessing_V1/us_pow` |

- **Source:** all price data were obtained from the [Tushare](https://tushare.pro) financial data platform (2010-01-04 to 2025-11-14).
- **Content:** daily price and volume data for 24 (Medical-1), 79 (Medical-2) and 30 (Power) Chinese stocks, and 20 U.S. healthcare and 20 U.S. utility stocks.
- **Data split:** training from 2010-01-01 to 2022-12-31, validation from 2023-01-01 to 2024-06-30, and test from 2024-07-01 to 2025-11-14.
- **Availability:** Tushare data are provided under the platform's own terms, so raw price data are not redistributed here. The stock codes and date ranges in `data/stock_lists/` allow the datasets to be downloaded again with a Tushare account. A download script is included for the U.S. data (`01_data_preparation/us/us_01_download.py`). The Chinese data preparation starts from the downloaded daily-quote files (`tomAB-.csv` for healthcare, `toAB-.csv` for utilities).

## Code Information

| Folder                  | Content                                                                                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `01_data_preparation/`  | Data quality screening, feature engineering, first-stage feature selection, chronological split and standardization (`china_medical/`, `china_power/`, `us/`) |
| `02_feature_selection/` | Second-stage feature screening with a Transformer-based permutation importance model                                                                          |
| `03_model_training/`    | Training of the primary seed-42 GRU, BiLSTM and LSTM models                                                                                                   |
| `04_xai_analysis/`      | Attribution analyses (IG, KernelSHAP, perturbation tests, sanity checks, robustness, multi-seed stability) and figure/table generation                        |
| `05_revision_analysis/` | Block bootstrap of predictive performance and corrected Medical-2 multi-seed retraining                                                                       |
| `06_stage6/`            | Final analysis stage: reprocessing, retraining and full attribution validation of the U.S. datasets, corrected Chinese multi-seed IG, cross-market analyses, final figures and checks of all reported numbers |
| `models/`               | All trained model weights (PyTorch `state_dict`)                                                                                                              |
| `results/`              | Result tables (CSV) and figures produced by the scripts in `04_xai_analysis/` and `05_revision_analysis/`; console logs of `06_stage6/` in `results/stage6_logs/` |
| `data/stock_lists/`     | Stock codes and date coverage of the five datasets                                                                                                            |
| `environment/`          | Python package versions used                                                                                                                                  |
| `tools/`                | Helper scripts used to assemble and check this repository                                                                                                     |

### Notes on the code

1. **Origin of the scripts.** The scripts were originally run as Jupyter notebook cells. The data preparation notebooks were converted into the step scripts below, keeping only the cells that were run, in pipeline order. Cells with exploratory notes, font settings, unused market-index downloads and superseded download attempts were left out.
2. **Language.** Comments, docstrings and console messages were translated from Chinese into English for this repository, and Chinese output file names were renamed to English. The program logic was not changed: for every script, the Python syntax tree of the translated version was checked to be identical to the original, apart from the text of strings. The console logs of `06_stage6/` were translated in the same way; only the message text was changed, not the numbers.
3. **Absolute paths.** Some scripts contain absolute paths of the original server (for example `/home/i/...`); adjust them to your own folders. The `06_stage6/` scripts read and write folders under `stage6/` relative to the working directory.

## Requirements

- **Operating system:** Ubuntu 22.04 (other Linux systems should also work)
- **Hardware:** an NVIDIA GPU is recommended; experiments were run on an NVIDIA GeForce RTX 3090 (CUDA 12.1)
- **Python:** 3.10
- **Main libraries:** PyTorch (recurrent models), Captum (Integrated Gradients), scikit-learn (baselines and statistics), LightGBM and XGBoost (first-stage feature ranking), pandas, NumPy, SciPy, Matplotlib, and Tushare (data download)

The exact package versions are listed in `environment/versions.json` and `environment/requirements_freeze.txt`. To install them:

```
pip install -r environment/requirements_freeze.txt
```

A Tushare account and token are required to download the raw price data.

## Usage Instructions

1. Clone this repository and install the requirements.
2. Download the raw daily data from Tushare using the stock lists in `data/stock_lists/` (for the U.S. data, set your Tushare token in `01_data_preparation/us/us_01_download.py`).
3. Place the data so that the working directory contains the `XAI_cn/` data folders, and the U.S. feature files `us_medical_features_full.csv` and `us_power_features_full.csv` read by `06_stage6/stage6_A_preprocessing_V1.py`.
4. Adjust any absolute paths in the scripts to your own folders.
5. Run the scripts in the order below. To reproduce only the analyses without retraining, use the trained weights in `models/`.

### Step 1. Data preparation (`01_data_preparation/`)

| Order | Script                                                 | Purpose                                                                                                 |
|-------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 1     | `china_medical/medical_01_date_filter.py`              | Keep records from 2000-01-01 onward                                                                     |
| 2     | `china_medical/medical_02_missing_value_check.py`      | Missing-value check                                                                                     |
| 3     | `china_medical/medical_03_quality_diagnostics.py`      | Diagnostic only: data quality, suspension and missing-day analysis (no files used later)                |
| 4     | `china_medical/medical_04_quality_grading.py`          | Quality scoring and Grade-A / Grade-B classification                                                    |
| 5     | `china_medical/medical_05_features_selection_split.py` | Feature engineering, first-stage feature selection, split and standardization (Medical-1 and Medical-2) |
| 1–5   | `china_power/power_01` to `power_05`                   | Same steps for the Power dataset                                                                        |
| 1     | `us/us_01_download.py`                                 | Download U.S. daily data (set your own Tushare token)                                                   |
| 2     | `us/us_02_feature_engineering.py`                      | Feature engineering                                                                                     |
| 3     | `us/us_03_feature_selection.py`                        | First-stage feature ranking                                                                             |
| 4     | `us/us_04_split_standardize.py`                        | Split and standardization                                                                               |
| 5     | `us/us_05_copy_to_datasets.py`                         | Copy processed U.S. data to the folder used for training                                                |

For the U.S. datasets, the first-stage selection, split and standardization used in the paper are those of `06_stage6/stage6_A_preprocessing_V1.py` (Step 6), which supersedes U.S. steps 3–5 above.

### Steps 2–5. Feature screening, training and analysis

| Step | Script                                                       | Produces (paper)                                                                          |
|------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 2    | `02_feature_selection/transformer_permutation_importance.py` | Second-stage feature screening (Section 2.2)                                              |
| 3    | `03_model_training/m_*.py`, `p_*.py`, `us_*.py`              | Primary seed-42 models                                                                    |
| 4    | `04_xai_analysis/xai_cell_A_V9.py`                           | Parameter randomization, IG attribution, temporal occlusion (Figure 4a, Table 4)          |
| 4    | `04_xai_analysis/xai_cell_B_V6.py`                           | IG versus grouped KernelSHAP, perturbation-based behavioral relevance (Table 3, Figure 2) |
| 4    | `04_xai_analysis/xai_cell_C_V3.py`                           | Conventional baselines (Table 2, Supplementary Table S1)                                  |
| 4    | `04_xai_analysis/xai_cell_D_V1.py`                           | Small input-perturbation robustness (Table 6, Figure 4c)                                  |
| 4    | `04_xai_analysis/xai_cell_E_V2.py`                           | Label randomization sanity check (Table 5, Figure 4b)                                     |
| 4    | `04_xai_analysis/xai_cell_F_V1.py`                           | Temporal KernelSHAP and block attribution mass (Table 4, Figure 3)                        |
| 4    | `04_xai_analysis/xai_cell_H_V1.py`                           | Benjamini–Hochberg correction of Spearman tests (Tables 3 and 4)                          |
| 4    | `04_xai_analysis/xai_cell_I_V2.py`                           | Supplementary Table S1                                                                    |
| 4    | `04_xai_analysis/xai_cell_G_V8.py`                           | Figures 2–4 (named Figure 1–3 in the script output)                                       |
| 4    | `04_xai_analysis/xai_cell_J_V1.py`, `xai_cell_J_V2.py`       | Five-seed stability (Supplementary Table S2)                                              |
| 4    | `04_xai_analysis/xai_cell_K_V1.py`                           | Medical-2 GRU reconstructed-configuration sensitivity analysis (Supplementary Table S2)   |
| 5    | `05_revision_analysis/stage5_cell_A_bootstrap_V4.py`         | Paired moving-block bootstrap (Table 2 confidence intervals and p-values)                 |
| 5    | `05_revision_analysis/stage5_cell_B_retrain_med2_V2.py`      | Medical-2 BiLSTM and LSTM additional seeds (Supplementary Table S2)                       |

### Step 6. Final analysis stage (`06_stage6/`)

The U.S. results reported in the paper, the final figures and the final tables come from the scripts below, which supersede the U.S. parts of Steps 3–5 and `xai_cell_G_V8.py`. The Chinese results are still those of Steps 3–5; scripts K, L, M and N read them.

| Order | Script                               | Purpose (paper)                                                                                          |
|-------|--------------------------------------|----------------------------------------------------------------------------------------------------------|
| A     | `stage6_A_preprocessing_V1.py`       | U.S. label alignment, first-stage feature selection, split and standardization (Section 2.2, Table 1)   |
| B     | `stage6_B_screening_V3.py`           | Diagnostic run of the Transformer-based screening on the U.S. datasets (Section 2.2)                     |
| C     | `stage6_C_training_V2.py`            | U.S. seed-42 GRU, LSTM and BiLSTM models (Table 2)                                                       |
| D     | `stage6_D_baseline_V1.py`            | U.S. conventional baselines and paired moving-block bootstrap; BH correction across five datasets (Table 2, Supplementary Table S1) |
| E     | `stage6_E_attribution_V2.py`         | U.S. parameter randomization, IG attribution and temporal occlusion                                      |
| F     | `stage6_F_kernelshap_V4.py`          | U.S. grouped KernelSHAP and perturbation-based behavioral relevance                                      |
| G     | `stage6_G_temporal_V2.py`            | U.S. temporal KernelSHAP and block attribution mass                                                      |
| H     | `stage6_H_robustness_V4.py`          | U.S. small input-perturbation robustness                                                                 |
| I     | `stage6_I_labelrandom_V2.py`         | U.S. label randomization sanity check                                                                    |
| J     | `stage6_J_multiseed_V1.py`           | U.S. five-seed stability (Supplementary Table S2)                                                        |
| K     | `stage6_K_cnmultiseed_V7.py`         | Corrected IG of the Chinese five-seed analysis, without retraining (Supplementary Table S2)              |
| L     | `stage6_L_crossmarket_V5.py`         | Cross-market merging, BH correction and the two cross-dimension analyses (Sections 3.6 and 3.7)          |
| M     | `stage6_M_figures_V2.py`             | Figures 2–7 and Supplementary Figure S1                                                                  |
| N     | `stage6_N_paperstats_V5.py`          | Final read-only check of Tables 1–6, Supplementary Tables S1–S2 and the numbers in the text             |

The output files of `06_stage6/` are not included in this repository; their console logs are in `results/stage6_logs/`.

Figure 1 of the paper is a conceptual diagram and is not produced by code.

## Methodology

1. **Data screening.** Chinese stocks were graded into Grade A and Grade B using data completeness, listing history, price-error checks and a quality score.
2. **Feature selection.** A pool of 50 candidate features was reduced in two stages: an ensemble ranking (LightGBM, XGBoost, Random Forest, F-statistic, mutual information and Pearson correlation), followed by Transformer-based permutation importance for the Chinese datasets. For the U.S. datasets, the Transformer-based screening did not yield a usable ranking, so the first-stage features were used directly.
3. **Labels and sequences.** Next-day returns were divided into down, hold and up classes using training-set tercile thresholds. Inputs are 60-trading-day sliding windows.
4. **Data split.** The data were split chronologically into training, validation and test periods, and standardized using training-set statistics only.
5. **Models.** GRU, LSTM and BiLSTM models were trained with seed 42 (primary) and four additional seeds (123, 777, 0 and 2024).
6. **Evaluation.** Macro-F1 (primary) and accuracy were computed on the test set. Logistic Regression, Extra Trees and a naive momentum classifier were used as conventional baselines, and the best recurrent model was compared with the strongest baseline using a paired moving-block bootstrap over test trading days. Because the model pairs were selected by test-set Macro-F1, these comparisons are exploratory.
7. **Attribution validation.** Explanations were evaluated for cross-method agreement (IG versus grouped KernelSHAP), behavioral relevance (top-k perturbation), parameter and label dependence (randomization sanity checks), temporal consistency (temporal IG, temporal KernelSHAP and occlusion), robustness (Gaussian input noise) and stability across training seeds. Two cross-dimension analyses compared cross-method agreement with behavioral relevance, and input-perturbation robustness with training-seed stability.

### Notes on provenance

1. **Seed-42 checkpoints.** Because of an implementation detail in the original training scripts (the best model state was stored by reference), the saved seed-42 checkpoints contain the weights from the epoch at which early stopping was triggered, not from the epoch with the best validation Macro-F1. All reported seed-42 results use these saved checkpoints. The U.S. seed-42 models retrained in `06_stage6/stage6_C_training_V2.py` follow the same saving rule. All additional-seed models were saved at the best validation epoch.
2. **Missing original training script.** The training script that produced the Medical-2 GRU seed-42 checkpoint was not kept. This model is treated as a reconstructed configuration in Supplementary Table S2. All U.S. seed-42 models were retrained in `06_stage6/stage6_C_training_V2.py`, which supersedes the U.S. training scripts in `03_model_training/`.
3. **Second-stage feature screening.** `transformer_permutation_importance.py` was run separately for each Chinese dataset by changing the dataset setting in `main()`. On the U.S. datasets, the probe model of the Transformer-based screening learned no usable signal (`06_stage6/stage6_B_screening_V3.py`), so the first-stage features (22 for US-Med and 19 for US-Pow) were used directly (Section 2.2).
4. **Superseded multi-seed runs.**
   - `xai_cell_J_V1`: its Medical-1 BiLSTM seeds used an incorrect configuration and were retrained in `xai_cell_J_V2`. Its Medical-1 GRU and LSTM seeds were reused by `xai_cell_J_V2`.
   - `xai_cell_J_V2`: its Medical-2 BiLSTM and LSTM seeds used incorrect scheduler and early-stopping settings and were replaced by `stage5_cell_B_retrain_med2_V2`. Those rows in the J_V2 result files are not used in the paper, and the corresponding models are not included. Its U.S. seeds were superseded by `06_stage6/stage6_J_multiseed_V1.py`; the U.S. rows in the J_V2 result files are not used in the paper, and the corresponding models were removed.
   - Chinese five-seed IG: the original code left between-layer RNN dropout active during IG. `06_stage6/stage6_K_cnmultiseed_V7.py` recomputed IG for the same models with dropout-free inference copies; the paper uses these corrected results.
   - U.S. results in `04_xai_analysis/` and `05_revision_analysis/` result files are superseded by `06_stage6/`.
5. **cuDNN.** cuDNN could not be initialized on the server when `stage5_cell_B_retrain_med2_V2` and the `06_stage6/` scripts were run, so those runs used PyTorch's native CUDA RNN kernels.
6. **Logistic Regression baseline.** When refitted in `stage5_cell_A_bootstrap_V4`, the Medical-2 Logistic Regression baseline gave a slightly different Macro-F1 value (0.3757) from that reported in Table 2, because of numerical differences in the iterative solver. This is stated in the Table 2 note.

## Citations

If you use this code, please cite the paper (citation details will be added after publication) and the archived release of this repository ([https://doi.org/10.5281/zenodo.23789348](https://doi.org/10.5281/zenodo.23789348)).

Data source:

- Tushare financial data platform. https://tushare.pro

Main methods used:

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML 2017, PMLR 70, 3319–3328.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017, 4765–4774.
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. NeurIPS 2018, 9525–9536.
- Kokhlikyan, N., et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. arXiv:2009.07896.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3), 1217–1241.

## License & Contribution Guidelines

Code is released under the MIT License (see `LICENSE`). Tushare data are not covered by this license and remain subject to the Tushare terms of use.

Questions, bug reports and suggestions are welcome through GitHub Issues. Contributions can be submitted as pull requests; please describe the change and the scripts it affects.