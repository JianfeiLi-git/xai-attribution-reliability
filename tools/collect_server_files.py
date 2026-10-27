# -*- coding: utf-8 -*-
"""
collect_server_files.py

Run once on the original server. It copies, from fixed paths only:
  1. all model checkpoints used in the paper  -> models/
  2. result tables (CSV/JSON) and figures     -> results/
  3. stock code lists of the five datasets    -> data/stock_lists/
  4. software environment information         -> environment/
and finally scans the whole repository for leftover Tushare tokens.

Nothing is deleted or changed in the original folders.
Usage (from the repository root):
    python tools/collect_server_files.py --project-root /home/i/2026_0712/stock_i
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--project-root", default="/home/i/2026_0712/stock_i")
args = parser.parse_args()
ROOT = Path(args.project_root)
CN = ROOT / "XAI_cn"
US = ROOT / "xai_us"
PJCS = ROOT / "xai_PJCS"

missing = []
copied = []


def copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append((str(dst.relative_to(REPO)), src.stat().st_size))


def copy_glob(src_dir, pattern, dst_dir, exclude=()):
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        missing.append(str(src_dir))
        return
    files = sorted(p for p in src_dir.glob(pattern)
                   if p.is_file() and not any(x in p.name for x in exclude))
    if not files:
        missing.append(f"{src_dir}/{pattern} (no files)")
    for p in files:
        copy_file(p, Path(dst_dir) / p.name)


# 1. Seed-42 checkpoints (primary models)
SEED42 = {
    "Medical-1": [CN / "models_XAI/rnn_enhanced_dataset2/gru_dataset1_SELECTED_13_seed42.pt",
                  CN / "models_XAI/rnn_enhanced_dataset2/bilstm_dataset1_SELECTED_13_seed42.pt",
                  CN / "models_XAI/lstm_enhanced/lstm_dataset1_SELECTED_13_seed42.pt"],
    "Medical-2": [CN / "models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_14_seed42.pt",
                  CN / "models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_14_seed42.pt",
                  CN / "models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_14_seed42.pt"],
    "Power": [CN / "e3/models_XAI/rnn_enhanced_dataset2/gru_dataset2_SELECTED_12_seed42.pt",
              CN / "e3/models_XAI/rnn_enhanced_dataset2/bilstm_dataset2_SELECTED_12_seed42.pt",
              CN / "e3/models_XAI/lstm_enhanced/lstm_dataset2_SELECTED_12_seed42.pt"],
    "US-Med": [US / "models_XAI/rnn_models_us/gru_dataset4_m_SELECTED_13_seed42.pt",
               US / "models_XAI/rnn_bilstm_us_60/bilstm_dataset4_m_SELECTED_13_seed42.pt",
               US / "models_XAI/lstm_us_60/lstm_dataset4_m_SELECTED_13_seed42.pt"],
    "US-Pow": [US / "models_XAI/rnn_models_us/gru_dataset5_p_SELECTED_12_seed42.pt",
               US / "models_XAI/rnn_bilstm_us_60/bilstm_dataset5_p_SELECTED_12_seed42.pt",
               US / "models_XAI/lstm_us_60/lstm_dataset5_p_SELECTED_12_seed42.pt"],
}
for ds, paths in SEED42.items():
    for p in paths:
        copy_file(p, REPO / "models/seed42" / ds / p.name)

# 2. Additional-seed models
# J_V2: Medical-2 BiLSTM/LSTM excluded (superseded by stage5_cell_B_retrain_med2_V2)
copy_glob(ROOT / "xai_cell_J_V2/models", "*.pt", REPO / "models/multiseed_J_V2",
          exclude=("Medical-2_BiLSTM", "Medical-2_LSTM"))
copy_glob(ROOT / "xai_cell_K_V1/models", "*.pt", REPO / "models/multiseed_K_V1_medical2_gru_reconstructed")
copy_glob(PJCS / "stage5/cell_B_retrain_med2_V2/models", "*.pt", REPO / "models/multiseed_stage5_B_V2_medical2")

# 3. Result tables and figures
for cell in ["xai_cell_A_V9", "xai_cell_B_V6", "xai_cell_C_V3", "xai_cell_D_V1", "xai_cell_E_V2",
             "xai_cell_F_V1", "xai_cell_H_V1", "xai_cell_I_V2"]:
    copy_glob(ROOT / "0924" / cell, "*.csv", REPO / "results" / cell)
copy_glob(ROOT / "0924/xai_cell_G_V8", "*.png", REPO / "results/xai_cell_G_V8")
for cell in ["xai_cell_J_V2", "xai_cell_K_V1"]:
    copy_glob(ROOT / cell, "*.csv", REPO / "results" / cell)
for cell in ["cell_A_bootstrap_V4", "cell_B_retrain_med2_V2"]:
    copy_glob(PJCS / "stage5" / cell, "*.csv", REPO / "results" / f"stage5_{cell}")
    copy_glob(PJCS / "stage5" / cell, "*.json", REPO / "results" / f"stage5_{cell}")

# 4. Stock code lists (identifiers and date coverage only, no price data)
DATA = {
    "Medical-1": CN / "datasets/processed/dataset1_A_grade",
    "Medical-2": CN / "datasets/processed/dataset2_AB_grade",
    "Power": CN / "e3/datasets/processed/dataset2_AB_grade",
    "US-Med": US / "datasets/processed/dataset4_m",
    "US-Pow": US / "datasets/processed/dataset5_p",
}
out_dir = REPO / "data/stock_lists"
out_dir.mkdir(parents=True, exist_ok=True)
for ds, d in DATA.items():
    parts = []
    for split in ["train", "val", "test"]:
        f = d / f"{split}_X.csv"
        if not f.is_file():
            missing.append(str(f))
            continue
        parts.append(pd.read_csv(f, usecols=["stock_code", "trade_date"]))
    if parts:
        df = pd.concat(parts)
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
        lst = (df.groupby("stock_code")["trade_date"]
               .agg(first_date="min", last_date="max", n_rows="count").reset_index())
        lst.to_csv(out_dir / f"{ds}_stock_list.csv", index=False)
        print(f"{ds}: {len(lst)} stocks")

# 5. Software environment
env = REPO / "environment"
env.mkdir(exist_ok=True)
info = {"python": sys.version, "platform": platform.platform()}
try:
    import torch
    info.update(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as e:
    info["torch"] = f"not available: {e}"
for mod in ["numpy", "pandas", "sklearn", "scipy", "captum", "lightgbm", "xgboost", "matplotlib", "tushare"]:
    try:
        info[mod] = __import__(mod).__version__
    except Exception:
        info[mod] = "not installed"
json.dump(info, open(env / "versions.json", "w"), indent=2)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
(env / "requirements_freeze.txt").write_text(freeze)

# 6. Report and final secret scan
print("--------------------")
print(f"Copied files: {len(copied)}")
big = [(p, s) for p, s in copied if s > 95 * 1024 * 1024]
print(f"Total size: {sum(s for _, s in copied) / 1024 / 1024:.1f} MB")
if big:
    print("WARNING: files larger than 95 MB (GitHub limit is 100 MB per file):")
    for p, s in big:
        print(f"  {p}: {s / 1024 / 1024:.1f} MB")
print(f"Missing paths: {len(missing)}")
for m in missing:
    print(f"  MISSING: {m}")

hits = []
for p in REPO.rglob("*"):
    if p.is_file() and p.suffix in {".py", ".ipynb", ".txt", ".md", ".json", ".csv"}:
        s = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\b[0-9a-f]{56}\b", s):
            hits.append(str(p.relative_to(REPO)))
print(f"Token scan: {'CLEAN' if not hits else 'FOUND IN ' + ', '.join(hits)}")
