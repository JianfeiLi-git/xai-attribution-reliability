# -*- coding: utf-8 -*-
"""
check_results_text.py

Scan the result tables (CSV/JSON) in results/ for Chinese text in column names or values.
Nothing is changed; the script only reports what it finds.
Usage (from the repository root):
    python tools/check_results_text.py
"""

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF\u3000-\u303F]")

files = sorted(list((REPO / "results").rglob("*.csv")) + list((REPO / "results").rglob("*.json")))
print(f"Files checked: {len(files)}")
found = 0
for f in files:
    rel = f.relative_to(REPO)
    if f.suffix == ".json":
        text = f.read_text(encoding="utf-8", errors="ignore")
        hits = sorted(set(m.group() for m in re.finditer(r"[^\s\"',:{}\[\]]*" + CJK.pattern + r"[^\s\"',:{}\[\]]*", text)))
        if hits:
            found += 1
            print(f"{rel}: {len(hits)} Chinese strings, e.g. {hits[:5]}")
        continue
    try:
        df = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"{rel}: could not read ({e})")
        continue
    bad_cols = [c for c in df.columns if CJK.search(str(c))]
    bad_vals = {}
    for c in df.columns:
        vals = sorted(set(v for v in df[c] if CJK.search(v)))
        if vals:
            bad_vals[c] = vals
    if bad_cols or bad_vals:
        found += 1
        print(f"{rel}")
        if bad_cols:
            print(f"  Chinese column names: {bad_cols}")
        for c, vals in bad_vals.items():
            print(f"  column '{c}': {len(vals)} Chinese values, e.g. {vals[:5]}")
print("--------------------")
print(f"Files with Chinese text: {found}")
