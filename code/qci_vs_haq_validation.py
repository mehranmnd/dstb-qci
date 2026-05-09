#!/usr/bin/env python3
"""
QCI vs HAQ Index validation (Paper 1)
=====================================
Computes Spearman and Pearson correlations between the DS-TB Quality of
Care Index (QCI) and the GBD Healthcare Access and Quality (HAQ) Index
across countries, using the latest year that is common to both data
sources. The HAQ Index local file (`data/HAQ.CSV`) covers 1990-2019; the
QCI runs 1990-2021. We validate at year 2019.

This is the external (non-SDI) validator referenced in Paper 1's
Strengths and Limitations section, addressing the reviewer concern that
SDI-only validation is partly circular because GBD's DisMod-MR borrows
SDI-related covariates.

Output:
  results/paper1/tables/table_qci_haq_validation.csv  (per-country merge)
  results/paper1/qci_haq_summary.json                 (correlation stats)

Source: HAQ.CSV is the GBD HAQ Index file (location, year, indicator,
age type, value with 95% UI). We use indicator_id=100 (HAQ Index)
restricted to age type "Overall" for the cross-country panel.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/mehranmamandipoor/Desktop/thesis"
HAQ_PATH = os.path.join(BASE, "data/HAQ.CSV")
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
OUT_TABLE = os.path.join(BASE, "results/paper1/tables/table_qci_haq_validation.csv")
OUT_JSON = os.path.join(BASE, "results/paper1/qci_haq_summary.json")

VALIDATION_YEAR = 2019  # latest year present in HAQ.CSV

# ── Load HAQ ───────────────────────────────────────────────────────────────────
print("Loading HAQ Index...")
haq = pd.read_csv(HAQ_PATH)
haq_idx = haq[
    (haq["indicator_id"] == 100)
    & (haq["haq_index_age_type"] == "Overall")
    & (haq["year_id"] == VALIDATION_YEAR)
][["location_name", "val"]].rename(
    columns={"location_name": "haq_location_name", "val": "haq_index"}
).reset_index(drop=True)
print(f"  HAQ Index ({VALIDATION_YEAR}, Overall): {len(haq_idx)} locations")

# ── Load QCI for the same year ────────────────────────────────────────────────
print("Loading QCI...")
qci = pd.read_csv(QCI_PATH)
qci_yr = qci[
    (qci["year"] == VALIDATION_YEAR)
    & (qci["sex_name"] == "Both")
    & (qci["age_name"] == "Age-standardized")
][["iso_location_name", "haq_location_name", "qci"]].reset_index(drop=True)
print(f"  QCI ({VALIDATION_YEAR}, Age-std, Both): {len(qci_yr)} rows")

# ── Merge on the GBD-formatted location name (haq_location_name in QCI table
# is already the IHME canonical name and matches HAQ.CSV's location_name) ──────
merged = qci_yr.merge(haq_idx, on="haq_location_name", how="inner")
merged = merged.dropna(subset=["qci", "haq_index"]).reset_index(drop=True)
print(f"  Merged: {len(merged)} matched countries")

# ── Statistics ─────────────────────────────────────────────────────────────────
spearman = stats.spearmanr(merged["qci"], merged["haq_index"])
pearson = stats.pearsonr(merged["qci"], merged["haq_index"])

print(f"\n  Spearman rho = {spearman.correlation:.4f}  p = {spearman.pvalue:.3e}")
print(f"  Pearson r    = {pearson.statistic:.4f}  p = {pearson.pvalue:.3e}")
print(f"  N countries  = {len(merged)}")
print(f"  QCI mean ± SD: {merged['qci'].mean():.2f} ± {merged['qci'].std():.2f}")
print(f"  HAQ mean ± SD: {merged['haq_index'].mean():.2f} ± {merged['haq_index'].std():.2f}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
merged.sort_values("qci", ascending=False).to_csv(OUT_TABLE, index=False, float_format="%.4f")
print(f"\n  Saved per-country table: {OUT_TABLE}")

summary = {
    "validation_year": VALIDATION_YEAR,
    "n_countries": int(len(merged)),
    "spearman_rho": round(float(spearman.correlation), 4),
    "spearman_p": float(spearman.pvalue),
    "pearson_r": round(float(pearson.statistic), 4),
    "pearson_p": float(pearson.pvalue),
    "qci_mean": round(float(merged["qci"].mean()), 2),
    "qci_sd": round(float(merged["qci"].std()), 2),
    "haq_mean": round(float(merged["haq_index"].mean()), 2),
    "haq_sd": round(float(merged["haq_index"].std()), 2),
    "qci_min": round(float(merged["qci"].min()), 2),
    "qci_max": round(float(merged["qci"].max()), 2),
    "haq_min": round(float(merged["haq_index"].min()), 2),
    "haq_max": round(float(merged["haq_index"].max()), 2),
}

with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved summary: {OUT_JSON}")
