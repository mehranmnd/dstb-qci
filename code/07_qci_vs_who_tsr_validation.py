#!/usr/bin/env python3
"""
Phase 3.1: External validation of QCI against WHO Treatment Success Rate.

Data source: WHO Global TB Programme bulk-download "Treatment outcomes"
file, downloaded once on 2026-05-09 from
https://extranet.who.int/tme/generateCSV.asp?ds=outcomes (verified at
https://www.who.int/teams/global-tuberculosis-programme/data via
WebFetch). Saved locally as data/who_tb_outcomes.csv (gitignored).

For 2019 (latest year both QCI and HAQ analyses use), we extract
c_new_tsr (Treatment Success Rate for newly notified cases, all forms,
under the post-2012 outcome definitions) and correlate it with QCI
across countries. Strong positive correlation supports QCI's external
construct validity beyond SDI.

Output:
  results/global/tables/table_qci_who_tsr_validation.csv
  results/global/qci_who_tsr_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHO_PATH = os.path.join(BASE, "data/who_tb_outcomes.csv")
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
OUT_TABLE = os.path.join(BASE, "results/global/tables/table_qci_who_tsr_validation.csv")
OUT_JSON = os.path.join(BASE, "results/global/qci_who_tsr_summary.json")

VALIDATION_YEAR = 2019

# Country name harmonisation: WHO data uses ISO short names; QCI uses
# `iso_location_name` (the IHME shortened form). Most match; a few need
# explicit mapping. Built incrementally from observation.
WHO_TO_QCI = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Democratic People's Republic of Korea": "North Korea",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Laos",
    "Micronesia (Federated States of)": "Federated States of Micronesia",
    "North Macedonia": "Macedonia",
    "Republic of Korea": "South Korea",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russian Federation",
    "Syrian Arab Republic": "Syria",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
    "Bahamas": "The Bahamas",
    "Gambia": "The Gambia",
    "Eswatini": "Eswatini",
    "Türkiye": "Turkey",
    "Turkey (Türkiye)": "Turkey",
}

print("Loading WHO TSR data...")
who = pd.read_csv(WHO_PATH)
print(f"  Total rows: {len(who)}")
who_yr = who[(who["year"] == VALIDATION_YEAR)
             & who["c_new_tsr"].notna()].copy()
print(f"  Rows for year {VALIDATION_YEAR} with non-null c_new_tsr: {len(who_yr)}")
who_yr["country_qci"] = who_yr["country"].map(WHO_TO_QCI).fillna(who_yr["country"])

print("\nLoading QCI...")
qci = pd.read_csv(QCI_PATH)
qci_yr = qci[(qci["year"] == VALIDATION_YEAR)
             & (qci["sex_name"] == "Both")
             & (qci["age_name"] == "Age-standardized")][
    ["iso_location_name", "qci"]].copy()
print(f"  QCI rows for year {VALIDATION_YEAR}: {len(qci_yr)}")

merged = qci_yr.merge(who_yr[["country_qci", "c_new_tsr"]],
                       left_on="iso_location_name", right_on="country_qci",
                       how="inner")
merged = merged.dropna(subset=["qci", "c_new_tsr"]).reset_index(drop=True)
print(f"\nMerged: {len(merged)} matched countries")

# Show any unmatched WHO countries (helps debug name mappings)
unmatched_who = sorted(set(who_yr["country_qci"]) - set(qci_yr["iso_location_name"]))
unmatched_qci = sorted(set(qci_yr["iso_location_name"]) - set(who_yr["country_qci"]))
print(f"\nWHO countries not matched in QCI: {len(unmatched_who)}")
if unmatched_who:
    print(f"  examples (first 10): {unmatched_who[:10]}")
print(f"QCI countries without WHO TSR for this year: {len(unmatched_qci)}")

if len(merged) < 30:
    print("\nWARNING: too few matches to draw a reliable conclusion; aborting.")
    raise SystemExit(1)

spearman = stats.spearmanr(merged["qci"], merged["c_new_tsr"])
pearson = stats.pearsonr(merged["qci"], merged["c_new_tsr"])

print(f"\n  Spearman ρ (QCI vs WHO TSR) = {spearman.correlation:.4f}, p = {spearman.pvalue:.2e}")
print(f"  Pearson r                   = {pearson.statistic:.4f}, p = {pearson.pvalue:.2e}")
print(f"  N = {len(merged)}")
print(f"  QCI mean ± SD : {merged['qci'].mean():.2f} ± {merged['qci'].std():.2f}")
print(f"  TSR mean ± SD : {merged['c_new_tsr'].mean():.2f} ± {merged['c_new_tsr'].std():.2f}")

os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
merged.sort_values("qci", ascending=False).to_csv(OUT_TABLE, index=False, float_format="%.4f")
print(f"\n  Saved: {OUT_TABLE}")

summary = {
    "validation_year": VALIDATION_YEAR,
    "data_source": "WHO Global TB Programme, bulk download 'Treatment outcomes' (https://extranet.who.int/tme/generateCSV.asp?ds=outcomes), accessed 2026-05-09",
    "indicator": "c_new_tsr (treatment success rate, newly notified cases, all forms, post-2012 definitions)",
    "n_countries": int(len(merged)),
    "spearman_rho": round(float(spearman.correlation), 4),
    "spearman_p": float(spearman.pvalue),
    "pearson_r": round(float(pearson.statistic), 4),
    "pearson_p": float(pearson.pvalue),
    "qci_mean": round(float(merged["qci"].mean()), 2),
    "qci_sd": round(float(merged["qci"].std()), 2),
    "tsr_mean": round(float(merged["c_new_tsr"].mean()), 2),
    "tsr_sd": round(float(merged["c_new_tsr"].std()), 2),
    "n_unmatched_who": int(len(unmatched_who)),
    "n_unmatched_qci": int(len(unmatched_qci)),
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {OUT_JSON}")
