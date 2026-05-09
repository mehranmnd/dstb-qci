#!/usr/bin/env python3
"""
Phase 2.3: Logit-transformed QCI sensitivity for Paper 1.

Addresses the ceiling-effect concern: QCI is bounded at [0, 100] and
above ~95 the index has limited room to rise, which can compress
absolute differences and AAPC at the high end. Reporting AAPC on the
logit scale (logit(QCI/100)) rescales the bounded variable so that
movement near 100 still produces meaningful unit changes, providing a
sensitivity check on whether observed trends survive the rescaling.

For each location with QCI series 1990-2021 we compute:
  - Linear AAPC: from log(QCI) (current paper specification)
  - Logit AAPC: slope of logit(QCI/100) regressed on year, expressed
    as percent-per-year on the logit scale.

The AAPCs are not directly comparable in scale (different transforms)
but their RANKINGS should agree if the linear specification is not
just a ceiling artefact. We report the Spearman correlation between
the two AAPC vectors as a robustness indicator.

Output:
  results/paper1/tables/table_logit_aapc_sensitivity.csv
  results/paper1/qci_logit_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/mehranmamandipoor/Desktop/thesis"
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
OUT_TABLE = os.path.join(BASE, "results/paper1/tables/table_logit_aapc_sensitivity.csv")
OUT_JSON = os.path.join(BASE, "results/paper1/qci_logit_summary.json")

EPS = 1e-3  # to avoid logit(1) = +inf when QCI hits 100 exactly

def linear_aapc(years, qci):
    """Log-linear AAPC: ln(QCI) ~ year. AAPC = (exp(slope) - 1) * 100."""
    qci = np.asarray(qci, float)
    years = np.asarray(years, float)
    mask = (qci > 0) & np.isfinite(qci)
    if mask.sum() < 3:
        return np.nan
    s, *_ = stats.linregress(years[mask], np.log(qci[mask]))
    return (np.exp(s) - 1) * 100


def logit_aapc(years, qci):
    """Logit AAPC: slope of logit(QCI/100) regressed on year, as %/yr."""
    qci = np.asarray(qci, float)
    years = np.asarray(years, float)
    p = np.clip(qci / 100.0, EPS, 1 - EPS)
    z = np.log(p / (1 - p))
    mask = np.isfinite(z)
    if mask.sum() < 3:
        return np.nan
    s, *_ = stats.linregress(years[mask], z[mask])
    return s * 100  # logit-units per year, expressed in %


# Load
df = pd.read_csv(QCI_PATH)
mask = (df["age_name"] == "Age-standardized") & (df["sex_name"] == "Both")
sub = df.loc[mask, ["iso_location_name", "year", "qci"]].sort_values(
    ["iso_location_name", "year"]).reset_index(drop=True)

rows = []
for loc, g in sub.groupby("iso_location_name"):
    full = g[(g["year"] >= 1990) & (g["year"] <= 2021)]
    if len(full) < 5:
        continue
    rows.append({
        "Location": loc,
        "QCI_1990": full[full["year"] == 1990]["qci"].iloc[0] if (full["year"] == 1990).any() else np.nan,
        "QCI_2021": full[full["year"] == 2021]["qci"].iloc[0] if (full["year"] == 2021).any() else np.nan,
        "AAPC_linear_pct": linear_aapc(full["year"].values, full["qci"].values),
        "AAPC_logit_pct":  logit_aapc(full["year"].values, full["qci"].values),
    })

df_out = pd.DataFrame(rows)
print(f"Locations analysed: {len(df_out)}")

valid = df_out.dropna(subset=["AAPC_linear_pct", "AAPC_logit_pct"])
spearman = stats.spearmanr(valid["AAPC_linear_pct"], valid["AAPC_logit_pct"])
pearson = stats.pearsonr(valid["AAPC_linear_pct"], valid["AAPC_logit_pct"])

print(f"\nSpearman ρ (linear vs logit AAPC) = {spearman.correlation:.4f}, p = {spearman.pvalue:.2e}")
print(f"Pearson r                        = {pearson.statistic:.4f}, p = {pearson.pvalue:.2e}")

# Compare ranks for top 10 / bottom 10
top10_lin = set(df_out.nlargest(10, "AAPC_linear_pct")["Location"])
top10_log = set(df_out.nlargest(10, "AAPC_logit_pct")["Location"])
bot10_lin = set(df_out.nsmallest(10, "AAPC_linear_pct")["Location"])
bot10_log = set(df_out.nsmallest(10, "AAPC_logit_pct")["Location"])
print(f"\nTop-10 overlap (linear ∩ logit): {len(top10_lin & top10_log)}/10")
print(f"Bottom-10 overlap: {len(bot10_lin & bot10_log)}/10")

# Find any sign-disagreements
sign_disagree = df_out[
    (np.sign(df_out["AAPC_linear_pct"]) != np.sign(df_out["AAPC_logit_pct"]))
    & df_out["AAPC_linear_pct"].notna()
    & df_out["AAPC_logit_pct"].notna()
]
print(f"\nSign disagreements: {len(sign_disagree)} location(s)")
if len(sign_disagree):
    print(sign_disagree.to_string(index=False))

os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
df_out.sort_values("AAPC_linear_pct", ascending=False).to_csv(
    OUT_TABLE, index=False, float_format="%.4f")
print(f"\nSaved table: {OUT_TABLE}")

# Summary
key_locations_global = ["Global"]
key_summary = {}
for L in key_locations_global:
    sub_L = df_out[df_out["Location"] == L]
    if len(sub_L):
        r = sub_L.iloc[0]
        key_summary[L] = {
            "AAPC_linear_pct": round(float(r["AAPC_linear_pct"]), 4),
            "AAPC_logit_pct": round(float(r["AAPC_logit_pct"]), 4),
        }

summary = {
    "n_locations": int(len(df_out)),
    "n_valid": int(len(valid)),
    "spearman_rho_linear_vs_logit": round(float(spearman.correlation), 4),
    "spearman_p": float(spearman.pvalue),
    "pearson_r_linear_vs_logit": round(float(pearson.statistic), 4),
    "top10_overlap": int(len(top10_lin & top10_log)),
    "bottom10_overlap": int(len(bot10_lin & bot10_log)),
    "sign_disagreements": int(len(sign_disagree)),
    "key_locations": key_summary,
    "interpretation": (
        "High Spearman correlation (>0.95) and high top/bottom 10 overlap "
        "would indicate that the linear-scale AAPC ranking is not an "
        "artefact of the 0-100 ceiling. Sign disagreements would flag "
        "locations whose direction of change reverses under the rescaling."
    ),
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {OUT_JSON}")
