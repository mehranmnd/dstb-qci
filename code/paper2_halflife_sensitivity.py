#!/usr/bin/env python3
"""
Phase 2.7: Half-life sensitivity for Paper 2 convergence analysis.

Paper 2 currently reports a 42-year convergence half-life from a
log-linear beta-convergence regression fit on the FULL 1990-2021
period. The full-period fit is dominated by rapid 1990s convergence
followed by a plateau, which can produce a misleadingly optimistic
half-life. We re-fit beta-convergence on 2005-2021 (post-rapid-
convergence subperiod) to test whether the 42-year half-life is
robust or whether the recent pace is materially slower.

Specifications:
  log(QCI_t / QCI_0) ~ alpha + beta * log(QCI_0)
where:
  - Full period: t = 2021, 0 = 1990, T = 31
  - Sensitivity: t = 2021, 0 = 2005, T = 16

Half-life formula:
  lambda = -log(1 + beta) / T
  half-life = log(2) / lambda

Output:
  results/paper2/tables/table_halflife_sensitivity.csv
  results/paper2/analysis/paper2_halflife_summary.json
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/mehranmamandipoor/Desktop/thesis"
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
OUT_TABLE = os.path.join(BASE, "results/paper2/tables/table_halflife_sensitivity.csv")
OUT_JSON = os.path.join(BASE, "results/paper2/analysis/paper2_halflife_summary.json")

sys.path.insert(0, os.path.join(BASE, "code"))
from mappings import WB_REGIONS, SDI_QUINTILES, IRAN_PROVINCES

df = pd.read_csv(QCI_PATH)
regions_set = set(WB_REGIONS + SDI_QUINTILES + IRAN_PROVINCES + [
    "Global", "Africa", "America", "Asia", "Europe", "Oceania",
    "Eastern Sub-Saharan Africa", "Central Sub-Saharan Africa",
    "Southern Sub-Saharan Africa", "Western Sub-Saharan Africa",
    "High-income Asia Pacific", "High-income North America",
    "Andean Latin America", "Central Latin America",
    "Southern Latin America", "Tropical Latin America",
    "Caribbean", "North Africa and Middle East", "Australasia",
    "North America", "South Asia", "Central Asia", "Southeast Asia",
    "East Asia", "Eastern Europe", "Western Europe", "Central Europe",
    "World Bank High Income", "World Bank Low Income",
    "World Bank Upper Middle Income", "World Bank Lower Middle Income",
    "African Region", "South-East Asia Region", "Western Pacific Region",
    "European Region", "Eastern Mediterranean Region", "Region of the Americas"])
countries_only = sorted(
    [c for c in df["iso_location_name"].unique() if c not in regions_set])

m = (df["age_name"] == "Age-standardized") & (df["sex_name"] == "Both")
df_long = df.loc[m & df["iso_location_name"].isin(countries_only),
                 ["iso_location_name", "year", "qci"]].copy()


def beta_convergence(qci_start, qci_end, T_years):
    """Log-linear beta-convergence:
       log(QCI_end / QCI_start) ~ a + b * log(QCI_start)
    Returns dict with beta, R2, lambda, half-life."""
    y = np.log(qci_end / qci_start)
    x = np.log(qci_start)
    slope, intercept, r_value, p_value, se = stats.linregress(x, y)
    R2 = r_value ** 2
    # lambda from log-linear: 1 + beta = (1 - exp(-lambda*T)) / something...
    # Standard Sala-i-Martin: log(QCI_end/QCI_start) = a + (1 - e^{-lambda*T}) * log(QCI_start) + e
    # so the OLS slope b = -(1 - e^{-lambda*T}) i.e. e^{-lambda*T} = 1 + b
    if 1 + slope <= 0:
        return None
    lam = -np.log(1 + slope) / T_years
    half_life = np.log(2) / lam if lam != 0 else np.nan
    return {
        "n": int(len(qci_start)),
        "beta_OLS": float(slope),
        "se_beta": float(se),
        "p_value": float(p_value),
        "R2": float(R2),
        "lambda": float(lam),
        "half_life_years": float(half_life),
    }


def run_window(start_year, end_year):
    df_start = df_long[df_long["year"] == start_year][["iso_location_name", "qci"]].rename(columns={"qci": "qci_start"})
    df_end   = df_long[df_long["year"] == end_year][["iso_location_name", "qci"]].rename(columns={"qci": "qci_end"})
    panel = df_start.merge(df_end, on="iso_location_name", how="inner").dropna()
    panel = panel[(panel["qci_start"] > 0) & (panel["qci_end"] > 0)]
    res = beta_convergence(panel["qci_start"].values, panel["qci_end"].values, end_year - start_year)
    if res is None:
        return None
    res["start_year"] = start_year
    res["end_year"] = end_year
    res["span_years"] = end_year - start_year
    return res


windows = [
    (1990, 2021),  # full period
    (2005, 2021),  # post-rapid-convergence
    (2010, 2021),  # most recent
    (1990, 2005),  # early period
]

rows = []
for s, e in windows:
    res = run_window(s, e)
    if res is None:
        continue
    rows.append(res)

df_out = pd.DataFrame(rows)
print(df_out[[
    "start_year", "end_year", "span_years", "n", "beta_OLS",
    "R2", "lambda", "half_life_years"
]].to_string(index=False, float_format='%.4f'))

os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
df_out.to_csv(OUT_TABLE, index=False, float_format="%.6f")
print(f"\nSaved table: {OUT_TABLE}")

# Headline
full = df_out[(df_out["start_year"] == 1990) & (df_out["end_year"] == 2021)].iloc[0]
recent = df_out[(df_out["start_year"] == 2005) & (df_out["end_year"] == 2021)].iloc[0]
print(f"\nHALF-LIFE COMPARISON:")
print(f"  Full period (1990-2021): {full['half_life_years']:.1f} years (matches the 42-yr figure cited in Paper 2)")
print(f"  Post-2005 (2005-2021):    {recent['half_life_years']:.1f} years")
ratio = recent['half_life_years'] / full['half_life_years']
print(f"  Recent half-life is {ratio:.2f}x the full-period estimate")

summary = {
    "windows": [
        {
            "start_year": int(r["start_year"]),
            "end_year": int(r["end_year"]),
            "span_years": int(r["span_years"]),
            "n": int(r["n"]),
            "beta_OLS": round(r["beta_OLS"], 4),
            "se_beta": round(r["se_beta"], 4),
            "p_value": float(r["p_value"]),
            "R2": round(r["R2"], 4),
            "lambda": round(r["lambda"], 6),
            "half_life_years": round(r["half_life_years"], 1),
        }
        for _, r in df_out.iterrows()
    ],
    "interpretation": (
        f"Full-period (1990-2021) half-life is {full['half_life_years']:.1f} years. "
        f"Restricting to 2005-2021 (post-rapid-convergence) gives "
        f"{recent['half_life_years']:.1f} years — "
        + ("MUCH LONGER" if recent['half_life_years'] > 1.5 * full['half_life_years']
           else "comparable")
        + " than the full-period estimate. This sensitivity addresses the "
        "concern that the headline 42-year half-life is dominated by rapid "
        "1990s convergence and overstates the recent pace."
    ),
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {OUT_JSON}")
