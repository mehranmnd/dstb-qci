#!/usr/bin/env python3
"""
Paper 2 sensitivity: population-weighted Concentration Index (CI).

Replicates the equal-country CI computed by `paper2_analysis.py` but
weights each country by its 2021 population (extracted from the IHME
data file via `extract_population.py`). This sensitivity addresses the
reviewer concern that equal-country weighting treats Tuvalu and India
the same — population-weighting better reflects the distribution of
individual-level TB burden across the world's population.

The population-weighted CI follows the standard convenience formula
(Wagstaff/Kakwani):
    CI_w = 2 * cov_w(h, R_w) / mean_w(h)
where R_w is the weighted fractional rank of countries by SDI:
    R_w[i] = (cumulative weight up to i - 0.5 * w[i]) / sum(weights)
when countries are sorted by SDI ascending. mean_w(h) and cov_w are
weighted by population.

This script also reports unweighted CI alongside for direct comparison.
The output table is appended to results/paper2/tables/.

Output: results/paper2/tables/table_ci_population_weighted.csv
"""

import os
import json
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
QCI_COMPLETE_PATH = os.path.join(BASE, "results/shared/qci_complete_data.csv")
POP_PATH = os.path.join(BASE, "results/shared/population_2021.csv")
OUT_TABLE = os.path.join(BASE, "results/paper2/tables/table_ci_population_weighted.csv")
OUT_JSON = os.path.join(BASE, "results/paper2/analysis/paper2_pop_weighted_ci.json")

sys.path.insert(0, os.path.join(BASE, "code"))
from mappings import SDI_VALUE_MAP_2021, WB_REGIONS, SDI_QUINTILES, IRAN_PROVINCES

# ── Country list (mirrors paper2_analysis.py) ─────────────────────────────────
df_qci_full = pd.read_csv(QCI_PATH)
regions_set = set(WB_REGIONS + SDI_QUINTILES + IRAN_PROVINCES + [
    "Global", "Africa", "America", "Asia", "Europe", "Oceania",
    "South Asia", "Central Asia", "Southeast Asia", "East Asia",
    "Eastern Europe", "Western Europe", "Central Europe",
    "Eastern Sub-Saharan Africa", "Central Sub-Saharan Africa",
    "Southern Sub-Saharan Africa", "Western Sub-Saharan Africa",
    "High-income Asia Pacific", "High-income North America",
    "Andean Latin America", "Central Latin America",
    "Southern Latin America", "Tropical Latin America",
    "Caribbean", "North Africa and Middle East", "Australasia",
    "North America",
    "World Bank High Income", "World Bank Low Income",
    "World Bank Upper Middle Income", "World Bank Lower Middle Income",
    "African Region", "South-East Asia Region", "Western Pacific Region",
    "European Region", "Eastern Mediterranean Region", "Region of the Americas"])
countries_only = sorted(
    [c for c in df_qci_full["iso_location_name"].unique() if c not in regions_set])
print(f"Countries: {len(countries_only)}")

# ── SDI map (CSV + dict fallback, matches paper2_analysis.py logic) ──────────
_sdi_src = pd.read_csv(QCI_COMPLETE_PATH,
                        usecols=["iso_location_name", "year", "sex_name",
                                 "age_name", "sdi_value_2021"])
_sdi_src = _sdi_src[(_sdi_src["year"] == 2021)
                    & (_sdi_src["sex_name"] == "Both")
                    & (_sdi_src["age_name"] == "Age-standardized")]
_sdi_csv = (_sdi_src.dropna(subset=["sdi_value_2021"])
                      [["iso_location_name", "sdi_value_2021"]]
                      .drop_duplicates(subset="iso_location_name")
                      .rename(columns={"sdi_value_2021": "sdi"})
                      .reset_index(drop=True))
_csv_set = set(_sdi_csv["iso_location_name"])
_merged = []
for _c, _v in zip(_sdi_csv["iso_location_name"], _sdi_csv["sdi"]):
    _merged.append({"iso_location_name": _c, "sdi": float(_v)})
for _c, _v in SDI_VALUE_MAP_2021.items():
    if _c not in _csv_set:
        _merged.append({"iso_location_name": _c, "sdi": float(_v)})
sdi_df = pd.DataFrame(_merged).reset_index(drop=True)
print(f"SDI map: {len(sdi_df)} countries")

# ── Population (from extract_population.py output) ────────────────────────────
pop = pd.read_csv(POP_PATH)
# The population file uses GBD location_name (e.g., 'Islamic Republic of Iran').
# QCI uses iso_location_name (e.g., 'Iran'). Bridge via the QCI table itself,
# which contains both columns.
qci_keys = (df_qci_full[["iso_location_name", "location_name"]]
            .drop_duplicates(subset="iso_location_name"))
pop_iso = pop.merge(qci_keys, on="location_name", how="inner")
pop_iso = pop_iso[["iso_location_name", "population_2021"]]
print(f"Population mapped to ISO names: {len(pop_iso)} locations")


def weighted_ci(h, r_var, w):
    """Population-weighted Concentration Index.

    h:      health/QCI values, length n
    r_var:  socioeconomic ranking variable (SDI), length n
    w:      population weights, length n (sum > 0)

    Returns (CI_w, mean_w(h)).
    """
    h = np.asarray(h, dtype=float)
    r_var = np.asarray(r_var, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(r_var)
    h, r_var, w = h[order], r_var[order], w[order]
    W = w.sum()
    cum_w = np.cumsum(w)
    # Fractional rank (Lerman-Yitzhaki style for weighted data)
    R = (cum_w - 0.5 * w) / W
    mu = np.sum(w * h) / W
    # Weighted covariance
    cov_hr = np.sum(w * (h - mu) * (R - 0.5)) / W
    return 2.0 * cov_hr / mu, mu


def unweighted_ci(h, r_var):
    """Equal-country CI for comparison (matches paper2_analysis.py)."""
    h = np.asarray(h, dtype=float)
    r_var = np.asarray(r_var, dtype=float)
    n = len(h)
    if n < 5:
        return np.nan, np.nan
    R = (np.argsort(np.argsort(r_var)) + 1) / n
    mu = np.mean(h)
    return 2 * np.cov(h, R)[0, 1] / mu, mu


# ── Compute CI per year, weighted and unweighted ──────────────────────────────
mask_country = df_qci_full["iso_location_name"].isin(countries_only)
mask_strat = (df_qci_full["sex_name"] == "Both") & (df_qci_full["age_name"] == "Age-standardized")

rows = []
for year in range(1990, 2022):
    d = df_qci_full[mask_country & mask_strat & (df_qci_full["year"] == year)]
    m = (d.merge(sdi_df, on="iso_location_name", how="inner")
           .merge(pop_iso, on="iso_location_name", how="inner")
           .dropna(subset=["qci", "sdi", "population_2021"]))
    if len(m) < 10:
        continue
    ci_unw, mu_unw = unweighted_ci(m["qci"].values, m["sdi"].values)
    ci_w, mu_w = weighted_ci(m["qci"].values, m["sdi"].values, m["population_2021"].values)
    rows.append({
        "Year": year,
        "N_countries": len(m),
        "CI_unweighted": ci_unw,
        "CI_pop_weighted": ci_w,
        "Diff (w - unw)": ci_w - ci_unw,
        "Mean_QCI_unw": mu_unw,
        "Mean_QCI_w": mu_w,
    })

df_out = pd.DataFrame(rows)
print("\nCI comparison (selected years):")
print(df_out[df_out["Year"].isin([1990, 2000, 2010, 2021])].to_string(index=False))

os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
df_out.to_csv(OUT_TABLE, index=False, float_format="%.6f")
print(f"\nSaved table: {OUT_TABLE}")

# Summary JSON
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
summary = {
    "n_countries_2021": int(df_out[df_out["Year"] == 2021]["N_countries"].iloc[0]),
    "ci_unweighted": {
        "1990": round(float(df_out[df_out["Year"] == 1990]["CI_unweighted"].iloc[0]), 4),
        "2021": round(float(df_out[df_out["Year"] == 2021]["CI_unweighted"].iloc[0]), 4),
    },
    "ci_pop_weighted": {
        "1990": round(float(df_out[df_out["Year"] == 1990]["CI_pop_weighted"].iloc[0]), 4),
        "2021": round(float(df_out[df_out["Year"] == 2021]["CI_pop_weighted"].iloc[0]), 4),
    },
    "interpretation_2021": (
        "Population-weighted CI is "
        + ("greater" if df_out[df_out["Year"] == 2021]["Diff (w - unw)"].iloc[0] > 0
           else "smaller")
        + " than equal-country CI in 2021."
    ),
}
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {OUT_JSON}")
