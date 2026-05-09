"""
QCI Uncertainty Estimation via Monte Carlo Simulation
=====================================================
Propagates GBD 95% uncertainty intervals through the QCI pipeline
using 1000 Monte Carlo draws per location-year row.

Author: Generated for thesis analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import joblib
import sys
import time

# -- Configuration ------------------------------------------------------------
N_DRAWS      = 1000
SEED         = 42
DATA_PATH    = "/Users/mehranmamandipoor/Desktop/thesis/results/shared/qci_complete_data.csv"
MODEL_PATH   = "/Users/mehranmamandipoor/Desktop/thesis/code/qci_pca_model.joblib"
OUTPUT_PATH  = "/Users/mehranmamandipoor/Desktop/thesis/results/shared/qci_uncertainty.txt"

FOCUS_LOCATIONS = [
    "Global",
    "High SDI", "High-middle SDI", "Middle SDI", "Low-middle SDI", "Low SDI",
    "Europe & Central Asia - WB",
    "Middle East & North Africa - WB",
    "Sub-Saharan Africa - WB",
    "East Asia & Pacific - WB",
    "South Asia - WB",
    "Latin America & Caribbean - WB",
    "North America",
    "Islamic Republic of Iran",
    "Islamic Republic of Afghanistan",
    "Kingdom of Lesotho",
    "Republic of Zimbabwe",
    "Republic of the Marshall Islands",
    "Central African Republic",
    "Bermuda",
    "New Zealand",
    "Republic of Malta",
    "Antigua and Barbuda",
    "Grand Duchy of Luxembourg",
]

FOCUS_YEARS = [1990, 2021]

# -- Load data & model --------------------------------------------------------
print("Loading data ...")
df_all = pd.read_csv(DATA_PATH)
model  = joblib.load(MODEL_PATH)

scaler_mean  = model["scaler_mean_"]
scaler_scale = model["scaler_scale_"]
pca_comp     = model["pca_components_"][0]
sign         = model["sign"]
lo_all       = model["lo_all"]
hi_all       = model["hi_all"]

# -- Subset: Age-standardized, Both sexes, focus locations/years ---------------
mask = (
    (df_all["sex_name"] == "Both") &
    (df_all["age_name"] == "Age-standardized") &
    (df_all["location_name"].isin(FOCUS_LOCATIONS)) &
    (df_all["year"].isin(FOCUS_YEARS))
)
df = df_all.loc[mask].copy().reset_index(drop=True)
print(f"Working subset: {len(df)} rows  "
      f"({df['location_name'].nunique()} locations x {df['year'].nunique()} years)")

if len(df) == 0:
    print("ERROR: No rows matched the filter -- check location names.")
    sys.exit(1)

# -- Helper: apply fitted PCA pipeline ----------------------------------------
def features_to_qci(mir, ylltoyld, daltoper):
    X = np.stack([mir, ylltoyld, daltoper], axis=-1)
    X_scaled = (X - scaler_mean) / scaler_scale
    pc1_raw = X_scaled @ pca_comp
    pc1_signed = pc1_raw * sign
    qci = (pc1_signed - lo_all) / (hi_all - lo_all) * 100.0
    return qci

# -- Monte Carlo simulation ---------------------------------------------------
print(f"Running Monte Carlo simulation with {N_DRAWS} draws ...")
rng = np.random.default_rng(SEED)
t0 = time.time()

results = []

for idx, row in df.iterrows():
    loc  = row["location_name"]
    year = row["year"]
    qci_point = row["pca_score"]

    measures = {}
    for var in ["Deaths", "Incidence", "YLLs", "YLDs", "DALYs", "Prevalence"]:
        val   = row[f"val_{var}"]
        lower = row[f"lower_{var}"]
        upper = row[f"upper_{var}"]
        std = (upper - lower) / (2 * 1.96)
        draws = rng.normal(loc=val, scale=max(std, 1e-15), size=N_DRAWS)
        draws = np.maximum(draws, 1e-15)
        measures[var] = draws

    mir_draws      = measures["Deaths"]   / measures["Incidence"]
    ylltoyld_draws = measures["YLLs"]     / measures["YLDs"]
    daltoper_draws = measures["DALYs"]    / measures["Prevalence"]

    qci_draws = features_to_qci(mir_draws, ylltoyld_draws, daltoper_draws)
    qci_draws = np.clip(qci_draws, 0, 100)

    q025 = np.percentile(qci_draws, 2.5)
    q975 = np.percentile(qci_draws, 97.5)
    q50  = np.percentile(qci_draws, 50.0)
    sd   = np.std(qci_draws)

    results.append({
        "location_name": loc,
        "year": year,
        "qci_point": qci_point,
        "qci_median_mc": q50,
        "qci_lower": q025,
        "qci_upper": q975,
        "qci_sd_mc": sd,
    })

elapsed = time.time() - t0
print(f"Simulation complete in {elapsed:.1f} s.")

# -- Assemble results ---------------------------------------------------------
res = pd.DataFrame(results)

SHORT = {
    "Islamic Republic of Iran":        "Iran",
    "Islamic Republic of Afghanistan": "Afghanistan",
    "Kingdom of Lesotho":              "Lesotho",
    "Republic of Zimbabwe":            "Zimbabwe",
    "Republic of the Marshall Islands":"Marshall Islands",
    "Central African Republic":        "Central African Rep.",
    "Grand Duchy of Luxembourg":       "Luxembourg",
    "Republic of Malta":               "Malta",
    "Europe & Central Asia - WB":      "Europe & C. Asia (WB)",
    "Middle East & North Africa - WB": "MENA (WB)",
    "Sub-Saharan Africa - WB":         "Sub-Saharan Africa (WB)",
    "East Asia & Pacific - WB":        "E. Asia & Pacific (WB)",
    "South Asia - WB":                 "South Asia (WB)",
    "Latin America & Caribbean - WB":  "Lat. Am. & Carib. (WB)",
}
res["display_name"] = res["location_name"].map(lambda x: SHORT.get(x, x))

ORDER = {n: i for i, n in enumerate(FOCUS_LOCATIONS)}
res["sort_key"] = res["location_name"].map(ORDER)
res = res.sort_values(["sort_key", "year"]).reset_index(drop=True)

# -- Build output text ---------------------------------------------------------
lines = []
lines.append("=" * 95)
lines.append("QCI UNCERTAINTY ESTIMATES -- Monte Carlo propagation of GBD 95% UI")
lines.append("=" * 95)
lines.append(f"  Draws per row        : {N_DRAWS}")
lines.append(f"  Distribution assumed : Normal, std = (upper - lower) / (2 x 1.96)")
lines.append(f"  Subset               : Both sexes, Age-standardized")
lines.append(f"  Years                : {FOCUS_YEARS}")
lines.append(f"  Locations            : {res['location_name'].nunique()}")
lines.append(f"  PCA model lo_all     : {lo_all:.4f}")
lines.append(f"  PCA model hi_all     : {hi_all:.4f}")
lines.append(f"  PCA sign             : {sign}")
lines.append(f"  Random seed          : {SEED}")
lines.append("")

hdr = f"{'Location':<30s} {'Year':>4s}  {'QCI (point)':>11s}  {'QCI_lower':>9s}  {'QCI_upper':>9s}  {'95% Width':>9s}"
sep = "-" * len(hdr)
lines.append(hdr)
lines.append(sep)

prev_group = None
for _, r in res.iterrows():
    group = r["sort_key"]
    if prev_group is not None and group != prev_group:
        lines.append("")
    prev_group = group

    width = r["qci_upper"] - r["qci_lower"]
    lines.append(
        f"{r['display_name']:<30s} {int(r['year']):>4d}  "
        f"{r['qci_point']:>11.2f}  {r['qci_lower']:>9.2f}  {r['qci_upper']:>9.2f}  "
        f"{width:>9.2f}"
    )

lines.append(sep)
lines.append("")

# -- Summary statistics --------------------------------------------------------
lines.append("SUMMARY STATISTICS")
lines.append("-" * 50)
widths = res["qci_upper"] - res["qci_lower"]
lines.append(f"  Mean 95% CI width       : {widths.mean():.2f} QCI points")
lines.append(f"  Median 95% CI width     : {widths.median():.2f} QCI points")
lines.append(f"  Min 95% CI width        : {widths.min():.2f}  ({res.loc[widths.idxmin(), 'display_name']}, {int(res.loc[widths.idxmin(), 'year'])})")
lines.append(f"  Max 95% CI width        : {widths.max():.2f}  ({res.loc[widths.idxmax(), 'display_name']}, {int(res.loc[widths.idxmax(), 'year'])})")
lines.append("")

for yr in FOCUS_YEARS:
    sub = res[res["year"] == yr]
    w = sub["qci_upper"] - sub["qci_lower"]
    lines.append(f"  Year {yr} -- mean width: {w.mean():.2f}, median: {w.median():.2f}")
lines.append("")

# -- Detailed per-location summary (2021) --------------------------------------
lines.append("DETAILED 2021 VALUES")
lines.append("-" * 80)
sub2021 = res[res["year"] == 2021].copy()
sub2021["width"] = sub2021["qci_upper"] - sub2021["qci_lower"]
for _, r in sub2021.iterrows():
    lines.append(
        f"  {r['display_name']:<30s}  QCI = {r['qci_point']:.2f}  "
        f"[{r['qci_lower']:.2f}, {r['qci_upper']:.2f}]  "
        f"width = {r['width']:.2f}"
    )
lines.append("")

# -- Note on interpretation ----------------------------------------------------
lines.append("INTERPRETATION NOTES")
lines.append("-" * 80)
lines.append("  - Narrow intervals (< 5 points) indicate high confidence in the QCI estimate.")
lines.append("  - Wider intervals may reflect greater uncertainty in the underlying GBD measures")
lines.append("    (deaths, incidence, prevalence, YLLs, YLDs, DALYs) for that location-year.")
lines.append("  - Aggregated locations (Global, SDI, WB regions) tend to have narrower intervals")
lines.append("    because their underlying GBD estimates are more precisely estimated.")
lines.append("  - Small countries / territories may show wider intervals due to sparse data.")
lines.append("")

output = "\n".join(lines)

with open(OUTPUT_PATH, "w") as f:
    f.write(output)

print(output)
print(f"\nResults saved to: {OUTPUT_PATH}")

csv_path = OUTPUT_PATH.replace(".txt", ".csv")
res.to_csv(csv_path, index=False)
print(f"CSV saved to:     {csv_path}")
