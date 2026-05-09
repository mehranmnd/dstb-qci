#!/usr/bin/env python3
"""
Phase 2.5: Joinpoint (segmented log-linear) regression for Paper 1.

The current Paper 1 specification fits a single log-linear AAPC over
1990-2021 plus a separate "recent period" AAPC over 2010-2021 with a
hand-chosen 2010 break. This is a known weakness: the 2010 cut is
arbitrary, and several locations (notably Zimbabwe and Lesotho, where
HIV/TB co-infection drove a peak then ART scale-up reversed it) clearly
have non-monotonic trajectories that a single slope misrepresents.

Here we fit a 1-, 2-, and 3-segment piecewise log-linear regression to
each focus location's QCI series (1990-2021), select the segment count
by Bayesian Information Criterion (BIC), and report the data-driven
joinpoint(s) plus segment AAPCs. We use `pwlf` for the breakpoint
search.

Output:
  results/paper1/tables/table_joinpoint_focus.csv  (per-location segment table)
  results/paper1/qci_joinpoint_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import pwlf

warnings.filterwarnings("ignore")

BASE = "/Users/mehranmamandipoor/Desktop/thesis"
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
OUT_TABLE = os.path.join(BASE, "results/paper1/tables/table_joinpoint_focus.csv")
OUT_JSON = os.path.join(BASE, "results/paper1/qci_joinpoint_summary.json")

# Locations the paper highlights or that show non-monotonic behaviour
FOCUS_LOCATIONS = [
    "Global",
    "High SDI", "High-middle SDI", "Middle SDI", "Low-middle SDI", "Low SDI",
    "Europe & Central Asia - WB", "Middle East & North Africa - WB",
    "Sub-Saharan Africa - WB", "East Asia & Pacific - WB",
    "South Asia - WB", "Latin America & Caribbean - WB", "North America",
    "Iran",
    "Afghanistan", "Lesotho", "Zimbabwe", "Eswatini", "Kenya",
    "Libya", "Burkina Faso",
    "Guatemala", "Rwanda", "Zambia", "Ethiopia",
    "United States",
]

MAX_SEGMENTS = 3

def fit_segments(years, log_qci, n_segments):
    pw = pwlf.PiecewiseLinFit(years, log_qci)
    pw.fit(n_segments)
    yhat = pw.predict(years)
    rss = float(np.sum((log_qci - yhat) ** 2))
    n = len(years)
    # Number of free parameters in n_segments piecewise linear with continuity:
    # n_segments + 1 break-points (2 fixed at end) and n_segments slopes plus 1
    # intercept. pwlf's effective parameter count: n_segments breakpoint
    # positions (interior) + (n_segments + 1) slopes/intercepts.
    # We approximate as 2*n_segments parameters (slopes + interior breaks).
    k = 2 * n_segments
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return pw, rss, aic, bic


def aapc_per_segment(pw):
    """Convert each segment's slope on log scale to %/year."""
    breaks = pw.fit_breaks
    slopes = pw.calc_slopes()
    segs = []
    for i, slope in enumerate(slopes):
        aapc_pct = (np.exp(slope) - 1.0) * 100.0
        segs.append({
            "segment": i + 1,
            "year_from": float(breaks[i]),
            "year_to": float(breaks[i + 1]),
            "slope_log": float(slope),
            "aapc_pct": round(float(aapc_pct), 4),
        })
    return segs


df = pd.read_csv(QCI_PATH)
m = (df["age_name"] == "Age-standardized") & (df["sex_name"] == "Both")
sub = df.loc[m, ["iso_location_name", "year", "qci"]].copy()

results = []
for loc in FOCUS_LOCATIONS:
    g = sub[sub["iso_location_name"] == loc].sort_values("year").reset_index(drop=True)
    if len(g) < 8:
        results.append({"Location": loc, "status": "skipped (too few obs)"})
        continue
    yrs = g["year"].values.astype(float)
    qci = g["qci"].values.astype(float)
    if (qci <= 0).any():
        results.append({"Location": loc, "status": "skipped (non-positive QCI)"})
        continue
    log_q = np.log(qci)

    fits = {}
    for n_seg in range(1, MAX_SEGMENTS + 1):
        try:
            pw, rss, aic, bic = fit_segments(yrs, log_q, n_seg)
            fits[n_seg] = {"pw": pw, "rss": rss, "aic": aic, "bic": bic}
        except Exception as e:
            fits[n_seg] = None

    valid_fits = {k: v for k, v in fits.items() if v is not None}
    if not valid_fits:
        results.append({"Location": loc, "status": "fit failed"})
        continue

    best_n = min(valid_fits, key=lambda k: valid_fits[k]["bic"])
    best = valid_fits[best_n]
    segs = aapc_per_segment(best["pw"])

    # Also report: AAPC under 1-segment for direct comparison to the paper
    one_seg_aapc = (np.exp(valid_fits[1]["pw"].calc_slopes()[0]) - 1.0) * 100.0
    breaks = list(best["pw"].fit_breaks)

    results.append({
        "Location": loc,
        "n_obs": int(len(g)),
        "best_n_segments": best_n,
        "joinpoints": breaks[1:-1] if best_n > 1 else [],  # interior breaks only
        "bic_1seg": round(float(valid_fits[1]["bic"]), 3),
        "bic_best": round(float(best["bic"]), 3),
        "aapc_1seg_pct": round(float(one_seg_aapc), 4),
        "segments_best": segs,
    })

print(f"Locations analysed: {len(results)}")
n_with_break = sum(1 for r in results if r.get("best_n_segments", 1) > 1)
print(f"Locations with >=1 joinpoint preferred by BIC: {n_with_break}")

# Flatten for CSV
rows = []
for r in results:
    if "segments_best" not in r:
        rows.append({"Location": r["Location"], "status": r.get("status")})
        continue
    for s in r["segments_best"]:
        rows.append({
            "Location": r["Location"],
            "best_n_segments": r["best_n_segments"],
            "segment": s["segment"],
            "year_from": s["year_from"],
            "year_to": s["year_to"],
            "aapc_pct": s["aapc_pct"],
            "aapc_1seg_pct": r["aapc_1seg_pct"],
            "bic_1seg": r["bic_1seg"],
            "bic_best": r["bic_best"],
            "joinpoints": ",".join(f"{b:.1f}" for b in r["joinpoints"]) if r["joinpoints"] else "",
        })
df_out = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
df_out.to_csv(OUT_TABLE, index=False, float_format="%.4f")
print(f"\nSaved table: {OUT_TABLE}")

# Highlight: who preferred a break point and where?
print("\nLocations where BIC preferred >1 segment:")
for r in results:
    if r.get("best_n_segments", 1) > 1:
        loc = r["Location"]
        jp = ", ".join(f"{b:.1f}" for b in r["joinpoints"])
        segs_str = "; ".join(
            f"{int(s['year_from'])}-{int(s['year_to'])}: {s['aapc_pct']:+.3f}%/yr"
            for s in r["segments_best"]
        )
        print(f"  {loc}: joinpoint(s) ~{jp} | {segs_str} | (1-seg AAPC: {r['aapc_1seg_pct']:+.3f}%/yr)")

summary = {
    "n_locations_analysed": int(len([r for r in results if "segments_best" in r])),
    "n_with_joinpoint_preferred": int(n_with_break),
    "locations_with_breaks": [
        {
            "location": r["Location"],
            "best_n_segments": r["best_n_segments"],
            "joinpoints": [round(b, 1) for b in r["joinpoints"]],
            "segments": r["segments_best"],
            "aapc_1seg_pct": r["aapc_1seg_pct"],
        }
        for r in results
        if r.get("best_n_segments", 1) > 1
    ],
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary: {OUT_JSON}")
