#!/usr/bin/env python3
"""
Extract per-country population for 2021 from the local IHME data file.

Population is derived as: Number / Rate * 100000, computed at the
location-year level using DS-TB All-ages, Both-sexes rows. Any cause
would do (population is shared across causes); we use cause 934
(Drug-susceptible tuberculosis) because it is what the rest of the
project already filters on, so the file is naturally indexed there.

This is used by Phase 2.2 (population-weighted concentration index).
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IHME_PATH = os.path.join(BASE, "data/ihme.csv")
OUT_PATH = os.path.join(BASE, "results/shared/population_2021.csv")

# Pull only the rows we need (filter at parse time to keep memory low).
# We want: cause_id == 934 (DS-TB), sex_id == 3 (Both), age_name == 'All ages',
# year == 2021, both metric_name == 'Number' and 'Rate' for the same row.
print("Reading IHME (this can take ~30s on 1.5GB file)...")
chunks = []
for chunk in pd.read_csv(
    IHME_PATH,
    usecols=["location_id", "location_name", "sex_id", "age_name",
             "cause_id", "metric_name", "year", "val"],
    chunksize=500_000,
):
    sel = chunk[
        (chunk["cause_id"] == 934)
        & (chunk["sex_id"] == 3)
        & (chunk["age_name"] == "All ages")
        & (chunk["year"] == 2021)
        & (chunk["metric_name"].isin(["Number", "Rate"]))
    ]
    if len(sel):
        chunks.append(sel)
df = pd.concat(chunks, ignore_index=True)
print(f"  Filtered: {len(df)} rows")

# Pivot Number and Rate side-by-side so we can divide them
# (we average across measures because all of them use the same
# population for a given location/year)
agg = (df.groupby(["location_id", "location_name", "metric_name"], as_index=False)
         ["val"].mean())
piv = agg.pivot(index=["location_id", "location_name"],
                columns="metric_name", values="val").reset_index()

# population = Number / (Rate / 100000) = Number * 100000 / Rate
piv["population_2021"] = piv["Number"] * 100000.0 / piv["Rate"]
out = piv[["location_id", "location_name", "population_2021"]].copy()
out = out.dropna(subset=["population_2021"]).sort_values("location_name").reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False, float_format="%.0f")
print(f"  Saved {len(out)} locations to {OUT_PATH}")
print(out.head(10))
print("\nWorld total (should be ~7.9B):",
      f"{out['population_2021'].sum() / 1e9:.2f} B")
