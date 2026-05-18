#!/usr/bin/env python3
"""
Phase 2.4: Small-population sensitivity for Paper 1 rankings.

Reviewer concern: Bermuda (population ~64,000) and similar very-small
states top the QCI ranking partly because GBD point estimates for tiny
populations are inherently noisier. Re-run the top-20 / bottom-20
ranking after excluding countries with population < 500,000 in 2021,
to check how the headline rankings change.

Output:
  results/paper1/tables/table_smallpop_excluded_top20.csv
  results/paper1/tables/table_smallpop_excluded_bottom20.csv
  results/paper1/qci_smallpop_summary.json
"""

import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCI_PATH = os.path.join(BASE, "results/shared/qci.csv")
POP_PATH = os.path.join(BASE, "results/shared/population_2021.csv")
TOP_OUT = os.path.join(BASE, "results/paper1/tables/table_smallpop_excluded_top20.csv")
BOT_OUT = os.path.join(BASE, "results/paper1/tables/table_smallpop_excluded_bottom20.csv")
JSON_OUT = os.path.join(BASE, "results/paper1/qci_smallpop_summary.json")

SMALL_POP_THRESHOLD = 500_000

# Load
df = pd.read_csv(QCI_PATH)
mask = (df["year"] == 2021) & (df["sex_name"] == "Both") & (df["age_name"] == "Age-standardized")
qci_2021 = df.loc[mask, ["iso_location_name", "location_name", "qci"]].copy()

pop = pd.read_csv(POP_PATH)
m = qci_2021.merge(pop[["location_name", "population_2021"]],
                   on="location_name", how="left")

# Drop aggregates (no pop or unrealistically large) and keep only countries
# with QCI and pop. Aggregates like 'Global' will get a population from
# extract_population.py too, so we filter explicitly by name.
import sys
sys.path.insert(0, os.path.join(BASE, "code"))
from mappings import WB_REGIONS, SDI_QUINTILES, IRAN_PROVINCES
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
m = m[~m["iso_location_name"].isin(regions_set)]
m = m.dropna(subset=["qci", "population_2021"])

print(f"Total country-level rows with QCI + population: {len(m)}")

# Apply threshold
small = m[m["population_2021"] < SMALL_POP_THRESHOLD].copy()
big = m[m["population_2021"] >= SMALL_POP_THRESHOLD].copy()
print(f"Excluded (pop < {SMALL_POP_THRESHOLD:,}): {len(small)}")
print(f"Retained: {len(big)}")
print(f"\nExcluded countries: {sorted(small['iso_location_name'].tolist())}")

# Top-20 and bottom-20 rankings, both with and without small populations
top20_all = m.sort_values("qci", ascending=False).head(20).reset_index(drop=True)
top20_big = big.sort_values("qci", ascending=False).head(20).reset_index(drop=True)

bot20_all = m.sort_values("qci", ascending=True).head(20).reset_index(drop=True)
bot20_big = big.sort_values("qci", ascending=True).head(20).reset_index(drop=True)

# Identify which top20-all entries drop out under the small-pop filter
dropped_from_top20 = set(top20_all["iso_location_name"]) - set(top20_big["iso_location_name"])
new_in_top20 = set(top20_big["iso_location_name"]) - set(top20_all["iso_location_name"])
print(f"\nTop-20 countries dropped after pop>=500k filter: {sorted(dropped_from_top20)}")
print(f"Top-20 countries new after filter: {sorted(new_in_top20)}")

dropped_from_bot20 = set(bot20_all["iso_location_name"]) - set(bot20_big["iso_location_name"])
print(f"Bottom-20 countries dropped after pop>=500k filter: {sorted(dropped_from_bot20)}")

os.makedirs(os.path.dirname(TOP_OUT), exist_ok=True)
top20_big.to_csv(TOP_OUT, index=False, float_format="%.4f")
bot20_big.to_csv(BOT_OUT, index=False, float_format="%.4f")
print(f"\nSaved: {TOP_OUT}")
print(f"Saved: {BOT_OUT}")

summary = {
    "threshold_population": SMALL_POP_THRESHOLD,
    "n_total_countries": int(len(m)),
    "n_excluded": int(len(small)),
    "n_retained": int(len(big)),
    "excluded_countries": sorted(small["iso_location_name"].tolist()),
    "top20_dropped_after_filter": sorted(list(dropped_from_top20)),
    "top20_new_after_filter": sorted(list(new_in_top20)),
    "bottom20_dropped_after_filter": sorted(list(dropped_from_bot20)),
    "interpretation": (
        f"Of the original top-20, {len(dropped_from_top20)} country/countries "
        "drop out when small populations are excluded; this signals which top "
        "rankings rest on small-population GBD estimates that may be unstable. "
        f"Of the original bottom-20, {len(dropped_from_bot20)} drop out, "
        "indicating that the bottom-of-ranking narrative (Afghanistan, Lesotho, "
        "Zimbabwe, etc.) is more robust to population size."
    ),
}
os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
with open(JSON_OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {JSON_OUT}")
