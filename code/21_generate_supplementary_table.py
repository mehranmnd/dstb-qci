#!/usr/bin/env python3
"""
Generate Supplementary Table S3: All-country QCI ranking table in LaTeX longtable format.

Reads aapc_results.csv, filters to countries, sorts by QCI 2021 descending,
and outputs a LaTeX longtable file for inclusion in supplementary.tex.

Usage:
    python generate_supplementary_table.py
"""

import sys
import os
import pandas as pd

# Add code directory to path so we can import mappings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mappings import SDI_COUNTRY_MAPPING

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE, "results/shared/aapc_results.csv")
OUTPUT_PATH = os.path.join(BASE, "results/paper1/tables/table_s3_all_countries.tex")

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT_PATH)
print(f"  Total rows: {len(df)}")

# Filter to countries only
countries = df[df["category"] == "COUNTRY"].copy()
print(f"  Country rows: {len(countries)}")

# Sort by QCI 2021 descending
countries = countries.sort_values("qci_2021", ascending=False).reset_index(drop=True)

# Add rank
countries["rank"] = range(1, len(countries) + 1)

# Map SDI group
countries["sdi_group"] = countries["location"].map(SDI_COUNTRY_MAPPING).fillna("--")

# Compute change
countries["change"] = countries["qci_2021"] - countries["qci_1990"]

# Format the 95% CI for the full-period AAPC
countries["ci_str"] = countries.apply(
    lambda r: f"({r['ci_lo_full']:.2f}, {r['ci_hi_full']:.2f})", axis=1
)

# ---------------------------------------------------------------------------
# Helper: escape LaTeX special characters in location names
# ---------------------------------------------------------------------------
def latex_escape(s):
    """Escape characters that are special in LaTeX."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, repl in replacements.items():
        s = s.replace(char, repl)
    return s


# ---------------------------------------------------------------------------
# Build LaTeX longtable
# ---------------------------------------------------------------------------
print("Generating LaTeX table...")

lines = []

# Preamble
lines.append(r"\begin{longtable}{r l r r r r l l}")
lines.append(r"\caption{Complete country ranking by Quality of Care Index in 2021 "
             r"(age-standardised, both sexes), with QCI in 1990, absolute change, "
             r"Average Annual Percentage Change (AAPC) over 1990--2021, and SDI group.}")
lines.append(r"\label{tab:all_countries} \\")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Rank} & \textbf{Country} & \textbf{QCI 2021} & "
    r"\textbf{QCI 1990} & \textbf{Change} & \textbf{AAPC} & "
    r"\textbf{95\% CI} & \textbf{SDI Group} \\"
)
lines.append(r"\midrule")
lines.append(r"\endfirsthead")

# Continuation header
lines.append(r"")
lines.append(r"\multicolumn{8}{l}{\textit{Table~S3 continued from previous page}} \\")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Rank} & \textbf{Country} & \textbf{QCI 2021} & "
    r"\textbf{QCI 1990} & \textbf{Change} & \textbf{AAPC} & "
    r"\textbf{95\% CI} & \textbf{SDI Group} \\"
)
lines.append(r"\midrule")
lines.append(r"\endhead")

# Continuation footer
lines.append(r"")
lines.append(r"\midrule")
lines.append(r"\multicolumn{8}{r}{\textit{Continued on next page}} \\")
lines.append(r"\endfoot")

# Last footer
lines.append(r"")
lines.append(r"\bottomrule")
lines.append(r"\endlastfoot")

# Data rows
for _, row in countries.iterrows():
    rank = int(row["rank"])
    country = latex_escape(str(row["location"]))
    qci_2021 = f'{row["qci_2021"]:.2f}'
    qci_1990 = f'{row["qci_1990"]:.2f}'
    change = f'{row["change"]:.2f}'
    aapc = f'{row["aapc_full"]:.2f}'
    ci = row["ci_str"]
    sdi = latex_escape(str(row["sdi_group"]))

    lines.append(
        f"{rank} & {country} & {qci_2021} & {qci_1990} & "
        f"{change} & {aapc} & {ci} & {sdi} \\\\"
    )

# End table
lines.append(r"")
lines.append(r"\end{longtable}")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\n")

print(f"  Output written to: {OUTPUT_PATH}")
print(f"  Total countries in table: {len(countries)}")
print("Done.")
