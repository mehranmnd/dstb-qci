#!/usr/bin/env bash
#
# run_all.sh — End-to-end Python pipeline for the DS-TB QCI thesis.
#
# Each step is a single `python3` invocation. The script aborts on the
# first failure (set -e). Downstream steps idempotently regenerate their
# outputs, so a partial run can be retried by re-invoking this script.
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Required input files (download per README.md; not in the git repo):
#   data/ihme.csv              GBD 2021 bulk DS-TB results
#   data/HAQ.CSV               GBD HAQ Index file
#   data/SDI_1950_2021.csv     GBD Socio-demographic Index 1950-2021
#   data/iran_shapefile/iran_provinces.geojson
#   data/who_tb_outcomes.csv   WHO TB Treatment Outcomes (optional;
#                              step 07 is skipped if absent)
#
# Required upstream outputs (produced by notebooks 02-03; see README):
#   results/shared/qci.csv
#   results/shared/qci_complete_data.csv
#
# Outputs:
#   results/shared/            shared CSVs (aapc, uncertainty, population, ...)
#   results/paper{1,2,3}/      per-paper figures, tables, summary JSONs

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3 -u}"
CODE=code

echo "================================================================"
echo "  DS-TB QCI - Python pipeline"
echo "  Started:   $(date -Iseconds)"
echo "================================================================"

# -- Sanity checks --------------------------------------------------------
for f in data/ihme.csv data/HAQ.CSV data/SDI_1950_2021.csv \
         data/iran_shapefile/iran_provinces.geojson \
         results/shared/qci.csv results/shared/qci_complete_data.csv; do
    if [ ! -f "$f" ]; then
        echo "MISSING input: $f"
        echo "  See README.md for download instructions and notebook prerequisites."
        exit 1
    fi
done
echo "Input files present."

run() {
    local script="$1"
    echo
    echo ">> $script"
    $PY "$CODE/$script"
}

# -- QCI-level analyses ---------------------------------------------------
run 01_aapc_analysis.py
run 02_qci_uncertainty.py
run 03_extract_population.py
run 04_pca_sensitivity_iran_vs_global.py
run 05_pca_3v4_comparison.py

# -- Validation against external indicators -------------------------------
run 06_qci_vs_haq_validation.py
if [ -f data/who_tb_outcomes.csv ]; then
    run 07_qci_vs_who_tsr_validation.py
else
    echo
    echo "SKIPPED: 07_qci_vs_who_tsr_validation.py (data/who_tb_outcomes.csv absent)"
fi

# -- QCI sensitivity analyses --------------------------------------------
run 08_qci_logit_sensitivity.py
run 09_global_smallpop_sensitivity.py
run 10_qci_joinpoint.py

# -- Equity: inequality & multilevel -------------------------------------
run 11_equity_analysis.py
run 12_equity_pop_weighted_ci.py
run 13_equity_halflife_sensitivity.py

# -- Global & equity figures ---------------------------------------------
run 14_global_figures.py
run 15_equity_figures.py

# -- Iran subnational ----------------------------------------------------
run 16_iran_analysis.py
run 17_iran_spatial_decomposition.py
run 18_iran_morans_shuffle.py
run 19_iran_shapley_robust.py

# -- Tables --------------------------------------------------------------
run 20_generate_tables.py
run 21_generate_supplementary_table.py

echo
echo "================================================================"
echo "  Pipeline complete: $(date -Iseconds)"
echo "  Outputs in:"
echo "    results/shared/"
echo "    results/global/, results/equity/, results/iran/"
echo "================================================================"
