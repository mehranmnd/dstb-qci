#!/usr/bin/env bash
#
# run_all.sh — End-to-end reproducibility script for the DS-TB QCI thesis.
#
# Runs every analysis script in dependency order. Assumes the input data
# files at the paths listed below exist (these are the GBD/IHME bulk
# downloads and ancillary CSVs; they are not included in the git repo
# because of size — see .gitignore).
#
# Inputs expected at:
#   data/ihme.csv              GBD 2021 bulk DS-TB results (~1.5GB)
#   data/HAQ.CSV               GBD HAQ Index file
#   data/SDI_1950_2021.csv     GBD Socio-demographic Index 1950-2021
#   data/who_tb_outcomes.csv   WHO TB Treatment Outcomes (download via:
#                              curl -o data/who_tb_outcomes.csv \
#                                "https://extranet.who.int/tme/generateCSV.asp?ds=outcomes")
#   data/iran_shapefile/iran_provinces.geojson
#
# Outputs written to:
#   results/shared/            shared CSVs (qci.csv, qci_uncertainty.csv,
#                              aapc_results.csv, population_2021.csv, etc.)
#   results/paper{1,2,3}/      per-paper figures, tables, JSON summaries
#
# This script is intentionally simple: each step is a single `python3`
# invocation. If any step fails, the script aborts (set -e). Re-run the
# whole script after fixing the failure; downstream steps idempotently
# regenerate their outputs.

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3 -u}"

echo "================================================================"
echo "  DS-TB QCI — full reproducibility pipeline"
echo "  Started:   $(date -Iseconds)"
echo "================================================================"

# -- 0. Sanity checks ---------------------------------------------------------
for f in data/ihme.csv data/HAQ.CSV data/SDI_1950_2021.csv \
         data/iran_shapefile/iran_provinces.geojson \
         results/shared/qci.csv results/shared/qci_complete_data.csv; do
    if [ ! -f "$f" ]; then
        echo "MISSING input: $f"
        exit 1
    fi
done
echo "Input files present."

# -- 1. PCA model artefact (regenerable from qci_complete_data.csv) -----------
# Built on demand by qci_uncertainty.py if not present.

# -- 2. AAPC analysis ---------------------------------------------------------
echo; echo ">> AAPC analysis ..."
$PY code/aapc_analysis.py

# -- 3. QCI uncertainty (Monte Carlo, log-normal sampling) -------------------
echo; echo ">> QCI uncertainty propagation (Monte Carlo) ..."
$PY code/qci_uncertainty.py

# -- 4. Population extraction (from ihme.csv) --------------------------------
echo; echo ">> Population extraction (Number/Rate from ihme.csv) ..."
$PY code/extract_population.py

# -- 5. PCA sensitivity (Iran vs global) -------------------------------------
echo; echo ">> PCA sensitivity (Iran-only vs global) ..."
$PY code/pca_sensitivity_iran_vs_global.py

# -- 6. PCA 3 vs 4 component comparison --------------------------------------
echo; echo ">> PCA 3 vs 4 component comparison ..."
$PY code/pca_3v4_comparison.py

# -- 7. Validation: QCI vs HAQ Index -----------------------------------------
echo; echo ">> Validation: QCI vs HAQ Index ..."
$PY code/qci_vs_haq_validation.py

# -- 8. Validation: QCI vs WHO TSR -------------------------------------------
if [ -f data/who_tb_outcomes.csv ]; then
    echo; echo ">> Validation: QCI vs WHO TSR ..."
    $PY code/qci_vs_who_tsr_validation.py
else
    echo; echo "SKIPPED: data/who_tb_outcomes.csv missing (run the curl in the header)."
fi

# -- 9. Logit-transformed AAPC sensitivity -----------------------------------
echo; echo ">> Logit-AAPC ceiling-effect sensitivity ..."
$PY code/qci_logit_sensitivity.py

# -- 10. Small-population sensitivity ----------------------------------------
echo; echo ">> Small-population sensitivity (Paper 1 rankings) ..."
$PY code/paper1_smallpop_sensitivity.py

# -- 11. Joinpoint regression ------------------------------------------------
echo; echo ">> Joinpoint regression (Paper 1 focus locations) ..."
$PY code/qci_joinpoint.py

# -- 12. Paper 2 analysis (Concentration Index, SII, Theil, multilevel) ------
echo; echo ">> Paper 2 main analysis ..."
$PY code/paper2_analysis.py

# -- 13. Paper 2 population-weighted CI sensitivity --------------------------
echo; echo ">> Paper 2 population-weighted CI sensitivity ..."
$PY code/paper2_pop_weighted_ci.py

# -- 14. Paper 2 half-life sensitivity ---------------------------------------
echo; echo ">> Paper 2 half-life sensitivity (post-2005 etc) ..."
$PY code/paper2_halflife_sensitivity.py

# -- 15. Paper 1 figures -----------------------------------------------------
echo; echo ">> Paper 1 figures ..."
$PY code/paper1_figures.py

# -- 16. Paper 2 figures -----------------------------------------------------
echo; echo ">> Paper 2 figures ..."
$PY code/paper2_figures.py

# -- 17. Paper 3 main analysis ----------------------------------------------
echo; echo ">> Paper 3 main analysis ..."
$PY code/paper3_analysis.py

# -- 18. Paper 3 spatial decomposition (Moran's I + LISA + Shapley) ---------
echo; echo ">> Paper 3 spatial decomposition ..."
$PY code/paper3_spatial_decomposition.py

# -- 19. Paper 3 Moran's shuffle baseline ------------------------------------
echo; echo ">> Paper 3 Moran's shuffle baseline ..."
$PY code/paper3_morans_shuffle.py

# -- 20. Paper 3 robust Shapley reference -----------------------------------
echo; echo ">> Paper 3 robust top-5 Shapley reference ..."
$PY code/paper3_shapley_robust.py

# -- 21. Tables / supplementary ---------------------------------------------
echo; echo ">> Generate tables ..."
$PY code/generate_tables.py
$PY code/generate_supplementary_table.py

echo
echo "================================================================"
echo "  Pipeline complete: $(date -Iseconds)"
echo "  Outputs in:"
echo "    results/shared/"
echo "    results/paper1/, results/paper2/, results/paper3/"
echo "================================================================"
