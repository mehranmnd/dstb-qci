# DS-TB Quality of Care Index

Analytic code for a three-paper thesis on a population-level Quality of Care Index (QCI) for drug-susceptible tuberculosis, derived from Global Burden of Disease (GBD) 2021 estimates.

- **Global** — global QCI, 204 countries, 1990-2021
- **Equity** — global inequality (concentration index, Theil, multilevel models)
- **Iran** — Iran subnational analysis (31 provinces, spatial decomposition)

Figures, tables, and the manuscripts themselves are withheld from this archive until peer-reviewed publication. Running the pipeline against a local copy of the GBD 2021 inputs regenerates every figure and every derived table used in the manuscripts.

## Repository contents

```
.
├── code/
│   ├── notebooks/                            Stage 0 — QCI construction from raw IHME data
│   │   ├── 01_creating_ihme_data.ipynb
│   │   ├── 02_cleaning_data.ipynb
│   │   ├── 03_pca_analysis.ipynb             produces results/shared/qci_complete_data.csv
│   │   ├── 04_world_plot.ipynb
│   │   ├── 05_statistical_analysis.ipynb
│   │   └── 06_reports.ipynb
│   ├── mappings.py                           Shared lookup constants
│   ├── 01_aapc_analysis.py                   Stage 1 — QCI-level analyses
│   ├── 02_qci_uncertainty.py                 Monte Carlo uncertainty propagation
│   ├── 03_extract_population.py
│   ├── 04_pca_sensitivity_iran_vs_global.py
│   ├── 05_pca_3v4_comparison.py
│   ├── 06_qci_vs_haq_validation.py           Stage 2 — external validation
│   ├── 07_qci_vs_who_tsr_validation.py
│   ├── 08_qci_logit_sensitivity.py           Stage 3 — sensitivity analyses
│   ├── 09_global_smallpop_sensitivity.py
│   ├── 10_qci_joinpoint.py
│   ├── 11_equity_analysis.py                 Stage 4 — equity (global inequality)
│   ├── 12_equity_pop_weighted_ci.py
│   ├── 13_equity_halflife_sensitivity.py
│   ├── 14_global_figures.py                  Stage 5 — figures
│   ├── 15_equity_figures.py
│   ├── 16_iran_analysis.py                   Stage 6 — Iran subnational
│   ├── 17_iran_spatial_decomposition.py
│   ├── 18_iran_morans_shuffle.py
│   ├── 19_iran_shapley_robust.py
│   ├── 20_generate_tables.py                 Stage 7 — tables (global)
│   └── 21_generate_supplementary_table.py
├── requirements.txt
├── run_all.sh                                One-shot pipeline for Stages 1-7
├── LICENSE
└── README.md
```

Each script writes its outputs under one of:

- `results/shared/` — files consumed by more than one downstream script (qci.csv, qci_uncertainty.csv, aapc_results.csv, population_2021.csv, …)
- `results/global/` — figures/tables/JSON for the global QCI analysis (Stages 3, 5, 7)
- `results/equity/` — figures/tables/JSON for the equity analysis
- `results/iran/`   — figures/tables/JSON for the Iran subnational analysis

## Reproducing the analyses

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain GBD 2021 data

Register at the [GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) and download the DS-TB extract under IHME's Free-of-Charge Non-Commercial User Agreement. The pipeline expects the following files (none are included in the repo):

| Path | Source |
|---|---|
| `data/ihme.csv` | GBD 2021 DS-TB bulk results (204 countries x 32 years x 6 measures x 5 age groups x both sexes) |
| `data/HAQ.CSV` | GBD HAQ Index (1990-2019, all causes) |
| `data/SDI_1950_2021.csv` | GBD Socio-demographic Index |
| `data/iran_shapefile/iran_provinces.geojson` | Iran province boundaries |
| `data/who_tb_outcomes.csv` | WHO TB Treatment Outcomes (`curl -o data/who_tb_outcomes.csv "https://extranet.who.int/tme/generateCSV.asp?ds=outcomes"`) — optional; step 07 is skipped if absent |

### 3. Stage 0 — build the QCI

Run the six notebooks under `code/notebooks/` in order (`01_` through `06_`). They cleanse the raw IHME bulk extract, fit the PCA, and write the canonical QCI table to `results/shared/qci.csv` plus its full feature matrix to `results/shared/qci_complete_data.csv`. Stages 1-7 depend on these two files.

### 4. Stages 1-7 — analyses, figures, tables

```bash
./run_all.sh
```

Each step is a single `python3` invocation. The script aborts on the first failure (`set -e`); downstream steps idempotently regenerate their outputs, so a partial run can be retried by re-invoking the script. Individual scripts can also be run directly (`python3 code/14_global_figures.py`) once their upstream inputs exist.

## License

Code is released under the MIT license — see [LICENSE](LICENSE). The license covers the code in this repository only; the GBD inputs are governed by IHME's Free-of-Charge Non-Commercial User Agreement.
