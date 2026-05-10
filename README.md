# DS-TB Quality of Care Index — Thesis Repository

This repository contains the analytic code, paper manuscripts, and intermediate results for a three-paper thesis on a population-level Quality of Care Index (QCI) for drug-susceptible tuberculosis, derived from Global Burden of Disease (GBD) 2021 estimates.

## Directory layout

```
thesis/
├── code/             Python scripts that produce every analysis output
│   └── notebooks/    Older Jupyter notebooks (kept for reference)
├── data/             Input data (most files gitignored — see .gitignore)
│   ├── ihme.csv                 GBD 2021 bulk DS-TB results (~1.5 GB)
│   ├── HAQ.CSV                  GBD HAQ Index (1990-2019, all causes)
│   ├── SDI_1950_2021.csv        GBD Socio-demographic Index
│   ├── who_tb_outcomes.csv      WHO TB Treatment Outcomes (download instructions in code/qci_vs_who_tsr_validation.py)
│   ├── iran_shapefile/          Iran province boundaries (GeoJSON)
│   └── ne_110m/                 Natural Earth country boundaries
├── docs/             Thesis drafts (PDF/RTF), proposal, audit notes
├── papers/           Canonical paper sources (one directory per paper)
│   ├── paper1/       main.tex, cover_letter.tex, supplementary.tex,
│   │                 GATHER_checklist.md
│   ├── paper2/       main.tex, GATHER_checklist.md
│   └── paper3/       main.tex, GATHER_checklist.md
├── results/          Analysis outputs (one directory per paper plus shared)
│   ├── shared/       qci.csv, qci_uncertainty.csv, aapc_results.csv,
│   │                 population_2021.csv, qci_complete_data.csv
│   ├── paper1/       figures/, tables/, summary JSONs
│   ├── paper2/       figures/, tables/, analysis/ (per-paper JSON outputs)
│   └── paper3/       figures/, tables/, analysis/, stats.json
├── archive.zip       Frozen snapshot of the previous Stata/notebook workflow
├── requirements.txt  Python dependencies
├── run_all.sh        End-to-end reproducibility pipeline
├── CHANGELOG.md      Append-only log of all automated changes (2026-05 fix plan)
├── CITATIONS_VERIFIED.md  PMID/DOI verification log for every new citation
├── FIX_PROGRESS.md   Per-task state of the 2026-05 fix plan
├── TODO_FOR_USER.md  Items the author must complete before submission
└── README.md         This file
```

## Where the figures live

Each paper's `main.tex` uses `\graphicspath{{../../results/paperN/figures/}}` so that figure files live in one place — alongside the analysis outputs that produced them — rather than being duplicated next to the manuscript source. To compile a paper:

```bash
cd papers/paper1
pdflatex main.tex
bibtex main          # if needed; bibliography is currently inline
pdflatex main.tex
pdflatex main.tex
```

## Reproducing the analyses

The full pipeline runs from raw inputs to all figures, tables, and summary files:

```bash
./run_all.sh
```

Each step in `run_all.sh` is a single Python script under `code/`; the script is intentionally simple so individual steps can be re-run on their own. Inputs that are gitignored (`data/ihme.csv` and similar bulk downloads) need to be present locally.

## What the 2026-05 fix plan changed

A comprehensive multi-phase fix pass landed in May 2026; see `CHANGELOG.md` for the full record (~25 commits). Highlights:

- **Phase 1 (reframing & cleanup)**: scope/collinearity disclosure, "modelled-not-measured" caveats, causal-language scrub, honest GATHER updates, log-normal Monte Carlo uncertainty propagation, dual-source SDI bug eliminated.
- **Phase 2 (sensitivities & new analyses)**: HAQ Index validation, population-weighted concentration index, logit-transformed AAPCs, small-population sensitivity, joinpoint regression, half-life sensitivity (post-2005), Moran's I shuffle baseline, robust top-5 Shapley reference.
- **Phase 3 (verified WebFetch)**: WHO Treatment Success Rate validation (negative correlation interpreted as complementary, not redundant).
- **Phase 4 (manuscript integration)**: all sensitivities folded into Methods, Results, and Discussion sections of the relevant papers.
- **Phase 5 (handoff)**: `run_all.sh`, `TODO_FOR_USER.md`, word-count audit.

## Single-target-journal policy

The repository deliberately holds **only one** version of each paper. Previous per-journal copies (`paper{1,2,3}/{bmj_global_health,...}/`, `submissions/paperN_*/`) were removed on 2026-05-09 to prevent drift between the canonical source and stale derivatives. When you choose a target journal:

1. Copy `papers/paperN/main.tex` to a working directory.
2. Reformat references, section structure, and length to that journal's house style.
3. Submit; do **not** check the journal-specific version back into the canonical `papers/paperN/` location.

If the paper is rejected and you target a different journal, repeat the process from the canonical source.

## Tracking your edits

The repository is git-tracked. The fix-plan commits are tagged `[fix-N.N]` in the commit log. When you complete items from `TODO_FOR_USER.md`, please commit them with a corresponding message tag (e.g., `[user-todo] author list finalised`).
