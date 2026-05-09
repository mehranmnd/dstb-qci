# GATHER Checklist for Paper 1

## Guidelines for Accurate and Transparent Health Estimates Reporting (GATHER)

**Paper:** Global, Regional, and National Quality of Care Index for Drug-Susceptible Tuberculosis, 1990-2021: A Systematic Analysis of the Global Burden of Disease Study

| # | Item | Reported | Location |
|---|------|----------|----------|
| 1 | Define the indicator(s), population(s), and time period(s) for which estimates were made. | Yes | Abstract; Methods, Data source and scope (para 1); Methods, Component ratios |
| 2 | List the funding sources for the work. | Yes | Abstract ("Funding: None"); Role of the funding source section |
| 3 | Describe how the data were identified and how the data were accessed. | Yes | Methods, Data source and scope (para 1); Data sharing statement |
| 4 | Specify the inclusion and exclusion criteria. Identify all ad-hoc exclusions. | Yes | Methods, Data source and scope (204 countries, 195 with complete data); Methods, Sensitivity analysis (PERtoINC exclusion) |
| 5 | Provide information on all included data sources and their main characteristics. For each data source used, report reference information or contact name/institution, population represented, data collection method, year(s) of data collection, sex and age range, diagnostic criteria or measurement method, and sample size, as relevant. | Yes | Methods, Data source and scope (GBD 2021 methodology references 7-9); 204 countries, 1990-2021, 5 age groups, 3 sex categories |
| 6 | Provide a detailed description of the analytical methods used. This description should cover: mathematical formulae, computational methods, data adjustments, methods for handling missing values, model validation, and uncertainty quantification. | Yes | Methods, Component ratios; Methods, PCA; Methods, Uncertainty quantification; Methods, Trend analysis; Methods, Validation |
| 7 | Describe methods for calculating uncertainty of the estimates. State which sources of uncertainty were, and were not, accounted for in the uncertainty analysis. | Partially | Methods, Uncertainty quantification (Monte Carlo 1000 draws, SD derivation from GBD UIs); Discussion, Strengths and limitations (limitation ii on model-based estimates and the normal-approximation caveat; limitation x on the GBD-modelled-data structure). Sources of uncertainty NOT accounted for: (a) covariance between the six GBD input measures (Deaths, Incidence, YLLs, YLDs, DALYs, Prevalence) was treated as independent, which underestimates true uncertainty; (b) the symmetric normal approximation derived from GBD 95\% UI bounds may underestimate tail uncertainty in settings with asymmetric GBD UIs; (c) GBD's structural modelling uncertainty (DisMod-MR 2.1 model specification) is not propagated through the QCI; (d) uncertainty in the PCA loadings themselves (treated as fixed parameters) is not propagated. |
| 8 | State how analytic and/or statistical choices were made, noting which choices could affect estimates. | Yes | Methods, Sensitivity analysis (3 vs 4 components); Methods, PCA (sign correction, scaling); Discussion, Limitations (iv - PCA weighting) |
| 9 | Provide the main results of the analysis, including measures of uncertainty. | Yes | Results (all subsections); Tables 1-4; Figures 1-9; 95% UIs reported throughout |
| 10 | Provide a comparison with previously published estimates, if possible. | Yes | Discussion, Comparison with existing metrics (HAQ Index, WHO treatment success rate) |
| 11 | Discuss limitations of the estimates. Include a discussion of any modelling assumptions or data limitations that affect interpretation. | Yes | Discussion, Strengths and limitations (7 limitations enumerated) |
| 12 | State how the analytic code and data set can be accessed. | Partially | Data sharing statement: input data are publicly available via the GBD Results Tool. The analytic code repository URL is a placeholder pending pre-submission archival to Zenodo; this is tracked as a pre-submission TODO and item 12 should be upgraded to "Yes" once the Zenodo DOI is inserted. |

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| GBD 2021 | Global Burden of Disease Study 2021, Institute for Health Metrics and Evaluation (IHME) | GBD Results Tool (https://vizhub.healthdata.org/gbd-results/) |

## Indicators

| Indicator | Definition | Source |
|-----------|------------|--------|
| MIR | Mortality-to-Incidence Ratio for DS-TB | GBD 2021 (Deaths/Incidence) |
| YLL/YLD | Years of Life Lost to Years Lived with Disability ratio | GBD 2021 |
| DALY/Prevalence | DALY-to-Prevalence ratio | GBD 2021 |
| QCI | Quality of Care Index: first principal component of standardised (MIR, YLL/YLD, DALY/Prevalence), scaled 0-100 | Derived; PC1 explains 87.98% of variance |

## Geographic Coverage

- **Countries/territories:** 204 (195 with complete temporal data)
- **Time period:** 1990-2021 (32 annual observations)
- **Age groups:** <5 years, 5-14 years, 15-49 years, 50-69 years, 70+ years, age-standardised, all ages
- **Sex:** Both sexes, Male, Female

## Statistical Methods

| Method | Application | Software |
|--------|-------------|----------|
| PCA | QCI construction from 3 component ratios | Python scikit-learn |
| Monte Carlo simulation | Uncertainty propagation (1000 draws) | Python numpy/scipy |
| Log-linear regression | AAPC with 95% CI | Python scipy.stats |
| Spearman/Pearson correlation | Construct validation against SDI | Python scipy.stats |
| Min-max normalisation | QCI scaling to 0-100 | Python numpy |
| Sensitivity analysis | 3-component vs 4-component PCA comparison | Python scikit-learn |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| PCA variance explained (PC1) | 87.98% |
| QCI scaling | Min-max to 0-100 (higher = better) |
| Monte Carlo draws | 1000 |
| Significance threshold | p < 0.05 |
| AAPC confidence interval | 95% (t-distribution) |
| Sensitivity analysis (3 vs 4 component rho) | 0.998 |

## Sensitivity Analyses

1. **Component selection:** Compared 3-component and 4-component PCA models. PERtoINC contributed only 0.37% to PC1 variance; Spearman rho between models = 0.998, with 19/20 top-ranked and 20/20 bottom-ranked countries overlapping. Three-component model selected for parsimony.

## Code Availability

Analysis code will be made available in a public repository upon publication. All analyses performed in Python 3 using:
- pandas, numpy, scipy for data management and statistics
- matplotlib, seaborn for visualisation
- scikit-learn for PCA
- geopandas for spatial visualisation
