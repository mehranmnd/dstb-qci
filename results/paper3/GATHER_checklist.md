# GATHER Checklist for Paper 3

## Guidelines for Accurate and Transparent Health Estimates Reporting (GATHER)

**Paper:** Subnational Quality of Care for Drug-Susceptible Tuberculosis in Iran: A Provincial Analysis, 1990-2021

| # | Item | Reported | Location |
|---|------|----------|----------|
| 1 | Define the indicator(s), population(s), and time period(s) for which estimates were made. | Yes | Abstract; Methods, Study Design (para 1) |
| 2 | List the funding sources for the work. | Yes | Acknowledgments |
| 3 | Describe how the data were identified and how the data were accessed. | Yes | Methods, Data Sources (para 1) |
| 4 | Specify the inclusion and exclusion criteria. Identify all ad-hoc exclusions. | Yes | Methods, Data Sources (para 1-2) |
| 5 | Provide information on all included data sources and their main characteristics. For each data source used, report reference information or contact name/institution, population represented, data collection method, year(s) of data collection, sex and age range, diagnostic criteria or measurement method, and sample size, as relevant. | Yes | Methods, Data Sources; GBD 2021 methodology references |
| 6 | Provide a detailed description of the analytical methods used. This description should cover: mathematical formulae, computational methods, data adjustments, methods for handling missing values, model validation, and uncertainty quantification. | Yes | Methods, QCI Construction (para 2-3); Methods, Statistical Analysis (para 1-3) |
| 7 | Describe methods for calculating uncertainty of the estimates. State which sources of uncertainty were, and were not, accounted for in the uncertainty analysis. | Partially | Methods, Statistical Analysis (AAPC CIs from t-distribution; Moran's I significance from 999 permutations); Discussion, Strengths and limitations. Sources of uncertainty NOT accounted for: (a) provincial-level GBD UIs are not propagated through the QCI for Iran's provinces; (b) Shapley decomposition point estimates are not accompanied by bootstrap CIs; (c) the Moran's I estimates partly reflect GBD's covariate-based spatial smoothing rather than only true spatial heterogeneity (acknowledged in Discussion). |
| 8 | State how analytic and/or statistical choices were made, noting which choices could affect estimates. | Yes | Methods, Statistical Analysis; Discussion |
| 9 | Provide the main results of the analysis, including measures of uncertainty. | Yes | Results, all subsections; Tables 1-8; Figures 1-13 |
| 10 | Provide a comparison with previously published estimates, if possible. | Yes | Discussion, Comparison with Literature (para 1-2); Table 5 (MENA comparison) |
| 11 | Discuss limitations of the estimates. Include a discussion of any modelling assumptions or data limitations that affect interpretation. | Yes | Discussion, Limitations (para 1-5) |
| 12 | State how the analytic code and data set can be accessed. | Partially | Input data are publicly available via the GBD Results Tool. The analytic code repository URL is a placeholder pending pre-submission archival to Zenodo; this is tracked as a pre-submission TODO. |

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| GBD 2021 | Global Burden of Disease Study 2021, Institute for Health Metrics and Evaluation (IHME) | GBD Results Tool (https://vizhub.healthdata.org/gbd-results/) |
| Natural Earth | Admin-1 boundaries for Iran provinces (v5.1.1, 1:10m cultural) | https://www.naturalearthdata.com/ |

## Indicators

| Indicator | Definition | Source |
|-----------|------------|--------|
| MIR | Mortality-to-Incidence Ratio for DS-TB | GBD 2021 (Deaths/Incidence) |
| YLL/YLD | Years of Life Lost to Years Lived with Disability ratio | GBD 2021 |
| DALY/Prevalence | DALY-to-Prevalence ratio | GBD 2021 |
| QCI | Quality of Care Index: first principal component of standardized (MIR, YLL/YLD, DALY/Prevalence), scaled 0-100 | Derived; PC1 explains 87.98% of variance |

## Geographic Coverage

- **Country:** Iran
- **Subnational units:** 31 provinces (ostans)
- **Time period:** 1990-2021 (32 annual observations)
- **Age groups:** Age-standardized, <5 years, 5-14 years, 15-49 years, 50-69 years, 70+ years
- **Sex:** Both sexes, Male, Female

## Statistical Methods

| Method | Application | Software |
|--------|-------------|----------|
| PCA | QCI construction from 3 component ratios | Python scikit-learn |
| Log-linear regression | AAPC with 95% CI | Python scipy.stats |
| Coefficient of Variation | Provincial inequality over time | Python numpy |
| Gini coefficient | Provincial inequality over time | Python numpy |
| Beta-convergence | Initial QCI vs change (catching-up) | Python scipy.stats |
| Sigma-convergence | SD trend over time (narrowing disparity) | Python scipy.stats |
| Gender Disparity Ratio | Female/Male QCI ratio by province | Python pandas |
| Global Moran's I | Spatial autocorrelation (Queen contiguity + KNN-5) | Python esda/libpysal |
| Local Moran's I (LISA) | Local spatial clusters (HH, LL, HL, LH) | Python esda |
| Shapley decomposition | Attribution of QCI gap to MIR, YLL/YLD, DALY/Prev | Python custom implementation |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| PCA variance explained (PC1) | 87.98% |
| QCI scaling | Min-max to 0-100 (higher = better) |
| Significance threshold | p < 0.05 |
| AAPC confidence interval | 95% (t-distribution) |
| Moran's I permutations | 999 |
| LISA significance threshold | p < 0.05 |
| Spatial weights | Queen contiguity (primary), KNN k=5 (sensitivity) |

## Sensitivity Analyses

1. **Spatial weights sensitivity:** Moran's I computed with both Queen contiguity (mean 4.8 neighbors) and KNN-5 weights; results consistent across both specifications.
2. **AAPC periods:** Computed for full period (1990-2021) and recent period (2010-2021) to assess trend changes.
3. **Decomposition model:** Linear model R-squared = 1.0000, confirming PCA linearity preserved in decomposition.

## Code Availability

Analysis code is available at request from the corresponding author. All analyses performed in Python 3 using:
- pandas, numpy, scipy for data management and statistics
- matplotlib, seaborn for visualization
- geopandas, shapely for spatial data handling
- esda, libpysal for spatial analysis
- scikit-learn for PCA
