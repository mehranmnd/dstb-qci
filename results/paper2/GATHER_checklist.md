# GATHER Checklist for Paper 2

## Guidelines for Accurate and Transparent Health Estimates Reporting (GATHER)

**Paper:** Inequality in Drug-Susceptible Tuberculosis Quality of Care: A Decomposition and Convergence Analysis Across 195 Countries, 1990-2021

| # | Item | Reported | Location |
|---|------|----------|----------|
| 1 | Define the indicator(s), population(s), and time period(s) for which estimates were made. | Yes | Abstract; Methods, Data source and study population (para 1) |
| 2 | List the funding sources for the work. | Yes | Abstract ("Funding: None") |
| 3 | Describe how the data were identified and how the data were accessed. | Yes | Methods, Data source and study population (para 1); Data sharing statement |
| 4 | Specify the inclusion and exclusion criteria. Identify all ad-hoc exclusions. | Yes | Methods, Data source and study population (195 countries with both QCI and SDI values; exclusion of subnational entities and aggregate regions) |
| 5 | Provide information on all included data sources and their main characteristics. For each data source used, report reference information or contact name/institution, population represented, data collection method, year(s) of data collection, sex and age range, diagnostic criteria or measurement method, and sample size, as relevant. | Yes | Methods, Data source and study population (QCI from Paper 1; GBD 2021; SDI); 195 countries, 1990-2021, 7 age groups, 3 sex categories |
| 6 | Provide a detailed description of the analytical methods used. This description should cover: mathematical formulae, computational methods, data adjustments, methods for handling missing values, model validation, and uncertainty quantification. | Yes | Methods (8 subsections: CI formula, SII/RII, Gini/Theil, Theil decomposition, Wagstaff decomposition, GDR, age-group inequality, convergence); Multilevel regression models |
| 7 | Describe methods for calculating uncertainty of the estimates. State which sources of uncertainty were, and were not, accounted for in the uncertainty analysis. | Partially | Methods, Concentration Index (jackknife SE, 95% CI); Methods, SII (OLS SE); Discussion, Limitations (vi - GBD uncertainty not propagated through inequality calculations) |
| 8 | State how analytic and/or statistical choices were made, noting which choices could affect estimates. | Yes | Methods (SDI as ranking variable; CI formula choice; Wagstaff decomposition specification); Discussion, Limitations (iii - CI sensitivity to ranking variable) |
| 9 | Provide the main results of the analysis, including measures of uncertainty. | Yes | Results (all subsections); Tables 1-5; Figures 1-8; 95% CIs for CI and SII |
| 10 | Provide a comparison with previously published estimates, if possible. | Yes | Discussion, Declining but slow convergence (comparison with health convergence literature); Introduction (equity measurement literature) |
| 11 | Discuss limitations of the estimates. Include a discussion of any modelling assumptions or data limitations that affect interpretation. | Yes | Discussion, Strengths and limitations (7 limitations enumerated) |
| 12 | State how the analytic code and data set can be accessed. | Yes | Data sharing statement (GBD Results Tool URL; code available upon publication) |

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| GBD 2021 | Global Burden of Disease Study 2021, IHME | GBD Results Tool (https://vizhub.healthdata.org/gbd-results/) |
| DS-TB QCI | Quality of Care Index (Paper 1) | Derived from GBD 2021 data |
| SDI | Socio-demographic Index, GBD 2021 | GBD Results Tool |
| Natural Earth | Country boundaries (110m resolution) | https://www.naturalearthdata.com/ |

## Indicators

| Indicator | Definition | Source |
|-----------|------------|--------|
| QCI | Quality of Care Index for DS-TB (0-100 scale) | Paper 1 |
| CI | Concentration Index of QCI ranked by SDI | Derived |
| SII | Slope Index of Inequality (QCI points) | Derived |
| RII | Relative Index of Inequality (SII/mean QCI) | Derived |
| Gini | Gini coefficient of QCI across countries | Derived |
| Theil T | Generalised entropy index (alpha=1) | Derived |
| GDR | Gender Disparity Ratio (Female QCI / Male QCI) | Derived |

## Geographic Coverage

- **Countries:** 195 (with QCI and SDI data)
- **Time period:** 1990-2021 (32 annual observations)
- **Age groups:** <5 years, 5-14 years, 15-49 years, 50-69 years, 70+ years, age-standardised
- **Sex:** Both sexes, Male, Female
- **Regions:** 7 World Bank regions; 5 SDI quintiles

## Statistical Methods

| Method | Application | Software |
|--------|-------------|----------|
| Concentration Index | Wealth-related inequality (SDI ranking) | Python numpy (covariance formula) |
| Jackknife SE | Standard errors for CI | Python numpy |
| SII/RII | Absolute/relative inequality gradient | Python scipy.stats (OLS) |
| Gini coefficient | Overall QCI dispersion | Python numpy |
| Theil T and L indices | Decomposable inequality measures | Python numpy |
| Theil decomposition | Between- vs within-region inequality | Python numpy |
| Wagstaff decomposition | CI decomposition by determinants | Python statsmodels (OLS) |
| Sigma-convergence | Declining cross-country SD over time | Python scipy.stats (linear regression) |
| Beta-convergence | Catch-up from lower baselines | Python scipy.stats (linear/log-linear regression) |
| Multilevel mixed-effects regression | SDI, sex, and age effects with country random intercepts | Python statsmodels (MixedLM) |
| Spearman correlation | GDR vs SDI association | Python scipy.stats |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Significance threshold | p < 0.05 |
| CI confidence interval | 95% (jackknife SE) |
| SII confidence interval | 95% (OLS SE) |
| Convergence half-life | 42.2 years |
| Wagstaff decomposition reference year | 2021 |
| Multilevel model random effect | Country-level random intercept |
| Multilevel model estimation | REML |

## Sensitivity Analyses

1. **Ranking variable:** SDI used as primary ranking variable for CI; alternative rankings acknowledged as limitation.
2. **Multiple inequality measures:** Gini, Theil, CI, SII, and RII computed to ensure robustness across different measurement approaches.
3. **Conditional beta-convergence:** Controlled for SDI to separate structural development effects from catch-up dynamics.

## Code Availability

Analysis code will be made available upon publication. All analyses performed in Python 3.11 using:
- pandas, numpy, scipy for data management and statistics
- statsmodels for mixed-effects regression and Wagstaff decomposition
- matplotlib, seaborn for visualisation
- geopandas for world map
- scikit-learn for PCA (QCI construction in Paper 1)
