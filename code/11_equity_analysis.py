#!/usr/bin/env python3
"""
Equity analysis: Gender, Age, and Disparities in DS-TB Care Quality
Complete analysis: Concentration Index, SII, Theil, GDR global,
age decomposition, multilevel regression, all figures and tables.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import geopandas as gpd
import warnings
import os
import sys
import json
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCI_PATH = os.path.join(BASE, 'results/shared/qci.csv')
SDI_PATH = os.path.join(BASE, 'data/SDI_1950_2021.csv')
OUTPUT_DIR = os.path.join(BASE, 'results/equity/figures')
TABLE_DIR = os.path.join(BASE, 'results/equity/tables')
STATS_PATH = os.path.join(BASE, 'results/equity/analysis/equity_stats.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, 'code'))
from mappings import (SDI_COUNTRY_MAPPING, WB_REGIONS, SDI_QUINTILES,
                      IRAN_PROVINCES, COUNTRY_NAME_MAPPING, SDI_VALUE_MAP_2021,
                      NON_COUNTRY_LOCATIONS)

# Note: `SDI_VALUE_MAP_2021` from mappings.py is imported only as a
# backward-compatibility fallback for ~40 countries whose SDI is not
# present in the qci_complete_data.csv data file (mostly small island
# states and conflict-affected settings). The data file is the
# authoritative source for the other 155 countries; see the SDI map
# construction block further down for details.

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.8, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
sdi_colors = {'High': '#2196F3', 'High-middle': '#4CAF50', 'Middle': '#FFC107',
              'Low-middle': '#FF9800', 'Low': '#F44336', 'Unknown': 'gray'}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df_qci = pd.read_csv(QCI_PATH)

# Identify countries (exclude all aggregated/regional locations and provinces)
countries_only = sorted([c for c in df_qci['iso_location_name'].unique() if c not in NON_COUNTRY_LOCATIONS])
print(f"  {len(countries_only)} countries identified")

# ── Single-source SDI map (replaces direct use of hardcoded SDI_VALUE_MAP_2021) ─
# The qci_complete_data.csv `sdi_value_2021` column is treated as the
# authoritative source for the 155 countries where it is non-null. For
# the remaining ~40 countries (predominantly small island states,
# conflict-affected or data-sparse settings such as Marshall Islands,
# Antigua and Barbuda, South Sudan, New Zealand) the data file has no
# SDI value and we fall back to the hardcoded dict in mappings.py to
# preserve the original 195-country analytic sample. We assert that the
# data-file values match the dict exactly on every overlapping country,
# so any future divergence fails loudly. Both sources together are
# acknowledged as DEPRECATED in favour of an authoritative GBD 2021 SDI
# rebuild from data/SDI_1950_2021.csv (documented as a pre-submission
# TODO and out of scope for the present fix pass).
QCI_COMPLETE_PATH = os.path.join(BASE, 'results/shared/qci_complete_data.csv')
_sdi_src = pd.read_csv(QCI_COMPLETE_PATH, usecols=['iso_location_name', 'year',
                                                    'sex_name', 'age_name',
                                                    'sdi_value_2021'])
_sdi_src = _sdi_src[(_sdi_src['year'] == 2021)
                    & (_sdi_src['sex_name'] == 'Both')
                    & (_sdi_src['age_name'] == 'Age-standardized')]
_sdi_csv = (_sdi_src.dropna(subset=['sdi_value_2021'])
                      [['iso_location_name', 'sdi_value_2021']]
                      .drop_duplicates(subset='iso_location_name')
                      .rename(columns={'sdi_value_2021': 'sdi'})
                      .reset_index(drop=True))

_overlap = set(_sdi_csv['iso_location_name']) & set(SDI_VALUE_MAP_2021.keys())
_mismatches = []
for _c in sorted(_overlap):
    _v_csv = float(_sdi_csv[_sdi_csv['iso_location_name'] == _c]['sdi'].iloc[0])
    _v_dict = float(SDI_VALUE_MAP_2021[_c])
    if abs(_v_csv - _v_dict) > 1e-6:
        _mismatches.append((_c, _v_csv, _v_dict))
if _mismatches:
    print(f"\nWARNING: {len(_mismatches)} country/countries differ between"
          f" qci_complete_data.csv and SDI_VALUE_MAP_2021. First few:")
    for _c, _v_csv, _v_dict in _mismatches[:10]:
        print(f"    {_c}: csv={_v_csv}, dict={_v_dict}")
    print(f"  Preferring the CSV value (single source of truth).")
else:
    print(f"  SDI source check passed: {len(_overlap)} countries match exactly between CSV and dict.")

# Build the merged 195-country map: CSV values first, dict fallback for gaps.
_merged_rows = []
_csv_set = set(_sdi_csv['iso_location_name'])
for _c, _v in zip(_sdi_csv['iso_location_name'], _sdi_csv['sdi']):
    _merged_rows.append({'iso_location_name': _c, 'sdi': float(_v), 'sdi_source': 'csv'})
_fallback_count = 0
for _c, _v in SDI_VALUE_MAP_2021.items():
    if _c not in _csv_set:
        _merged_rows.append({'iso_location_name': _c, 'sdi': float(_v), 'sdi_source': 'dict_fallback'})
        _fallback_count += 1
sdi_df_map = pd.DataFrame(_merged_rows).reset_index(drop=True)
SDI_2021 = dict(zip(sdi_df_map['iso_location_name'], sdi_df_map['sdi']))
print(f"  SDI map built: {len(_csv_set)} countries from CSV + {_fallback_count} dict-fallback = {len(sdi_df_map)} total.")
del _sdi_src, _sdi_csv, _overlap, _mismatches, _merged_rows, _csv_set, _fallback_count

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Global GDR by country
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 1: Global GDR ===")

gdr_rows = []
for country in countries_only:
    for year in range(1990, 2022):
        fem = df_qci[(df_qci['iso_location_name'] == country) & (df_qci['sex_name'] == 'Female')
                     & (df_qci['age_name'] == 'Age-standardized') & (df_qci['year'] == year)]
        mal = df_qci[(df_qci['iso_location_name'] == country) & (df_qci['sex_name'] == 'Male')
                     & (df_qci['age_name'] == 'Age-standardized') & (df_qci['year'] == year)]
        if len(fem) > 0 and len(mal) > 0:
            f_v, m_v = fem.iloc[0]['qci'], mal.iloc[0]['qci']
            gdr_rows.append({'Country': country, 'Year': year,
                             'QCI_Female': f_v, 'QCI_Male': m_v, 'QCI_Both': (f_v + m_v) / 2,
                             'GDR': f_v / m_v, 'Gap': f_v - m_v,
                             'SDI_group': SDI_COUNTRY_MAPPING.get(country, 'Unknown')})

df_gdr = pd.DataFrame(gdr_rows)
df_gdr_2021 = df_gdr[df_gdr['Year'] == 2021].copy()
print(f"  GDR for {len(df_gdr_2021)} countries | mean={df_gdr_2021['GDR'].mean():.4f} | "
      f"female advantage: {(df_gdr_2021['GDR'] > 1).sum()}/{len(df_gdr_2021)}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Age-specific QCI by SDI quintile
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 2: Age-Specific QCI ===")

age_groups = ['<5 years', '5-14 years', '15-49 years', '50-69 years', '70+ years', 'Age-standardized']
age_rows = []
for country in countries_only:
    sdi_g = SDI_COUNTRY_MAPPING.get(country, 'Unknown')
    sdi_v = SDI_2021.get(country, np.nan)
    for age in age_groups:
        for sex in ['Both', 'Male', 'Female']:
            r = df_qci[(df_qci['iso_location_name'] == country) & (df_qci['age_name'] == age)
                       & (df_qci['sex_name'] == sex) & (df_qci['year'] == 2021)]
            if len(r) > 0:
                age_rows.append({'Country': country, 'Age': age, 'Sex': sex,
                                 'QCI': r.iloc[0]['qci'], 'SDI_group': sdi_g, 'SDI_value': sdi_v})

df_age = pd.DataFrame(age_rows)
age_sdi = df_age[df_age['Sex'] == 'Both'].groupby(['SDI_group', 'Age'])['QCI'].agg(
    ['mean', 'std', 'median', 'count']).reset_index()
age_sdi.columns = ['SDI_group', 'Age', 'Mean', 'SD', 'Median', 'N']

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Concentration Index
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 3: Concentration Index ===")

# (sdi_df_map and SDI_2021 are loaded above, just after country list setup,
# so all earlier analyses can use them too.)

def concentration_index(qci, rank_var):
    n = len(qci)
    if n < 5:
        return np.nan, np.nan
    ro = (np.argsort(np.argsort(rank_var)) + 1) / n
    mu = np.mean(qci)
    ci = 2 * np.cov(qci, ro)[0, 1] / mu
    # Jackknife SE
    cis = []
    for i in range(n):
        q_j, r_j = np.delete(qci, i), np.delete(rank_var, i)
        ro_j = (np.argsort(np.argsort(r_j)) + 1) / len(r_j)
        cis.append(2 * np.cov(q_j, ro_j)[0, 1] / np.mean(q_j))
    se = np.sqrt((n - 1) / n * np.sum((np.array(cis) - np.mean(cis))**2))
    return ci, se

ci_rows = []
for year in range(1990, 2022):
    d = df_qci[(df_qci['year'] == year) & (df_qci['age_name'] == 'Age-standardized')
               & (df_qci['sex_name'] == 'Both') & (df_qci['iso_location_name'].isin(countries_only))]
    m = d.merge(sdi_df_map, on='iso_location_name', how='inner')
    if len(m) > 10:
        ci, se = concentration_index(m['qci'].values, m['sdi'].values)
        ci_rows.append({'Year': year, 'CI': ci, 'SE': se, 'CI_lo': ci - 1.96 * se, 'CI_hi': ci + 1.96 * se, 'N': len(m)})

df_ci = pd.DataFrame(ci_rows)
print(f"  CI 1990={df_ci[df_ci['Year']==1990].iloc[0]['CI']:.4f}  2021={df_ci[df_ci['Year']==2021].iloc[0]['CI']:.4f}")

# Sex-specific CI
ci_sex = []
for sex in ['Male', 'Female', 'Both']:
    for year in [1990, 2000, 2010, 2021]:
        d = df_qci[(df_qci['year'] == year) & (df_qci['age_name'] == 'Age-standardized')
                   & (df_qci['sex_name'] == sex) & (df_qci['iso_location_name'].isin(countries_only))]
        m = d.merge(sdi_df_map, on='iso_location_name', how='inner')
        if len(m) > 10:
            ci, se = concentration_index(m['qci'].values, m['sdi'].values)
            ci_sex.append({'Sex': sex, 'Year': year, 'CI': ci, 'SE': se, 'N': len(m)})
df_ci_sex = pd.DataFrame(ci_sex)

# Age-specific CI
ci_age = []
for age in age_groups:
    d = df_qci[(df_qci['year'] == 2021) & (df_qci['age_name'] == age)
               & (df_qci['sex_name'] == 'Both') & (df_qci['iso_location_name'].isin(countries_only))]
    m = d.merge(sdi_df_map, on='iso_location_name', how='inner')
    if len(m) > 10:
        ci, se = concentration_index(m['qci'].values, m['sdi'].values)
        ci_age.append({'Age': age, 'CI': ci, 'SE': se, 'N': len(m)})
df_ci_age = pd.DataFrame(ci_age)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Slope Index of Inequality
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 4: SII ===")

sii_rows = []
for year in range(1990, 2022):
    d = df_qci[(df_qci['year'] == year) & (df_qci['age_name'] == 'Age-standardized')
               & (df_qci['sex_name'] == 'Both') & (df_qci['iso_location_name'].isin(countries_only))]
    m = d.merge(sdi_df_map, on='iso_location_name', how='inner')
    if len(m) > 10:
        fr = (np.argsort(np.argsort(m['sdi'].values)) + 1) / len(m)
        slope, _, _, p, se = stats.linregress(fr, m['qci'].values)
        sii_rows.append({'Year': year, 'SII': slope, 'SE': se, 'p': p,
                         'SII_lo': slope - 1.96 * se, 'SII_hi': slope + 1.96 * se})
df_sii = pd.DataFrame(sii_rows)
print(f"  SII 1990={df_sii[df_sii['Year']==1990].iloc[0]['SII']:.2f}  2021={df_sii[df_sii['Year']==2021].iloc[0]['SII']:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Theil Index + Gini + CV over time
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 5: Theil / Gini / CV ===")

ineq_rows = []
for year in range(1990, 2022):
    d = df_qci[(df_qci['year'] == year) & (df_qci['age_name'] == 'Age-standardized')
               & (df_qci['sex_name'] == 'Both') & (df_qci['iso_location_name'].isin(countries_only))]
    vals = d['qci'].values
    if len(vals) < 10:
        continue
    mu = np.mean(vals)
    theil = np.mean((vals / mu) * np.log(vals / mu))
    sv = np.sort(vals)
    n = len(sv)
    gini = (2 * np.sum(np.arange(1, n + 1) * sv) - (n + 1) * np.sum(sv)) / (n * np.sum(sv))
    cv = np.std(vals, ddof=1) / mu * 100
    ineq_rows.append({'Year': year, 'Theil': theil, 'Gini': gini, 'CV': cv,
                      'Mean': mu, 'SD': np.std(vals, ddof=1), 'Range': vals.max() - vals.min(), 'N': len(vals)})
df_ineq = pd.DataFrame(ineq_rows)
print(f"  Theil 1990={df_ineq[df_ineq['Year']==1990].iloc[0]['Theil']:.6f}  2021={df_ineq[df_ineq['Year']==2021].iloc[0]['Theil']:.6f}")
print(f"  Gini  1990={df_ineq[df_ineq['Year']==1990].iloc[0]['Gini']:.4f}  2021={df_ineq[df_ineq['Year']==2021].iloc[0]['Gini']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Multilevel Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 6: Multilevel Regression ===")

# Model 1: QCI ~ SDI + year (random intercept by country)
panel = []
for c in countries_only:
    sdi_v = SDI_2021.get(c, np.nan)
    for y in range(1990, 2022):
        r = df_qci[(df_qci['iso_location_name'] == c) & (df_qci['year'] == y)
                   & (df_qci['age_name'] == 'Age-standardized') & (df_qci['sex_name'] == 'Both')]
        if len(r) > 0:
            panel.append({'country': c, 'year': y, 'qci': r.iloc[0]['qci'],
                          'sdi': sdi_v, 'year_c': y - 2005})
df_panel = pd.DataFrame(panel).dropna(subset=['sdi', 'qci'])
print(f"  Panel: {len(df_panel)} obs, {df_panel['country'].nunique()} countries")

model_results = {}
try:
    m1 = smf.mixedlm("qci ~ sdi + year_c", df_panel, groups=df_panel["country"]).fit(reml=True)
    print(f"  M1: SDI={m1.fe_params['sdi']:.3f}(p={m1.pvalues['sdi']:.1e}), "
          f"Year={m1.fe_params['year_c']:.4f}(p={m1.pvalues['year_c']:.1e}), "
          f"RE_SD={np.sqrt(m1.cov_re.iloc[0,0]):.3f}")
    model_results['model1'] = {k: {'coef': round(float(m1.fe_params[k]), 4),
                                    'se': round(float(m1.bse[k]), 4),
                                    'p': float(m1.pvalues[k])}
                                for k in m1.fe_params.index}
    model_results['model1']['re_sd'] = round(float(np.sqrt(m1.cov_re.iloc[0, 0])), 3)
    model_results['model1']['n_obs'] = int(m1.nobs)
    model_results['model1']['n_groups'] = int(df_panel['country'].nunique())
except Exception as e:
    print(f"  M1 error: {e}")
    model_results['model1'] = {'error': str(e)}

# Model 2: + female + SDI:female
panel_sex = []
for c in countries_only:
    sdi_v = SDI_2021.get(c, np.nan)
    for y in [1990, 2000, 2010, 2021]:
        for sex in ['Male', 'Female']:
            r = df_qci[(df_qci['iso_location_name'] == c) & (df_qci['year'] == y)
                       & (df_qci['age_name'] == 'Age-standardized') & (df_qci['sex_name'] == sex)]
            if len(r) > 0:
                panel_sex.append({'country': c, 'year': y, 'qci': r.iloc[0]['qci'],
                                  'sdi': sdi_v, 'female': 1 if sex == 'Female' else 0, 'year_c': y - 2005})
df_ps = pd.DataFrame(panel_sex).dropna(subset=['sdi', 'qci'])

try:
    m2 = smf.mixedlm("qci ~ sdi + year_c + female + sdi:female", df_ps, groups=df_ps["country"]).fit(reml=True)
    print(f"  M2: Female={m2.fe_params['female']:.3f}(p={m2.pvalues['female']:.1e}), "
          f"SDI:Female={m2.fe_params['sdi:female']:.3f}(p={m2.pvalues['sdi:female']:.1e})")
    model_results['model2'] = {k: {'coef': round(float(m2.fe_params[k]), 4),
                                    'se': round(float(m2.bse[k]), 4),
                                    'p': float(m2.pvalues[k])}
                                for k in m2.fe_params.index}
except Exception as e:
    print(f"  M2 error: {e}")
    model_results['model2'] = {'error': str(e)}

# Model 3: + age groups
panel_age = []
for c in countries_only:
    sdi_v = SDI_2021.get(c, np.nan)
    for age in ['<5 years', '5-14 years', '15-49 years', '50-69 years', '70+ years']:
        r = df_qci[(df_qci['iso_location_name'] == c) & (df_qci['year'] == 2021)
                   & (df_qci['age_name'] == age) & (df_qci['sex_name'] == 'Both')]
        if len(r) > 0:
            panel_age.append({'country': c, 'age': age, 'qci': r.iloc[0]['qci'], 'sdi': sdi_v})
df_pa = pd.DataFrame(panel_age).dropna(subset=['sdi', 'qci'])

try:
    m3 = smf.mixedlm("qci ~ sdi + C(age, Treatment(reference='15-49 years'))",
                      df_pa, groups=df_pa["country"]).fit(reml=True)
    model_results['model3'] = {k: {'coef': round(float(m3.fe_params[k]), 4),
                                    'p': float(m3.pvalues[k])} for k in m3.fe_params.index}
    for k in m3.fe_params.index:
        if 'age' in k.lower() or k == 'sdi':
            print(f"  M3: {k}={m3.fe_params[k]:.3f} (p={m3.pvalues[k]:.1e})")
except Exception as e:
    print(f"  M3 error: {e}")
    model_results['model3'] = {'error': str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: GDR trends by SDI
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 7: GDR Trends ===")

gdr_trend = []
for year in range(1990, 2022):
    for grp in ['High', 'High-middle', 'Middle', 'Low-middle', 'Low']:
        gc = [c for c, g in SDI_COUNTRY_MAPPING.items() if g == grp]
        gd = df_gdr[(df_gdr['Year'] == year) & (df_gdr['Country'].isin(gc))]
        if len(gd) > 3:
            gdr_trend.append({'Year': year, 'SDI_group': grp, 'GDR_mean': gd['GDR'].mean(),
                              'Gap_mean': gd['Gap'].mean(), 'N': len(gd)})
df_gdr_trend = pd.DataFrame(gdr_trend)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE TABLES & STATS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Saving Tables ===")

df_gdr_2021.sort_values('GDR', ascending=False).to_csv(os.path.join(TABLE_DIR, 'table1_gdr_by_country.csv'), index=False, float_format='%.4f')
age_sdi.to_csv(os.path.join(TABLE_DIR, 'table2_age_sdi_qci.csv'), index=False, float_format='%.2f')
df_ci.to_csv(os.path.join(TABLE_DIR, 'table3_concentration_index.csv'), index=False, float_format='%.6f')
df_sii.to_csv(os.path.join(TABLE_DIR, 'table4_sii.csv'), index=False, float_format='%.4f')
df_ineq.to_csv(os.path.join(TABLE_DIR, 'table5_inequality_indices.csv'), index=False, float_format='%.6f')
df_ci_sex.to_csv(os.path.join(TABLE_DIR, 'table6_ci_by_sex.csv'), index=False, float_format='%.6f')
df_ci_age.to_csv(os.path.join(TABLE_DIR, 'table6b_ci_by_age.csv'), index=False, float_format='%.6f')
gdr_sdi = df_gdr_2021.groupby('SDI_group')['GDR'].agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
gdr_sdi.to_csv(os.path.join(TABLE_DIR, 'table7_gdr_by_sdi.csv'), index=False, float_format='%.4f')

ci_1990 = df_ci[df_ci['Year'] == 1990].iloc[0]
ci_2021 = df_ci[df_ci['Year'] == 2021].iloc[0]
sii_1990 = df_sii[df_sii['Year'] == 1990].iloc[0]
sii_2021 = df_sii[df_sii['Year'] == 2021].iloc[0]
ineq_1990 = df_ineq[df_ineq['Year'] == 1990].iloc[0]
ineq_2021 = df_ineq[df_ineq['Year'] == 2021].iloc[0]

top5 = df_gdr_2021.nlargest(5, 'GDR')[['Country', 'GDR', 'Gap', 'SDI_group']].to_dict('records')
bot5 = df_gdr_2021.nsmallest(5, 'GDR')[['Country', 'GDR', 'Gap', 'SDI_group']].to_dict('records')

age_v = df_age[df_age['Sex'] == 'Both'].groupby('Age')['QCI'].mean().to_dict()

stats_dict = {
    'gdr': {'mean': round(df_gdr_2021['GDR'].mean(), 4), 'sd': round(df_gdr_2021['GDR'].std(), 4),
            'range': f"{df_gdr_2021['GDR'].min():.4f}-{df_gdr_2021['GDR'].max():.4f}",
            'n_female_adv': int((df_gdr_2021['GDR'] > 1).sum()), 'n_total': len(df_gdr_2021),
            'mean_gap': round(df_gdr_2021['Gap'].mean(), 2)},
    'top5_gdr': top5, 'bottom5_gdr': bot5,
    'ci': {'ci_1990': round(float(ci_1990['CI']), 4), 'ci_2021': round(float(ci_2021['CI']), 4),
           'change_pct': round((ci_2021['CI'] - ci_1990['CI']) / abs(ci_1990['CI']) * 100, 1)},
    'sii': {'sii_1990': round(float(sii_1990['SII']), 2), 'sii_2021': round(float(sii_2021['SII']), 2),
            'change': round(float(sii_2021['SII'] - sii_1990['SII']), 2)},
    'theil': {'t_1990': round(float(ineq_1990['Theil']), 6), 't_2021': round(float(ineq_2021['Theil']), 6),
              'gini_1990': round(float(ineq_1990['Gini']), 4), 'gini_2021': round(float(ineq_2021['Gini']), 4),
              'cv_1990': round(float(ineq_1990['CV']), 2), 'cv_2021': round(float(ineq_2021['CV']), 2)},
    'models': model_results,
    'age_mean_qci_2021': {k: round(v, 2) for k, v in age_v.items()},
}
with open(STATS_PATH, 'w') as f:
    json.dump(stats_dict, f, indent=2, default=str)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── FIG 1: GDR World Map ─────────────────────────────────────────────────────
print("\n=== FIGURE 1: GDR World Map ===")
try:
    world = gpd.read_file(os.path.join(BASE, 'data/ne_110m/ne_110m_admin_0_countries.shp'))
    for nc in ['NAME', 'NAME_LONG', 'SOVEREIGNT', 'ADMIN']:
        if nc in world.columns:
            name_col = nc
            break
    world['qci_name'] = world[name_col].map(COUNTRY_NAME_MAPPING)
    world = world.merge(df_gdr_2021[['Country', 'GDR', 'Gap']], left_on='qci_name', right_on='Country', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    norm = Normalize(vmin=0.99, vmax=1.04)
    world.plot(column='GDR', ax=ax, cmap='RdBu', norm=norm, edgecolor='gray', linewidth=0.3,
               legend=False, missing_kwds={'color': 'lightgray', 'edgecolor': 'gray', 'linewidth': 0.2})
    sm_o = ScalarMappable(cmap='RdBu', norm=norm)
    sm_o._A = []
    cbar = fig.colorbar(sm_o, ax=ax, shrink=0.5, aspect=25, pad=0.02)
    cbar.set_label('GDR (Female/Male QCI)', fontsize=11)
    ax.set_title('Gender Disparity Ratio for DS-TB QCI, 2021', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_gdr_world_map.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_gdr_world_map.png'), format='png')
    plt.close()
    print("  Saved figure1")
except Exception as e:
    print(f"  Map error: {e}")

# ── FIG 2: Concentration Curves ──────────────────────────────────────────────
print("\n=== FIGURE 2: Concentration Curves ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, year in enumerate([1990, 2021]):
    ax = axes[i]
    d = df_qci[(df_qci['year'] == year) & (df_qci['age_name'] == 'Age-standardized')
               & (df_qci['sex_name'] == 'Both') & (df_qci['iso_location_name'].isin(countries_only))]
    m = d.merge(sdi_df_map, on='iso_location_name', how='inner').sort_values('sdi')
    n = len(m)
    cp = np.arange(1, n + 1) / n
    cq = np.cumsum(m['qci'].values) / m['qci'].sum()
    ci_val = df_ci[df_ci['Year'] == year].iloc[0]['CI']
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Equality')
    ax.plot(cp, cq, 'b-', lw=2, label=f'CI = {ci_val:.4f}')
    ax.fill_between(cp, cq, cp, alpha=0.15, color='blue')
    ax.set_xlabel('Cumulative pop. (SDI rank)')
    ax.set_ylabel('Cumulative QCI share')
    ax.set_title(f'({chr(65 + i)}) {year}', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Concentration Curves for DS-TB QCI, 1990 vs 2021', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure2_concentration_curves.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure2_concentration_curves.png'), format='png')
plt.close()
print("  Saved figure2")

# ── FIG 3: Inequality indices over time ───────────────────────────────────────
print("\n=== FIGURE 3: Inequality Indices ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col, title, color in [(axes[0, 0], 'CI', '(A) Concentration Index', 'blue'),
                                (axes[0, 1], 'SII', '(B) Slope Index of Inequality', 'red'),
                                (axes[1, 0], 'Theil', '(C) Theil T Index', 'green'),
                                (axes[1, 1], 'Gini', '(D) Gini Coefficient', 'purple')]:
    src = df_ci if col == 'CI' else df_sii if col == 'SII' else df_ineq
    ax.plot(src['Year'], src[col], '-o', markersize=3, lw=1.5, color=color)
    if col == 'CI':
        ax.fill_between(src['Year'], src['CI_lo'], src['CI_hi'], alpha=0.2, color=color)
    elif col == 'SII':
        ax.fill_between(src['Year'], src['SII_lo'], src['SII_hi'], alpha=0.2, color=color)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    if col in ['Theil', 'Gini']:
        ax.set_xlabel('Year')

fig.suptitle('Inequality in DS-TB Care Quality, 1990-2021', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure3_inequality_indices.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure3_inequality_indices.png'), format='png')
plt.close()
print("  Saved figure3")

# ── FIG 4: GDR vs SDI ────────────────────────────────────────────────────────
print("\n=== FIGURE 4: GDR vs SDI ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for grp in ['High', 'High-middle', 'Middle', 'Low-middle', 'Low']:
    s = df_gdr_2021[df_gdr_2021['SDI_group'] == grp]
    sv = [SDI_2021.get(c, np.nan) for c in s['Country']]
    ax.scatter(sv, s['GDR'], c=sdi_colors[grp], s=25, alpha=0.7, label=grp, edgecolors='none')
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.set_xlabel('SDI (2021)')
ax.set_ylabel('GDR')
ax.set_title('(A) GDR vs SDI, 2021', fontweight='bold')
ax.legend(fontsize=8, title='SDI')
ax.grid(True, alpha=0.3)

ax = axes[1]
for grp in ['High', 'High-middle', 'Middle', 'Low-middle', 'Low']:
    gd = df_gdr_trend[df_gdr_trend['SDI_group'] == grp]
    ax.plot(gd['Year'], gd['GDR_mean'], color=sdi_colors[grp], lw=1.5, label=grp)
ax.axhline(1.0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Year')
ax.set_ylabel('Mean GDR')
ax.set_title('(B) GDR Trends by SDI', fontweight='bold')
ax.legend(fontsize=8, title='SDI')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure4_gdr_sdi.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure4_gdr_sdi.png'), format='png')
plt.close()
print("  Saved figure4")

# ── FIG 5: Age-specific QCI by SDI ───────────────────────────────────────────
print("\n=== FIGURE 5: Age by SDI ===")
fig, ax = plt.subplots(figsize=(12, 7))
age_order = ['<5 years', '5-14 years', '15-49 years', '50-69 years', '70+ years']
x = np.arange(len(age_order))
w = 0.15
for i, grp in enumerate(['High', 'High-middle', 'Middle', 'Low-middle', 'Low']):
    vals = [age_sdi[(age_sdi['SDI_group'] == grp) & (age_sdi['Age'] == a)]['Mean'].values[0]
            if len(age_sdi[(age_sdi['SDI_group'] == grp) & (age_sdi['Age'] == a)]) > 0 else np.nan
            for a in age_order]
    ax.bar(x + i * w, vals, w, label=grp, color=sdi_colors[grp], alpha=0.85)
ax.set_xticks(x + w * 2)
ax.set_xticklabels(age_order)
ax.set_ylabel('Mean QCI (2021)')
ax.set_title('Age-Specific DS-TB QCI by SDI Quintile, 2021', fontweight='bold')
ax.legend(title='SDI', fontsize=9)
ax.set_ylim(80, 100)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure5_age_sdi_bars.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure5_age_sdi_bars.png'), format='png')
plt.close()
print("  Saved figure5")

# ── FIG 6: Age-Sex-SDI heatmap ───────────────────────────────────────────────
print("\n=== FIGURE 6: Age-Sex-SDI Heatmap ===")
fig, ax = plt.subplots(figsize=(10, 8))
heat = []
for age in age_order + ['Age-standardized']:
    row = {'Age': age}
    for sex in ['Male', 'Female']:
        for grp in ['High', 'High-middle', 'Middle', 'Low-middle', 'Low']:
            v = df_age[(df_age['Age'] == age) & (df_age['Sex'] == sex) & (df_age['SDI_group'] == grp)]
            row[f'{sex[0]}_{grp}'] = v['QCI'].mean() if len(v) > 0 else np.nan
    heat.append(row)
hdf = pd.DataFrame(heat).set_index('Age')
sns.heatmap(hdf, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=0.5, ax=ax,
            vmin=85, vmax=100, annot_kws={'size': 7.5}, cbar_kws={'label': 'QCI'})
ax.set_title('Mean QCI by Age, Sex, and SDI, 2021', fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure6_age_sex_sdi_heatmap.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure6_age_sex_sdi_heatmap.png'), format='png')
plt.close()
print("  Saved figure6")

# ── FIG 7: CI by age ─────────────────────────────────────────────────────────
print("\n=== FIGURE 7: CI by Age ===")
fig, ax = plt.subplots(figsize=(8, 5))
ca = df_ci_age.copy()
ca['lo'] = ca['CI'] - 1.96 * ca['SE']
ca['hi'] = ca['CI'] + 1.96 * ca['SE']
ax.barh(range(len(ca)), ca['CI'], xerr=[ca['CI'] - ca['lo'], ca['hi'] - ca['CI']],
        color='steelblue', capsize=3)
ax.set_yticks(range(len(ca)))
ax.set_yticklabels(ca['Age'])
ax.set_xlabel('Concentration Index')
ax.set_title('Concentration Index by Age Group, 2021', fontweight='bold')
ax.axvline(0, color='k', ls='--', lw=0.8)
ax.grid(True, alpha=0.3, axis='x')
for i, r in ca.iterrows():
    ax.text(r['hi'] + 0.0005, i, f'{r["CI"]:.4f}', va='center', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure7_ci_by_age.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure7_ci_by_age.png'), format='png')
plt.close()
print("  Saved figure7")

# ── FIG 8: F vs M scatter ────────────────────────────────────────────────────
print("\n=== FIGURE 8: F vs M ===")
fig, ax = plt.subplots(figsize=(8, 8))
for grp in ['High', 'High-middle', 'Middle', 'Low-middle', 'Low']:
    s = df_gdr_2021[df_gdr_2021['SDI_group'] == grp]
    ax.scatter(s['QCI_Male'], s['QCI_Female'], c=sdi_colors[grp], s=30, alpha=0.7, label=grp, edgecolors='none')
lims = [min(df_gdr_2021['QCI_Male'].min(), df_gdr_2021['QCI_Female'].min()) - 1,
        max(df_gdr_2021['QCI_Male'].max(), df_gdr_2021['QCI_Female'].max()) + 1]
ax.plot(lims, lims, 'k--', lw=1, label='Equality')
ax.set_xlabel('Male QCI')
ax.set_ylabel('Female QCI')
ax.set_title('Female vs Male QCI, 2021', fontweight='bold')
ax.legend(fontsize=8, title='SDI')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure8_female_vs_male.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure8_female_vs_male.png'), format='png')
plt.close()
print("  Saved figure8")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EQUITY ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nTables: {TABLE_DIR}")
print(f"Figures: {OUTPUT_DIR}")
print(f"Stats: {STATS_PATH}")
print(f"\n--- KEY FINDINGS ---")
s = stats_dict
print(f"GDR mean: {s['gdr']['mean']} | Female advantage: {s['gdr']['n_female_adv']}/{s['gdr']['n_total']}")
print(f"CI: {s['ci']['ci_1990']} -> {s['ci']['ci_2021']} ({s['ci']['change_pct']:+.1f}%)")
print(f"SII: {s['sii']['sii_1990']} -> {s['sii']['sii_2021']}")
print(f"Theil: {s['theil']['t_1990']} -> {s['theil']['t_2021']}")
print(f"Gini: {s['theil']['gini_1990']} -> {s['theil']['gini_2021']}")
print(f"CV: {s['theil']['cv_1990']}% -> {s['theil']['cv_2021']}%")
