#!/usr/bin/env python3
"""
Iran subnational analysis: Quality of Care for Drug-Susceptible Tuberculosis
A Provincial Analysis, 1990-2021

Complete analysis script generating all tables, figures, and statistics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import geopandas as gpd
import warnings
import os
import json
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCI_PATH = os.path.join(BASE, 'results/shared/qci.csv')
QCI_COMPLETE_PATH = os.path.join(BASE, 'results/shared/qci_complete_data.csv')
QCI_UNCERTAINTY_PATH = os.path.join(BASE, 'results/shared/qci_uncertainty.csv')
AAPC_PATH = os.path.join(BASE, 'results/shared/aapc_results.csv')
OUTPUT_DIR = os.path.join(BASE, 'results/iran/figures')
TABLE_DIR = os.path.join(BASE, 'results/iran/tables')
STATS_PATH = os.path.join(BASE, 'results/iran/stats.json')
SHAPEFILE_DIR = os.path.join(BASE, 'data/iran_shapefile')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(SHAPEFILE_DIR, exist_ok=True)

# ── Iran Provinces ─────────────────────────────────────────────────────────────
IRAN_PROVINCES = [
    'Alborz', 'Ardebil', 'Bushehr', 'Chahar Mahaal and Bakhtiari',
    'East Azarbayejan', 'Fars', 'Gilan', 'Golestan', 'Hamadan',
    'Hormozgan', 'Ilam', 'Isfahan', 'Kerman', 'Kermanshah',
    'Khorasan-e-Razavi', 'Khuzestan', 'Kohgiluyeh and Boyer-Ahmad',
    'Kurdistan', 'Lorestan', 'Markazi', 'Mazandaran', 'North Khorasan',
    'Qazvin', 'Qom', 'Semnan', 'Sistan and Baluchistan', 'South Khorasan',
    'Tehran', 'West Azarbayejan', 'Yazd', 'Zanjan',
]

# Province name mapping: qci data name -> shapefile name
# This will be finalized after downloading the shapefile
PROV_NAME_MAP = {}

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.8,
    'lines.markersize': 4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df_qci = pd.read_csv(QCI_PATH)
df_complete = pd.read_csv(QCI_COMPLETE_PATH)

# Filter Iran provinces + national
iran_mask = df_qci['iso_location_name'].isin(IRAN_PROVINCES + ['Iran'])
df_iran = df_qci[iran_mask].copy()

iran_mask_c = df_complete['iso_location_name'].isin(IRAN_PROVINCES + ['Iran'])
df_iran_c = df_complete[iran_mask_c].copy()

print(f"  Iran data: {len(df_iran)} rows, {df_iran['iso_location_name'].nunique()} locations")

# Also load MENA/Global data for comparison
comparison_locs = ['Iran', 'Global', 'Middle East & North Africa - WB',
                   'North Africa and Middle East', 'High-middle SDI']
comp_mask = df_qci['iso_location_name'].isin(comparison_locs)
df_comp = df_qci[comp_mask].copy()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: AAPC for all 31 provinces
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 1: Provincial AAPC ===")

def compute_aapc(qci_series, year_series):
    """Log-linear AAPC with 95% CI."""
    mask = ~(qci_series.isna() | year_series.isna())
    q = qci_series[mask].values
    y = year_series[mask].values.astype(float)
    if len(q) < 3:
        return np.nan, np.nan, np.nan
    ln_q = np.log(q)
    slope, intercept, r, p, se = stats.linregress(y, ln_q)
    aapc = (np.exp(slope) - 1) * 100
    t_crit = stats.t.ppf(0.975, df=len(y) - 2)
    ci_lo = (np.exp(slope - t_crit * se) - 1) * 100
    ci_hi = (np.exp(slope + t_crit * se) - 1) * 100
    return aapc, ci_lo, ci_hi

# Age-standardized, both sexes
prov_as = df_iran[(df_iran['age_name'] == 'Age-standardized') & (df_iran['sex_name'] == 'Both')]

aapc_rows = []
for prov in IRAN_PROVINCES + ['Iran']:
    pdata = prov_as[prov_as['iso_location_name'] == prov].sort_values('year')
    full = pdata[(pdata['year'] >= 1990) & (pdata['year'] <= 2021)]
    recent = pdata[(pdata['year'] >= 2010) & (pdata['year'] <= 2021)]
    aapc_f, ci_lo_f, ci_hi_f = compute_aapc(full['qci'], full['year'])
    aapc_r, ci_lo_r, ci_hi_r = compute_aapc(recent['qci'], recent['year'])
    qci_1990 = full[full['year'] == 1990]['qci'].values
    qci_2021 = full[full['year'] == 2021]['qci'].values
    qci_2000 = full[full['year'] == 2000]['qci'].values
    qci_2010 = full[full['year'] == 2010]['qci'].values
    aapc_rows.append({
        'Province': prov,
        'QCI_1990': qci_1990[0] if len(qci_1990) else np.nan,
        'QCI_2000': qci_2000[0] if len(qci_2000) else np.nan,
        'QCI_2010': qci_2010[0] if len(qci_2010) else np.nan,
        'QCI_2021': qci_2021[0] if len(qci_2021) else np.nan,
        'Change_1990_2021': (qci_2021[0] - qci_1990[0]) if (len(qci_2021) and len(qci_1990)) else np.nan,
        'AAPC_full': aapc_f, 'CI_lo_full': ci_lo_f, 'CI_hi_full': ci_hi_f,
        'AAPC_recent': aapc_r, 'CI_lo_recent': ci_lo_r, 'CI_hi_recent': ci_hi_r,
    })

df_aapc = pd.DataFrame(aapc_rows)
df_aapc = df_aapc.sort_values('QCI_2021', ascending=False).reset_index(drop=True)
print(f"  Computed AAPC for {len(df_aapc)} locations")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Component analysis per province
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 2: Component Analysis ===")

prov_comp_as = df_iran_c[(df_iran_c['age_name'] == 'Age-standardized') & (df_iran_c['sex_name'] == 'Both')]

comp_rows = []
for prov in IRAN_PROVINCES:
    p2021 = prov_comp_as[(prov_comp_as['iso_location_name'] == prov) & (prov_comp_as['year'] == 2021)]
    p1990 = prov_comp_as[(prov_comp_as['iso_location_name'] == prov) & (prov_comp_as['year'] == 1990)]
    if len(p2021) == 0 or len(p1990) == 0:
        continue
    r2021 = p2021.iloc[0]
    r1990 = p1990.iloc[0]
    comp_rows.append({
        'Province': prov,
        'MIR_1990': r1990['MIR'], 'MIR_2021': r2021['MIR'],
        'MIR_change_pct': ((r2021['MIR'] - r1990['MIR']) / r1990['MIR'] * 100) if r1990['MIR'] != 0 else np.nan,
        'YLLtoYLD_1990': r1990['YLLtoYLD'], 'YLLtoYLD_2021': r2021['YLLtoYLD'],
        'YLLtoYLD_change_pct': ((r2021['YLLtoYLD'] - r1990['YLLtoYLD']) / r1990['YLLtoYLD'] * 100) if r1990['YLLtoYLD'] != 0 else np.nan,
        'DALtoPER_1990': r1990['DALtoPER'], 'DALtoPER_2021': r2021['DALtoPER'],
        'DALtoPER_change_pct': ((r2021['DALtoPER'] - r1990['DALtoPER']) / r1990['DALtoPER'] * 100) if r1990['DALtoPER'] != 0 else np.nan,
        'QCI_2021': r2021['pca_score'],
    })

df_comp_prov = pd.DataFrame(comp_rows)
df_comp_prov = df_comp_prov.sort_values('QCI_2021', ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Provincial inequality metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 3: Provincial Inequality Metrics ===")

# Compute CV and range across provinces for each year
inequality_rows = []
for year in range(1990, 2022):
    yr_data = prov_as[(prov_as['iso_location_name'].isin(IRAN_PROVINCES)) & (prov_as['year'] == year)]
    if len(yr_data) < 10:
        continue
    qci_vals = yr_data['qci'].values
    mean_val = np.mean(qci_vals)
    sd_val = np.std(qci_vals, ddof=1)
    cv = (sd_val / mean_val) * 100
    iqr = np.percentile(qci_vals, 75) - np.percentile(qci_vals, 25)
    rng = np.max(qci_vals) - np.min(qci_vals)
    # Gini coefficient
    sorted_vals = np.sort(qci_vals)
    n = len(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_vals) - (n+1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))

    inequality_rows.append({
        'Year': year, 'Mean': mean_val, 'SD': sd_val, 'CV': cv,
        'Min': np.min(qci_vals), 'Max': np.max(qci_vals),
        'Range': rng, 'IQR': iqr, 'Gini': gini,
        'P10': np.percentile(qci_vals, 10), 'P90': np.percentile(qci_vals, 90),
    })

df_ineq = pd.DataFrame(inequality_rows)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Age-sex analysis by province
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 4: Age-Sex Provincial Analysis ===")

age_sex_rows = []
for prov in IRAN_PROVINCES:
    for sex in ['Both', 'Male', 'Female']:
        for age in ['Age-standardized', '<5 years', '5-14 years', '15-49 years', '50-69 years', '70+ years']:
            row_data = df_iran[(df_iran['iso_location_name'] == prov) &
                               (df_iran['sex_name'] == sex) &
                               (df_iran['age_name'] == age) &
                               (df_iran['year'] == 2021)]
            if len(row_data) == 0:
                continue
            age_sex_rows.append({
                'Province': prov, 'Sex': sex, 'Age': age,
                'QCI_2021': row_data.iloc[0]['qci'],
            })

df_age_sex = pd.DataFrame(age_sex_rows)

# GDR by province
gdr_rows = []
for prov in IRAN_PROVINCES:
    fem = df_age_sex[(df_age_sex['Province'] == prov) & (df_age_sex['Sex'] == 'Female') & (df_age_sex['Age'] == 'Age-standardized')]
    mal = df_age_sex[(df_age_sex['Province'] == prov) & (df_age_sex['Sex'] == 'Male') & (df_age_sex['Age'] == 'Age-standardized')]
    if len(fem) > 0 and len(mal) > 0:
        gdr = fem.iloc[0]['QCI_2021'] / mal.iloc[0]['QCI_2021']
        gap = fem.iloc[0]['QCI_2021'] - mal.iloc[0]['QCI_2021']
        gdr_rows.append({'Province': prov, 'QCI_Female': fem.iloc[0]['QCI_2021'],
                         'QCI_Male': mal.iloc[0]['QCI_2021'], 'GDR': gdr, 'Gap': gap})

df_gdr = pd.DataFrame(gdr_rows).sort_values('GDR', ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Comparison with national and regional benchmarks
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 5: National/Regional Benchmarks ===")

bench_as = df_qci[(df_qci['age_name'] == 'Age-standardized') & (df_qci['sex_name'] == 'Both')]

comparison_data = {}
for loc in ['Iran', 'Global', 'Middle East & North Africa - WB', 'High-middle SDI']:
    loc_data = bench_as[bench_as['iso_location_name'] == loc]
    if len(loc_data) > 0:
        q1990 = loc_data[loc_data['year'] == 1990]['qci'].values
        q2021 = loc_data[loc_data['year'] == 2021]['qci'].values
        comparison_data[loc] = {
            'QCI_1990': q1990[0] if len(q1990) else np.nan,
            'QCI_2021': q2021[0] if len(q2021) else np.nan,
        }

# Also get similar-SDI MENA countries
mena_countries_in_data = ['Turkey', 'Saudi Arabia', 'Iraq', 'Algeria', 'Egypt', 'Morocco',
                          'Tunisia', 'Jordan', 'Lebanon', 'Libya', 'Syria', 'Yemen',
                          'Oman', 'Kuwait', 'Bahrain', 'Qatar', 'United Arab Emirates', 'Palestine']
mena_comp_rows = []
for c in mena_countries_in_data:
    cdata = bench_as[bench_as['iso_location_name'] == c]
    if len(cdata) > 0:
        q2021 = cdata[cdata['year'] == 2021]['qci'].values
        q1990 = cdata[cdata['year'] == 1990]['qci'].values
        if len(q2021) and len(q1990):
            mena_comp_rows.append({'Country': c, 'QCI_2021': q2021[0], 'QCI_1990': q1990[0],
                                   'Change': q2021[0] - q1990[0]})
df_mena_comp = pd.DataFrame(mena_comp_rows).sort_values('QCI_2021', ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Sistan-Baluchistan deep dive
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ANALYSIS 6: Sistan-Baluchistan Deep Dive ===")

sb = 'Sistan and Baluchistan'
sb_ts = prov_as[prov_as['iso_location_name'] == sb].sort_values('year')
sb_comp = prov_comp_as[prov_comp_as['iso_location_name'] == sb].sort_values('year')

# Best province for comparison
best_prov = 'Chahar Mahaal and Bakhtiari'
bp_ts = prov_as[prov_as['iso_location_name'] == best_prov].sort_values('year')
bp_comp = prov_comp_as[prov_comp_as['iso_location_name'] == best_prov].sort_values('year')

# ══════════════════════════════════════════════════════════════════════════════
# SAVE TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Saving Tables ===")

# Table 1: Provincial QCI ranking with AAPC
table1 = df_aapc[df_aapc['Province'] != 'Iran'].copy()
table1['Rank'] = range(1, len(table1) + 1)
# Add component ratios for 2021
for idx, row in table1.iterrows():
    comp_row = df_comp_prov[df_comp_prov['Province'] == row['Province']]
    if len(comp_row) > 0:
        table1.loc[idx, 'MIR_2021'] = comp_row.iloc[0]['MIR_2021']
        table1.loc[idx, 'YLLtoYLD_2021'] = comp_row.iloc[0]['YLLtoYLD_2021']
        table1.loc[idx, 'DALtoPER_2021'] = comp_row.iloc[0]['DALtoPER_2021']

cols_t1 = ['Rank', 'Province', 'QCI_1990', 'QCI_2000', 'QCI_2010', 'QCI_2021',
           'Change_1990_2021', 'AAPC_full', 'CI_lo_full', 'CI_hi_full',
           'AAPC_recent', 'CI_lo_recent', 'CI_hi_recent',
           'MIR_2021', 'YLLtoYLD_2021', 'DALtoPER_2021']
table1[cols_t1].to_csv(os.path.join(TABLE_DIR, 'table1_provincial_qci_ranking.csv'), index=False, float_format='%.4f')
print(f"  Table 1 saved: {len(table1)} provinces")

# Table 2: Component analysis
df_comp_prov.to_csv(os.path.join(TABLE_DIR, 'table2_component_analysis.csv'), index=False, float_format='%.4f')

# Table 3: Inequality metrics over time
df_ineq.to_csv(os.path.join(TABLE_DIR, 'table3_inequality_metrics.csv'), index=False, float_format='%.4f')

# Table 4: Gender disparity by province
df_gdr.to_csv(os.path.join(TABLE_DIR, 'table4_gender_disparity.csv'), index=False, float_format='%.4f')

# Table 5: MENA comparison
df_mena_comp.to_csv(os.path.join(TABLE_DIR, 'table5_mena_comparison.csv'), index=False, float_format='%.4f')

# ══════════════════════════════════════════════════════════════════════════════
# COLLECT STATISTICS FOR PAPER
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Collecting Statistics ===")

iran_row = df_aapc[df_aapc['Province'] == 'Iran'].iloc[0]
prov_only = df_aapc[df_aapc['Province'] != 'Iran']

stats_dict = {
    'iran_national': {
        'qci_1990': round(iran_row['QCI_1990'], 2),
        'qci_2021': round(iran_row['QCI_2021'], 2),
        'change': round(iran_row['Change_1990_2021'], 2),
        'aapc_full': round(iran_row['AAPC_full'], 3),
        'aapc_full_ci': f"{iran_row['CI_lo_full']:.3f} to {iran_row['CI_hi_full']:.3f}",
        'aapc_recent': round(iran_row['AAPC_recent'], 3),
    },
    'best_province': {
        'name': prov_only.iloc[0]['Province'],
        'qci_2021': round(prov_only.iloc[0]['QCI_2021'], 2),
        'qci_1990': round(prov_only.iloc[0]['QCI_1990'], 2),
    },
    'worst_province': {
        'name': prov_only.iloc[-1]['Province'],
        'qci_2021': round(prov_only.iloc[-1]['QCI_2021'], 2),
        'qci_1990': round(prov_only.iloc[-1]['QCI_1990'], 2),
    },
    'provincial_range_2021': round(prov_only['QCI_2021'].max() - prov_only['QCI_2021'].min(), 2),
    'provincial_range_1990': round(prov_only['QCI_1990'].max() - prov_only['QCI_1990'].min(), 2),
    'mean_qci_2021': round(prov_only['QCI_2021'].mean(), 2),
    'sd_qci_2021': round(prov_only['QCI_2021'].std(), 2),
    'median_qci_2021': round(prov_only['QCI_2021'].median(), 2),
    'all_above_95': bool((prov_only['QCI_2021'] > 95).all()),
    'n_above_97': int((prov_only['QCI_2021'] > 97).sum()),
    'n_above_98': int((prov_only['QCI_2021'] > 98).sum()),
    'inequality': {
        'cv_1990': round(df_ineq[df_ineq['Year'] == 1990].iloc[0]['CV'], 3) if len(df_ineq[df_ineq['Year'] == 1990]) else None,
        'cv_2021': round(df_ineq[df_ineq['Year'] == 2021].iloc[0]['CV'], 3) if len(df_ineq[df_ineq['Year'] == 2021]) else None,
        'gini_1990': round(df_ineq[df_ineq['Year'] == 1990].iloc[0]['Gini'], 4) if len(df_ineq[df_ineq['Year'] == 1990]) else None,
        'gini_2021': round(df_ineq[df_ineq['Year'] == 2021].iloc[0]['Gini'], 4) if len(df_ineq[df_ineq['Year'] == 2021]) else None,
        'range_1990': round(df_ineq[df_ineq['Year'] == 1990].iloc[0]['Range'], 2) if len(df_ineq[df_ineq['Year'] == 1990]) else None,
        'range_2021': round(df_ineq[df_ineq['Year'] == 2021].iloc[0]['Range'], 2) if len(df_ineq[df_ineq['Year'] == 2021]) else None,
    },
    'fastest_improving': prov_only.sort_values('AAPC_full', ascending=False).iloc[0]['Province'],
    'fastest_improving_aapc': round(prov_only.sort_values('AAPC_full', ascending=False).iloc[0]['AAPC_full'], 3),
    'largest_absolute_gain': prov_only.sort_values('Change_1990_2021', ascending=False).iloc[0]['Province'],
    'largest_absolute_gain_val': round(prov_only.sort_values('Change_1990_2021', ascending=False).iloc[0]['Change_1990_2021'], 2),
    'gdr_mean': round(df_gdr['GDR'].mean(), 4),
    'gdr_range': f"{df_gdr['GDR'].min():.4f} - {df_gdr['GDR'].max():.4f}",
    'mean_gap_fm': round(df_gdr['Gap'].mean(), 2),
}

# Sistan deep dive stats
sb_2021 = sb_comp[sb_comp['year'] == 2021].iloc[0]
sb_1990 = sb_comp[sb_comp['year'] == 1990].iloc[0]
stats_dict['sistan'] = {
    'mir_2021': round(sb_2021['MIR'], 4),
    'mir_1990': round(sb_1990['MIR'], 4),
    'mir_change_pct': round((sb_2021['MIR'] - sb_1990['MIR']) / sb_1990['MIR'] * 100, 1),
    'yll_yld_2021': round(sb_2021['YLLtoYLD'], 2),
    'dal_per_2021': round(sb_2021['DALtoPER'], 2),
    'qci_2021': round(sb_ts[sb_ts['year'] == 2021].iloc[0]['qci'], 2),
    'qci_1990': round(sb_ts[sb_ts['year'] == 1990].iloc[0]['qci'], 2),
    'improvement': round(sb_ts[sb_ts['year'] == 2021].iloc[0]['qci'] - sb_ts[sb_ts['year'] == 1990].iloc[0]['qci'], 2),
}

# Best province stats
bp_2021 = bp_comp[bp_comp['year'] == 2021].iloc[0]
stats_dict['chahar_mahaal'] = {
    'mir_2021': round(bp_2021['MIR'], 4),
    'yll_yld_2021': round(bp_2021['YLLtoYLD'], 2),
    'dal_per_2021': round(bp_2021['DALtoPER'], 2),
}

with open(STATS_PATH, 'w') as f:
    json.dump(stats_dict, f, indent=2, default=str)
print(f"  Statistics saved to {STATS_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── FIGURE 1: Provincial choropleth maps (1990 vs 2021) ──────────────────────
print("\n=== FIGURE 1: Choropleth Maps ===")

# Try to download Iran shapefile
shapefile_path = os.path.join(SHAPEFILE_DIR, 'iran_provinces.shp')
geojson_path = os.path.join(SHAPEFILE_DIR, 'iran_provinces.geojson')

# Try multiple approaches for Iran province boundaries
iran_gdf = None

# First check if shapefile exists already
if os.path.exists(shapefile_path):
    iran_gdf = gpd.read_file(shapefile_path)
elif os.path.exists(geojson_path):
    iran_gdf = gpd.read_file(geojson_path)
else:
    # Try downloading from natural earth admin-1 (contains subnational divisions)
    try:
        import urllib.request
        # Natural Earth admin-1 states/provinces
        ne_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
        zip_path = os.path.join(SHAPEFILE_DIR, 'ne_admin1.zip')
        print("  Downloading Natural Earth admin-1 data...")
        urllib.request.urlretrieve(ne_url, zip_path)
        # Read and filter for Iran
        gdf_all = gpd.read_file(f"zip://{zip_path}")
        iran_gdf = gdf_all[gdf_all['admin'] == 'Iran'].copy()
        if len(iran_gdf) == 0:
            iran_gdf = gdf_all[gdf_all['iso_a2'] == 'IR'].copy()
        print(f"  Found {len(iran_gdf)} Iran provinces in Natural Earth data")
        if len(iran_gdf) > 0:
            iran_gdf.to_file(geojson_path, driver='GeoJSON')
    except Exception as e:
        print(f"  Could not download shapefile: {e}")
        print("  Will create non-map visualizations instead")

# Name mapping from Natural Earth to our data
NE_TO_QCI = {
    'Alborz': 'Alborz',
    'Ardabīl': 'Ardebil', 'Ardabil': 'Ardebil', 'Ardebil': 'Ardebil',
    'Būshehr': 'Bushehr', 'Bushehr': 'Bushehr', 'Bushehr (Bushire)': 'Bushehr',
    'Chahār Maḩāll va Bakhtīārī': 'Chahar Mahaal and Bakhtiari',
    'Chahar Mahall va Bakhtiari': 'Chahar Mahaal and Bakhtiari',
    'Chahar Mahaal and Bakhtiari': 'Chahar Mahaal and Bakhtiari',
    'Chahar Mahall and Bakhtiari': 'Chahar Mahaal and Bakhtiari',
    'Āz̄arbāyjān-e Sharqī': 'East Azarbayejan',
    'East Azarbaijan': 'East Azarbayejan', 'East Azerbaijan': 'East Azarbayejan',
    'East Azarbayejan': 'East Azarbayejan', 'Azarbayjan-e Sharqi': 'East Azarbayejan',
    'East Azarbaijan': 'East Azarbayejan',
    'Fārs': 'Fars', 'Fars': 'Fars',
    'Gīlān': 'Gilan', 'Gilan': 'Gilan',
    'Golestān': 'Golestan', 'Golestan': 'Golestan',
    'Hamadān': 'Hamadan', 'Hamadan': 'Hamadan', 'Hamedan': 'Hamadan',
    'Hormozgān': 'Hormozgan', 'Hormozgan': 'Hormozgan',
    'Īlām': 'Ilam', 'Ilam': 'Ilam',
    'Eşfahān': 'Isfahan', 'Isfahan': 'Isfahan', 'Esfahan': 'Isfahan',
    'Esfahan': 'Isfahan',
    'Kermān': 'Kerman', 'Kerman': 'Kerman',
    'Kermānshāh': 'Kermanshah', 'Kermanshah': 'Kermanshah',
    'Khorāsān-e Raẕavī': 'Khorasan-e-Razavi', 'Razavi Khorasan': 'Khorasan-e-Razavi',
    'Khorasan-e Razavi': 'Khorasan-e-Razavi', 'Khorasan-e-Razavi': 'Khorasan-e-Razavi',
    'Razavi Khorasan (Khorasan-e Razavi)': 'Khorasan-e-Razavi',
    'Khūzestān': 'Khuzestan', 'Khuzestan': 'Khuzestan',
    'Kohgīlūyeh va Būyer Aḩmad': 'Kohgiluyeh and Boyer-Ahmad',
    'Kohgiluyeh va Buyer Ahmad': 'Kohgiluyeh and Boyer-Ahmad',
    'Kohgiluyeh and Boyer-Ahmad': 'Kohgiluyeh and Boyer-Ahmad',
    'Kohgiluyeh va Boyerahmad': 'Kohgiluyeh and Boyer-Ahmad',
    'Kohgiluyeh and Buyer Ahmad': 'Kohgiluyeh and Boyer-Ahmad',
    'Kordestān': 'Kurdistan', 'Kurdistan': 'Kurdistan', 'Kordestan': 'Kurdistan',
    'Kordestan': 'Kurdistan',
    'Lorestān': 'Lorestan', 'Lorestan': 'Lorestan',
    'Markazī': 'Markazi', 'Markazi': 'Markazi',
    'Māzandarān': 'Mazandaran', 'Mazandaran': 'Mazandaran',
    'Khorāsān-e Shomālī': 'North Khorasan', 'North Khorasan': 'North Khorasan',
    'Qazvīn': 'Qazvin', 'Qazvin': 'Qazvin',
    'Qom': 'Qom',
    'Semnān': 'Semnan', 'Semnan': 'Semnan',
    'Sīstān va Balūchestān': 'Sistan and Baluchistan',
    'Sistan va Baluchestan': 'Sistan and Baluchistan',
    'Sistan and Baluchistan': 'Sistan and Baluchistan',
    'Sistan and Baluchestan': 'Sistan and Baluchistan',
    'Sistan and Baluchestan': 'Sistan and Baluchistan',
    'Khorāsān-e Jonūbī': 'South Khorasan', 'South Khorasan': 'South Khorasan',
    'Tehrān': 'Tehran', 'Tehran': 'Tehran',
    'Āz̄arbāyjān-e Gharbī': 'West Azarbayejan',
    'West Azarbaijan': 'West Azarbayejan', 'West Azerbaijan': 'West Azarbayejan',
    'West Azarbayejan': 'West Azarbayejan', 'Azarbayjan-e Gharbi': 'West Azarbayejan',
    'West Azarbaijan': 'West Azarbayejan',
    'Yazd': 'Yazd',
    'Zanjān': 'Zanjan', 'Zanjan': 'Zanjan',
}

def make_choropleth(iran_gdf, qci_col_data, title, fname, vmin=None, vmax=None, cmap_name='RdYlGn'):
    """Create a choropleth map of Iran provinces."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Try to match province names
    name_col = None
    for col in ['name', 'NAME', 'Name', 'name_en', 'NAME_1', 'name_1', 'NAME_EN', 'gn_name', 'woe_name']:
        if col in iran_gdf.columns:
            name_col = col
            break

    if name_col is None:
        print(f"  Warning: Could not find name column in shapefile. Columns: {iran_gdf.columns.tolist()}")
        plt.close()
        return False

    # Map names
    iran_gdf = iran_gdf.copy()
    iran_gdf['qci_province'] = iran_gdf[name_col].map(NE_TO_QCI)

    # Check matching
    matched = iran_gdf['qci_province'].notna().sum()
    unmatched = iran_gdf[iran_gdf['qci_province'].isna()][name_col].tolist()
    if unmatched:
        print(f"  Unmatched provinces: {unmatched}")

    # Merge QCI data
    iran_gdf = iran_gdf.merge(qci_col_data, left_on='qci_province', right_on='Province', how='left')

    if vmin is None:
        vmin = iran_gdf['QCI'].min() - 0.5
    if vmax is None:
        vmax = iran_gdf['QCI'].max() + 0.5

    # Plot
    iran_gdf.plot(column='QCI', ax=ax, cmap=cmap_name, edgecolor='black',
                  linewidth=0.5, legend=False, vmin=vmin, vmax=vmax,
                  missing_kwds={'color': 'lightgray', 'edgecolor': 'black', 'linewidth': 0.5})

    # Add province labels
    for idx, row in iran_gdf.iterrows():
        if pd.notna(row.get('QCI')):
            centroid = row.geometry.centroid
            label = f"{row['QCI']:.1f}"
            ax.annotate(label, (centroid.x, centroid.y), fontsize=7, ha='center', va='center',
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, linewidth=0))

    # Colorbar
    sm = ScalarMappable(cmap=cmap_name, norm=Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('QCI Score', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_axis_off()

    fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), format='png')
    plt.close()
    print(f"  Saved {fname}")
    return True

if iran_gdf is not None and len(iran_gdf) > 0:
    # Prepare 2021 data
    qci_2021_data = prov_only[['Province', 'QCI_2021']].rename(columns={'QCI_2021': 'QCI'})
    qci_1990_data = prov_only[['Province', 'QCI_1990']].rename(columns={'QCI_1990': 'QCI'})
    qci_change_data = prov_only[['Province', 'Change_1990_2021']].rename(columns={'Change_1990_2021': 'QCI'})

    make_choropleth(iran_gdf, qci_2021_data,
                    'DS-TB Quality of Care Index by Province, Iran, 2021',
                    'figure1a_choropleth_2021', vmin=95, vmax=99)
    make_choropleth(iran_gdf, qci_1990_data,
                    'DS-TB Quality of Care Index by Province, Iran, 1990',
                    'figure1b_choropleth_1990', vmin=89, vmax=98)
    make_choropleth(iran_gdf, qci_change_data,
                    'Change in QCI by Province, Iran, 1990-2021',
                    'figure1c_choropleth_change', cmap_name='YlOrRd')
else:
    print("  Skipping choropleth maps (no shapefile available)")
    # Create a horizontal bar chart as alternative
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    for ax_idx, (year, col) in enumerate([(2021, 'QCI_2021'), (1990, 'QCI_1990')]):
        ax = axes[ax_idx]
        sorted_data = prov_only.sort_values(col, ascending=True)
        colors = plt.cm.RdYlGn((sorted_data[col] - sorted_data[col].min()) /
                                (sorted_data[col].max() - sorted_data[col].min()))
        ax.barh(range(len(sorted_data)), sorted_data[col], color=colors)
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['Province'], fontsize=8)
        ax.set_xlabel('QCI Score')
        ax.set_title(f'Provincial QCI, {year}', fontweight='bold')
        for i, (_, row) in enumerate(sorted_data.iterrows()):
            ax.text(row[col] + 0.05, i, f'{row[col]:.1f}', va='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_provincial_ranking_bars.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure1_provincial_ranking_bars.png'), format='png')
    plt.close()
    print("  Saved figure1_provincial_ranking_bars (alternative to choropleth)")


# ── FIGURE 2: Time series for all 31 provinces ───────────────────────────────
print("\n=== FIGURE 2: Provincial Time Series ===")

fig, ax = plt.subplots(figsize=(14, 7))

# Sort provinces by 2021 QCI for consistent coloring
sorted_provs = prov_only.sort_values('QCI_2021', ascending=False)['Province'].tolist()

# Color scheme: top 5 green, bottom 5 red, middle gray
top5 = sorted_provs[:5]
bottom5 = sorted_provs[-5:]
middle = sorted_provs[5:-5]

# Also plot Iran national
iran_ts = prov_as[prov_as['iso_location_name'] == 'Iran'].sort_values('year')

for prov in middle:
    pdata = prov_as[prov_as['iso_location_name'] == prov].sort_values('year')
    ax.plot(pdata['year'], pdata['qci'], color='#D0D0D0', linewidth=0.6, alpha=0.5)

green_shades = ['#1a9850', '#33a02c', '#66bd63', '#91cf60', '#b2df8a']
for i, prov in enumerate(top5):
    pdata = prov_as[prov_as['iso_location_name'] == prov].sort_values('year')
    ax.plot(pdata['year'], pdata['qci'], color=green_shades[i], linewidth=1.8, label=prov)

red_shades = ['#d73027', '#e31a1c', '#fc4e2a', '#fd8d3c', '#feb24c']
for i, prov in enumerate(bottom5):
    pdata = prov_as[prov_as['iso_location_name'] == prov].sort_values('year')
    ax.plot(pdata['year'], pdata['qci'], color=red_shades[i], linewidth=1.8, label=prov)

ax.plot(iran_ts['year'], iran_ts['qci'], color='black', linewidth=2.5, linestyle='--', label='Iran (national)')

ax.set_xlabel('Year')
ax.set_ylabel('Quality of Care Index (QCI)')
ax.set_title('DS-TB Quality of Care Index Trajectories, Iranian Provinces, 1990-2021', fontweight='bold')
ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.95, edgecolor='#cccccc')
ax.set_xlim(1990, 2021)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

fig.savefig(os.path.join(OUTPUT_DIR, 'figure2_provincial_timeseries.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure2_provincial_timeseries.png'), format='png')
plt.close()
print("  Saved figure2_provincial_timeseries")


# ── FIGURE 3: Component ratio analysis (heatmap) ─────────────────────────────
print("\n=== FIGURE 3: Component Heatmap ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 10))
comp_labels = ['MIR_2021', 'YLLtoYLD_2021', 'DALtoPER_2021']
comp_titles = ['Mortality-to-Incidence Ratio (MIR)', 'YLL-to-YLD Ratio', 'DALY-to-Prevalence Ratio']
cmaps = ['YlOrRd', 'YlOrRd', 'YlOrRd']

sorted_comp = df_comp_prov.sort_values('QCI_2021', ascending=True)

for i, (comp, title, cmap) in enumerate(zip(comp_labels, comp_titles, cmaps)):
    ax = axes[i]
    vals = sorted_comp[comp].values
    colors = plt.cm.YlOrRd(Normalize(vmin=vals.min(), vmax=vals.max())(vals))
    ax.barh(range(len(sorted_comp)), vals, color=colors)
    ax.set_yticks(range(len(sorted_comp)))
    if i == 0:
        ax.set_yticklabels(sorted_comp['Province'], fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Ratio Value')
    for j, v in enumerate(vals):
        ax.text(v + (vals.max() - vals.min()) * 0.02, j, f'{v:.3f}' if comp == 'MIR_2021' else f'{v:.2f}',
                va='center', fontsize=7.5)

fig.suptitle('Component Ratios by Province, Iran, 2021', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure3_component_heatmap.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure3_component_heatmap.png'), format='png')
plt.close()
print("  Saved figure3_component_heatmap")


# ── FIGURE 4: Inequality metrics over time ────────────────────────────────────
print("\n=== FIGURE 4: Inequality Over Time ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: CV
ax = axes[0, 0]
ax.plot(df_ineq['Year'], df_ineq['CV'], 'b-o', markersize=3, linewidth=1.5)
ax.set_ylabel('Coefficient of Variation (%)')
ax.set_title('(A) Coefficient of Variation', fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel B: Gini
ax = axes[0, 1]
ax.plot(df_ineq['Year'], df_ineq['Gini'], 'r-o', markersize=3, linewidth=1.5)
ax.set_ylabel('Gini Coefficient')
ax.set_title('(B) Gini Coefficient', fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel C: Range and IQR
ax = axes[1, 0]
ax.plot(df_ineq['Year'], df_ineq['Range'], 'g-o', markersize=3, linewidth=1.5, label='Range (Max-Min)')
ax.plot(df_ineq['Year'], df_ineq['IQR'], 'm-s', markersize=3, linewidth=1.5, label='IQR')
ax.set_ylabel('QCI Points')
ax.set_xlabel('Year')
ax.set_title('(C) Range and Interquartile Range', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel D: P10 and P90
ax = axes[1, 1]
ax.fill_between(df_ineq['Year'], df_ineq['P10'], df_ineq['P90'], alpha=0.2, color='blue', label='P10-P90 range')
ax.plot(df_ineq['Year'], df_ineq['Mean'], 'k-', linewidth=2, label='Mean')
ax.plot(df_ineq['Year'], df_ineq['Min'], 'r--', linewidth=1, label='Min')
ax.plot(df_ineq['Year'], df_ineq['Max'], 'g--', linewidth=1, label='Max')
ax.set_ylabel('QCI Score')
ax.set_xlabel('Year')
ax.set_title('(D) QCI Distribution Over Time', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('Provincial Inequality in DS-TB Care Quality, Iran, 1990-2021',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure4_inequality_metrics.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure4_inequality_metrics.png'), format='png')
plt.close()
print("  Saved figure4_inequality_metrics")


# ── FIGURE 5: Sistan-Baluchistan deep dive ────────────────────────────────────
print("\n=== FIGURE 5: Sistan-Baluchistan Deep Dive ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: QCI trajectory comparison
ax = axes[0, 0]
ax.plot(sb_ts['year'], sb_ts['qci'], 'r-o', markersize=3, linewidth=2, label='Sistan and Baluchistan')
ax.plot(bp_ts['year'], bp_ts['qci'], 'g-s', markersize=3, linewidth=2, label=best_prov)
ax.plot(iran_ts['year'], iran_ts['qci'], 'k--', linewidth=1.5, label='Iran (national)')
ax.set_ylabel('QCI Score')
ax.set_title('(A) QCI Trajectory', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: MIR over time
ax = axes[0, 1]
sb_mir = sb_comp['MIR'].values
bp_mir = bp_comp['MIR'].values
ax.plot(sb_comp['year'], sb_mir, 'r-o', markersize=3, linewidth=2, label='Sistan and Baluchistan')
ax.plot(bp_comp['year'], bp_mir, 'g-s', markersize=3, linewidth=2, label=best_prov)
ax.set_ylabel('MIR')
ax.set_title('(B) Mortality-to-Incidence Ratio', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: YLLtoYLD over time
ax = axes[1, 0]
ax.plot(sb_comp['year'], sb_comp['YLLtoYLD'], 'r-o', markersize=3, linewidth=2, label='Sistan and Baluchistan')
ax.plot(bp_comp['year'], bp_comp['YLLtoYLD'], 'g-s', markersize=3, linewidth=2, label=best_prov)
ax.set_ylabel('YLL/YLD Ratio')
ax.set_xlabel('Year')
ax.set_title('(C) YLL-to-YLD Ratio', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel D: DALtoPER over time
ax = axes[1, 1]
ax.plot(sb_comp['year'], sb_comp['DALtoPER'], 'r-o', markersize=3, linewidth=2, label='Sistan and Baluchistan')
ax.plot(bp_comp['year'], bp_comp['DALtoPER'], 'g-s', markersize=3, linewidth=2, label=best_prov)
ax.set_ylabel('DALY/Prevalence Ratio')
ax.set_xlabel('Year')
ax.set_title('(D) DALY-to-Prevalence Ratio', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('Sistan and Baluchistan vs Chahar Mahaal and Bakhtiari: Component Analysis',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure5_sistan_deep_dive.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure5_sistan_deep_dive.png'), format='png')
plt.close()
print("  Saved figure5_sistan_deep_dive")


# ── FIGURE 6: Age-sex heatmap by province ─────────────────────────────────────
print("\n=== FIGURE 6: Age-Sex Heatmap ===")

# Create heatmap: rows=provinces (sorted by AS QCI), columns=age groups, for both sexes
age_order = ['<5 years', '5-14 years', '15-49 years', '50-69 years', '70+ years', 'Age-standardized']
prov_order = prov_only.sort_values('QCI_2021', ascending=False)['Province'].tolist()

# Both sexes heatmap
heat_data = []
for prov in prov_order:
    row = {'Province': prov}
    for age in age_order:
        val = df_age_sex[(df_age_sex['Province'] == prov) & (df_age_sex['Sex'] == 'Both') & (df_age_sex['Age'] == age)]
        row[age] = val.iloc[0]['QCI_2021'] if len(val) > 0 else np.nan
    heat_data.append(row)

heat_df = pd.DataFrame(heat_data).set_index('Province')

fig, ax = plt.subplots(figsize=(12, 13))
sns.heatmap(heat_df, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=0.5,
            ax=ax, vmin=88, vmax=100, annot_kws={'size': 8, 'fontweight': 'bold'},
            cbar_kws={'label': 'QCI Score', 'shrink': 0.7})
ax.set_title('QCI by Age Group and Province, Iran, 2021 (Both Sexes)', fontweight='bold', fontsize=13)
ax.set_ylabel('')
ax.set_xlabel('')
ax.tick_params(axis='y', labelsize=9)
plt.xticks(rotation=30, ha='right')

fig.savefig(os.path.join(OUTPUT_DIR, 'figure6_age_province_heatmap.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure6_age_province_heatmap.png'), format='png')
plt.close()
print("  Saved figure6_age_province_heatmap")


# ── FIGURE 7: GDR by province ────────────────────────────────────────────────
print("\n=== FIGURE 7: GDR by Province ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: GDR bar chart
ax = axes[0]
gdr_sorted = df_gdr.sort_values('GDR', ascending=True)
# Gradient color based on GDR value
gdr_norm = (gdr_sorted['GDR'] - gdr_sorted['GDR'].min()) / (gdr_sorted['GDR'].max() - gdr_sorted['GDR'].min())
colors = [plt.cm.RdBu(0.3 + v * 0.5) for v in gdr_norm]
ax.barh(range(len(gdr_sorted)), gdr_sorted['GDR'], color=colors)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax.set_yticks(range(len(gdr_sorted)))
ax.set_yticklabels(gdr_sorted['Province'], fontsize=8)
ax.set_xlabel('Gender Disparity Ratio (Female/Male)')
ax.set_title('(A) GDR by Province', fontweight='bold')
for i, (_, row) in enumerate(gdr_sorted.iterrows()):
    ax.text(row['GDR'] + 0.0003, i, f'{row["GDR"]:.4f}', va='center', fontsize=7)

# Panel B: Female vs Male scatter
ax = axes[1]
ax.scatter(df_gdr['QCI_Male'], df_gdr['QCI_Female'], c='steelblue', s=30, alpha=0.7)
lims = [min(df_gdr['QCI_Male'].min(), df_gdr['QCI_Female'].min()) - 0.3,
        max(df_gdr['QCI_Male'].max(), df_gdr['QCI_Female'].max()) + 0.3]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Equality line')
ax.set_xlabel('Male QCI')
ax.set_ylabel('Female QCI')
ax.set_title('(B) Female vs Male QCI by Province', fontweight='bold')
ax.legend()
# Label a few outliers
for _, row in df_gdr.nlargest(3, 'Gap').iterrows():
    ax.annotate(row['Province'], (row['QCI_Male'], row['QCI_Female']),
                fontsize=7, ha='left', xytext=(5, 5), textcoords='offset points')
for _, row in df_gdr.nsmallest(2, 'Gap').iterrows():
    ax.annotate(row['Province'], (row['QCI_Male'], row['QCI_Female']),
                fontsize=7, ha='left', xytext=(5, -8), textcoords='offset points')
ax.grid(True, alpha=0.3)

fig.suptitle('Gender Disparities in DS-TB Care Quality, Iranian Provinces, 2021',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure7_gender_disparity.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure7_gender_disparity.png'), format='png')
plt.close()
print("  Saved figure7_gender_disparity")


# ── FIGURE 8: Iran vs MENA benchmarks ────────────────────────────────────────
print("\n=== FIGURE 8: Iran vs Benchmarks ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Time series comparison
ax = axes[0]
benchmark_locs = {
    'Iran': ('Iran', 'black', '-', 2.5),
    'Global': ('Global', 'blue', '--', 1.5),
    'Middle East & North Africa - WB': ('MENA', 'green', '--', 1.5),
    'High-middle SDI': ('High-middle SDI', 'orange', '--', 1.5),
}

bench_as_data = df_qci[(df_qci['age_name'] == 'Age-standardized') & (df_qci['sex_name'] == 'Both')]

for loc, (label, color, ls, lw) in benchmark_locs.items():
    ldata = bench_as_data[bench_as_data['iso_location_name'] == loc].sort_values('year')
    if len(ldata) > 0:
        ax.plot(ldata['year'], ldata['qci'], color=color, linestyle=ls, linewidth=lw, label=label)

# Add band for provincial range
for year in range(1990, 2022):
    yr_prov = prov_as[(prov_as['iso_location_name'].isin(IRAN_PROVINCES)) & (prov_as['year'] == year)]
    if len(yr_prov) > 0:
        ax.fill_between([year, year], yr_prov['qci'].min(), yr_prov['qci'].max(),
                        alpha=0.1, color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('QCI Score')
ax.set_title('(A) Iran vs Regional and Global Benchmarks', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(1990, 2021)

# Panel B: MENA country comparison bar chart (2021)
ax = axes[1]
mena_sorted = df_mena_comp.sort_values('QCI_2021', ascending=True)
# Add Iran to the comparison
iran_2021 = bench_as_data[(bench_as_data['iso_location_name'] == 'Iran') & (bench_as_data['year'] == 2021)]['qci'].values[0]
iran_1990 = bench_as_data[(bench_as_data['iso_location_name'] == 'Iran') & (bench_as_data['year'] == 1990)]['qci'].values[0]
mena_plus_iran = pd.concat([mena_sorted,
                             pd.DataFrame([{'Country': 'Iran', 'QCI_2021': iran_2021, 'QCI_1990': iran_1990}])],
                            ignore_index=True)
mena_plus_iran = mena_plus_iran.sort_values('QCI_2021', ascending=True)

colors = ['#4CAF50' if c == 'Iran' else '#2196F3' for c in mena_plus_iran['Country']]
ax.barh(range(len(mena_plus_iran)), mena_plus_iran['QCI_2021'], color=colors)
ax.set_yticks(range(len(mena_plus_iran)))
ax.set_yticklabels(mena_plus_iran['Country'], fontsize=8)
ax.set_xlabel('QCI Score (2021)')
ax.set_title('(B) Iran vs MENA Countries, 2021', fontweight='bold')
ax.set_xlim(left=max(0, mena_plus_iran['QCI_2021'].min() - 2))
for i, (_, row) in enumerate(mena_plus_iran.iterrows()):
    ax.text(row['QCI_2021'] + 0.05, i, f'{row["QCI_2021"]:.1f}', va='center', fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure8_iran_vs_benchmarks.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure8_iran_vs_benchmarks.png'), format='png')
plt.close()
print("  Saved figure8_iran_vs_benchmarks")


# ── FIGURE 9: AAPC forest plot for provinces ──────────────────────────────────
print("\n=== FIGURE 9: AAPC Forest Plot ===")

fig, ax = plt.subplots(figsize=(10, 12))

aapc_sorted = prov_only.sort_values('AAPC_full', ascending=True).reset_index(drop=True)
y_pos = range(len(aapc_sorted))

ax.errorbar(aapc_sorted['AAPC_full'], y_pos,
            xerr=[aapc_sorted['AAPC_full'] - aapc_sorted['CI_lo_full'],
                  aapc_sorted['CI_hi_full'] - aapc_sorted['AAPC_full']],
            fmt='o', color='steelblue', markersize=5, capsize=3, elinewidth=1)

# Add Iran national line
iran_aapc = iran_row['AAPC_full']
ax.axvline(x=iran_aapc, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Iran national ({iran_aapc:.3f}%)')

ax.set_yticks(y_pos)
ax.set_yticklabels(aapc_sorted['Province'], fontsize=9)
ax.set_xlabel('AAPC (%, 1990-2021)')
ax.set_title('Average Annual Percent Change in QCI, Iranian Provinces, 1990-2021',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3, axis='x', linewidth=0.5)

# Add annotations for top/bottom
for i, (_, row) in enumerate(aapc_sorted.iterrows()):
    ax.text(row['CI_hi_full'] + 0.003, i, f'{row["AAPC_full"]:.3f}%', va='center', fontsize=8, color='#444444')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure9_aapc_forest_plot.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure9_aapc_forest_plot.png'), format='png')
plt.close()
print("  Saved figure9_aapc_forest_plot")


# ── FIGURE 10: Convergence/divergence analysis ───────────────────────────────
print("\n=== FIGURE 10: Convergence Analysis ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: 1990 QCI vs absolute change (sigma-convergence test)
ax = axes[0]
ax.scatter(prov_only['QCI_1990'], prov_only['Change_1990_2021'], c='steelblue', s=40, alpha=0.7)
# Fit regression line
slope, intercept, r, p, se = stats.linregress(prov_only['QCI_1990'], prov_only['Change_1990_2021'])
x_line = np.linspace(prov_only['QCI_1990'].min(), prov_only['QCI_1990'].max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5)
ax.set_xlabel('QCI in 1990')
ax.set_ylabel('Absolute Change, 1990-2021')
ax.set_title(f'(A) Beta-Convergence (r={r:.3f}, p={p:.4f})', fontweight='bold')
ax.grid(True, alpha=0.3)
# Label extreme points
for _, row in prov_only.nlargest(3, 'Change_1990_2021').iterrows():
    ax.annotate(row['Province'], (row['QCI_1990'], row['Change_1990_2021']),
                fontsize=7, xytext=(5, 5), textcoords='offset points')
for _, row in prov_only.nsmallest(2, 'Change_1990_2021').iterrows():
    ax.annotate(row['Province'], (row['QCI_1990'], row['Change_1990_2021']),
                fontsize=7, xytext=(5, -8), textcoords='offset points')

stats_dict['convergence'] = {
    'beta_slope': round(slope, 4),
    'beta_r': round(r, 4),
    'beta_p': round(p, 6),
    'interpretation': 'Negative slope indicates beta-convergence (catching up)' if slope < 0 else 'No beta-convergence'
}

# Panel B: SD over time (sigma-convergence)
ax = axes[1]
ax.plot(df_ineq['Year'], df_ineq['SD'], 'b-o', markersize=3, linewidth=1.5)
# Fit trend
slope_s, intercept_s, r_s, p_s, _ = stats.linregress(df_ineq['Year'], df_ineq['SD'])
x_trend = np.linspace(1990, 2021, 100)
ax.plot(x_trend, slope_s * x_trend + intercept_s, 'r--', linewidth=1, alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Standard Deviation of Provincial QCI')
ax.set_title(f'(B) Sigma-Convergence (slope={slope_s:.4f}/yr, p={p_s:.4f})', fontweight='bold')
ax.grid(True, alpha=0.3)

stats_dict['sigma_convergence'] = {
    'sd_slope': round(slope_s, 5),
    'sd_r': round(r_s, 4),
    'sd_p': round(p_s, 6),
    'interpretation': 'Negative slope indicates sigma-convergence (narrowing disparity)' if slope_s < 0 else 'No sigma-convergence'
}

fig.suptitle('Convergence in Provincial DS-TB Care Quality, Iran, 1990-2021',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure10_convergence.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure10_convergence.png'), format='png')
plt.close()
print("  Saved figure10_convergence")


# ── Save final stats ─────────────────────────────────────────────────────────
with open(STATS_PATH, 'w') as f:
    json.dump(stats_dict, f, indent=2, default=str)

# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("IRAN ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nTables saved to: {TABLE_DIR}")
print(f"Figures saved to: {OUTPUT_DIR}")
print(f"Statistics saved to: {STATS_PATH}")

print("\n--- KEY FINDINGS ---")
print(f"Iran national QCI: {stats_dict['iran_national']['qci_1990']} (1990) -> {stats_dict['iran_national']['qci_2021']} (2021)")
print(f"Best province: {stats_dict['best_province']['name']} ({stats_dict['best_province']['qci_2021']})")
print(f"Worst province: {stats_dict['worst_province']['name']} ({stats_dict['worst_province']['qci_2021']})")
print(f"Provincial range 2021: {stats_dict['provincial_range_2021']} points")
print(f"Provincial range 1990: {stats_dict['provincial_range_1990']} points")
print(f"All provinces above 95: {stats_dict['all_above_95']}")
print(f"Mean GDR: {stats_dict['gdr_mean']}")
print(f"Inequality CV 1990->2021: {stats_dict['inequality']['cv_1990']}% -> {stats_dict['inequality']['cv_2021']}%")
print(f"Gini 1990->2021: {stats_dict['inequality']['gini_1990']} -> {stats_dict['inequality']['gini_2021']}")
if 'convergence' in stats_dict:
    print(f"Beta-convergence: r={stats_dict['convergence']['beta_r']}, p={stats_dict['convergence']['beta_p']}")
if 'sigma_convergence' in stats_dict:
    print(f"Sigma-convergence: slope={stats_dict['sigma_convergence']['sd_slope']}/yr, p={stats_dict['sigma_convergence']['sd_p']}")
