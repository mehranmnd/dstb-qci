#!/usr/bin/env python3
"""
Paper 3 Enhancement: Spatial Autocorrelation and Decomposition Analysis

Adds:
  1. Global Moran's I for provincial QCI (1990, 2000, 2010, 2021)
  2. Local Moran's I (LISA) cluster map for 2021
  3. Shapley decomposition of QCI gap attribution (MIR vs YLLtoYLD vs DALtoPER)
  4. Updated figures (LISA map, decomposition chart)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import geopandas as gpd
from scipy import stats
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCI_PATH = os.path.join(BASE, 'results/shared/qci.csv')
QCI_COMPLETE_PATH = os.path.join(BASE, 'results/shared/qci_complete_data.csv')
OUTPUT_DIR = os.path.join(BASE, 'results/paper3/figures')
TABLE_DIR = os.path.join(BASE, 'results/paper3/tables')
STATS_PATH = os.path.join(BASE, 'results/paper3/stats.json')
GEOJSON_PATH = os.path.join(BASE, 'data/iran_shapefile/iran_provinces.geojson')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
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
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ── Iran province list ───────────────────────────────────────────────────────
IRAN_PROVINCES = [
    'Alborz', 'Ardebil', 'Bushehr', 'Chahar Mahaal and Bakhtiari',
    'East Azarbayejan', 'Fars', 'Gilan', 'Golestan', 'Hamadan',
    'Hormozgan', 'Ilam', 'Isfahan', 'Kerman', 'Kermanshah',
    'Khorasan-e-Razavi', 'Khuzestan', 'Kohgiluyeh and Boyer-Ahmad',
    'Kurdistan', 'Lorestan', 'Markazi', 'Mazandaran', 'North Khorasan',
    'Qazvin', 'Qom', 'Semnan', 'Sistan and Baluchistan', 'South Khorasan',
    'Tehran', 'West Azarbayejan', 'Yazd', 'Zanjan',
]

# Name mapping: Natural Earth -> QCI data
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
    'Fārs': 'Fars', 'Fars': 'Fars',
    'Gīlān': 'Gilan', 'Gilan': 'Gilan',
    'Golestān': 'Golestan', 'Golestan': 'Golestan',
    'Hamadān': 'Hamadan', 'Hamadan': 'Hamadan', 'Hamedan': 'Hamadan',
    'Hormozgān': 'Hormozgan', 'Hormozgan': 'Hormozgan',
    'Īlām': 'Ilam', 'Ilam': 'Ilam',
    'Eşfahān': 'Isfahan', 'Isfahan': 'Isfahan', 'Esfahan': 'Isfahan',
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
    'Khorāsān-e Jonūbī': 'South Khorasan', 'South Khorasan': 'South Khorasan',
    'Tehrān': 'Tehran', 'Tehran': 'Tehran',
    'Āz̄arbāyjān-e Gharbī': 'West Azarbayejan',
    'West Azarbaijan': 'West Azarbayejan', 'West Azerbaijan': 'West Azarbayejan',
    'West Azarbayejan': 'West Azarbayejan', 'Azarbayjan-e Gharbi': 'West Azarbayejan',
    'Yazd': 'Yazd',
    'Zanjān': 'Zanjan', 'Zanjan': 'Zanjan',
}

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df_qci = pd.read_csv(QCI_PATH)
df_complete = pd.read_csv(QCI_COMPLETE_PATH)

iran_mask = df_qci['iso_location_name'].isin(IRAN_PROVINCES)
df_iran = df_qci[iran_mask].copy()
prov_as = df_iran[(df_iran['age_name'] == 'Age-standardized') & (df_iran['sex_name'] == 'Both')]

iran_mask_c = df_complete['iso_location_name'].isin(IRAN_PROVINCES)
df_iran_c = df_complete[iran_mask_c].copy()
prov_comp_as = df_iran_c[(df_iran_c['age_name'] == 'Age-standardized') & (df_iran_c['sex_name'] == 'Both')]

# Load existing stats
with open(STATS_PATH, 'r') as f:
    stats_dict = json.load(f)

# Load GeoJSON
print("Loading GeoJSON...")
iran_gdf = gpd.read_file(GEOJSON_PATH)

# Find name column
name_col = None
for col in ['name', 'NAME', 'Name', 'name_en', 'NAME_1', 'name_1', 'NAME_EN', 'gn_name', 'woe_name']:
    if col in iran_gdf.columns:
        name_col = col
        break

print(f"  Name column: {name_col}")
iran_gdf['qci_province'] = iran_gdf[name_col].map(NE_TO_QCI)
matched = iran_gdf['qci_province'].notna().sum()
print(f"  Matched {matched}/{len(iran_gdf)} provinces")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: SPATIAL AUTOCORRELATION (Moran's I)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SPATIAL ANALYSIS: Moran's I ===")

from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local

# Build spatial weights from province geometry
# Use Queen contiguity (provinces sharing a border)
try:
    w_queen = Queen.from_dataframe(iran_gdf, use_index=False)
    w_queen.transform = 'R'  # Row-standardize
    print(f"  Queen weights: {w_queen.n} observations, mean neighbors: {w_queen.mean_neighbors:.1f}")
except Exception as e:
    print(f"  Queen weights failed: {e}, falling back to KNN")
    w_queen = KNN.from_dataframe(iran_gdf, k=5)
    w_queen.transform = 'R'

# Also compute KNN weights for robustness
w_knn = KNN.from_dataframe(iran_gdf, k=5)
w_knn.transform = 'R'

# Compute Global Moran's I for QCI at multiple time points
moran_results = []
for year in [1990, 2000, 2010, 2021]:
    yr_data = prov_as[prov_as['year'] == year][['iso_location_name', 'qci']].copy()

    # Merge with GeoDataFrame to get correct order
    gdf_yr = iran_gdf[['qci_province', 'geometry']].copy()
    gdf_yr = gdf_yr[gdf_yr['qci_province'].notna()]
    gdf_yr = gdf_yr.merge(yr_data, left_on='qci_province', right_on='iso_location_name', how='left')

    if gdf_yr['qci'].isna().sum() > 0:
        print(f"  Warning: {gdf_yr['qci'].isna().sum()} missing QCI values for {year}")
        gdf_yr = gdf_yr.dropna(subset=['qci'])

    # Rebuild weights for this subset if needed
    if len(gdf_yr) == len(iran_gdf):
        w_use = w_queen
    else:
        try:
            w_use = Queen.from_dataframe(gdf_yr.reset_index(drop=True), use_index=False)
            w_use.transform = 'R'
        except Exception:
            w_use = KNN.from_dataframe(gdf_yr.reset_index(drop=True), k=5)
            w_use.transform = 'R'

    # Global Moran's I
    moran = Moran(gdf_yr['qci'].values, w_use, permutations=999)

    # Also with KNN for robustness
    if len(gdf_yr) == len(iran_gdf):
        moran_knn = Moran(gdf_yr['qci'].values, w_knn, permutations=999)
    else:
        w_knn_sub = KNN.from_dataframe(gdf_yr.reset_index(drop=True), k=5)
        w_knn_sub.transform = 'R'
        moran_knn = Moran(gdf_yr['qci'].values, w_knn_sub, permutations=999)

    result = {
        'Year': year,
        'Morans_I_Queen': round(moran.I, 4),
        'p_value_Queen': round(moran.p_sim, 4),
        'z_score_Queen': round(moran.z_sim, 4),
        'Expected_I': round(moran.EI, 4),
        'Morans_I_KNN5': round(moran_knn.I, 4),
        'p_value_KNN5': round(moran_knn.p_sim, 4),
    }
    moran_results.append(result)
    print(f"  {year}: Moran's I={moran.I:.4f}, p={moran.p_sim:.4f} (Queen); I={moran_knn.I:.4f}, p={moran_knn.p_sim:.4f} (KNN-5)")

df_moran = pd.DataFrame(moran_results)
df_moran.to_csv(os.path.join(TABLE_DIR, 'table6_morans_i.csv'), index=False, float_format='%.4f')
print(f"  Table 6 saved: Moran's I results")

# ── LISA clusters for 2021 ───────────────────────────────────────────────────
print("\n=== LISA Cluster Analysis (2021) ===")

# Get 2021 data merged with GeoDataFrame
gdf_2021 = iran_gdf[['qci_province', 'geometry']].copy()
gdf_2021 = gdf_2021[gdf_2021['qci_province'].notna()].reset_index(drop=True)
qci_2021 = prov_as[prov_as['year'] == 2021][['iso_location_name', 'qci']].copy()
gdf_2021 = gdf_2021.merge(qci_2021, left_on='qci_province', right_on='iso_location_name', how='left')

# Rebuild weights for matched provinces
try:
    w_lisa = Queen.from_dataframe(gdf_2021, use_index=False)
    w_lisa.transform = 'R'
except Exception:
    w_lisa = KNN.from_dataframe(gdf_2021, k=5)
    w_lisa.transform = 'R'

# Local Moran's I
lisa = Moran_Local(gdf_2021['qci'].values, w_lisa, permutations=999)

# LISA cluster types: 1=HH, 2=LH, 3=LL, 4=HL, 0=NS
# esda labels: q values are 1-4 (HH, LH, LL, HL); significant at p < 0.05
sig_threshold = 0.05
gdf_2021['lisa_I'] = lisa.Is
gdf_2021['lisa_p'] = lisa.p_sim
gdf_2021['lisa_q'] = lisa.q  # quadrant: 1=HH, 2=LH, 3=LL, 4=HL

# Classify clusters
def classify_lisa(row):
    if row['lisa_p'] > sig_threshold:
        return 'Not Significant'
    q = row['lisa_q']
    if q == 1:
        return 'High-High'
    elif q == 2:
        return 'Low-High'
    elif q == 3:
        return 'Low-Low'
    elif q == 4:
        return 'High-Low'
    return 'Not Significant'

gdf_2021['lisa_cluster'] = gdf_2021.apply(classify_lisa, axis=1)

# Count clusters
cluster_counts = gdf_2021['lisa_cluster'].value_counts()
print(f"  LISA clusters: {dict(cluster_counts)}")

# Save LISA results table
lisa_table = gdf_2021[['qci_province', 'qci', 'lisa_I', 'lisa_p', 'lisa_cluster']].copy()
lisa_table.columns = ['Province', 'QCI_2021', 'Local_I', 'p_value', 'Cluster']
lisa_table = lisa_table.sort_values('QCI_2021', ascending=False)
lisa_table.to_csv(os.path.join(TABLE_DIR, 'table7_lisa_clusters.csv'), index=False, float_format='%.4f')
print(f"  Table 7 saved: LISA cluster assignments")

# ── FIGURE 11: LISA Cluster Map ──────────────────────────────────────────────
print("\n=== FIGURE 11: LISA Cluster Map ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.2]})

# Panel A: Moran scatterplot
ax = axes[0]
# Standardize QCI
qci_vals = gdf_2021['qci'].values
z = (qci_vals - qci_vals.mean()) / qci_vals.std()
# Spatial lag
lag = np.array([np.sum(w_lisa.weights[i] * z[list(w_lisa.neighbors[i])]) for i in range(len(z))])

colors_scatter = []
for i in range(len(z)):
    if gdf_2021.iloc[i]['lisa_p'] > sig_threshold:
        colors_scatter.append('#CCCCCC')
    else:
        q = gdf_2021.iloc[i]['lisa_q']
        if q == 1:
            colors_scatter.append('#d7191c')  # HH red
        elif q == 2:
            colors_scatter.append('#abd9e9')  # LH light blue
        elif q == 3:
            colors_scatter.append('#2c7bb6')  # LL blue
        elif q == 4:
            colors_scatter.append('#fdae61')  # HL orange
        else:
            colors_scatter.append('#CCCCCC')

ax.scatter(z, lag, c=colors_scatter, s=40, edgecolors='black', linewidth=0.5, zorder=5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Regression line
slope_m, intercept_m, _, _, _ = stats.linregress(z, lag)
x_line = np.linspace(z.min(), z.max(), 100)
ax.plot(x_line, slope_m * x_line + intercept_m, 'r-', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Standardized QCI (z)')
ax.set_ylabel('Spatial Lag of QCI')
moran_2021 = df_moran[df_moran['Year'] == 2021].iloc[0]
ax.set_title(f"(A) Moran Scatterplot (I={moran_2021['Morans_I_Queen']:.3f}, p={moran_2021['p_value_Queen']:.3f})",
             fontweight='bold')

# Label significant provinces
for i in range(len(z)):
    if gdf_2021.iloc[i]['lisa_p'] <= sig_threshold:
        ax.annotate(gdf_2021.iloc[i]['qci_province'], (z[i], lag[i]),
                    fontsize=7, ha='left', xytext=(4, 4), textcoords='offset points')

ax.grid(True, alpha=0.2)
# Add quadrant labels
xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.text(xlims[1]*0.7, ylims[1]*0.7, 'HH', fontsize=14, color='#d7191c', alpha=0.4, fontweight='bold')
ax.text(xlims[0]*0.7, ylims[1]*0.7, 'LH', fontsize=14, color='#abd9e9', alpha=0.6, fontweight='bold')
ax.text(xlims[0]*0.7, ylims[0]*0.7, 'LL', fontsize=14, color='#2c7bb6', alpha=0.4, fontweight='bold')
ax.text(xlims[1]*0.7, ylims[0]*0.7, 'HL', fontsize=14, color='#fdae61', alpha=0.6, fontweight='bold')

# Panel B: LISA Cluster Map
ax = axes[1]
cluster_colors = {
    'High-High': '#d7191c',
    'Low-Low': '#2c7bb6',
    'High-Low': '#fdae61',
    'Low-High': '#abd9e9',
    'Not Significant': '#f0f0f0',
}

for cluster_type, color in cluster_colors.items():
    subset = gdf_2021[gdf_2021['lisa_cluster'] == cluster_type]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5)

# Add province labels
for idx, row in gdf_2021.iterrows():
    centroid = row.geometry.centroid
    label = row['qci_province']
    if len(label) > 12:
        # Abbreviate long names
        parts = label.split()
        if len(parts) > 2:
            label = parts[0][:3] + '.'
        else:
            label = label[:10] + '.'
    fontsize = 6.5 if row['lisa_cluster'] == 'Not Significant' else 7.5
    fontweight = 'normal' if row['lisa_cluster'] == 'Not Significant' else 'bold'
    ax.annotate(label, (centroid.x, centroid.y), fontsize=fontsize, ha='center', va='center',
                fontweight=fontweight)

ax.set_title('(B) LISA Cluster Map, 2021', fontweight='bold')
ax.set_axis_off()

# Legend
legend_elements = [Patch(facecolor=c, edgecolor='black', label=l)
                   for l, c in cluster_colors.items() if l in gdf_2021['lisa_cluster'].values]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)

fig.suptitle('Spatial Autocorrelation of DS-TB Quality of Care, Iran, 2021',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure11_lisa_clusters.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure11_lisa_clusters.png'), format='png')
plt.close()
print("  Saved figure11_lisa_clusters")


# ── FIGURE 12: Moran's I Over Time ──────────────────────────────────────────
print("\n=== FIGURE 12: Moran's I Over Time ===")

# Compute Moran's I for every year 1990-2021
moran_yearly = []
for year in range(1990, 2022):
    yr_data = prov_as[prov_as['year'] == year][['iso_location_name', 'qci']].copy()
    gdf_yr = iran_gdf[['qci_province', 'geometry']].copy()
    gdf_yr = gdf_yr[gdf_yr['qci_province'].notna()].reset_index(drop=True)
    gdf_yr = gdf_yr.merge(yr_data, left_on='qci_province', right_on='iso_location_name', how='left')
    gdf_yr = gdf_yr.dropna(subset=['qci']).reset_index(drop=True)

    if len(gdf_yr) < 10:
        continue

    try:
        w_yr = Queen.from_dataframe(gdf_yr, use_index=False)
        w_yr.transform = 'R'
    except Exception:
        w_yr = KNN.from_dataframe(gdf_yr, k=5)
        w_yr.transform = 'R'

    moran_yr = Moran(gdf_yr['qci'].values, w_yr, permutations=999)
    moran_yearly.append({
        'Year': year,
        'Morans_I': moran_yr.I,
        'p_value': moran_yr.p_sim,
        'z_score': moran_yr.z_sim,
    })

df_moran_yearly = pd.DataFrame(moran_yearly)

fig, ax = plt.subplots(figsize=(10, 5))
# Color by significance
colors_ts = ['#d7191c' if p < 0.05 else '#2c7bb6' for p in df_moran_yearly['p_value']]
ax.bar(df_moran_yearly['Year'], df_moran_yearly['Morans_I'], color=colors_ts, alpha=0.8, width=0.8)
ax.axhline(0, color='black', linewidth=0.5)

# Add significance line
ax.plot(df_moran_yearly['Year'], df_moran_yearly['Morans_I'], 'k-', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Year')
ax.set_ylabel("Moran's I")
ax.set_title("Global Moran's I for Provincial QCI, Iran, 1990-2021", fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# Legend
legend_elements = [
    Patch(facecolor='#d7191c', label='Significant (p<0.05)'),
    Patch(facecolor='#2c7bb6', label='Not significant'),
]
ax.legend(handles=legend_elements, fontsize=9)

fig.savefig(os.path.join(OUTPUT_DIR, 'figure12_morans_i_timeseries.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure12_morans_i_timeseries.png'), format='png')
plt.close()
print("  Saved figure12_morans_i_timeseries")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: SHAPLEY DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== DECOMPOSITION ANALYSIS: Shapley Value ===")

# The QCI is derived from PCA on three component ratios:
#   MIR (Mortality-to-Incidence Ratio)
#   YLLtoYLD (YLL/YLD ratio)
#   DALtoPER (DALY/Prevalence ratio)
#
# We decompose the provincial QCI gap relative to the national best
# using Shapley values to attribute how much of the gap comes from each component.
#
# Shapley approach: For each coalition of features, compute the "QCI" from those
# features at the province level and the rest at the reference level.
# Since QCI = PC1 of (MIR, YLLtoYLD, DALtoPER), we use the PCA loadings.

from itertools import combinations

# Get PCA loadings from the original PCA
# The QCI was computed via PCA on standardized (MIR, YLLtoYLD, DALtoPER)
# We need the PC1 loadings and the standardization parameters

# Recompute PCA to get exact loadings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Use all 2021 provincial data
comp_2021 = prov_comp_as[prov_comp_as['year'] == 2021][['iso_location_name', 'MIR', 'YLLtoYLD', 'DALtoPER', 'pca_score']].copy()
comp_2021 = comp_2021[comp_2021['iso_location_name'].isin(IRAN_PROVINCES)].reset_index(drop=True)

# But the original PCA was on ALL countries, not just Iran
# So we use a simpler approach: direct marginal contribution

# Approach: For each province, compute the gap from the best province (Chahar Mahaal)
# Then decompose: how much would QCI improve if we replaced each ratio with the best province's value?

best_prov = 'Chahar Mahaal and Bakhtiari'
best_row = comp_2021[comp_2021['iso_location_name'] == best_prov].iloc[0]

# For Shapley decomposition, we need a function that maps (MIR, YLLtoYLD, DALtoPER) -> QCI
# Since QCI = f(MIR, YLLtoYLD, DALtoPER) via PCA, and PCA is linear,
# we can approximate this using the original PCA model.

# Alternative: Use a simpler percentile-rank approach or the actual PCA model
# Since we have the complete data, let's use the regression approach:
# Fit QCI_2021 ~ MIR_2021 + YLLtoYLD_2021 + DALtoPER_2021 across all 31 provinces

from numpy.linalg import lstsq

X = comp_2021[['MIR', 'YLLtoYLD', 'DALtoPER']].values
y = comp_2021['pca_score'].values

# Add intercept
X_int = np.column_stack([np.ones(len(X)), X])
beta, residuals, rank, sv = lstsq(X_int, y, rcond=None)
print(f"  Linear model R²: {1 - np.sum((y - X_int @ beta)**2) / np.sum((y - y.mean())**2):.4f}")
print(f"  Coefficients: intercept={beta[0]:.4f}, MIR={beta[1]:.4f}, YLLtoYLD={beta[2]:.4f}, DALtoPER={beta[3]:.4f}")

# Function to predict QCI from components
def predict_qci(mir, yllyld, dalper):
    return beta[0] + beta[1]*mir + beta[2]*yllyld + beta[3]*dalper

# Shapley decomposition for each province
features = ['MIR', 'YLLtoYLD', 'DALtoPER']
n_features = len(features)
feature_indices = {f: i for i, f in enumerate(features)}

shapley_rows = []
for _, prov_row in comp_2021.iterrows():
    prov = prov_row['iso_location_name']
    if prov == best_prov:
        shapley_rows.append({
            'Province': prov,
            'QCI_2021': prov_row['pca_score'],
            'Gap_from_best': 0,
            'Shapley_MIR': 0, 'Shapley_YLLtoYLD': 0, 'Shapley_DALtoPER': 0,
            'Pct_MIR': 0, 'Pct_YLLtoYLD': 0, 'Pct_DALtoPER': 0,
        })
        continue

    # Province and reference (best) component values
    prov_vals = {f: prov_row[f] for f in features}
    ref_vals = {f: best_row[f] for f in features}

    # Total gap
    qci_prov = predict_qci(prov_vals['MIR'], prov_vals['YLLtoYLD'], prov_vals['DALtoPER'])
    qci_ref = predict_qci(ref_vals['MIR'], ref_vals['YLLtoYLD'], ref_vals['DALtoPER'])
    total_gap = qci_ref - qci_prov

    # Shapley values
    shapley = {f: 0.0 for f in features}
    for f in features:
        other_features = [of for of in features if of != f]
        # All coalitions not containing f
        for size in range(n_features):
            for coalition in combinations(other_features, size):
                coalition_set = set(coalition)
                # v(S union {f}) - v(S)
                # S: coalition members use reference values, others use province values
                # S union {f}: coalition + f use reference values, rest use province values

                # v(S): QCI with S features at reference, rest at province
                vals_s = {}
                for feat in features:
                    if feat in coalition_set:
                        vals_s[feat] = ref_vals[feat]
                    else:
                        vals_s[feat] = prov_vals[feat]
                v_s = predict_qci(vals_s['MIR'], vals_s['YLLtoYLD'], vals_s['DALtoPER'])

                # v(S union {f}): same but with f also at reference
                vals_sf = vals_s.copy()
                vals_sf[f] = ref_vals[f]
                v_sf = predict_qci(vals_sf['MIR'], vals_sf['YLLtoYLD'], vals_sf['DALtoPER'])

                # Marginal contribution
                marginal = v_sf - v_s

                # Weight: |S|! * (n - |S| - 1)! / n!
                s_size = len(coalition_set)
                import math
                weight = math.factorial(s_size) * math.factorial(n_features - s_size - 1) / math.factorial(n_features)

                shapley[f] += weight * marginal

    # Percentage attribution
    total_shapley = sum(shapley.values())
    pct = {f: (shapley[f] / total_shapley * 100) if total_shapley != 0 else 0 for f in features}

    shapley_rows.append({
        'Province': prov,
        'QCI_2021': prov_row['pca_score'],
        'Gap_from_best': round(total_gap, 4),
        'Shapley_MIR': round(shapley['MIR'], 4),
        'Shapley_YLLtoYLD': round(shapley['YLLtoYLD'], 4),
        'Shapley_DALtoPER': round(shapley['DALtoPER'], 4),
        'Pct_MIR': round(pct['MIR'], 1),
        'Pct_YLLtoYLD': round(pct['YLLtoYLD'], 1),
        'Pct_DALtoPER': round(pct['DALtoPER'], 1),
    })

df_shapley = pd.DataFrame(shapley_rows).sort_values('Gap_from_best', ascending=False)
df_shapley.to_csv(os.path.join(TABLE_DIR, 'table8_shapley_decomposition.csv'), index=False, float_format='%.4f')
print(f"  Table 8 saved: Shapley decomposition")

# Print summary
print("\n  Top 5 provinces by QCI gap from best:")
for _, row in df_shapley.head(5).iterrows():
    print(f"    {row['Province']}: Gap={row['Gap_from_best']:.2f} | MIR={row['Pct_MIR']:.0f}% | "
          f"YLLtoYLD={row['Pct_YLLtoYLD']:.0f}% | DALtoPER={row['Pct_DALtoPER']:.0f}%")

# Mean decomposition across all provinces (excluding best)
non_best = df_shapley[df_shapley['Province'] != best_prov]
mean_pct = {
    'MIR': non_best['Pct_MIR'].mean(),
    'YLLtoYLD': non_best['Pct_YLLtoYLD'].mean(),
    'DALtoPER': non_best['Pct_DALtoPER'].mean(),
}
print(f"\n  Mean attribution across all provinces:")
print(f"    MIR: {mean_pct['MIR']:.1f}%, YLLtoYLD: {mean_pct['YLLtoYLD']:.1f}%, DALtoPER: {mean_pct['DALtoPER']:.1f}%")


# ── FIGURE 13: Shapley Decomposition Chart ───────────────────────────────────
print("\n=== FIGURE 13: Decomposition Chart ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Panel A: Stacked bar chart of Shapley contributions
ax = axes[0]
decomp_plot = df_shapley[df_shapley['Province'] != best_prov].sort_values('Gap_from_best', ascending=True)

y_pos = range(len(decomp_plot))
bar_mir = decomp_plot['Shapley_MIR'].values
bar_yll = decomp_plot['Shapley_YLLtoYLD'].values
bar_dal = decomp_plot['Shapley_DALtoPER'].values

ax.barh(y_pos, bar_mir, color='#e74c3c', label='MIR', height=0.7)
ax.barh(y_pos, bar_yll, left=bar_mir, color='#3498db', label='YLL/YLD', height=0.7)
ax.barh(y_pos, bar_dal, left=bar_mir + bar_yll, color='#2ecc71', label='DALY/Prev', height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(decomp_plot['Province'], fontsize=9)
ax.set_xlabel('Shapley Contribution to QCI Gap (points)')
ax.set_title('(A) Decomposition of QCI Gap from Best Province', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.2, axis='x', linewidth=0.5)

# Panel B: Percentage stacked bar
ax = axes[1]
ax.barh(y_pos, decomp_plot['Pct_MIR'].values, color='#e74c3c', label='MIR', height=0.7)
ax.barh(y_pos, decomp_plot['Pct_YLLtoYLD'].values, left=decomp_plot['Pct_MIR'].values,
        color='#3498db', label='YLL/YLD', height=0.7)
ax.barh(y_pos, decomp_plot['Pct_DALtoPER'].values,
        left=decomp_plot['Pct_MIR'].values + decomp_plot['Pct_YLLtoYLD'].values,
        color='#2ecc71', label='DALY/Prev', height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([])
ax.set_xlabel('Contribution (%)')
ax.set_title('(B) Percentage Attribution of QCI Gap', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.2, axis='x')

# Add percentage labels
for i, (_, row) in enumerate(decomp_plot.iterrows()):
    ax.text(row['Pct_MIR']/2, i, f"{row['Pct_MIR']:.0f}%", ha='center', va='center', fontsize=7, color='white', fontweight='bold')

fig.suptitle('Shapley Decomposition of Provincial QCI Gaps, Iran, 2021',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure13_shapley_decomposition.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure13_shapley_decomposition.png'), format='png')
plt.close()
print("  Saved figure13_shapley_decomposition")


# ══════════════════════════════════════════════════════════════════════════════
# UPDATE STATS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Updating Statistics ===")

# Spatial stats
stats_dict['spatial'] = {
    'morans_i_2021': round(df_moran[df_moran['Year'] == 2021].iloc[0]['Morans_I_Queen'], 4),
    'morans_i_p_2021': round(df_moran[df_moran['Year'] == 2021].iloc[0]['p_value_Queen'], 4),
    'morans_i_1990': round(df_moran[df_moran['Year'] == 1990].iloc[0]['Morans_I_Queen'], 4),
    'morans_i_p_1990': round(df_moran[df_moran['Year'] == 1990].iloc[0]['p_value_Queen'], 4),
    'lisa_hh_count': int(cluster_counts.get('High-High', 0)),
    'lisa_ll_count': int(cluster_counts.get('Low-Low', 0)),
    'lisa_hl_count': int(cluster_counts.get('High-Low', 0)),
    'lisa_lh_count': int(cluster_counts.get('Low-High', 0)),
    'lisa_ns_count': int(cluster_counts.get('Not Significant', 0)),
    'lisa_hh_provinces': gdf_2021[gdf_2021['lisa_cluster'] == 'High-High']['qci_province'].tolist(),
    'lisa_ll_provinces': gdf_2021[gdf_2021['lisa_cluster'] == 'Low-Low']['qci_province'].tolist(),
    'lisa_hl_provinces': gdf_2021[gdf_2021['lisa_cluster'] == 'High-Low']['qci_province'].tolist(),
    'lisa_lh_provinces': gdf_2021[gdf_2021['lisa_cluster'] == 'Low-High']['qci_province'].tolist(),
}

# Decomposition stats
stats_dict['decomposition'] = {
    'model_r2': round(1 - np.sum((y - X_int @ beta)**2) / np.sum((y - y.mean())**2), 4),
    'mean_pct_mir': round(mean_pct['MIR'], 1),
    'mean_pct_yllyld': round(mean_pct['YLLtoYLD'], 1),
    'mean_pct_dalper': round(mean_pct['DALtoPER'], 1),
    'sistan_gap': round(df_shapley[df_shapley['Province'] == 'Sistan and Baluchistan']['Gap_from_best'].values[0], 2),
    'sistan_pct_mir': round(df_shapley[df_shapley['Province'] == 'Sistan and Baluchistan']['Pct_MIR'].values[0], 1),
}

with open(STATS_PATH, 'w') as f:
    json.dump(stats_dict, f, indent=2, default=str)
print(f"  Statistics updated in {STATS_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SPATIAL & DECOMPOSITION ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nNew tables:")
print(f"  Table 6: {os.path.join(TABLE_DIR, 'table6_morans_i.csv')}")
print(f"  Table 7: {os.path.join(TABLE_DIR, 'table7_lisa_clusters.csv')}")
print(f"  Table 8: {os.path.join(TABLE_DIR, 'table8_shapley_decomposition.csv')}")

print(f"\nNew figures:")
print(f"  Figure 11: LISA cluster map")
print(f"  Figure 12: Moran's I time series")
print(f"  Figure 13: Shapley decomposition")

print(f"\n--- KEY SPATIAL FINDINGS ---")
for _, row in df_moran.iterrows():
    sig = "***" if row['p_value_Queen'] < 0.001 else ("**" if row['p_value_Queen'] < 0.01 else ("*" if row['p_value_Queen'] < 0.05 else "ns"))
    print(f"  {int(row['Year'])}: Moran's I = {row['Morans_I_Queen']:.4f} {sig}")

print(f"\n--- LISA CLUSTERS ---")
for cluster_type in ['High-High', 'Low-Low', 'High-Low', 'Low-High']:
    provs = gdf_2021[gdf_2021['lisa_cluster'] == cluster_type]['qci_province'].tolist()
    if provs:
        print(f"  {cluster_type}: {', '.join(provs)}")

print(f"\n--- DECOMPOSITION FINDINGS ---")
print(f"  Mean attribution: MIR={mean_pct['MIR']:.1f}%, YLLtoYLD={mean_pct['YLLtoYLD']:.1f}%, DALtoPER={mean_pct['DALtoPER']:.1f}%")
sb_row = df_shapley[df_shapley['Province'] == 'Sistan and Baluchistan'].iloc[0]
print(f"  Sistan: Gap={sb_row['Gap_from_best']:.2f}, MIR={sb_row['Pct_MIR']:.0f}%, YLLtoYLD={sb_row['Pct_YLLtoYLD']:.0f}%, DALtoPER={sb_row['Pct_DALtoPER']:.0f}%")
