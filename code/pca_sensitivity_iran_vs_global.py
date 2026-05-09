#!/usr/bin/env python3
"""
PCA Sensitivity Analysis: Iran-only PCA vs Global PCA

This script tests whether the global PCA parameterisation (used in Paper 1)
is appropriate for the Iran subnational analysis (Paper 3) by:

1. Running PCA on Iran's 31 provinces only (age-standardized, both-sexes)
2. Comparing loadings, explained variance, and resulting QCI scores
   with the globally-derived QCI
3. Computing Spearman/Pearson correlations and rank agreement
4. Producing a supplementary table and figure for Paper 3

Methodology precedent: The GBD Healthcare Access and Quality (HAQ) Index
derives PCA weights at the country level and applies them to subnational
locations (Lancet, 2018; doi:10.1016/S0140-6736(18)30994-2).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = '/Users/mehranmamandipoor/Desktop/thesis'
QCI_COMPLETE_PATH = os.path.join(BASE, 'results/shared/qci_complete_data.csv')
OUTPUT_DIR = os.path.join(BASE, 'results/paper3/figures')
TABLE_DIR = os.path.join(BASE, 'results/paper3/tables')
STATS_PATH = os.path.join(BASE, 'results/paper3/pca_sensitivity_results.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

FEATURES = ['MIR', 'YLLtoYLD', 'DALtoPER']

IRAN_PROVINCES = [
    'Alborz', 'Ardebil', 'Bushehr', 'Chahar Mahaal and Bakhtiari',
    'East Azarbayejan', 'Fars', 'Gilan', 'Golestan', 'Hamadan',
    'Hormozgan', 'Ilam', 'Isfahan', 'Kerman', 'Kermanshah',
    'Khorasan-e-Razavi', 'Khuzestan', 'Kohgiluyeh and Boyer-Ahmad',
    'Kurdistan', 'Lorestan', 'Markazi', 'Mazandaran', 'North Khorasan',
    'Qazvin', 'Qom', 'Semnan', 'Sistan and Baluchistan', 'South Khorasan',
    'Tehran', 'West Azarbayejan', 'Yazd', 'Zanjan',
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(QCI_COMPLETE_PATH)

# Global training set: age-standardized, both-sexes (same as original PCA)
df_global_train = df[(df['age_name'] == 'Age-standardized') &
                     (df['sex_name'] == 'Both')].dropna(subset=FEATURES).copy()

# Iran provinces only: age-standardized, both-sexes
df_iran = df_global_train[df_global_train['iso_location_name'].isin(IRAN_PROVINCES)].copy()

print(f"  Global training set: {len(df_global_train)} observations "
      f"({df_global_train['iso_location_name'].nunique()} locations x "
      f"{df_global_train['year'].nunique()} years)")
print(f"  Iran subset: {len(df_iran)} observations "
      f"({df_iran['iso_location_name'].nunique()} provinces x "
      f"{df_iran['year'].nunique()} years)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RUN GLOBAL PCA (replicate original)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== GLOBAL PCA (replication) ===")

scaler_global = StandardScaler().fit(df_global_train[FEATURES].values)
Z_global = scaler_global.transform(df_global_train[FEATURES].values)
pca_global = PCA(n_components=1).fit(Z_global)

pc1_global_all = pca_global.transform(Z_global).ravel()
# Orient: higher = better (negative correlation with MIR)
sign_global = -1 if np.corrcoef(pc1_global_all, df_global_train['MIR'].values)[0, 1] > 0 else 1
pc1_global_all *= sign_global

loadings_global = pca_global.components_[0] * sign_global
var_explained_global = pca_global.explained_variance_ratio_[0] * 100

print(f"  Variance explained: {var_explained_global:.2f}%")
print(f"  Loadings: MIR={loadings_global[0]:.4f}, "
      f"YLLtoYLD={loadings_global[1]:.4f}, DALtoPER={loadings_global[2]:.4f}")

# Extract Iran scores from global PCA
iran_idx = df_global_train['iso_location_name'].isin(IRAN_PROVINCES)
pc1_iran_from_global = pc1_global_all[iran_idx.values]

# Rescale Iran scores to 0-100 using GLOBAL min-max (as done in original)
lo_global = pc1_global_all.min()
hi_global = pc1_global_all.max()
qci_iran_global = 100 * (pc1_iran_from_global - lo_global) / (hi_global - lo_global)

# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN IRAN-ONLY PCA
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== IRAN-ONLY PCA ===")

iran_features = df_iran[FEATURES].values
scaler_iran = StandardScaler().fit(iran_features)
Z_iran = scaler_iran.transform(iran_features)
pca_iran = PCA(n_components=1).fit(Z_iran)

pc1_iran_local = pca_iran.transform(Z_iran).ravel()
# Orient: higher = better
sign_iran = -1 if np.corrcoef(pc1_iran_local, df_iran['MIR'].values)[0, 1] > 0 else 1
pc1_iran_local *= sign_iran

loadings_iran = pca_iran.components_[0] * sign_iran
var_explained_iran = pca_iran.explained_variance_ratio_[0] * 100

print(f"  Variance explained: {var_explained_iran:.2f}%")
print(f"  Loadings: MIR={loadings_iran[0]:.4f}, "
      f"YLLtoYLD={loadings_iran[1]:.4f}, DALtoPER={loadings_iran[2]:.4f}")

# Rescale Iran-only scores to 0-100 using Iran-only min-max
lo_iran = pc1_iran_local.min()
hi_iran = pc1_iran_local.max()
qci_iran_local = 100 * (pc1_iran_local - lo_iran) / (hi_iran - lo_iran)

# ══════════════════════════════════════════════════════════════════════════════
# 4. ALSO RUN MENA-ONLY PCA (intermediate check)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== MENA-ONLY PCA ===")

mena_countries = ['Iran', 'Turkey', 'Saudi Arabia', 'Iraq', 'Algeria', 'Egypt',
                  'Morocco', 'Tunisia', 'Jordan', 'Lebanon', 'Libya', 'Syria',
                  'Yemen', 'Oman', 'Kuwait', 'Bahrain', 'Qatar',
                  'United Arab Emirates', 'Palestine'] + IRAN_PROVINCES

df_mena = df_global_train[df_global_train['iso_location_name'].isin(mena_countries)].copy()
print(f"  MENA set: {len(df_mena)} observations ({df_mena['iso_location_name'].nunique()} locations)")

mena_features = df_mena[FEATURES].values
scaler_mena = StandardScaler().fit(mena_features)
Z_mena = scaler_mena.transform(mena_features)
pca_mena = PCA(n_components=1).fit(Z_mena)

pc1_mena_all = pca_mena.transform(Z_mena).ravel()
sign_mena = -1 if np.corrcoef(pc1_mena_all, df_mena['MIR'].values)[0, 1] > 0 else 1
pc1_mena_all *= sign_mena

loadings_mena = pca_mena.components_[0] * sign_mena
var_explained_mena = pca_mena.explained_variance_ratio_[0] * 100

# Extract Iran provinces from MENA PCA
iran_in_mena_idx = df_mena['iso_location_name'].isin(IRAN_PROVINCES)
pc1_iran_from_mena = pc1_mena_all[iran_in_mena_idx.values]
lo_mena = pc1_mena_all.min()
hi_mena = pc1_mena_all.max()
qci_iran_mena = 100 * (pc1_iran_from_mena - lo_mena) / (hi_mena - lo_mena)

print(f"  Variance explained: {var_explained_mena:.2f}%")
print(f"  Loadings: MIR={loadings_mena[0]:.4f}, "
      f"YLLtoYLD={loadings_mena[1]:.4f}, DALtoPER={loadings_mena[2]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARE LOADINGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LOADING COMPARISON")
print("=" * 70)

loadings_table = pd.DataFrame({
    'Component': FEATURES,
    'Global_Loading': loadings_global,
    'Global_Squared': (loadings_global**2 / (loadings_global**2).sum()),
    'Iran_Loading': loadings_iran,
    'Iran_Squared': (loadings_iran**2 / (loadings_iran**2).sum()),
    'MENA_Loading': loadings_mena,
    'MENA_Squared': (loadings_mena**2 / (loadings_mena**2).sum()),
})

print(loadings_table.to_string(index=False, float_format='%.4f'))

# Cosine similarity between loading vectors
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_global_iran = cosine_sim(loadings_global, loadings_iran)
cos_global_mena = cosine_sim(loadings_global, loadings_mena)
cos_iran_mena = cosine_sim(loadings_iran, loadings_mena)

print(f"\nCosine similarity (loading vectors):")
print(f"  Global vs Iran-only: {cos_global_iran:.6f}")
print(f"  Global vs MENA-only: {cos_global_mena:.6f}")
print(f"  Iran-only vs MENA:   {cos_iran_mena:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARE QCI SCORES (Iran provinces, all years)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SCORE COMPARISON (Iran provinces, age-standardized, both-sexes)")
print("=" * 70)

# All pairwise correlations
pearson_global_iran, p_pearson_gi = stats.pearsonr(qci_iran_global, qci_iran_local)
spearman_global_iran, p_spearman_gi = stats.spearmanr(qci_iran_global, qci_iran_local)

pearson_global_mena, p_pearson_gm = stats.pearsonr(qci_iran_global, qci_iran_mena)
spearman_global_mena, p_spearman_gm = stats.spearmanr(qci_iran_global, qci_iran_mena)

print(f"\nGlobal QCI vs Iran-only QCI:")
print(f"  Pearson r  = {pearson_global_iran:.6f}  (p = {p_pearson_gi:.2e})")
print(f"  Spearman rho = {spearman_global_iran:.6f}  (p = {p_spearman_gi:.2e})")

print(f"\nGlobal QCI vs MENA QCI:")
print(f"  Pearson r  = {pearson_global_mena:.6f}  (p = {p_pearson_gm:.2e})")
print(f"  Spearman rho = {spearman_global_mena:.6f}  (p = {p_spearman_gm:.2e})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. RANK COMPARISON (2021 cross-section)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RANK COMPARISON (2021 snapshot)")
print("=" * 70)

# Get 2021 data
iran_2021_mask = (df_iran['year'] == 2021)
provinces_2021 = df_iran[iran_2021_mask]['iso_location_name'].values

# Get global QCI for 2021
qci_g_2021 = qci_iran_global[iran_2021_mask.values]
qci_l_2021 = qci_iran_local[iran_2021_mask.values]
qci_m_2021 = qci_iran_mena[iran_2021_mask.values]

rank_df = pd.DataFrame({
    'Province': provinces_2021,
    'QCI_Global': qci_g_2021,
    'QCI_Iran_Only': qci_l_2021,
    'QCI_MENA': qci_m_2021,
})

rank_df['Rank_Global'] = rank_df['QCI_Global'].rank(ascending=False).astype(int)
rank_df['Rank_Iran_Only'] = rank_df['QCI_Iran_Only'].rank(ascending=False).astype(int)
rank_df['Rank_MENA'] = rank_df['QCI_MENA'].rank(ascending=False).astype(int)
rank_df['Rank_Diff_Global_vs_Iran'] = (rank_df['Rank_Global'] - rank_df['Rank_Iran_Only']).abs()
rank_df = rank_df.sort_values('Rank_Global')

print("\nProvincial rankings (2021):")
print(rank_df[['Province', 'Rank_Global', 'Rank_Iran_Only', 'Rank_MENA', 'Rank_Diff_Global_vs_Iran']].to_string(index=False))

max_rank_diff = rank_df['Rank_Diff_Global_vs_Iran'].max()
mean_rank_diff = rank_df['Rank_Diff_Global_vs_Iran'].mean()
median_rank_diff = rank_df['Rank_Diff_Global_vs_Iran'].median()
n_same_rank = (rank_df['Rank_Diff_Global_vs_Iran'] == 0).sum()
n_diff_1 = (rank_df['Rank_Diff_Global_vs_Iran'] <= 1).sum()
n_diff_2 = (rank_df['Rank_Diff_Global_vs_Iran'] <= 2).sum()

# Kendall's tau for rank concordance
tau_gi, p_tau_gi = stats.kendalltau(rank_df['Rank_Global'], rank_df['Rank_Iran_Only'])
tau_gm, p_tau_gm = stats.kendalltau(rank_df['Rank_Global'], rank_df['Rank_MENA'])

print(f"\nRank agreement (2021):")
print(f"  Max rank difference: {max_rank_diff}")
print(f"  Mean rank difference: {mean_rank_diff:.2f}")
print(f"  Median rank difference: {median_rank_diff:.1f}")
print(f"  Provinces with identical rank: {n_same_rank}/31")
print(f"  Provinces within 1 rank: {n_diff_1}/31")
print(f"  Provinces within 2 ranks: {n_diff_2}/31")
print(f"  Kendall's tau (Global vs Iran-only): {tau_gi:.4f} (p={p_tau_gi:.2e})")
print(f"  Kendall's tau (Global vs MENA): {tau_gm:.4f} (p={p_tau_gm:.2e})")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TOP/BOTTOM AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TOP/BOTTOM AGREEMENT (2021)")
print("=" * 70)

top5_global = set(rank_df.nsmallest(5, 'Rank_Global')['Province'])
top5_iran = set(rank_df.nsmallest(5, 'Rank_Iran_Only')['Province'])
bottom5_global = set(rank_df.nlargest(5, 'Rank_Global')['Province'])
bottom5_iran = set(rank_df.nlargest(5, 'Rank_Iran_Only')['Province'])

print(f"Top 5 agreement: {len(top5_global & top5_iran)}/5 provinces overlap")
print(f"  Global top 5: {sorted(top5_global)}")
print(f"  Iran-only top 5: {sorted(top5_iran)}")
print(f"Bottom 5 agreement: {len(bottom5_global & bottom5_iran)}/5 provinces overlap")
print(f"  Global bottom 5: {sorted(bottom5_global)}")
print(f"  Iran-only bottom 5: {sorted(bottom5_iran)}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE SUPPLEMENTARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Saving supplementary table ===")

supp_table = rank_df[['Province', 'QCI_Global', 'QCI_Iran_Only', 'QCI_MENA',
                       'Rank_Global', 'Rank_Iran_Only', 'Rank_MENA',
                       'Rank_Diff_Global_vs_Iran']].copy()
supp_table.to_csv(os.path.join(TABLE_DIR, 'table_supp_pca_sensitivity.csv'),
                  index=False, float_format='%.2f')
print(f"  Saved: {TABLE_DIR}/table_supp_pca_sensitivity.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SUPPLEMENTARY FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Creating supplementary figure ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 11))

# Panel A: Loading comparison bar chart
ax = axes[0, 0]
x = np.arange(len(FEATURES))
width = 0.25
ax.bar(x - width, np.abs(loadings_global), width, label='Global', color='#2196F3', alpha=0.8)
ax.bar(x, np.abs(loadings_iran), width, label='Iran-only', color='#FF9800', alpha=0.8)
ax.bar(x + width, np.abs(loadings_mena), width, label='MENA', color='#4CAF50', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES)
ax.set_ylabel('|Loading| on PC1')
ax.set_title('(A) PC1 Loadings by PCA Scope', fontweight='bold')
ax.legend(fontsize=8)
# Annotate variance explained
ax.text(0.02, 0.98, f'Var. explained:\n  Global: {var_explained_global:.1f}%\n'
        f'  Iran: {var_explained_iran:.1f}%\n  MENA: {var_explained_mena:.1f}%',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Global QCI vs Iran-only QCI scatter (all province-years)
ax = axes[0, 1]
ax.scatter(qci_iran_global, qci_iran_local, alpha=0.3, s=10, c='steelblue')
# Fit line
slope_fit, intercept_fit, r_fit, _, _ = stats.linregress(qci_iran_global, qci_iran_local)
x_line = np.linspace(qci_iran_global.min(), qci_iran_global.max(), 100)
ax.plot(x_line, slope_fit * x_line + intercept_fit, 'r-', linewidth=1.5,
        label=f'Fit: r={r_fit:.4f}')
ax.set_xlabel('QCI (Global PCA)')
ax.set_ylabel('QCI (Iran-only PCA)')
ax.set_title('(B) Score Agreement (All Province-Years)', fontweight='bold')
ax.legend(fontsize=8)
ax.text(0.02, 0.88, f'Spearman rho = {spearman_global_iran:.4f}\n'
        f'Pearson r = {pearson_global_iran:.4f}',
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Panel C: 2021 rank comparison
ax = axes[1, 0]
ax.scatter(rank_df['Rank_Global'], rank_df['Rank_Iran_Only'], c='steelblue', s=40, zorder=5)
ax.plot([0.5, 31.5], [0.5, 31.5], 'k--', alpha=0.5, label='Perfect agreement')
ax.set_xlabel('Rank (Global PCA)')
ax.set_ylabel('Rank (Iran-only PCA)')
ax.set_title(f'(C) Provincial Rank Agreement, 2021\n'
             f'(Kendall tau={tau_gi:.3f})', fontweight='bold')
ax.set_xlim(0.5, 31.5)
ax.set_ylim(0.5, 31.5)
ax.legend(fontsize=8, loc='lower right')
# Label provinces with largest rank shifts
for _, row in rank_df.nlargest(3, 'Rank_Diff_Global_vs_Iran').iterrows():
    ax.annotate(row['Province'], (row['Rank_Global'], row['Rank_Iran_Only']),
                fontsize=6, xytext=(4, 4), textcoords='offset points')

# Panel D: Squared loading contributions comparison (pie-like grouped bar)
ax = axes[1, 1]
sq_global = (loadings_global**2 / (loadings_global**2).sum()) * 100
sq_iran = (loadings_iran**2 / (loadings_iran**2).sum()) * 100
sq_mena = (loadings_mena**2 / (loadings_mena**2).sum()) * 100

x = np.arange(len(FEATURES))
ax.bar(x - width, sq_global, width, label='Global', color='#2196F3', alpha=0.8)
ax.bar(x, sq_iran, width, label='Iran-only', color='#FF9800', alpha=0.8)
ax.bar(x + width, sq_mena, width, label='MENA', color='#4CAF50', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES)
ax.set_ylabel('Contribution to PC1 (%)')
ax.set_title('(D) Component Contributions by PCA Scope', fontweight='bold')
ax.legend(fontsize=8)
# Annotate values
for i in range(3):
    ax.text(i - width, sq_global[i] + 0.5, f'{sq_global[i]:.1f}%', ha='center', fontsize=7)
    ax.text(i, sq_iran[i] + 0.5, f'{sq_iran[i]:.1f}%', ha='center', fontsize=7)
    ax.text(i + width, sq_mena[i] + 0.5, f'{sq_mena[i]:.1f}%', ha='center', fontsize=7)

fig.suptitle('Supplementary Figure: PCA Sensitivity Analysis\n'
             'Global vs Iran-only vs MENA PCA Parameterisation',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figure_supp_pca_sensitivity.pdf'), format='pdf')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure_supp_pca_sensitivity.png'), format='png')
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figure_supp_pca_sensitivity.pdf/png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. SAVE ALL RESULTS AS JSON
# ══════════════════════════════════════════════════════════════════════════════
results = {
    'description': 'PCA sensitivity analysis: Global vs Iran-only vs MENA parameterisation',
    'global_pca': {
        'n_observations': int(len(df_global_train)),
        'n_locations': int(df_global_train['iso_location_name'].nunique()),
        'variance_explained_pct': round(var_explained_global, 2),
        'loadings': {f: round(float(l), 4) for f, l in zip(FEATURES, loadings_global)},
        'squared_contributions_pct': {f: round(float(s), 2) for f, s in zip(FEATURES, sq_global)},
    },
    'iran_only_pca': {
        'n_observations': int(len(df_iran)),
        'n_provinces': int(df_iran['iso_location_name'].nunique()),
        'variance_explained_pct': round(var_explained_iran, 2),
        'loadings': {f: round(float(l), 4) for f, l in zip(FEATURES, loadings_iran)},
        'squared_contributions_pct': {f: round(float(s), 2) for f, s in zip(FEATURES, sq_iran)},
    },
    'mena_pca': {
        'n_observations': int(len(df_mena)),
        'n_locations': int(df_mena['iso_location_name'].nunique()),
        'variance_explained_pct': round(var_explained_mena, 2),
        'loadings': {f: round(float(l), 4) for f, l in zip(FEATURES, loadings_mena)},
        'squared_contributions_pct': {f: round(float(s), 2) for f, s in zip(FEATURES, sq_mena)},
    },
    'score_correlations_all_province_years': {
        'global_vs_iran_only': {
            'pearson_r': round(float(pearson_global_iran), 6),
            'spearman_rho': round(float(spearman_global_iran), 6),
        },
        'global_vs_mena': {
            'pearson_r': round(float(pearson_global_mena), 6),
            'spearman_rho': round(float(spearman_global_mena), 6),
        },
    },
    'rank_agreement_2021': {
        'kendall_tau_global_vs_iran': round(float(tau_gi), 4),
        'kendall_tau_global_vs_mena': round(float(tau_gm), 4),
        'max_rank_difference': int(max_rank_diff),
        'mean_rank_difference': round(float(mean_rank_diff), 2),
        'median_rank_difference': round(float(median_rank_diff), 1),
        'provinces_identical_rank': int(n_same_rank),
        'provinces_within_1_rank': int(n_diff_1),
        'provinces_within_2_ranks': int(n_diff_2),
        'top5_overlap': int(len(top5_global & top5_iran)),
        'bottom5_overlap': int(len(bottom5_global & bottom5_iran)),
    },
    'cosine_similarity': {
        'global_vs_iran': round(float(cos_global_iran), 6),
        'global_vs_mena': round(float(cos_global_mena), 6),
        'iran_vs_mena': round(float(cos_iran_mena), 6),
    },
}

with open(STATS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {STATS_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
PCA Sensitivity Analysis Complete
---------------------------------
Variance Explained:
  Global: {var_explained_global:.2f}%  |  Iran-only: {var_explained_iran:.2f}%  |  MENA: {var_explained_mena:.2f}%

Loading Cosine Similarity:
  Global vs Iran-only: {cos_global_iran:.6f}
  Global vs MENA:      {cos_global_mena:.6f}

Score Correlation (all province-years):
  Pearson r:    {pearson_global_iran:.6f}
  Spearman rho: {spearman_global_iran:.6f}

Rank Agreement (2021):
  Kendall's tau: {tau_gi:.4f}
  Mean rank diff: {mean_rank_diff:.2f} positions
  Top-5 overlap: {len(top5_global & top5_iran)}/5
  Bottom-5 overlap: {len(bottom5_global & bottom5_iran)}/5

CONCLUSION: {'Global PCA parameters are appropriate for Iran subnational analysis.' if spearman_global_iran > 0.99 else 'Results warrant further investigation.'}
""")
