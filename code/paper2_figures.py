"""
Paper 2: Publication-Quality Figures for Inequality Analysis
"""

import os
import json
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

BASE = "/Users/mehranmamandipoor/Desktop/thesis"
RESULTS = os.path.join(BASE, "results")
P2 = os.path.join(RESULTS, "paper2/analysis")
FIG_DIR = os.path.join(RESULTS, "paper2/figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "code"))
from mappings import SDI_COUNTRY_MAPPING, SDI_VALUE_MAP_2021

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "Low": "#d73027",
    "Low-middle": "#fc8d59",
    "Middle": "#fee08b",
    "High-middle": "#91bfdb",
    "High": "#4575b4",
}

REGION_COLORS = {
    "Sub-Saharan Africa": "#d73027",
    "South Asia": "#fc8d59",
    "Middle East & North Africa": "#fee08b",
    "Latin America & Caribbean": "#91cf60",
    "East Asia & Pacific": "#1a9850",
    "Europe & Central Asia": "#4575b4",
    "North America": "#762a83",
}

# Load results
ci_df = pd.read_csv(os.path.join(P2, "concentration_index_by_year.csv"))
sii_df = pd.read_csv(os.path.join(P2, "sii_rii_by_year.csv"))
disp_df = pd.read_csv(os.path.join(P2, "dispersion_indices_by_year.csv"))
gdr_trend = pd.read_csv(os.path.join(P2, "gdr_trend.csv"))
gdr_all = pd.read_csv(os.path.join(P2, "gdr_country_year.csv"))
conv_data = pd.read_csv(os.path.join(P2, "convergence_data.csv"))
age_ineq = pd.read_csv(os.path.join(P2, "age_inequality_by_country.csv"))

with open(os.path.join(P2, "paper2_summary_stats.json")) as f:
    summary = json.load(f)

# ────────────────────────────────────────────────────────────────────
# FIGURE 1: Concentration Index over time
# ────────────────────────────────────────────────────────────────────
print("Figure 1: Concentration Index over time...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ci_df["year"], ci_df["CI"], color="#2166ac", linewidth=2, marker="o", markersize=3)
ax.fill_between(ci_df["year"], ci_df["CI_lower"], ci_df["CI_upper"], alpha=0.2, color="#2166ac")
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("Concentration Index (CI)")
ax.set_title("Concentration Index of DS-TB QCI by SDI, 1990-2021")
ax.set_xlim(1990, 2021)

# Add annotation
ax.annotate(f"CI 1990: {ci_df.iloc[0]['CI']:.4f}\nCI 2021: {ci_df.iloc[-1]['CI']:.4f}",
            xy=(2015, ci_df.iloc[-1]["CI"]),
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_concentration_index.png"))
fig.savefig(os.path.join(FIG_DIR, "fig1_concentration_index.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 2: Multi-panel inequality trends (Gini, Theil, SII, CI)
# ────────────────────────────────────────────────────────────────────
print("Figure 2: Multi-panel inequality trends...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Panel A: Gini
ax = axes[0, 0]
ax.plot(disp_df["year"], disp_df["Gini"], color="#d73027", linewidth=2, marker="o", markersize=3)
ax.set_ylabel("Gini Coefficient")
ax.set_title("A. Gini Coefficient")
ax.set_xlim(1990, 2021)

# Panel B: Theil
ax = axes[0, 1]
ax.plot(disp_df["year"], disp_df["Theil_T"], color="#fc8d59", linewidth=2, marker="o", markersize=3, label="Theil T (GE1)")
ax.plot(disp_df["year"], disp_df["Theil_L"], color="#91bfdb", linewidth=2, marker="s", markersize=3, label="Theil L (GE0)")
ax.set_ylabel("Index Value")
ax.set_title("B. Generalised Entropy Indices")
ax.legend()
ax.set_xlim(1990, 2021)

# Panel C: SII
ax = axes[1, 0]
ax.plot(sii_df["year"], sii_df["SII"], color="#4575b4", linewidth=2, marker="o", markersize=3)
ax.fill_between(sii_df["year"], sii_df["SII_lower"], sii_df["SII_upper"], alpha=0.2, color="#4575b4")
ax.set_xlabel("Year")
ax.set_ylabel("SII (QCI points)")
ax.set_title("C. Slope Index of Inequality")
ax.set_xlim(1990, 2021)

# Panel D: RII
ax = axes[1, 1]
ax.plot(sii_df["year"], sii_df["RII"], color="#1a9850", linewidth=2, marker="o", markersize=3)
ax.fill_between(sii_df["year"], sii_df["RII_lower"], sii_df["RII_upper"], alpha=0.2, color="#1a9850")
ax.set_xlabel("Year")
ax.set_ylabel("RII (ratio)")
ax.set_title("D. Relative Index of Inequality")
ax.set_xlim(1990, 2021)

fig.suptitle("Trends in QCI Inequality Measures, 1990-2021", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_inequality_trends.png"))
fig.savefig(os.path.join(FIG_DIR, "fig2_inequality_trends.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 3: Concentration Curve (2021)
# ────────────────────────────────────────────────────────────────────
print("Figure 3: Concentration curve...")
qci_simple = pd.read_csv(os.path.join(RESULTS, "shared/qci.csv"))

# Build WB_REGION_MAPPING (same as in analysis)
# (reusing the same mapping from the analysis script)
from paper2_analysis import WB_REGION_MAPPING, exclude_set

df_cc = qci_simple[~qci_simple["iso_location_name"].isin(exclude_set)].copy()
df_cc["sdi_value"] = df_cc["iso_location_name"].map(SDI_VALUE_MAP_2021)
df_cc["sdi_group"] = df_cc["iso_location_name"].map(SDI_COUNTRY_MAPPING)

# 2021, age-std, both sexes
cc_data = df_cc[(df_cc["year"] == 2021) & (df_cc["age_name"] == "Age-standardized") &
                (df_cc["sex_name"] == "Both")].dropna(subset=["sdi_value", "qci"]).copy()
cc_data = cc_data.sort_values("sdi_value")
n = len(cc_data)
cc_data["cum_pop"] = np.arange(1, n + 1) / n
cc_data["cum_qci"] = cc_data["qci"].cumsum() / cc_data["qci"].sum()

# Also for 1990
cc_1990 = df_cc[(df_cc["year"] == 1990) & (df_cc["age_name"] == "Age-standardized") &
                (df_cc["sex_name"] == "Both")].dropna(subset=["sdi_value", "qci"]).copy()
cc_1990 = cc_1990.sort_values("sdi_value")
n_1990 = len(cc_1990)
cc_1990["cum_pop"] = np.arange(1, n_1990 + 1) / n_1990
cc_1990["cum_qci"] = cc_1990["qci"].cumsum() / cc_1990["qci"].sum()

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Line of equality")
ax.plot(cc_1990["cum_pop"], cc_1990["cum_qci"], color="#fc8d59", linewidth=2, label="1990")
ax.plot(cc_data["cum_pop"], cc_data["cum_qci"], color="#4575b4", linewidth=2, label="2021")
ax.fill_between(cc_data["cum_pop"], cc_data["cum_qci"], cc_data["cum_pop"],
                alpha=0.1, color="#4575b4")
ax.set_xlabel("Cumulative proportion of countries (ranked by SDI)")
ax.set_ylabel("Cumulative proportion of QCI")
ax.set_title("Concentration Curve: DS-TB QCI by SDI")
ax.legend(loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_concentration_curve.png"))
fig.savefig(os.path.join(FIG_DIR, "fig3_concentration_curve.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 4: Sigma and Beta Convergence
# ────────────────────────────────────────────────────────────────────
print("Figure 4: Convergence analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Sigma-convergence
ax1.plot(disp_df["year"], disp_df["SD"], color="#2166ac", linewidth=2, marker="o", markersize=4)
sigma_fit = np.polyfit(disp_df["year"], disp_df["SD"], 1)
ax1.plot(disp_df["year"], np.polyval(sigma_fit, disp_df["year"]), "r--", linewidth=1.5, alpha=0.7)
ax1.set_xlabel("Year")
ax1.set_ylabel("Standard deviation of QCI")
ax1.set_title("A. Sigma-convergence")
ax1.annotate(f"Slope: {sigma_fit[0]:.4f}/year\np < 0.001",
             xy=(2005, disp_df["SD"].median()), fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))
ax1.set_xlim(1990, 2021)

# Panel B: Beta-convergence
for sdi_grp, color in COLORS.items():
    mask = conv_data["sdi_group"] == sdi_grp
    ax2.scatter(conv_data.loc[mask, "qci_1990"], conv_data.loc[mask, "change"],
                c=color, s=25, alpha=0.8, label=sdi_grp, edgecolors="white", linewidths=0.3)

# Add fit line
valid = conv_data.dropna(subset=["qci_1990", "change"])
slope, intercept = np.polyfit(valid["qci_1990"], valid["change"], 1)
x_range = np.linspace(valid["qci_1990"].min(), valid["qci_1990"].max(), 100)
ax2.plot(x_range, slope * x_range + intercept, "k-", linewidth=2, alpha=0.7)
ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax2.set_xlabel("QCI in 1990 (baseline)")
ax2.set_ylabel("QCI change, 1990-2021")
ax2.set_title("B. Beta-convergence")
ax2.legend(title="SDI group", loc="upper right", fontsize=8)
r2 = summary["beta_convergence_r2"]
ax2.annotate(f"Beta = {slope:.3f}\nR² = {r2:.3f}\np < 0.001",
             xy=(77, 10), fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

fig.suptitle("Convergence in DS-TB Quality of Care, 1990-2021", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_convergence.png"))
fig.savefig(os.path.join(FIG_DIR, "fig4_convergence.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 5: GDR Analysis
# ────────────────────────────────────────────────────────────────────
print("Figure 5: Gender disparity ratio...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: GDR trend over time (mean ± SD)
ax = axes[0]
ax.plot(gdr_trend["year"], gdr_trend["mean_GDR"], color="#8e24aa", linewidth=2, marker="o", markersize=3)
ax.fill_between(gdr_trend["year"],
                gdr_trend["mean_GDR"] - gdr_trend["sd_GDR"],
                gdr_trend["mean_GDR"] + gdr_trend["sd_GDR"],
                alpha=0.15, color="#8e24aa")
ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("Gender Disparity Ratio (F/M)")
ax.set_title("A. GDR Trend Over Time")
ax.set_xlim(1990, 2021)

# Panel B: GDR by WB region (2021, boxplot)
ax = axes[1]
gdr_2021 = gdr_all[gdr_all["year"] == 2021].copy()
region_order = gdr_2021.groupby("wb_region")["GDR"].median().sort_values(ascending=False).index.tolist()
box_data = [gdr_2021[gdr_2021["wb_region"] == r]["GDR"].values for r in region_order]
bp = ax.boxplot(box_data, labels=[r[:20] for r in region_order], vert=True, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, region in zip(bp["boxes"], region_order):
    patch.set_facecolor(REGION_COLORS.get(region, "#cccccc"))
    patch.set_alpha(0.7)
ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
ax.set_ylabel("GDR (F/M)")
ax.set_title("B. GDR by Region (2021)")
ax.tick_params(axis="x", rotation=45)

# Panel C: GDR vs SDI (2021)
ax = axes[2]
for sdi_grp, color in COLORS.items():
    mask = gdr_2021["sdi_group"] == sdi_grp
    ax.scatter(gdr_2021.loc[mask, "sdi_value"], gdr_2021.loc[mask, "GDR"],
               c=color, s=25, alpha=0.8, label=sdi_grp, edgecolors="white", linewidths=0.3)
ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
# Fit line
valid_gdr = gdr_2021.dropna(subset=["sdi_value", "GDR"])
slope_gdr, intercept_gdr = np.polyfit(valid_gdr["sdi_value"], valid_gdr["GDR"], 1)
x_gdr = np.linspace(0.38, 0.92, 100)
ax.plot(x_gdr, slope_gdr * x_gdr + intercept_gdr, "k-", linewidth=1.5, alpha=0.7)
rho_gdr, p_gdr = stats.spearmanr(valid_gdr["sdi_value"], valid_gdr["GDR"])
ax.set_xlabel("SDI (2021)")
ax.set_ylabel("GDR (F/M)")
ax.set_title("C. GDR vs SDI (2021)")
ax.legend(title="SDI group", fontsize=7, loc="upper right")
ax.annotate(f"rho = {rho_gdr:.3f}\np < 0.001",
            xy=(0.42, 0.96), fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

fig.suptitle("Gender Disparity in DS-TB Quality of Care", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_gender_disparity.png"))
fig.savefig(os.path.join(FIG_DIR, "fig5_gender_disparity.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 6: Age-Group Inequality
# ────────────────────────────────────────────────────────────────────
print("Figure 6: Age-group inequality...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: CI by age group (2021) bar chart
ci_age = pd.read_csv(os.path.join(P2, "ci_by_age_group_2021.csv"))
colors_age = ["#d73027", "#fee08b", "#91cf60", "#4575b4", "#762a83"]
ax1.barh(ci_age["age_group"], ci_age["CI"], xerr=1.96*ci_age["CI_se"],
         color=colors_age, edgecolor="white", height=0.6, capsize=4)
ax1.set_xlabel("Concentration Index (CI)")
ax1.set_title("A. CI of QCI by Age Group (2021)")
ax1.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
ax1.invert_yaxis()

# Panel B: Age range (max-min QCI across age groups) by SDI group over time
ax2_data = age_ineq.groupby(["sdi_group", "year"])["age_range"].mean().reset_index()
for sdi_grp, color in COLORS.items():
    mask = ax2_data["sdi_group"] == sdi_grp
    if mask.sum() > 0:
        ax2.plot(ax2_data.loc[mask, "year"], ax2_data.loc[mask, "age_range"],
                 color=color, linewidth=2, label=sdi_grp)
ax2.set_xlabel("Year")
ax2.set_ylabel("Mean age-group QCI range (max-min)")
ax2.set_title("B. Age-Group QCI Disparity by SDI Group")
ax2.legend(title="SDI group", fontsize=8)
ax2.set_xlim(1990, 2021)

fig.suptitle("Age-Related Inequality in DS-TB Quality of Care", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_age_inequality.png"))
fig.savefig(os.path.join(FIG_DIR, "fig6_age_inequality.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 7: Wagstaff Decomposition
# ────────────────────────────────────────────────────────────────────
print("Figure 7: Wagstaff decomposition...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for ax, year in zip([ax1, ax2], [1990, 2021]):
    decomp = pd.read_csv(os.path.join(P2, f"wagstaff_decomposition_{year}.csv"))
    # Filter to main components only
    main_comps = decomp[decomp["determinant"].isin(["MIR", "YLLtoYLD", "DALtoPER", "SDI", "Residual"])]
    main_comps = main_comps[main_comps["contribution"].abs() > 1e-8]

    colors_decomp = {"MIR": "#d73027", "YLLtoYLD": "#4575b4", "DALtoPER": "#1a9850",
                     "SDI": "#762a83", "Residual": "#999999"}

    bars = ax.barh(main_comps["determinant"], main_comps["contribution"],
                   color=[colors_decomp.get(d, "#999999") for d in main_comps["determinant"]],
                   edgecolor="white", height=0.6)

    # Add percentage labels
    for bar, pct in zip(bars, main_comps["pct_of_total"]):
        width = bar.get_width()
        ax.text(width + 0.0002, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    ax.set_xlabel("Contribution to CI")
    ax.set_title(f"{'A' if year == 1990 else 'B'}. Decomposition ({year})")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.invert_yaxis()

fig.suptitle("Wagstaff Decomposition of QCI Concentration Index",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig7_decomposition.png"))
fig.savefig(os.path.join(FIG_DIR, "fig7_decomposition.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 8: Theil Decomposition (Between vs Within)
# ────────────────────────────────────────────────────────────────────
print("Figure 8: Theil between/within decomposition...")
theil_bw = pd.read_csv(os.path.join(P2, "theil_decomposition_between_within.csv"))

fig, ax = plt.subplots(figsize=(8, 5))
width = 0.6
x = np.arange(len(theil_bw))
ax.bar(x, theil_bw["Between"], width, label="Between-region", color="#4575b4")
ax.bar(x, theil_bw["Within"], width, bottom=theil_bw["Between"],
       label="Within-region", color="#fc8d59")
ax.set_xticks(x)
ax.set_xticklabels(theil_bw["year"].astype(int))
ax.set_xlabel("Year")
ax.set_ylabel("Theil Index")
ax.set_title("Theil Index Decomposition: Between vs Within Region")
ax.legend()

# Add percentage labels
for i, row in theil_bw.iterrows():
    ax.text(i, row["Total_Theil"] + 0.00002, f'{row["Pct_between"]:.1f}% B',
            ha='center', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_theil_decomposition.png"))
fig.savefig(os.path.join(FIG_DIR, "fig8_theil_decomposition.pdf"))
plt.close(fig)

# ────────────────────────────────────────────────────────────────────
# FIGURE 9: Summary Dashboard (Graphical Abstract)
# ────────────────────────────────────────────────────────────────────
print("Figure 9: Summary dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# A: CI trend
ax = fig.add_subplot(gs[0, 0])
ax.plot(ci_df["year"], ci_df["CI"], color="#2166ac", linewidth=2)
ax.fill_between(ci_df["year"], ci_df["CI_lower"], ci_df["CI_upper"], alpha=0.2, color="#2166ac")
ax.set_title("A. Concentration Index", fontweight="bold")
ax.set_ylabel("CI")
ax.set_xlim(1990, 2021)

# B: Gini
ax = fig.add_subplot(gs[0, 1])
ax.plot(disp_df["year"], disp_df["Gini"], color="#d73027", linewidth=2)
ax.set_title("B. Gini Coefficient", fontweight="bold")
ax.set_ylabel("Gini")
ax.set_xlim(1990, 2021)

# C: GDR trend
ax = fig.add_subplot(gs[0, 2])
ax.plot(gdr_trend["year"], gdr_trend["mean_GDR"], color="#8e24aa", linewidth=2)
ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
ax.set_title("C. Gender Disparity Ratio", fontweight="bold")
ax.set_ylabel("GDR (F/M)")
ax.set_xlim(1990, 2021)

# D: Beta-convergence
ax = fig.add_subplot(gs[1, 0])
for sdi_grp, color in COLORS.items():
    mask = conv_data["sdi_group"] == sdi_grp
    ax.scatter(conv_data.loc[mask, "qci_1990"], conv_data.loc[mask, "change"],
               c=color, s=15, alpha=0.7, edgecolors="none")
valid = conv_data.dropna(subset=["qci_1990", "change"])
sl, ic = np.polyfit(valid["qci_1990"], valid["change"], 1)
xr = np.linspace(valid["qci_1990"].min(), valid["qci_1990"].max(), 100)
ax.plot(xr, sl * xr + ic, "k-", linewidth=1.5)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax.set_title("D. Beta-Convergence", fontweight="bold")
ax.set_xlabel("QCI 1990")
ax.set_ylabel("Change")

# E: Age-group CI
ax = fig.add_subplot(gs[1, 1])
colors_age = ["#d73027", "#fee08b", "#91cf60", "#4575b4", "#762a83"]
ax.barh(ci_age["age_group"], ci_age["CI"], color=colors_age, edgecolor="white", height=0.6)
ax.set_title("E. CI by Age Group", fontweight="bold")
ax.set_xlabel("CI")
ax.invert_yaxis()

# F: Sigma-convergence
ax = fig.add_subplot(gs[1, 2])
ax.plot(disp_df["year"], disp_df["SD"], color="#2166ac", linewidth=2)
ax.set_title("F. Sigma-Convergence (SD)", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("SD")
ax.set_xlim(1990, 2021)

fig.suptitle("Inequality in DS-TB Quality of Care: Summary of Key Findings, 1990-2021",
             fontsize=16, fontweight="bold", y=1.01)
fig.savefig(os.path.join(FIG_DIR, "fig9_summary_dashboard.png"))
fig.savefig(os.path.join(FIG_DIR, "fig9_summary_dashboard.pdf"))
plt.close(fig)

print(f"\nAll figures saved to: {FIG_DIR}")
print(f"Files: {sorted(os.listdir(FIG_DIR))}")
