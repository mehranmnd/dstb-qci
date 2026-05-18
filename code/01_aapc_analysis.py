#!/usr/bin/env python3
"""
Average Annual Percent Change (AAPC) Analysis for QCI Trends (1990-2021)

Computes AAPC via log-linear regression: ln(QCI) = beta_0 + beta_1*year
AAPC = (exp(beta_1) - 1) * 100;  95% CI from standard error of beta_1.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCI_PATH = os.path.join(BASE, 'results/shared/qci.csv')
QCI_COMPLETE_PATH = os.path.join(BASE, 'results/shared/qci_complete_data.csv')
OUTPUT_PATH = os.path.join(BASE, 'results/shared/aapc_results.txt')

print("Loading data...")
df = pd.read_csv(QCI_PATH)
df_complete = pd.read_csv(QCI_COMPLETE_PATH)

mask = (df['age_name'] == 'Age-standardized') & (df['sex_name'] == 'Both')
data = df.loc[mask, ['iso_location_name', 'year', 'qci']].copy()
data = data.sort_values(['iso_location_name', 'year']).reset_index(drop=True)
print(f"Filtered data: {len(data)} rows, {data['iso_location_name'].nunique()} locations")

countries = sorted(df_complete[df_complete['sdi_group'].notna()]['iso_location_name'].unique())
print(f"Identified {len(countries)} countries from sdi_group")

WB_REGIONS = [
    'Europe & Central Asia - WB', 'Middle East & North Africa - WB',
    'Sub-Saharan Africa - WB', 'East Asia & Pacific - WB',
    'South Asia - WB', 'Latin America & Caribbean - WB', 'North America',
]
SDI_QUINTILES = ['High SDI', 'High-middle SDI', 'Middle SDI', 'Low-middle SDI', 'Low SDI']


def compute_aapc(subset):
    """Fit ln(QCI) = b0 + b1*year. Return AAPC and 95% CI."""
    subset = subset.dropna(subset=['qci'])
    if len(subset) < 3:
        return np.nan, np.nan, np.nan
    y = np.log(subset['qci'].values)
    x = subset['year'].values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    aapc = (np.exp(slope) - 1) * 100
    n = len(x)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_lower = (np.exp(slope - t_crit * std_err) - 1) * 100
    ci_upper = (np.exp(slope + t_crit * std_err) - 1) * 100
    return aapc, ci_lower, ci_upper


def get_qci_value(loc_data, year):
    row = loc_data[loc_data['year'] == year]
    if len(row) == 1:
        return row['qci'].values[0]
    return np.nan


def analyze_location(loc_name, loc_data):
    full = loc_data[(loc_data['year'] >= 1990) & (loc_data['year'] <= 2021)]
    recent = loc_data[(loc_data['year'] >= 2010) & (loc_data['year'] <= 2021)]
    aapc_full, ci_lo_full, ci_hi_full = compute_aapc(full)
    aapc_recent, ci_lo_recent, ci_hi_recent = compute_aapc(recent)
    qci_1990 = get_qci_value(loc_data, 1990)
    qci_2021 = get_qci_value(loc_data, 2021)
    return {
        'location': loc_name,
        'aapc_full': aapc_full, 'ci_lo_full': ci_lo_full, 'ci_hi_full': ci_hi_full,
        'aapc_recent': aapc_recent, 'ci_lo_recent': ci_lo_recent, 'ci_hi_recent': ci_hi_recent,
        'qci_1990': qci_1990, 'qci_2021': qci_2021,
    }


# --- Run all analyses ---
results = []

print("Computing AAPC for Global...")
results.append(analyze_location('Global', data[data['iso_location_name'] == 'Global']))

print("Computing AAPC for WB regions...")
for region in WB_REGIONS:
    loc_data = data[data['iso_location_name'] == region]
    if len(loc_data) > 0:
        results.append(analyze_location(region, loc_data))

print("Computing AAPC for SDI quintiles...")
for sdi in SDI_QUINTILES:
    loc_data = data[data['iso_location_name'] == sdi]
    if len(loc_data) > 0:
        results.append(analyze_location(sdi, loc_data))

print("Computing AAPC for Iran...")
results.append(analyze_location('Iran', data[data['iso_location_name'] == 'Iran']))

print(f"Computing AAPC for {len(countries)} countries...")
for country in countries:
    if country == 'Iran':
        continue
    loc_data = data[data['iso_location_name'] == country]
    if len(loc_data) > 0:
        results.append(analyze_location(country, loc_data))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('aapc_full', ascending=False).reset_index(drop=True)


def classify(loc):
    if loc == 'Global':
        return 0, 'GLOBAL'
    elif loc in WB_REGIONS:
        return 1, 'WB REGION'
    elif loc in SDI_QUINTILES:
        return 2, 'SDI QUINTILE'
    elif loc == 'Iran':
        return 3, 'IRAN'
    else:
        return 4, 'COUNTRY'


results_df['sort_key'] = results_df['location'].apply(lambda x: classify(x)[0])
results_df['category'] = results_df['location'].apply(lambda x: classify(x)[1])


def fmt_aapc(val, lo, hi):
    if pd.isna(val):
        return 'N/A', 'N/A'
    return f"{val:+.3f}", f"({lo:+.3f}, {hi:+.3f})"


def fmt_qci(val):
    if pd.isna(val):
        return 'N/A'
    return f"{val:.2f}"


# --- Build output ---
print(f"\nWriting results to {OUTPUT_PATH}...")
lines = []
SEP = "=" * 140
sep = "-" * 140

lines.append(SEP)
lines.append("AVERAGE ANNUAL PERCENT CHANGE (AAPC) IN QCI, 1990-2021")
lines.append("Method: Log-linear regression  ln(QCI) = beta_0 + beta_1*year")
lines.append("AAPC = (exp(beta_1) - 1) x 100;  95% CI from t-distribution on beta_1")
lines.append(SEP)
lines.append("")

n_pos = (results_df['aapc_full'] > 0).sum()
n_neg = (results_df['aapc_full'] < 0).sum()
lines.append(f"Total locations analyzed: {len(results_df)}")
lines.append(f"  Positive AAPC (full period): {n_pos}")
lines.append(f"  Negative AAPC (full period): {n_neg}")
lines.append("")

sections = [
    ('GLOBAL', 'GLOBAL'),
    ('WB REGION', 'WORLD BANK REGIONS (sorted by AAPC, full period)'),
    ('SDI QUINTILE', 'SDI QUINTILES (sorted by AAPC, full period)'),
    ('IRAN', 'IRAN'),
    ('COUNTRY', 'INDIVIDUAL COUNTRIES (sorted by AAPC, full period)'),
]

hdr = "{:<45s} {:>10s}  {:>25s}  {:>10s}  {:>25s}  {:>10s}  {:>10s}".format(
    "Location", "AAPC_full%", "95% CI (full)", "AAPC_rec%", "95% CI (recent)", "QCI_1990", "QCI_2021"
)

for cat_key, cat_title in sections:
    section = results_df[results_df['category'] == cat_key].sort_values('aapc_full', ascending=False)
    if len(section) == 0:
        continue
    lines.append(sep)
    lines.append(f"  {cat_title}")
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)
    for _, row in section.iterrows():
        af, cf = fmt_aapc(row['aapc_full'], row['ci_lo_full'], row['ci_hi_full'])
        ar, cr = fmt_aapc(row['aapc_recent'], row['ci_lo_recent'], row['ci_hi_recent'])
        lines.append("{:<45s} {:>10s}  {:>25s}  {:>10s}  {:>25s}  {:>10s}  {:>10s}".format(
            row['location'][:45], af, cf, ar, cr, fmt_qci(row['qci_1990']), fmt_qci(row['qci_2021'])))
    lines.append("")

# Top/Bottom 10
country_df = results_df[results_df['category'] == 'COUNTRY'].sort_values('aapc_full', ascending=False)

lines.append(sep)
lines.append("  TOP 10 COUNTRIES BY AAPC (FULL PERIOD, HIGHEST IMPROVEMENT)")
lines.append(sep)
lines.append(hdr)
lines.append(sep)
for _, row in country_df.head(10).iterrows():
    af, cf = fmt_aapc(row['aapc_full'], row['ci_lo_full'], row['ci_hi_full'])
    ar, cr = fmt_aapc(row['aapc_recent'], row['ci_lo_recent'], row['ci_hi_recent'])
    lines.append("{:<45s} {:>10s}  {:>25s}  {:>10s}  {:>25s}  {:>10s}  {:>10s}".format(
        row['location'][:45], af, cf, ar, cr, fmt_qci(row['qci_1990']), fmt_qci(row['qci_2021'])))

lines.append("")
lines.append(sep)
lines.append("  BOTTOM 10 COUNTRIES BY AAPC (FULL PERIOD, LARGEST DECLINE)")
lines.append(sep)
lines.append(hdr)
lines.append(sep)
for _, row in country_df.tail(10).iterrows():
    af, cf = fmt_aapc(row['aapc_full'], row['ci_lo_full'], row['ci_hi_full'])
    ar, cr = fmt_aapc(row['aapc_recent'], row['ci_lo_recent'], row['ci_hi_recent'])
    lines.append("{:<45s} {:>10s}  {:>25s}  {:>10s}  {:>25s}  {:>10s}  {:>10s}".format(
        row['location'][:45], af, cf, ar, cr, fmt_qci(row['qci_1990']), fmt_qci(row['qci_2021'])))

lines.append("")

# Iran summary
ir_row = results_df[results_df['location'] == 'Iran']
all_ranked = pd.concat([country_df, ir_row]).drop_duplicates(subset='location').sort_values('aapc_full', ascending=False).reset_index(drop=True)
iran_rank = int(all_ranked[all_ranked['location'] == 'Iran'].index[0]) + 1
total_ranked = len(all_ranked)
ir = ir_row.iloc[0]

lines.append(sep)
lines.append("  IRAN SUMMARY")
lines.append(sep)
lines.append(f"  Iran rank among {total_ranked} locations: {iran_rank}/{total_ranked} (1 = highest AAPC)")
lines.append(f"  AAPC (full period, 1990-2021): {ir['aapc_full']:+.4f}%  95% CI: ({ir['ci_lo_full']:+.4f}%, {ir['ci_hi_full']:+.4f}%)")
lines.append(f"  AAPC (recent, 2010-2021):      {ir['aapc_recent']:+.4f}%  95% CI: ({ir['ci_lo_recent']:+.4f}%, {ir['ci_hi_recent']:+.4f}%)")
lines.append(f"  QCI 1990: {ir['qci_1990']:.4f}")
lines.append(f"  QCI 2021: {ir['qci_2021']:.4f}")
lines.append(f"  Absolute change: {ir['qci_2021'] - ir['qci_1990']:+.4f}")
lines.append(f"  Relative change: {((ir['qci_2021'] - ir['qci_1990']) / ir['qci_1990']) * 100:+.2f}%")
lines.append("")

# Global summary
gl = results_df[results_df['location'] == 'Global'].iloc[0]
lines.append(sep)
lines.append("  GLOBAL SUMMARY")
lines.append(sep)
lines.append(f"  AAPC (full period, 1990-2021): {gl['aapc_full']:+.4f}%  95% CI: ({gl['ci_lo_full']:+.4f}%, {gl['ci_hi_full']:+.4f}%)")
lines.append(f"  AAPC (recent, 2010-2021):      {gl['aapc_recent']:+.4f}%  95% CI: ({gl['ci_lo_recent']:+.4f}%, {gl['ci_hi_recent']:+.4f}%)")
lines.append(f"  QCI 1990: {gl['qci_1990']:.4f}")
lines.append(f"  QCI 2021: {gl['qci_2021']:.4f}")
lines.append(f"  Absolute change: {gl['qci_2021'] - gl['qci_1990']:+.4f}")
lines.append(f"  Relative change: {((gl['qci_2021'] - gl['qci_1990']) / gl['qci_1990']) * 100:+.2f}%")
lines.append("")

# Acceleration/deceleration
lines.append(sep)
lines.append("  ACCELERATION / DECELERATION ANALYSIS")
lines.append("  (Comparing recent 2010-2021 AAPC to full 1990-2021 AAPC)")
lines.append(sep)

cdf = country_df.copy()
cdf['diff'] = cdf['aapc_recent'] - cdf['aapc_full']
accel = cdf[cdf['diff'] > 0].sort_values('diff', ascending=False)
decel = cdf[cdf['diff'] < 0].sort_values('diff', ascending=True)
lines.append(f"  Countries with accelerating improvement (recent > full): {len(accel)}")
lines.append(f"  Countries with decelerating improvement (recent < full): {len(decel)}")
lines.append("")
lines.append("  Top 5 accelerating:")
for _, row in accel.head(5).iterrows():
    lines.append(f"    {row['location']:<35s}  full: {row['aapc_full']:+.3f}%  recent: {row['aapc_recent']:+.3f}%  diff: {row['diff']:+.3f}pp")
lines.append("")
lines.append("  Top 5 decelerating:")
for _, row in decel.head(5).iterrows():
    lines.append(f"    {row['location']:<35s}  full: {row['aapc_full']:+.3f}%  recent: {row['aapc_recent']:+.3f}%  diff: {row['diff']:+.3f}pp")
lines.append("")
lines.append(SEP)
lines.append("End of AAPC analysis.")

output_text = "\n".join(lines)
with open(OUTPUT_PATH, 'w') as f:
    f.write(output_text)
print(output_text)
print(f"\nResults saved to: {OUTPUT_PATH}")

# Save CSV
csv_path = OUTPUT_PATH.replace('.txt', '.csv')
export = results_df[['location', 'category', 'aapc_full', 'ci_lo_full', 'ci_hi_full',
                      'aapc_recent', 'ci_lo_recent', 'ci_hi_recent', 'qci_1990', 'qci_2021']].copy()
export.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")
