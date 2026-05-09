#!/usr/bin/env python3
"""
Generate publication-ready tables for DS-TB QCI Paper 1.
Outputs clean CSV files suitable for direct import into Word/LaTeX.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# Configuration
QCI_PATH = "/Users/mehranmamandipoor/Desktop/thesis/results/shared/qci.csv"
COMPLETE_PATH = "/Users/mehranmamandipoor/Desktop/thesis/results/shared/qci_complete_data.csv"
OUTPUT_DIR = "/Users/mehranmamandipoor/Desktop/thesis/results/paper1/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
SEP = "=" * 80
print(SEP)
print("Loading data...")
print(SEP)

qci_df = pd.read_csv(QCI_PATH)
complete_df = pd.read_csv(COMPLETE_PATH)
print(f"  qci.csv: {len(qci_df):>8,} rows")
print(f"  qci_complete_data.csv: {len(complete_df):>8,} rows")
print()


def print_table(df, title):
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)
    print(df.to_string(index=False))
    print()


def save_table(df, filename, title):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print_table(df, title)
    print(f"  --> Saved to: {path}")
    print()


# Base filters
mask_std_both = (qci_df["age_name"] == "Age-standardized") & (qci_df["sex_name"] == "Both")
qci_std = qci_df[mask_std_both].copy()
mask_std_both_c = (complete_df["age_name"] == "Age-standardized") & (complete_df["sex_name"] == "Both")
comp_std = complete_df[mask_std_both_c].copy()


# ================================================================
# TABLE 1: Global and Regional QCI Summary
# ================================================================
print()
print("#" * 80)
print("# TABLE 1: Global and Regional QCI Summary")
print("#" * 80)

table1_locations = [
    "Global",
    "Europe & Central Asia - WB",
    "Middle East & North Africa - WB",
    "Sub-Saharan Africa - WB",
    "East Asia & Pacific - WB",
    "South Asia - WB",
    "Latin America & Caribbean - WB",
    "North America",
    "High SDI",
    "High-middle SDI",
    "Middle SDI",
    "Low-middle SDI",
    "Low SDI",
]

display_names = {
    "Global": "Global",
    "Europe & Central Asia - WB": "Europe & Central Asia",
    "Middle East & North Africa - WB": "Middle East & North Africa",
    "Sub-Saharan Africa - WB": "Sub-Saharan Africa",
    "East Asia & Pacific - WB": "East Asia & Pacific",
    "South Asia - WB": "South Asia",
    "Latin America & Caribbean - WB": "Latin America & Caribbean",
    "North America": "North America",
    "High SDI": "High SDI",
    "High-middle SDI": "High-middle SDI",
    "Middle SDI": "Middle SDI",
    "Low-middle SDI": "Low-middle SDI",
    "Low SDI": "Low SDI",
}

rows_t1 = []
for loc in table1_locations:
    row = {"Location": display_names[loc]}
    for yr in [1990, 2000, 2010, 2021]:
        val = qci_std[(qci_std["location_name"] == loc) & (qci_std["year"] == yr)]["qci"]
        row[f"QCI_{yr}"] = round(val.values[0], 2) if len(val) > 0 else np.nan
    if not pd.isna(row.get("QCI_1990")) and not pd.isna(row.get("QCI_2021")):
        row["Change_1990_2021"] = round(row["QCI_2021"] - row["QCI_1990"], 2)
    else:
        row["Change_1990_2021"] = np.nan
    comp_row = comp_std[(comp_std["location_name"] == loc) & (comp_std["year"] == 2021)]
    if len(comp_row) > 0:
        cr = comp_row.iloc[0]
        row["MIR_2021"] = round(cr["MIR"], 4)
        row["YLLtoYLD_2021"] = round(cr["YLLtoYLD"], 2)
        row["DALtoPER_2021"] = round(cr["DALtoPER"], 2)
    else:
        row["MIR_2021"] = np.nan
        row["YLLtoYLD_2021"] = np.nan
        row["DALtoPER_2021"] = np.nan
    rows_t1.append(row)

table1 = pd.DataFrame(rows_t1)
save_table(table1, "table1_global_regional_qci_summary.csv",
           "Table 1: Global and Regional QCI Summary (Age-standardized, Both sexes)")


# ================================================================
# TABLE 2: Top 20 and Bottom 20 Countries by QCI in 2021
# ================================================================
print()
print("#" * 80)
print("# TABLE 2: Top 20 and Bottom 20 Countries by QCI in 2021")
print("#" * 80)

countries_2021 = comp_std[
    (comp_std["year"] == 2021) & (comp_std["sdi_group"].notna())
].copy()

countries_1990 = qci_std[
    (qci_std["year"] == 1990)
][["location_name", "qci"]].rename(columns={"qci": "QCI_1990"})

countries_2021 = countries_2021.merge(countries_1990, on="location_name", how="left")

countries_2021["Country"] = countries_2021["haq_location_name"]
countries_2021["QCI_2021"] = countries_2021["pca_score"].round(2)
countries_2021["QCI_1990"] = countries_2021["QCI_1990"].round(2)
countries_2021["Change"] = (countries_2021["QCI_2021"] - countries_2021["QCI_1990"]).round(2)
countries_2021["MIR_2021"] = countries_2021["MIR"].round(4)
countries_2021["YLLtoYLD_2021"] = countries_2021["YLLtoYLD"].round(2)
countries_2021["DALtoPER_2021"] = countries_2021["DALtoPER"].round(2)
countries_2021["SDI_Group"] = countries_2021["sdi_group"]

cols_out = ["Rank", "Country", "QCI_2021", "QCI_1990", "Change", "SDI_Group",
            "MIR_2021", "YLLtoYLD_2021", "DALtoPER_2021"]

top20 = countries_2021.nlargest(20, "QCI_2021").reset_index(drop=True)
top20["Rank"] = range(1, 21)
top20_out = top20[cols_out].copy()

bot20 = countries_2021.nsmallest(20, "QCI_2021").reset_index(drop=True)
bot20["Rank"] = range(1, 21)
bot20_out = bot20[cols_out].copy()

save_table(top20_out, "table2a_top20_countries_qci_2021.csv",
           "Table 2A: Top 20 Countries by QCI in 2021 (Age-standardized, Both sexes)")

save_table(bot20_out, "table2b_bottom20_countries_qci_2021.csv",
           "Table 2B: Bottom 20 Countries by QCI in 2021 (Age-standardized, Both sexes)")

combined = pd.concat([top20_out, bot20_out], ignore_index=True)
save_table(combined, "table2_top_bottom_20_countries_qci_2021.csv",
           "Table 2: Top 20 & Bottom 20 Countries by QCI in 2021")


# ================================================================
# TABLE 3: QCI by Age Group and Sex (Global, 2021)
# ================================================================
print()
print("#" * 80)
print("# TABLE 3: QCI by Age Group and Sex (Global, 2021)")
print("#" * 80)

age_order = ["<5 years", "5-14 years", "15-49 years", "50-69 years", "70+ years",
             "Age-standardized", "All ages"]
age_display = {
    "<5 years": "<5",
    "5-14 years": "5-14",
    "15-49 years": "15-49",
    "50-69 years": "50-69",
    "70+ years": "70+",
    "Age-standardized": "Age-standardized",
    "All ages": "All ages",
}

global_2021 = qci_df[
    (qci_df["location_name"] == "Global") & (qci_df["year"] == 2021)
].copy()

rows_t3 = []
for age in age_order:
    row = {"Age_Group": age_display[age]}
    for sex in ["Both", "Female", "Male"]:
        val = global_2021[
            (global_2021["age_name"] == age) & (global_2021["sex_name"] == sex)
        ]["qci"]
        row[f"QCI_{sex}"] = round(val.values[0], 2) if len(val) > 0 else np.nan
    if not pd.isna(row.get("QCI_Female")) and not pd.isna(row.get("QCI_Male")) and row["QCI_Male"] != 0:
        row["GDR"] = round(row["QCI_Female"] / row["QCI_Male"], 4)
    else:
        row["GDR"] = np.nan
    rows_t3.append(row)

table3 = pd.DataFrame(rows_t3)
save_table(table3, "table3_qci_age_sex_global_2021.csv",
           "Table 3: QCI by Age Group and Sex (Global, 2021)")


# ================================================================
# TABLE 4: Iran Provincial QCI (2021)
# ================================================================
print()
print("#" * 80)
print("# TABLE 4: Iran Provincial QCI (2021)")
print("#" * 80)

iran_provinces = [
    "Alborz", "Ardebil", "Bushehr", "Chahar Mahaal and Bakhtiari",
    "East Azarbayejan", "Fars", "Gilan", "Golestan", "Hamadan",
    "Hormozgan", "Ilam", "Isfahan", "Kerman", "Kermanshah",
    "Khorasan-e-Razavi", "Khuzestan", "Kohgiluyeh and Boyer-Ahmad",
    "Kurdistan", "Lorestan", "Markazi", "Mazandaran", "North Khorasan",
    "Qazvin", "Qom", "Semnan", "Sistan and Baluchistan", "South Khorasan",
    "Tehran", "West Azarbayejan", "Yazd", "Zanjan",
]

rows_t4 = []
for prov in iran_provinces:
    row = {"Province": prov}
    val_2021 = qci_std[(qci_std["location_name"] == prov) & (qci_std["year"] == 2021)]["qci"]
    row["QCI_2021"] = round(val_2021.values[0], 2) if len(val_2021) > 0 else np.nan
    val_1990 = qci_std[(qci_std["location_name"] == prov) & (qci_std["year"] == 1990)]["qci"]
    row["QCI_1990"] = round(val_1990.values[0], 2) if len(val_1990) > 0 else np.nan
    if not pd.isna(row.get("QCI_2021")) and not pd.isna(row.get("QCI_1990")):
        row["Change"] = round(row["QCI_2021"] - row["QCI_1990"], 2)
    else:
        row["Change"] = np.nan
    comp_row = comp_std[(comp_std["location_name"] == prov) & (comp_std["year"] == 2021)]
    if len(comp_row) > 0:
        cr = comp_row.iloc[0]
        row["MIR_2021"] = round(cr["MIR"], 4)
        row["YLLtoYLD_2021"] = round(cr["YLLtoYLD"], 2)
        row["DALtoPER_2021"] = round(cr["DALtoPER"], 2)
    else:
        row["MIR_2021"] = np.nan
        row["YLLtoYLD_2021"] = np.nan
        row["DALtoPER_2021"] = np.nan
    rows_t4.append(row)

table4 = pd.DataFrame(rows_t4)
table4 = table4.sort_values("QCI_2021", ascending=False).reset_index(drop=True)
save_table(table4, "table4_iran_provincial_qci_2021.csv",
           "Table 4: Iran Provincial QCI (2021, Age-standardized, Both sexes)")


# ================================================================
# TABLE 5: Validation Summary - QCI vs SDI Correlations
# ================================================================
print()
print("#" * 80)
print("# TABLE 5: Validation Summary")
print("#" * 80)

validation_df = comp_std[
    (comp_std["year"] == 2021) &
    (comp_std["sdi_group"].notna())
].copy()

qci_vals = validation_df["pca_score"].values
sdi_vals = validation_df["sdi_value_2021"].values

valid_mask = ~(np.isnan(qci_vals) | np.isnan(sdi_vals))
qci_clean = qci_vals[valid_mask]
sdi_clean = sdi_vals[valid_mask]

n_countries = len(qci_clean)

spearman_r, spearman_p = stats.spearmanr(qci_clean, sdi_clean)
pearson_r, pearson_p = stats.pearsonr(qci_clean, sdi_clean)

rows_t5 = [
    {
        "Metric": "QCI vs SDI (Spearman rho)",
        "Coefficient": round(spearman_r, 4),
        "P_value": f"{spearman_p:.2e}",
        "N_countries": n_countries,
        "Interpretation": "Monotonic association",
    },
    {
        "Metric": "QCI vs SDI (Pearson r)",
        "Coefficient": round(pearson_r, 4),
        "P_value": f"{pearson_p:.2e}",
        "N_countries": n_countries,
        "Interpretation": "Linear association",
    },
    {
        "Metric": "QCI 2021 - Mean",
        "Coefficient": round(float(np.mean(qci_clean)), 2),
        "P_value": "-",
        "N_countries": n_countries,
        "Interpretation": "Across all countries",
    },
    {
        "Metric": "QCI 2021 - Median",
        "Coefficient": round(float(np.median(qci_clean)), 2),
        "P_value": "-",
        "N_countries": n_countries,
        "Interpretation": "Across all countries",
    },
    {
        "Metric": "QCI 2021 - SD",
        "Coefficient": round(float(np.std(qci_clean, ddof=1)), 2),
        "P_value": "-",
        "N_countries": n_countries,
        "Interpretation": "Across all countries",
    },
    {
        "Metric": "QCI 2021 - Min",
        "Coefficient": round(float(np.min(qci_clean)), 2),
        "P_value": "-",
        "N_countries": n_countries,
        "Interpretation": "Across all countries",
    },
    {
        "Metric": "QCI 2021 - Max",
        "Coefficient": round(float(np.max(qci_clean)), 2),
        "P_value": "-",
        "N_countries": n_countries,
        "Interpretation": "Across all countries",
    },
    {
        "Metric": "QCI 2021 - IQR",
        "Coefficient": round(float(np.percentile(qci_clean, 75) - np.percentile(qci_clean, 25)), 2),
        "P_value": "-",
        "N_countries": n_countries,
        "Interpretation": "Q3 - Q1",
    },
]

table5 = pd.DataFrame(rows_t5)
save_table(table5, "table5_validation_summary.csv",
           "Table 5: Validation Summary - QCI Correlation with SDI (2021, Countries only)")


# Final summary
print()
print(SEP)
print("  ALL TABLES GENERATED SUCCESSFULLY")
print(SEP)
print(f"  Output directory: {OUTPUT_DIR}")
print()
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith(".csv"):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<55s}  ({size:>6,} bytes)")
print()
print(SEP)
