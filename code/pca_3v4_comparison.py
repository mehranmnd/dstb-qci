#!/usr/bin/env python3
"""
PCA Comparison: 3-feature vs 4-feature QCI
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import io

DATA   = '/Users/mehranmamandipoor/Desktop/thesis/results/shared/model_data.csv'
REPORT = '/Users/mehranmamandipoor/Desktop/thesis/results/shared/pca_comparison_3v4.txt'

FEAT3 = ["MIR", "YLLtoYLD", "DALtoPER"]
FEAT4 = ["MIR", "YLLtoYLD", "DALtoPER", "PERtoINC"]

df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

train_mask = (df["age_name"] == "Age-standardized") & (df["sex_name"] == "Both")
df_train   = df.loc[train_mask].copy()
print(f"Training rows (Age-standardized, Both): {len(df_train):,}")


def run_pca_pipeline(df_full, df_tr, features, label, t_mask):
    info = {}
    info["label"] = label
    info["features"] = features

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_tr[features])

    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_train)

    info["var_explained"] = pca.explained_variance_ratio_[0]
    info["loadings"]      = dict(zip(features, pca.components_[0]))

    sq = pca.components_[0] ** 2
    info["sq_shares"] = dict(zip(features, sq / sq.sum() * 100))

    X_all  = scaler.transform(df_full[features])
    scores = pca.transform(X_all).ravel()

    mir_train_scores = pca.transform(X_train).ravel()
    corr_mir = np.corrcoef(df_tr["MIR"].values, mir_train_scores)[0, 1]
    info["mir_corr_raw"] = corr_mir
    if corr_mir > 0:
        scores          = -scores
        mir_train_scores = -mir_train_scores
        info["flipped"]  = True
        info["loadings"] = {k: -v for k, v in info["loadings"].items()}
    else:
        info["flipped"] = False

    qci = (scores - scores.min()) / (scores.max() - scores.min()) * 100
    info["qci_col"] = f"QCI_{len(features)}feat"
    df_full[info["qci_col"]] = qci

    train_idx = df_full.index[t_mask]
    info["train_qci"] = df_full.loc[train_idx, info["qci_col"]].values

    return info


res3 = run_pca_pipeline(df, df_train, FEAT3, "3-feature", train_mask)
res4 = run_pca_pipeline(df, df_train, FEAT4, "4-feature", train_mask)

comp = df.loc[train_mask, ["location_name", "year", "QCI_3feat", "QCI_4feat"]].copy()
rho, pval = spearmanr(comp["QCI_3feat"], comp["QCI_4feat"])

latest_year = comp["year"].max()
comp_latest = comp[comp["year"] == latest_year].copy()
comp_latest["rank3"] = comp_latest["QCI_3feat"].rank(ascending=False).astype(int)
comp_latest["rank4"] = comp_latest["QCI_4feat"].rank(ascending=False).astype(int)
n_countries = len(comp_latest)

top20_3 = set(comp_latest.nsmallest(20, "rank3")["location_name"])
top20_4 = set(comp_latest.nsmallest(20, "rank4")["location_name"])
bot20_3 = set(comp_latest.nlargest(20, "rank3")["location_name"])
bot20_4 = set(comp_latest.nlargest(20, "rank4")["location_name"])

age_var = {}
for age in df["age_name"].unique():
    sub = df[(df["age_name"] == age) & (df["sex_name"] == "Both")].dropna(subset=FEAT4)
    if len(sub) < 10:
        continue
    sc = StandardScaler()
    X  = sc.fit_transform(sub[FEAT4])
    pc = PCA(n_components=1, random_state=42)
    pc.fit(X)
    age_var[age] = {
        "n": len(sub),
        "var_pc1": pc.explained_variance_ratio_[0],
        "loadings": dict(zip(FEAT4, pc.components_[0]))
    }

buf = io.StringIO()
w = buf.write

w("=" * 72 + chr(10))
w("  PCA COMPARISON: 3-FEATURE vs 4-FEATURE QCI" + chr(10))
w("=" * 72 + chr(10) + chr(10))

for res in [res3, res4]:
    w("-" * 72 + chr(10))
    w(f"  {res['label'].upper()} MODEL  (features: {res['features']})" + chr(10))
    w("-" * 72 + chr(10))
    w(f"  Variance explained by PC1: {res['var_explained']:.4f}  "
      f"({res['var_explained']*100:.2f}%)" + chr(10))
    w(f"  PC1 flipped (so higher=better): {res['flipped']}" + chr(10))
    w(f"  Pearson corr(MIR, raw PC1) before flip: {res['mir_corr_raw']:.4f}" + chr(10) + chr(10))
    w(f"  {'Feature':<12} {'Loading':>10} {'Squared':>10} {'Contribution%':>15}" + chr(10))
    w(f"  {'--------':<12} {'--------':>10} {'--------':>10} {'-------------':>15}" + chr(10))
    for f in res["features"]:
        ld = res["loadings"][f]
        sq = res["sq_shares"][f]
        w(f"  {f:<12} {ld:>10.4f} {ld**2:>10.4f} {sq:>14.2f}%" + chr(10))
    w(chr(10))

w("=" * 72 + chr(10))
w("  COMPARISON (on Age-standardized, Both sexes, all years)" + chr(10))
w("=" * 72 + chr(10) + chr(10))
w(f"  Spearman rho:  {rho:.6f}" + chr(10))
w(f"  Spearman p:    {pval:.2e}" + chr(10) + chr(10))

w(f"  Year used for rank comparison: {latest_year}" + chr(10))
w(f"  Countries in that year: {n_countries}" + chr(10) + chr(10))

w(f"  Top-20 overlap:    {len(top20_3 & top20_4)} / 20" + chr(10))
w(f"  Top-20 only in 3f: {top20_3 - top20_4}" + chr(10))
w(f"  Top-20 only in 4f: {top20_4 - top20_3}" + chr(10) + chr(10))
w(f"  Bottom-20 overlap:    {len(bot20_3 & bot20_4)} / 20" + chr(10))
w(f"  Bottom-20 only in 3f: {bot20_3 - bot20_4}" + chr(10))
w(f"  Bottom-20 only in 4f: {bot20_4 - bot20_3}" + chr(10) + chr(10))

w("-" * 72 + chr(10))
w("  PERtoINC in 4-feature model" + chr(10))
w("-" * 72 + chr(10))
w(f"  Loading:        {res4['loadings']['PERtoINC']:.4f}" + chr(10))
w(f"  Contribution%:  {res4['sq_shares']['PERtoINC']:.2f}%" + chr(10) + chr(10))

comp_latest["rank_diff"] = comp_latest["rank3"] - comp_latest["rank4"]
biggest_movers = comp_latest.reindex(comp_latest["rank_diff"].abs().nlargest(15).index)
w("  Biggest rank movers (rank3 - rank4, positive = improved in 4f):" + chr(10))
w(f"  {'Country':<45} {'Rank3':>6} {'Rank4':>6} {'Diff':>6}" + chr(10))
for _, row in biggest_movers.sort_values("rank_diff", key=abs, ascending=False).iterrows():
    w(f"  {row['location_name']:<45} {int(row['rank3']):>6} {int(row['rank4']):>6} {int(row['rank_diff']):>6}" + chr(10))
w(chr(10))

w("=" * 72 + chr(10))
w("  4-FEATURE PCA BY AGE GROUP (Both sexes, n_components=1)" + chr(10))
w("  (checking if ~82.6% variance explained appears in any subgroup)" + chr(10))
w("=" * 72 + chr(10) + chr(10))
w(f"  {'Age group':<22} {'N':>7} {'Var expl PC1':>14}   Loadings" + chr(10))
w(f"  {'--------------------':<22} {'------':>7} {'-------------':>14}   {'----------------------------------------'}" + chr(10))
for age in sorted(age_var.keys()):
    v = age_var[age]
    ld_str = "  ".join(f"{f}:{v['loadings'][f]:+.3f}" for f in FEAT4)
    w(f"  {age:<22} {v['n']:>7} {v['var_pc1']*100:>13.2f}%   {ld_str}" + chr(10))
w(chr(10))

w("=" * 72 + chr(10))
w("  CONCLUSION" + chr(10))
w("=" * 72 + chr(10))
if rho > 0.99:
    w("  The 3-feature and 4-feature QCI scores are nearly identical" + chr(10))
    w(f"  (Spearman rho = {rho:.6f}). Adding PERtoINC has negligible impact" + chr(10))
    w("  on country rankings. The 3-feature model is preferred for parsimony." + chr(10))
elif rho > 0.95:
    w("  The two QCI scores are very highly correlated but not identical." + chr(10))
    w("  PERtoINC adds some information; consider whether the gain justifies" + chr(10))
    w("  the added complexity." + chr(10))
else:
    w("  The two QCI scores differ meaningfully. Investigate PERtoINC role." + chr(10))
w(chr(10))

report = buf.getvalue()
print(report)

with open(REPORT, "w") as fh:
    fh.write(report)
print(f"Report saved to: {REPORT}")
