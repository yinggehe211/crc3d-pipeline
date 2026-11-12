# -*- coding: utf-8 -*-
# a22d_chrcenter.py — branch-2 registered retry (single attempt): re-run the
#   registered K-means after per-chromosome centering
# a22c diagnostic: clusters = chromosome stratification (A2=100% chr2, A1=77% chr1),
#   but the per-chr ARI median 0.283 > a21h best 0.224 -> within-chr resolution DID
#   improve; the failure is chr-level mean offsets (failure form #3, novel)
# Registration statement: per-chr centering = subtract each PC's mean over that
#   chr's fitted bins (unsupervised translation, NOT sign/rotation; within-chr
#   geometry unchanged) — does not violate the §16.12 prohibition; criteria
#   identical verbatim to a21d: ARI >= 0.6
# Prediction (fixed before results): global ARI should rise to the per-chr range
#   0.25-0.40; the A/B layer should exceed 0.4
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")

# ---- 1) data + per-chr centering ----
pc   = pd.read_csv(os.path.join(RES, "a22b_genome_PCs.csv"))
anch = pd.read_csv(os.path.join(RES, "a21c_bin_anchor.csv"), usecols=["bin", "label"])
pc = pc.merge(anch, on="bin", how="left")
pc["label"] = pc["label"].fillna("-")
PCS = ["PC1", "PC2", "PC3", "PC4"]
chr_means = pc.groupby("chrom")[PCS].transform("mean")
for p in PCS: pc[f"{p}c"] = pc[p] - chr_means[p]
PSC = [f"{p}c" for p in PCS]
X  = pc[PSC].values.astype(float)
yA = pc["label"].values
mE = yA != "-"
print(f"Fitted bins: {len(pc)}, anchor bins: {int(mE.sum())}")
print("Per-chr PC1 centering magnitude (range of chr means):")
print(pc.groupby("chrom")["PC1"].mean().describe().round(4).to_string())
labs6 = [l for l in ["A1", "A2", "B1", "B2", "B3", "B4"] if (yA == l).sum() >= 10]
K = len(labs6)

def ari(y1, y2):
    i1 = {v: i for i, v in enumerate(np.unique(y1))}
    i2 = {v: i for i, v in enumerate(np.unique(y2))}
    M = np.zeros((len(i1), len(i2)), dtype=np.int64)
    for u, v in zip(y1, y2): M[i1[u], i2[v]] += 1
    c2 = lambda x: x * (x - 1) / 2
    s_ij, s_a, s_b = c2(M).sum(), c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    return (s_ij - s_a * s_b / c2(M.sum())) / ((s_a + s_b) / 2 - s_a * s_b / c2(M.sum()))

# ---- 2) registered K-means (parameters verbatim identical to a21d; only change = centered PCs) ----
W = np.where(mE, 1.0, 0.2)
C = np.vstack([X[yA == l].mean(0) for l in labs6])
for it in range(100):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    a  = d2.argmin(1)
    Cn = C.copy()
    for k in range(K):
        mk = a == k
        if mk.sum(): Cn[k] = (X[mk] * W[mk, None]).sum(0) / W[mk].sum()
    if np.allclose(Cn, C, atol=1e-10): break
    C = Cn
print(f"K-means converged: {it + 1} iterations")
pred_lab = np.array([labs6[x] for x in a])
A6 = ari(pred_lab[mE], yA[mE])
ab_true = np.array(["A" if s.startswith("A") else "B" for s in yA[mE]])
ab_pred = np.array(["A" if pred_lab[j].startswith("A") else "B" for j in np.where(mE)[0]])
AB = ari(ab_pred, ab_true)
print("\n=== a22d registered criteria (per-chr centered PC space) ===")
print(f"6-class ARI = {A6:.4f} (n={int(mE.sum())}) -> {'PASS gate' if A6 >= 0.6 else 'FAIL gate'}")
print(f"A/B binary ARI = {AB:.4f} (a22c=0.0077, a21g=0.2619)")

# ---- 3) cluster composition recheck + confusion table + per-chr ----
pc["cluster"] = pred_lab
print("\n[Recheck] cluster chr composition (a22c reference: A2=100% chr2):")
rows = []
for k in range(K):
    mk = pred_lab == labs6[k]
    chrc = pc.loc[mk, "chrom"].value_counts(normalize=True)
    ent = float(-(chrc * np.log(chrc)).sum())
    rows.append({"cluster": labs6[k], "n": int(mk.sum()), "n_chr": int(len(chrc)),
                 "max_chr": chrc.index[0], "max_frac": round(float(chrc.iloc[0]), 3),
                 "chr_entropy": round(ent, 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nConfusion table:")
print(pd.crosstab(pd.Series(pred_lab[mE], name="pred"),
                  pd.Series(yA[mE], name="anchor")).to_string())
print("\nPer-chr ARI:")
per = []
for c, idx in pc.groupby("chrom").indices.items():
    i = np.asarray(idx); mm = yA[i] != "-"
    if mm.sum() >= 30:
        r = ari(pred_lab[i][mm], yA[i][mm])
        per.append({"chr": c, "n": int(mm.sum()), "ARI": round(r, 3)})
P = pd.DataFrame(per)
print(P.to_string(index=False))
print(f"Per-chr ARI median: {P.ARI.median():.3f} (a22c reference = 0.283)")

# ---- 4) supervised control ----
Cg   = np.vstack([X[yA == l].mean(0) for l in labs6])
d2g  = ((X[:, None, :] - Cg[None, :, :]) ** 2).sum(2)
predg = np.array([labs6[i] for i in d2g.argmin(1)])
Asup = ari(predg[mE], yA[mE])
Acc  = float(np.mean(predg[mE] == yA[mE]))
print(f"\nSupervised nearest-centroid: acc={Acc:.4f}, ARI={Asup:.4f}")

# ---- 5) write out ----
pc.rename(columns={"chrom": "chr"}).to_csv(
    os.path.join(RES, "a22d_subcomp_labels_chrcenter.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([
    {"run": "a21g_perchr_v4",           "ARI6": 0.1216,   "ARI_AB": 0.2619},
    {"run": "a21h_procrustes_ceiling",  "ARI6": 0.2242,   "ARI_AB": 0.4038},
    {"run": "a22c_joint_raw",           "ARI6": 0.1875,   "ARI_AB": 0.0077},
    {"run": "a22d_joint_chrcenter",     "ARI6": round(A6, 4), "ARI_AB": round(AB, 4)},
    {"run": "a22d_sup_centroid",        "ARI6": round(Asup, 4), "ARI_AB": ""},
]).to_csv(os.path.join(RES, "a22d_ari_summary.csv"), index=False, encoding="utf-8-sig")
print("Saved: a22d_subcomp_labels_chrcenter.csv / a22d_ari_summary.csv")
