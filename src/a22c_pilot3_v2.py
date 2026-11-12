# -*- coding: utf-8 -*-
# a22c_pilot3_v2.py v2 — v1 crash root cause: the a21c anchor table carries its own
#   chromosome column; after merge the column became chrom_x/chrom_y and
#   groupby("chrom") raised KeyError. v2 fix: take only bin/label from the anchor
#   table. v1 already produced core numbers (ARI6=0.1875/AB=0.0077/supervised
#   0.1828) which are NOT recomputed; this run completes the diagnostic sections.
# Registered criteria (a21d verbatim): k=6 / weights 1.0 anchor - 0.2 non-anchor /
#   centroids = anchor means / PC1-4 Euclidean / ARI >= 0.6
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")

# ---- 1) data: a22b PCs + a21c anchor table (two columns only, to avoid column collision) ----
pc   = pd.read_csv(os.path.join(RES, "a22b_genome_PCs.csv"))
anch = pd.read_csv(os.path.join(RES, "a21c_bin_anchor.csv"), usecols=["bin", "label"])
pc = pc.merge(anch, on="bin", how="left")
pc["label"] = pc["label"].fillna("-")
PCS = ["PC1", "PC2", "PC3", "PC4"]
X  = pc[PCS].values.astype(float)
yA = pc["label"].values
mE = yA != "-"
print(f"Fitted bins: {len(pc)}, anchor bins: {int(mE.sum())}")
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

# ---- 2) Run A: registered K-means ----
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
print("\n=== W7-1 registered criteria ===")
print(f"6-class ARI = {A6:.4f} (n={int(mE.sum())}) -> {'PASS gate' if A6 >= 0.6 else 'FAIL gate'}")
print(f"A/B binary ARI = {AB:.4f}")

# ---- 3) diagnostic 1: chromosome composition of each cluster (tests the 'cluster = chromosome stratification' hypothesis) ----
pc["cluster"] = pred_lab
print("\n[Diagnostic] per-cluster chr-composition entropy + largest-chr share:")
rows = []
for k in range(K):
    mk = pred_lab == labs6[k]
    chrc = pc.loc[mk, "chrom"].value_counts(normalize=True)
    ent = float(-(chrc * np.log(chrc)).sum())
    rows.append({"cluster": labs6[k], "n": int(mk.sum()), "n_chr": int(len(chrc)),
                 "max_chr": chrc.index[0], "max_frac": round(float(chrc.iloc[0]), 3),
                 "chr_entropy": round(ent, 3)})
print(pd.DataFrame(rows).to_string(index=False))

# ---- 4) per-chr ARI + confusion table ----
print("\nPer-chr ARI:")
for c, idx in pc.groupby("chrom").indices.items():
    i = np.asarray(idx); mm = yA[i] != "-"
    if mm.sum() >= 30:
        print(f"  {c}: n={int(mm.sum())} ARI={ari(pred_lab[i][mm], yA[i][mm]):.3f}")
print("\nConfusion table:")
CT = pd.crosstab(pd.Series(pred_lab[mE], name="pred"), pd.Series(yA[mE], name="anchor"))
print(CT.to_string())

# ---- 5) control: global supervised nearest-centroid ----
Cg   = np.vstack([X[yA == l].mean(0) for l in labs6])
d2g  = ((X[:, None, :] - Cg[None, :, :]) ** 2).sum(2)
predg = np.array([labs6[i] for i in d2g.argmin(1)])
Asup = ari(predg[mE], yA[mE])
Acc  = float(np.mean(predg[mE] == yA[mE]))
print(f"\nSupervised nearest-centroid: acc={Acc:.4f}, ARI={Asup:.4f} (a21h per-chr=0.1827)")

# ---- 6) write out ----
pc.rename(columns={"chrom": "chr"}).to_csv(
    os.path.join(RES, "a22c_subcomp_labels.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([
    {"run": "jointeig_kmeans_v3",    "ARI6": round(A6, 4),   "ARI_AB": round(AB, 4)},
    {"run": "jointeig_sup_centroid", "ARI6": round(Asup, 4),
     "ARI_AB": round(ari(np.array(["A" if predg[j].startswith("A") else "B"
                                    for j in np.where(mE)[0]]), ab_true), 4)},
]).to_csv(os.path.join(RES, "a22c_ari_summary.csv"), index=False, encoding="utf-8-sig")
print("Saved: a22c_subcomp_labels.csv / a22c_ari_summary.csv")
