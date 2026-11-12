# -*- coding: utf-8 -*-
# a24a_w5_align_embed.py — cohort-wide PC1 sign alignment + QC recheck + Layer-1 joint embedding
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
REF  = "FHC-1_100000"  # sign reference: normal control

mat = np.load(os.path.join(RES, "a22f_layer1_matrix.npy"))          # (40, 30968)
samples = pd.read_csv(os.path.join(RES, "a22f_sample_index.csv"))["sample"].tolist()
bed = pd.read_csv(os.path.join(ROOT, "data", "raw", "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
bed["bin"] -= 1
bed = bed[bed["bin"] < mat.shape[1]]
chrom_arr = np.full(mat.shape[1], "NA", dtype=object)
chrom_arr[bed["bin"].values] = bed["chrom"].values
chroms = [c for c in bed["chrom"].unique() if c not in ("chrY",)]   # chrY excluded from alignment/QC
ridx = samples.index(REF)

# ---- 1) sign alignment: per-chromosome flip toward the reference ----
aligned = mat.copy()
flips = []
for i, sm in enumerate(samples):
    if i == ridx: continue
    n_flip = 0
    for c in chroms:
        idx = np.where(chrom_arr == c)[0]
        x, r = mat[i, idx].astype(float), mat[ridx, idx].astype(float)
        if np.std(x) == 0 or np.std(r) == 0: continue
        if np.corrcoef(x, r)[0, 1] < 0:
            aligned[i, idx] = -x; n_flip += 1
    flips.append(n_flip)
print(f"=== 1) sign alignment (reference={REF}) ===")
print(f"Chromosomes flipped per sample: mean={np.mean(flips):.1f} max={max(flips)} "
      f"(~12/23 flips is normal: PC1 signs are arbitrary by construction)")

# ---- 2) post-alignment recheck ----
# 2-a A fraction
rows = []
for i, sm in enumerate(samples):
    v = aligned[i]; ok = v != 0
    frac = float((v[ok] > 0).mean())
    rows.append({"sample": sm, "A_fraction_aligned": round(frac, 3),
                 "criterion": "PASS" if 0.35 <= frac <= 0.75 else "REVIEW"})
QC2b = pd.DataFrame(rows)
print("\n=== 2-a A fraction after alignment ===")
print(QC2b.to_string(index=False))
bad = QC2b[QC2b["criterion"] != "PASS"]
print(f"Samples flagged REVIEW: {list(bad['sample']) if len(bad) else 'none'}")

# 2-b homolog-pair consistency (core-name normalization: strip -sb, trailing N, duplicate numbering)
def core(name):
    b = name.replace("_100000", "")
    b = re.sub(r"-sb$", "", b)
    b = re.sub(r"N$", "", b)
    return b
groups = {}
for i, sm in enumerate(samples):
    groups.setdefault(core(sm), []).append(i)

def pair_corr(a, b):
    rs = []
    for c in chroms:
        idx = np.where(chrom_arr == c)[0]
        if len(idx) < 50: continue
        x, y = aligned[a, idx].astype(float), aligned[b, idx].astype(float)
        if np.std(x) == 0 or np.std(y) == 0: continue
        rs.append(abs(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(rs)) if rs else np.nan

print("\n=== 2-b homolog-pair consistency (aligned; T/N + sb + replicates) ===")
import itertools
prows, base = [], []
for a, b in itertools.combinations(range(len(samples)), 2):
    base.append(pair_corr(a, b))
print(f"Whole-cohort baseline mean: {np.nanmean(base):.3f}")
for gname, mem in groups.items():
    if len(mem) < 2: continue
    for j in range(len(mem)):
        for k in range(j + 1, len(mem)):
            a, b = mem[j], mem[k]
            v = pair_corr(a, b)
            prows.append({"core_name": gname,
                          "sample_pair": f"{samples[a]} vs {samples[b]}",
                          "mean_abs_corr": round(v, 3),
                          "vs_baseline": round(v - np.nanmean(base), 3),
                          "criterion": "PASS" if v > np.nanmean(base) else "below baseline!"})
PF = pd.DataFrame(prows).sort_values("mean_abs_corr")
print(PF.to_string(index=False))

# ---- 3) joint embedding: PCA ----
from sklearn.decomposition import PCA
X = aligned.copy()
ok_bins = (np.abs(X) > 0).sum(axis=0) >= int(0.8 * len(samples))  # bins valid in >=80% of samples
X = X[:, ok_bins].astype(np.float64)
X = X - X.mean(axis=0, keepdims=True)
p = PCA(n_components=10).fit(X)
Z = p.transform(X)
print(f"\n=== 3) Layer-1 joint embedding (valid bins {ok_bins.sum()}) ===")
for k in range(5):
    print(f"PC{k+1} variance explained: {p.explained_variance_ratio_[k]*100:.1f}%")
EMB = pd.DataFrame({"sample": samples})
for k in range(Z.shape[1]): EMB[f"PC{k+1}"] = Z[:, k].round(4)
EMB["core_name"] = [core(s) for s in samples]
EMB.to_csv(os.path.join(RES, "a24a_layer1_embedding.csv"), index=False, encoding="utf-8-sig")
np.save(os.path.join(RES, "a24a_layer1_aligned.npy"), aligned)
QC2b.to_csv(os.path.join(RES, "a24a_ab_fraction_aligned.csv"), index=False, encoding="utf-8-sig")
PF.to_csv(os.path.join(RES, "a24a_pairs_aligned.csv"), index=False, encoding="utf-8-sig")
print("\nSaved: a24a_layer1_embedding.csv, a24a_layer1_aligned.npy, "
      "a24a_ab_fraction_aligned.csv, a24a_pairs_aligned.csv")
