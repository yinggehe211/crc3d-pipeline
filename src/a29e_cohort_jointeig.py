# -*- coding: utf-8 -*-
# a29e_cohort_jointeig.py — cohort-wide, subtype-gate-free primary analysis
#   (first leg of the main line after Option-B verdict)
# ===== Registered statistics (declared before results, per the a20f/a21d/a27a
#        convention) =====
# Input: a24a_layer1_aligned.npy (40x30968, FHC-1 sign-aligned; continuity
#   verified by a29c/a29d0)
# Mapping: column i <-> binID i+1 (basis = §16.13 a22f domain measurement,
#   max 0-based = 30967 + a29d0 sliding-window chrY signature cross-check);
#   bed binIDs 30969/30970 untouched and dropped; chrY/chrM excluded (chrY
#   per the a23b anomaly record, 0.326)
# Validation: chrY empirical-signature assert (mean within-chr |spearman| <
#   0.5x autosomal mean); on failure stop without writing
# Primary statistic: per-chr (23 chrs, excl chrY/chrM) mean of pairwise
#   |Spearman| = pair score; same-type pairs (N-N ∪ T-T, n=70) vs cross-type
#   (N-T, n=66) MW one-sided + type-label permutation test (10000 draws)
# Secondary: average-linkage (1-|spearman| distance) hierarchical clustering
#   k=2 vs N/T labels ARI (27 after dedup, cell lines excluded); sensitivity
#   surface = full 40-sample table (with replicates), primary statistic rerun
# Dedup key: (base name, type) two-part key — a24a's core name would merge
#   cross-type samples into one key; single-key dedup would wrongly drop them
#   (this script's verdict)
# Interpretation note: BRD3179/3179N are not in this pool (they run the a22b
#   joint-decomposition line). Reading: same-type > cross-type significant =
#   the compartment skeleton carries sample-type information (weak narrative,
#   consistent with a23b sb > T/N > baseline); not significant = type info is
#   not in the skeleton, and the cohort narrative keeps the §16.10 two-layer
#   "shared program + patient-specific deviation" — both branches pre-registered.
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(20261111)
ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
RAW  = os.path.join(ROOT, "data", "raw")
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
M = np.load(os.path.join(RES, "a24a_layer1_aligned.npy"))
idx = pd.read_csv(os.path.join(RES, "a22f_sample_index.csv"))
n_s, n_c = M.shape
print(f"matrix {M.shape}, bed {len(bed)} bins, binID domain "
      f"{bed['bin'].min()}-{bed['bin'].max()}")
assert n_c == 30968, f"columns {n_c} != 30968, mapping premise broken"
assert bed["bin"].max() >= 30968, "bed domain insufficient, binID rule broken"

# --- mapping: column i -> binID i+1 ---
cols = pd.DataFrame({"bin": np.arange(1, n_c + 1)})
bmap = cols.merge(bed[["chrom", "bin"]], on="bin", how="left")
assert bmap["chrom"].notna().all(), "column->bin mapping has misses, domain assumption broken"
chroms = bmap["chrom"].values

# --- extreme-value census (±0.7071 artifact quantification, archived) ---
ext = (np.abs(M) >= 0.7).sum(axis=1)
print(f"|v|>=0.7 extreme bins: median={int(np.median(ext))}, max={ext.max()} (sample axis)")

# --- dedup: (base, type) two-part key ---
CELL = ("FHC", "HCT116", "LS174T", "RKO", "SW480", "Wi38")
types, bases, seen = [], [], set()
keep = []
for i, s in enumerate(idx["sample"]):
    base = s.replace("_100000", "").replace("-sb", "")
    t = "celline" if base.startswith(CELL) else ("normal" if base.endswith("N") else "tumor")
    k = (base, t)
    if k in seen: continue
    seen.add(k); keep.append(i); types.append(t); bases.append(base)
keep = np.array(keep); types = np.array(types)
D = M[keep].astype(np.float64)
print(f"dedup: 40 -> {len(keep)} groups (N={sum(t=='normal' for t in types)}, "
      f"T={sum(t=='tumor' for t in types)}, celline={sum(t=='celline' for t in types)})")

# --- chrY empirical-signature validation ---
def pair_abs_spearman(X):
    # X: (samples x bins) -> pairwise |spearman| matrix (rank transform then Pearson)
    R = np.apply_along_axis(lambda v: pd.Series(v).rank().values, 1, X)
    C = np.corrcoef(R)
    return np.abs(C)
all_chr = [c for c in pd.unique(bmap["chrom"]) if c not in ("chrY", "chrM")]
print(f"chromosomes in: {len(all_chr)} (excl chrY/chrM)")
chr_scores = {}
mean_auto = []
for c in all_chr:
    cmask = chroms == c
    S = pair_abs_spearman(D[:, cmask])
    iu = np.triu_indices(len(keep), 1)
    chr_scores[c] = S[iu]
    mean_auto.append(S[iu].mean())
auto_mean = float(np.mean(mean_auto))
ymask = chroms == "chrY"
y_mean = float(pair_abs_spearman(D[:, ymask])[np.triu_indices(len(keep), 1)].mean())
print(f"autosomal mean |rho|={auto_mean:.4f}, chrY={y_mean:.4f}")
assert y_mean < 0.5 * auto_mean, \
    f"chrY signature validation FAILED ({y_mean:.3f} vs {0.5*auto_mean:.3f}), mapping may be shifted — stopping"
print("chrY signature validation passed -> mapping confirmed")

# --- primary statistic: pair score = per-chr mean ---
iu = np.triu_indices(len(keep), 1)
P = np.mean([chr_scores[c] for c in all_chr], axis=0)
nc = (types != "celline")
same  = nc[iu[0]] & nc[iu[1]] & (types[iu[0]] == types[iu[1]])
cross = nc[iu[0]] & nc[iu[1]] & (types[iu[0]] != types[iu[1]])
s_same, s_cross = P[same], P[cross]
u, p_mw = mannwhitneyu(s_same, s_cross, alternative="greater")
print(f"\nPrimary statistic: same-type pairs n={len(s_same)} mean={s_same.mean():.4f} | "
      f"cross-type pairs n={len(s_cross)} mean={s_cross.mean():.4f} | "
      f"diff={s_same.mean()-s_cross.mean():+.4f}")
print(f"MW one-sided p={p_mw:.3e}")
nt_rows = np.where(types != "celline")[0]
row_to_nt = {int(r): k for k, r in enumerate(nt_rows)}
nt_mask = (types[iu[0]] != "celline") & (types[iu[1]] != "celline")
pi = np.array([row_to_nt[int(r)] for r in iu[0][nt_mask]])
pj = np.array([row_to_nt[int(r)] for r in iu[1][nt_mask]])
Pnt = P[nt_mask]
nl = types[nt_rows]
perm_stat = []
n_perm = 10000
base_stat = s_same.mean() - s_cross.mean()
for _ in range(n_perm):
    pl = rng.permutation(nl)
    ls = pl[pi] == pl[pj]
    if ls.sum() == 0 or (~ls).sum() == 0: continue
    perm_stat.append(Pnt[ls].mean() - Pnt[~ls].mean())
perm_stat = np.array(perm_stat)
p_perm = float((perm_stat >= base_stat).mean())
print(f"Permutation test ({n_perm} draws) p={p_perm:.4f}")

# --- secondary: hierarchical clustering ARI ---
nt_idx = np.where(types != "celline")[0]
R_full = np.zeros((len(nt_idx), len(nt_idx)))
for a in range(len(nt_idx)):
    for b in range(a + 1, len(nt_idx)):
        ia, ib = nt_idx[a], nt_idx[b]
        v = 1.0 - float(pair_abs_spearman(np.stack([D[ia], D[ib]]))[0, 1])
        R_full[a, b] = R_full[b, a] = v
Z = linkage(squareform(R_full, checks=False), method="average")
for k in (2, 3):
    cl = fcluster(Z, k, "maxclust")
    ari = adjusted_rand_score(types[nt_idx], cl)
    print(f"hierarchical clustering k={k} ARI (N/T labels, n={len(nt_idx)}) = {ari:.4f}")

# --- sensitivity surface: full 40-sample table ---
M40 = M.astype(np.float64)
tt = []
for i, s in enumerate(idx["sample"]):
    base = s.replace("_100000", "").replace("-sb", "")
    tt.append("celline" if base.startswith(CELL)
              else ("normal" if base.endswith("N") else "tumor"))
t40 = np.array(tt)
iu40 = np.triu_indices(n_s, 1)
chr_means = []
for c in all_chr:
    cmask = chroms == c
    S = pair_abs_spearman(M40[:, cmask])
    chr_means.append(S[iu40])
P40 = np.mean(chr_means, axis=0)
nc40 = (t40 != "celline")
same40 = nc40[iu40[0]] & nc40[iu40[1]] & (t40[iu40[0]] == t40[iu40[1]])
cross40 = nc40[iu40[0]] & nc40[iu40[1]] & (t40[iu40[0]] != t40[iu40[1]])
print(f"\nSensitivity (40 samples): same-type mean={P40[same40].mean():.4f} vs "
      f"cross-type mean={P40[cross40].mean():.4f}, "
      f"MW p={mannwhitneyu(P40[same40], P40[cross40], alternative='greater')[1]:.3e}")

# --- write out ---
rows = [{"chrom": c, "mean_abs_rho": float(chr_scores[c].mean())} for c in all_chr]
rows.append({"chrom": "chrY", "mean_abs_rho": y_mean})
pd.DataFrame(rows).to_csv(os.path.join(RES, "a29e_per_chr_mean_rho.csv"),
                          index=False, encoding="utf-8")
pd.DataFrame({"pair": [f"{bases[iu[0][k]]}~{bases[iu[1][k]]}" for k in range(len(P))],
              "type_i": types[iu[0]], "type_j": types[iu[1]],
              "same_type": same, "pair_score": P}).to_csv(
    os.path.join(RES, "a29e_pair_scores.csv"), index=False, encoding="utf-8")
pd.DataFrame({"test": ["MW_same_vs_cross", "permutation",
                       "clusterARI_k2", "clusterARI_k3"],
              "value": [p_mw, p_perm,
                        adjusted_rand_score(types[nt_idx], fcluster(Z, 2, "maxclust")),
                        adjusted_rand_score(types[nt_idx], fcluster(Z, 3, "maxclust"))]}
             ).to_csv(os.path.join(RES, "a29e_tests.csv"), index=False, encoding="utf-8")
print("\nSaved: a29e_per_chr_mean_rho.csv / a29e_pair_scores.csv / a29e_tests.csv")
