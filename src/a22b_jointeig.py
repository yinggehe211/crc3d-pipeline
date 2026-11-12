# -*- coding: utf-8 -*-
# a22b_jointeig.py v3 — v2 crash root cause: np.where evaluates BOTH branches in full;
#   non-cis entries got band=27 which lies outside ub, so searchsorted returned
#   len(ub)=27 -> out-of-range indexing. v3 fix: a full-band lookup table
#   (cis band slot = band mean; all other slots = trans global expectation),
#   E retrieved with a single fancy-index. §1-3/§5-8 unchanged from v2.
# a22a probe findings: global 0-based indexing (= binID-1); trans fraction 83.22%
#   (dilute Hi-C); CSR 0.87 GB feasible.
# Pre-registration (W7-1): per-chromosome sign flips / post-hoc rotations are
#   FORBIDDEN; PC1 aligned to gene density via a single global sign flip only.
import gzip, os, time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RES  = os.path.join(ROOT, "results")
FP   = os.path.join(RAW, "GSM3930260_BRD3179N_100000_iced.matrix.gz")

# ---- 1) bed + validity mask (binID basis) ----
bed_all = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                      sep="\t", header=None, names=["chrom", "bstart", "bend", "bin"])
z = np.load(os.path.join(PROC, "mats_cache_N_100kb.npz"))
valid_ids = set()
for c, g in bed_all.groupby("chrom"):
    if c == "chrY": continue
    vv = z[f"valid_{c}"]
    valid_ids.update(g.sort_values("bstart")["bin"].values[np.where(vv)[0]].tolist())
bed = bed_all[bed_all.chrom != "chrY"].sort_values("bin").reset_index(drop=True)
KB = bed[bed["bin"].isin(valid_ids)].reset_index(drop=True)
bids = KB["bin"].values
print(f"Fitted bins (valid & non-chrY): {len(KB)}")

# ---- 2) streaming matrix read (0-based index -> binID -> filter to fitted bins) ----
t0 = time.time()
tot = n_upper = 0
buf_i, buf_j, buf_v = [], [], []
bid_set = np.array(sorted(valid_ids))
for ch in pd.read_csv(FP, sep="\t", header=None, names=["i", "j", "v"], chunksize=5_000_000):
    tot += len(ch)
    n_upper += int((ch["i"].values > ch["j"].values).sum())
    ci = ch["i"].values + 1
    cj = ch["j"].values + 1
    m = np.isin(ci, bid_set) & np.isin(cj, bid_set)
    if m.sum():
        buf_i.append(np.searchsorted(bid_set, ci[m]))
        buf_j.append(np.searchsorted(bid_set, cj[m]))
        buf_v.append(ch["v"].values[m].astype(np.float32))
    if tot % 20_000_000 < 5_000_000:
        print(f" ...{tot/1e6:.0f}M lines, kept={sum(len(x) for x in buf_i)/1e6:.1f}M, "
              f"{time.time()-t0:.0f}s, flush", flush=True)
        with open(os.path.join(RES, "a22b.progress"), "w") as fd:
            fd.write(f"{tot},{time.time()-t0}\n")
I = np.concatenate(buf_i); J = np.concatenate(buf_j); V = np.concatenate(buf_v)
del buf_i, buf_j, buf_v
print(f"Read complete: total {tot/1e6:.0f}M lines -> kept nnz {len(V)/1e6:.1f}M "
      f"({time.time()-t0:.0f}s); i>j row share={100*n_upper/max(tot,1):.1f}%")

# ---- 3) upper-triangle completion -> symmetrize ----
if n_upper < tot * 0.01:
    bi = np.concatenate([I, J]); bj = np.concatenate([J, I])
    bv = np.concatenate([V, V]).astype(np.float64)
else:
    bi, bj, bv = I, J, V.astype(np.float64)
del I, J, V
N = len(KB)
M = coo_matrix((bv, (bi, bj)), shape=(N, N)).tocsr()
M.setdiag(0); M.eliminate_zeros()
M = ((M + M.T) * 0.5).tocsr()
print(f"Symmetric matrix: nnz={M.nnz/1e6:.1f}M, {time.time()-t0:.0f}s")

# ---- 4) O/E (v3 lookup-table version): cis distance bands (36 log-spaced) + single trans expectation ----
chcode  = pd.factorize(KB["chrom"])[0]
cis_mask = chcode[bi] == chcode[bj]
d_all = np.abs(bids[bi] - bids[bj])
edges = np.unique(np.round(np.logspace(0, np.log10(d_all.max() + 1), 37))).astype(int)
band  = np.searchsorted(edges, d_all, side="right")
v_cis = bv[cis_mask]; b_cis = band[cis_mask]
ub = np.unique(b_cis)
global_mean = float(bv.mean())
# full-band lookup table: length=len(edges)+1; cis band slot = band mean
#   (fallback to cis global mean when <30 entries); all other slots = trans expectation
lookup = np.full(len(edges) + 1, global_mean)
for b in ub:
    sel = b_cis == b
    lookup[b] = v_cis[sel].mean() if sel.sum() >= 30 else v_cis.mean()
E = lookup[band]
nz = E > 0
OE = coo_matrix((bv[nz] / E[nz], (bi[nz], bj[nz])), shape=(N, N)).tocsr()
OE.setdiag(0); OE.eliminate_zeros()
print(f"O/E built: trans expectation={global_mean:.4f}, cis bands={len(ub)}, "
      f"nnz={OE.nnz/1e6:.1f}M, {time.time()-t0:.0f}s")
del M, bi, bj, bv, E, lookup

# ---- 5) eigsh k=12 ----
vals, vecs = eigsh(OE, k=12, which="LA")
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]
print(f"eigsh done ({time.time()-t0:.0f}s): top eigenvalues={np.round(vals[:6], 3)}")

# ---- 6) PC1-4 global sign = gene-density alignment (no anchors; the ONLY permitted orientation operation) ----
dens = np.zeros(N)
bs_map = {c: g.sort_values("bstart") for c, g in bed_all.groupby("chrom")}
with gzip.open(os.path.join(RAW, "gencode.v19.annotation.gtf.gz"), "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t")
        if p[2] != "gene": continue
        c = p[0] if p[0].startswith("chr") else "chr" + p[0]
        g = bs_map.get(c)
        if g is None: continue
        mid = (int(p[3]) - 1 + int(p[4])) // 2
        st, bx = g.bstart.values, g.bin.values
        j = np.searchsorted(st, mid) - 1
        if 0 <= j < len(st) and st[j] <= mid < g.bend.values[j]:
            kk = np.searchsorted(bid_set, bx[j])
            if kk < len(bid_set) and bid_set[kk] == bx[j]:
                dens[kk] += 1
lg = np.log1p(dens)
for pi in range(4):
    r = spearmanr(vecs[:, pi], lg).statistic
    if r < 0: vecs[:, pi] *= -1; r = -r
    print(f"PC{pi+1}: eig={vals[pi]:.3f}, Spearman vs gene density={r:.3f}")

# ---- 7) anchor-free QC: per-chr PC1 vs density (verdict readout: # negative-rho chrs vs a21f's 12/22) ----
print("\nPer-chr PC1 vs gene density Spearman (anchor-free QC):")
qc = []
for c, idx in KB.groupby("chrom").indices.items():
    i = np.asarray(idx)
    r = spearmanr(vecs[i, 0], lg[i]).statistic
    qc.append({"chr": c, "n": len(i), "rho_PC1_gdens": round(r, 3)})
Q = pd.DataFrame(qc)
print(Q.to_string(index=False))
print(f"Chrs with negative rho: {int((Q.rho_PC1_gdens < 0).sum())}/{len(Q)} "
      f"(a21f reference: 12/22 flipped)")

# ---- 8) write out ----
out = KB[["bin", "chrom", "bstart", "bend"]].copy()
for pi in range(4):
    out[f"PC{pi+1}"] = vecs[:, pi]
out.to_csv(os.path.join(RES, "a22b_genome_PCs.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"eig": vals}).to_csv(os.path.join(RES, "a22b_eigvals.csv"), index=False)
Q.to_csv(os.path.join(RES, "a22b_qc_per_chr.csv"), index=False)
print(f"\nSaved: a22b_genome_PCs.csv / a22b_eigvals.csv / a22b_qc_per_chr.csv "
      f"(total {time.time()-t0:.0f}s)")
