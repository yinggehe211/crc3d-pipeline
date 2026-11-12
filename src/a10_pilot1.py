# -*- coding: utf-8 -*-
# a10_pilot1.py - Pilot 1: genome-wide per-chromosome compartment PC1 (N/T) + global sign unification + N/T switch bins + Xu second cross-check
# Requires: two iced matrices + binAnnotation under data/raw; per-sample compartment.txt under Xu_repo Input_compartment
import gzip, os, glob
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
XU_DIR = os.path.join(ROOT, "Xu_repo", "CRC_metastasis_code-main", "Fig3b", "Input_compartment")
OUT = os.path.join(ROOT, "results", "pilot")
os.makedirs(OUT, exist_ok=True)

SAMPLES = {
    "BRD3179N_normal": "GSM3930260_BRD3179N_100000_iced.matrix.gz",
    "BRD3179_tumor":   "GSM3930259_BRD3179_100000_iced.matrix.gz",
}
# sign adjudicated by the chr1 cross-check: we are sign-flipped vs Xu -> global flip factor
GLOBAL_SIGN = -1  # if a rerun of the cross-check flips sign again, change here

CENTROMERE = {"chr1": 121_500_000}  # only chr1 needs per-arm (cross-check); all other chromosomes aligned whole


def load_triplet(path):
    with gzip.open(path, "rt") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else r"\s+"
    return pd.read_csv(path, sep=sep, header=None, compression="gzip",
                       engine="c", names=["i", "j", "v"])


def compartment_pc1(sub, bin_starts, binsize=100000):
    """Distance-decay normalization -> Pearson -> eigendecomposition; relaxed flip gate for small chromosomes"""
    d = np.abs(np.subtract.outer(bin_starts, bin_starts)) // binsize
    dmax = int(d.max())
    sum_d = np.bincount(d.ravel(), weights=sub.ravel(), minlength=dmax + 1)
    cnt_d = np.bincount(d.ravel(), minlength=dmax + 1)
    exp_d = np.divide(sum_d, np.maximum(cnt_d, 1))
    oe = sub / np.maximum(exp_d[d], 1e-9)
    np.fill_diagonal(oe, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(oe)
    c = np.nan_to_num(c)
    np.fill_diagonal(c, 0.0)
    w, v = np.linalg.eigh(c)
    order = np.argsort(w)[::-1][:6]
    n = sub.shape[0]
    lo = 3 if n < 800 else 10
    for k in order:
        ev = v[:, k]
        flips = int((ev[:-1] * ev[1:] < 0).sum())
        if lo <= flips <= 300:
            if (ev > 0).sum() and np.median(ev[ev > 0]) < abs(np.median(ev[ev < 0])):
                ev = -ev
            return ev, flips, f"eig{k}"
    ev = v[:, order[0]]
    return ev, int((ev[:-1] * ev[1:] < 0).sum()), f"eig{order[0]}(fallback)"


# ---- binAnnotation ----
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]
bed["start_i"] = bed["start"].astype(int)
bed["bin0"] = pd.to_numeric(bed["binID"]).astype(int) - 1

# ---- Xu compartment file inventory ----
xu_files = sorted(glob.glob(os.path.join(XU_DIR, "*", "compartment.txt")))
print("Xu compartment files:")
for f in xu_files:
    print("  ", os.path.basename(os.path.dirname(f)))
xu_by_sample = {os.path.basename(os.path.dirname(f)): f for f in xu_files}


def load_xu(path):
    xu = pd.read_csv(path, sep="\t")
    xu.columns = [c.strip() for c in xu.columns]
    xu["score_num"] = pd.to_numeric(xu.score, errors="coerce")
    return xu[xu.score_num.notna()]


pc1_tables, stats = {}, []
for tag, fn in SAMPLES.items():
    print(f"\n===== {tag} =====")
    df = load_triplet(os.path.join(RAW, fn))
    n_bins = int(df[["i", "j"]].max().max()) + 1
    m = sp.coo_matrix((df.v.values, (df.i.values, df.j.values)),
                      shape=(n_bins, n_bins)).tocsr()
    m = m + m.T

    rows = []
    for chrom in [f"chr{i}" for i in range(1, 23)]:
        cb = bed[(bed.chr == chrom) & (bed.bin0 >= 0) & (bed.bin0 < n_bins)].reset_index(drop=True)
        if len(cb) < 50:
            print(f"  {chrom}: bins={len(cb)} skipped (too short)")
            continue
        ids = cb["bin0"].values.astype(int)
        starts = cb["start_i"].values
        sub = m[ids][:, ids].toarray().astype(np.float64)
        ev, flips, method = compartment_pc1(sub, starts)
        ev = ev * GLOBAL_SIGN  # unified to the Xu coordinate system
        rows.append(pd.DataFrame({"chr": chrom, "binID": cb["binID"], "bin0": ids,
                                  "start": starts, "PC1": ev, "flip_method": method}))
        stats.append({"sample": tag, "chr": chrom, "bins": len(ids),
                      "flips": flips, "method": method})
        print(f"  {chrom}: bins={len(ids)} flips={flips} [{method}]")
    pc1_tables[tag] = pd.concat(rows, ignore_index=True)
    pc1_tables[tag].to_csv(os.path.join(OUT, f"pilot1_{tag}_genome_PC1.csv"),
                           index=False, encoding="utf-8")

# ---- genome-wide second cross-check: our N vs all Xu samples ----
print("\n===== genome-wide cross-check (normal samples, 100-kb aggregated to 500 kb) =====")
our_n = pc1_tables["BRD3179N_normal"].copy()
our_n["bin500"] = (our_n.start // 500000) * 500000
agg = our_n.groupby(["chr", "bin500"]).PC1.mean().reset_index()
for sname, spath in xu_by_sample.items():
    xu = load_xu(spath)
    mg = agg.merge(xu[["seqnames", "start", "score_num"]],
                   left_on=["chr", "bin500"],
                   right_on=["seqnames", xu.start.astype(int) - 1], how="inner")
    if len(mg) < 100:
        continue
    a = ((mg.PC1 > 0) == (mg.score_num > 0)).mean()
    print(f"  vs Xu {sname}: agreement {a:.2%} (n={len(mg)})  [aligned via global sign]")
    stats.append({"sample": f"genome_vs_Xu_{sname}", "agree": round(float(a), 4), "n": len(mg)})

# ---- N/T compartment switch bins ----
print("\n===== N/T compartment switches (unified coordinate system) =====")
n = pc1_tables["BRD3179N_normal"][["chr", "bin0", "start", "PC1"]].rename(columns={"PC1": "PC1_N"})
t = pc1_tables["BRD3179_tumor"][["chr", "bin0", "PC1"]].rename(columns={"PC1": "PC1_T"})
nt = n.merge(t, on=["chr", "bin0"])
nt["comp_N"] = np.where(nt.PC1_N > 0, "A", "B")
nt["comp_T"] = np.where(nt.PC1_T > 0, "A", "B")
nt["switch"] = nt.comp_N != nt.comp_T
nsw = int(nt["switch"].sum())
print(f"  genome-wide bins={len(nt)} | A/B switch bins={nsw} ({nsw/len(nt):.1%})")
print(f"  direction split: B->A {int(((nt.comp_N=='B')&(nt.comp_T=='A')).sum())} | "
      f"A->B {int(((nt.comp_N=='A')&(nt.comp_T=='B')).sum())}")
nt[["chr", "bin0", "start", "PC1_N", "PC1_T", "comp_N", "comp_T", "switch"]].to_csv(
    os.path.join(OUT, "pilot1_NT_switch_bins.csv"), index=False, encoding="utf-8")

pd.DataFrame(stats).to_csv(os.path.join(OUT, "pilot1_summary.csv"),
                           index=False, encoding="utf-8")
print("\n=== Pilot 1 written: genome_PC1 (two samples) + NT_switch_bins + summary ===")
