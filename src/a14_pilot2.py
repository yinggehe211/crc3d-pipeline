# -*- coding: utf-8 -*-
# a14_pilot2.py - Pilot 2: CNV estimation (raw-matrix read-depth ratio) + dosage attribution of the 158 switch bins
# Attribution: A->B bins should be enriched for copy loss, B->A for copy gain (dosage-driven);
#       switches without accompanying CNV = non-dosage (epigenetic-reprogramming candidates, core asset of the paper)
import gzip, os
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results", "pilot")
os.makedirs(OUT, exist_ok=True)

RAW_M = {
    "BRD3179N_normal": "GSM3930258_BRD3179N_100000.matrix.gz",
    "BRD3179_tumor":   "GSM3930259_BRD3179_100000.matrix.gz",
}


def load_counts(path):
    """raw (non-iced) matrix -> genome-wide row sums = per-bin read depth"""
    df = pd.read_csv(path, sep="\t", header=None, compression="gzip",
                     engine="c", names=["i", "j", "v"])
    n = int(df[["i", "j"]].max().max()) + 1
    m = sp.coo_matrix((df.v.values, (df.i.values, df.j.values)),
                      shape=(n, n)).tocsr()
    m = m + m.T
    return np.asarray(m.sum(axis=1)).ravel()


cov = {}
for tag, fn in RAW_M.items():
    p = os.path.join(RAW, fn)
    if not os.path.exists(p):
        raise SystemExit(f"missing {fn} - download the raw matrices from GEO first (check GSM ids in the filelist)")
    cov[tag] = load_counts(p)
    print(f"{tag}: read-depth bins={len(cov[tag])} median={np.median(cov[tag]):.0f}")

# CNV: log2(T/N), per-chromosome median de-biasing (each tumor chromosome has its own ploidy)
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]
bed["bin0"] = pd.to_numeric(bed["binID"]).astype(int) - 1
bed["start_i"] = bed["start"].astype(int)

cnv = pd.DataFrame({"chr": bed.chr, "bin0": bed.bin0, "start": bed.start_i})
cnv["log2cnv"] = np.log2((cov["BRD3179_tumor"] + 1) / (cov["BRD3179N_normal"] + 1))
med_by_chr = cnv.groupby("chr").log2cnv.transform("median")
cnv["log2cnv_chrnorm"] = cnv.log2cnv - med_by_chr  # per-chromosome normalization

# read-depth self-check: copy-number sawtooth should be segmented along chromosomes (autocorrelation > 0)
cnv_sorted = cnv.sort_values(["chr", "bin0"])
ac = np.corrcoef(cnv_sorted.log2cnv_chrnorm[:-1], cnv_sorted.log2cnv_chrnorm[1:])[0, 1]
print(f"adjacent-bin CNV autocorrelation: {ac:.3f} (gate > 0.3; proves the read-depth signal is segmented copy number, not noise)")

# ---- switch-bin attribution ----
sw = pd.read_csv(os.path.join(OUT, "pilot1b_NT_switch_confident.csv"))
sw = sw.merge(cnv[["chr", "bin0", "log2cnv_chrnorm"]], on=["chr", "bin0"], how="left")
bg = cnv.merge(
    sw[["chr", "bin0"]].assign(is_sw=True), on=["chr", "bin0"], how="left")
bg["is_sw"] = bg.is_sw.fillna(False)

sw["dir"] = np.where((sw.comp_N == "B") & (sw.comp_T == "A"), "B2A", "A2B")
gain = sw.log2cnv_chrnorm > 0.3
loss = sw.log2cnv_chrnorm < -0.3
print(f"\n===== CNV attribution of the 158 high-confidence switch bins =====")
print(f"  switch-bin CNV quantiles: median={sw.log2cnv_chrnorm.median():.2f} "
      f"| |CNV|>0.3 fraction: {100*(sw.log2cnv_chrnorm.abs()>0.3).mean():.0f}% "
      f"(genome-wide background: {100*(bg.log2cnv_chrnorm.abs()>0.3).mean():.0f}%)")
for d in ("B2A", "A2B"):
    s = sw[sw.dir == d]
    print(f"  {d}: n={len(s)} | copy gain {(s.log2cnv_chrnorm>0.3).mean():.0%} "
          f"| copy loss {(s.log2cnv_chrnorm<-0.3).mean():.0%} "
          f"| median CNV={s.log2cnv_chrnorm.median():+.2f}")
# key gate: median CNV of A->B bins should be significantly below background (loss-driven contraction)
from scipy.stats import mannwhitneyu
u, p_ab = mannwhitneyu(sw[sw.dir == "A2B"].log2cnv_chrnorm.dropna(),
                       bg[~bg.is_sw].log2cnv_chrnorm.dropna())
u2, p_ba = mannwhitneyu(sw[sw.dir == "B2A"].log2cnv_chrnorm.dropna(),
                        bg[~bg.is_sw].log2cnv_chrnorm.dropna())
print(f"  * A2B vs background CNV distribution: p={p_ab:.2e} (expected significantly < 0)")
print(f"  * B2A vs background CNV distribution: p={p_ba:.2e} (expected significantly > 0)")
nondose = sw[sw.log2cnv_chrnorm.abs() <= 0.3]
print(f"  non-dosage switches (epigenetic-reprogramming candidates): {len(nondose)} bins "
      f"(A2B {int((nondose.dir=='A2B').sum())} / B2A {int((nondose.dir=='B2A').sum())})")

sw.to_csv(os.path.join(OUT, "pilot2_switch_CNV_attributed.csv"),
          index=False, encoding="utf-8")
cnv.to_csv(os.path.join(OUT, "pilot2_bin_CNV_100kb.csv"),
           index=False, encoding="utf-8")
print("\n=== Pilot 2 written: switch_CNV_attributed + bin_CNV_100kb ===")
