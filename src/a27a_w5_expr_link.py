# -*- coding: utf-8 -*-
# a27a_w5_expr_link.py — BRD3179 compartment switches vs same-patient RNA-seq
#   (GSM4513967) expression linkage
# Registered statistics (declared before results, per the a20f lesson):
#   Primary (bin level): tumor TPM (median log2 per bin) of B->A bins >
#     stableB bins (MW one-sided p<0.05); tumor TPM of A->B bins <
#     stableA bins (MW one-sided p<0.05)
#   Control (gene level): same direction = corroboration
# Join gate 1: pilot1b-vs-bed (chr,start) match rate >= 95%
# Join gate 2: .sf transcripts vs v19 GTF, two-sided normalization
#   (strip _N suffix + strip version) TPM match rate >= 90%
# PC1 source = Pilot 1b anchored version; noise threshold THR = 0.939
#   (cohort-level calibration from the sb replicate); amplitude gate
#   |PC1| > 1e-5 guards against "-0.0" pseudo-flips (effective eigenvector
#   amplitude ~0.01-0.05)
# Lessons logged: (1) join normalization must be two-sided (this round's 0.00%
#   lesson); (2) never name columns after DataFrame methods (a "ge" column
#   shadowed the built-in method; renamed gstart/gend); (3) never assume file
#   names from memory (actual .sf = GSM4513967_BRD3179.sf.txt.gz)
import os, re, gzip
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
PIL  = os.path.join(RES, "pilot")
RAW  = os.path.join(ROOT, "data", "raw")

# ---- 1) PC1: Pilot 1b anchored version + cohort scale + noise threshold ----
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
bed["bin"] -= 1
aligned = np.load(os.path.join(RES, "a24a_layer1_aligned.npy"))
samples = pd.read_csv(os.path.join(RES, "a22f_sample_index.csv"))["sample"].tolist()
smap = {s: i for i, s in enumerate(samples)}
bed = bed[bed["bin"] < aligned.shape[1]].sort_values("bin").reset_index(drop=True)
chrom_arr = np.full(aligned.shape[1], "NA", dtype=object)
chrom_arr[bed["bin"].values] = bed["chrom"].values
chroms = [c for c in bed["chrom"].unique() if c != "chrY"]
scale = {}
for c in chroms:
    idx = np.where(chrom_arr == c)[0]
    ss = [np.std(aligned[i, idx][aligned[i, idx] != 0]) for i in range(len(samples))]
    scale[c] = float(np.median([s for s in ss if s > 0]))
Tdf = pd.read_csv(os.path.join(PIL, "pilot1_BRD3179_tumor_genome_PC1.csv"))
Ndf = pd.read_csv(os.path.join(PIL, "pilot1_BRD3179N_normal_genome_PC1.csv"))
mg = bed.merge(Tdf[["chr", "start", "PC1"]].rename(columns={"PC1": "T_pc1"}),
               left_on=["chrom", "start"], right_on=["chr", "start"], how="inner")
mg = mg.merge(Ndf[["chr", "start", "PC1"]].rename(columns={"PC1": "N_pc1"}),
              left_on=["chrom", "start"], right_on=["chr", "start"], how="inner")
rate = len(mg) / len(Tdf)
print(f"join(chr,start): bed {len(bed)} bins, pilot1b {len(Tdf)} rows, "
      f"matched {len(mg)} (rate {rate*100:.1f}%) "
      f"{'PASS gate' if rate >= 0.95 else '!! GATE FAILED, aborting'}")
assert rate >= 0.95
T = np.zeros(aligned.shape[1]); N = np.zeros(aligned.shape[1])
T[mg["bin"].values] = mg["T_pc1"].values
N[mg["bin"].values] = mg["N_pc1"].values
inb = np.zeros(aligned.shape[1], dtype=bool); inb[mg["bin"].values] = True
EPS = 1e-5
valid = inb & (np.abs(T) > EPS) & (np.abs(N) > EPS)
print(f"Amplitude gate |PC1|>1e-5: blocked {int(inb.sum() - valid.sum())} "
      f"/ {int(inb.sum())} bins")
d = np.zeros(aligned.shape[1])
for c in chroms:
    idx = np.where(chrom_arr == c)[0]
    d[idx] = (T[idx] - N[idx]) / scale[c]
THR = 0.939
flip = valid & (np.abs(d) > THR) & (np.sign(T) != np.sign(N))
group = np.full(aligned.shape[1], "NA", dtype=object)
nf = valid & ~flip
group[nf] = np.where(T[nf] > 0, "stableA", "stableB")
group[flip & (T > 0)] = "B2A"
group[flip & (T < 0)] = "A2B"
print(f"THR={THR:.3f}; BRD3179 T/N: valid={int(valid.sum())}, "
      f"switches={int(flip.sum())} (B2A={int((group=='B2A').sum())}, "
      f"A2B={int((group=='A2B').sum())})\n")

# ---- 2) v19 GTF: gene coordinates + transcript->gene mapping (two-sided normalized join) ----
gene_rows, tx2gene = [], {}
with gzip.open(os.path.join(RAW, "gencode.v19.annotation.gtf.gz"), "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t")
        a = p[8]
        gid = a.split('gene_id "')[1].split('"')[0]
        if p[2] == "gene":
            gene_rows.append((gid, a.split('gene_name "')[1].split('"')[0],
                              p[0], int(p[3]) - 1, int(p[4])))
        elif p[2] == "transcript":
            tid = re.sub(r"\.\d+$", "", a.split('transcript_id "')[1].split('"')[0])
            tx2gene[tid] = gid
G = pd.DataFrame(gene_rows, columns=["gene_id", "gene_name", "chrom", "gstart", "gend"])

# ---- 3) .sf parsing + two-sided normalized join ----
sf = pd.read_csv(os.path.join(RAW, "GSM4513967_BRD3179.sf.txt.gz"), sep="\t")
norm = lambda s: re.sub(r"\.\d+$", "", re.sub(r"_\d+$", "", s))
sf_ids = sf["Name"].map(norm).tolist()
print("Pre-join preview:")
print("  .sf first 5:", sf_ids[:5])
print("  GTF first 5:", list(tx2gene.keys())[:5])
sf["tx_norm"] = sf_ids
sf["gene_id"] = sf["tx_norm"].map(tx2gene)
m_rate = sf.loc[sf.TPM > 0, "gene_id"].notna().mean()
print(f"TPM-weighted match rate: {m_rate*100:.2f}% "
      f"{'PASS gate (>=90%)' if m_rate >= 0.9 else '!! GATE FAILED, aborting'}")
assert m_rate >= 0.9
gtpm = sf.groupby("gene_id")["TPM"].sum()
G["TPM"] = G["gene_id"].map(gtpm)
G = G[G.TPM.notna()].copy()
G["log2TPM"] = np.log2(G.TPM + 1)

# ---- 4) gene-body midpoint -> bin -> group ----
bstart = {c: g["start"].values for c, g in bed.groupby("chrom")}
bidx   = {c: g["bin"].values  for c, g in bed.groupby("chrom")}
bend   = {c: g["end"].values  for c, g in bed.groupby("chrom")}
def bin_of(row):
    st, ix, en = bstart.get(row.chrom), bidx.get(row.chrom), bend.get(row.chrom)
    if st is None: return -1
    j = np.searchsorted(st, row.mid) - 1
    if 0 <= j < len(st) and st[j] <= row.mid < en[j]: return ix[j]
    return -1
G["mid"] = (G["gstart"] + G["gend"]) // 2
G["bin"] = [bin_of(r) for r in G.itertuples()]
G = G[G["bin"] >= 0].copy()
G["group"] = [group[b] for b in G["bin"]]
G = G[G.group != "NA"]
print(f"Genes binned with a group: {len(G)}\n")

# ---- 5) primary tests (bin level) + control (gene level) ----
btpm = G.groupby("bin")["log2TPM"].median()
bg = pd.Series({b: group[b] for b in btpm.index})
rows = []
for lvl in ["bin(primary)", "gene(control)"]:
    for h1, (a_, b_, alt) in {
            "H1: B2A>stableB": ("B2A", "stableB", "greater"),
            "H2: A2B<stableA": ("A2B", "stableA", "less")}.items():
        if lvl.startswith("bin"):
            x = btpm[bg == a_]; y = btpm[bg == b_]
        else:
            x = G.loc[G.group == a_, "log2TPM"]; y = G.loc[G.group == b_, "log2TPM"]
        u, p = stats.mannwhitneyu(x, y, alternative=alt)
        rows.append({"level": lvl, "test": h1, "n1": len(x), "n2": len(y),
                     "median1": round(x.median(), 3), "median2": round(y.median(), 3),
                     "p": f"{p:.2e}",
                     "direction_holds": "yes" if p < 0.05 else "no"})
R = pd.DataFrame(rows)
print("=== Grouped expression tests ===")
print(R.to_string(index=False))
print("\nPer-group bin median log2TPM:",
      {k: round(v, 3) for k, v in btpm.groupby(bg).median().items()})

# ---- 6) W5 shared-region cross-validation ----
print("\n=== W5 shared-region key genes on the BRD3179 expression surface ===")
for g in ["AXIN2", "IFIT2", "IFIT3", "ZEB1", "MTDH"]:
    sub = G[G.gene_name == g]
    if not len(sub):
        print(f"  {g}: v19 TPM=0 or unmapped"); continue
    r = sub.iloc[0]; b = int(r["bin"])
    print(f"  {g} [{r.chrom}:{r.gstart/1e6:.2f}Mb] bin group={group[b]} "
          f"T_PC1={T[b]:+.4f} N_PC1={N[b]:+.4f} TPM={r.TPM:.1f}")
S = pd.read_csv(os.path.join(RES, "a26c_shared_regions_v19.csv"))
seg = S["region"].str.extract(r"([\d.]+)-([\d.]+)Mb")
S["start"] = seg[0].astype(float) * 1e6
S["end"]   = seg[1].astype(float) * 1e6
S["chrom"] = S["chrom"].astype(str)
def in_shared(row):
    ov = S[(S.chrom == row.chrom) & (S.start < row.gend) & (S.end > row.gstart)]
    return ov.iloc[0]["direction"] if len(ov) else "-"
G["shared"] = [in_shared(r) for r in G.itertuples()]
for direction, expect in [("B->A", "high"), ("A->B", "low")]:
    x = G[G.shared == direction].log2TPM
    y = G[G.shared == "-"].log2TPM
    alt = "greater" if expect == "high" else "less"
    u, p = stats.mannwhitneyu(x, y, alternative=alt)
    print(f"  Shared {direction} genes (n={len(x)}) median log2TPM={x.median():.2f} "
          f"vs background={y.median():.2f}, one-sided p={p:.2e} "
          f"{'yes, ' + expect + 'er than background' if p < 0.05 else 'NOT significant'}")

# ---- write out ----
G[["gene_id", "gene_name", "chrom", "gstart", "gend", "TPM", "log2TPM",
   "bin", "group", "shared"]].to_csv(
    os.path.join(RES, "a27a_gene_expr_groups.csv"), index=False, encoding="utf-8-sig")
R.to_csv(os.path.join(RES, "a27a_group_tests.csv"), index=False, encoding="utf-8-sig")
print("\nSaved: a27a_gene_expr_groups.csv, a27a_group_tests.csv")
