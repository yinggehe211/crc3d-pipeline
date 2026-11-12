# -*- coding: utf-8 -*-
# a26a_w5_cross_patient.py — BRD3462 switch regions + shared/patient-specific
#   compartment changes across the two patients
import os, gzip
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
RAW  = os.path.join(ROOT, "data", "raw")
aligned = np.load(os.path.join(RES, "a24a_layer1_aligned.npy"))
samples = pd.read_csv(os.path.join(RES, "a22f_sample_index.csv"))["sample"].tolist()
smap = {s: i for i, s in enumerate(samples)}
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
bed["bin"] -= 1
bed = bed[bed["bin"] < aligned.shape[1]].sort_values("bin").reset_index(drop=True)
chrom_arr = np.full(aligned.shape[1], "NA", dtype=object)
chrom_arr[bed["bin"].values] = bed["chrom"].values
chroms = [c for c in bed["chrom"].unique() if c != "chrY"]
scale = {}
for c in chroms:
    idx = np.where(chrom_arr == c)[0]
    ss = [np.std(aligned[i, idx][aligned[i, idx] != 0]) for i in range(len(samples))]
    ss = [s for s in ss if s > 0]
    scale[c] = float(np.median(ss))

def get(name): return aligned[smap[name + "_100000"]].astype(float)

def scaled_diff(a, b):
    d = np.zeros(aligned.shape[1])
    for c in chroms:
        idx = np.where(chrom_arr == c)[0]
        d[idx] = (aligned[smap[a + "_100000"], idx] - aligned[smap[b + "_100000"], idx]) / scale[c]
    return d

nz = scaled_diff("BRD3462N", "BRD3462N-sb")
THR = float(np.percentile(np.abs(nz[nz != 0]), 99))
print(f"Noise 99th-percentile threshold: {THR:.3f}\n")

def switches(t, n):
    T, N = get(t), get(n)
    d = scaled_diff(t, n)
    valid = (T != 0) & (N != 0)
    flip = valid & (np.abs(d) > THR) & (np.sign(T) != np.sign(N))
    return flip, d

def to_regions(flip, d, tumor):
    T = get(tumor)
    sub = bed[flip].copy()
    sub["direction"] = np.where(T[flip] > 0, "B->A", "A->B")
    sub["diff"] = d[flip]
    sub = sub.sort_values(["chrom", "start"]).reset_index(drop=True)
    regs, cur = [], None
    for _, b in sub.iterrows():
        if cur and cur["chrom"] == b["chrom"] and b["start"] - cur["end"] <= 200_000:
            cur["end"] = b["end"]; cur["nbins"] += 1
        else:
            if cur: regs.append(cur)
            cur = {"chrom": b["chrom"], "start": b["start"], "end": b["end"],
                   "nbins": 1, "direction": b["direction"]}
    if cur: regs.append(cur)
    R = pd.DataFrame(regs)
    R["lenMb"] = ((R["end"] - R["start"]) / 1e6).round(2)
    print(f"{tumor} T/N: {int(flip.sum())} bins -> {len(R)} regions "
          f"(>=1Mb: {int((R.lenMb>=1).sum())})")
    return R

f3187, d3187 = switches("BRD3187", "BRD3187N")
f3462, d3462 = switches("BRD3462", "BRD3462N")
R1 = to_regions(f3187, d3187, "BRD3187")
R2 = to_regions(f3462, d3462, "BRD3462")

# ---- gene annotation cache ----
gtf_path = os.path.join(RAW, "gencode.v44.annotation.gtf.gz")
genes = []
with gzip.open(gtf_path, "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t")
        if p[2] != "gene": continue
        genes.append((p[0], int(p[3]) - 1, int(p[4]),
                      p[8].split('gene_name "')[1].split('"')[0]))
G = pd.DataFrame(genes, columns=["chrom", "gs", "ge", "gene_name"])

def add_genes(R):
    out = []
    for _, r in R.iterrows():
        ov = G[(G["chrom"] == r["chrom"]) & (G["gs"] < r["end"]) & (G["ge"] > r["start"])]
        out.append(",".join(ov.gene_name.head(10)))
    R["genes"] = out
    return R
R1, R2 = add_genes(R1), add_genes(R2)

# ---- patient intersection: overlapping intervals on the same chromosome + same direction = shared ----
print("\n=== Shared switch regions (present in both patients with matching direction) ===")
shared = []
for _, a in R1.iterrows():
    ov = R2[(R2["chrom"] == a["chrom"]) & (R2["start"] < a["end"]) & (R2["end"] > a["start"])]
    for _, b in ov.iterrows():
        if a["direction"] == b["direction"]:
            shared.append({"chrom": a["chrom"],
                           "region": f"{a['start']/1e6:.1f}-{a['end']/1e6:.1f}Mb",
                           "direction": a["direction"],
                           "genes_BRD3187": a["genes"][:60],
                           "genes_BRD3462": b["genes"][:60]})
            break
S = pd.DataFrame(shared)
if len(S):
    print(S.to_string(index=False, max_colwidth=55))
else:
    print("No shared regions — compartment switches are fully patient-specific")
print(f"\nStatistics: BRD3187 {len(R1)} regions vs BRD3462 {len(R2)} regions, "
      f"shared {len(S)}")
R1.to_csv(os.path.join(RES, "a26a_BRD3187_regions.csv"), index=False, encoding="utf-8-sig")
R2.to_csv(os.path.join(RES, "a26a_BRD3462_regions.csv"), index=False, encoding="utf-8-sig")
S.to_csv(os.path.join(RES, "a26a_shared_regions.csv"), index=False, encoding="utf-8-sig")
print("Saved: a26a_BRD3187_regions.csv, a26a_BRD3462_regions.csv, a26a_shared_regions.csv")
