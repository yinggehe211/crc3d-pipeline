# -*- coding: utf-8 -*-
# a25b_w5_switch_regions.py — merge contiguous switch bins into regions + chr1 focus + CRC landmark overlay
import os, gzip
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
RAW  = os.path.join(ROOT, "data", "raw")
sw = pd.read_csv(os.path.join(RES, "a24d_BRD3187_switch_bins.csv"))
sw = sw.sort_values(["chrom", "start"]).reset_index(drop=True)

# ---- 1) contiguity: same chromosome, adjacent switch bins with gap <= 200kb merged into a region ----
GAP = 200_000
regions, cur = [], None
for _, b in sw.iterrows():
    if cur and cur["chrom"] == b.chrom and b.start - cur["end"] <= GAP:
        cur["end"] = b["end"]; cur["nbins"] += 1
        cur["B2A"] += (b["direction"] == "B->A"); cur["A2B"] += (b["direction"] == "A->B")
        cur["maxdiff"] = max(cur["maxdiff"], abs(b.diff_scaled))
    else:
        if cur: regions.append(cur)
        cur = {"chrom": b.chrom, "start": b.start, "end": b["end"], "nbins": 1,
               "B2A": int(b["direction"] == "B->A"),
               "A2B": int(b["direction"] == "A->B"),
               "maxdiff": abs(b.diff_scaled)}
if cur: regions.append(cur)
R = pd.DataFrame(regions)
R["lenMb"] = ((R["end"] - R.start) / 1e6).round(2)
R["direction"] = np.where(R.B2A >= R.A2B, "B->A", "A->B")
R = R.sort_values("nbins", ascending=False).reset_index(drop=True)
print(f"=== 1) contiguity result: 622 bins -> {len(R)} switch regions ===")
print(f"Length distribution: median={R.lenMb.median():.1f}Mb max={R.lenMb.max():.1f}Mb; "
      f"large regions >=5Mb: {int((R.lenMb>=5).sum())}")

# ---- 2) gene cache (reuses the GTF downloaded by a25a) ----
gtf_path = os.path.join(RAW, "gencode.v44.annotation.gtf.gz")
genes = []
with gzip.open(gtf_path, "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t")
        if p[2] != "gene": continue
        if p[0] not in set(R.chrom): continue
        gname = p[8].split('gene_name "')[1].split('"')[0]
        genes.append((p[0], int(p[3]) - 1, int(p[4]), gname))
G = pd.DataFrame(genes, columns=["chrom", "gs", "ge", "gene_name"])
R["genes"] = [
    ",".join(G[(G["chrom"] == r["chrom"]) & (G["gs"] < r["end"]) &
               (G["ge"] > r["start"])].gene_name.head(12))
    for _, r in R.iterrows()]

# ---- 3) CRC landmark overlay ----
LANDMARKS = [
    ("APC (5q21) tumor suppressor",  "chr5",  112_100_000, 112_400_000),
    ("SMAD2/4 + DCC (18q21)",        "chr18",  48_500_000,  59_000_000),
    ("MYC (8q24) oncogene",          "chr8",  127_700_000, 127_900_000),
    ("TP53 (17p13) tumor suppressor","chr17",   7_600_000,   7_800_000),
    ("1p36 deletion region",         "chr1",    1_000_000,  15_000_000),
    ("KRAS (12p12)",                 "chr12",  25_200_000,  25_400_000),
    ("8q gained arm",                "chr8",   80_000_000, 145_000_000),
    ("20q gained arm",               "chr20",  32_000_000,  64_000_000),
]
print("\n=== 2) CRC landmarks vs switch-region coverage ===")
for name, c, s, e in LANDMARKS:
    ov = R[(R.chrom == c) & (R.start < e) & (R["end"] > s)]
    if len(ov):
        for _, o in ov.iterrows():
            print(f"  HIT {name}: covered {o.nbins}bins/{o.lenMb}Mb "
                  f"[{o.chrom}:{o.start/1e6:.1f}-{o['end']/1e6:.1f}Mb] direction={o['direction']}")
    else:
        print(f"  --- {name}: no switch-region coverage")

# ---- 4) chr1 fine mapping ----
print("\n=== 3) chr1 switch regions (sorted by bins) ===")
C1 = R[R.chrom == "chr1"].sort_values("nbins", ascending=False)
for _, r in C1.head(12).iterrows():
    print(f"chr1:{r.start/1e6:.1f}-{r['end']/1e6:.1f}Mb ({r.lenMb}Mb) "
          f"{r.nbins}bins direction={r['direction']} genes: {r.genes[:80]}")

# ---- write out ----
R_out = R[["chrom", "start", "end", "lenMb", "nbins", "B2A", "A2B",
           "direction", "maxdiff", "genes"]]
R_out.to_csv(os.path.join(RES, "a25b_switch_regions.csv"), index=False, encoding="utf-8-sig")
print(f"\nSaved: a25b_switch_regions.csv ({len(R)} regions, with gene lists)")
