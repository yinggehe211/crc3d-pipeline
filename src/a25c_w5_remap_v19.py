# -*- coding: utf-8 -*-
# a25c_w5_remap_v19.py — GENCODE v19 (hg19) remapping: gene columns of switch bins,
#   regions and shared regions all corrected
import os, gzip
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
RAW  = os.path.join(ROOT, "data", "raw")

# ---- 1) HEAD probe + triple download guard ----
url = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
       "release_19/gencode.v19.annotation.gtf.gz")
h = requests.head(url, allow_redirects=True, timeout=60)
size = int(h.headers["Content-Length"])
print(f"HEAD probe: {url}\n  Content-Length = {size/1e6:.1f}MB "
      f"{'OK' if size > 30e6 else 'ABNORMAL!'}")
assert size > 30e6, "HEAD size abnormal, aborting"
gtf_path = os.path.join(RAW, "gencode.v19.annotation.gtf.gz")
if not (os.path.exists(gtf_path) and os.path.getsize(gtf_path) == size
        and not os.path.exists(gtf_path + ".aria2")):
    print("Downloading GENCODE v19 (~40MB)...")
    r = requests.get(url, timeout=600); r.raise_for_status()
    assert len(r.content) == size, "download size != HEAD, aborting"
    open(gtf_path, "wb").write(r.content)
    print("Guard passed: no .aria2 + size == HEAD\n")

# ---- 2) parse gene rows (once, reused throughout) ----
genes = []
with gzip.open(gtf_path, "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t")
        if p[2] != "gene": continue
        a = p[8]
        genes.append((p[0], int(p[3]) - 1, int(p[4]),
                      a.split('gene_name "')[1].split('"')[0],
                      a.split('gene_type "')[1].split('"')[0]))
G = pd.DataFrame(genes, columns=["chrom", "gs", "ge", "gene_name", "gene_type"])
Gf = G.set_index("gene_name")
print(f"v19 genes: {len(G)}\n")

def hit_genes(c, s, e, k=10):
    ov = G[(G["chrom"] == c) & (G["gs"] < e) & (G["ge"] > s)]
    return ",".join(ov.gene_name.head(k))

# ---- 3) remap 1: BRD3187 switch bins -> gene list ----
sw = pd.read_csv(os.path.join(RES, "a24d_BRD3187_switch_bins.csv"))
sw["gene_name"] = [hit_genes(r["chrom"], r["start"], r["end"], k=20) for _, r in sw.iterrows()]
sw["diff_scaled"] = sw["diff_scaled"].round(4)
sw.to_csv(os.path.join(RES, "a25c_switch_bins_genes_v19.csv"), index=False, encoding="utf-8-sig")
print(f"(1) switch-bin genes remapped: a25c_switch_bins_genes_v19.csv ({len(sw)} rows)")

# ---- 4) remap 2: a25b regions + a26a per-patient regions/shared regions ----
for fname, gcol, out in [("a25b_switch_regions.csv", "genes", "a25c_regions_v19.csv"),
                         ("a26a_BRD3187_regions.csv", "genes", "a26c_BRD3187_regions_v19.csv"),
                         ("a26a_BRD3462_regions.csv", "genes", "a26c_BRD3462_regions_v19.csv")]:
    R = pd.read_csv(os.path.join(RES, fname))
    R[gcol] = [hit_genes(r["chrom"], r["start"], r["end"]) for _, r in R.iterrows()]
    R.to_csv(os.path.join(RES, out), index=False, encoding="utf-8-sig")
    print(f"(2) {fname} -> {out}")

S  = pd.read_csv(os.path.join(RES, "a26a_shared_regions.csv"))
R1 = pd.read_csv(os.path.join(RES, "a26c_BRD3187_regions_v19.csv"))
R2 = pd.read_csv(os.path.join(RES, "a26c_BRD3462_regions_v19.csv"))

def region_genes(R, c, s, e):
    ov = R[(R["chrom"] == c) & (R["start"] < e) & (R["end"] > s)]
    return ov.iloc[0]["genes"] if len(ov) else ""

S["genes_BRD3187_v19"] = [region_genes(R1, r["chrom"],
                              float(r["region"].split("-")[0]) * 1e6,
                              float(r["region"].split("-")[1].replace("Mb", "")) * 1e6)
                          for _, r in S.iterrows()]
S["genes_BRD3462_v19"] = [region_genes(R2, r["chrom"],
                              float(r["region"].split("-")[0]) * 1e6,
                              float(r["region"].split("-")[1].replace("Mb", "")) * 1e6)
                          for _, r in S.iterrows()]
S.to_csv(os.path.join(RES, "a26c_shared_regions_v19.csv"), index=False, encoding="utf-8-sig")
print("  a26a_shared_regions.csv -> a26c_shared_regions_v19.csv")

# ---- 5) landmark loci: true hg19 coordinates from the v19 GTF (memorized hardcodes forbidden) ----
print("\n=== (3) landmark loci hg19 coordinates (verbatim from v19 GTF) vs both patients' switch regions ===")
for g in ["APC", "DCC", "SMAD2", "SMAD4", "MYC", "TP53", "KRAS", "MSH3", "MTDH", "ANKRD1"]:
    if g not in Gf.index:
        print(f"  {g}: absent in v19"); continue
    row = Gf.loc[g]
    if isinstance(row, pd.DataFrame): row = row.iloc[0]
    c, s, e = row["chrom"], int(row["gs"]), int(row["ge"])
    ov1 = R1[(R1["chrom"] == c) & (R1["start"] < e) & (R1["end"] > s)]
    ov2 = R2[(R2["chrom"] == c) & (R2["start"] < e) & (R2["end"] > s)]
    d1 = ov1.iloc[0]["direction"] if len(ov1) else "-"
    d2 = ov2.iloc[0]["direction"] if len(ov2) else "-"
    print(f"  {g} [{c}:{s/1e6:.2f}-{e/1e6:.2f}Mb] 3187:{d1} 3462:{d2}")

# ---- 6) rough random-expectation estimate for shared regions ----
tot2 = (R2["end"] - R2["start"]).sum()
p = tot2 / 2.8e9
exp = len(R1) * p * 0.5  # 0.5 = probability of matching direction
print(f"\n=== (4) shared-region statistics ===")
print(f"Observed shared regions: {len(S)}; random expectation ~{exp:.1f} "
      f"(BRD3462 coverage {tot2/1e6:.0f}Mb / 2.8Gb, direction 1/2) "
      f"-> enrichment ~{len(S)/max(exp,0.1):.0f}x")
print("\nShared regions v19 genes (first 10):")
print(S.head(10)[["chrom", "region", "direction", "genes_BRD3187_v19", "genes_BRD3462_v19"]]
      .to_string(index=False, max_colwidth=60))
