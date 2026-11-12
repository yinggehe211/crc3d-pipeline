# -*- coding: utf-8 -*-
# a25a_w5_switch_genes.py — map switch bins to genes (auto-detect hg19/hg38 + GENCODE)
import os, gzip
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
RAW  = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)
sw = pd.read_csv(os.path.join(RES, "a24d_BRD3187_switch_bins.csv"))
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])

# ---- 1) determine genome build: chr1 full length ----
c1 = bed[bed.chrom == "chr1"]["end"].max()
build = "hg38" if c1 < 2.5e8 else "hg19"
print(f"chr1 end={c1/1e6:.2f}Mb -> build determined as {build}")
gtf_name = {"hg19": "gencode.v19.annotation.gtf.gz",
            "hg38": "gencode.v44.annotation.gtf.gz"}[build]
url = {"hg19": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
               "release_19/gencode.v19.annotation.gtf.gz",
       "hg38": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
               "release_44/gencode.v44.annotation.gtf.gz"}[build]
gtf_path = os.path.join(RAW, gtf_name)
if not os.path.exists(gtf_path):
    print(f"Downloading GENCODE {build} annotation (~40MB)...")
    r = requests.get(url, timeout=600); r.raise_for_status()
    open(gtf_path, "wb").write(r.content)
    print(f"Download complete: {os.path.getsize(gtf_path)/1e6:.0f}MB")

# ---- 2) parse gene bodies (gene rows: protein_coding, lncRNA, etc.) ----
genes = []
with gzip.open(gtf_path, "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.rstrip("\n").split("\t")
        if p[2] != "gene": continue
        attr = p[8]
        gname = attr.split('gene_name "')[1].split('"')[0]
        gtype = attr.split('gene_type "')[1].split('"')[0]
        genes.append((p[0], int(p[3]) - 1, int(p[4]), gname, gtype))
G = pd.DataFrame(genes, columns=["chrom", "start", "end", "gene_name", "gene_type"])
G = G[G.chrom.isin(sw.chrom.unique())]
print(f"Gene annotation: {len(G)} genes ({G.gene_type.value_counts().head(3).to_dict()})")

# ---- 3) intersection: switch bins overlapping gene bodies ----
rows = []
for _, b in sw.iterrows():
    ov = G[(G.chrom == b.chrom) & (G.start < b["end"]) & (G["end"] > b.start)]
    for _, g in ov.iterrows():
        rows.append({"chrom": b.chrom, "bin_start": b.start, "bin_end": b["end"],
                     "direction": b["direction"], "diff_scaled": b.diff_scaled,
                     "gene_name": g.gene_name, "gene_type": g.gene_type,
                     "gene_start": g.start, "gene_end": g["end"]})
M = pd.DataFrame(rows)
print("\n=== Mapping results ===")
print(f"{len(sw)} switch bins -> {len(M)} gene associations "
      f"({M.gene_name.nunique()} unique genes)")
print(f"Direction distribution:\n{M['direction'].value_counts().to_string()}")
print(f"\nTop-5 gene types: {M.gene_type.value_counts().head(5).to_dict()}")

# ---- 4) gene list output (dedup: keep the record with max |diff| per gene) ----
M["absdiff"] = M.diff_scaled.abs()
GL = (M.sort_values("absdiff", ascending=False)
       .drop_duplicates("gene_name")
       .sort_values(["direction", "absdiff"], ascending=[True, False]))
out_cols = ["gene_name", "direction", "chrom", "bin_start", "bin_end",
            "diff_scaled", "gene_type"]
GL[out_cols].to_csv(os.path.join(RES, "a25a_switch_genes.csv"), index=False, encoding="utf-8-sig")
M.to_csv(os.path.join(RES, "a25a_switch_gene_pairs.csv"), index=False, encoding="utf-8-sig")
print(f"\nSaved: a25a_switch_genes.csv ({len(GL)} genes, sorted by |diff|), "
      f"a25a_switch_gene_pairs.csv")
print("\nB->A genes Top15 (compartments activated in tumor):")
print(GL[GL["direction"] == "B->A"][out_cols].head(15).to_string(index=False))
print("\nA->B genes Top15 (compartments silenced in tumor):")
print(GL[GL["direction"] == "A->B"][out_cols].head(15).to_string(index=False))
