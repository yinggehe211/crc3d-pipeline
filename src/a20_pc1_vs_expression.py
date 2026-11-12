# -*- coding: utf-8 -*-
# a20_pc1_vs_expression.py - PC1 vs same-patient RNA-seq (GSM4513967 Salmon .sf)
# build=hg19 (adjudicated in a18) -> GTF=GENCODE v19; join rule = strip _N suffix on both sides + strip version number (§16.7)
# Gate (§9 Pilot 1 residual): per-chr |Spearman| median >= 0.4
import gzip, os, subprocess, urllib.request, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
OUT  = os.path.join(ROOT, "results", "pilot")
GTF  = os.path.join(RAW, "gencode.v19.annotation.gtf.gz")
GTF_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
           "release_19/gencode.v19.annotation.gtf.gz")
SF   = os.path.join(RAW, "GSM4513967_BRD3179.sf.txt.gz")

# ---- 1. GTF: HEAD probe -> download (per §13 rule 3: download only after URL verified) ----
if not os.path.exists(GTF):
    req = urllib.request.Request(GTF_URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        clen = int(r.headers["Content-Length"])
    print(f"HEAD ok: {GTF_URL}\n  Content-Length = {clen/1e6:.1f} MB")
    assert 5e6 < clen < 5e7, "GTF size abnormal (expected ~10 MB scale), abort"
    subprocess.run(["aria2c", "--check-certificate=false", "-c",
                    "-d", RAW, GTF_URL], check=True)
    assert os.path.getsize(GTF) == clen, "download size differs from HEAD (sparse-file artifact gate, per §17 discipline)"

# ---- 2. GTF parse: transcript_id -> gene_id (+gene_name), gene coordinates ----
print("parsing GTF ...")
tx2gene, genes = {}, {}
t0 = time.time()
with gzip.open(GTF, "rt", encoding="utf-8") as f:
    for ln in f:
        if ln[0] == "#": continue
        p = ln.rstrip("\n").split("\t")
        feat, chrom = p[2], p[0]
        if feat not in ("transcript", "gene"): continue
        attr = p[8]
        i1 = attr.find('gene_id "') + 9;  gid = attr[i1:attr.find('"', i1)]
        if feat == "transcript":
            i2 = attr.find('transcript_id "') + 15
            tid = attr[i2:attr.find('"', i2)]
            tx2gene[tid] = gid
        else:
            i3 = attr.find('gene_name "') + 12
            gname = attr[i3:attr.find('"', i3)] if i3 > 11 else gid
            genes[gid] = (chrom, int(p[3]), int(p[4]), gname)
print(f"  transcripts={len(tx2gene):,} genes={len(genes):,} {time.time()-t0:.0f}s")

# ---- 3. .sf: ID normalization on both sides + aggregate TPM by gene ----
def norm_tid(s):                      # ENST00000456328.2_1 -> ENST00000456328
    return s.split("_")[0].split(".")[0]
sf = pd.read_csv(SF, sep="\t")
sf["tx"] = sf["Name"].map(norm_tid)
sf["gid"] = sf["tx"].map(tx2gene)
matched = sf["gid"].notna().mean()
print(f"\n.sf transcript->GTF gene match rate: {100*matched:.1f}% ({matched*len(sf):,.0f}/{len(sf):,})")
assert matched > 0.9, f"match rate too low, re-check the join rule (§13 rule 5)"
gtpm = sf.groupby("gid")["TPM"].sum()
gtpm = gtpm[gtpm > 0]
print(f"expressed genes (TPM>0): {len(gtpm):,}")

# ---- 4. gene midpoint -> 100-kb bin (hg19) ----
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]
bed["bin0"] = pd.to_numeric(bed["binID"]).astype(int) - 1
bed["start_i"] = bed["start"].astype(int)
bed["end_i"] = bed["end"].astype(int)
rows = []
for gid, tpm in gtpm.items():
    if gid not in genes: continue
    c, s, e, gname = genes[gid]
    mid = (s + e) // 2
    g = bed[bed.chr == c]
    hit = g[(g.start_i <= mid) & (mid < g.end_i) | 
            (g.start_i == g[g.start_i <= mid].start_i.max())]
    # conservative approach: searchsorted semantics (last bin with start <= midpoint)
    gg = g[g.start_i <= mid]
    if len(gg) == 0: continue
    b = gg.sort_values("start_i").iloc[-1]
    rows.append({"gid": gid, "gname": gname, "chr": c, "bin0": int(b.bin0),
                 "log2tpm": float(np.log2(tpm + 1))})
gmap = pd.DataFrame(rows)
print(f"gene->bin mapping success: {len(gmap):,} ({100*len(gmap)/len(gtpm):.1f}%)")
gmap.to_csv(os.path.join(OUT, "a20_gene_TPM_bins.csv"), index=False, encoding="utf-8")

# ---- 5. PC1 (pilot1, column names known) join + per-chr Spearman ----
res = []
for t, pat in (("N", "pilot1_BRD3179N_normal_genome_PC1.csv"),
               ("T", "pilot1_BRD3179_tumor_genome_PC1.csv")):
    pc = pd.read_csv(os.path.join(OUT, pat))[["chr", "bin0", "PC1"]]
    m = gmap.merge(pc, on=["chr", "bin0"], how="inner")
    print(f"\n[{t}] joined bins: {len(m):,}")
    for c, g in m.groupby("chr"):
        if len(g) >= 100:
            rho, pval = spearmanr(g.PC1, g.log2tpm)
            res.append({"sample": t, "chr": c, "n": len(g),
                        "spearman_rho": rho, "p": pval})
rr = pd.DataFrame(res)
rr["abs_rho"] = rr.spearman_rho.abs()
med = rr.groupby("sample")["abs_rho"].median()
print("\n=== per-chr Spearman (PC1 vs log2TPM) ===")
print(rr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n* |rho| median: N={med.get('N', float('nan')):.3f}  T={med.get('T', float('nan')):.3f}"
      f"  (§9 gate: >= 0.4 -> Pilot 1 residual closed)")
print("  sign note: positive rho = A-type-bin genes highly expressed (compartment direction correct); negative-value chr = PC1 sign arbitrary per chr (§16.2),")
print("  read |r|; chromosomes with very low |rho| (< 0.2) listed separately for review")
rr.to_csv(os.path.join(OUT, "a20_pc1_vs_expression.csv"), index=False, encoding="utf-8")
