# -*- coding: utf-8 -*-
# a28a_expr_validation.py v2 — v1 crash attribution: GTF "gene" rows carry no
#   transcript_id attribute, the regex fell through, all tids became empty
#   strings, match rate 0.0000; the assert intercepted before any write
#   (no output pollution). v2 fix: read GTF "transcript" rows (tid + gene_id +
#   coordinates complete), join .sf directly on transcripts, drop the
#   unnecessary gene layer (same mapping approach as a27a; the §16.8 lesson-3
#   family struck again).
# Registered statistics (declared before results): monotonic gradient of
#   bin-level median log2TPM across the six labels A1>A2>B1>B2~=B3>B4;
#   A cluster (A1+A2) vs B cluster (B1-B4) MW one-sided p<0.001
import gzip, os, re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
RES  = os.path.join(ROOT, "results")

# ---- 1) labels ----
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "bstart", "bend", "bin"])
lab = pd.read_csv(os.path.join(RES, "a22d_subcomp_labels_chrcenter.csv"))
print(f"Labeled bins: {len(lab)}, {lab.cluster.value_counts().to_dict()}")

# ---- 2) GTF v19 transcript rows -> (tid, bin coordinates) ----
rows = []
with gzip.open(os.path.join(RAW, "gencode.v19.annotation.gtf.gz"), "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split("\t")
        if p[2] != "transcript": continue
        tx = re.search(r'transcript_id "([^"]+)"', p[8])
        if tx is None: continue
        c = p[0] if p[0].startswith("chr") else "chr" + p[0]
        rows.append((tx.group(1), c, int(p[3]), int(p[4])))
T = pd.DataFrame(rows, columns=["tid", "tchr", "ts", "te"])
print(f"GTF transcripts: {len(T)}; tid head3: {T.tid.head(3).tolist()}")

# ---- 3) .sf read-in, two-sided normalization (strip _N suffix + strip version) ----
sf = pd.read_csv(os.path.join(RAW, "GSM4513967_BRD3179.sf.txt.gz"), sep="\t")
sf["tid_norm"] = (sf["Name"].str.replace(r"_\d+$", "", regex=True)
                            .str.replace(r"\.\d+$", "", regex=True))
T["tid_norm"] = (T["tid"].str.replace(r"_\d+$", "", regex=True)
                          .str.replace(r"\.\d+$", "", regex=True))
print("Two-sided ID head-5 comparison: sf=", sf.tid_norm.head(5).tolist(),
      " gtf=", T.tid_norm.head(5).tolist())
m = sf.merge(T, on="tid_norm", how="inner")
match_tpm = m["TPM"].sum() / sf["TPM"].sum()
print(f"TPM-weighted match rate: {match_tpm:.4f} (gate >= 0.90)")
assert match_tpm >= 0.90, "match rate below gate, aborting"

# ---- 4) transcript -> bin (binID = bed row) ----
bed_idx = {c: g.sort_values("bstart").reset_index(drop=True)
           for c, g in bed.groupby("chrom")}
def tx_bin(row):
    g = bed_idx.get(row.tchr)
    if g is None: return -1
    st, bx = g.bstart.values, g.bin.values
    mid = (row.ts - 1 + row.te) // 2
    j = np.searchsorted(st, mid) - 1
    if 0 <= j < len(st) and st[j] <= mid < g.bend.values[j]: return int(bx[j])
    return -1
m["bin"] = [tx_bin(r) for r in m.itertuples()]
m = m[m["bin"] > 0]
print(f"Transcripts falling into bins: {len(m)}")

# ---- 5) bin-level aggregation: TPM-weighted mean of log2TPM ----
m["wc"] = m["TPM"].clip(lower=1e-6)
m["lv"] = np.log2(m["TPM"] + 1)
agg = m.groupby("bin").apply(
    lambda g: np.average(g["lv"], weights=g["wc"]), include_groups=False)
E = pd.DataFrame({"bin": agg.index, "log2TPM": agg.values})
D = lab.merge(E, on="bin", how="inner")
print(f"labels ∩ expressed bins: {len(D)}")

# ---- 6) registered tests ----
med = D.groupby("cluster")["log2TPM"].median()
print("\nBin-level median log2TPM per label:")
print(med.round(4).to_string())
order_exp = ["A1", "A2", "B1", "B2", "B3", "B4"]
mono = all(med[order_exp[i]] >= med[order_exp[i + 1]] for i in range(5))
Avals = D.loc[D.cluster.isin(["A1", "A2"]), "log2TPM"]
Bvals = D.loc[D.cluster.isin(["B1", "B2", "B3", "B4"]), "log2TPM"]
stat, p = mannwhitneyu(Avals, Bvals, alternative="greater")
print(f"\nMonotonic gradient (A1>=A2>=B1>=B2>=B3>=B4): {'yes' if mono else 'NO'}")
print(f"A cluster vs B cluster MW one-sided: p={p:.2e} (gate <0.001) "
      f"-> {'PASS' if p < 0.001 else 'FAIL'}")
r = spearmanr(D["PC1"], D["log2TPM"]).statistic
print(f"PC1 vs log2TPM spearman: {r:.3f}")
D.to_csv(os.path.join(RES, "a28a_label_expr.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([{**{f"med_{k}": round(med[k], 4) for k in order_exp},
               "monotone": mono, "MW_p": f"{p:.2e}", "PC1_rho": round(r, 3),
               "match_tpm": round(match_tpm, 4)}]).to_csv(
    os.path.join(RES, "a28a_expr_tests.csv"), index=False, encoding="utf-8-sig")
print("Saved: a28a_label_expr.csv / a28a_expr_tests.csv")
