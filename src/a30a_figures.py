# -*- coding: utf-8 -*-
# a30a_figures.py — first leg of the a30x figure set (purely descriptive, zero
#   hypothesis tests, per the §16.16 todo)
# ===== Panel registration (declared in script header) =====
# A embedding: a24a_layer1_embedding.csv PC1/PC2, colored N blue / T red /
#   C grey, sb replicate pairs linked
# B per-chr |rho|: a29e_per_chr_mean_rho.csv bar chart, chrY highlighted red
#   (collapse signature made visible)
# C switch heatmap: a24c_diff_*.bedgraph x3 (two patients' T/N + 5AZA positive
#   control) ordered by bed binID, symmetric scale vmin/vmax = ±2 (the
#   high-confidence threshold 0.94 sits inside the range; 5AZA median 0.647
#   visible), missing bins = grey
# Lessons applied: print then assert the column names of any cited artifact
#   (§16.11 lesson 3); all in-figure text in English (font safety)
# Output: results/figs/a30a_embedding.png / a30b_per_chr_rho.png /
#   a30c_switch_heatmap.png
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
RAW  = os.path.join(ROOT, "data", "raw")
FIGS = os.path.join(RES, "figs")
os.makedirs(FIGS, exist_ok=True)
CELL = ("FHC", "HCT116", "LS174T", "RKO", "SW480", "Wi38")

def stype(s):
    base = s.replace("_100000", "").replace("-sb", "")
    return "celline" if base.startswith(CELL) else ("normal" if base.endswith("N") else "tumor")

def sbase(s):
    return s.replace("_100000", "").replace("-sb", "")

# ============ Panel A: embedding ============
emb = pd.read_csv(os.path.join(RES, "a24a_layer1_embedding.csv"))
print(f"[A] embedding columns: {emb.columns.tolist()}")
assert {"sample", "PC1", "PC2"}.issubset(emb.columns), "embedding columns unexpected, stopping"
emb["type"] = emb["sample"].map(stype)
emb["base"] = emb["sample"].map(sbase)
CMAP = {"normal": "#1f77b4", "tumor": "#d62728", "celline": "#999999"}
MK = {"normal": "o", "tumor": "^", "celline": "s"}
fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
for t in ("normal", "tumor", "celline"):
    sub = emb[emb["type"] == t]
    ax.scatter(sub["PC1"], sub["PC2"], c=CMAP[t], marker=MK[t], s=45,
               label=f"{t} (n={len(sub)})", alpha=0.85, edgecolors="k", linewidths=0.4)
# sb technical-replicate pairs (visualizing the sb axis that drives the
# high-correlation source ordering)
sb_pairs = emb[emb["base"].duplicated(keep=False)].sort_values("base")
for b, g in sb_pairs.groupby("base"):
    if len(g) == 2:
        ax.plot(g["PC1"], g["PC2"], "-", c="#2ca02c", lw=1.0, alpha=0.8, zorder=0)
ax.plot([], [], "-", c="#2ca02c", label="same-patient technical reps (sb)")
ax.set_xlabel("PC1 (25.8% var)")
ax.set_ylabel("PC2")
ax.set_title("Sample embedding of per-bin compartment vectors (40 samples)")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "a30a_embedding.png"), bbox_inches="tight")
plt.close(fig)
print("[A] a30a_embedding.png done")

# ============ Panel B: per-chr |rho| ============
pc = pd.read_csv(os.path.join(RES, "a29e_per_chr_mean_rho.csv"))
print(f"[B] per-chr columns: {pc.columns.tolist()}")
assert {"chrom", "mean_abs_rho"}.issubset(pc.columns), "per-chr columns unexpected, stopping"
def chrkey(c):
    c = c.replace("chr", "")
    return 23 if c in ("X", "Y", "M") else int(c)
pc = pc.sort_values("chrom", key=lambda s: s.map(chrkey)).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
colors = ["#d62728" if c == "chrY" else "#1f77b4" for c in pc["chrom"]]
ax.bar(range(len(pc)), pc["mean_abs_rho"], color=colors)
ax.set_xticks(range(len(pc)))
ax.set_xticklabels(pc["chrom"], rotation=60, fontsize=7)
ax.set_ylabel("Mean pairwise |Spearman rho|")
ax.set_title("Cross-sample compartment concordance per chromosome\n"
             "(37 deduplicated samples; chrY collapse = expected, sex-composition artifact)")
auto = pc.loc[pc["chrom"] != "chrY", "mean_abs_rho"].mean()
ax.axhline(auto, ls="--", c="k", lw=0.8)
ax.text(len(pc) - 0.5, auto, f" auto mean={auto:.3f}", fontsize=8, va="bottom", ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "a30b_per_chr_rho.png"), bbox_inches="tight")
plt.close(fig)
print("[B] a30b_per_chr_rho.png done")

# ============ Panel C: switch heatmap ============
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
grids = {(r.chrom, r.start): i for i, r in enumerate(bed.itertuples())}
CONTRASTS = [
    ("BRD3187 T vs N",    "a24c_diff_BRD3187_vs_BRD3187N.bedgraph"),
    ("BRD3462 T vs N",    "a24c_diff_BRD3462_vs_BRD3462N.bedgraph"),
    ("HCT116 5AZA vs 0h", "a24c_diff_HCT116_5AZA_vs_0h.bedgraph"),
]
H = np.full((len(CONTRASTS), len(bed)), np.nan)
for k, (lab, fn) in enumerate(CONTRASTS):
    fp = os.path.join(RES, fn)
    with open(fp, "r", encoding="utf-8") as f:
        head = [next(f).rstrip("\n") for _ in range(2)]
    print(f"[C] {fn} first two lines: {head}")
    df = pd.read_csv(fp, sep=None, engine="python", header=None)
    if df.shape[1] == 4 and str(df.iloc[0, 0]).startswith("chr"):
        pass  # headerless 4 columns
    else:
        df = pd.read_csv(fp, sep=None, engine="python")  # headered fallback
    assert {"chrom", "start"}.issubset(df.columns) or df.shape[1] == 4, \
        f"{fn} column layout unclear, stopping"
    df.columns = range(df.shape[1])
    hits = 0
    for r in df.itertuples(index=False):
        key = (r[0], int(r[1]))
        if key in grids:
            H[k, grids[key]] = float(r[3]); hits += 1
    print(f"[C] {lab}: matched {hits}/{len(df)} rows ({hits/len(df)*100:.1f}%)")
    assert hits / len(df) > 0.8, f"{fn} alignment to bed too low, coordinate convention suspect, stopping"
fig, ax = plt.subplots(figsize=(11, 3.4), dpi=300)
im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
ax.set_yticks(range(len(CONTRASTS)))
ax.set_yticklabels([c[0] for c in CONTRASTS], fontsize=9)
ax.set_xlabel("Genomic bins (100 kb, hg19, ordered chr1 -> chrY)")
ax.set_title("Standardized compartment shift (per-chr std-normalized PC1 diff)\n"
             "grey = no coverage; |diff|>0.94 = high-confidence switch (a24c threshold)")
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("diff (std units)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "a30c_switch_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("[C] a30c_switch_heatmap.png done")

# ============ manifest ============
pd.DataFrame({
    "panel":  ["A_embedding", "B_per_chr_rho", "C_switch_heatmap"],
    "png":    ["a30a_embedding.png", "a30b_per_chr_rho.png", "a30c_switch_heatmap.png"],
    "source": ["a24a_layer1_embedding.csv", "a29e_per_chr_mean_rho.csv",
               "a24c_diff_*.bedgraph x3"],
    "note":   ["PC1/PC2, type colored, sb pairs linked",
               "chrY highlighted; auto mean line",
               "vmin/vmax=+/-2 std units; bed binID-ordered"],
}).to_csv(os.path.join(RES, "a30a_fig_manifest.csv"), index=False, encoding="utf-8-sig")
print("\nSaved: results/figs/ x3 + a30a_fig_manifest.csv")
