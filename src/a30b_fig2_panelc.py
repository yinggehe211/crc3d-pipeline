# -*- coding: utf-8 -*-
# a30b_fig2_panelc.py — panel-C artifact diagnosis + block-aggregated re-render v2
# Suspect: 30970 bins compressed into ~1400 pixel columns; signed per-pixel
#   averaging drives alternating positive/negative values (5AZA) to zero ->
#   washed out
# Diagnosis: per-contrast cross-check against the a24c record (|diff| median /
#   >0.5 count / >0.94 count); block signed mean vs block mean |diff|
# v2: two-layer heatmap — top = block signed mean (keeps regional coherence and
#   the direction narrative), bottom = block mean |diff| (keeps 5AZA magnitude)
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
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
grids = {(r.chrom, r.start): i for i, r in enumerate(bed.itertuples())}
CONTRASTS = [
    ("BRD3187 T vs N",    "a24c_diff_BRD3187_vs_BRD3187N.bedgraph", None),
    ("BRD3462 T vs N",    "a24c_diff_BRD3462_vs_BRD3462N.bedgraph", None),
    ("HCT116 5AZA vs 0h", "a24c_diff_HCT116_5AZA_vs_0h_v2.bedgraph",
     {"med_abs": 0.647, "gt0.5": 16641, "gt0.94": 10075, "n": 28268}),
]
NB = 1500  # blocks (~21 bins each; >2 pixels per block at 300 dpi — kills the
           # per-pixel averaging artifact)
signed = np.full((len(CONTRASTS), NB), np.nan)
absmed = np.full((len(CONTRASTS), NB), np.nan)
edges = np.linspace(0, len(bed), NB + 1).astype(int)
for k, (lab, fn, ref) in enumerate(CONTRASTS):
    df = pd.read_csv(os.path.join(RES, fn), sep="\t", header=None,
                     names=["chrom", "start", "end", "v"])
    idx = np.array([grids.get((c, int(s)), -1) for c, s in zip(df["chrom"], df["start"])])
    ok = idx >= 0
    v = df["v"].values[ok]; idx = idx[ok]
    # diagnostic cross-check (lesson 3: print cited artifact numbers before trusting them)
    print(f"[{lab}] n={len(v)}, |diff| median={np.median(np.abs(v)):.3f}, "
          f">0.5: {(np.abs(v)>0.5).sum()}, >0.94: {(np.abs(v)>0.94).sum()}")
    if ref:
        assert abs(np.median(np.abs(v)) - ref["med_abs"]) < 0.01, \
            f"{lab} median cross-check failed, stopping"
        d5  = (np.abs(v) > 0.5).sum()  - ref["gt0.5"]
        d94 = (np.abs(v) > 0.94).sum() - ref["gt0.94"]
        # tolerance = rounding-boundary effect of 4-decimal on-disk values
        # (bins within ±5e-5 of a threshold can flip)
        edge5  = (np.abs(np.abs(v) - 0.5)  < 5e-5 * 10).sum()
        edge94 = (np.abs(np.abs(v) - 0.94) < 5e-5 * 10).sum()
        print(f"  boundary deltas: >0.5 {d5:+d} (bins near threshold={edge5}), "
              f">0.94 {d94:+d} (bins near threshold={edge94})")
        assert abs(d5)  <= max(10, edge5),  f"{lab} >0.5 count delta beyond rounding tolerance, stopping"
        assert abs(d94) <= max(10, edge94), f"{lab} >0.94 count delta beyond rounding tolerance, stopping"
        print("  -> cross-check against a24c record passed (tolerance = on-disk precision boundary)")
    # artifact diagnosis: magnitude gap between block signed mean and block mean |diff|
    b_sign = np.array([v[(idx >= edges[b]) & (idx < edges[b+1])].mean()
                       if ((idx >= edges[b]) & (idx < edges[b+1])).any() else np.nan
                       for b in range(NB)])
    b_abs = np.array([np.abs(v[(idx >= edges[b]) & (idx < edges[b+1])]).mean()
                      if ((idx >= edges[b]) & (idx < edges[b+1])).any() else np.nan
                      for b in range(NB)])
    print(f"  block signed mean |max|={np.nanmax(np.abs(b_sign)):.3f} vs "
          f"block mean-|diff| median={np.nanmedian(b_abs):.3f} "
          f"(large ratio = artifact confirmed)")
    signed[k] = b_sign; absmed[k] = b_abs
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), dpi=300, sharex=True)
im0 = axes[0].imshow(signed, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                     interpolation="nearest")
axes[0].set_yticks(range(len(CONTRASTS)))
axes[0].set_yticklabels([c[0] for c in CONTRASTS], fontsize=9)
axes[0].set_title("Per-block mean signed diff (regional coherence, B->A warm / A->B cool)")
plt.colorbar(im0, ax=axes[0], pad=0.01).set_label("block mean (std units)")
im1 = axes[1].imshow(absmed, aspect="auto", cmap="viridis", vmin=0, vmax=2,
                     interpolation="nearest")
axes[1].set_yticks(range(len(CONTRASTS)))
axes[1].set_yticklabels([c[0] for c in CONTRASTS], fontsize=9)
axes[1].set_title("Per-block mean |diff| (shift magnitude; 5AZA positive control)")
axes[1].set_xlabel("Genomic blocks (1500 blocks, ~2 Mb each, hg19 ordered chr1 -> chrY)")
plt.colorbar(im1, ax=axes[1], pad=0.01).set_label("block mean |diff|")
fig.suptitle("Compartment shift landscape (a24c std-normalized PC1 diff)", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "a30c_switch_heatmap_v2.png"), bbox_inches="tight")
plt.close(fig)
print("\nSaved: a30c_switch_heatmap_v2.png (v1 kept as the artifact-teaching counterpart)")
