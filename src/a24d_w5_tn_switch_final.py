# -*- coding: utf-8 -*-
# a24d_w5_tn_switch_final.py — final A/B switch statistics (full-length mask, both T/N pairs)
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
aligned = np.load(os.path.join(RES, "a24a_layer1_aligned.npy"))
samples = pd.read_csv(os.path.join(RES, "a22f_sample_index.csv"))["sample"].tolist()
smap = {s: i for i, s in enumerate(samples)}
bed = pd.read_csv(os.path.join(ROOT, "data", "raw", "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
bed["bin"] -= 1
bed = bed[bed["bin"] < aligned.shape[1]].sort_values("bin")
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

# noise threshold (full length)
nz = scaled_diff("BRD3462N", "BRD3462N-sb")
thr = float(np.percentile(np.abs(nz[nz != 0]), 99))
print(f"Noise 99th-percentile threshold: {thr:.3f}\n")

def switch_report(t, n, d):
    T, N = get(t), get(n)
    valid = (T != 0) & (N != 0)
    hi = np.abs(d) > thr
    flip = valid & hi & (np.sign(T) != np.sign(N))
    a_gain = int((flip & (T > 0)).sum())   # N:B -> T:A
    b_gain = int((flip & (T < 0)).sum())   # N:A -> T:B
    stableA = int((valid & (T > 0) & (N > 0)).sum())
    stableB = int((valid & (T < 0) & (N < 0)).sum())
    print(f"--- {t} vs {n} ---")
    print(f"  switch bins (above threshold + sign flip): {int(flip.sum())} "
          f"| B->A (A compartment gained)={a_gain} | A->B (B compartment gained)={b_gain} "
          f"| ratio A/B={a_gain/max(b_gain,1):.2f}")
    print(f"  stable compartments: stableA={stableA} stableB={stableB} "
          f"(of {int(valid.sum())} valid bins)")
    return {"comparison": f"{t} vs {n}", "switch_bins": int(flip.sum()),
            "B_to_A": a_gain, "A_to_B": b_gain,
            "stable_A": stableA, "stable_B": stableB}, flip

rows, flips = [], {}
for t, n in [("BRD3187", "BRD3187N"), ("BRD3462", "BRD3462N")]:
    r, flip = switch_report(t, n, scaled_diff(t, n))
    rows.append(r); flips[t] = flip

# chromosome distribution of BRD3187 switches
bed_f = bed.reset_index(drop=True).copy()
bed_f["flip3187"] = flips["BRD3187"]
per = bed_f[bed_f.flip3187].groupby("chrom").size().sort_values(ascending=False)
tot = bed_f.groupby("chrom").size()
print("\n=== BRD3187 switch bins chromosome distribution (Top10) ===")
for c in per.head(10).index:
    print(f"{c}: {per[c]} / {tot[c]} ({per[c]/tot[c]*100:.1f}%)")

# write out: switch-site list (BRD3187, all flip sites with bed coordinates and diff values)
d3187 = scaled_diff("BRD3187", "BRD3187N")
out = bed_f[bed_f.flip3187][["chrom", "start", "end"]].copy()
out["diff_scaled"] = d3187[bed_f.flip3187.values].round(4)
out["T_PC1"] = get("BRD3187")[bed_f.flip3187.values].round(5)
out["N_PC1"] = get("BRD3187N")[bed_f.flip3187.values].round(5)
out["direction"] = np.where(out["T_PC1"] > 0, "B->A", "A->B")
out.to_csv(os.path.join(RES, "a24d_BRD3187_switch_bins.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(os.path.join(RES, "a24d_switch_summary.csv"), index=False, encoding="utf-8-sig")
print(f"\nSaved: a24d_BRD3187_switch_bins.csv ({len(out)} switch sites, "
      f"with genomic coordinates + direction), a24d_switch_summary.csv")
