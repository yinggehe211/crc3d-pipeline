# -*- coding: utf-8 -*-
# a24c_w5_tn_diff_scaled.py — differential compartment scoring (scale-corrected):
#   within-chromosome standardization + noise-floor threshold
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
coords = bed.set_index("bin")[["chrom", "start", "end"]]
chrom_arr = np.full(aligned.shape[1], "NA", dtype=object)
chrom_arr[bed["bin"].values] = bed["chrom"].values
chroms = [c for c in bed["chrom"].unique() if c != "chrY"]

# ---- per-chromosome amplitude scale: median across samples of within-chr std ----
scale = {}
for c in chroms:
    idx = np.where(chrom_arr == c)[0]
    ss = [np.std(aligned[i, idx][aligned[i, idx] != 0]) for i in range(len(samples))]
    ss = [s for s in ss if s > 0]
    scale[c] = float(np.median(ss)) if ss else np.nan
print("=== PC1 raw amplitude diagnostic (typical per-chr std, expected 0.01-0.1) ===")
print({c: round(v, 4) for c, v in list(scale.items())[:6]}, "...")

def get(name): return aligned[smap[name + "_100000"]].astype(float)

def scaled_diff(a, b):
    """Standardized diff: in units of each chromosome's typical PC1 amplitude"""
    d = np.zeros(aligned.shape[1])
    for c in chroms:
        idx = np.where(chrom_arr == c)[0]
        d[idx] = (aligned[smap[a + "_100000"], idx] - aligned[smap[b + "_100000"], idx]) / scale[c]
    return d

# ---- noise floor (sb replicate) ----
nz = scaled_diff("BRD3462N", "BRD3462N-sb")
ok = nz != 0
thr = float(np.percentile(np.abs(nz[ok]), 99))
print(f"\nNoise floor: |diff| median={np.median(np.abs(nz[ok])):.3f} "
      f"99th-percentile threshold={thr:.3f} max={np.abs(nz[ok]).max():.3f}")

def summarize(d, label, save=None):
    ok = d != 0; dd = d[ok]
    hi = np.abs(dd) > thr
    row = {"comparison": label, "valid_bins": int(ok.sum()),
           "median_abs_diff_scaled": round(float(np.median(np.abs(dd))), 3),
           f"abs_diff_gt_{thr:.2f}": int(hi.sum()),
           "abs_diff_gt_0.5": int((np.abs(dd) > 0.5).sum()),
           "abs_diff_gt_1.0": int((np.abs(dd) > 1.0).sum())}
    if save:
        df = coords[ok].copy(); df["score"] = dd.round(4)
        df.to_csv(os.path.join(RES, save), sep="\t", header=False, index=False)
    return row, ok, hi, d

rows, beds = [], []
for t, n, tag in [("BRD3187", "BRD3187N", "T/N"), ("BRD3462", "BRD3462N", "T/N")]:
    r, ok, hi, d = summarize(scaled_diff(t, n), f"{tag}: {t} vs {n}",
                             f"a24c_diff_{t}_vs_{n}.bedgraph")
    rows.append(r); beds.append((t, n, d, ok, hi))
# the drug-control diff must be standardized separately (get() returns raw scale)
d5 = np.zeros(aligned.shape[1])
h5a = (get("HCT116-5AZA-24h-1") + get("HCT116-5AZA-24h-2")) / 2
h0  = (get("HCT116-0h-1") + get("HCT116-0h-2")) / 2
for c in chroms:
    idx = np.where(chrom_arr == c)[0]
    d5[idx] = (h5a[idx] - h0[idx]) / scale[c]
r5, ok5, hi5, _ = summarize(d5, "drug_HCT116_5AZA24h_vs_0h")
rows[-1] = r5

S = pd.DataFrame(rows)
S.to_csv(os.path.join(RES, "a24c_diff_summary.csv"), index=False, encoding="utf-8-sig")
print("\n=== Differential compartment summary (standardized units, threshold = noise 99th pct) ===")
print(S.to_string(index=False))

# ---- A/B switch statistics: PC1 sign flip + above threshold, for the T/N pairs ----
print("\n=== A<->B switch bins (sign flip + above noise threshold) ===")
for t, n, d, ok, hi in beds:
    T, N = get(t), get(n)
    flip = ok & (np.sign(T) != np.sign(N)) & hi
    a_gain = int((flip & (T > 0) & (N < 0)).sum())
    b_gain = int((flip & (T < 0) & (N > 0)).sum())
    print(f"{t} vs {n}: total switches={int(flip.sum())} "
          f"B->A (A gain)={a_gain} A->B (B gain)={b_gain}")

# ---- chromosome distribution of BRD3187 switches ----
t, n, d, ok, hi = beds[0]
T, N = get(t), get(n)
flip = ok & (np.sign(T) != np.sign(N)) & hi
cc = coords.copy(); cc["flip"] = flip
per = cc[cc.flip].groupby("chrom").size().sort_values(ascending=False)
tot = cc.groupby("chrom").size()
print("\n=== BRD3187 switch bins chromosome distribution (Top10) ===")
for c in per.head(10).index:
    print(f"{c}: {per[c]} / {tot[c]} ({per[c]/tot[c]*100:.1f}%)")
