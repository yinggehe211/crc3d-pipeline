# -*- coding: utf-8 -*-
# a24b_w5_tn_diff.py — T/N differential compartment scoring (+ noise floor + drug control) -> bedgraph
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
aligned = np.load(os.path.join(RES, "a24a_layer1_aligned.npy"))     # (40, 30968)
samples = pd.read_csv(os.path.join(RES, "a22f_sample_index.csv"))["sample"].tolist()
smap = {s: i for i, s in enumerate(samples)}
bed = pd.read_csv(os.path.join(ROOT, "data", "raw", "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "start", "end", "bin"])
bed["bin"] -= 1
bed = bed[bed["bin"] < aligned.shape[1]].sort_values("bin")
coords = bed.set_index("bin")[["chrom", "start", "end"]]

def get(name): return aligned[smap[name + "_100000"]].astype(float)

def diff_stats(d, label):
    ok = (d != 0); dd = d[ok]
    return {"comparison": label, "valid_bins": int(ok.sum()),
            "frac_diff_gt0_A_gain": round(float((dd > 0).mean()), 3),
            "abs_diff_gt_0.3": int((np.abs(dd) > 0.3).sum()),
            "abs_diff_gt_0.5": int((np.abs(dd) > 0.5).sum()),
            "median_abs_diff": round(float(np.median(np.abs(dd))), 3)}

def save_bedgraph(d, fname):
    ok = d != 0
    df = coords[ok].copy()
    df["score"] = d[ok].round(4)
    p = os.path.join(RES, fname)
    df.to_csv(p, sep="\t", header=False, index=False)
    return p

rows, files = [], []

# ---- 1) noise floor: BRD3462N vs BRD3462N-sb ----
noise = np.abs(get("BRD3462N") - get("BRD3462N-sb"))
rows.append(diff_stats(noise, "noise_floor_BRD3462N_vs_BRD3462Nsb"))
print(f"Noise floor: |diff| median={np.median(np.abs(noise[noise!=0])):.3f} "
      f"bins with |diff|>0.3={int((noise>0.3).sum())}")

# ---- 2) paired T/N ----
for t, n in [("BRD3187", "BRD3187N"), ("BRD3462", "BRD3462N")]:
    d = get(t) - get(n)
    rows.append(diff_stats(d, f"T/N: {t} vs {n}"))
    # 'high-confidence' switch sites: exceeding the noise floor by 3x
    hi = np.abs(d) > max(3 * np.median(np.abs(noise[noise != 0])), 0.3)
    rows[-1]["high_conf_switch_bins"] = int(hi[get(t) * get(n) != 0].sum())
    files.append(save_bedgraph(d, f"a24b_diff_{t}_vs_{n}.bedgraph"))

# ---- 3) drug control: HCT116 5AZA vs 0h (replicate means) ----
d5 = ((get("HCT116-5AZA-24h-1") + get("HCT116-5AZA-24h-2")) / 2
      - (get("HCT116-0h-1") + get("HCT116-0h-2")) / 2)
rows.append(diff_stats(d5, "drug_HCT116_5AZA24h_vs_0h"))
files.append(save_bedgraph(d5, "a24b_diff_HCT116_5AZA_vs_0h.bedgraph"))

# ---- 4) chromosome-level summary: distribution of switch bins for BRD3187 T/N ----
d = get("BRD3187") - get("BRD3187N")
ok = d != 0
thr = max(3 * np.median(np.abs(noise[noise != 0])), 0.3)
hi = ok & (np.abs(d) > thr)
cc = coords.copy(); cc["d"] = d; cc["hi"] = hi
per_chrom = cc[cc.hi].groupby("chrom").size().sort_values(ascending=False)
tot_bins = cc.groupby("chrom").size()
print("\n=== Chromosome distribution of high-confidence switch bins (Top10) ===")
for c in per_chrom.head(10).index:
    print(f"{c}: {per_chrom[c]} / {tot_bins[c]} bins "
          f"({per_chrom[c]/tot_bins[c]*100:.1f}%)")
S = pd.DataFrame(rows)
S.to_csv(os.path.join(RES, "a24b_diff_summary.csv"), index=False, encoding="utf-8-sig")
print("\n=== Differential compartment summary ===")
print(S.to_string(index=False))
print(f"\n{len(files)} bedgraph files written: {[os.path.basename(f) for f in files]}")
print("(loadable directly in IGV / WashU; value = PC1_T - PC1_N, positive = A gain, negative = B gain)")
