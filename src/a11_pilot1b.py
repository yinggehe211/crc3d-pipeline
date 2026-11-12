# -*- coding: utf-8 -*-
# a11_pilot1b.py - Pilot 1b: per-chromosome PC1 sign anchoring (Xu CRC-03-N as reference) + N/T switches (correlation alignment + confidence filter)
# Input: results/pilot/pilot1_*_genome_PC1.csv (written by a10; no matrix re-decomposition needed)
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "pilot")
XU_DIR = os.path.join(ROOT, "Xu_repo", "CRC_metastasis_code-main", "Fig3b", "Input_compartment")


def load_xu(sample):
    xu = pd.read_csv(os.path.join(XU_DIR, sample, "compartment.txt"), sep="\t")
    xu.columns = [c.strip() for c in xu.columns]
    xu["score_num"] = pd.to_numeric(xu.score, errors="coerce")
    xu = xu[xu.score_num.notna()].copy()
    xu["bin500"] = xu.start.astype(int) - 1  # a10 adjudicated 1-based, offset -1
    return xu[["seqnames", "bin500", "score_num"]].rename(columns={"seqnames": "chr"})


ours = {}
for tag in ("BRD3179N_normal", "BRD3179_tumor"):
    ours[tag] = pd.read_csv(os.path.join(OUT, f"pilot1_{tag}_genome_PC1.csv"))

stats = []

# ---- 1) per-chromosome sign anchoring: our N vs Xu CRC-03-N ----
print("===== per-chromosome sign anchoring (our N vs Xu CRC-03-N, 500 kb) =====")
xn = load_xu("CRC-03-N-HiC")
n_agg = (ours["BRD3179N_normal"].assign(bin500=lambda d: (d.start // 500000) * 500000)
         .groupby(["chr", "bin500"]).PC1.mean().reset_index())
n_aligned_parts = []
for chrom, g in n_agg.groupby("chr"):
    gx = xn[xn.chr == chrom]
    mg = g.merge(gx, on=["chr", "bin500"], how="inner")
    if len(mg) < 20:
        continue
    a_direct = ((mg.PC1 > 0) == (mg.score_num > 0)).mean()
    sign = 1 if a_direct >= 0.5 else -1
    a = max(a_direct, 1 - a_direct)
    n_aligned_parts.append(sign * g[["bin500", "PC1"]].assign(chr=chrom))
    stats.append({"item": "chr_sign_anchor_N", "chr": chrom, "sign": sign,
                  "agree": round(float(a), 4), "n": len(mg)})
    print(f"  {chrom}: sign={sign:+d}  aligned agreement {a:.1%} (n={len(mg)})")
n_aligned = pd.concat(n_aligned_parts)
a_all = n_aligned.assign(pc=lambda d: d.PC1).merge(
    xn, on=["chr", "bin500"])
a_all["pc"] = a_all["pc"] if "pc" in a_all else a_all.PC1
# re-merge (PC1 column-name conflict handling)
a_all = pd.concat(n_aligned_parts).rename(columns={"PC1": "PC1_sig"})
a_all["bin500"] = a_all["bin500"].astype(int)
a_all = a_all.merge(xn, on=["chr", "bin500"], how="inner")
agree_genome = ((a_all.PC1_sig > 0) == (a_all.score_num > 0)).mean()
print(f"\n  * genome-wide agreement (after per-chromosome anchoring, vs Xu-N): {agree_genome:.2%} "
      f"(n={len(a_all)}) gate >= 80%")
stats.append({"item": "genome_agree_N_vs_XuN_aligned", "agree": round(float(agree_genome), 4),
              "n": len(a_all)})

# the other two Xu samples, same protocol (reference values)
for smp in ("CRC-03-T-HiC", "CRC-03-M-HiC"):
    xs = load_xu(smp)
    mg = a_all.merge(xs, on=["chr", "bin500"], suffixes=("_N", "_s"))
    a = ((mg.PC1_sig > 0) == (mg.score_num_s > 0)).mean()
    print(f"  vs Xu {smp}: {a:.2%} (n={len(mg)})")
    stats.append({"item": f"genome_agree_vs_{smp}", "agree": round(float(a), 4), "n": len(mg)})

# ---- 2) N/T switches: T sign aligned to N per chromosome by correlation (no Xu) + confidence filter ----
print("\n===== N/T compartment switches (per-chromosome correlation alignment + confidence filter) =====")
nn = ours["BRD3179N_normal"][["chr", "bin0", "start", "PC1"]].rename(columns={"PC1": "PC1_N"})
tt = ours["BRD3179_tumor"][["chr", "bin0", "PC1"]].rename(columns={"PC1": "PC1_T"})
nt = nn.merge(tt, on=["chr", "bin0"])
parts = []
for chrom, g in nt.groupby("chr"):
    r = np.corrcoef(g.PC1_N, g.PC1_T)[0, 1]
    s = 1 if r >= 0 else -1
    g = g.copy()
    g["PC1_T"] = g.PC1_T * s
    parts.append(g)
    stats.append({"item": "NT_corr_align", "chr": chrom, "pearson_r": round(float(r), 3),
                  "sign": s})
nt2 = pd.concat(parts)
r_med = np.median([abs(s["pearson_r"]) for s in stats if s["item"] == "NT_corr_align"])
print(f"  N/T same-chromosome PC1 correlation |r| median: {r_med:.3f} (cross-sample conservation of the compartment skeleton)")

q25n, q25t = nt2.PC1_N.abs().quantile(0.25), nt2.PC1_T.abs().quantile(0.25)
nt2["comp_N"] = np.where(nt2.PC1_N > 0, "A", "B")
nt2["comp_T"] = np.where(nt2.PC1_T > 0, "A", "B")
nt2["switch_raw"] = nt2.comp_N != nt2.comp_T
nt2["switch_conf"] = (nt2.switch_raw & (nt2.PC1_N.abs() > q25n)
                      & (nt2.PC1_T.abs() > q25t))
n_raw, n_conf = int(nt2.switch_raw.sum()), int(nt2.switch_conf.sum())
ba = int(((nt2.comp_N == "B") & (nt2.comp_T == "A") & nt2.switch_conf).sum())
ab = int(((nt2.comp_N == "A") & (nt2.comp_T == "B") & nt2.switch_conf).sum())
print(f"  raw switches: {n_raw} bins ({n_raw/len(nt2):.1%})")
print(f"  * high-confidence switches (|PC1|>Q25 two-sided): {n_conf} bins ({n_conf/len(nt2):.1%})"
      f" | B→A {ba} | A→B {ab}")
nt2[nt2.switch_conf].to_csv(os.path.join(OUT, "pilot1b_NT_switch_confident.csv"),
                            index=False, encoding="utf-8")
stats.append({"item": "NT_switch", "raw": n_raw, "confident": n_conf,
              "B2A": ba, "A2B": ab, "total_bins": len(nt2)})

pd.DataFrame(stats).to_csv(os.path.join(OUT, "pilot1b_summary.csv"),
                           index=False, encoding="utf-8")
print("\n=== Pilot 1b written: switch_confident + summary ===")
