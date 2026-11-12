# -*- coding: utf-8 -*-
# a13_pilot1d.py - Pilot 1d: continuous-value validation of compartment change (primary test) + 100-kb switch overlap (supplementary)
# Primary test: dPC1 (ours, T-N) vs dscore (Xu, CRC-03 T-N) correlation genome-wide / per chromosome
# Supplementary: raw switch set (unfiltered) vs Xu switches, OR at 100-kb resolution
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, fisher_exact

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "pilot")
XU_DIR = os.path.join(ROOT, "Xu_repo", "CRC_metastasis_code-main", "Fig3b", "Input_compartment")


def load_xu(sample):
    xu = pd.read_csv(os.path.join(XU_DIR, sample, "compartment.txt"), sep="\t")
    xu.columns = [c.strip() for c in xu.columns]
    xu["score_num"] = pd.to_numeric(xu.score, errors="coerce")
    xu = xu[xu.score_num.notna()].copy()
    xu["bin500"] = xu.start.astype(int) - 1  # 1-based offset
    return xu[["seqnames", "bin500", "score_num"]].rename(columns={"seqnames": "chr"})


def load_ours(tag):
    return pd.read_csv(os.path.join(OUT, f"pilot1_{tag}_genome_PC1.csv"))


n_raw, t_raw = load_ours("BRD3179N_normal"), load_ours("BRD3179_tumor")
xn, xt = load_xu("CRC-03-N-HiC"), load_xu("CRC-03-T-HiC")
xu = xn.merge(xt, on=["chr", "bin500"], suffixes=("_N", "_T"))
xu["dscore"] = xu.score_num_T - xu.score_num_N

# ---- sign unification: N anchored to Xu-N (per chromosome), T aligned to N by correlation (per chromosome) ----
stats = []
n_sig_parts, t_sig_parts = [], []
for chrom in sorted(n_raw.chr.unique()):
    gn = n_raw[n_raw.chr == chrom]
    gt = t_raw[t_raw.chr == chrom]
    # N: vs Xu-N@500kb
    g5 = (gn.assign(bin500=(gn.start // 500000) * 500000)
          .groupby("bin500").PC1.mean().reset_index())
    mg = g5.merge(xn, on="bin500")
    if len(mg) >= 20:
        s_n = 1 if (((mg.PC1 > 0) == (mg.score_num > 0)).mean()) >= 0.5 else -1
    else:
        s_n = 1
    n_sig_parts.append(gn.assign(PC1=gn.PC1 * s_n))
    # T: correlation-aligned vs N @100kb
    mm = gn.merge(gt[["chr", "bin0", "PC1"]], on=["chr", "bin0"], suffixes=("_N", "_T"))
    r = np.corrcoef(mm.PC1_N, mm.PC1_T)[0, 1]
    s_t = s_n * (1 if r >= 0 else -1)
    t_sig_parts.append(gt.assign(PC1=gt.PC1 * s_t))
    stats.append({"item": "sign_unify", "chr": chrom, "sign_N": s_n, "sign_T": s_t,
                  "NT_r": round(float(r), 3)})

n_sig = pd.concat(n_sig_parts)[["chr", "bin0", "start", "PC1"]].rename(columns={"PC1": "PC1_N"})
t_sig = pd.concat(t_sig_parts)[["chr", "bin0", "PC1"]].rename(columns={"PC1": "PC1_T"})

# ---- primary test 1: dPC1 vs dscore @500kb ----
nt = n_sig.merge(t_sig, on=["chr", "bin0"])
nt["dPC1"] = nt.PC1_T - nt.PC1_N
nt["bin500"] = (nt.start // 500000) * 500000
d_ours = nt.groupby(["chr", "bin500"]).dPC1.mean().reset_index()
mg = d_ours.merge(xu, on=["chr", "bin500"], how="inner")
r_p, p_p = pearsonr(mg.dPC1, mg.dscore)
r_s, p_s = spearmanr(mg.dPC1, mg.dscore)
print("===== primary test: compartment-change continuous-value correlation (dPC1_ours vs dscore_Xu, T-N) =====")
print(f"  * genome-wide Pearson r={r_p:.3f} (p={p_p:.2e}) | Spearman rho={r_s:.3f} (p={p_s:.2e})  n={len(mg)}")
pos = 0
for chrom, g in mg.groupby("chr"):
    if len(g) >= 50:
        r_c, p_c = pearsonr(g.dPC1, g.dscore)
        flag = "✓" if r_c > 0.2 else ("~" if r_c > 0 else "✗")
        if r_c > 0.2:
            pos += 1
        stats.append({"item": "dcorr", "chr": chrom, "r": round(float(r_c), 3),
                      "p": float(p_c), "n": len(g)})
        print(f"  {chrom}: r={r_c:+.3f} {flag} (n={len(g)})")
big = [s for s in stats if s["item"] == "dcorr"]
print(f"  fraction of large chromosomes (n>=50) with r>0.2: {pos}/{len(big)}")

# ---- supplementary test: raw switch set (unfiltered) @100kb vs Xu switches ----
print("\n===== supplementary test: 100-kb switch overlap (raw set, unfiltered) =====")
nt["switch_raw"] = (np.where(nt.PC1_N > 0, "A", "B") != np.where(nt.PC1_T > 0, "A", "B"))
# Xu 500-kb labels expanded to 100 kb: each bin500 covers [bin500, bin500+500kb)
xu_sw = xu[xu.dscore != 0].copy()  # score changed (opposite-sign cases taken here)
xu_sw_strict = xu[(xu.score_num_N > 0) != (xu.score_num_T > 0)].copy()
xu_sw_strict["chr"] = xu_sw_strict["chr"].astype(str)
# 100-kb mapping
xu_bins = xu_sw_strict[["chr", "bin500"]]
ours_map = nt[["chr", "bin0", "start", "switch_raw"]].copy()
# owning 500-kb bin of each 100-kb bin: start // 500000 * 500000, but Xu bin500 already equals that value
ours_map["bin500"] = (ours_map.start // 500000) * 500000
ours_map["chr"] = ours_map["chr"].astype(str)
mg2 = ours_map.merge(xu_bins.assign(xu_switch=True), on=["chr", "bin500"], how="left")
mg2["xu_switch"] = mg2.xu_switch.fillna(False)
a = int((mg2.switch_raw & mg2.xu_switch).sum())
b = int((mg2.switch_raw & ~mg2.xu_switch).sum())
c = int((~mg2.switch_raw & mg2.xu_switch).sum())
d = int((~mg2.switch_raw & ~mg2.xu_switch).sum())
or_, p = fisher_exact([[a, b], [c, d]])
print(f"  our raw switches {a+b} bins | Xu switches mapped {c+a} bins | both {a}")
print(f"  * OR={or_:.2f}  Fisher p={p:.2e} (100-kb resolution, raw set)")

pd.DataFrame(stats).to_csv(os.path.join(OUT, "pilot1d_summary.csv"), index=False, encoding="utf-8")
print("\n=== Pilot 1d written pilot1d_summary.csv ===")
