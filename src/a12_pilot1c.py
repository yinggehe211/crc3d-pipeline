# -*- coding: utf-8 -*-
# a12_pilot1c.py - Pilot 1c: third-party validation of the switch set (our 158 bins vs Xu N->T calls)
# Tests: 1) overlap enrichment (OR)  2) direction concordance (B->A / A->B)
import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "pilot")
XU_DIR = os.path.join(ROOT, "Xu_repo", "CRC_metastasis_code-main", "Fig3b", "Input_compartment")


def load_xu(sample):
    xu = pd.read_csv(os.path.join(XU_DIR, sample, "compartment.txt"), sep="\t")
    xu.columns = [c.strip() for c in xu.columns]
    xu["score_num"] = pd.to_numeric(xu.score, errors="coerce")
    xu = xu[xu.score_num.notna()].copy()
    xu["bin500"] = xu.start.astype(int) - 1  # 1-based offset (a10 adjudicated)
    return xu[["seqnames", "bin500", "score_num"]].rename(columns={"seqnames": "chr"})


xn, xt = load_xu("CRC-03-N-HiC"), load_xu("CRC-03-T-HiC")
xu = xn.merge(xt, on=["chr", "bin500"], suffixes=("_N", "_T"))
xu["xu_switch"] = (xu.score_num_N > 0) != (xu.score_num_T > 0)
xu["xu_dir"] = np.where(xu.xu_switch,
                        np.where((xu.score_num_N < 0) & (xu.score_num_T > 0), "B2A",
                                 np.where((xu.score_num_N > 0) & (xu.score_num_T < 0), "A2B", "")), "")
print(f"Xu 500-kb bins labeled in both N/T: {len(xu)} | Xu-called switches: {int(xu.xu_switch.sum())}")

# our high-confidence switches (100 kb) -> aggregated to 500 kb (majority switch direction per bin)
ours = pd.read_csv(os.path.join(OUT, "pilot1b_NT_switch_confident.csv"))
ours["bin500"] = (ours.start // 500000) * 500000
o_sw = (ours.groupby(["chr", "bin500"])
        .agg(n_sw_100kb=("switch_conf", "sum"))
        .reset_index())
o_sw["our_switch"] = True
o_sw["our_dir"] = np.where(
    ours.groupby(["chr", "bin500"]).apply(
        lambda g: "B2A" if ((g.comp_N == "B") & (g.comp_T == "A")).sum() >
                            ((g.comp_N == "A") & (g.comp_T == "B")).sum() else "A2B")
        .reset_index(drop=True).values, None, None)  # placeholder, recomputed explicitly below
# the apply formulation above is error-prone; recompute explicitly:
dir_df = (ours.assign(b2a=(ours.comp_N == "B") & (ours.comp_T == "A"))
          .groupby(["chr", "bin500"]).b2a.sum().reset_index())
o_sw = o_sw.merge(dir_df, on=["chr", "bin500"])
o_sw["our_dir"] = np.where(o_sw.b2a * 2 >= o_sw.n_sw_100kb, "B2A", "A2B")
o_sw = o_sw[["chr", "bin500", "our_switch", "our_dir", "n_sw_100kb"]]

mg = xu.merge(o_sw, on=["chr", "bin500"], how="left")
mg["our_switch"] = mg["our_switch"].fillna(False)
total = len(mg)
a = int(((mg.our_switch) & (mg.xu_switch)).sum())
b = int(((mg.our_switch) & (~mg.xu_switch)).sum())
c = int(((~mg.our_switch) & (mg.xu_switch)).sum())
d = total - a - b - c
or_, p = fisher_exact([[a, b], [c, d]])
print(f"\n===== overlap enrichment (500-kb resolution) =====")
print(f"  switch called by both: {a} | ours only: {b} | Xu only: {c} | neither: {d}")
print(f"  * enrichment OR={or_:.1f}  Fisher p={p:.2e}")
print(f"  fraction of our switch bins inside the Xu switch set: {a}/{a+b} = {a/max(a+b,1):.1%}")

both = mg[(mg.our_switch) & (mg.xu_switch)]
dir_agree = (both.our_dir == both.xu_dir).mean()
print(f"\n===== direction concordance (bins switched in both, n={len(both)}) =====")
print(f"  * direction agreement (B2A/A2B): {dir_agree:.1%} [chance baseline 50%]")
for d_ in ("B2A", "A2B"):
    sub = both[both.xu_dir == d_]
    if len(sub):
        print(f"    {d_}: n={len(sub)} our direction matches {(sub.our_dir==sub.xu_dir).mean():.1%}")

pd.DataFrame({"OR": [or_], "p": [p], "both": [a], "ours_only": [b],
              "xu_only": [c], "neither": [d],
              "dir_agree": [round(float(dir_agree), 4)]}).to_csv(
    os.path.join(OUT, "pilot1c_switch_validation.csv"), index=False)
print("\n=== Pilot 1c written pilot1c_switch_validation.csv ===")
