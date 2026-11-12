# -*- coding: utf-8 -*-
# a28d_bin_enrich.py v2 — W7-1 anchor-free validation, cell 2: K27me3 enrichment
#   over B3/B4, registered AUC >= 0.7
# v1 depended on pyBigWig (no win64 wheel) -> v2 reads the pure-Python parsing
#   output CSVs produced by a28e
# Primary test (registered, §16.12 verbatim): B3∪B4 bins vs the rest,
#   BRD3179N K27me3 (same-patient paired)
# Direction check: median of A1∪A2 < global median of the B cluster;
#   replication surface (exploratory, no gate): BRD3187N K27me3 / BRD3187 K9me3
# The .bw files lack input controls -> rank-based AUC is robust to monotone
#   transforms; archived
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")
lab = pd.read_csv(os.path.join(RES, "a22d_subcomp_labels_chrcenter.csv"))
print(f"Labeled bins: {len(lab)}, {lab.cluster.value_counts().to_dict()}")

def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    r1 = ranks[:len(pos)].sum()
    return (r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

FILES = [("BRD3179N_K27me3", "a28e_BRD3179N_K27me3_binsig.csv", True),
         ("BRD3187N_K27me3", "a28e_BRD3187N_K27me3_binsig.csv", False),
         ("BRD3187_K9me3",   "a28e_BRD3187_K9me3_binsig.csv",   False)]
# masks are computed on D.cluster AFTER the merge (strictly aligned with D row
# order; precomputing on lab was a v2 bug)
rows = []
for tag, fn, is_primary in FILES:
    fp = os.path.join(RES, fn)
    if not os.path.exists(fp):
        print(f"[{tag}] parsing output missing: {fn}, skipping"); continue
    E = pd.read_csv(fp)
    D = lab.merge(E, on="bin", how="inner")
    D = D.assign(pos=D.cluster.isin(["B3", "B4"]).values,
                 A=D.cluster.isin(["A1", "A2"]).values)
    med = D.groupby("cluster")["mean"].median()
    print(f"\n[{tag}] covered bins={len(D)}; median signal per label:")
    print(med.round(4).to_string())
    a_ = auc(D["mean"][D.pos].values, D["mean"][~D.pos].values)
    dir_ok = D["mean"][D.A].median() < med[["B1", "B2", "B3", "B4"]].median()
    verdict = ("PASS gate" if (is_primary and a_ >= 0.7)
               else ("FAIL gate" if is_primary else "(exploratory)"))
    print(f"B3∪B4 vs rest AUC={a_:.4f} {verdict}")
    print(f"Direction check, A-cluster median < B-cluster median: "
          f"{'yes' if dir_ok else 'NO'}")
    rows.append({"track": tag, "primary": is_primary, "n_cov": len(D),
                 "AUC_B34": round(a_, 4), "A_below_B": dir_ok,
                 **{f"med_{k}": round(med.get(k, np.nan), 4)
                    for k in ["A1", "A2", "B1", "B2", "B3", "B4"]}})
pd.DataFrame(rows).to_csv(os.path.join(RES, "a28d_enrich_summary.csv"),
                          index=False, encoding="utf-8-sig")
print("\nSaved: a28d_enrich_summary.csv (a28e_*_binsig.csv kept as recomputation basis)")
