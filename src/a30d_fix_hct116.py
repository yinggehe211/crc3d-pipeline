# -*- coding: utf-8 -*-
# a30d_fix_hct116.py — repair of the a24c HCT116 standardization-not-wired
#   incident: calibration-transfer reconstruction of the v2 bedgraph
# Principle: a24c = a24b raw x per-chr constant (reciprocal of the
#   cross-sample std median); regress the factors from the BRD3187 file pair,
#   verify transferability on BRD3462 (independent check), then apply to the
#   HCT116 raw -> v2, cross-check counts against a24c_summary
# Output: a24c_diff_HCT116_5AZA_vs_0h_v2.bedgraph (old file untouched, kept
#   for the incident audit)
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")

def load_bg(fn):
    df = pd.read_csv(os.path.join(RES, fn), sep="\t", header=None,
                     names=["chrom", "start", "end", "v"])
    df["key"] = df["chrom"] + ":" + df["start"].astype(str)
    return df

pairs = {
    "BRD3187": ("a24b_diff_BRD3187_vs_BRD3187N.bedgraph",
                "a24c_diff_BRD3187_vs_BRD3187N.bedgraph"),
    "BRD3462": ("a24b_diff_BRD3462_vs_BRD3462N.bedgraph",
                "a24c_diff_BRD3462_vs_BRD3462N.bedgraph"),
    "HCT116":  ("a24b_diff_HCT116_5AZA_vs_0h.bedgraph", None),
}
db, dc = load_bg(pairs["BRD3187"][0]), load_bg(pairs["BRD3187"][1])
assert len(db) == len(dc), "BRD3187 b/c row counts differ, key-alignment premise broken"

# --- 1) per-chr factor regression (BRD3187 file pair) ---
factors = {}
print("===== 1) per-chr factors (BRD3187: a24c = raw x f) =====")
for c, g in db.groupby("chrom"):
    gc = dc.loc[dc["key"].isin(set(g["key"]))].set_index("key").loc[g["key"]]
    b, v = g["v"].values.astype(float), gc["v"].values.astype(float)
    nz = np.abs(b) > 1e-9
    f = float((b[nz] * v[nz]).sum() / (b[nz] ** 2).sum())
    resid = np.abs(v - f * b).max()
    factors[c] = f
    # tolerance scales with the factor: raw on disk at 4 decimals gives a
    # per-bin rounding of ±5e-5, amplified by factor f
    tol = 1.2 * 5e-5 * f
    print(f"  {c:6s} f={f:7.2f} max residual={resid:.2e} (tolerance={tol:.2e})")
    assert resid < tol, f"{c} linear residual beyond rounding-amplified tolerance, recipe assumption broken"
fs = np.array(list(factors.values()))
print(f"factor domain: {fs.min():.1f}-{fs.max():.1f} (chr13~112 has the smallest "
      f"cross-sample std; logged; residual = the true verification)")
assert 20 < fs.min() and fs.max() < 200, "factor domain extremely anomalous, stopping"

# --- 2) transferability verification (BRD3462, key-aligned) ---
print("\n===== 2) transferability check: BRD3462 raw x factors vs a24c file =====")
db2, dc2 = load_bg(pairs["BRD3462"][0]), load_bg(pairs["BRD3462"][1])
m = db2.merge(dc2, on="key", suffixes=("_b", "_c"))
assert len(m) > 0.99 * len(dc2), f"BRD3462 key alignment too low ({len(m)}/{len(dc2)}), stopping"
f2 = m["chrom_b"].map(factors).values
pred = m["v_b"].values * f2
diff = np.abs(pred - m["v_c"].values)
ftol = 1.2 * (5e-5 * f2 + 5e-5)
print(f"  aligned {len(m)}/{len(dc2)} bins, |pred-truth| max={diff.max():.2e}, "
      f"median={np.median(diff):.2e} (per-row tolerance max={ftol.max():.2e})")
assert (diff <= ftol).all(), \
    "transferability FAILED: factor is not a uniform constant across contrasts, recipe assumption broken"
print("  -> transferability passed: per-chr factors are a uniform scale across "
      "contrasts, safe to apply to HCT116")

# --- 3) HCT116 v2 reconstruction + cross-check against a24c_summary ---
print("\n===== 3) HCT116 v2 reconstruction =====")
dh = load_bg(pairs["HCT116"][0])
fac = dh["chrom"].map(factors)
missing = sorted(set(dh["chrom"][fac.isna()]))
print(f"  chromosomes without a factor: {missing} "
      f"(expected=['chrY'], per the a24c standardization convention excluding chrY)")
assert missing == ["chrY"], f"chromosomes without a factor are not chrY-only: {missing}, stopping"
keep = fac.notna().values
n_raw = len(dh)
dh = dh[keep].reset_index(drop=True)
v2 = dh["v"].values * fac[keep].values
print(f"  bins dropped (no factor): {n_raw - len(dh)} (28268-28188=80, matches "
      f"the a24c masking convention)")
assert len(dh) == 28188, f"post-drop n={len(dh)} != summary 28188, stopping"
med_abs = float(np.median(np.abs(v2)))
n94 = int((np.abs(v2) > 0.94).sum()); n50 = int((np.abs(v2) > 0.5).sum())
n10 = int((np.abs(v2) > 1.0).sum())
REF = {"med": 0.647, "n94": 10075, "n50": 16641, "n10": 9346}
print(f"  v2: |diff| median={med_abs:.3f} (summary {REF['med']}), "
      f">0.94: {n94} ({REF['n94']}), >0.5: {n50} ({REF['n50']}), "
      f">1.0: {n10} ({REF['n10']}), n={len(v2)}")
assert abs(med_abs - REF["med"]) < 0.02, "median cross-check failed, stopping"
# tolerance = 4-decimal raw x factor~50 rounding amplification (±2.5e-3)
# + the 80-bin masking-convention difference
for k, got, ref in ((">0.94", n94, REF["n94"]), (">0.5", n50, REF["n50"]),
                    (">1.0", n10, REF["n10"])):
    d = got - ref
    edge = int((np.abs(np.abs(v2) - float(k[1:])) < 0.005).sum())
    print(f"  {k}: delta {d:+d} (bins within ±0.005 of threshold={edge})")
    assert abs(d) <= max(300, edge), f"{k} count delta beyond tolerance, stopping"
print("  -> cross-check passed: the HCT116 numbers in a24c_summary are "
      "independently recomputed and confirmed")
out = dh[["chrom", "start", "end"]].copy()
out["v"] = np.round(v2, 4)
fp = os.path.join(RES, "a24c_diff_HCT116_5AZA_vs_0h_v2.bedgraph")
out.to_csv(fp, sep="\t", header=False, index=False, encoding="utf-8")
print(f"\nSaved: {os.path.basename(fp)} ({len(out)} bins); old file kept for the incident audit")
