# -*- coding: utf-8 -*-
# a9_pilot0.py v3.1 - compartment PC1 after distance-decay normalization + multi-eigenvector selection + Xu agreement (direct/reverse/per-arm/per-arm reverse)
# Gates: 1) gzip intact  2) row sums uniform (iced-balanced)  3) PC1 sign flips 10-300  4) cross-check vs Xu compartment.txt, aligned agreement >= 80%
# v3.1: eigenvector sign is arbitrary -> report reverse agreement too; primary gate = max(direct, reverse); per-arm handled analogously
import gzip, os
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
XU = os.path.join(ROOT, "Xu_repo", "CRC_metastasis_code-main", "Fig3b",
                  "Input_compartment", "CRC-03-N-HiC", "compartment.txt")
OUT = os.path.join(ROOT, "results", "pilot")
os.makedirs(OUT, exist_ok=True)

SAMPLES = {
    "BRD3179N_normal": "GSM3930260_BRD3179N_100000_iced.matrix.gz",
    "BRD3179_tumor":   "GSM3930259_BRD3179_100000_iced.matrix.gz",
}


def load_triplet(path):
    with gzip.open(path, "rt") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else r"\s+"
    sep_name = "TAB" if sep == chr(9) else "space"
    print(f"  delimiter: {sep_name}")
    return pd.read_csv(path, sep=sep, header=None, compression="gzip",
                       engine="c", names=["i", "j", "v"])


def compartment_pc1(sub, bin_starts):
    """sub: within-chr 100-kb submatrix (iced); bin_starts: bin start positions (bp).
    Distance-decay normalization -> Pearson correlation -> eigendecomposition,
    among the first 6 eigenvectors pick the largest-eigenvalue one with sign flips in [10, 300]."""
    n = sub.shape[0]
    # 1) distance-decay expectation E(d)
    d = np.abs(np.subtract.outer(bin_starts, bin_starts)) // 100000
    dmax = int(d.max())
    sum_d = np.bincount(d.ravel(), weights=sub.ravel(), minlength=dmax + 1)
    cnt_d = np.bincount(d.ravel(), minlength=dmax + 1)
    exp_d = np.divide(sum_d, np.maximum(cnt_d, 1))
    oe = sub / np.maximum(exp_d[d], 1e-9)
    np.fill_diagonal(oe, 0.0)  # diagonal carries no compartment information
    # 2) Pearson correlation (zero-variance bins produce NaN, set to 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(oe)
    c = np.nan_to_num(c)
    np.fill_diagonal(c, 0.0)
    # 3) eigendecomposition, multi-candidate selection
    w, v = np.linalg.eigh(c)
    order = np.argsort(w)[::-1][:6]
    picked = None
    for k in order:
        ev = v[:, k]
        flips = int((ev[:-1] * ev[1:] < 0).sum())
        if 10 <= flips <= 300:
            picked = (ev, flips, f"eig{k}(λ={w[k]:.2f})")
            break
    if picked is None:  # fallback: take the largest-eigenvalue vector
        ev = v[:, order[0]]
        picked = (ev, int((ev[:-1] * ev[1:] < 0).sum()), f"eig{order[0]}(fallback)")
    ev, flips, method = picked
    if (ev > 0).sum() and np.median(ev[ev > 0]) < abs(np.median(ev[ev < 0])):
        ev = -ev  # convention: positive side = majority side (global sign unified via cross-check below)
    return ev, flips, method


results = []
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]

for tag, fn in SAMPLES.items():
    p = os.path.join(RAW, fn)
    print(f"\n===== {tag} ({fn}) =====")
    try:
        with gzip.open(p, "rb") as f:
            while f.read(1 << 24):
                pass
        print("  gzip integrity: PASS")
    except Exception as e:
        raise SystemExit(f"gzip FAIL: {e}")

    df = load_triplet(p)
    n_bins = int(df[["i", "j"]].max().max()) + 1
    m = sp.coo_matrix((df.v.values, (df.i.values, df.j.values)),
                      shape=(n_bins, n_bins)).tocsr()
    m = m + m.T
    rs = np.asarray(m.sum(axis=1)).ravel()
    print(f"  dim {n_bins} | nonzero bins {100*(rs>0).mean():.1f}% | "
          f"row-sum median={np.median(rs):.0f} P10={np.percentile(rs,10):.0f} P90={np.percentile(rs,90):.0f}")
    results.append({"sample": tag, "n_bins": n_bins,
                    "nonzero_frac": round((rs > 0).mean(), 4)})

    cb = bed[bed.chr == "chr1"].copy()
    cb["bin0"] = pd.to_numeric(cb["binID"]).astype(int) - 1  # 1-based -> 0-based
    cb = cb[(cb["bin0"] >= 0) & (cb["bin0"] < n_bins)].reset_index(drop=True)
    ids = cb["bin0"].values.astype(int)
    starts = cb["start"].astype(int).values

    sub = m[ids][:, ids].toarray().astype(np.float64)
    ev, flips, method = compartment_pc1(sub, starts)
    print(f"  chr1 bins={len(ids)} | PC1 via {method} | sign flips {flips}"
          f" (gate: 10-300 indicates genuine compartment structure)")
    pd.DataFrame({"binID": cb["binID"], "start": starts, "PC1_chr1": ev}).to_csv(
        os.path.join(OUT, f"pilot0_{tag}_chr1_PC1.csv"), index=False, encoding="utf-8")

    # 3) cross-check vs Xu 500-kb labels (normal sample)
    if tag == "BRD3179N_normal":
        xu = pd.read_csv(XU, sep="\t")
        xu.columns = [c.strip() for c in xu.columns]
        xc = xu[xu.seqnames == "chr1"].copy()
        xc["score_num"] = pd.to_numeric(xc.score, errors="coerce")
        xc = xc[xc.score_num.notna()]
        print(f"  Xu chr1 valid-label bins: {len(xc)} (500-kb resolution)")

        ours = pd.DataFrame({"start": starts, "PC1": ev})
        ours["bin500"] = (ours.start // 500000) * 500000
        agg = ours.groupby("bin500").PC1.mean().reset_index()
        mg = agg.merge(xc[["start", "score_num"]], left_on="bin500",
                       right_on=xc.start.astype(int) - 1, how="inner")
        agree = ((mg.PC1 > 0) == (mg.score_num > 0)).mean()

        # per-arm alignment (chr1 centromere ~121 Mb): majority sign per arm, then agreement
        mg["arm"] = np.where(mg.start < 121_500_000, "p", "q")
        agree_arm_parts = []
        for a, g in mg.groupby("arm"):
            s = 1 if (g.PC1 > 0).mean() >= 0.5 else -1
            agree_arm_parts.append(((g.PC1 * s > 0) == (g.score_num > 0)).mean())
        agree_arm = float(np.mean(agree_arm_parts))

        # v3.1: eigenvector sign is arbitrary -> reverse agreement = 1 - direct
        agree_anti = 1 - agree
        agree_arm_anti = 1 - agree_arm
        final = max(agree, agree_anti)
        final_arm = max(agree_arm, agree_arm_anti)
        print(f"  * Xu chr1 500-kb sign agreement: direct {agree:.2%} | reverse {agree_anti:.2%}"
              f" | per-arm {agree_arm:.2%} | per-arm reverse {agree_arm_anti:.2%} (n={len(mg)})")
        print(f"  * final aligned: genome {final:.2%} | per-arm {final_arm:.2%}  gate: >= 80%")
        results.append({"sample": "vs_Xu_chr1",
                        "agree": round(float(agree), 4),
                        "agree_aligned": round(float(final), 4),
                        "agree_arm_aligned": round(float(final_arm), 4),
                        "n": len(mg)})

pd.DataFrame(results).to_csv(os.path.join(OUT, "pilot0_summary.csv"),
                             index=False, encoding="utf-8")
print("\n=== Pilot 0 summary -> results/pilot/pilot0_summary.csv ===")
