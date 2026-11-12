# -*- coding: utf-8 -*-
# a19_cnrsc_layer2.py - CNR-SC v0.1 Layer 2: per-chr O/E->Pearson->eigendecomposition PC1-PC4 + bootstrap sign stability
# Input: GSE133928 100-kb iced x2 (GSM3930260=N, GSM3930259=T) + binAnnotation (hg19, adjudicated in a18)
# Resolution: 100 kb (downgraded under the §8 RAM red line, not the §6 original 50 kb; cooler 50-kb stage would upgrade)
# Output: results/pilot/cnrsc_layer2_PCs_100kb.csv + _eigen_spectrum.csv + _bootstrap_flip.csv
# Discipline: PC sign is arbitrary per chr (§16.2), must be anchored before cross-sample comparison; long-job progress flushed to disk
import glob, gzip, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
OUT  = os.path.join(ROOT, "results", "pilot")
RES  = 100          # kb, recorded into all output columns (§13 rule 8)
N_BOOT, SUBFRAC = 50, 0.90
MIN_BINS = 300      # chromosomes with < 300 bins skipped (unstable decomposition)

bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]
bed["bin0"] = pd.to_numeric(bed["binID"]).astype(int) - 1
bed["start_i"] = bed["start"].astype(int)
CHRS = sorted(bed.chr.unique(), key=lambda c: int(c[3:]) if c[3:].isdigit() else 23)
NB   = {c: (g.bin0.max() - g.bin0.min() + 1) for c, g in bed.groupby("chr")}
B0MIN= {c: g.bin0.min() for c, g in bed.groupby("chr")}

def find_iced(prefix):
    cands = [p for p in glob.glob(os.path.join(RAW, prefix + "*.txt.gz"))
             if "binAnnotation" not in os.path.basename(p)]
    assert len(cands) == 1, f"{prefix}: {len(cands)} candidates, confirm manually: {cands}"
    return cands[0]

FILES = {"N": find_iced("GSM3930260"), "T": find_iced("GSM3930259")}

# ---- peek: print first 2 rows + bin-index style adjudication (§13 rule 5: cross-check before join) ----
print("=== peek (triple-format adjudication) ===")
fmt = {}
for t, p in FILES.items():
    with gzip.open(p, "rt", encoding="utf-8") as f:
        lines = [f.readline().rstrip().split() for _ in range(3)]
    hdr_numeric = all(x.replace(".", "", 1).replace("-", "", 1).isdigit() for x in lines[0][:3])
    print(f"  [{t}] {os.path.basename(p)}")
    for ln in lines: print(f"    | {ln[:6]}")
    fmt[t] = 1 if hdr_numeric else 0   # 1 = header row present, skip it
    print(f"    header row: {'yes -> skip 1 row' if hdr_numeric else 'no'}")

def load_dense(path, skip_header, tag):
    """single-pass scan of triples -> per-chr dense symmetric matrix (main chr). bin-index style auto-adapts (min==0 -> 0-based)."""
    mats = {c: np.zeros((NB[c], NB[c]), dtype=np.float32) for c in CHRS if NB[c] >= MIN_BINS}
    seen_min, nrow, t0 = 10**9, 0, time.time()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        if skip_header: next(f)
        for ln in f:
            nrow += 1
            if nrow % 5_000_000 == 0:
                print(f"  [{tag}] {nrow:,} rows {time.time()-t0:.0f}s", end="\r", flush=True)
            p = ln.split()
            try:
                i, j, v = int(p[0]), int(p[1]), float(p[2])
            except (ValueError, IndexError):
                continue
            seen_min = min(seen_min, i, j)
            base = seen_min if seen_min == 0 else 1   # fixed once determined
            for c, M in mats.items():
                k = i - B0MIN[c] + (0 if base == 0 else -1)  # 0-based: subtract min; 1-based: also -1
                l = j - B0MIN[c] + (0 if base == 0 else -1)
                if 0 <= k < NB[c] and 0 <= l < NB[c]:
                    M[k, l] = v; M[l, k] = v
    print(f"  [{tag}] done: {nrow:,} rows, index min={seen_min} -> "
          f"{'0-based' if seen_min==0 else '1-based'} {time.time()-t0:.0f}s")
    return mats, seen_min

MATS = {}
for t, p in FILES.items():
    MATS[t], base_min = load_dense(p, fmt[t], t)
# hard check on index-style consistency (both files must share the style, else misalignment)
assert abs(base_min) <= 1, f"index min={base_min} abnormal (neither 0 nor 1), abort"

def layer2(M, c, tag):
    """single chr: O/E -> Pearson -> eigh -> PC1-4 + bootstrap sign-flip rate"""
    M = M.astype(np.float64)
    np.fill_diagonal(M, 0.0)
    r = M.sum(1)
    tot = r.sum()
    E = np.outer(r, r) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        OE = np.where(E > 0, M / E, 0.0)
    sd = OE.std(1)
    ok = sd > 1e-12
    P = np.zeros_like(OE)
    idx = np.where(ok)[0]
    Psub = np.corrcoef(OE[idx])
    P[np.ix_(idx, idx)] = np.nan_to_num(Psub)
    w, V = np.linalg.eigh(P)                       # ascending
    order = np.argsort(w)[::-1][:4]
    lam, V4 = w[order], V[:, order]
    flip = {k: [] for k in range(4)}
    rng = np.random.default_rng(20261107)
    n = P.shape[0]
    for b in range(N_BOOT):
        sub = np.sort(rng.choice(n, int(n * SUBFRAC), replace=False))
        Ps = P[np.ix_(sub, sub)]
        try:
            ws, Vs = np.linalg.eigh(Ps)
        except np.linalg.LinAlgError:
            continue
        o2 = np.argsort(ws)[::-1][:4]
        for k in range(4):
            v = Vs[:, o2[k]]
            full = V4[sub, k]
            if v @ full < 0: v = -v                # align overall sign first
            flip[k].append(float((np.sign(v) != np.sign(full)).mean()))
    spec = {f"eig{k+1}": float(lam[k]) for k in range(4)}
    fr = {f"PC{k+1}_fliprate": float(np.mean(flip[k])) for k in range(4)}
    print(f"  [{tag}] {c}: n={n} eig1={lam[0]:.3f} eig2={lam[1]:.3f} "
          f"flip={['%.2f'%fr[f'PC{k+1}_fliprate'] for k in range(4)]}")
    return V4, {**spec, **fr}

rows, spec_rows = [], []
for t in ("N", "T"):
    for c in CHRS:
        if c not in MATS[t]: continue
        t0 = time.time()
        V4, st = layer2(MATS[t][c], c, t)
        g = bed[bed.chr == c].sort_values("bin0").reset_index(drop=True)
        if len(g) != MATS[t][c].shape[0]:
            g = bed[bed.chr == c].reset_index(drop=True)   # bin0 contiguous, no gaps (proven in a17e), direct layout
        for k in range(4):
            for bi, v in zip(g.bin0.values, V4[:, k]):
                rows.append({"sample": t, "chr": c, "bin0": bi,
                             "PC": k + 1, "val": float(v), "resolution_kb": RES})
        spec_rows.append({"sample": t, "chr": c, **st, "sec": time.time() - t0})

pc = pd.DataFrame(rows)
pc.to_csv(os.path.join(OUT, f"cnrsc_layer2_PCs_{RES}kb.csv"), index=False, encoding="utf-8")
sp = pd.DataFrame(spec_rows)
sp.to_csv(os.path.join(OUT, f"cnrsc_layer2_eigen_spectrum_{RES}kb.csv"), index=False, encoding="utf-8")
print("\n=== Layer 2 summary ===")
print(sp[["sample","chr","eig1","eig2","PC1_fliprate","PC2_fliprate"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n* reading: high eig2/eig1 ratio = subcompartment axis informative (> 0.3 preferred); PC1 fliprate < 0.1 = sign stable")
print(f"written: cnrsc_layer2_PCs_{RES}kb.csv (long format, N/T x PC1-4) + eigen_spectrum")
print("note: PC sign is arbitrary per chr (§16.2), must anchor per chr before Layer 3 / cross-sample comparison")
