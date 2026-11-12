# -*- coding: utf-8 -*-
# a16d_allvalidpairs_depth.py — per-bin raw read depth from allValidPairs
#   (substitute for the non-existent raw matrix; see handover §17)
# Steps: HEAD size probe -> aria2 download (-c resume) -> peek column-order
#   detection -> pandas chunked counting -> CSV written (.done idempotent).
# The .gz files may be deleted after counting if space is tight; delete
# manually — the script never deletes data.
import os, re, sys, gzip, subprocess, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
OUT  = os.path.join(ROOT, "results", "pilot")
os.makedirs(OUT, exist_ok=True)
SAMPLES  = {"BRD3179N_normal": "GSM3930260", "BRD3179_tumor": "GSM3930259"}
SOFT_PAT = re.compile(r"GSM\d+_BRD3179N?_hic_allValidPairs\.txt\.gz$")

def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

# ---- 1) resolve allValidPairs URLs from per-sample suppl directories ----
found = {}
for tag, gsm in SAMPLES.items():
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/"
    gz_files = re.findall(r'href="([^"]+\.gz)"', http_get(base))
    hits = [h for h in gz_files if SOFT_PAT.fullmatch(h)]
    assert len(hits) == 1, f"{tag}: allValidPairs matched {len(hits)} files (expected 1): {gz_files}"
    url = base + hits[0]
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    found[tag] = (url, size)
    print(f"{tag}: {hits[0]} remote size {size/1e9:.2f} GB")

# ---- 2) aria2 download (skip when present with matching size) ----
todo = []
for tag, (url, size) in found.items():
    fn = url.rsplit("/", 1)[-1]
    local = os.path.join(RAW, fn)
    if os.path.exists(local) and os.path.getsize(local) == size:
        print(f"Already present with matching size, skipping download: {fn}")
    else:
        todo.append(url)
if todo:
    urls_file = os.path.join(RAW, "a16_urls.txt")
    with open(urls_file, "w", encoding="utf-8") as f:
        f.write("\n".join(todo) + "\n")
    subprocess.check_call(["aria2c", "--check-certificate=false", "-c", "-d", RAW,
                           "-j", "1", "-x", "4", "-s", "16", "--summary-interval=30",
                           "-i", urls_file])

# ---- 3) binAnnotation -> chr -> (starts, bin0) lookup structure ----
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]
bed["bin0"]   = pd.to_numeric(bed["binID"]).astype(int) - 1
bed["start_i"] = bed["start"].astype(int)
lookup = {}
for chrom, g in bed.groupby("chr"):
    g = g.sort_values("start_i")
    lookup[chrom] = (g.start_i.values, g.bin0.values.astype(np.int64))
print(f"binAnnotation: {len(bed)} bins, {len(lookup)} chromosomes")

def bins_of(chrom_arr, pos_arr):
    """Vectorized: coordinate -> bin0 (searchsorted); out-of-range/unknown chromosomes dropped"""
    out = np.full(len(pos_arr), -1, dtype=np.int64)
    for chrom in pd.unique(chrom_arr):
        if chrom not in lookup: continue
        starts, bins0 = lookup[chrom]
        sel = chrom_arr == chrom
        idx = np.searchsorted(starts, pos_arr[sel], side="right") - 1
        valid = (idx >= 0) & (idx < len(starts))
        b = np.full(sel.sum(), -1, dtype=np.int64)
        b[valid] = bins0[idx[valid]]
        out[sel] = b
    return out

# ---- 4) peek column-order detection + chunked counting ----
def process(tag, gsm):
    fn = [u.rsplit("/", 1)[-1] for t, (u, s) in found.items() if t == tag][0]
    local = os.path.join(RAW, fn)
    out_csv = os.path.join(OUT, f"depth_{tag}_100kb.csv")
    if os.path.exists(out_csv + ".done"):
        print(f"{tag}: depth CSV already exists (.done), skipping")
        return
    # peek first 3 lines
    with gzip.open(local, "rt") as f:
        head = [f.readline().rstrip("\n") for _ in range(3)]
    print(f"\n{tag} first 3 lines:")
    for h in head: print(" ", h)
    cols0 = head[0].split()
    chr_cols = [i for i, v in enumerate(cols0) if re.fullmatch(r"chr[\dXYM]+", v)]
    assert len(chr_cols) == 2, f"Detected {len(chr_cols)} chromosome columns (expected 2): {cols0}"
    pos_cols = []
    for c in chr_cols:
        after = [i for i in range(c + 1, len(cols0)) if re.fullmatch(r"\d+", cols0[i])]
        assert after, f"No coordinate column found after chr column {c}: {cols0}"
        pos_cols.append(c + 1)  # HiC-Pro convention: position column immediately follows chr column
    print(f" Detected: chr cols={chr_cols}, pos cols={pos_cols}")
    total = np.zeros(len(bed), dtype=np.int64)
    n_lines, chunk_no, n_bad = 0, 0, 0
    usecols = sorted(chr_cols + pos_cols)
    chr_idx = [usecols.index(c) for c in chr_cols]
    pos_idx = [usecols.index(p) for p in pos_cols]
    for chunk in pd.read_csv(local, sep=r"\s+", header=None, compression="gzip",
                             engine="c", usecols=usecols, chunksize=5_000_000,
                             dtype={c: str for c in chr_cols}):
        arr = chunk.values
        ch1, ch2 = arr[:, chr_idx[0]], arr[:, chr_idx[1]]
        p1 = pd.to_numeric(arr[:, pos_idx[0]], errors="coerce")
        p2 = pd.to_numeric(arr[:, pos_idx[1]], errors="coerce")
        bad = ~(np.isfinite(p1) & np.isfinite(p2))
        n_bad += int(bad.sum())
        p1 = np.where(np.isfinite(p1), p1, -1).astype(np.int64)
        p2 = np.where(np.isfinite(p2), p2, -1).astype(np.int64)
        b = np.concatenate([bins_of(ch1, p1), bins_of(ch2, p2)])
        b = b[b >= 0]
        np.add.at(total, b, 1)
        n_lines += len(chunk); chunk_no += 1
        print(f" chunk{chunk_no}: cumulative {n_lines:,} lines", flush=True)
        if chunk_no % 20 == 0:  # mid-run checkpoint write
            pd.DataFrame({"bin0": bed.bin0, "depth": total}).to_csv(out_csv + ".tmp", index=False)
    pd.DataFrame({"bin0": bed.bin0, "depth": total}).to_csv(out_csv, index=False)
    open(out_csv + ".done", "w").write(f"lines={n_lines}\n")
    nz = (total > 0).mean()
    print(f" {tag}: {n_lines:,} pairs (skipped bad lines {n_bad}) | "
          f"nonzero bins {nz:.1%} | median depth {np.median(total):.0f}")

for tag in SAMPLES:
    process(tag, SAMPLES[tag])
print("\n=== depth CSVs ready -> python a14b_pilot2.py ===")
