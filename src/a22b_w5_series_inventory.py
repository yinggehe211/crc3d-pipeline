# -*- coding: utf-8 -*-
# a22b_w5_series_inventory.py — W5: GSE133928 full-cohort inventory + aria2 download-list generation
# Confirmed: GSM3930259 (BRD3179 T) / GSM3930260 (BRD3179 N) belong to GSE133928
#   "A topological atlas reveals layers of genome reorganization in colorectal cancer"
# Note: BRD3179 also has a 40kb iced matrix (899 MB) — the "higher resolution"
#   revision path is immediately feasible
# Usage: python a22b_w5_series_inventory.py
import os, re, requests, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
HEAD = {"User-Agent": "Mozilla/5.0 (research inventory)"}
BASE_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133928/suppl/"

# ---- 1. fetch the full suppl directory listing (HTTPS index) ----
print(f"Accessing {BASE_FTP} ...")
r = requests.get(BASE_FTP, headers=HEAD, timeout=60)
r.raise_for_status()
files = re.findall(r'href="([^"]+\.(?:matrix|txt)\.gz)"', r.text)
sizes = dict(re.findall(r'href="([^"]+\.(?:matrix|txt)\.gz)"[^<]*</a>\s*</td>'
                        r'<td[^>]*>\s*([\d.]+ [KMG]b|\d+)\s*</td>', r.text))
print(f"Total suppl files: {len(files)}")

# ---- 2. parse per sample: GSM id + biological sample name + resolution/kind ----
rows = []
for fn in files:
    m = re.match(r"(GSM\d+)_(.+?)_(\d+)_(iced\.matrix)\.gz$", fn)
    if m:
        gsm, bio, res, kind = m.groups()
        rows.append({"gsm": gsm, "sample": bio, "res_kb": int(res), "kind": "matrix",
                     "size": sizes.get(fn, "?"), "fname": fn})
        continue
    m = re.match(r"(GSM\d+)_(.+?)_hic_allValidPairs\.txt\.gz$", fn)
    if m:
        rows.append({"gsm": m.group(1), "sample": m.group(2), "res_kb": None,
                     "kind": "allValidPairs", "size": sizes.get(fn, "?"), "fname": fn})
D = pd.DataFrame(rows)
if D.empty:
    raise SystemExit("No sample files parsed — paste the raw HTML for manual inspection")

# ---- 3. cohort census table ----
D.to_csv(os.path.join(OUT, "a22b_GSE133928_inventory.csv"), index=False, encoding="utf-8")
mat = D[D.kind == "matrix"]
print("\n=== GSE133928 cohort census ===")
print(f"Samples (by GSM): {mat.gsm.nunique()}  iced matrices: {len(mat)}")
print("\n=== Matrix availability per sample ===")
piv = mat.pivot_table(index=["gsm", "sample"], columns="res_kb", values="fname", aggfunc="count")
print(piv.to_string())
print("\n=== Matrix size distribution (by resolution) ===")
print(mat.groupby("res_kb")["size"].apply(list).to_string())

# ---- 4. W5 download list: iced matrices only (allValidPairs skipped; matrices suffice for Layers 1-3) ----
aria = os.path.join(OUT, "a22b_aria2_download.txt")
have = {"GSM3930259", "GSM3930260"}  # BRD3179 pair already on disk
with open(aria, "w", encoding="utf-8") as f:
    for _, x in mat.iterrows():
        if x.gsm in have: continue
        f.write(f"{BASE_FTP}{x.fname}\n"
                f"  out=data/raw/{x.fname}\n"
                f"  referer=https://www.ncbi.nlm.nih.gov/geo/\n")
n_dl = sum(1 for _, x in mat.iterrows() if x.gsm not in have)
print(f"\n=== Download list written: {aria} ===")
print(f"Matrices to download: {n_dl} (on-disk BRD3179 pair excluded)")

# ---- 5. side check: does the 40kb matrix cover the full cohort (higher-resolution feasibility) ----
res40 = mat[mat.res_kb == 40]
print(f"\n40kb coverage: {res40.gsm.nunique()}/{mat.gsm.nunique()} samples")
if res40.gsm.nunique() == mat.gsm.nunique():
    print("* 40kb available for the full cohort — resolution-upgrade path for the Layer-3 retry is viable")
print("\nNext: batch-download via aria2c, then adapt the pilot pipeline into a multi-sample batch version")
