# -*- coding: utf-8 -*-
# a18_probe_rnaseq_build.py — three-in-one reconnaissance ("probe first, then act")
# (1) GSM4513967 Salmon .sf download + header peek (2.8MB)
# (2) decisive probe of binAnnotation genome build (chr1 rightmost bin end vs
#     hg19/hg38 full length)
# (3) GSE133928 series-level suppl listing (looking for SNIPER calls / cnv.csv
#     style products; tool-side access failed earlier, retry via urllib here)
import gzip, os, re, subprocess, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
SF = os.path.join(RAW, "GSM4513967_BRD3179.sf.txt.gz")
URL = ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4513nnn/GSM4513967/suppl/"
       "GSM4513967_BRD3179.sf.txt.gz")
BUILD_REF = {"hg19": 249250621, "hg38": 248956422}  # chr1 full length

# ---- (1) download .sf (aria2, per download discipline; skip if present) ----
if not os.path.exists(SF) and not os.path.exists(SF + ".aria2"):
    cmd = ["aria2c", "--check-certificate=false", "-c", "-x", "4", "-s", "4",
           "-d", RAW, URL]
    print("downloading .sf ...")
    subprocess.run(cmd, check=True)

# ---- (2) build probe: rightmost bin end of chr1 in binAnnotation ----
bed = os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz")
mx = 0
with gzip.open(bed, "rt", encoding="utf-8") as f:
    for ln in f:
        p = ln.split("\t")
        if p[0] == "chr1" and int(p[2]) > mx: mx = int(p[2])
print(f"\nbinAnnotation chr1 rightmost bin end = {mx:,}")
hits = {b: abs(mx - v) for b, v in BUILD_REF.items()}
best = min(hits, key=hits.get)
print(f"  vs hg19 {BUILD_REF['hg19']:,} diff {hits['hg19']:,}")
print(f"  vs hg38 {BUILD_REF['hg38']:,} diff {hits['hg38']:,}")
verdict = best if hits[best] < 50000 else "AMBIGUOUS"
print(f"  >> VERDICT: binAnnotation = {verdict} "
      f"(the GTF for gene->bin mapping must use the same build)")

# ---- (3) .sf header peek + row count ----
if os.path.exists(SF) and not os.path.exists(SF + ".aria2"):
    n = 0
    with gzip.open(SF, "rt", encoding="utf-8") as f:
        head = [f.readline().rstrip() for _ in range(3)]
        for _ in f: n += 1
    print(f"\n--- GSM4513967_BRD3179.sf ---")
    print(f"  rows (transcripts) = {n:,}")
    for h in head: print(f"  | {h}")
    tids = [head[1].split("\t")[0]] + ([head[2].split("\t")[0]] if len(head) > 2 else [])
    print(f"  transcript ID samples: {tids}")
    ver = ("versioned (ENST...\\d+.$)" if re.search(r"ENST\d+\.\d+", tids[0])
           else "unversioned")
    print(f"  ID format: {ver} -> basis for GTF selection")

# ---- (4) GSE133928 series-level suppl listing ----
print("\n--- GSE133928 series-level suppl ---")
try:
    req = urllib.request.Request(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1339nnn/GSE133928/suppl/",
        headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    files = re.findall(r'href="(GSE133928[^"]+)"', html)
    if files:
        for f_ in files: print(f"  {f_}")
        kw = [f_ for f_ in files if re.search(
            r"subcomp|sniper|cnv|table|compartment", f_, re.I)]
        print(f"\n  keyword hits (subcomp/sniper/cnv/compartment): {kw if kw else 'none'}")
        if kw:
            print("  >> SNIPER-style products exist at series level -> Pilot 3 could add "
                  "'vs original Johnstone calls' cross-check")
        else:
            print("  >> none at series level either -> Pilot 3 route stays with the "
                  "GM12878 anchor verdict, unchanged")
    else:
        print("  (empty directory or only anomalous files)")
except Exception as e:
    print(f"  access failed: {e}\n  -> non-blocking; route verdict already made (GM12878 anchor)")
