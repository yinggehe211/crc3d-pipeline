# -*- coding: utf-8 -*-
# a21_cnrsc_layer3_anchors.py - Layer 3: Rao2014 GM12878 anchors + per-chr weighted K-means
# Anchors: enumerate GEO directories for the subcompartment bed (hg19==hg19); Pilot 3 gate: N-sample ARI >= 0.6
# Per-chr centroid = mean of that chr's anchor bins in its own PC space -> degenerate-chr rotation absorbed by the fallback (§16.7)
# Output: per-bin subcompartment labels + N/T switch-event table (AB-stable but subtype-switching = quantification of the title claim)
import os, re, subprocess, urllib.request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW, OUT = (os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "results", "pilot"))
RS = 20261108
PCS_F = os.path.join(OUT, "cnrsc_layer2_PCs_100kb_distexp.csv")
SPEC_F = os.path.join(OUT, "cnrsc_layer2_eigen_spectrum_100kb_distexp.csv")
CNV_F = os.path.join(OUT, "pilot2_bin_CNV_100kb_methyl_v3.csv")
PCS = ["PC1", "PC2", "PC3", "PC4"]

def head_clen(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None

# ---- 1. enumerate GEO directories for the subcompartment bed (§13 rule 3: enumerate, never guess) ----
bed_url = None
for d in ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551652/suppl/",
          "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"):
    try:
        with urllib.request.urlopen(d, timeout=60) as r:
            html = r.read().decode("utf-8", "ignore")
        cands = re.findall(r'href="([^"]*subcompartment[^"]*\.bed(?:\.gz)?)"', html)
        print(f"enumerate {d}  candidates: {cands}")
        if cands:
            bed_url = d + sorted(cands)[0]; break
    except Exception as e:
        print(f"  enumeration failed: {e}")
assert bed_url, "no subcompartment bed in either directory -> paste output and report, do not guess URLs"
fn = bed_url.split("/")[-1]; bed_path = os.path.join(RAW, fn)
clen = head_clen(bed_url); assert clen, "HEAD failed"
if not (os.path.exists(bed_path) and not os.path.exists(bed_path + ".aria2")
        and os.path.getsize(bed_path) == clen):
    subprocess.run(["aria2c", "--check-certificate=false", "-c",
                    "--file-allocation=none", "-x", "16", "-s", "16",
                    "--max-tries=0", "--retry-wait=5", "-d", RAW, bed_url], check=True)
assert os.path.getsize(bed_path) == clen and not os.path.exists(bed_path + ".aria2")
print(f"anchors ready: {fn} ({clen/1e6:.2f} MB)")

# ---- 2. anchors -> 100-kb bins (hg19 midpoint; §16.2 binID-1 discipline) ----
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, compression="gzip", dtype=str)
bed.columns = ["chr", "start", "end", "binID"]
bed["bin0"] = pd.to_numeric(bed["binID"]).astype(int) - 1
bed["s"] = bed["start"].astype(int)
bb_all = {c: g.sort_values("s") for c, g in bed.groupby("chr")}

comp = "gzip" if fn.endswith(".gz") else None
anch = pd.read_csv(bed_path, sep="\t", header=None, compression=comp)
anch = anch.iloc[:, :4]; anch.columns = ["chr", "start", "end", "state"]
anch["mid"] = (anch.start + anch.end) // 2
print(f"anchor intervals: {len(anch):,}\n{anch.state.value_counts().to_string()}")
rows = []
for c, g in anch.groupby("chr"):
    bb = bb_all.get(c)
    if bb is None: continue
    pos = np.searchsorted(bb.s.values, g.mid.values, side="right") - 1
    ok = pos >= 0
    rows.append(pd.DataFrame({"chr": c, "bin0": bb.bin0.values[pos[ok]],
                              "state": g.state.values[ok]}))
amap = pd.concat(rows).drop_duplicates(["chr", "bin0"], keep="first")
print(f"anchor->bin mapping: {len(amap):,} bins\n{amap.state.value_counts().to_string()}")

# ---- 3. distexp PC + degeneracy flags + CNV weights ----
pcs = pd.read_csv(PCS_F)
P = pcs.pivot_table(index=["sample", "chr", "bin0"], columns="PC",
                    values="val").reset_index()
P.columns = ["sample", "chr", "bin0"] + PCS
spec = pd.read_csv(SPEC_F)
spec["ratio"] = spec.eig2 / spec.eig1
DEG = set(map(tuple, spec.loc[spec.ratio > 0.85, ["sample", "chr"]].values))
P["degenerate"] = [ (s, c) in DEG for s, c in zip(P["sample"], P["chr"]) ]
print(f"degenerate chr-samples ( eig2/eig1>0.85 ): {sorted(DEG)}")

cnv = pd.read_csv(CNV_F)
print(f"CNV columns: {list(cnv.columns)}")   # §13 rule 5 spirit: inspect columns before join
cand_bin = [x for x in ("bin0", "binID") if x in cnv.columns]
assert cand_bin, "CNV file has no bin column -> paste column names and report"
if cand_bin[0] == "binID":
    cnv["bin0"] = pd.to_numeric(cnv["binID"]).astype(int) - 1
cand_cnv = [x for x in cnv.columns if "log2" in x.lower() or "cnv" in x.lower()]
assert cand_cnv, "CNV value column not identified -> paste column names and report"
cnv = cnv.rename(columns={cand_cnv[0]: "log2cnv"})[["bin0", "log2cnv"]]
P = P.merge(cnv, on="bin0", how="left")
P["w"] = np.clip(1.0 / (1.0 + P.log2cnv.abs().fillna(0.0)), 0.25, 1.0)
print(f"CNV weights: median={P.w.median():.3f} (BRD3179 near-diploid -> near-identity, §16.6 expected)")

# ---- 4. per-chr anchored weighted K-means ----
A_COL = "A1 A2 B1 B2 B3 B4".split()
def run_chr(g):
    g = g.merge(amap[amap.chr == g.name], on=["chr", "bin0"], how="left")
    anc = g.dropna(subset=["state"])
    classes = sorted(set(anc.state) & set(A_COL + list(anc.state.unique())))
    X = g[PCS].values
    if len(anc) >= 30 and anc.state.nunique() >= 3:
        cents = (anc.groupby("state")[PCS].mean()
                 .loc[sorted(anc.state.unique())].values)
        cls = sorted(anc.state.unique())
        km = KMeans(n_clusters=len(cls), init=cents, n_init=1, random_state=RS)
        lab = km.fit_predict(X, sample_weight=g.w.values)
        l2c = {i: cls[int(np.argmin(np.linalg.norm(cents - cen, axis=1)))]
               for i, cen in enumerate(km.cluster_centers_)}
        g["subcomp"] = [l2c[l] for l in lab]
        cn = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        g["conf"] = np.max(xn @ cn.T, axis=1)[np.arange(len(g)), lab]
        g["mode"] = "kmeans"
    else:   # small chr / insufficient anchor coverage: nearest-anchor neighbor (fallback, per §16.2 small-chr caveat)
        if len(anc) == 0:
            g["subcomp"] = None; g["conf"] = 0.0; g["mode"] = "no_anchor"
            return g
        d = np.linalg.norm(X[:, None, :] - anc[PCS].values[None, :, :], axis=2)
        nn = np.argmin(d, axis=1)
        g["subcomp"] = anc.state.values[nn]
        g["conf"] = 1.0 - d[np.arange(len(g)), nn] / (d.max() + 1e-12)
        g["mode"] = "nn_fallback"
    return g

res, aris = [], []
for t in ("N", "T"):
    for c, g in P[P["sample"] == t].groupby("chr"):
        g2 = run_chr(g)
        res.append(g2)
        an = g2.dropna(subset=["state"])
        if len(an) >= 30:
            aris.append({"sample": t, "chr": c, "n_anch": len(an),
                         "ARI": adjusted_rand_score(an.state, an.subcomp)})
L = pd.concat(res, ignore_index=True)
ari = pd.DataFrame(aris)
print("\n=== Pilot 3: per-chr ARI (vs GM12878 anchors) ===")
print(ari.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n* ARI median: " + "  ".join(f"{k}={v:.3f}" for k, v in
      ari.groupby("sample").ARI.median().items()) + "  (gate >= 0.6, N samples primary)")
print(f"  clustering-mode distribution:\n{L.groupby(['sample','mode']).size().to_string()}")

# ---- 5. N/T switch events (labels anchor-aligned, cross-sample comparable) ----
nL = L[L["sample"] == "N"][["chr", "bin0", "subcomp", "conf"]].rename(
    columns={"subcomp": "sub_N", "conf": "conf_N"})
tL = L[L["sample"] == "T"][["chr", "bin0", "subcomp", "conf"]].rename(
    columns={"subcomp": "sub_T", "conf": "conf_T"})
sw = nL.merge(tL, on=["chr", "bin0"]).dropna(subset=["sub_N", "sub_T"])
sw["switch"] = sw.sub_N != sw.sub_T
sw["AB_N"] = sw.sub_N.str[0].map({"A": "A", "B": "B"})
sw["AB_T"] = sw.sub_T.str[0].map({"A": "A", "B": "B"})
sw["AB_switch"] = sw.AB_N != sw.AB_T
sw["high_conf"] = (sw.conf_N >= 0.6) & (sw.conf_T >= 0.6)
sw = sw.merge(P[P["sample"] == "N"][["chr", "bin0", "log2cnv", "degenerate"]],
              on=["chr", "bin0"], how="left")
print(f"\n=== switch events (labeled in both samples: {len(sw):,} bins) ===")
for hc in (False, True):
    s = sw[sw.high_conf] if hc else sw
    ab_sw = s[s.AB_switch]; sub_sw = s[s.switch & ~s.AB_switch]
    print(f"  {'high-confidence' if hc else 'all'}: total switches={len(s[s.switch]):,}  "
          f"A/B switches={len(ab_sw):,}  AB-stable but subtype-switched={len(sub_sw):,}")
hc_sw = sw[sw.high_conf & sw.switch & ~sw.AB_switch]
print(f"  median CNV of AB-stable-but-subtype-switched (high-confidence)={hc_sw.log2cnv.median():.3f} "
      f"(dosage attribution later per §16.6 method)")
L.to_csv(os.path.join(OUT, "cnrsc_layer3_labels_100kb.csv"),
         index=False, encoding="utf-8")
sw.to_csv(os.path.join(OUT, "cnrsc_layer3_switch_events.csv"),
          index=False, encoding="utf-8")
ari.to_csv(os.path.join(OUT, "cnrsc_layer3_pilot3_ARI.csv"),
           index=False, encoding="utf-8")
print("\nwritten: cnrsc_layer3_labels/switch_events/pilot3_ARI")
