# -*- coding: utf-8 -*-
# a28e_bw_extract.py v2 — R-tree self-calibration + data-block level audit fixes
# v1 failure chain (3 rounds): layout hardcoded from memory, R-tree item stride
#   misremembered -> child pointer out of bounds -> short-read crash.
# v2: layout self-calibration (candidate stride scan + hard interval
#   constraints); data-block layer fixes the 24-byte header and the type-1
#   interleaved structure
import struct, zlib, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, "data", "raw")
RES  = os.path.join(ROOT, "results")
bed = pd.read_csv(os.path.join(RAW, "GSE133928_binAnnotation100Kb.bed.gz"),
                  sep="\t", header=None, names=["chrom", "bstart", "bend", "bin"])
lab_raw = pd.read_csv(os.path.join(RES, "a22d_subcomp_labels_chrcenter.csv"))
print(f"a22d columns: {lab_raw.columns.tolist()}")
lab = lab_raw[["bin", "cluster"]].copy()
D = lab.merge(bed, on="bin", how="inner")
assert not any(c.endswith(("_x", "_y")) for c in D.columns), "merge still has suffix columns"
print(f"labels ∩ bed: {len(D)}")
chrom_groups = {}
for c, g in D.groupby("chrom"):
    g = g.sort_values("bstart")
    chrom_groups[c] = (g.bstart.values.astype(np.int64),
                       g.bend.values.astype(np.int64),
                       g.bin.values.astype(np.int64))
acc = {c: {"bs": v[0], "be": v[1], "bid": v[2],
           "sums": np.zeros(len(v[0])), "w": np.zeros(len(v[0]))}
       for c, v in chrom_groups.items()}

def read_at(f, off, n):
    f.seek(off); return f.read(n)

def node_header(f, off):
    b = read_at(f, off, 4)
    if len(b) < 4: return None, None
    isLeaf, _r, cnt = struct.unpack("<BBH", b)
    return isLeaf, cnt

def calib_internal(raw, cnt, lo, hi):
    # internal items: child pointer = last 8 bytes of each item; candidate
    # strides 16/20/24/28/32; hard constraint: all pointers within (lo, hi)
    hits = []
    for stride in (16, 20, 24, 28, 32):
        if cnt * stride > len(raw): continue
        offs = [int.from_bytes(raw[i*stride+stride-8: i*stride+stride], "little")
                for i in range(cnt)]
        if all(lo < v < hi for v in offs): hits.append(stride)
    return hits

def calib_leaf(raw, cnt, lo, hi):
    # leaf item candidates (stride, doff field offset): (24,12)/(28,16)/(32,16)
    hits = []
    for stride, p in ((24, 12), (28, 16), (32, 16)):
        if cnt * stride > len(raw): continue
        ok = True
        for i in range(cnt):
            o = int.from_bytes(raw[i*stride+p: i*stride+p+8], "little")
            s = int.from_bytes(raw[i*stride+p+8: i*stride+p+12], "little")
            if not (lo < o < hi and s > 0 and o + s <= hi): ok = False; break
        if ok: hits.append((stride, p))
    return hits

def walk_rtree(f, root_off, filesize, dataOff, idxOff):
    blocks = []; stack = [root_off]; seen = set()
    _int_stride = None; _leaf_layout = None
    while stack:
        off = stack.pop()
        if off in seen: continue
        seen.add(off)
        isLeaf, cnt = node_header(f, off)
        assert isLeaf is not None, f"R-tree node short read: off={off}, filesize={filesize}"
        raw = read_at(f, off + 4, cnt * 48)
        if isLeaf:
            if _leaf_layout is None:
                hits = calib_leaf(raw, cnt, dataOff, idxOff)
                if not hits:
                    print(f"  leaf calibration failed (cnt={cnt}); first 256 bytes of node:")
                    print("  " + raw[:256].hex())
                assert hits, "leaf item layout calibration failed"
                _leaf_layout = hits[0]
                print(f"  R-tree leaf item calibration: stride/doff-offset={_leaf_layout} (candidates {hits})")
            stride, p = _leaf_layout
            for i in range(cnt):
                doff, dsz = struct.unpack_from("<QI", raw, i*stride + p)
                blocks.append((doff, dsz))
        else:
            if _int_stride is None:
                hits = calib_internal(raw, cnt, idxOff, filesize)
                if not hits:
                    print(f"  internal calibration failed (cnt={cnt}); first 256 bytes of node:")
                    print("  " + raw[:256].hex())
                assert hits, "internal item layout calibration failed"
                _int_stride = hits[0]
                print(f"  R-tree internal item calibration: stride={_int_stride} (candidates {hits})")
            stride = _int_stride
            for i in range(cnt):
                (child,) = struct.unpack_from("<Q", raw, i*stride + stride - 8)
                stack.append(child)
    return blocks

def parse_bw(fpath):
    f = open(fpath, "rb")
    filesize = os.path.getsize(fpath)
    hdr = read_at(f, 0, 64)
    magic, ver, zooms, chrOff, dataOff, idxOff, fc, dfc, _a, _s, bufSize, _r2 = \
        struct.unpack("<IHHQQQHHQQIQ", hdr)
    assert magic == 0x888FFC26, "bad magic number"
    print(f"  filesize={filesize/1e6:.1f}MB chrOff={chrOff} "
          f"dataOff={dataOff} idxOff={idxOff} bufSize={bufSize}")
    # --- chrom B+ tree (battle-tested) ---
    bhdr = read_at(f, chrOff, 32)
    _m, _bs, keySize, _vs, _cnt, _res = struct.unpack("<IIIIQQ", bhdr)
    assert _m == 0x78CA8C91, f"chrom B+ tree bad magic: {hex(_m)}"
    name2id = {}
    stack = [chrOff + 32]
    while stack:
        off = stack.pop()
        isLeaf, cnt = node_header(f, off)
        pos = off + 4
        if isLeaf:
            item = keySize + 8
            raw = read_at(f, pos, cnt * item)
            for i in range(cnt):
                key = raw[i*item: i*item + keySize].rstrip(b"\x00").decode()
                cid, csz = struct.unpack_from("<II", raw, i*item + keySize)
                name2id[key] = cid
        else:
            item_in = keySize + 8
            raw = read_at(f, pos, cnt * item_in)
            for i in range(cnt):
                (child,) = struct.unpack_from("<Q", raw, i*item_in + keySize)
                stack.append(child)
    id2name = {v: k for k, v in name2id.items()}
    print(f"  chroms={len(name2id)}, sample={list(name2id)[:5]}")
    # --- R-tree (self-calibrating) ---
    rhdr = read_at(f, idxOff, 48)
    _m, _blk, itemCount, *_rest = struct.unpack("<IIQIIIIQII", rhdr)
    print(f"  R-tree magic={hex(_m)} blockSize={_blk} itemCount={itemCount}")
    blocks = walk_rtree(f, idxOff + 48, filesize, dataOff, idxOff)
    print(f"  leaf blocks={len(blocks)} (declared {itemCount}), doff range="
          f"{min(b[0] for b in blocks)/1e6:.1f}-"
          f"{max(b[0]+b[1] for b in blocks)/1e6:.1f}MB")
    # --- data block parsing + vectorized overlap accumulation (audit-fixed) ---
    dt1 = np.dtype([("s", "<u4"), ("e", "<u4"), ("v", "<f4")])
    dt2 = np.dtype([("s", "<u4"), ("v", "<f4")])
    n_iv = 0; skipped4 = 0; bad_blocks = 0
    for bi, (doff, dsz) in enumerate(blocks):
        raw = read_at(f, doff, dsz)
        if bufSize > 0:
            try: raw = zlib.decompress(raw)
            except zlib.error: pass
        if len(raw) < 24: continue
        cid, s, e, step, span, typ = struct.unpack_from("<IIIIIB", raw, 0)
        A = acc.get(id2name.get(cid))
        if A is None: continue
        if typ == 1:
            n = (len(raw) - 24) // 12
            it = np.frombuffer(raw, dt1, n, 24)
            S = it["s"].astype(np.int64); E = it["e"].astype(np.int64)
            V = it["v"].astype(np.float64)
        elif typ == 2:
            n = (len(raw) - 24) // 8
            it = np.frombuffer(raw, dt2, n, 24)
            S = it["s"].astype(np.int64); V = it["v"].astype(np.float64)
            E = S + max(span, 1)
        elif typ == 3:
            n = (e - s) // max(step, 1)
            V = np.frombuffer(raw, "<f4", n, 24).astype(np.float64)
            S = s + np.arange(n, dtype=np.int64) * step
            E = S + span
        else:
            skipped4 += 1; continue
        ok = np.isfinite(V) & (E >= S) & (S >= s) & (E <= e)
        if len(S) > 10:
            frac = ok.mean()
            if frac < 0.5:
                bad_blocks += 1
                if bad_blocks <= 3:
                    print(f"  warning: block {bi} entry validation pass rate={frac:.2f} (typ={typ})")
        S, E, V = S[ok], E[ok], V[ok]
        if len(S) == 0: continue
        n_iv += len(S)
        bs, be = A["bs"], A["be"]
        lo = np.searchsorted(be, S, "right")
        hi = np.searchsorted(bs, E, "left")
        cnts = np.maximum(hi - lo, 0)
        tot = int(cnts.sum())
        if tot == 0: continue
        rep = np.repeat(np.arange(len(S)), cnts)
        within = np.arange(tot) - np.repeat(np.cumsum(cnts) - cnts, cnts)
        idx = np.repeat(lo, cnts) + within
        ov = np.minimum(E[rep], be[idx]) - np.maximum(S[rep], bs[idx])
        np.add.at(A["sums"], idx, V[rep] * ov)
        np.add.at(A["w"], idx, ov)
        if (bi + 1) % 5000 == 0:
            print(f"  block {bi+1}/{len(blocks)}...", flush=True)
    f.close()
    print(f"  total intervals={n_iv}, type-4 skipped blocks={skipped4}, "
          f"validation-abnormal blocks={bad_blocks}")
    return acc

TRACKS = [("BRD3179N_K27me3", "GSM3930224_BRD3179N-K27me3.bw", True),
          ("BRD3187N_K27me3", "GSM3930233_BRD3187N-K27me3.bw", False),
          ("BRD3187_K9me3",   "GSM3930232_BRD3187-K9me3.bw",   False)]
for tag, fn, primary in TRACKS:
    fp = os.path.join(RAW, fn)
    print(f"\n[{tag}] {fn}")
    acc = parse_bw(fp)
    rows = []
    for c, A in acc.items():
        m = A["w"] > 0
        rows.append(pd.DataFrame({"chrom": c, "bin": A["bid"][m],
                                  "mean": A["sums"][m] / A["w"][m]}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(RES, f"a28e_{tag}_binsig.csv"),
               index=False, encoding="utf-8-sig")
    print(f"  covered labeled bins={len(out)}/{len(D)} -> a28e_{tag}_binsig.csv")
print("\nAll tracks parsed -> python a28d_bin_enrich.py")
