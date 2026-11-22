#!/usr/bin/env python3
"""Rewrite the GReLU table of an NPUC-compiled NCP (PORT-32 N3d; local, for the POC: our own tool, our own NCP).

What NPUC writes (V, libnpuc_backend NPU2020::NcpMtcConvolution::ncp_mtc_set_file @0x1422d8, ";GReLU configuration",
when conv mode (byte +0x1a4) is 2 or 3): NFUBR 0x40163800 = 0xff, then 16 registers 0x40163044..0x40163080, each
(int16 A) | (uint16 B) << 16, and in mode 3 two more, 0x40163084 and 0x40163088, then 0x40163800 = 0.
Reading (G, from the values): four 4-entry groups of (slope B, offset A) pieces with the thresholds T1 = 0x...84 and
T2 = 0x...88: piece 0 below T1, piece 1 between T1 and T2, piece 2 above T2. The stock NFD.ncp (runs correctly):
group 0 = (0,0) (1,0) (0,0x60) (0,0), T1 0, T2 0x60 (a ReLU6 in its fixed point). Ours, no activation (quant 3):
every group (16,0) (16,0) (0,0) (0,0), T1 = T2 = 0: above 0 the piece is (0,0), which is exactly the defect measured
(every acc >= 0 comes back 0; in quant 1 the pieces are (16,0x80) (16,0x80) (0,0) (0,0) and acc >= 0 gives the
byte 0, not the zero point). Our ReLU compile: group 0 all (0,0), groups 1-3 (16,0) (16,0) (0,0) (0,0): all zeros out.
The instructions are 6 bytes (u32 value, u16 opcode; ncp-isa.py); only value fields are rewritten, so the NCP keeps
its size, layout and every other byte.

Several convolutions = several GReLU blocks (one table each, in instruction order): every spec works block by block,
from that block's own values (npu-run, 2026-11-21: the 3-layer example has two).
The ReLU table (npu-run run 20261121-132851, its OFM all the zero point after "piece2"): NPUC writes group 0 = the
constant zero point (A = zp, B = 0) in pieces 0 and 1, and the identity (zp, slope) in groups 1-3; the stock NFD uses
group 0 only (groups 1-3 zero). So NPUC's identity piece sits one group too far: "relu" puts it in group 0's pieces 1
and 2 (piece 0, below T1 = 0, stays the zero point) and makes groups 1-3 copies of group 0.

  ncp-grelu-patch.py <in.ncp> <out.ncp> <spec>
    spec "all-pieces":   every (0,0) piece of a group whose piece 0 is (B,A) != (0,0) becomes (B,A): one line, no
                         threshold matters (the bypass made identity everywhere)
    spec "piece2":       only piece 2 of each group becomes its piece 0 (the "above T2" piece)   [no activation: V]
    spec "relu":         a ReLU block (group 0 piece 0 slope 0, an identity piece in group 1): group 0 = zp, id, id,
                         piece 3 kept; groups 1-3 = group 0                                        [G, to be run]
    spec "auto":         per block: "relu" for a ReLU table, "piece2" for a no-activation one, else untouched; an NCP
                         without any GReLU block is copied unchanged (a pooling-only network)
    spec "t2max":        T2 (0x40163088) = 0x7fffffff, the table untouched (piece 1 then covers 0 <= x)
    spec "reg=val,...":  explicit, e.g. 0x4016304c=0x00100000 (every block)
Prints every change; exit 1 if a register the spec needs is not in the NCP.
"""
import hashlib
import struct
import sys

BASE = 0x40160000
TABLE = [0x40163044 + 4 * i for i in range(16)]
T1, T2 = 0x40163084, 0x40163088


def slots(d):
    """{reg: [file offset of each 6-byte set_reg instruction writing it]} for the NFUBR block"""
    h = struct.unpack_from("<32I", d, 0)
    gvo, gvc, body = h[14], h[15], h[17]
    res = {}
    for g in range(gvc):
        gv = struct.unpack_from("<10I", d, gvo + 40 * g)
        off, size = body + gv[8], gv[9]
        for i in range(size // 6):
            o = off + 6 * i
            val, op = struct.unpack_from("<IH", d, o)
            if op >> 12 == 0xA and op >> 8 not in (0x3A, 0x2B):
                res.setdefault(BASE + (op & 0xFFF) * 4, []).append(o)
    return res


def kind(vals):
    """a block's 16 table values -> "relu", "none" or "other" """
    g = [vals[4 * i:4 * i + 4] for i in range(4)]
    if g[0][0] >> 16 == 0 and g[1][0] >> 16 != 0 and g[0][2] == 0:
        return "relu"
    if g[0][0] >> 16 != 0 and g[0][2] == 0:
        return "none"
    return "other"


def block_want(vals, spec):
    """{table index: new value} for one block"""
    want = {}
    if spec in ("all-pieces", "piece2"):
        for gi in range(4):
            p0 = vals[4 * gi]
            if p0 == 0:
                continue
            for k in range(4):
                if vals[4 * gi + k] == 0 and (spec == "all-pieces" or k == 2):
                    want[4 * gi + k] = p0
    elif spec == "relu":
        ident = vals[4]
        g0 = [vals[0], ident, ident, vals[3]]
        for gi in range(4):
            for k in range(4):
                if vals[4 * gi + k] != g0[k]:
                    want[4 * gi + k] = g0[k]
    return want


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, spec = sys.argv[1:]
    d = bytearray(open(src, "rb").read())
    s = slots(d)
    nb = len(s.get(TABLE[0], []))
    if spec == "auto" and not any(r in s for r in TABLE):
        open(dst, "wb").write(d)
        print("ncp-grelu-patch: %s (auto): no GReLU block, copied unchanged -> %s, %d B, sha256 %s" % (
            src, dst, len(d), hashlib.sha256(d).hexdigest()[:16]))
        return
    missing = [r for r in TABLE + [T1, T2] if r not in s]
    if missing:
        sys.exit("ncp-grelu-patch: the NCP has no GReLU write for %s" % ", ".join("%#x" % r for r in missing))
    if any(len(s[r]) != nb for r in TABLE):
        sys.exit("ncp-grelu-patch: the table registers are written %s times: not whole blocks" % sorted({len(s[r]) for r in TABLE}))
    rd = lambda o: struct.unpack_from("<I", d, o)[0]
    changes = []  # (reg, file offset, new value)
    for b in range(nb):
        vals = [rd(s[r][b]) for r in TABLE]
        k = kind(vals)
        sp = spec
        if spec == "auto":
            sp = {"relu": "relu", "none": "piece2"}.get(k)
            print("ncp-grelu-patch: block %d: %s table -> %s" % (b, k, sp or "untouched"))
        elif spec == "relu" and k != "relu":
            sys.exit("ncp-grelu-patch: block %d is not a ReLU table (%s): %s" % (b, k, " ".join("%#x" % v for v in vals)))
        if sp in ("all-pieces", "piece2", "relu"):
            for i, v in sorted(block_want(vals, sp).items()):
                changes.append((TABLE[i], s[TABLE[i]][b], v))
        elif sp == "t2max":
            changes.append((T2, s[T2][min(b, len(s[T2]) - 1)], 0x7FFFFFFF))
        elif sp is not None:
            for kv in sp.split(","):
                r, v = kv.split("=")
                r, v = int(r, 0), int(v, 0)
                if r not in s:
                    sys.exit("ncp-grelu-patch: %#x is not written by this NCP" % r)
                if b < len(s[r]):
                    changes.append((r, s[r][b], v))
    for r, o, v in changes:
        print("ncp-grelu-patch: %#010x at file offset %#x: %#010x -> %#010x" % (r, o, rd(o), v))
        struct.pack_into("<I", d, o, v)
    open(dst, "wb").write(d)
    print("ncp-grelu-patch: %s (%s) -> %s, %d B, %d blocks, sha256 %s -> %s" % (src, spec, dst, len(d), nb,
          hashlib.sha256(open(src, "rb").read()).hexdigest()[:16], hashlib.sha256(d).hexdigest()[:16]))


if __name__ == "__main__":
    main()
