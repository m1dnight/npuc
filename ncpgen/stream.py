#!/usr/bin/env python3
"""The CMDQ instruction streams of an NCP's groups, generated from npunet's lowered form, byte for byte as Samsung's
NPUC 1.36.1.a emits them (docs/format/cmdq.md holds the register map and its evidence).

  stream.py check <variant dir>...     compare group_streams(<dir>/lowered.json) with the isa bytes of <dir>/net.ncp
                                       (compiled by the oracle, oracle/npuc-local.sh; nothing is compiled
                                       here): one line per variant (exact / differs / unsupported) and a total
  stream.py show <variant dir> [--diff]   our streams as text (with --diff: only the instructions that differ)
  stream.py events <variant dir>...    replay NPUC's event schedule from <dir>/log.txt's memory profile and compare
                                       its ev2set lines and waits with NPUC's streams (cmdq.md 7b)

Interface: group_streams(low, fixed=False) -> list[bytes], low = lowered.json's dict; fixed is grelu.grelu_regs'
(True: the corrected activation table, "robust", False: NPUC's own). plan_dmas(low) lists the DMAs for frame.py.

Instruction encoding (V, NPUC's compile log): 6 bytes, u32 value then u16 opcode. set_reg: opcode = block << 12 |
word index (CMDQ 8 @0x401e0000, MTC 9 @0x40120000, NFUBR a @0x40160000, INTCOM b @0x40100000); nop = 0/0;
waiti3(irN, rM, v) = 0x3a00 | N << 4 | M; wait3(rN, rM, v) = 0x2b00 | N << 4 | M (value v in both).

The model, in NPUC's own terms (its log: IR elements, events, SRAM chunks), for the family it covers: chains of
convolutions (stride 1 or 2, kernels up to 13, SAME or VALID, any channel count up to 255, several output passes,
weight sub-batches with partial sums, weight rings) and max/avg pools, whole or tiled (NPUC's strata from
frame.plan: per-tile groups with trim, then every layer's pad, reformat, conv or pool; cmdq.md 7b), and untiled
graphs (npuc-net 2: eltwise add as an ARM intrinsic group, concat in DRAM or SRAM, standalone ReLU/ReLU6 as NPUC's
dummy conv; cmdq.md 7c). The activation table is grelu.py's, the weight batches' rows and sub-batches weights.py's.
Anything outside raises Unsupported rather than guess.
"""
import argparse
import json
import os
import re
import sys

if not __package__:     # run as a script: the package is this file's directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncpgen import frame, grelu, weights  # noqa: E402
from ncpgen.common import (CMDQ, INTC, MTC, NFU, Memo, cdiv, concat_in_sram, decode_isa, encode_isa,  # noqa: E402
                           exec_order, is_graph, multiplier, ncp_parts, network_key, requant, scale_value, set_reg_addr,
                           set_reg_op, GRAPH_KINDS)


class Unsupported(Exception):
    """a network this module cannot generate the streams of (not modelled: refused rather than guessed)"""


class NeedsTiling(Unsupported):
    """a layer NPUC tiles where the planner expected it whole"""


# ---- instruction encoding -------------------------------------------------------------------------------------------
R = {i: CMDQ + 0x100 + 4 * i for i in range(1, 16)}       # r1..r15 (V: names in NPUC's log)
EV2SET, EV2, SWI = CMDQ + 0x2024, CMDQ + 0x2028, CMDQ + 0x180
GRELU_GUARD = CMDQ + 0x50


class Code:
    """a group's instruction list: (opcode, value) pairs"""

    def __init__(self):
        self.ins = []

    def w(self, addr, val):
        """set_reg addr = val"""
        self.ins.append((set_reg_op(addr), val & 0xFFFFFFFF))

    def nop(self, n=1):
        self.ins.extend([(0, 0)] * n)

    def waiti3(self, ir, reg, val):
        self.ins.append((0x3A00 | ir << 4 | reg, val))

    def wait3(self, ra, rb, val):
        self.ins.append((0x2B00 | ra << 4 | rb, val))

    def wait_event(self, mask, trailing=True):
        """wait for DMA events (V: r1, 4 nops, waiti3 ir6, 4 nops, clear with ev2, 4 nops)"""
        self.w(R[1], mask)
        self.nop(4)
        self.waiti3(6, 1, mask)
        self.nop(4)
        self.w(EV2, mask)
        if trailing:
            self.nop(4)

    def bytes(self):
        return encode_isa(self.ins)


# ---- how NPUC spreads channels over the 4 cores ----------------------------------------------------------------------
class Dist:
    """channel planes (2 channels each) over the cores: ppc planes per core, the first `cores` cores used (V: core
    masks and per-core registers of the ci*/co* variants)"""

    def __init__(self, c):
        self.c = c
        self.planes = cdiv(c, 2)
        self.ppc = cdiv(self.planes, 4)
        self.cores = cdiv(self.planes, self.ppc)
        self.mask = (1 << self.cores) - 1


def cells(h, w):
    """4-line x 8-column cells of one channel plane (64 bytes each, V)"""
    return cdiv(h, 4) * cdiv(w, 8)


# ---- the GReLU hook ----------------------------------------------------------------------------------------------
def layer_shift(layer):
    """the requantisation shift of a conv (NPUC's ScaleONFiller::setScale, common.requant)"""
    return requant(multiplier(*(scale_value(layer[k]) for k in ("in_scale", "w_scale", "out_scale"))))[2]


def grelu_writes(layer, quant_mode, p, fixed=False, cache=None):
    """the activation table's writes for pass p of a conv, CMDQ 0x401e0050 = 1 ... = 0 included: grelu.grelu_regs
    (every pass's block), split here at the 0x401e0050 brackets. cache: a dict kept while one plan is emitted (the
    layer's blocks are made once, not once per pass)"""
    hit = cache.get((id(layer), quant_mode, fixed)) if cache is not None else None
    if hit is not None and hit[0] is layer:
        blocks = hit[1]
    else:
        blocks = _grelu_blocks(layer, quant_mode, fixed)
        if cache is not None:
            cache[(id(layer), quant_mode, fixed)] = (layer, blocks)
    if p >= len(blocks):
        raise Unsupported("grelu.py gave %d blocks, pass %d wanted" % (len(blocks), p))
    return blocks[p]


def _grelu_blocks(layer, quant_mode, fixed):
    regs = list(grelu.grelu_regs(layer, quant_mode, fixed=fixed))
    blocks, cur = [], None
    guard = any(a == GRELU_GUARD for a, v in regs)
    start = (GRELU_GUARD, 1) if guard else (NFU + 0x3800, 0xFF)
    end = (GRELU_GUARD, 0) if guard else (NFU + 0x3800, 0)
    for a, v in regs:
        if (a, v) == start:
            cur = []
        cur.append((a, v))
        if (a, v) == end:
            blocks.append(cur if guard else [(GRELU_GUARD, 1)] + cur + [(GRELU_GUARD, 0)])
            cur = None
    return blocks


# ---- the weight blocks' geometry -------------------------------------------------------------------------------------
def weight_rows(L, q):
    """(rows of 64 B per weight batch, sub-batches, channels per sub-batch) of a conv: weights.py's packing (a packing
    it refuses is refused here)"""
    try:
        geo = weights.conv_block(L, q)[0]
    except ValueError as e:
        raise Unsupported("weights.py: %s" % e)
    return geo["rows_per_batch"], geo["sub_batches"], geo["sub_channels"][0]


def out_channel(dout, core, p, slot, c):
    """the output channel computed by `core` in pass p, plane slot `slot`, pair member c (V: bias/shift registers of
    co1..co32), or None"""
    if core >= dout.cores:
        return None
    plane_in_core = 2 * p + slot
    if plane_in_core >= dout.ppc:
        return None
    plane = core * dout.ppc + plane_in_core
    ch = 2 * plane + c
    return ch if ch < dout.c else None


# ---- NFU width pieces ------------------------------------------------------------------------------------------------
NFU_MAX_W = 112      # an NFU pad/pool/trim command reads at most 112 input columns (V: w112/w116, w*pool, dw440)
NFU_MAX_H = 248      # G: and at most 248 input rows (dh480 241 rows one command, dh500 252 and 253 rows two)
RU_MAX_W = 128       # V: an NFU reformat command reads at most 128 input rows and columns (w128s2, w256s2, w300s2,
#                      the stems 224x224x3: pieces row-major)


def nfu_pieces(w, wo, k, s, pl, pr, unit=8, limit=None):
    """the pieces of an NFU pad (k = s = 1), trim (pl = -skip) or pool command along one axis: n = ceil(input / 112)
    for columns (unit 8), ceil(input / 248) for rows (unit 4), pieces of round_up(ceil(wo / n), unit) outputs (V:
    w104..w250, w*pool, w*s2, the 225-wide stems (2 x 120), dw440 (225 -> 3 x 80); rows: dh500 253 and 252 -> 2,
    dh480 241 -> 1, G for the 248); each reads from the cell holding its first input (skip = the inputs before it in
    that cell); the leading pad on the first piece only, the trailing one on the last only"""
    limit = limit or (NFU_MAX_W if unit == 8 else NFU_MAX_H)
    n = cdiv(w, limit)
    pw = cdiv(cdiv(wo, n), unit) * unit
    out = []
    o0 = 0
    while o0 < wo:
        o1 = min(wo, o0 + pw)
        a = max(0, o0 * s - pl)
        b = min(w, (o1 - 1) * s + k - pl)
        out.append({"o0": o0, "o1": o1, "a": a, "b": b, "cell": a // unit, "skip": a % unit,
                    "first": o0 == 0, "last": o1 == wo})
        o0 = o1
    return out


def nfu_grid(h, ho, kh, sh, pt, pb, w, wo, kw, sw, pl, pr):
    """row pieces x column pieces of an NFU command, rows outer (G for the order: rows split only in dh500)"""
    return [(rc, pc) for rc in nfu_pieces(h, ho, kh, sh, pt, pb, 4) for pc in nfu_pieces(w, wo, kw, sw, pl, pr)]


SRAM_LIMIT = 0x40000  # bytes per core NPUC lets a layer (or a tile) use before it tiles (V: t200 vs t208, ts128c32)


def weight_ring(d):
    """pass slots the weight chunk holds: every pass when the conv node fits (its input, output, partial sums and
    weights within 256 KiB), else a ring of R slots, R_max = what fits beside the feature maps and R balanced over
    the rounds, R = ceil(passes / ceil(passes / R_max)) (V: c256 6 of 16, r_c256h24 8, r_c320h24 7 of 20,
    r_c512h16 7 of 32, r_c512h24 2; all: r_c256h16, r_c192h32, co512)"""
    slot = 2 * d["wunits"] * 64
    cin = d["ref"] if d.get("ref") else d["pad"] if d.get("pad") else d["in"]
    fm = cin.size + d["out"].size + (d["ps"].size if d.get("ps") else 0)
    passes = d["passes"]
    if fm + passes * slot <= SRAM_LIMIT:
        return passes
    rmax = (SRAM_LIMIT - fm) // slot
    if rmax < 1:
        raise NeedsTiling("tiling (no weight slot fits beside %d B of feature maps)" % fm)
    return cdiv(passes, cdiv(passes, rmax))


def unflatten_tiles(d, ho, wo):
    """row tiles of the output read-back (the unflatten): NPUC's policy on the output node (getBestProcessingParams,
    frame.md 6.10, frame.py's _unflatten_tiles; V on every output node's log, 2080 compiles): whole when its input
    (the output's planes, 16-rounded channels behind a conv) and its output, per core in cells, fit 256 KiB together;
    else the fewest tileable row counts n >= 2 (n tiles of round_up(ceil(H / n), 4) rows, the last the rest, all
    non-empty) whose first tile fits twice in 256 KiB (V: ts208 one; t2d-300x300 two; rb120w72 three; tiling3
    tr-2nd four, tr-320 six). d: "dout", and "conv" (or "passes") when a conv makes the output"""
    dout = d["dout"]
    din = Dist(cdiv(dout.c, 16) * 16) if d.get("conv", "passes" in d) else dout
    if (din.ppc + dout.ppc) * cells(ho, wo) * 64 <= SRAM_LIMIT:
        return 1
    for n in range(2, cdiv(ho, 4) + 1):
        t = cdiv(cdiv(ho, n), 4) * 4
        if not (t * (n - 1) < ho):         # calcBasicTileLengths: exactly n non-empty tiles
            continue
        if 2 * dout.ppc * cells(t, wo) * 64 <= SRAM_LIMIT:
            return n
    raise Unsupported("no row tiling of the %d x %d read-back" % (ho, wo))


# ---- planning: groups, IR elements, events, SRAM ---------------------------------------------------------------------
class Op:
    """an IR element of a group: its kind, its dependencies (indices in the group) and its fields"""
    def __init__(self, kind, **kw):
        self.kind = kind
        self.deps = []
        self.__dict__.update(kw)


class Chunk:
    """an SRAM region (bytes per core) with its live range in node order (NPUC's liveness analysis, its log)"""

    def __init__(self, cid, name, size, live, batch, tail=False):
        self.cid, self.name, self.size, self.live, self.batch = cid, name, size, live, batch
        self.tail = tail                        # placed after the batch's other chunks (partial sums)
        self.off = None


def _place(c, placed, align=64):
    busy = sorted((m.off, m.off + m.size) for m in placed
                  if m.live[0] <= c.live[1] and c.live[0] <= m.live[1] and m.size)
    o = 0
    for x, y in busy:
        if o + c.size <= x:
            break
        o = max(o, cdiv(y, align) * align)
    c.off = o


def _next_permutation(a):
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return True


def allocate(chunks, limit=None):
    """NPUC's PolicyMaker allocation (V: its "Request ... Allocated" log lines over the sweeps): batch by batch (the
    first citizens, the input, then each layer's chunks), each chunk at the lowest 64-aligned offset clear of every
    placed chunk whose live range (inclusive, in node order) overlaps. Within a batch it tries "bigger first" (size
    descending, equal sizes the later chunk first; partial sums last), then "backward" (the reverse, partial sums
    still last), then every next permutation of the backward order, and keeps the first order in which no chunk
    ends past the limit (V: r_c256co128, six orders tried)"""
    limit = SRAM_LIMIT if limit is None else limit
    placed = []
    for bt in sorted(set(c.batch for c in chunks)):
        batch = [c for c in chunks if c.batch == bt]
        main = sorted((c for c in batch if not c.tail), key=lambda c: (-c.size, -c.cid))
        tail = [c for c in batch if c.tail]
        first = main + tail
        back = list(reversed(main)) + tail
        idx = list(range(len(back)))
        tries = [first, back]
        ok = False
        n = 0
        while True:
            order = tries.pop(0) if tries else None
            if order is None:
                if not _next_permutation(idx):
                    break
                order = [back[i] for i in idx]
            mine = []
            for c in order:
                if getattr(c, "fixed", False):  # placed by the caller (the graph's first citizens)
                    pass
                elif getattr(c, "top", False):  # a concat in SRAM: at the top (V: cat16, cat-32-16)
                    c.off = (limit - c.size) // 64 * 64
                else:
                    _place(c, placed + mine)
                mine.append(c)
            if all(c.off + c.size <= limit for c in mine):
                ok = True
                break
            n += 1
            if n > 5040:
                break
        if not ok:                              # nothing fits: bigger first, the caller sees the overflow
            mine = []
            for c in first:
                _place(c, placed + mine)
                mine.append(c)
        placed += mine


class Plan:
    """groups of IR elements as NPUC builds them for a chain of layers (V: its log's IR lists, "create group" lines):
    one group per layer (the first also loads the input, the last stores the output; feature maps between layers stay
    in SRAM), then the unflatten group"""

    def __init__(self, low):
        self.low = low
        self.q = low["quant_mode"]
        layers = low["layers"]
        if not layers:
            raise Unsupported("no layer")
        self.ev = 0
        self.groups = []
        self.grelu_cache = {}       # grelu_writes' blocks while this plan is emitted
        if is_graph(low):
            self.graph()
            return
        prev = low["input"]["name"]
        for L in layers:
            if L["input"] != prev:
                raise Unsupported("not a chain (%s reads %s)" % (L["name"], L["input"]))
            prev = L["name"]
            if L["kind"] == "conv":
                if L["stride"] not in ([1, 1], [2, 2]):
                    raise Unsupported("conv stride %s" % L["stride"])
            elif L["kind"] != "pool":
                raise Unsupported("layer kind %s" % L["kind"])
        fp = self.frame_plan()
        if fp is not None:
            self.from_frame(fp)
            return
        # untiled: one SRAM-forwarded run of every layer, loading from and storing to DRAM (a layer NPUC tiles raises
        # NeedsTiling: frame.plan tiles every network NPUC tiles)
        self.layout(layers, final=True, inp=tuple(low["input"]["chw"]))
        for li, L in enumerate(layers):
            if "copy" in self.info[li]:
                self.groups.append(self.copy_group(self.info[li]))
            ev0 = self.ev
            self.groups += self.cut(self.layer_group(li, L, first=li == 0, last=li == len(layers) - 1), ev0)
        last = self.info[-1]
        nu = unflatten_tiles(last, last["ho"], last["wo"])
        dout = last["dout"]
        self.out_chw = (dout.c, last["ho"], last["wo"])     # NPUC's own output size (the read-back's profile)
        for u in range(nu):
            # a row-tiled unflatten loads per core when the output has several planes per core (V: ts192c16)
            nl = dout.cores if nu > 1 and dout.ppc > 1 else 1
            ops = [Op("load", what="dram", flag=self.event(), utile=len(self.groups), core=k if nl > 1 else None,
                      rows=(u * cdiv(cdiv(last["ho"], nu), 4) * 4,
                            min(last["ho"], (u + 1) * cdiv(cdiv(last["ho"], nu), 4) * 4))) for k in range(nl)]
            self.groups.append(ops + [Op("nopop", deps=list(range(nl))), Op("unflatten")])

    # ---- tiled plans from frame.py -----------------------------------------------------
    def frame_plan(self):
        """frame.plan(low) when NPUC tiles this network: frame.py's tiling decisions (strata, tiles, back-propagated
        ranges, SRAM chunks; V on its sweep), so that both halves of the NCP agree. None (the planner below) when
        frame.plan refuses the network or does not tile it."""
        try:
            fp = frame.plan(self.low)
        except Exception:
            return None
        if not fp.get("tiled"):
            return None
        # the strata are NPUC's own (its policy, read from its code; frame.md 6.10): halo pools (final-chains f012,
        # tiling2 g*), convs with weight sub-batches (f002, f059; their PARTIAL_SUM chunks in st["chunks"]) and tiled
        # weight rings included
        return fp

    def from_frame(self, fp):
        layers = self.low["layers"]
        shapes = fp["shapes"]
        last = None
        for st in fp["strata"]:
            lis = st["layers"]
            if st["kind"] == "single":
                run = [layers[li] for li in lis]
                self.layout(run, final=False, inp=tuple(shapes[lis[0]]["in"]))
                for k, L in enumerate(run):
                    if "copy" in self.info[k]:
                        self.groups.append(self.copy_group(self.info[k]))
                    ev0 = self.ev
                    self.groups += self.cut(self.layer_group(k, L, first=k == 0, last=k == len(run) - 1), ev0)
                last = self.info[-1]
                continue
            self.frame_stratum(st, shapes)
            co, ho, wo = shapes[lis[-1]]["out"]
            last = {"dout": Dist(co), "ho": ho, "wo": wo, "conv": layers[lis[-1]]["kind"] == "conv"}
        # the read-back tiles: our rule (frame.py's per-tile rule splits t2d-300x300 in 3, NPUC in 2)
        nu = unflatten_tiles(last, last["ho"], last["wo"])
        dout = last["dout"]
        self.out_chw = (dout.c, last["ho"], last["wo"])
        for u in range(nu):
            nl = dout.cores if nu > 1 and dout.ppc > 1 else 1
            ops = [Op("load", what="dram", flag=self.event(), utile=u, core=k if nl > 1 else None)
                   for k in range(nl)]
            self.groups.append(ops + [Op("nopop", deps=list(range(nl))), Op("unflatten")])

    def frame_stratum(self, st, shapes):
        """the groups of one tiled stratum as frame.plan cut it: per tile, its input region loaded, a TRIM when the
        stratum has one, then per layer its pad (in every tile of a padding layer; the tile's own edges), reformat,
        conv (weights loaded in tile 0, or a ring reloaded every tile; partial sums) or pool, and the store (V: tiled
        sweeps, frame/fs/*, final-chains, tiling2); tiles reaching the output's end as tile_to_end extends them"""
        layers = self.low["layers"]
        lis = st["layers"]
        tiles = st["tiles"]
        if any(self.tile_to_end(t, lis, shapes) != t for t in tiles):
            raise Unsupported("a tile that frame.plan does not extend to the end of the output")
        chunks = st["chunks"]
        ch = {}
        wch = []
        for c in chunks:
            if c["kind"] == "WEIGHT":
                wch.append(c)
            else:
                ch[(c["kind"], c.get("layer"))] = c
        ring = st.get("ring", {})
        wstep = st.get("wstep", {})
        has_trim = any(o["op"] == "TRIM" for o in st["ops"])
        split = ch[("SPLIT", lis[0])]["off"] // 64
        trim = ch[("TRIM", lis[0])]["off"] // 64 if has_trim else None
        convs = [li for li in lis if layers[li]["kind"] == "conv"]
        wbase = {li: wch[k]["off"] // 64 for k, li in enumerate(convs)}
        cin0 = Dist(shapes[lis[0]]["in"][0])
        for ti, t in enumerate(tiles):
            ops = []
            nl = cin0.cores if cin0.ppc > 1 else 1
            l0, l1, m0, m1 = t["load"]
            for k in range(nl):
                ops.append(Op("load", what="ifm", flag=self.event(), tile=ti, core=k if nl > 1 else None,
                              sram=split * 64, region=(l0, l1 + 1, m0, m1 + 1)))
            loads = list(range(nl))
            i0, i1, j0, j1 = t["layers"][lis[0]]["in"]
            hl, wl, hn, wn = l1 - l0 + 1, m1 - m0 + 1, i1 - i0 + 1, j1 - j0 + 1
            if has_trim:
                for sl in range(cin0.ppc):
                    for rc, pc in nfu_grid(hl, hn, 1, 1, -(i0 - l0), 0, wl, wn, 1, 1, -(j0 - m0), 0):
                        ops.append(Op("trim", L=layers[lis[0]], din=cin0, hl=hl, wl=wl, hn=hn, wn=wn, rc=rc,
                                      skiph=i0 - l0, pc=pc, src=split + sl * cells(hl, wl),
                                      dst=trim + sl * cells(hn, wn), deps=loads, cm=cin0.mask,
                                      one_layer=len(lis) == 1))
                src = trim
            else:
                src = split
            pending = loads                     # the first consumer waits for the input loads
            full = True
            for li in lis:
                L = layers[li]
                tl = t["layers"][li]
                a0, a1, b0, b1 = tl["in"]
                o0, o1, q0, q1 = tl["out"]
                h, w, ho, wo = a1 - a0 + 1, b1 - b0 + 1, o1 - o0 + 1, q1 - q0 + 1
                tp = list(tl["pads"])
                din, dout = Dist(shapes[li]["in"][0]), Dist(shapes[li]["out"][0])
                if L["kind"] == "pool":
                    for sl in range(din.ppc):
                        for rc, pc in nfu_grid(h, ho, L["k"][0], L["stride"][0], tp[0], tp[1],
                                               w, wo, L["k"][1], L["stride"][1], tp[2], tp[3]):
                            ops.append(Op("pool", L=L, din=din, h=h, w=w, ho=ho, wo=wo, pc=pc, rc=rc,
                                          src=src + sl * cells(h, w),
                                          dst=ch[("POOL", li)]["off"] // 64 + sl * cells(ho, wo),
                                          deps=pending, pads=tp, cm=self.slot_mask(din, sl, full)))
                    pending = []
                    src = ch[("POOL", li)]["off"] // 64
                    continue
                kh, kw = L["k"]
                s2d = L["stride"][0]
                hin, win = h, w
                if any(L["pads"]):        # every tile of a padding layer pads, even with no pad of its own (V: f002 g3)
                    hp, wp = h + tp[0] + tp[1], w + tp[2] + tp[3]
                    pad = ch[("PAD", li)]["off"] // 64
                    for sl in range(din.ppc):
                        for rc, pc in nfu_grid(h, hp, 1, 1, tp[0], tp[1], w, wp, 1, 1, tp[2], tp[3]):
                            ops.append(Op("pad", L=L, din=din, h=h, w=w, hp=hp, wp=wp, pc=pc, rc=rc, tpads=tp,
                                          src=src + sl * cells(h, w), dst=pad + sl * cells(hp, wp), deps=pending,
                                          cm=self.slot_mask(din, sl, full)))
                    pending = []
                    src, hin, win = pad, hp, wp
                kin = 2 * din.ppc
                if s2d == 2:
                    hr, wr = cdiv(hin, 2), cdiv(win, 2)
                    ref = ch[("REFORMAT", li)]["off"] // 64
                    for sl in range(din.ppc):
                        for y0 in range(0, hin, RU_MAX_W):
                            for x0 in range(0, win, RU_MAX_W):
                                ops.append(Op("reformat", L=L, din=din, hp=hin, wp=win, hr=hr, wr=wr, x0=x0,
                                              x1=min(win, x0 + RU_MAX_W), y0=y0, y1=min(hin, y0 + RU_MAX_W),
                                              src=src + sl * cells(hin, win), dst=ref + 4 * sl * cells(hr, wr),
                                              deps=pending, cm=self.slot_mask(din, sl, full)))
                    pending = []
                    src, hin, win, kin = ref, hr, wr, 8 * din.ppc
                passes = cdiv(dout.ppc, 2)
                wunits, nsub, cpb = weight_rows(L, self.q)
                # the weights: all resident (loaded by tile 0, kept), or a ring of R slots every tile reloads, pass p
                # into slot p mod R, `wstep` bytes apart (frame.md 6.6; V: c512, r_c384, t-32x32-256-512)
                r = ring.get(li) or passes
                step = wstep.get(li) if r < passes else None
                step = step or 2 * wunits * 64
                psc = ch.get(("PARTIAL_SUM", li))
                for p in range(passes):
                    deps = list(pending)
                    if ti == 0 or r < passes:
                        ops.append(Op("load", what="weight", flag=self.event(), layer=L["name"], p=p,
                                      sram=wbase[li] * 64 + (p % r) * step))
                        deps.append(len(ops) - 1)
                    ops.append(Op("conv", L=L, din=din, dout=dout, p=p, passes=passes, hin=hin, win=win, kin=kin,
                                  k=[cdiv(kh, s2d), cdiv(kw, s2d)], ho=ho, wo=wo, src=src,
                                  ofm=ch[("CONV", li)]["off"] // 64, wbase=(wbase[li] * 64 + (p % r) * step) // 64,
                                  wunits=wunits, nsub=nsub, cpb=cpb, wslot=0,
                                  ps=psc["off"] // 64 if psc else None, deps=deps))
                pending = []
                src = ch[("CONV", li)]["off"] // 64
                full = dout.planes == dout.cores * dout.ppc
            dlast = Dist(shapes[lis[-1]]["out"][0])
            ns = dlast.cores if dlast.ppc > 1 else 1
            o0, o1, q0, q1 = t["out"]
            ops.append(Op("store", flags=[self.event() for _ in range(ns)], tile=ti, sram=src * 64,
                          region=(o0, o1 + 1, q0, q1 + 1)))
            self.groups.append(ops)

    def tile_to_end(self, t, lis, shapes):
        """a tile that ends on the stratum output's last row (column) computes every layer's output to that layer's
        last row (column), its input to the tensor's end with the layer's trailing pad: NPUC's layout adds the
        rounding left-over of each layer (0 <= d < s) to the last tile (frame.md 6.10), so a layer whose output is
        not fully read by the next one (a VALID pool on an odd size) is still computed whole there; an untiled axis
        is one such tile (V: tiling2 e03, e04: 2x2 pools on 27 x 31 after a 3x3 conv: pool 2 computes 31 columns, the
        stride-2 conv before pads 128 columns to 129). frame.plan's tiles are already extended (frame_stratum checks
        it: its chunks are sized on them)."""
        layers = self.low["layers"]
        out_end = (shapes[lis[-1]]["out"][1] - 1, shapes[lis[-1]]["out"][2] - 1)
        o = t["out"]
        ends = (o[1] >= out_end[0], o[3] >= out_end[1])
        if not any(ends):
            return t
        per = {li: dict(x) for li, x in t["layers"].items()}
        for li in lis:
            L = layers[li]
            _, hi, wi = shapes[li]["in"]
            _, ho, wo = shapes[li]["out"]
            i0, i1, j0, j1 = per[li]["in"]
            o0, o1, q0, q1 = per[li]["out"]
            pt, pb, pl, pr = per[li]["pads"]
            if ends[0]:
                i1, o1, pb = hi - 1, ho - 1, L["pads"][1]
            if ends[1]:
                j1, q1, pr = wi - 1, wo - 1, L["pads"][3]
            per[li] = {"in": (i0, i1, j0, j1), "out": (o0, o1, q0, q1), "pads": (pt, pb, pl, pr)}
        l0, l1, m0, m1 = t["load"]
        i0, i1, j0, j1 = per[lis[0]]["in"]
        return dict(t, layers=per, load=(l0, max(l1, i1), m0, max(m1, j1)))

    def event(self):
        """DMA n of the network signals event 1 << (n mod 16) (V: io trig_flag of co224..co255)"""
        f = 1 << (self.ev % 16)
        self.ev += 1
        return f

    def layout(self, layers, final=True, inp=None):
        """the node order, the chunks and their offsets of an SRAM-forwarded run of layers (input from DRAM)"""
        c0, h0, w0 = inp or self.low["input"]["chw"]
        whole = [c0, h0, w0] == list(self.low["input"]["chw"]) and len(layers) == len(self.low["layers"])
        chunks = []
        order = 1                                       # node 0 = the input
        self.info = []
        inp = Chunk(0, "INPUT", Dist(c0).ppc * cells(h0, w0) * 64, None, 0)
        chunks.append(inp)
        prev_out = inp
        h, w = h0, w0                                   # NPUC's own feature-map sizes (may exceed lowered.json's)
        full = True                                     # every plane slot of every core written (see slot_mask)
        for li, L in enumerate(layers):
            cin = L["in_chw"][0]
            cout = L["out_chw"][0]
            d = {"din": Dist(cin), "dout": Dist(cout), "h": h, "w": w, "pads": list(L["pads"]), "full": full}
            kh, kw = L["k"]
            sh, sw = L["stride"]
            pt, pb, pl, pr = L["pads"]
            ho, wo = (h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1
            if L["kind"] == "pool" and (ho - 1) * sh + kh - pt - pb - h >= 0 > (wo - 1) * sw + kw - pl - pr - w:
                # "Padding in Pooling has negative value!": when only the width's padding is negative NPUC computes
                # one more output column with a right pad instead (V: r004, r017, r029, r035, r045, r055, r063,
                # r065; not when the height's is negative too: r006, r013, r025; nor for the height: r007, r068)
                # its right pad: the smaller of the negated padding and the extra window's overhang (V: r2-139,
                # rnd067, rnd057: k3 s3 on 52, 10, 8 columns, 1 each; k2 s2: 1)
                d["pads"][3] = min(w + pl + pr - (wo - 1) * sw - kw, wo * sw + kw - pl - w)
                wo += 1
            d["ho"], d["wo"] = ho, wo
            head = order
            if prev_out.live is None:
                prev_out.live = [head, head]           # NPUC's "Hack on LA": the input lives at its consumer only
            else:
                prev_out.live[1] = head
            if L["kind"] == "conv":
                pt, pb, pl, pr = L["pads"]
                d["haspad"] = any(L["pads"])
                d["hp"], d["wp"] = h + pt + pb, w + pl + pr
                d["passes"] = cdiv(d["dout"].ppc, 2)
                d["s2d"] = s2d = L["stride"][0]
                d["wunits"], d["nsub"], d["cpb"] = weight_rows(L, self.q)
                if d["haspad"]:
                    d["pad"] = Chunk(len(chunks), "PAD", d["din"].ppc * cells(d["hp"], d["wp"]) * 64,
                                     [order, order + 1], li + 1)
                    chunks.append(d["pad"])
                    order += 1
                if s2d == 2:                            # stride 2: NFU reformat (space to depth), V: s2* variants
                    d["hr"], d["wr"] = cdiv(d["hp"], 2), cdiv(d["wp"], 2)
                    d["ref"] = Chunk(len(chunks), "REFORMAT", d["din"].ppc * 4 * cells(d["hr"], d["wr"]) * 64,
                                     [order, order + 1], li + 1)
                    chunks.append(d["ref"])
                    order += 1
                ofm_planes = cdiv(cdiv(cout, 16) * 16, 8)   # "Convolution Ofm will be rounded to 16 ... HW bug"
                d["out"] = Chunk(len(chunks), "CONV", ofm_planes * cells(ho, wo) * 64, [order, order + 1], li + 1)
                chunks.append(d["out"])
                d["weight"] = Chunk(len(chunks), "WEIGHT", d["passes"] * 2 * d["wunits"] * 64, [head, order], li + 1)
                chunks.append(d["weight"])
                d["ps"] = None
                if d["nsub"] > 1:                       # weights in several sub-batches: a partial-sum buffer,
                    hin, win = (d["hr"], d["wr"]) if s2d == 2 else (d["hp"], d["wp"])   # placed after the
                    d["ps"] = Chunk(len(chunks), "PARTIAL_SUM", 2 * 7 * cells(hin, win) * 64,  # layer's other
                                    [order, order], li + 1, tail=True)                  # chunks (V: ci64k7...)
                    chunks.append(d["ps"])
            else:
                d["out"] = Chunk(len(chunks), "POOL", d["dout"].ppc * cells(ho, wo) * 64, [order, order + 1], li + 1)
                chunks.append(d["out"])
            d["in"] = prev_out
            prev_out = d["out"]
            order += 1
            self.info.append(d)
            h, w = ho, wo
            if L["kind"] == "conv":
                full = d["dout"].planes == d["dout"].cores * d["dout"].ppc
        for d in self.info:
            if "weight" in d:
                d["ring"] = weight_ring(d)
                d["weight"].size = d["ring"] * 2 * d["wunits"] * 64
                if d["ring"] < d["passes"] and d["ring"] <= 4:
                    # a ring of 4 slots or fewer lives on its conv only (frame.md 6.7; V: f043, wl-512-h21)
                    d["weight"].live = [d["out"].live[0], d["out"].live[0]]
        allocate(chunks)
        self.chunks = chunks
        if whole:
            self.frame_single(chunks)
        # NPUC tiles a layer (SPLIT/TRIM/CONCAT nodes, several tiles per group: not modelled) when the SRAM a core
        # needs passes 256 KiB or an output dimension passes 255 (V: t200 one tile at 245120 B, t208 tiled at
        # 264896 B; w250 one tile, w256 and h300 tiled, w256s2 and w300s2 (outputs 128, 150) one tile)
        need = max(c.off + c.size for c in chunks)
        if need > SRAM_LIMIT:
            raise NeedsTiling("tiling (SRAM %d B per core > 256 KiB)" % need)
        for L, d in zip(layers, self.info):
            if L["kind"] == "conv" and max(d["ho"], d["wo"]) > 255:     # pools are not tiled (V: w300pool)
                raise NeedsTiling("tiling (an output dimension of %s > 255)" % L["name"])
        d = self.info[-1]
        if final and unflatten_tiles(d, d["ho"], d["wo"]) > 1:
            raise Unsupported("tiling (only the unflatten of %d x %d)" % (d["ho"], d["wo"]))

    def frame_single(self, chunks):
        """take the SRAM plan of an untiled network from frame.plan (frame.md 6.6-6.7: the weight ring's slot step
        when Cout is not a multiple of 16, a ring's live range, NPUC's trim-and-spill copy when no allocation order
        fits) when frame.py is present and the whole network is this one block: every chunk's offset and size, the
        ring and step of each WEIGHT chunk, the copy (a TRIM chunk) before a layer"""
        try:
            fp = frame.plan(self.low)
        except Exception:
            return
        if fp.get("tiled") or "chunks" not in fp:
            return
        fc = {(c["kind"], c.get("layer", -1)): c for c in fp["chunks"]}
        for ck in chunks:
            c = fc.get((ck.name, ck.batch - 1))
            if c is None:
                return                      # not the same block: keep ours
        for ck in chunks:
            c = fc[(ck.name, ck.batch - 1)]
            ck.off, ck.size = c["off"], c["size"]
            if ck.name == "WEIGHT":
                d = self.info[ck.batch - 1]
                d["ring"] = c.get("ring") or d["ring"]
                d["wstep"] = c.get("step")
        for c in fp["chunks"]:
            if c["kind"] == "TRIM" and c.get("copy"):
                d = self.info[c["layer"]]
                d["copy"] = {"src": c["src"], "dst": c["off"], "size": c["size"]}
                d["in"] = Chunk(None, "TRIM", c["size"], list(c["live"]), d["in"].batch)
                d["in"].off = c["off"]

    def copy_group(self, d):
        """NPUC's trim-and-spill copy (frame.md 6.6; V: ws-191-256x2, r3-076/077/096/111): its own group, no DMA:
        one NFU copy (TRIM mode, no skip) per plane slot of the layer's input and NFU command piece, from the
        spilled chunk to the new one, every core (V: the oracle's streams, per frame.py; tiling2 b03, b05)"""
        din, h, w = d["din"], d["h"], d["w"]
        src, dst = d["copy"]["src"] // 64, d["copy"]["dst"] // 64
        # NFU command pieces as any NFU op (V: tiling2 b03, b05: 138 and 128 columns in 2 pieces)
        return [Op("trim", L=None, din=din, hl=h, wl=w, hn=h, wn=w, rc=rc, skiph=0, pc=pc,
                   src=src + s * cells(h, w), dst=dst + s * cells(h, w), deps=[], cm=0xF) for s in range(din.ppc)
                for rc, pc in nfu_grid(h, h, 1, 1, 0, 0, w, w, 1, 1, 0, 0)]

    # NPUC's per-layer CMDQ size estimate (V: its log, "Heuristically predicted size of CMDQ" on 1619 layer blocks):
    # 16350 + 1020 per conv pass + 60 per padded plane slot + 150 per pooled one + 102 per reformatted one + 1500 per
    # eltwise; the plane-slot ops count once whatever their NFU pieces
    CMDQ_COST = {"conv": 1020, "pad": 60, "pool": 150, "reformat": 102, "trim": 60, "eltnop": 1500}
    CMDQ_LIMIT = 61440

    def cut(self, ops, ev0):
        """NPUC's "CMDQ seems too big" (V: the cut sweep `stream_variants_cut.txt`, big4, MobileNet's 1024 x 1024 1x1):
        when a layer's estimate passes 60 KiB, its instructions (conv passes, plane-slot NFU ops) go in chunks of n =
        (61440 // (estimate share per instruction)) // 2, each chunk its own NCP group (loads with their consumer); a
        layer that stores gets one store per chunk, of the chunk's passes. The DMAs' events follow the new order."""
        inst = []
        for i, o in enumerate(ops):
            if o.kind == "conv" or (o.kind in self.CMDQ_COST and o.kind != "conv"
                                    and (not hasattr(o, "pc") or (o.pc["first"] and o.rc["first"]))):
                inst.append(i)
        cost = sum(self.CMDQ_COST.get(ops[i].kind, 0) for i in inst)
        if 16350 + cost <= self.CMDQ_LIMIT or not inst:
            return [ops]
        n = (self.CMDQ_LIMIT // (cost // len(inst))) // 2
        st = [o for o in ops if o.kind == "store"]
        convs = [o for o in ops if o.kind == "conv"]
        if len(st) > 1 or not convs or any(o.kind not in ("load", "store", "conv", "pad", "reformat")
                                           for o in ops):
            raise Unsupported("a CMDQ cut of this layer (not modelled)")
        # chunk boundaries: before the instruction numbered k n (its loads go with it)
        starts = set(inst[k] for k in range(n, len(inst), n))
        chunks, cur = [], []
        pending = []                        # loads waiting for their consumer's chunk
        for i, o in enumerate(ops):
            if o.kind == "store":
                continue
            if o.kind == "load":
                pending.append(i)
                continue
            if i in starts and cur:
                chunks.append(cur)
                cur = []
            cur += pending + [i]
            pending = []
        cur += pending
        chunks.append(cur)
        dout = convs[0].dout
        oc = cells(convs[0].ho, convs[0].wo)
        # the input loads of a layer an NFU op reads first follow the NFU ops: each chunk loads the plane slots it
        # pads, in 4 parts of ceil(slots / 4) when 16 or more (V: k3c64-ci512: 56 slots in 4 loads of 14, then 8)
        ifm = [i for i, o in enumerate(ops) if o.kind == "load" and o.what == "ifm"]
        nfu = [i for i, o in enumerate(ops) if o.kind in ("pad", "reformat", "pool") and set(o.deps) & set(ifm)]
        if nfu and ops[ifm[0]].sram is not None:
            base, hw = ops[ifm[0]].sram, cells(ops[nfu[0]].h, ops[nfu[0]].w) if hasattr(ops[nfu[0]], "h") else None
            for k, ch in enumerate(chunks):
                ch[:] = [i for i in ch if i not in ifm]
                sl = sorted(set(ops[i].slot for i in ch if i in nfu))
                if not sl:
                    continue
                part = cdiv(len(sl), 4) if len(sl) >= 16 else len(sl)
                newl = {}
                for j in range(0, len(sl), part):
                    lo = Op("load", what="ifm", flag=0, sram=base + sl[j] * (hw or 0) * 64,
                            slots=(sl[j], sl[min(j + part, len(sl)) - 1] + 1))
                    ops.append(lo)
                    for x in sl[j:j + part]:
                        newl[x] = len(ops) - 1
                res, seen = [], set()
                for i in ch:
                    if i in nfu:
                        li = newl[ops[i].slot]
                        if li not in seen:
                            res.append(li)
                            seen.add(li)
                        ops[i].deps = [li]
                    res.append(i)
                ch[:] = res
        out = []
        for ch in chunks:
            idx = {old: k for k, old in enumerate(ch)}
            new = []
            for old in ch:
                o = ops[old]
                o.deps = [idx[x] for x in o.deps if x in idx]
                new.append(o)
            ps = [o.p for o in new if o.kind == "conv"]
            if st and ps:
                p0, p1 = min(ps), max(ps) + 1
                cnt = [min(2 * (p1 - p0), max(0, min(dout.ppc, dout.planes - k * dout.ppc) - 2 * p0))
                       for k in range(dout.cores)]
                new.append(Op("store", flags=[0] * len(set(cnt) - {0}), sram=st[0].sram + 2 * p0 * oc * 64, cnt=cnt,
                              **{k: getattr(st[0], k) for k in ("tensor",) if hasattr(st[0], k)}))
            out.append(new)
        self.ev = ev0                       # the events again, in the new order
        for new in out:
            for o in new:
                if o.kind == "load" and not getattr(o, "arm", False):
                    o.flag = self.event()
                elif o.kind == "store":
                    o.flags = [self.event() for _ in o.flags]
        return out

    @staticmethod
    def slot_mask(dist, s, full):
        """the cores an NFU op on plane slot s runs on: the slots its input tensor has written. A tensor loaded from
        DRAM, or a conv output whose planes fill every core, has all of them (V: ci9; pfirst20s, the pads after a
        pool on a loaded input); a conv output whose last core holds fewer planes has there only the slots of the
        conv's 2-slot passes, 2 floor(ppc / 2) (V, last core's slots: c10p, c14p, r015 (ppc 2: 2), c20p, c22p, r013
        (ppc 3: 2), c26p, c28p (ppc 4: 4), c34p, c38p (ppc 5: 4), c50p (ppc 7: 6)). Pads, pools and reformats keep
        their input's written slots (r045, c34pp: pool after pool; c20pad, c34pad, r038, r061, c20s2)."""
        if full or s < 2 * (dist.ppc // 2):
            return dist.mask
        return dist.mask >> 1

    def layer_group(self, li, L, first, last):
        d = self.info[li]
        din, dout = d["din"], d["dout"]
        h, w, ho, wo = d["h"], d["w"], d["ho"], d["wo"]
        ops = []
        # the stage that reads the input feature map: a pad, else a stride-2 reformat, else a pool, else the conv
        stage = ("pad" if d.get("haspad") else "reformat" if d.get("s2d") == 2 else "conv") \
            if L["kind"] == "conv" else "pool"
        # with 16 or more plane slots per core and a pad or pool first, NPUC loads the input in 4 chunks of
        # ceil(ppc / 4) slots, each just before the NFU ops of its slots (V: ci124..ci256, ci128s2, pfirst128; not
        # for a reformat or conv first: ci128s2h20, ci128v)
        split = first and stage in ("pad", "pool") and din.ppc >= 16
        chunk = cdiv(din.ppc, 4)
        # a tensor loaded whole (an MTC op reads it first) comes in one DMA per distinct per-core plane count, one
        # loaded per core in one (V: rc-114-272v: 114 channels, 1x1 conv: 2; res20, padded: 1)
        parts = d.get("parts")
        if parts is None:
            core = stage in ("pad", "pool", "reformat") and (din.ppc > 1 or din.c % 2)
            parts = 1 if core or din.planes - (din.cores - 1) * din.ppc == din.ppc else 2
        if first and not split:
            for j in range(parts):
                ops.append(Op("load", what="ifm", flag=self.event(), sram=d["in"].off,
                              **({"part": j} if parts > 1 else {})))
        cur = [list(range(parts))]

        def ifm_dep(slot):
            if not first:
                return []
            if split and slot % chunk == 0:
                ops.append(Op("load", what="ifm", flag=self.event(), sram=d["in"].off + slot * cells(h, w) * 64,
                              slots=(slot, slot + chunk)))
                cur[0] = [len(ops) - 1]
            return list(cur[0])

        src = d["in"].off // 64
        pbp = False
        if L["kind"] == "conv":
            if d["haspad"]:
                pl, pr = L["pads"][2:]
                for s in range(din.ppc):
                    dep = ifm_dep(s)
                    for rc, pc in nfu_grid(h, d["hp"], 1, 1, L["pads"][0], L["pads"][1], w, d["wp"], 1, 1, pl, pr):
                        ops.append(Op("pad", L=L, din=din, h=h, w=w, hp=d["hp"], wp=d["wp"], pc=pc, rc=rc, slot=s,
                                      src=src + s * cells(h, w), dst=d["pad"].off // 64 + s * cells(d["hp"], d["wp"]),
                                      deps=dep, cm=self.slot_mask(din, s, d["full"])))
                src = d["pad"].off // 64
            hin, win, kin = d["hp"], d["wp"], 2 * din.ppc
            if d["s2d"] == 2:
                dst = d["ref"].off // 64
                rc = cells(d["hr"], d["wr"])
                for s in range(din.ppc):
                    dep = ifm_dep(s) if stage == "reformat" else []
                    for y0 in range(0, d["hp"], RU_MAX_W):
                        for x0 in range(0, d["wp"], RU_MAX_W):
                            ops.append(Op("reformat", L=L, din=din, hp=d["hp"], wp=d["wp"], hr=d["hr"], wr=d["wr"], slot=s,
                                          x0=x0, x1=min(d["wp"], x0 + RU_MAX_W),
                                          y0=y0, y1=min(d["hp"], y0 + RU_MAX_W),
                                          src=src + s * cells(d["hp"], d["wp"]), dst=dst + 4 * s * rc, deps=dep,
                                          cm=self.slot_mask(din, s, d["full"])))
                src, hin, win, kin = dst, d["hr"], d["wr"], 8 * din.ppc
            # a ring slot: frame.md's step (Cout not a multiple of 16), else 2 x the planner's rows (a conv reading a
            # concat: its slots sit at its planning rows, the second half at the emitted rows; V: g103, g167)
            step = d.get("wstep") or 2 * d.get("prows", d["wunits"]) * 64
            # pass by pass stores: the last conv with a weight ring, Cout not a multiple of 16, no sub-batches
            # stores each pass's two plane slots after its conv (frame.md 6.7; V: rc-167-488..510, rc-200-500)
            pbp = last and d["ring"] < d["passes"] and dout.c % 16 and d["nsub"] == 1
            # not in a layer NPUC cuts for its CMDQ size: that one stores chunk by chunk (V: k1c1000-h1)
            est = 16350 + 1020 * d["passes"] + din.ppc * (60 * d["haspad"] + 102 * (d["s2d"] == 2))
            pbp = pbp and est <= self.CMDQ_LIMIT
            # a kernel over 4 MiB in all is loaded core by core, 4 DMAs a pass (V: k3c64-ci512, s2c64; one up to
            # 2.6 MB: u512-ci512)
            npc = 4 if 2 * d["wunits"] * 64 * d["passes"] * 4 > 4 << 20 else 1
            for p in range(d["passes"]):
                for k in range(npc):
                    ops.append(Op("load", what="weight", flag=self.event(), layer=L["name"], p=p,
                                  sram=d["weight"].off + (p % d["ring"]) * step, **({"core": k} if npc > 1 else {})))
                deps = list(range(len(ops) - npc, len(ops)))
                if first and stage == "conv":
                    deps += list(range(parts))
                ops.append(Op("conv", L=L, din=din, dout=dout, p=p, passes=d["passes"], hin=hin, win=win,
                              kin=kin, k=[cdiv(k, d["s2d"]) for k in L["k"]],
                              ho=ho, wo=wo, src=src, ofm=d["out"].off // 64,
                              wbase=(d["weight"].off + (p % d["ring"]) * step) // 64,
                              wunits=d["wunits"], nsub=d["nsub"], cpb=d["cpb"], wslot=0,
                              ps=d["ps"].off // 64 if d["ps"] else None, deps=deps))
                if pbp:                     # one DMA per distinct per-core count of the pass's planes (V: rc-167-500)
                    pl = [dout.planes - k * dout.ppc for k in range(dout.cores)]
                    n = len({min(2, max(0, min(x, dout.ppc) - 2 * p)) for x in pl} - {0})
                    ops.append(Op("store", flags=[self.event() for _ in range(n)],
                                  sram=d["out"].off + 2 * p * cells(ho, wo) * 64, part_pass=p))
                    ops += [Op("skip") for _ in range(n - 1)]   # each DMA its own IR element (V: rc-167-500)
        else:
            pl, pr = d["pads"][2:]
            for s in range(din.ppc):
                dep = ifm_dep(s)
                for rc, pc in nfu_grid(h, ho, L["k"][0], L["stride"][0], d["pads"][0], d["pads"][1],
                                       w, wo, L["k"][1], L["stride"][1], pl, pr):
                    ops.append(Op("pool", L=L, din=din, h=h, w=w, ho=ho, wo=wo, pc=pc, rc=rc, slot=s,
                                  src=src + s * cells(h, w),
                                  dst=d["out"].off // 64 + s * cells(ho, wo), deps=dep, pads=d["pads"],
                                  cm=self.slot_mask(din, s, d["full"])))
        if last and not pbp:
            # one store per distinct per-core plane count (NPUC merges the per-core DMAs of equal shape): a conv's
            # cores hold their real planes (co9: 2, 2, 1 -> 2 stores); a pool's cores run the slots of slot_mask, so
            # a pool on an uneven conv output takes 2 stores when ppc is odd (r013, r068: 20 channels, ppc 3), 1
            # when it is even (r015, r067: 9 channels, ppc 2)
            rem = dout.planes - (dout.cores - 1) * dout.ppc
            if L["kind"] == "pool":
                nstore = 1 if d["full"] or dout.ppc % 2 == 0 else 2
            else:
                nstore = 1 if rem == dout.ppc else 2
            segs = d.get("segs")
            if segs and len(segs) > 1:
                # the planes of a concat in SRAM (through a pool): one store per input's slots per core, putting the
                # channels back in order (V: g141 3 x 4, g169 2 x 4); one per core when the inputs' slots are all
                # alike (a strided store; V: g141's 16 + 16)
                if len({n for _, n in segs}) == 1:
                    segs = segs[:1]
                srams = [d["out"].off + s0 * cells(ho, wo) * 64 for _ in range(dout.cores) for s0, _ in segs]
                ops.append(Op("store", flags=[self.event() for _ in srams], sram=d["out"].off, srams=srams))
                return ops
            ops.append(Op("store", flags=[self.event() for _ in range(nstore)], sram=d["out"].off))
        return ops

    # ---- graphs: eltwise add, concat, standalone ReLU (npuc-net 2) ---------------------------------------------------
    def graph(self):
        """a network with graph ops, untiled (dossier cmdq.md 7c). NPUC runs the layers in its own order (exec_order),
        one group per layer (none for a concat), and keeps every tensor in SRAM until its last reader, except where it
        goes through DRAM: the network output, the inputs of an eltwise that reads the network input (that eltwise
        loads both operands back), the inputs of a concat (each producer stores into the concat's buffer; its reader
        loads it). The network input is loaded by its first reader, and again by a later reader unless the layer run
        just before also read it or a concat shares it (one INPUT chunk then)."""
        low = self.low
        layers = low["layers"]
        src = low["input"]["name"]
        by = {L["name"]: L for L in layers}
        ins = {L["name"]: list(L["inputs"]) if "inputs" in L else [L["input"]] for L in layers}
        cons = {}
        for L in layers:
            k = L["kind"]
            if k == "conv" and L["stride"] not in ([1, 1], [2, 2]):
                raise Unsupported("conv stride %s" % L["stride"])
            if k == "relu" and L.get("type", "ReLU") not in ("ReLU", "ReLU6"):
                raise Unsupported("standalone %s" % L.get("type"))
            if k not in ("conv", "pool") + GRAPH_KINDS:
                raise Unsupported("layer kind %s" % k)
            for i in ins[L["name"]]:
                if i != src and i not in by:
                    raise Unsupported("%s reads %s, not an earlier layer" % (L["name"], i))
                cons.setdefault(i, []).append(L["name"])
        out = low.get("npu_output") or layers[-1]["name"]
        if [L["name"] for L in layers if L["name"] not in cons] != [out]:
            raise Unsupported("not a single output")
        order = exec_order(out, ins, cons, src)
        if len(order) != len(layers):
            raise Unsupported("layers the output does not depend on")
        order = [by[n] for n in order]
        for L in order:
            if L["kind"] == "concat":
                if any(i == src or by[i]["kind"] == "concat" for i in ins[L["name"]]):
                    raise Unsupported("concat of the input or of a concat")
            if L["kind"] == "eltwise" and len(ins[L["name"]]) != 2:
                raise Unsupported("eltwise of %d inputs" % len(ins[L["name"]]))
        # atoms (NPUC's nodes, in order): pad, reformat, conv; pool; relu: its conv; eltwise: 2; concat: 1
        c0, h0, w0 = low["input"]["chw"]
        chs, hw, head, tail, geo = {src: c0}, {src: (h0, w0)}, {}, {}, {}
        pos = 1
        for L in order:
            n, k = L["name"], L["kind"]
            c, (h, w) = chs[ins[n][0]], hw[ins[n][0]]
            pads = [0, 0, 0, 0]
            if k in ("conv", "pool"):
                kh, kw = L["k"]
                sh, sw = L["stride"]
                pads = list(L["pads"])
                pt, pb, pl, pr = pads
                ho, wo = (h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1
                if k == "pool" and (ho - 1) * sh + kh - pt - pb - h >= 0 > (wo - 1) * sw + kw - pl - pr - w:
                    pads[3] = min(w + pl + pr - (wo - 1) * sw - kw, wo * sw + kw - pl - w)  # NPUC's wider pool
                    wo += 1
                cout = L["out_chw"][0]
                na = 1 + (k == "conv" and any(pads)) + (k == "conv" and sh == 2)
            else:
                if any(hw[i] != (h, w) for i in ins[n]):
                    raise Unsupported("%s: inputs of different sizes" % n)
                ho, wo = h, w
                cout = sum(chs[i] for i in ins[n]) if k == "concat" else c
                if k == "eltwise" and chs[ins[n][1]] != c:
                    raise Unsupported("%s: inputs of different channel counts" % n)
                na = 2 if k == "eltwise" else 1
            head[n], tail[n] = pos, pos + na - 1
            pos += na
            chs[n], hw[n] = cout, (ho, wo)
            geo[n] = (c, h, w, cout, ho, wo, pads)
        unfl = pos
        try:
            winfo = {x["name"]: x for x in weights.weight_layer_info(low) if x["kind"] == "conv"}
        except ValueError as e:
            raise Unsupported("weights.py: %s" % e)
        cats = [L["name"] for L in order if L["kind"] == "concat"]
        scats = [kn for kn in cats if concat_in_sram(kn, out, by, ins, cons)]
        # tensors their readers load from DRAM: the network input, a concat's buffer
        dsrc = [src] + [kn for kn in cats if kn not in scats and kn != out]
        eltd = {L["name"] for L in order if L["kind"] == "eltwise" and any(i in dsrc for i in ins[L["name"]])}
        into = {}                                           # producer -> (SRAM concat, its first plane slot)
        for kn in scats:
            o = 0
            for i in ins[kn]:
                if len(cons[i]) != 1:
                    raise Unsupported("%s feeds a concat in SRAM and another layer" % i)
                into[i] = (kn, o)
                o += Dist(chs[i]).ppc
        stored = {}                                         # tensor -> where its producer stores it
        if by[out]["kind"] != "concat":
            stored[out] = ("out", out)
        for e in sorted(eltd):
            for i in ins[e]:
                if i not in dsrc:
                    stored.setdefault(i, ("im", i))
        for kn in cats:
            if kn in scats:
                continue
            for i in ins[kn]:
                if i in stored:
                    raise Unsupported("%s goes to DRAM twice" % i)
                stored[i] = ("cat", kn)

        def end_of(t):
            e = [head[r] for r in cons.get(t, []) if by[r]["kind"] != "concat"]
            if t == out:
                e.append(unfl)
            return max(e) if e else tail[t]

        chunks = []

        def chunk(name, size, live, batch, tl=False):    # batch: (0, live start, n) the first citizens, else
                                                          # (1,) the INPUT chunks, (2, i) layer i's
            ck = Chunk(len(chunks), name, size, list(live), batch, tl)
            chunks.append(ck)
            return ck

        insize = Dist(c0).ppc * cells(h0, w0) * 64

        def bulb_root(t):
            # where a concat's branch starts: back through one-input layers to a tensor read by several layers
            while t != src and by[t]["kind"] in ("conv", "pool", "relu") and len(cons[t]) == 1:
                t = ins[t][0]
            return t

        shared = any(all(bulb_root(i) == src for i in ins[kn]) for kn in cats)
        # a concat whose branches all start at one computed tensor ("Input sharing" from a layer): when that layer
        # is a single node (a pool, a ReLU, a conv without pad or stride), NPUC stores it to DRAM, its first reader
        # loads it back and the other branches share it in SRAM (V: g005, g008, g009, g030, g047); after a padded
        # conv or an eltwise the branches share it in SRAM straight away (V: g003, g012, g025, g036)
        roots = set()
        for kn in cats:
            rs = {bulb_root(i) for i in ins[kn]}
            r = min(rs)
            if len(rs) == 1 and r != src and by[r]["kind"] != "concat" and head[r] == tail[r]:
                roots |= rs
        for r in roots:
            stored.setdefault(r, ("im", r))
        # where each reader finds a tensor loaded from DRAM: the network input shared by a concat's branches, one
        # chunk live from 0 (V: cat2, cat3, cat-chain); a concat read by one layer, one first citizen from the concat
        # (V: cat2, cat3); else one chunk per reader, live at it only (NPUC's "Hack on LA"; V: add2, res; g006, g022)
        rch = {}
        catch, store = set(), {}
        xidx = {L["name"]: i for i, L in enumerate(order)}
        for t in dsrc:
            rs = cons.get(t, [])
            if not rs:
                continue
            nm = "INPUT" if t == src else "CONCAT"
            size = insize if t == src else Dist(chs[t]).ppc * cells(*hw[t]) * 64
            if t == src and shared:
                ck = chunk(nm, size, [0, max(head[r] for r in rs)], (0, 0, 0))
                rch.update({(t, r): ck for r in rs})
            elif t != src and len(rs) == 1:
                rch[t, rs[0]] = chunk(nm, size, [tail[t], head[rs[0]]], (0, tail[t], 1))
            else:
                for r in rs:
                    rch[t, r] = chunk(nm, size, [head[r], head[r]], (1,) if t == src else (2, xidx[t]))
            if t != src:
                catch.add(t)
        for kn in scats:                                    # concat in SRAM: from its first producer's conv to its
            c, h, w, cout, ho, wo, _ = geo[kn]              # last reader (V: cat16, cat-32-16, g021)
            t0 = min(tail[i] for i in ins[kn])
            store[kn] = chunk("CONCAT_STORAGE", Dist(cout).ppc * cells(ho, wo) * 64,
                              [t0, max(head[r] for r in cons[kn])], (0, t0, 1))
        tens, full, info = {src: None}, {src: True}, {}
        segs = {kn: [(into[i][1], Dist(chs[i]).ppc) for i in ins[kn]] for kn in scats}  # an SRAM concat's
        #                                                                     per-core plane segments (start, size)
        for idx, L in enumerate(order):
            n, k = L["name"], L["kind"]
            c, h, w, cout, ho, wo, pads = geo[n]
            bt = (2, idx)
            i0 = ins[n][0]
            cin = rch[i0, n] if (i0, n) in rch else tens[i0]
            d = {"din": Dist(c), "dout": Dist(cout), "h": h, "w": w, "ho": ho, "wo": wo, "pads": pads,
                 "full": full[i0], "in": cin}
            at = head[n]
            if k == "conv":
                d["haspad"] = any(pads)
                d["hp"], d["wp"] = h + pads[0] + pads[1], w + pads[2] + pads[3]
                d["passes"] = cdiv(d["dout"].ppc, 2)
                d["s2d"] = s2d = L["stride"][0]
                if n in winfo:              # the emitted block's geometry and NPUC's planning rows (weights.py; V: weights/cat)
                    wi = winfo[n]
                    d["wunits"], d["nsub"], d["cpb"] = wi["rows_per_batch"], wi["sub_batches"], wi["sub_channels"][0]
                    d["prows"] = wi["planning_rows"]
                else:
                    d["wunits"], d["nsub"], d["cpb"] = weight_rows(L, self.q)
                if d["haspad"]:
                    d["pad"] = chunk("PAD", d["din"].ppc * cells(d["hp"], d["wp"]) * 64, [at, at + 1], bt)
                    at += 1
                if s2d == 2:
                    d["hr"], d["wr"] = cdiv(d["hp"], 2), cdiv(d["wp"], 2)
                    d["ref"] = chunk("REFORMAT", d["din"].ppc * 4 * cells(d["hr"], d["wr"]) * 64, [at, at + 1], bt)
                    at += 1
                osize = cdiv(cdiv(cout, 16) * 16, 8) * cells(ho, wo) * 64
                if n in into:                                   # writes its planes into the concat's chunk
                    d["out"] = Chunk(None, "CONV", osize, [at, at], bt)
                    d["into"] = into[n]
                else:
                    d["out"] = chunk("CONV", osize, [at, end_of(n)], bt)
                d["weight"] = chunk("WEIGHT", 0, [head[n], at], bt)
                d["ps"] = None
                if d["nsub"] > 1:
                    hin, win = (d["hr"], d["wr"]) if s2d == 2 else (d["hp"], d["wp"])
                    d["ps"] = chunk("PARTIAL_SUM", 2 * 7 * cells(hin, win) * 64, [at, at], bt, True)
                full[n] = d["dout"].planes == d["dout"].cores * d["dout"].ppc
            elif k == "pool":
                if n in into:                                   # writes its planes into the concat's chunk
                    d["out"] = Chunk(None, "POOL", d["dout"].ppc * cells(ho, wo) * 64, [at, at], bt)
                    d["into"] = into[n]
                else:
                    d["out"] = chunk("POOL", d["dout"].ppc * cells(ho, wo) * 64, [at, end_of(n)], bt)
                full[n] = full[i0]
            elif k == "relu":
                # a 1x1 conv over the C channels, all weights at the zero point (weights.py weight_layers)
                R = {"kind": "conv", "name": n, "cout": c, "in_chw": [c, 1, 1], "k": [1, 1], "stride": [1, 1],
                     "w_zp": 128, "weights_oihw_u8": [128] * (c * c)}
                d["passes"] = cdiv(d["dout"].ppc, 2)
                d["wunits"], d["nsub"], d["cpb"] = weight_rows(R, self.q)
                d["out"] = chunk("CONV", cdiv(cdiv(cout, 16) * 16, 8) * cells(ho, wo) * 64, [at, end_of(n)], bt)
                d["weight"] = chunk("WEIGHT", 0, [at, at], bt)
                d["ps"] = None
                d["whole"] = i0 in store or i0 in catch or (i0 != src and by[i0]["kind"] == "conv") \
                    or (i0 != src and by[i0]["kind"] == "relu" and info[i0].get("whole", False))   # (V: g555)
                full[n] = d["dout"].planes == d["dout"].cores * d["dout"].ppc
            elif k == "eltwise":
                d["meta_chunks"] = [chunk("SHIFT_OUTPUT1", 32, [at, at + 1], bt),
                                    chunk("SHIFT_OUTPUT2", 32, [at, at + 1], bt),
                                    chunk("CU_BIN", 128, [at, at + 1], bt)]
                d["out"] = chunk("ELTWISE", d["dout"].ppc * cells(ho, wo) * 64, [at, end_of(n)], bt)
                # META_DATA: 64 B past 4 planes per core (the eltwise in two halves?), else 32 (V: res40, res64)
                d["meta"] = chunk("META_DATA", 64 if d["din"].ppc > 4 else 32, [at, at + 1], bt)
                d["ins"] = [rch[i, n] if (i, n) in rch else tens[i] for i in ins[n]]
                full[n] = True
            else:
                d["out"] = chunk("CONCAT", Dist(cout).ppc * cells(ho, wo) * 64, [unfl, unfl], bt) \
                    if n == out else store.get(n)
                full[n] = True
            if k == "pool" and i0 in segs:                  # a pool keeps its input's segments
                segs[n] = d["segs"] = segs[i0]
            if n in roots:                                  # a stored bulb root: a first citizen (V: g030),
                d["out"].batch = (0, tail[n], 0)            # all its slots written once loaded back (V: g021)
                full[n] = True
            tens[n] = d["out"]
            info[n] = d
        for n, d in info.items():
            if "weight" in d:
                # the WEIGHT chunk as NPUC's planner sizes it: its own pack of the kernel (planning rows; for a conv
                # reading a concat in SRAM the unpermuted channel order), while the loads use the emitted block's rows
                # (weights.weight_layer_info; V: c16_8_8-k5-pool, c32_16-k5-conv, g161, g103, g167)
                pr = d.get("prows", d["wunits"])
                d["ring"] = weight_ring(dict(d, wunits=pr))
                d["weight"].size = d["ring"] * 2 * pr * 64
                if d["ring"] < d["passes"] and d["ring"] <= 4:      # a short ring lives on its conv only
                    d["weight"].live = [d["out"].live[0], d["out"].live[0]]
                if by[n]["kind"] == "relu" and not d["whole"]:  # NPUC sizes the per-pass dummy conv's chunk as 4
                    # cores' worth over the cores used (V: relu-c1..c24); a whole-reading one as a conv's (V: g029)
                    d["weight"].size = cdiv(4 * d["weight"].size, d["dout"].cores)
        # the first citizens, in order of their live ranges: at 0 when free there, else at the top of SRAM, below
        # any top one live at the same time (V: cat16, g002, g021, g028 at the top; g010, g012 at 0; g141, g169)
        fc = sorted((ck for ck in chunks if ck.batch[0] == 0), key=lambda ck: ck.batch)
        for i, ck in enumerate(fc):
            ov = [o for o in fc[:i] if o.live[0] <= ck.live[1] and ck.live[0] <= o.live[1] and o.size]
            if not any(o.off < ck.size for o in ov):
                ck.off = 0
            else:
                ck.off = (min([SRAM_LIMIT] + [o.off for o in ov if o.top]) - ck.size) // 64 * 64
            ck.top = ck.off > 0
            ck.fixed = True
        allocate(chunks)
        self.chunks = chunks
        need = max(ck.off + ck.size for ck in chunks if not getattr(ck, "top", False))
        if need > min([SRAM_LIMIT] + [ck.off for ck in chunks if getattr(ck, "top", False)]):
            raise Unsupported("tiling in a graph (SRAM %d B per core)" % need)
        for n, d in info.items():
            if "into" in d:
                kn, slot = d["into"]
                d["out"].off = store[kn].off + slot * cells(d["ho"], d["wo"]) * 64
        for L in order:
            if L["kind"] == "conv" and max(hw[L["name"]]) > 255:
                raise Unsupported("tiling in a graph (an output dimension of %s > 255)" % L["name"])
        # groups

        def load_kind(L, d):
            # NPUC loads an input per core when an NFU op reads it first and a core holds several planes, else
            # whole (its LIR: "LOAD shape 1x8x32x28" x 4 vs "1x32x32x28"; V: g020, g025)
            nfu = L["kind"] == "pool" or (L["kind"] == "conv" and (d["haspad"] or d.get("s2d") == 2))
            return "core" if nfu and (d["din"].ppc > 1 or d["din"].c % 2) else "whole"     # (g175: 3 channels)

        self.info = []
        prev = None                         # what the layer run just before loaded from DRAM, and how
        for L in order:
            n, k = L["name"], L["kind"]
            d = info[n]
            i0 = ins[n][0]
            if k == "concat":
                continue
            # a reader of a tensor in DRAM loads it, unless the input is shared by a concat's branches and an earlier
            # branch loaded it, or the layer just before loaded it the same way (NPUC's LIR skips an identical load;
            # V: add2, in-two vs add-branches, g020; g022: a concat's buffer read by two convs)
            firstr = next(x["name"] for x in order if x["name"] in cons[i0]) if i0 in cons else None
            ld = (i0, load_kind(L, d)) if i0 in dsrc and k != "eltwise" else None
            if i0 == src and shared:
                first = n == firstr
            elif ld is not None:
                first = ld != prev
            else:
                first = i0 in roots and n == firstr
            prev = ld
            if i0 != src and k != "eltwise":
                # an intermediate loaded whole comes in one DMA per distinct per-core plane count; loaded per core,
                # in one (V: g008's ReLU, 20 channels: 2; g030, g047's padded convs, 36 and 20 channels: 1)
                dl = d["din"]
                even = dl.planes - (dl.cores - 1) * dl.ppc == dl.ppc
                d["parts"] = 1 if even or load_kind(L, d) == "core" else 2
            last = n in stored
            ev0 = self.ev
            if k in ("conv", "pool"):
                self.info.append(d)
                ops = self.layer_group(len(self.info) - 1, L, first, last)
            elif k == "relu":
                ops = self.relu_group(L, d, first, last)
            else:
                ops = self.elt_group(L, d, ins[n], dsrc, last)
            for op in ops:
                if op.kind == "load" and op.what == "ifm":
                    op.tensor = i0
                elif op.kind == "store":
                    op.tensor = stored[n][1]
            self.groups += self.cut(ops, ev0) if k != "eltwise" else [ops]
        if by[out]["kind"] == "concat":
            o = {"dout": Dist(chs[out]), "ho": hw[out][0], "wo": hw[out][1]}
        else:
            o = info[out]
        nu = unflatten_tiles(o, o["ho"], o["wo"])
        if nu > 1:
            raise Unsupported("tiling in a graph (the unflatten of %d x %d)" % (o["ho"], o["wo"]))
        self.out_chw = (o["dout"].c, o["ho"], o["wo"])
        self.groups.append([Op("load", what="dram", flag=self.event(), utile=len(self.groups), tensor=out),
                            Op("nopop", deps=[0]), Op("unflatten")])

    def relu_group(self, L, d, first, last):
        """a standalone ReLU: NPUC's 1x1 dummy conv, one pass per pair of plane slots (V: res, relu-c4..c64)"""
        din, dout = d["din"], d["dout"]
        ops = []
        parts = d.get("parts", 1) if first else 0
        for j in range(parts):
            ops.append(Op("load", what="ifm", flag=self.event(), sram=d["in"].off,
                          **({"part": j} if parts > 1 else {})))
        # a pass's weight slot: the chunk (4 cores' worth over the cores used) over the passes, in whole rows (V:
        # relu-c18: 342 B, 2 passes 192 B apart)
        stride = cdiv(d["weight"].size // d["ring"], 64)
        # each pass reads its own 2 plane slots (4 channels), except from a tensor a conv wrote (a conv's output, a
        # concat in SRAM) or one loaded from DRAM (a concat's buffer), which every pass reads whole (V: cat16-relu,
        # g008; ops-check relu-q3, a ReLU after a conv; per pass: after a pool or an eltwise, relu-c32, res32-k1)
        whole = d.get("whole", False)
        for p in range(d["passes"]):
            wb = d["weight"].off // 64 + (p % d["ring"]) * stride
            ops.append(Op("load", what="weight", flag=self.event(), layer=L["name"], p=p, sram=wb * 64))
            deps = [len(ops) - 1] + list(range(parts))
            ops.append(Op("conv", relu=not whole, relu_whole=whole, L=L, din=din, dout=dout, p=p,
                          passes=d["passes"], hin=d["h"], win=d["w"], kin=2 * din.ppc if whole else 4, k=[1, 1],
                          ho=d["ho"], wo=d["wo"], src=d["in"].off // 64, ofm=d["out"].off // 64, wbase=wb,
                          wunits=d["wunits"], nsub=d["nsub"], cpb=d["cpb"], wslot=0, ps=None, deps=deps))
        if last:
            rem = dout.planes - (dout.cores - 1) * dout.ppc
            ops.append(Op("store", flags=[self.event() for _ in range(1 if rem == dout.ppc else 2)],
                          sram=d["out"].off))
        return ops

    def elt_group(self, L, d, names, dsrc, last):
        """an eltwise add: an ARM intrinsic group (swi 0x10000 | group), its DMAs done by the firmware: the operands
        loaded back from DRAM when one is the network input (the input first; one load per distinct per-core plane
        count), the 32 B META_DATA, then one IR element per core, a nop, and the store when the result goes to
        DRAM. Their event flags count from 1 within the group, except META_DATA's, which takes the network's next
        flag when the planes per core are 4 or fewer (V: res, res24, res32-big, add2; fresh: res40, res48, res64,
        add-pool48, add2-64-*)"""
        din, dout = d["din"], d["dout"]
        ops = []
        fresh = [0]

        def fflag():
            fresh[0] += 1
            return 1 << (fresh[0] - 1)

        # two loads per operand when the cores hold different plane counts (V: res20, g006), and for exactly 4
        # channels, where an operand a conv wrote comes as its two plane slots (V: rc4-*, g120, g134; not 2, 3, 6, 12)
        rem = din.planes - (din.cores - 1) * din.ppc
        four = din.c == 4
        parts = 1 if rem == din.ppc and not four else 2
        if any(t in dsrc for t in names):
            for t, ck in sorted(zip(names, d["ins"]), key=lambda x: x[0] not in dsrc):
                for part in range(parts):
                    slot = part if four and t not in dsrc else 0
                    if slot:                # the second plane slot's load takes the network's next flag (V: g134,
                        flag = self.event()                 # rc4-*, g120)
                    else:
                        flag = fflag()
                    ops.append(Op("load", what="eltwise_in", arm=True, flag=flag, tensor=t, part=part,
                                  sram=ck.off + slot * cells(d["h"], d["w"]) * 64))
        if din.ppc <= 4 and din.cores > 1:  # the network's next flag, even one the group's loads use (V: g537)
            flag = self.event()
        else:
            flag = fflag()
        ops.append(Op("load", what="eltwise_meta", arm=True, flag=flag, sram=d["meta"].off))
        # 4 channels loaded back: 4 elements (V: rc4-*); in SRAM, one per core (V: g513)
        ops += [Op("elt") for _ in range(4 if four and any(t in dsrc for t in names) else din.cores)]
        if din.cores == 1:
            # one core: no hand-over nop; the ARM step waits on an event of its own, a network flag with no DMA
            # behind it (V: rc2-*)
            if last:
                raise Unsupported("a one-core eltwise whose result goes to DRAM")
            ops.append(Op("eltnop2", store=None, flags=[self.event()]))
            return ops
        ops.append(Op("eltnop", more=last))
        if last:
            rem = dout.planes - (dout.cores - 1) * dout.ppc
            ops.append(Op("store", arm=True, flags=[self.event() for _ in range(1 if rem == dout.ppc else 2)],
                          sram=d["out"].off))
            si = len(ops) - 1
            ops += [Op("skip") for _ in ops[si].flags[1:]]     # each store DMA its own IR element (V: g540)
            ops.append(Op("eltnop2", store=si))
        return ops


# ---- emission ----------------------------------------------------------------------------------------------------
def emit_pad(c, op, q, tail):
    """an NFU pad command (one plane slot, one piece)"""
    L, din = op.L, op.din
    cm = op.cm
    pt, pb, pl, pr = getattr(op, "tpads", L["pads"])
    u8 = (1 << 26) if q == 1 else 0
    c.w(R[1], 4)                                    # ";must be more than 4 commands before 'wait' command"
    c.w(INTC + 0x28, cm)
    c.w(INTC + 0x24, cm)
    c.w(NFU + 0x800, 5)
    c.w(NFU + 0x228, 0x10000)
    c.w(NFU + 0x800, 0xA)
    c.w(NFU + 0x228, 0x10000)
    c.w(NFU + 0x800, cm)
    pc, rc = op.pc, op.rc
    pl, pr = (pl if pc["first"] else 0), (pr if pc["last"] else 0)
    pt, pb = (pt if rc["first"] else 0), (pb if rc["last"] else 0)
    c.w(NFU + 0x208, pb | pt << 4 | pr << 8 | pl << 12)
    c.w(NFU + 0x214, u8 | (2 * pc["skip"]) << 20 | rc["skip"] << 18 | (2 * (pc["b"] - 8 * pc["cell"])) << 8 |
        (rc["b"] - 4 * rc["cell"]))
    c.w(NFU + 0x218, u8 | (2 * (pc["o1"] - pc["o0"])) << 8 | (rc["o1"] - rc["o0"]))
    c.w(NFU + 0x218, u8 | (2 * (pc["o1"] - pc["o0"])) << 8 | (rc["o1"] - rc["o0"]))
    c.w(NFU + 0x22C, (0x10000 | L["in_zp"]) if q == 1 else 0)
    c.w(NFU + 0x20C, cdiv(op.w, 8) << 16 | (op.src + rc["cell"] * cdiv(op.w, 8) + pc["cell"]))
    c.w(NFU + 0x210, cdiv(op.wp, 8) << 16 | (op.dst + rc["o0"] // 4 * cdiv(op.wp, 8) + pc["o0"] // 8))
    c.w(NFU + 0x200, 0x30000)
    c.w(NFU + 0x2FC, cm)
    c.nop(0 if tail else 4)
    c.waiti3(0, 1, 4)
    c.w(INTC + 0x24, 0)
    c.w(INTC + 0x28, cm)


def emit_trim(c, op, q, tail):
    """NFU copy of a tile's input region (mode 0x20100): from the loaded region (start aligned to a cell) to the
    rows and columns the tile needs (V: tw256, th300, ts208, tw320c16)"""
    cm = op.cm
    pc = op.pc
    u8 = (1 << 26) if q == 1 else 0
    c.w(R[1], 4)
    c.w(INTC + 0x28, cm)
    c.w(INTC + 0x24, cm)
    c.w(NFU + 0x800, 5)
    c.w(NFU + 0x228, 0x10000)
    c.w(NFU + 0x800, 0xA)
    c.w(NFU + 0x228, 0x10000)
    c.w(NFU + 0x800, cm)
    c.w(NFU + 0x208, 0)
    rc = op.rc
    c.w(NFU + 0x214, u8 | (2 * pc["skip"]) << 20 | rc["skip"] << 18 | (2 * (pc["b"] - 8 * pc["cell"])) << 8 |
        (rc["b"] - 4 * rc["cell"]))
    c.w(NFU + 0x218, u8 | (2 * (pc["o1"] - pc["o0"])) << 8 | (rc["o1"] - rc["o0"]))
    c.w(NFU + 0x218, u8 | (2 * (pc["o1"] - pc["o0"])) << 8 | (rc["o1"] - rc["o0"]))
    c.w(NFU + 0x22C, 0x10000 if q == 1 else 0)
    c.w(NFU + 0x20C, cdiv(op.wl, 8) << 16 | (op.src + rc["cell"] * cdiv(op.wl, 8) + pc["cell"]))
    c.w(NFU + 0x210, cdiv(op.wn, 8) << 16 | (op.dst + rc["o0"] // 4 * cdiv(op.wn, 8) + pc["o0"] // 8))
    c.w(NFU + 0x200, 0x20100)
    c.w(NFU + 0x2FC, cm)
    c.nop(0 if tail else 4)
    c.waiti3(0, 1, 4)
    c.w(INTC + 0x24, 0)
    c.w(INTC + 0x28, cm)


def emit_pool(c, op, q, tail):
    """NFU pooling (V: pmax*, pavg* variants)"""
    L, din = op.L, op.din
    cm = op.cm
    pt, pb, pl, pr = op.pads
    kh, kw = L["k"]
    sh, sw = L["stride"]
    pc, rc = op.pc, op.rc

    def used(p, unit):          # the input extent field: from the start (first piece), from the cell (last), or
        if p["first"]:          # the piece's own inputs (middle pieces; V: w240max3, w200avg3; G for rows)
            return p["b"]
        if p["last"]:
            return p["b"] - unit * p["cell"]
        return p["b"] - p["a"]
    hu, wu = used(rc, 4), used(pc, 8)
    pl, pr = (pl if pc["first"] else 0), (pr if pc["last"] else 0)
    pt, pb = (pt if rc["first"] else 0), (pb if rc["last"] else 0)
    u8 = (1 << 26) if q == 1 else 0
    c.w(R[1], 4)
    c.w(INTC + 0x28, cm)
    c.w(INTC + 0x24, cm)
    c.w(NFU + 0x800, 5)
    c.w(NFU + 0x228, 0x10000)
    c.w(NFU + 0x800, 0xA)
    c.w(NFU + 0x228, 0x10000)
    c.w(NFU + 0x800, cm)
    c.w(NFU + 0x208, pb | pt << 4 | pr << 8 | pl << 12)
    c.w(NFU + 0x214, u8 | (2 * pc["skip"]) << 20 | rc["skip"] << 18 | (2 * wu) << 8 | hu)
    c.w(NFU + 0x218, u8 | (2 * (pc["o1"] - pc["o0"])) << 8 | (rc["o1"] - rc["o0"]))
    c.w(NFU + 0x218, u8 | (2 * (pc["o1"] - pc["o0"])) << 8 | (rc["o1"] - rc["o0"]))
    c.w(NFU + 0x22C, (0x10000 | L["in_zp"]) if q == 1 else 0)
    c.w(NFU + 0x204, (2 if L["type"] == "AvgPool" else 0) << 16 | kw << 12 | kh << 8 | sw << 4 | sh)
    c.w(NFU + 0x20C, cdiv(op.w, 8) << 16 | (op.src + rc["cell"] * cdiv(op.w, 8) + pc["cell"]))
    c.w(NFU + 0x210, cdiv(op.wo, 8) << 16 | (op.dst + rc["o0"] // 4 * cdiv(op.wo, 8) + pc["o0"] // 8))
    c.w(NFU + 0x200, 0)
    c.w(NFU + 0x2FC, cm)
    c.nop(0 if tail else 4)
    c.waiti3(0, 1, 4)
    c.w(INTC + 0x24, 0)
    c.w(INTC + 0x28, cm)


def emit_reformat(c, op, q, tail):
    """NFU reformat for a stride-2 conv: space to depth of one input plane slot into 4 phase planes, in two halves
    (bit 30, bit 31 of 0x4016030c; V: s2* variants)"""
    L, din = op.L, op.din
    cm = op.cm
    for half in (1, 2):
        c.w(INTC + 0x18, cm)
        c.w(INTC + 0x14, cm)
        c.w(NFU + 0x800, 5)
        c.w(NFU + 0x310, 0x10000)
        c.w(NFU + 0x800, 0xA)
        c.w(NFU + 0x310, 0x10000)
        c.w(NFU + 0x800, cm)
        pw, ph = op.x1 - op.x0, op.y1 - op.y0
        c.w(NFU + 0x30C, half << 30 | cdiv(pw, 8) << 24 | cdiv(ph, 4) << 16 | pw << 8 | ph)
        c.w(R[1], 2)                            # ";must be more than 4 commands before 'wait' command"
        c.w(NFU + 0x304, cdiv(op.wp, 8) << 16 | (op.src + op.y0 // 4 * cdiv(op.wp, 8) + op.x0 // 8))
        c.w(NFU + 0x308, cdiv(op.wr, 8) << 16 | (op.dst + op.y0 // 8 * cdiv(op.wr, 8) + op.x0 // 16))
        c.w(NFU + 0x300, cells(op.hr, op.wr) << 16)
        c.w(NFU + 0x314, (0x10000 | L["in_zp"]) if q == 1 else 0)
        c.w(NFU + 0x318, 0)
        c.w(NFU + 0x3FC, cm)
        c.nop(0 if (tail and half == 2) else 4)
        c.waiti3(0, 1, 2)
        c.w(INTC + 0x14, 0)
        c.w(INTC + 0x18, cm)


def shift_code(s):
    """the MTC's encoding of a requantisation shift"""
    return s << 11 | s << 6


def emit_conv(c, op, q, fixed, tail, cache=None):
    """one MTC convolution pass: its registers, weights, partial sums, GReLU table and shifts"""
    L, din, dout, p = op.L, op.din, op.dout, op.p
    kh, kw = op.k
    c.w(MTC + 0x004, 0x420)
    c.w(MTC + 0x040, 0x07000000 if q == 1 else 0x300)
    if q == 1:
        c.w(MTC + 0x1B0, L["in_zp"] << 16 | L["w_zp"])
        c.w(MTC + 0x1B4, L["out_zp"])
    c.w(MTC + 0x008, 0)
    adv = getattr(op, "relu", False)        # a standalone ReLU's dummy conv: pass p reads plane slots 2p, 2p + 1
    relu = adv or getattr(op, "relu_whole", False)
    c.w(MTC + 0x780, cells(op.ho, op.wo) & 0x3FF)    # 10 bits: NPUC masks (V: t192, t200)
    c.w(MTC + 0x0A0, cdiv(op.win, 8) << 16 | (op.src + (2 * p * cells(op.hin, op.win) if adv else 0)))
    c.w(MTC + 0x180, cells(op.hin, op.win))
    for k in range(din.cores):
        c.w(MTC + 0x240 + 4 * k, op.kin)
    c.w(MTC + 0x044, din.mask)
    c.w(MTC + 0x1A0, op.hin << 16 | op.win)
    c.w(MTC + 0x788, op.ho << 16 | op.wo)
    c.w(MTC + 0x19C, 0)
    c.w(MTC + 0x480, kh << 16 | kw)
    c.w(MTC + 0x488, (cdiv(kw - 1, 8) + 1) << 8 | (cdiv(kh - 1, 4) + 1))
    c.w(MTC + 0x784, op.passes)
    # bit 28 asymmetric (quant 1), bits 8..15 the GReLU mode (3), low byte: the weight sub-batch count when the
    # output is more than one cell, else 1 (G: ci64k7, ci128k5, ci256h6, ci128s2h16 vs ci256 4x4, ci128s2 8x8)
    rounds = op.nsub if cells(op.ho, op.wo) > 1 else 1
    # the ReLU's dummy conv: GReLU mode 1, no table (V: res, relu-*); a ReLU6's: mode 3 with NPUC's clamp table
    # (grelu.py npuc_relu6_blocks; V: ops-check relu6-q3)
    table = not relu or L.get("type") == "ReLU6"
    c.w(MTC + 0x940, (0x10000000 if q == 1 else 0) | (0x300 if table else 0x100) | rounds)
    for a in range(0x400, 0x410, 4):
        c.w(MTC + a, 0)
    for a in range(0x440, 0x460, 4):
        c.w(MTC + a, 0)
    c.w(R[1], 0x100)                                # ";must be more than 4 commands before 'wait' command"
    for k in range(din.cores):
        for j in range(2):
            c.w(MTC + 0x300 + 8 * k + 4 * j, op.wunits << 16 | (op.wbase + (2 * op.wslot + j) * op.wunits))
    c.w(MTC + 0x580, 0xFFFFFF)
    c.w(MTC + 0x048, 0xFFFF)
    c.w(MTC + 0x04C, 0xFFFF)
    for a, v in (grelu_writes(L, q, p, fixed, cache) if table else []):
        c.w(a, v)
    c.w(MTC + 0x200, op.nsub << 16 | op.cpb)    # weight sub-batches x channels each (V: ci128s2, ci256)
    c.w(MTC + 0x0B0, cdiv(op.wo, 8) << 16)
    oc = cells(op.ho, op.wo)
    for k in range(4):
        for j in range(2):
            c.w(MTC + 0x700 + 8 * k + 4 * j, op.ofm + (2 * p + j) * oc)
    sc = 0 if relu else shift_code(layer_shift(L))
    for k in range(4):
        for j in range(2):
            v = 0
            for cc in range(2):
                if out_channel(dout, k, p, j, cc) is not None:
                    v |= sc << (16 * cc)
            c.w(MTC + 0x900 + 8 * k + 4 * j, v)
    for k in range(4):
        for j in range(2):
            for cc in range(2):
                o = out_channel(dout, k, p, j, cc)
                b = L["bias"][o] if o is not None and not relu else 0
                a = MTC + 0x800 + 32 * k + 8 * (2 * j + cc)
                c.w(a, b & 0xFFFFFFFF)
                c.w(a + 4, 0xFFFFFFFF if b < 0 else 0)
    if op.ps is None:
        for a in range(0x590, 0x5A0, 4):
            c.w(MTC + a, 0)
    else:                                       # partial sums (V: ci64k7, ci128k5, ci255k7, ci256, ci128s2...)
        ic = cells(op.hin, op.win)
        c.w(MTC + 0x540, 7 * cdiv(op.win, 8) << 16 | 7 * ic)
        for k in range(4):
            c.w(MTC + 0x590 + 4 * k, 0xF << 4 * k)
        for k in range(4):
            c.w(MTC + 0x500 + 4 * k, (op.ps + 7 * ic) << 16 | op.ps)
    c.w(INTC + 0x08, 0x11)
    c.w(INTC + 0x04, 0x11)
    c.w(MTC + 0x000, 0x11)
    c.nop(0 if tail else 4)              # no nops when the op ends its group (V: pmax, two, cpc)
    c.waiti3(0, 1, 0x100)
    c.w(INTC + 0x04, 0)
    c.w(INTC + 0x08, 0x11)


def clear_regs(c):
    """the registers NPUC clears at the end of a group"""
    for i in range(1, 16):
        c.w(R[i], 0)


def group_schedule(ops, plan):
    """NPUC's event schedule of one group from its generated memory profile: (lines {op index: ev2set mask},
    waits {op index: mask}, schedule, elements), or None when the group is not modelled (an ARM group) or its
    events disagree with the DMAs' flags"""
    els = npuc_profile(ops, plan)
    if not els or not any(_is_dma(e) for e in els):
        return None
    want, store = {}, None
    for e in els:
        i = e["idx"]
        o = ops[i] if i < len(ops) else None
        if o is not None and o.kind == "load":
            want[i] = o.flag
        elif o is not None and o.kind == "store":
            store, nxt = o, 1
            want[i] = o.flags[0]
        elif _is_dma(e):            # a store's further DMAs (skip elements, or appended)
            if store is None or nxt >= len(store.flags):
                return None
            want[i] = store.flags[nxt]
            nxt += 1
    first = next(e["idx"] for e in els if _is_dma(e))
    sc = npuc_schedule(els, want[first].bit_length() - 1)
    for i, f in want.items():
        if [1 << ev for ev in sc["sig"][i]] != [f]:
            return None
    lines, waits = npuc_lines(els, sc)
    return lines, waits, sc, els


def emit_group(gi, ops, plan, last, fixed):
    """the instructions of one group (its ops in NPUC's order, the DMA triggers and waits of its event schedule)"""
    c = Code()
    for op in ops:
        op.group = gi
    q = plan.q
    sched = group_schedule(ops, plan)
    if sched is not None:
        # the io batch ids (SetBatchTransferInGroup; V on every compile, also groups of 64 loads or more: k3c64-ci512
        # one id for its loads, tiling3 ra rings loaded core by core one per trigger)
        ids, plan.bid = npuc_batch_ids(sched[3], sched[2], getattr(plan, "bid", 0))
        for i, b in ids.items():
            if i < len(ops) and ops[i].kind in ("load", "store"):
                ops[i].bid = b
    for i, op in enumerate(ops):
        if op.kind in ("load", "elt", "skip") or getattr(op, "arm", False):
            continue                    # the ARM's DMAs and the eltwise's IR elements: nothing in the CMDQ
        if op.kind == "store":
            if sched is None:
                plan.bid = getattr(plan, "bid", 0) + 1
                op.bid = plan.bid
            m = 0
            for f in op.flags:
                m |= f
            c.w(EV2SET, m << 16)
            c.nop(4)
            more = any(o.kind != "skip" for o in ops[i + 1:])
            c.wait_event(m, trailing=more)      # mid-group (pass by pass): 4 nops after (V: rc-167-488)
            continue
        c.w(R[9], i)
        if op.kind in ("nopop", "eltnop", "eltnop2"):
            c.nop(5)
        if sched is not None:
            # the trigger batches: NPUC's event schedule on the group's memory profile (cmdq.md 7b): an ev2set line
            # where a wait finds no line open (its mask ORs in the waits up to the next trigger or pool reset), then
            # the element's own wait
            if i in sched[0]:
                c.w(EV2SET, sched[0][i] << 16)
            if i in sched[1]:
                c.wait_event(sched[1][i])
        elif any(ops[d].kind == "load" and not getattr(ops[d], "arm", False) for d in op.deps):
            raise Unsupported("the trigger batches of a group whose memory profile is not modelled")
        tail = i == len(ops) - 1
        if op.kind == "pad":
            emit_pad(c, op, q, tail)
        elif op.kind == "conv":
            emit_conv(c, op, q, fixed, tail, plan.grelu_cache)
        elif op.kind == "pool":
            emit_pool(c, op, q, tail)
        elif op.kind == "reformat":
            emit_reformat(c, op, q, tail)
        elif op.kind == "trim":
            emit_trim(c, op, q, tail)
        elif op.kind == "nopop":
            pass
        elif op.kind == "eltnop":       # hand the eltwise to the ARM, wait for it (V: res, add2, res-c3)
            plan.bid = getattr(plan, "bid", 0) + 1
            for o in ops:
                if o.kind == "load" and getattr(o, "arm", False):
                    o.bid = plan.bid
            c.w(SWI, 0x10000 | gi)
            c.nop(1)
            c.w(R[1], 0xFFFF0000 | gi)
            c.nop(4)
            c.wait3(15, 1, 0xFFFF0000 | gi)
            c.nop(4)
            c.w(R[15], 0)
            if op.more:
                c.nop(4)
        elif op.kind == "eltnop2":      # its result to DRAM: the store's event, then the ARM again (V: add2)
            if op.store is None:        # one core: the ARM loads' batch here, an event with no DMA (V: rc2-*)
                plan.bid = getattr(plan, "bid", 0) + 1
                for o in ops:
                    if o.kind == "load" and getattr(o, "arm", False):
                        o.bid = plan.bid
                flags = op.flags
            else:
                st = ops[op.store]
                plan.bid += 1
                st.bid = plan.bid
                flags = st.flags
            m = 0
            for f in flags:
                m |= f
            c.w(EV2SET, m << 16)
            c.wait_event(m)
            c.w(SWI, 0x10000 | gi)
            c.nop(1)
            c.w(SWI, 0xFFFF0000 | gi)
            c.nop(4)
            c.w(R[1], 0xFFFF0000 | gi)
            c.nop(4)
            c.wait3(15, 1, 0xFFFF0000 | gi)
            c.w(R[15], 0)
            c.w(R[1], 0)
            c.w(R[3], 0)
        elif op.kind == "unflatten":
            c.ins.pop()          # the unflatten carries no r9 write of its own (V)
            c.w(SWI, 0x10000 | gi)
            c.nop(1)
            c.w(R[1], 0xFFFF0000 | gi)
            c.nop(4)
            c.wait3(15, 1, 0xFFFF0000 | gi)
            c.nop(4)
            c.w(R[15], 0)
            c.w(R[1], 0)
            c.w(R[3], 0)
        else:
            raise Unsupported(op.kind)
    clear_regs(c)
    c.w(R[10], gi)
    if last:
        c.w(SWI, 0x8000)
    c.nop(4)
    while len(c.ins) % 8:
        c.nop(1)
    return c


def group_streams(low, fixed=False):
    """the isa bytes of every group of the network (the isa area padded to 64 bytes)"""
    plan = Plan(low)
    codes = [emit_group(gi, ops, plan, gi == len(plan.groups) - 1, fixed) for gi, ops in enumerate(plan.groups)]
    out = [c.bytes() for c in codes]
    total = sum(len(b) for b in out)
    out[-1] += b"\0" * ((-total) % 64)       # the isa area as a whole is a multiple of 64 bytes (V)
    return out


def plan_dmas(low, fixed=False):
    """every DMA of the network in io-descriptor order, as the stream uses it (for the container, frame.py):
    dicts with n, flag (trig_flag), bid (the io batch id, 1-based: NPUC's SetBatchTransferInGroup on the group's event
    schedule, npuc_batch_ids; V), sgid (= the NCP group), kind (ifm / weight / store / dram), sram (byte offset
    in each core's SRAM), and for tiles: tile, core (per-core DMA) and region (rows y0..y1, columns x0..x1 of the
    tensor: the loaded input region, or the stored output region); weight loads: layer, p (pass). Memoised by the network's
    content (frame.py asks for it from each plan)."""
    return _DMAS.get((network_key(low), fixed, frame._IN_STREAM), lambda: _plan_dmas(low, fixed))


_DMAS = Memo(8)


def _plan_dmas(low, fixed):
    plan = Plan(low)
    for gi, ops in enumerate(plan.groups):
        emit_group(gi, ops, plan, gi == len(plan.groups) - 1, fixed)
    out = []
    for ops in plan.groups:
        for op in ops:
            if op.kind == "load":
                out.append({"kind": op.what, "flag": op.flag, "bid": getattr(op, "bid", None), "sgid": op.group,
                            **{k: getattr(op, k) for k in ("tile", "core", "sram", "region", "layer", "p", "slots",
                                                             "rows") if hasattr(op, k)}})
            elif op.kind == "store":
                for i, f in enumerate(op.flags):
                    out.append({"kind": "store", "flag": f, "bid": op.bid, "sgid": op.group, "part": i,
                                **{k: getattr(op, k) for k in ("tile", "sram", "region", "tensor") if hasattr(op, k)}})
                    if hasattr(op, "srams"):
                        out[-1]["sram"] = op.srams[i]
    for n, d in enumerate(out):
        d["n"] = n
    return out


# ---- NPUC's event schedule (cmdq.md 7b) ---------------------------------------------------------------------------------
# NPUC places its DMA triggers from a dependency graph over each group's IR elements (EventSchedule): a memory map of
# the elements' SRAM reads and writes gives the graph, the representatives keep one link per chain (a compute -> DMA
# link is a trigger), a 16-event pool gives the DMA -> compute links their events, and the codegen writes one ev2set
# line per run of waits between two triggers or pool resets. emit_group runs it on npuc_profile's generated profile;
# `events` replays it on the profile NPUC logs ("IRIndex: n", "<op> n Pushes SRAM: a, b", "<op> Refers a, b").
def _profile(block):
    """the elements of one group's memory profile: index, op name, reads and writes (inclusive byte ranges)"""
    els, cur = [], None
    for line in ("IRIndex: 0 \n" + block).splitlines():
        if "IRIndex" not in line and "SRAM" not in line and "Refers" not in line and "Event Summary" not in line:
            continue
        line = re.sub(r"^alog \d/NPUC: \[[^\]]*\]\[[^\]]*\]\[[^\]]*\]\[[^\]]*\] ", "", line).strip()
        if line.startswith("Event Summary"):
            break
        m = re.match(r"IRIndex: (\d+)", line)
        if m:
            cur = {"idx": int(m.group(1)), "name": None, "refs": [], "pushes": []}
            els.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"(ncp_\w+) \d+ Pushes SRAM: (0x[0-9a-f]+), (0x[0-9a-f]+)", line)
        if m:
            cur["name"] = m.group(1)
            cur["pushes"].append((int(m.group(2), 16), int(m.group(3), 16)))
            continue
        m = re.match(r"(ncp_\w+) Refers (0x[0-9a-f]+), (0x[0-9a-f]+)", line)
        if m:
            cur["name"] = cur["name"] or m.group(1)
            r = (int(m.group(2), 16), int(m.group(3), 16))
            if not (cur["name"] == "ncp_blob_load" and cur["pushes"] and r == cur["pushes"][-1]):
                cur["refs"].append(r)       # (a load's ref of its own write is the write's check, not a read)
    return els


def _is_dma(e):
    return e["name"] in ("ncp_blob_load", "ncp_blob_store")


def _etype(e):
    """NPUC's IRElement type: 0 a DMA, 1 a CMDQ op (NFU, MTC, the nop), 2 an ARM intrinsic (eltwise, unflatten)"""
    if _is_dma(e):
        return 0
    return 2 if e["name"] in ("ncp_eltwise_op", "ncp_unflatten") else 1


# ---- NPUC's memory profile, generated (cmdq.md 7b) --------------------------------------------------------------------
# The per-element SRAM reads and writes NPUC's GetMemoryProfile gives its dependency graph, as it logs them (V: every
# element of every group of the chain and tiled sets, element by element against _profile's parse of the logs): byte
# ranges per IDP (core k at k x 0x40000; the MTC's weight view continues at 0x100000).
IDP_SPAN = 0x40000
_PROFILE_NAMES = {"load": "ncp_blob_load", "store": "ncp_blob_store", "skip": "ncp_blob_store",
                  "pad": "ncp_nfu_pooling", "pool": "ncp_nfu_pooling", "trim": "ncp_nfu_pooling",
                  "reformat": "ncp_nfu_reformat", "conv": "ncp_mtc_convolution", "nopop": "ncp_nop",
                  "unflatten": "ncp_unflatten"}


def _span(a, n):
    return (a, a + n - 1)


def _distinct(vals):
    out = []
    for v in vals:
        if v and v not in out:
            out.append(v)
    return out


def npuc_profile(ops, plan):
    """the memory profile of one group's IR elements (ops), as NPUC logs it: [{idx, name, refs, pushes}], refs and
    pushes inclusive byte ranges; None for a group it does not model (ARM groups: eltwise). Formats (V):
    - load: one write per IDP of its cores, its SRAM offset + k x 0x40000, the bytes per core (input: planes per core x
      cells; weights: the slot, 2 x rows x 64, on the conv's input cores, or its own core); a per-core load writes
      plane j at IDP core + j (as logged); an input loaded in parts covers the cores of its per-core plane count; the
      read-back's load: per IDP when one plane per core or 4 cores of full planes, else the planes in one range;
    - store: reads per IDP the planes of each core, one element per distinct per-core count (a tile's per-core stores:
      one element per core, one range per plane);
    - pad, pool, trim (one NFU piece): one read per core of 2 x rows x columns bytes from the piece's first cell (rows
      and columns up to the piece's input end, a pool's last piece to the input's end), and its output rows (the
      piece's width in cells, rows of 4 lines) written back to back from the piece's first output cell;
    - reformat: its input rows (piece width, row stride of the plane) and per phase plane its output rows;
    - conv: every input row of every MTC input plane (kin / 2) on every input core, then 8 weight halves (k x 0x40000
      + the slot + (k odd) x half; empty ranges at k x 0x40000 past 2 x the input cores), then its pass's two OFM plane
      slots, row by row, on the cores of the 16-rounded output;
    - nop: reads and writes each IDP whole; unflatten: reads per IDP 64 bytes a plane (its channels x 32 in one range
      unless they fill the cores evenly)."""
    els, extra, pend_skip = [], [], []

    def consumer(i):
        return next((o for o in ops if i in getattr(o, "deps", [])), None)

    def producer(i):
        return next((o for o in reversed(ops[:i]) if o.kind in ("conv", "pool", "pad", "reformat", "trim")), None)
    for i, op in enumerate(ops):
        k = op.kind
        if k in ("elt", "eltnop", "eltnop2") or getattr(op, "arm", False) or k not in _PROFILE_NAMES:
            return None
        refs, pushes = [], []
        core = getattr(op, "core", None)
        if k == "load":
            c = consumer(i)
            if op.what == "weight":
                if c is None or c.kind != "conv":
                    return None
                for d in ([core] if core is not None else range(c.din.cores)):
                    pushes.append(_span(op.sram + d * IDP_SPAN, 2 * c.wunits * 64))
            elif op.what == "dram":
                C, H, W = getattr(plan, "out_chw", None) or plan.low["npu_out_chw"]
                nu = sum(1 for g in plan.groups if any(o.kind == "unflatten" for o in g))
                u = getattr(plan, "_uf_done", 0)
                th = cdiv(cdiv(H, nu), 4) * 4
                rows = min(H, (u + 1) * th) - u * th
                dd = Dist(C)
                if dd.ppc == 1 or (dd.cores == 4 and C == 8 * dd.ppc):
                    if core is not None:
                        pushes += [_span((core + j) * IDP_SPAN, cells(rows, W) * 64) for j in range(dd.ppc)]
                    else:
                        pushes += [_span(d * IDP_SPAN, dd.ppc * cells(rows, W) * 64) for d in range(dd.cores)]
                else:
                    pushes.append(_span(0, dd.planes * cells(rows, W) * 64))
                if C == 2 * dd.cores * dd.ppc and (dd.ppc == 1 or dd.cores == 4):
                    plan._uf_refs = [_span(d * IDP_SPAN, dd.ppc * 64) for d in range(dd.cores)]
                else:
                    plan._uf_refs = [_span(0, C * 32)]
            else:
                if c is None:
                    return None
                h, w = {"conv": ("hin", "win"), "trim": ("hl", "wl"), "reformat": ("hp", "wp")}.get(c.kind, ("h", "w"))
                din, h, w = c.din, getattr(c, h), getattr(c, w)
                plane = cells(h, w) * 64
                if core is not None:            # all ppc planes, also on a short last core (V: tw300c20)
                    pushes += [_span(op.sram + (core + j) * IDP_SPAN, plane) for j in range(din.ppc)]
                elif getattr(op, "part", None) is not None:
                    cnt = [min(din.ppc, din.planes - d * din.ppc) for d in range(din.cores)]
                    v = _distinct(cnt)[op.part]
                    pushes += [_span(op.sram + d * IDP_SPAN, v * plane) for d in range(din.cores) if cnt[d] == v]
                else:
                    n = din.ppc * plane
                    if getattr(op, "slots", None):
                        n = (min(op.slots[1], din.ppc) - op.slots[0]) * plane
                    pushes += [_span(op.sram + d * IDP_SPAN, n) for d in range(din.cores)]
        elif k == "store":
            p = producer(i)
            if p is None or p.kind not in ("conv", "pool"):
                return None
            if p.kind == "conv":
                dl = p.dout
                cnt = [min(dl.ppc, dl.planes - d * dl.ppc) for d in range(dl.cores)]
                if getattr(op, "part_pass", None) is not None:
                    cnt = [min(2, max(0, x - 2 * op.part_pass)) for x in cnt]
                if getattr(op, "cnt", None) is not None:
                    cnt = op.cnt
            else:
                dl = p.din
                cnt = [sum(1 for o in ops if o.kind == "pool" and o.L is p.L and o.pc["first"] and o.rc["first"]
                           and o.cm >> d & 1) for d in range(dl.cores)]
            oc = cells(p.ho, p.wo) * 64
            if hasattr(op, "tile") and len(op.flags) > 1:
                parts = [[_span(op.sram + d * IDP_SPAN + j * oc, oc) for j in range(cnt[d])] for d in range(dl.cores)]
            else:
                parts = [[_span(op.sram + d * IDP_SPAN, v * oc) for d in range(dl.cores) if cnt[d] == v]
                         for v in _distinct(cnt)]
            refs = parts[0]
            for j, rr in enumerate(parts[1:]):
                if i + 1 + j < len(ops) and ops[i + 1 + j].kind == "skip":
                    pend_skip.append(rr)
                else:
                    extra.append({"idx": i + 1 + j, "name": "ncp_blob_store", "refs": rr, "pushes": []})
        elif k == "skip":
            refs = pend_skip.pop(0) if pend_skip else []
        elif k in ("pad", "pool", "trim"):
            pc, rc = op.pc, op.rc
            if k == "pool":
                w, wo = op.w, op.wo
                hi = (op.h if rc["last"] else rc["b"]) - 4 * rc["cell"]
                wi = (w if pc["last"] else pc["b"]) - 8 * pc["cell"]
            else:
                w, wo = (op.w, op.wp) if k == "pad" else (op.wl, op.wn)
                wi, hi = pc["b"] - 8 * pc["cell"], rc["b"] - 4 * rc["cell"]
            a = (op.src + rc["cell"] * cdiv(w, 8) + pc["cell"]) * 64
            dm = [d for d in range(4) if op.cm >> d & 1]
            refs = [_span(a + d * IDP_SPAN, 2 * hi * wi) for d in dm]
            a = (op.dst + rc["o0"] // 4 * cdiv(wo, 8) + pc["o0"] // 8) * 64
            rw = cdiv(pc["o1"] - pc["o0"], 8) * 64
            pushes = [_span(a + d * IDP_SPAN + r * rw, rw) for d in dm for r in range(cdiv(rc["o1"] - rc["o0"], 4))]
        elif k == "reformat":
            pw, ph = op.x1 - op.x0, op.y1 - op.y0
            dm = [d for d in range(4) if op.cm >> d & 1]
            a = (op.src + op.y0 // 4 * cdiv(op.wp, 8) + op.x0 // 8) * 64
            rs, rw = cdiv(op.wp, 8) * 64, cdiv(pw, 8) * 64
            refs = [_span(a + d * IDP_SPAN + r * rs, rw) for d in dm for r in range(cdiv(ph, 4))]
            a = (op.dst + op.y0 // 8 * cdiv(op.wr, 8) + op.x0 // 16) * 64
            rs, rw = cdiv(op.wr, 8) * 64, cdiv(cdiv(pw, 2), 8) * 64
            pl = cells(op.hr, op.wr) * 64
            pushes = [_span(a + d * IDP_SPAN + f * pl + r * rs, rw) for d in dm for f in range(4)
                      for r in range(cdiv(cdiv(ph, 2), 4))]
        elif k == "conv":
            rw = cdiv(op.win, 8) * 64
            # a standalone ReLU's dummy conv reads its pass's own plane slots (emit_conv's 0x401200a0; V: g502)
            a = (op.src + (2 * op.p * cells(op.hin, op.win) if getattr(op, "relu", False) else 0)) * 64
            refs = [_span(a + d * IDP_SPAN + r * rw, rw) for d in range(op.din.cores)
                    for r in range(op.kin // 2 * cdiv(op.hin, 4))]
            half = op.wunits * 64
            for d in range(8):
                if d < 2 * op.din.cores:
                    refs.append(_span(op.wbase * 64 + d * IDP_SPAN + (d & 1) * half, half))
                else:
                    refs.append((d * IDP_SPAN, d * IDP_SPAN - 1))
            rw = cdiv(op.wo, 8) * 64
            a = op.ofm * 64 + 2 * op.p * cells(op.ho, op.wo) * 64
            pushes = [_span(a + d * IDP_SPAN + r * rw, rw) for d in range(Dist(cdiv(op.dout.c, 16) * 16).cores)
                      for r in range(2 * cdiv(op.ho, 4))]
        elif k == "nopop":
            refs = [_span(d * IDP_SPAN, IDP_SPAN) for d in range(4)]
            pushes = list(refs)
        elif k == "unflatten":
            refs = getattr(plan, "_uf_refs", None)
            if refs is None:
                return None
            plan._uf_done = getattr(plan, "_uf_done", 0) + 1
        els.append({"idx": i, "name": _PROFILE_NAMES[k], "refs": refs, "pushes": pushes})
    if extra:
        if els[-1]["idx"] != extra[0]["idx"] - 1:
            return None
        els += extra
    return els


def npuc_event_graph(els):
    """BuildGraphByDataDeps with its MemoryManager: MemObjs [start, end, references, owner]; a range is found by a
    MemObj that holds its start or its end (end - 1 tolerated); a read takes the last such MemObj (counting a reference
    on each), links its owner -> reader and becomes its owner, or is pushed when none; a load's write is checked the
    same way; every write is pushed: each MemObj it meets gives owner -> writer (or each of the owner's successors ->
    writer) and is dropped, the write appended. Returns (predecessors, successors) by element index."""
    # the MemObj list in insertion order (a push appends, removals keep the order), found by 4 KiB page: a MemObj is
    # hit when it holds the range's start or meets [end - 1, end]; the reference counts never change an edge
    pred = {e["idx"]: set() for e in els}
    succ = {e["idx"]: set() for e in els}
    pages = {}
    seq = [0]

    def hits(o):
        found = {}
        for pg in {o[0] >> 12, (o[1] - 1) >> 12, o[1] >> 12}:
            for m in pages.get(pg, ()):
                if m[4] and (m[0] <= o[0] <= m[1] or (m[0] <= o[1] and o[1] - 1 <= m[1])):
                    found[m[2]] = m
        return [found[k] for k in sorted(found)]

    def edge(a, b):
        if a != b:
            succ[a].add(b)
            pred[b].add(a)

    def ref(o):
        h = hits(o)
        return h[-1] if h else None

    def push(o, who):
        for m in hits(o):
            for s in (list(succ[m[3]]) or [m[3]]):
                edge(s, who)
            m[4] = False
        seq[0] += 1
        m = [o[0], o[1], seq[0], who, True]
        for pg in range(o[0] >> 12, (o[1] >> 12) + 1):
            lst = pages.setdefault(pg, [])
            if len(lst) > 64:
                lst[:] = [x for x in lst if x[4]]
            lst.append(m)
    for e in els:
        i = e["idx"]
        for o in e["refs"]:
            m = ref(o)
            if m is None:
                push(o, i)
            else:
                edge(m[3], i)
                m[3] = i
        for o in e["pushes"]:
            if e["name"] == "ncp_blob_load":
                m = ref(o)
                if m is not None:
                    edge(m[3], i)
                    m[3] = i
            push(o, i)
    return pred, succ


def npuc_reps(els, pred, succ):
    """UpdateRepresentative then ReorganizeRepresent (V, code and disassembly): each element's latest predecessor and
    earliest successor of another type (a DMA's successor: a CMDQ or ARM element, or a store; a load's predecessor:
    no DMA), then in IR order, with a count of the DMAs passed since the last reset:
    - a CMDQ element whose successor is the very next element (a DMA): the count resets;
    - one whose successor is a later DMA: the link moves to the last CMDQ element before that DMA (an earlier DMA a
      CMDQ element in between feeds takes its place; the walk ends on the 16th DMA of the count, which becomes the
      target); the CMDQ elements in between point to that last one (dropping the DMA links they had); the count
      resets;
    - one with no successor: it points to the next CMDQ element that has one (the count resets: the device flag of
      that branch is off, tiling3 rb015), unless the 16th DMA of the count comes first: then the latest CMDQ element
      before that DMA with no successor of its own (or itself) links to it, and it points to that element (the count
      resets). This forced link keeps a run of DMAs to 15 events (a first conv with 16+ passes: ra/rc g0).
    Returns (prv, nxt) by element index."""
    by = {e["idx"]: e for e in els}
    order = [e["idx"] for e in els]
    pos = {k: n for n, k in enumerate(order)}
    ty = {e["idx"]: _etype(e) for e in els}
    prv, nxt = {}, {}
    for e in els:
        i, t = e["idx"], ty[e["idx"]]
        cs = [x for x in succ[i] if ty[x] != t or (t == 0 and by[x]["name"] == "ncp_blob_store")]
        nxt[i] = min(cs) if cs else None
        if t == 0 and e["name"] == "ncp_blob_load":
            ps = [x for x in pred[i] if ty[x] != 0]
        elif t == 0:
            ps = list(pred[i])
        else:
            ps = [x for x in pred[i] if ty[x] != t]
        prv[i] = max(ps) if ps else None
    cnt = 0
    for n, i in enumerate(order):
        if ty[i] == 0:
            cnt += 1
            continue
        nn = nxt[i]
        if nn is None:
            c = cnt
            for j in order[n + 1:]:
                if ty[j] == 0:
                    c += 1
                    if c >= 16:
                        k = pos[j] - 1
                        while k > n and ty[order[k]] != 1:
                            k -= 1
                        x = order[k]
                        while x != i and nxt[x] is not None:
                            k = pos[x] - 1
                            while k > n and ty[order[k]] != 1:
                                k -= 1
                            x = order[k]
                        nxt[i] = x
                        nxt[x] = j
                        prv[j] = x
                        cnt = 0
                        break
                elif ty[j] == 1 and nxt[j] is not None:
                    nxt[i] = j
                    if prv[j] is None:
                        prv[j] = i
                    cnt = 0
                    break
            continue
        if ty[nn] == 1:
            continue
        if pos[nn] - n == 1:
            cnt = 0
            continue
        cand, last = nn, i
        k = n + 1
        while k < len(order) and pos[order[k]] < pos[cand]:
            f = order[k]
            if ty[f] == 1:
                fn = nxt[f]
                if fn is not None and ty[fn] == 0 and pos[fn] < pos[cand]:
                    cand = fn
                last = f
            else:
                cnt += 1
                if cnt > 15:
                    cand = f
                    break
            k += 1
        if last != i:
            for g in reversed(order[n:pos[last]]):
                if ty[g] == 1:
                    p = nxt[g]
                    if p is not None and ty[p] == 0 and prv.get(p) == g:
                        prv[p] = None
                    nxt[g] = last
        nxt[last] = cand
        if last != i:
            nxt[i] = last
        prv[cand] = last
        cnt = 0
    return prv, nxt


def npuc_schedule(els, cursor=0):
    """UpdateEventWaitFieldDataDeps (V, code): the events of one group. In IR order, after releasing the events whose
    release position is this element: an element linked to one of another type takes an event, a DMA linked to a
    CMDQ element from the 16-event pool (round robin from the network-wide cursor; when all 16 are taken the pool
    resets, the element gets flag 75, and the consumer of the DMA whose event last filled the pool gets flag 77),
    released where the consumer's chain of representatives reaches a DMA (the group's end when none); any other link
    (to a DMA or an ARM element) from a one-event pool, released at its target; the event goes into the target's wait
    list, except for a store followed by a load (flag 73). A DMA with no successor takes a pool event released at the
    next CMDQ element (flag 73). Returns {"prv", "nxt", "sig", "wait", "f73", "f75", "f77", "cursor"}."""
    pred, succ = npuc_event_graph(els)
    prv, nxt = npuc_reps(els, pred, succ)
    by = {e["idx"]: e for e in els}
    ty = {e["idx"]: _etype(e) for e in els}
    end = els[-1]["idx"] + 1
    st = {"m": 0, "cur": cursor}
    rel = {}
    sig = {e["idx"]: [] for e in els}
    wait = {e["idx"]: [] for e in els}
    f73, f75, f77 = set(), set(), set()
    last_fill = None

    def alloc():
        c = st["cur"]
        for ev in list(range(c, 16)) + list(range(0, min(c, 16))):
            if not st["m"] >> ev & 1:
                st["m"] |= 1 << ev
                st["cur"] = ev + 1
                return ev
        return -1
    for n, e in enumerate(els):
        i = e["idx"]
        for b in rel.pop(i, []):
            st["m"] &= ~b
        if e["name"] in ("ncp_unflatten", "ncp_permute", "ncp_flatten"):
            continue
        dst = nxt[i]
        if dst is not None:
            if ty[dst] == ty[i]:
                continue
            if ty[dst] == 1:
                ev = alloc()
                if ev < 0:
                    st["m"] = 0
                    ev = alloc()
                    f75.add(i)
                    if last_fill is not None:
                        f77.add(last_fill)
                    keep = None
                else:
                    keep = dst if st["m"] == 0xFFFF else last_fill
                r = nxt[dst]
                while r is not None and ty[r] == 1:
                    r = nxt[r]
                rel.setdefault(end if r is None else r, []).append(1 << ev)
            else:
                ev = 0
                keep = dst if st["m"] == 0xFFFF else last_fill
            sig[i].append(ev)
            last_fill = keep
            if e["name"] == "ncp_blob_store" and n + 1 < len(els) and els[n + 1]["name"] == "ncp_blob_load":
                f73.add(i)
                continue
            wait[dst].append(ev)
            continue
        if ty[i] != 0:
            continue
        ev = alloc()
        f73.add(i)
        if ev < 0:
            continue
        sig[i].append(ev)
        nc = next((x["idx"] for x in els[n + 1:] if ty[x["idx"]] == 1), None)
        if nc is not None:
            rel.setdefault(nc, []).append(1 << ev)
    return {"prv": prv, "nxt": nxt, "sig": sig, "wait": wait, "f73": f73, "f75": f75, "f77": f77,
            "cursor": st["cur"]}


def npuc_lines(els, sc):
    """the ev2set lines and waits the codegen emits for the CMDQ elements (V, code: IsaFunc2020::addWaitFromCPU,
    addSwiTrigger, PostProcessing): an element with a wait list waits for those events; it opens a new ev2set line
    when the running trigger mask is empty, else ORs its events into the open line (which is rewritten); the
    running mask is cleared after an element that signals a DMA (its trigger) and after one flagged 77. Returns
    (lines {idx: mask}, waits {idx: mask})."""
    run, cur, lines, waits = 0, None, {}, {}
    for e in els:
        i = e["idx"]
        if _etype(e) == 1:
            m = 0
            for ev in sc["wait"][i]:
                m |= 1 << ev
            if m:
                waits[i] = m
                if run == 0:
                    cur = i
                run |= m
                lines[cur] = run
            if sc["sig"][i]:
                run = 0
        if i in sc["f77"]:
            run = 0
    return lines, waits


def npuc_batch_ids(els, sc, bid=0):
    """BlobTransfer::SetBatchTransferInGroup (V, code): each run of DMAs of one name gets a batch id: a new one when
    its head has a wait (a CMDQ element signals it), the previous run's otherwise; after a run whose last DMA has
    flag 73 this run and the next get new ids. Returns ({idx: bid}, the last bid)."""
    out = {}
    runs_, cur = [], []
    for e in els:
        if _is_dma(e) and cur and cur[-1]["name"] == e["name"]:
            cur.append(e)
            continue
        if cur:
            runs_.append(cur)
        cur = [e] if _is_dma(e) else []
    if cur:
        runs_.append(cur)
    prev, force = None, False
    for r in runs_:
        if prev is None or force:
            bid += 1
            force = False
        elif prev[-1]["idx"] in sc["f73"]:
            bid += 1
            force = True
        elif sc["wait"][r[0]["idx"]]:
            bid += 1
        for e in r:
            out[e["idx"]] = bid
        prev = r
    return out, bid


def _ncp_lines(stream):
    """NPUC's stream: its ev2set lines and waits by CMDQ element (r9), as npuc_lines gives them"""
    r9, done = None, False
    lines, waits = {}, {}
    for op, v in decode(stream):
        if op == 0x8000 | (0x124 >> 2):
            r9, done = v, False
        elif op == 0x8000 | (0x2024 >> 2):
            if not done and r9 is not None:
                lines[r9] = v >> 16
        elif op == 0x3A61 and not done and r9 is not None:
            waits[r9] = waits.get(r9, 0) | v
        elif op == 0x3A01:
            done = True
    return lines, waits


def events(dirs):
    """replay NPUC's event schedule from each compile's logged memory profile and compare its ev2set lines and waits
    (position, mask) with NPUC's streams; ARM groups (eltwise) keep the firmware's events: after one the network's
    cursor continues at stream.py's own next flag"""
    ok = tot = 0
    for d in dirs:
        try:
            want = ncp_streams(open(os.path.join(d, "net.ncp"), "rb").read())
            txt = open(os.path.join(d, "log.txt"), errors="replace").read()
        except OSError:
            continue
        blocks = re.split(r"\n[^\n]*IRIndex: 0 \n", txt)[1:]
        cursor, flags = 0, None
        for g, blk in enumerate(blocks[:len(want)]):
            els = _profile(blk)
            if not els:
                continue
            if any(e["name"] == "ncp_eltwise_op" for e in els):
                if flags is None:
                    try:
                        flags = {}
                        for x in plan_dmas(json.load(open(os.path.join(d, "lowered.json")))):
                            flags.setdefault(x["sgid"], x["flag"].bit_length() - 1)
                    except (OSError, ValueError, Unsupported):
                        pass
                later = [k for k in flags if k > g]
                if later:
                    cursor = flags[min(later)]
                continue
            sc = npuc_schedule(els, cursor)
            cursor = sc["cursor"]
            if not any(_is_dma(e) for e in els):
                continue
            got = npuc_lines(els, sc)
            nl = _ncp_lines(want[g])
            tot += 1
            ok += got == nl
            if got != nl:
                print("%s g%d: NPUC %s, replay %s" % (os.path.basename(os.path.normpath(d)), g, _fmt_lines(nl),
                                                      _fmt_lines(got)))
    print("events: %d of %d groups" % (ok, tot))


def _fmt_lines(lw):
    return " ".join("%d:%x" % kv for kv in sorted(lw[0].items())) + " / waits " + \
        " ".join("%d:%x" % kv for kv in sorted(lw[1].items()))


# ---- reading the oracle's NCP --------------------------------------------------------------------------------------
REG_NAMES = {CMDQ: "CMDQ", MTC: "MTC", NFU: "NFUBR", INTC: "INTCOM"}


def ncp_streams(ncp):
    """the isa bytes of each group of an NCP"""
    return ncp_parts(ncp)["streams"]


def fmt(op, v):
    """one instruction as text"""
    a = set_reg_addr(op)
    if a is not None:
        return "%-6s %#010x = %#010x" % (REG_NAMES[a & ~0x3FFF], a, v)
    if op == 0 and v == 0:
        return "nop"
    return "op%04x %#010x" % (op, v)


decode = decode_isa


def check(dirs):
    """our streams against each variant's net.ncp; True when none differs"""
    n_ok = n_bad = n_uns = 0
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        try:
            low = json.load(open(os.path.join(d, "lowered.json")))
            want = ncp_streams(open(os.path.join(d, "net.ncp"), "rb").read())
        except (OSError, ValueError) as e:
            print("%-14s missing (%s)" % (name, e))
            n_uns += 1
            continue
        try:
            got = group_streams(low)
        except Unsupported as e:
            print("%-14s unsupported: %s" % (name, e))
            n_uns += 1
            continue
        if got == want:
            print("%-14s exact (%d groups, %d B)" % (name, len(got), sum(map(len, got))))
            n_ok += 1
            continue
        n_bad += 1
        why = "groups %d vs %d" % (len(got), len(want)) if len(got) != len(want) else ""
        for g, (a, b) in enumerate(zip(got, want)):
            if a != b:
                da, db = decode(a), decode(b)
                i = next((i for i in range(min(len(da), len(db))) if da[i] != db[i]), min(len(da), len(db)))
                why = why or "g%d @%d: ours %s, NPUC %s (%d vs %d B)" % (
                    g, i, fmt(*da[i]) if i < len(da) else "end", fmt(*db[i]) if i < len(db) else "end", len(a), len(b))
                break
        print("%-14s differs: %s" % (name, why))
    print("total: %d exact, %d differ, %d unsupported, of %d" % (n_ok, n_bad, n_uns, len(dirs)))
    return n_bad == 0


def show(d, diff=False):
    """print our streams (with diff: only the instructions that differ from NPUC's)"""
    low = json.load(open(os.path.join(d, "lowered.json")))
    got = group_streams(low)
    want = ncp_streams(open(os.path.join(d, "net.ncp"), "rb").read()) if diff else None
    for g, b in enumerate(got):
        da = decode(b)
        db = decode(want[g]) if want and g < len(want) else []
        for i, x in enumerate(da):
            y = db[i] if i < len(db) else None
            if diff and x == y:
                continue
            print("g%d %4d %-36s%s" % (g, i, fmt(*x), ("   NPUC " + fmt(*y)) if diff and y else ""))
        if diff and len(db) > len(da):
            for i in range(len(da), len(db)):
                print("g%d %4d %-36s   NPUC %s" % (g, i, "-", fmt(*db[i])))


def main(argv=None):
    """the command line (module docstring)"""
    ap = argparse.ArgumentParser(prog="stream.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="our streams against NPUC's compiles")
    p.add_argument("dirs", nargs="+")
    p = sub.add_parser("show", help="our streams as text")
    p.add_argument("dir")
    p.add_argument("--diff", action="store_true", help="only the instructions that differ from NPUC's")
    p = sub.add_parser("events", help="NPUC's event schedule replayed from its logs")
    p.add_argument("dirs", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return 0 if check(a.dirs) else 1
    if a.cmd == "show":
        show(a.dir, a.diff)
    else:
        events(a.dirs)
    return 0


if __name__ == "__main__":
    from ncpgen import stream as _pkg    # the package's copy of this module (one copy of every module)
    sys.exit(_pkg.main())
