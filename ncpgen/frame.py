#!/usr/bin/env python3
"""The NCP container, the ARM intrinsics and the NPU's data layout (docs/format/frame.md). Pure Python, our own code: every rule below was read off NPUC 1.36.1.a's output (the
oracle, oracle/npuc-local.sh) or its disassembly; nothing here is a copied vendor table or blob.

  frame.py check <variant dir>...      assemble each variant from the oracle NCP's own streams and weight blocks (cut
                                       out of it) with frame() and intrinsics(), compare the whole file byte for byte
  frame.py plan <variant dir|lowered.json> [weight sizes]   print the plan (groups, SRAM chunks, vectors, DMAs)
  frame.py layout C H W [c y x]        the IFM/OFM sizes, and the byte address of element (c, y, x)
  frame.py selftest                    the layout against the NPU-measured 8x16x16 map, the A32 encoder

Interface (ncpgen.py, stream.py, npunet.py):
  frame(low, streams, intrinsics, weights, fix_npuc=False) -> bytes   the whole NCP from its parts (streams = isa
                         bytes per group, intrinsics = bytes per group, weights = WEIGHT blocks in NCP order)
  intrinsics(low) -> [bytes per group]
  plan(low, weight_sizes=None, strict=True, fix_npuc=False) -> dict   NPUC's plan: internal shapes, ops, SRAM chunks
                         (offset/size/live range), vectors, io descriptors, groups (the stream generator needs the same
                         SRAM offsets: chunks[k]["off"]); tiled plans add "strata" (blocks, tiles, per-tile ranges) and
                         "outputs" (the output groups)
  ifm_addr(c, y, x, C, H, W), ifm_size(C, H, W)       the input byte layout (format 0x120 cells, IDP-major)
  ofm_addr(c, y, x, C, H, W), ofm_size(C, H, W)       the output byte layout (dense CHW, format 0x110)
  ncp_layout(low, ncp)                                a checked drop-in for ncpgen/npunet.py's ncp_layout (any shape)

Scope (frame_sweep.py and the other parts' sweeps; docs/format/frame.md section 1): chains of conv (k 1..7,
stride 1 or 2, SAME/VALID, 1..512 channels) and max/avg pool layers, in one tile or tiled (plan_tiled: strata of
chained layers cut into tiles by rows, columns or both, single-tile blocks between them, split output read-back; which
layers, chains and tile counts by NPUC's own policy read from its code, npuc_policy, frame.md 6.10), with
weights resident or streamed through a ring of slots, PARTIAL_SUM chunks for sub-batched weights, and NPUC's copy
group when no allocation order fits a layer (frame.md 6.6, 6.7); graph networks with eltwise adds, concats and
standalone ReLUs (plan_graph, from stream.py's plan; frame.md 6.8). frame()'s weights are the WEIGHT blocks in
the NCP's vector order (weights.py's weight_blocks; for graphs the execution order, META_DATA included).

Tags in comments: V = seen in the oracle's bytes/log or the disassembly, G = guess (consistent with all variants,
not proven).
"""
import argparse
import itertools
import json
import os
import struct
import sys

if not __package__:     # run as a script: the package is this file's directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncpgen import stream, weights  # noqa: E402
from ncpgen.common import (MV_IM, MV_IN, MV_OT, MV_P0, MV_WT, Memo, cdiv, decode_isa, exec_order, first_diff,  # noqa
                           is_graph, ncp_header, ncp_parts, ncp_vectors, net_outputs, network_key, rup, topology)

# ---- constants of the format (V: NCPObjectGenerator2020::PackingHeader, dossier/npu-compiler.md 1.2) ----------------
MAGIC1, MAGIC2 = 0x0C0FFEE0, 0xC0DEC0DE
HDR_VERSION, INTRINSIC_VERSION, NET_ID = 22, 21, 100
AV_SIZE, MV_SIZE, GV_SIZE, IO_SIZE = 16, 32, 40, 124
RQ_PER_IO = 360            # V: "rq vector size" = 360 x io count, the zero tail after ncp_end
IDP = 4                    # V: four IDP SRAMs (OT_PIX0..3), each addressed 0x40000 apart, 1 MiB vector each
IDP_STRIDE = 0x40000
SRAM_VEC = 1 << 20         # V: av size of OT_PIX0..3
FMT_CELL, FMT_CHW = 0x120, 0x110   # V: 288 = the NPU's cell format, 272 = dense CHW (output intrinsic)
FW_TABLE = 0x1D000         # V: the firmware's intrinsic table (entries 3, 5, 8 used)
SRAM_LIMIT = IDP_STRIDE    # V: a plan whose live chunks exceed one IDP's 256 KiB SRAM streams its weights through a
#                            ring (a smaller WEIGHT chunk), or is tiled by NPUC
SRAM_SAFE = 240 * 1024     # G: when weights.py cannot tell a conv's sub-batches (its PARTIAL_SUM chunk), a plan above
#                            this may already overflow NPUC's count: refuse


# ---- the data layout -------------------------------------------------------------------------------------------------
def cell(C, H, W):
    """How a C x H x W tensor sits in the four IDP SRAMs and in a DRAM blob of format 0x120 (V for the sizes and
    descriptors of every variant; V for the byte order of 8x16x16 by the N3b impulse probe; G for other shapes):
      - cells of 4 lines x 8 columns x 2 channels (64 B): Hp = H rounded up to 4, Wp = W rounded up to 8; a plane holds
        two channels, Hp x 2Wp bytes;
      - channels are dealt to IDPs in blocks: cpi channels per IDP, cpi = ceil(C/4) rounded up to even, n = ceil(C/cpi)
        IDPs used; IDP i holds channels i*cpi .. i*cpi+cpi-1 as pp = cpi/2 planes;
      - in DRAM the IDPs follow each other (IDP-major), each n*pp planes long (the DMA copies IDP i's block to SRAM
        i*0x40000)."""
    Hp, Wp = rup(H, 4), rup(W, 8)
    cpi = rup(cdiv(C, 4), 2)
    n = cdiv(C, cpi)
    pp = cpi // 2
    plane = Hp * 2 * Wp
    return {"C": C, "H": H, "W": W, "Hp": Hp, "Wp": Wp, "cpi": cpi, "n": n, "pp": pp, "plane": plane,
            "per_idp": pp * plane, "size": n * pp * plane}


def ifm_addr(c, y, x, C, H, W):
    """byte offset of input element (c, y, x) in the IFM buffer the NCP's IN_FMAP DMA reads (format 0x120)"""
    L = cell(C, H, W)
    i, loc = divmod(c, L["cpi"])
    p, s = divmod(loc, 2)
    return (i * L["pp"] + p) * L["plane"] + (y // 4) * 4 * 2 * L["Wp"] + (x // 8) * 64 + s * 32 + (y % 4) * 8 + x % 8


def ifm_size(C, H, W):
    """the IFM buffer's size in bytes (cell())"""
    return cell(C, H, W)["size"]


def ofm_addr(c, y, x, C, H, W):
    """the output (OT_FMAP) is dense CHW: the output intrinsic converts format 0x120 to 0x110 (V for the sizes of every
    variant; V on the NPU for 8x16x16 and 8x8x8)"""
    return (c * H + y) * W + x


def ofm_size(C, H, W):
    """the OFM's size in bytes (dense CHW)"""
    return C * H * W


# ---- the plan: NPUC's single-tile schedule, SRAM allocation, DRAM vectors and DMAs ------------------------------------
def internal_shapes(low):
    """NPUC's backend shapes (C, H, W) of every layer's input and output (V: its IR graph dumps). They equal the
    lowered (TFLite) shapes except where a VALID pool drops columns: NPUC's NNPoolLayer::computeOutputShape warns
    'Padding in Pooling has negative value!' and, when the vertical padding (out-1)*s + k - in is >= 0 but the
    horizontal one is negative, keeps one more output column (V on 25 variants; G for the exact condition). The wider
    tensor then flows on: a conv computes its output as (in + pads - k) // s + 1 from the wider input (its lowered pads
    unchanged), a pool recomputes the same rule from it; the output intrinsic finally drops the extra column (its
    destination row stride is negative)."""
    shapes = []
    C, H, W = low["input"]["chw"]
    for L in low["layers"]:
        kh, kw = L["k"]
        sh, sw = L["stride"]
        pt, pb, pl, pr = L["pads"]
        Co = L["out_chw"][0]
        Hp_, Wp_ = H + pt + pb, W + pl + pr
        Ho, Wo = (Hp_ - kh) // sh + 1, (Wp_ - kw) // sw + 1
        if L["kind"] == "pool" and not any(L["pads"]):
            padh, padw = (Ho - 1) * sh + kh - H, (Wo - 1) * sw + kw - W
            if padw < 0 <= padh:
                Wo += 1
        shapes.append({"in": (C, H, W), "out": (Co, Ho, Wo)})
        C, H, W = Co, Ho, Wo
    return shapes


def _layer_ops(L, li, shp):
    """the atomic operations NPUC lowers one layer to (V: its 'LA started for node' / 'Creating an operation' prints),
    each with the SRAM chunk it defines (per-IDP size), on NPUC's internal shapes"""
    C, H, W = shp["in"]
    Co, Ho, Wo = shp["out"]
    pt, pb, pl, pr = L["pads"]
    ops = []
    if L["kind"] == "conv":
        sh, sw = L["stride"]
        Hc, Wc = H, W
        if any(L["pads"]):
            Hc, Wc = H + pt + pb, W + pl + pr
            ops.append({"op": "PAD", "layer": li, "size": cell(C, Hc, Wc)["per_idp"]})
        if (sh, sw) != (1, 1):
            # V (stride 2): a space-to-depth inside each IDP: its cpi channels become cpi*sh*sw phase channels of
            # ceil(Hpad/sh) x ceil(Wpad/sw), i.e. cpi*sh*sw/2 planes per IDP (G for strides > 2)
            r = cell(C, cdiv(Hc, sh), cdiv(Wc, sw))
            ops.append({"op": "REFORMAT", "layer": li, "size": r["cpi"] * sh * sw // 2 * r["plane"]})
        # V: "Convolution Ofm will be rounded to 16 ... due to HW bug": the SRAM OFM holds a multiple of 16 channels
        ops.append({"op": "CONV", "layer": li, "size": cell(rup(Co, 16), Ho, Wo)["per_idp"],
                    "direct": not any(L["pads"]) and (sh, sw) == (1, 1)})
    elif L["kind"] == "pool":
        ops.append({"op": "POOL", "layer": li, "size": cell(Co, Ho, Wo)["per_idp"]})
    else:
        raise NotImplementedError("layer kind %r" % L["kind"])
    return ops


def _place(order, placed):
    """first fit of chunks in this order beside the placed ones: the lowest 64-aligned offset clear of every placed
    chunk whose live range overlaps (inclusive; V: r3-095, the chunk after a ring of 183504 B sits at 183552)"""
    out = []
    for ch in order:
        busy = sorted((p["off"], p["off"] + p["size"]) for p in placed + out
                      if p["live"][0] <= ch["live"][1] and ch["live"][0] <= p["live"][1])
        off = 0
        for a, b in busy:
            if off + ch["size"] <= a:
                break
            off = max(off, rup(b, 64))
        out.append(dict(ch, off=off))
    return out


def _first_fit(chunks, placed=None, cap=None, overflow=False, ties_created=False):
    """V: NPUC's allocator, as its log shows it: per allocation group (the input, then each layer), the chunks
    'bigger first' (largest first; ties: the chunk created last first, i.e. WEIGHT, CONV, PAD: V on all ties seen),
    each at the lowest offset clear of every placed chunk whose live range [first op, last op] (inclusive) overlaps;
    PARTIAL_SUM chunks ("tail") after the group's others (V: c256, ci64k7). With a capacity, when a group overflows
    it (V: "Cannot allocate with current strategy, bail out!"): 'backward' allocation, the chunks by the start of
    their live range, latest first (equal starts: smaller first, the reverse of bigger first), tails still last (V: the
    45 logged compiles that try it, r_c256co128, r3-064, t-32x32-256-512, final-chains f014, tiling2 b10: a CONV and
    its WEIGHT starting together go smaller first), then "all other permutations" of that order, tails included
    (lexicographic, as std::next_permutation; V: r_c256co128's third try puts its PARTIAL_SUM before its PAD).
    Returns the placed chunks; when nothing fits, None, or with overflow=True the bigger-first placement (the caller
    sees the overflow). ties_created: equal sizes in creation order (NPUC's retry after a copy, _single_block)."""
    fixed = list(placed or [])
    done = list(fixed)
    for grp in chunks:
        main = sorted((c for c in grp if not c.get("tail")),
                      key=lambda c: (-c["size"], grp.index(c) if ties_created else -grp.index(c)))
        tail = [c for c in grp if c.get("tail")]
        res = _place(main + tail, done)
        if cap is not None and max([c["off"] + c["size"] for c in res] or [0]) > cap:
            back = sorted(main[::-1], key=lambda c: -c["live"][0]) + tail
            for t in itertools.islice(itertools.permutations(back), 0, 5041):
                res = _place(list(t), done)
                if max(c["off"] + c["size"] for c in res) <= cap:
                    break
            else:
                if not overflow:
                    return None
                res = _place(main + tail, done)
        done += res
    return done[len(fixed):]


def _im_vector(C, H, W, meta=None, round16=True):
    """the DRAM blob a layer's output is stored to (IM_FMAP) and its memory vector (V: DRAM size = roundup(C, 16)
    channels in cells; the vector from NCPObjectGenerator2020::FillVectors @0x9c230, disassembled): when w*h*c of the
    tensor covers the blob, the vector is (W, H, C) (W rounded to 8, H to 4, C to even, G for the rounding unit of W);
    else ("Shape is not matching with allocation size") w = size/4/2, h 4, c 2 when size/4 <= 8192, otherwise w = the
    largest power of two <= 8192 dividing size/4 (not below 8, G) and c = size/4/w rounded up to even. av size = w*h*c
    + 10*(w*h*c >> 8) + 10."""
    L = cell(rup(C, 16) if round16 else C, H, W)
    size = L["n"] * L["pp"] * L["plane"]
    if size <= (C * H * W if meta is None else meta):
        w, h, c = rup(W, 8), rup(H, 4), rup(C, 2)
    else:
        q = size // 4
        h = 4
        if q <= 8192:
            w, c = q // 2, 2
        else:
            w = 8192
            while q % w and w > 8:
                w //= 2
            c = q // w
            c += c & 1
    raw = w * h * c
    return {"w": w, "h": h, "c": c, "dram": size, "av": raw + 10 * (raw >> 8) + 10}


def _blob(vec, off, n, c, h, w, ws=0, cs=0, ns=0, fmt=FMT_CELL, bpp=128):
    return [vec, off, fmt, bpp, n, c, h, w, ws, cs & 0xFFFFFFFF, ns]


def _sram_dma(idps, h, w):
    """the SRAM side of a DMA over 'idps' consecutive IDPs of h x w bytes each: one blob per IDP 0x40000 apart (V)"""
    return idps, h, w, (IDP_STRIDE - h * w) if idps > 1 else 0


def _read_back(dmas, g, ol, Co, v_im, fix_npuc):
    """the output group's read-back of the stored output (V): spread over the IDPs as stored when every used IDP is
    full and (4 IDPs or 2 channels per IDP) (observed rule, G for its reason), else gathered into IDP 0; returns the
    output intrinsic's (n, c)"""
    spread = Co == ol["n"] * ol["cpi"] and (ol["n"] == 4 or ol["cpi"] == 2)
    if spread and ol["n"] > 1:
        idps, h, w, cs = _sram_dma(ol["n"], ol["pp"] * ol["Hp"], 2 * ol["Wp"])
        dmas.append((g, 0, _blob(v_im, 0, 1, idps, h, w), _blob(0, 0, 1, idps, h, w, cs=cs)))
        return ol["n"], ol["cpi"]
    if ol["n"] == 1:
        sz = cdiv(Co, 2) * ol["plane"]
        dmas.append((g, 0, _blob(v_im, 0, 1, 1, sz // 64, 64), _blob(0, 0, 1, 1, sz // 64, 64)))
        return (1, ol["cpi"]) if spread else (1, Co)
    if ol["pp"] == 1:
        # V: one plane per IDP gathered: NPUC's IR reads plane k from DRAM offset -k*plane (its 'STRIDE' pattern):
        # DRAM c_stride -2*plane. G: a bug, the planes after the first are read from before the buffer.
        h, w = ol["Hp"], 2 * ol["Wp"]
        dmas.append((g, 0, _blob(v_im, 0, 1, ol["n"], h, w, cs=0 if fix_npuc else -2 * ol["plane"]),
                     _blob(0, 0, 1, ol["n"], h, w)))
        return 1, Co
    sz = cdiv(Co, 2) * ol["plane"]
    dmas.append((g, 0, _blob(v_im, 0, 1, 1, sz // 64, 64), _blob(0, 0, 1, 1, sz // 64, 64)))
    return 1, Co


def _ps_size(shp, L):
    """V (stream.py, cmdq.md 5): a conv whose weights come in sub-batches accumulates in a PARTIAL_SUM chunk of
    2 x 7 planes over its MTC input (padded, or the space-to-depth input for stride 2), 896 B per cell"""
    C, H, W = shp["in"]
    pt, pb, pl, pr = L["pads"]
    h, w = H + pt + pb, W + pl + pr
    if tuple(L["stride"]) != (1, 1):
        h, w = cdiv(h, L["stride"][0]), cdiv(w, L["stride"][1])
    return 2 * 7 * (rup(h, 4) // 4) * (rup(w, 8) // 8) * 64


def _ring_chunk(per, cout, r, passes):
    """the WEIGHT chunk of R slots and the step between slots (V: ws-272-248k5, r3-046, r3-095, r3-103 and the ringc
    sweep): NPUC sizes a ring slot from the bytes per output channel, e = 16 x ceil(per / Cout) (per = the conv's
    weight bytes per IDP), which is the load size when Cout is a multiple of 16 and a little more otherwise; the
    chunk holds R x e bytes and slot k starts at k x round_up(e, 64) (so the last slot may pass the chunk's end by up
    to 48 bytes: r3-095). Resident weights (R = passes): the whole block."""
    if r == passes:
        return per, per // passes
    e = 16 * cdiv(per, cout)
    return r * e, rup(e, 64)


def _ring(passes, slot, fm):
    """V (stream.py's rule, cmdq.md 7b; c256, r_c256h24, r_c320h24, r_c512h16, r_c512h24): the WEIGHT chunk of a
    conv holds every pass when its node fits (feature maps fm = input, output, partial sums, plus every pass's slot,
    within 256 KiB), else a ring of R slots: R_max = (256 KiB - fm) // slot, R = ceil(passes / ceil(passes / R_max)).
    None when not even one slot fits (NPUC tiles)."""
    if fm + passes * slot <= SRAM_LIMIT:
        return passes
    rmax = (SRAM_LIMIT - fm) // slot
    if rmax < 1:
        return None
    return cdiv(passes, cdiv(passes, rmax))


def _single_block(layers, shapes, lis, wsize, v_src, v_dst, v_w, g0, dmas, fix_npuc, nsub=None, best=None,
                  spill_vec=None):
    """a run of layers NPUC keeps in one tile (one group per layer from g0, feature maps forwarded in SRAM): its SRAM
    chunks and DMAs (the input from DRAM vector v_src, the weights, the last output stored to v_dst). nsub: the
    weight sub-batch count per conv (weights.py), which adds a PARTIAL_SUM chunk; a conv whose weights do not fit
    beside its feature maps streams them through a ring (_ring; best: NPUC's own choice per layer, npuc_policy, which
    rules when given). Returns the placed chunks, the peak and the ops (the peak is None when a conv cannot hold even
    one pass: NPUC tiles)."""
    nsub = nsub or {}
    C0, H0, W0 = shapes[lis[0]]["in"]
    ops = []
    for li in lis:
        ops += _layer_ops(layers[li], li, shapes[li])
    for k, o in enumerate(ops):
        o["order"] = k + 1
    # SRAM chunks with live ranges (V: the LA dump 'order#[a ~ b]')
    inl = cell(C0, H0, W0)
    chunks = [[{"kind": "INPUT", "size": inl["per_idp"], "live": (1, 1), "layer": -1}]]
    ring = {}
    prev = inl["per_idp"]
    for li in lis:
        L = layers[li]
        grp = []
        lops = [o for o in ops if o["layer"] == li]
        for o in lops:
            nxt = o["order"] + 1   # its consumer is the next op (the next layer's first op or the unflatten)
            grp.append({"kind": o["op"], "size": o["size"], "live": (o["order"], nxt), "layer": li, "opr": o})
        if L["kind"] == "conv":
            nin = cell(*shapes[li]["in"])["n"]
            if wsize[li] % nin:
                raise ValueError("weight block %d B does not split over %d IDPs" % (wsize[li], nin))
            per = wsize[li] // nin
            passes = cdiv(L["cout"], 16)
            ps = _ps_size(shapes[li], L) if (nsub.get(li) or 1) > 1 else 0
            cin = lops[-2]["size"] if len(lops) > 1 else prev      # the MTC's input: REFORMAT, PAD or the input
            r = _ring(passes, per // passes, cin + lops[-1]["size"] + ps)
            short = r is not None and r < passes and r <= 4
            if best is not None:
                # V (NPUC's policy, frame.md 6.10; tiling2 a02: a 6-slot ring with an extended lifetime where all
                # 12 passes would fit with a short one): its ring and the ring's lifetime
                r = best[li]["nbuf"] or passes
                short = not best[li]["extended"]
            if r is None:
                return None, None, ops
            ring[li] = r
            wsz, wstep = _ring_chunk(per, L["cout"], r, passes)
            # V (wlive sweep): a ring of at most 4 slots lives on the conv only (R 1..4: 8 compiles), else from the
            # layer's first op like resident weights (R 5..16): NPUC's short and extended weight lifetimes
            w0 = lops[-1]["order"] if short else lops[0]["order"]
            grp.append({"kind": "WEIGHT", "size": wsz, "live": (w0, lops[-1]["order"]), "layer": li, "ring": r,
                        "step": wstep})
            if ps:
                grp.append({"kind": "PARTIAL_SUM", "size": ps, "live": (lops[-1]["order"], lops[-1]["order"]),
                            "layer": li, "tail": True})
        prev = lops[-1]["size"]
        chunks.append(grp)
    # group by group; when no order fits a layer's chunks, NPUC "relies on trim and spill" (V: its log, ws-191-256x2,
    # r3-076, r3-077, r3-096, r3-111): the layer's input (the previous layer's output chunk, "active IFM") is copied
    # by a TRIM (an NFU copy, its own NCP group without DMAs or intrinsic) into a new chunk at the start of the SRAM
    # ("reload IFM into start/end of available address space": 0 in 18 compiles), or at its end (0x40000 - size) when
    # the copy at 0 would overlap the input itself (V: final-chains f014, input at 23040 of 27648 B, copy at 234496;
    # G for the exact test), live at the layer's first op only; the input's live range is cut before it; then the
    # layer's chunks bigger first again, equal sizes in creation order (PAD before CONV: ws-191-256x2, r3-096)
    placed = []
    for k, grp in enumerate(chunks):
        res = _first_fit([grp], placed, cap=SRAM_LIMIT)
        if res is None and k >= 2:
            # NPUC's trim and spill (frame.md 6.7, 6.12; the stream part's model of its GlobalMemoryAllocator): a copy
            # of the input (a TRIM chunk, its own group before the layer's), or a spill (the previous layer stores its
            # output, this layer loads it into a chunk of its own, placed after its other chunks)
            res = _chain_trim_spill(grp, placed, lis[k - 1], lis[k - 2])
        if res is None:
            res = _first_fit([grp], placed, cap=SRAM_LIMIT, overflow=True)
        placed += res
    peak = max(c["off"] + c["size"] for c in placed)
    out_chunk = {c["layer"]: c for c in placed if c["kind"] in ("CONV", "POOL")}
    w_chunk = {c["layer"]: c for c in placed if c["kind"] == "WEIGHT"}
    in_chunk = placed[0]
    copies = set(c["layer"] for c in placed if c.get("copy"))
    grp_of, g = {}, g0
    for li in lis:
        g += li in copies               # the copy's group comes before its layer's
        grp_of[li] = g
        g += 1

    def load_weights(li, ks=None):
        L = layers[li]
        nin = cell(*shapes[li]["in"])["n"]
        per = wsize[li] // nin
        nl = cdiv(L["cout"], 16)        # V: one kernel load per 16 output channels
        part = per // nl
        for k in (range(nl) if ks is None else ks):
            # V: load k goes to slot k mod R of the WEIGHT chunk (a ring when R < the loads; stream.py's io)
            _weight_load(dmas, grp_of[li], v_w[li], per, part, k, nin, nl,
                         w_chunk[li]["off"] + (k % ring[li]) * w_chunk[li]["step"])

    # the input (V: split per IDP plane count when a stride-1 conv without padding reads it directly; one blob
    # when a PAD, REFORMAT or POOL op does, except that a PAD or POOL loads plane by plane and NPUC's DMACluster
    # cuts n*pp >= 64 (CODEGEN_MINIMAL_LOADS_NUM_FOR_BATCH_CLUSTERING) such loads into 4 (CODEGEN_CLUSTER_NUM_BATCHES)
    # clusters of ceil(pp/4) planes per IDP, the last one the rest), then every conv's weights in its own group
    def in_load(g, vsrc, lay, op0, off):
        direct = op0.get("direct", False)
        if op0["op"] in ("PAD", "POOL") and lay["n"] > 1 and lay["n"] * lay["pp"] >= 64:
            per = cdiv(lay["pp"], 4)
            k0 = 0
            while k0 < lay["pp"]:
                k1 = min(lay["pp"], k0 + per)
                sz = (k1 - k0) * lay["plane"]
                idps, h, w, cs = _sram_dma(lay["n"], sz // (2 * lay["Wp"]), 2 * lay["Wp"])
                dmas.append((g, 0, _blob(vsrc, k0 * lay["plane"], 1, idps, h, w, cs=lay["per_idp"] - sz),
                             _blob(0, off + k0 * lay["plane"], 1, idps, h, w, cs=cs)))
                k0 = k1
        else:
            for vec_, idps, h, w, cs, doff in _spread_dmas(lay, direct):
                dmas.append((g, 0, _blob(vsrc, doff, 1, idps, h, w), _blob(vec_, off, 1, idps, h, w, cs=cs)))

    def out_store(li, vdst):
        # V: when the layer is a conv on a wider input (internal_shapes), the DRAM side of its store steps planes by
        # the true width's plane: DRAM c_stride = pp * (true plane - internal plane) (negative; seen with one plane
        # per IDP; G: NPUC's planes then overlap in DRAM). A pool that widened its own output stores with c_stride 0.
        Co_, Ho_, Wo_ = layers[li]["out_chw"]
        _, Hi_, Wi_ = shapes[li]["out"]
        ol_ = cell(Co_, Hi_, Wi_)
        dcs = ol_["pp"] * (cell(Co_, Ho_, Wo_)["plane"] - ol_["plane"]) if Wi_ != Wo_ and layers[li]["kind"] == "conv" \
            else 0
        if fix_npuc:
            dcs = 0
        for vec_, idps, h, w, cs, doff in _spread_dmas(ol_, True, layers[li]["kind"] == "pool"):
            dmas.append((grp_of[li], 1, _blob(vec_, out_chunk[li]["off"], 1, idps, h, w, cs=cs),
                         _blob(vdst, doff, 1, idps, h, w, cs=dcs if idps > 1 else 0)))

    in_load(g0, v_src, inl, ops[0], in_chunk["off"])
    # a spill (frame.md 6.12): the previous layer stores its output (its own IM vector, spill_vec), the reader loads
    # it into its chunk, as a block's input (V: mobilenet-v1-100-224, conv3's 56 x 56 x 128 output before dw3)
    spills = {c["layer"]: c for c in placed if c.get("spill")}
    last = lis[-1]
    oc = out_chunk[last]
    Co, Ho, Wo = layers[last]["out_chw"]
    _, Hi, Wi = shapes[last]["out"]
    ol = cell(Co, Hi, Wi)
    # V (rc-167-488/500/504/510, rc-200-500; G for the condition): a last conv whose weights stream through a ring,
    # with Cout not a multiple of 16 and no sub-batches, stores its output pass by pass, each pass's store right
    # after that pass's weight load (16 output channels = 2 planes per IDP)
    Ll = layers[last]
    passwise = Ll["kind"] == "conv" and ring.get(last, 0) < cdiv(Ll["cout"], 16) and Ll["cout"] % 16 and \
        nsub.get(last) == 1 and _cmdq_estimate(Ll, shapes[last]) <= CMDQ_LIMIT    # (cut: chunk by chunk, 6.9)
    for k, li in enumerate(lis):
        if li in spills:
            if spill_vec is not None:
                out_store(lis[k - 1], spill_vec[li])
                in_load(grp_of[li], spill_vec[li], cell(*shapes[li]["in"]),
                        next(o for o in ops if o["layer"] == li), spills[li]["off"])
        if layers[li]["kind"] == "conv" and not (passwise and li == last):
            load_weights(li)
    if passwise:
        for p_ in range(cdiv(Ll["cout"], 16)):
            load_weights(last, [p_])
            _pass_store(dmas, grp_of[last], ol, oc["off"], v_dst, p_)
        return placed, peak, ops
    # the last layer's output, stored in its group (always split)
    # V: when the last layer is a conv on a wider input (internal_shapes), the DRAM side of its store steps planes by
    # the true width's plane: DRAM c_stride = pp * (true plane - internal plane) (negative; seen with one plane per
    # IDP; G: NPUC's planes then overlap in DRAM). A pool that widened its own output stores with c_stride 0.
    out_store(last, v_dst)
    return placed, peak, ops


def _spill_vectors(layers, shapes, lis, wsize, nsub, best, fix_npuc, vec):
    """the IM vectors of a single block's spilled tensors (frame.md 6.12): a dry run of its allocation finds the
    spills; each spilled producer's output gets a vector by the IM rule, in order. {reader layer: vector}"""
    placed, _, _ = _single_block(layers, shapes, lis, wsize, 0, 0, {li: 0 for li in lis}, 0, [], fix_npuc, nsub,
                                 best)
    out = {}
    for c in placed:
        if c.get("spill"):
            p_ = lis[lis.index(c["layer"]) - 1]
            c_, h_, w_ = shapes[p_]["out"]
            Lt = layers[p_]["out_chw"]
            im = _im_vector(c_, h_, w_, Lt[0] * Lt[1] * Lt[2])
            out[c["layer"]] = vec(MV_IM, im["av"], (im["w"], im["h"], im["c"]))
    return out


def _chain_trim_spill(grp, placed, li, prev):
    """stream._trim_spill on one layer's chunks of a chain's single block: placed (dicts, in placement order) as the
    allocator's state, the previous layer's output read by this layer's first op. Returns the layer's chunks, with
    a TRIM chunk ("copy") or a spilled reload ("spill": an INPUT chunk this layer loads) appended; the input's live
    range is cut as NPUC cuts it."""
    wrap = []
    for d in placed:
        c = stream.Chunk(None, d["kind"], d["size"], list(d["live"]), None, d.get("tail", False))
        c.off, c.layer, c.src = d["off"], d.get("layer"), d
        if d.get("layer") == prev and d["kind"] in ("CONV", "POOL"):
            c.tensor, c.readers = prev, [(li, min(x["live"][0] for x in grp))]
        wrap.append(c)
    pos = {}
    stream._register(pos, wrap)
    mine = []
    for j, d in enumerate(grp):
        c = stream.Chunk(j, d["kind"], d["size"], list(d["live"]), (2, li), d.get("tail", False))
        c.layer, c.src = li, d
        mine.append(c)
    ts = {}
    try:
        got = stream._trim_spill(mine, wrap, pos, SRAM_LIMIT, ts)
    except stream.Unsupported as e:
        raise NotImplementedError(str(e))
    for c in wrap:                          # the input's live range as NPUC cut it
        c.src["live"] = tuple(c.live)
    out = []
    for c in got:
        if c.cid is not None and getattr(c, "src", None) is not None and c.src in grp:
            out.append(dict(c.src, off=c.off))
    for X, T in ts.get("trims", []):
        out.append({"kind": "TRIM", "size": T.size, "live": tuple(T.live), "layer": li, "copy": True, "src": X.off,
                    "off": T.off})
    for t, r, n in ts.get("spills", []):
        out.append({"kind": "INPUT", "size": n.size, "live": tuple(n.live), "layer": li, "spill": True, "off": n.off})
    return out


def _weight_load(dmas, g, v_w, per, part, k, nin, nl, sram_off):
    """the load of pass k of a conv's weights (per bytes per IDP, part per pass) into SRAM offset sram_off: one DMA over
    the IDPs (DRAM c_stride per - part), or, when the kernel passes 4 MiB in all (4 x per; stream.py's rule),
    one per IDP through its own SRAM vector (V: k3c64-ci512, s2c64 cut; tiling3 ra002, ra067 rings: source c x per +
    k x part, part / 64 rows of 64 B)"""
    if 4 * per > 4 << 20 and nin > 1:
        for c in range(nin):
            dmas.append((g, 0, _blob(v_w, c * per + k * part, 1, 1, part // 64, 64),
                         _blob(c, sram_off, 1, 1, part // 64, 64)))
        return
    idps, h, w, cs = _sram_dma(nin, part // 64, 64)
    dmas.append((g, 0, _blob(v_w, k * part, 1, idps, h, w, cs=(per - part) if nl > 1 and nin > 1 else 0),
                 _blob(0, sram_off, 1, idps, h, w, cs=cs)))


def _pass_store(dmas, g, lay, sram_off, v_dst, p):
    """the store of pass p of a conv output (planes 2p, 2p+1 of each IDP; V: rc-167-500): the IDPs holding as many
    of them as IDP 0 in one blob (h = planes x Hp rows of 2Wp, SRAM c_stride 0x40000 - h*w, DRAM c_stride = the
    IDP's block - h*w), an IDP holding fewer (the last one) on its own through its SRAM vector, 64 B wide"""
    k = [max(0, min(2, cdiv(min(lay["cpi"], lay["C"] - i * lay["cpi"]), 2) - 2 * p)) for i in range(lay["n"])]
    full = 0
    while full < lay["n"] and k[full] == k[0]:
        full += 1
    off = 2 * p * lay["plane"]
    if k[0]:
        h, w = k[0] * lay["Hp"], 2 * lay["Wp"]
        dmas.append((g, 1, _blob(0, sram_off + off, 1, full, h, w, cs=IDP_STRIDE - h * w if full > 1 else 0),
                     _blob(v_dst, off, 1, full, h, w, cs=lay["per_idp"] - h * w if full > 1 else 0)))
    for i in range(full, lay["n"]):
        if k[i]:
            sz = k[i] * lay["plane"]
            dmas.append((g, 1, _blob(i, sram_off + off, 1, 1, sz // 64, 64),
                         _blob(v_dst, i * lay["per_idp"] + off, 1, 1, sz // 64, 64)))


def _mark_copies(groups, copies, g0, lis):
    """name the copy groups of a single block (from g0, layers lis): {"copy": {"layer", "src", "dst", "size"}}, the
    TRIM before that layer (_single_block)"""
    by = {c["layer"]: c for c in copies}
    g = g0
    for li in lis:
        if li in by:
            c = by[li]
            groups[g]["copy"] = {"layer": li, "src": c["src"], "dst": c["off"], "size": c["size"]}
            g += 1
        g += 1


def _spread_dmas(lay, split, pool=False):
    """the DMAs that move a tensor between a DRAM blob (IDP-major) and the IDP SRAMs (V): one blob over the IDPs
    (h = pp*Hp rows of 2Wp bytes, SRAM c_stride = 0x40000 - h*w); a single IDP as one blob 64 B wide; with
    'split', the IDPs holding fewer planes than the first (the last one) move on their own, through their own
    SRAM vector (OT_PIX<i>), 64 B wide. A pool's output: its last IDP, when not full, moves pp rounded down to even
    planes whatever it holds (V on 9 variants). Returns [(sram vector, idps, h, w, sram c_stride, dram offset)]."""
    counts = [min(lay["cpi"], lay["C"] - i * lay["cpi"]) for i in range(lay["n"])]
    planes = [cdiv(c, 2) for c in counts]
    if pool and planes[-1] < lay["pp"]:
        planes[-1] = lay["pp"] - lay["pp"] % 2
    if lay["n"] == 1:
        return [(0, 1, lay["per_idp"] // 64, 64, 0, 0)]
    full = sum(1 for p in planes if p == lay["pp"]) if split else lay["n"]
    idps, h, w, cs = _sram_dma(full, lay["pp"] * lay["Hp"], 2 * lay["Wp"])
    out = [(0, idps, h, w, cs, 0)]
    for i in range(full, lay["n"]):
        sz = planes[i] * lay["plane"]
        out.append((i, 1, sz // 64, 64, 0, sum(planes[:i]) * lay["plane"]))
    return out


def _io(dmas):
    """io descriptors: uid in order, batch = a run of DMAs of one direction in one group, at most 15 (frame() then
    takes the batches the streams declare), trigger bit = (uid - 1) mod 16 (V; the wrap seen at the 17th DMA)"""
    io = []
    batch, prev, run = 0, None, 0
    for k, (g, d, src, dst) in enumerate(dmas):
        if (g, d) != prev or run == 15:
            batch += 1
            prev, run = (g, d), 0
        run += 1
        io.append([k + 1, batch, g, 0, 1 << (k % 16), 2, 0, 1, 0] + src + dst)
    return io


def plan(low, weight_sizes=None, strict=True, fix_npuc=False):
    """NPUC's plan: groups, ops, SRAM chunks, vectors and DMAs (see the module docstring); plan_tiled() when a layer
    needs tiling. weight_sizes: the byte size of each conv's WEIGHT block in layer order (they set the SRAM
    allocation); by default weights.py's when it can pack them, else a dense estimate (not exact
    for compressed weights). strict=False skips the refusal of plans near 256 KiB whose partial sums weights.py cannot
    tell (for callers that only need the structure).
    fix_npuc=True writes the two DMA strides NPUC gets wrong (G until run on the NPU, docs/format/frame.md
    'NPUC defects'): the gathered read-back of one-plane-per-IDP outputs (c_stride 0, not -2*plane) and the store of
    a conv whose input was widened (DRAM c_stride 0); default False = NPUC's bytes.
    Memoised by the network's content (stream.py and this module ask for the same plan many times)."""
    if _POL_TRACE is not None:     # frame_sweep.py's trace of the policy's estimates: computed, not remembered
        return _plan(low, weight_sizes, strict, fix_npuc)
    key = (network_key(low), None if weight_sizes is None else tuple(weight_sizes), bool(strict), bool(fix_npuc),
           _IN_STREAM)
    return _PLANS.get(key, lambda: _plan(low, weight_sizes, strict, fix_npuc))


_PLANS = Memo(16)


def _plan(low, weight_sizes, strict, fix_npuc):
    if is_graph(low):
        return plan_graph(low, weight_sizes)
    layers = low["layers"]
    C0, H0, W0 = low["input"]["chw"]
    for L in layers:
        if L["kind"] == "pool" and L["stride"][0] != L["stride"][1]:
            raise NotImplementedError("%s: pooling strides %dx%d (not modelled; stream.pool_strides)"
                                      % ((L["name"],) + tuple(L["stride"])))
    convs = [i for i, L in enumerate(layers) if L["kind"] == "conv"]
    if weight_sizes is None:
        weight_sizes = _weight_sizes(low)
    if len(weight_sizes) != len(convs):
        raise ValueError("%d weight blocks for %d convs" % (len(weight_sizes), len(convs)))
    wsize = dict(zip(convs, weight_sizes))
    shapes = internal_shapes(low)
    nsub = _sub_batches(low)
    policy = npuc_policy(low, wsize)
    if policy["strata"]:        # V: NPUC's policy tiles a layer (frame.md 6.10)
        return plan_tiled(low, wsize, fix_npuc, nsub, policy)
    Co, Ho, Wo = layers[-1]["out_chw"]          # the true output (OT_FMAP)
    _, Hi, Wi = shapes[-1]["out"]               # NPUC's internal one (the SRAM, IM and DMAs)
    im = _im_vector(Co, Hi, Wi, Co * Ho * Wi)       # V (final-chains f036): a widened output's IM by its width
    av, mv = [], []

    def vec(t, size, shape=(0, 0, 0)):
        av.append([len(av), 0, 0, size])
        mv.append([t, 1, shape[0], shape[1], shape[2], 0, 0, len(av) - 1])
        return len(av) - 1

    # vectors: OT_PIX0..3 (the SRAMs), IN_FMAP, OT_FMAP, IM_FMAP (the stored output), WEIGHT per conv, IM 8x8x2 (V)
    for k in range(IDP):
        vec(MV_P0 + k, SRAM_VEC)
    inl = cell(C0, H0, W0)
    v_in = vec(MV_IN, inl["size"], (W0, H0, C0))
    v_out = vec(MV_OT, ofm_size(Co, Ho, Wo), (Wo, Ho, Co))
    lis = list(range(len(layers)))
    # a spilled tensor (frame.md 6.12) has its own IM vector, before the output's (in the order of the stores)
    spill_vec = _spill_vectors(layers, shapes, lis, wsize, nsub, policy["best"], fix_npuc, vec)
    v_im = vec(MV_IM, im["av"], (im["w"], im["h"], im["c"]))
    v_w = {li: vec(MV_WT, wsize[li]) for li in convs}
    vec(MV_IM, 138, (8, 8, 2))   # V: always present, 8x8x2 + 10
    dmas = []   # (group, direction 0 load / 1 store, src blob, dst blob)
    placed, peak, ops = _single_block(layers, shapes, lis, wsize, v_in, v_im, v_w, 0, dmas,
                                      fix_npuc, nsub, policy["best"], spill_vec)
    if peak is None or peak > SRAM_LIMIT:
        raise NotImplementedError("NPUC keeps every layer whole but our allocation needs %s B" % peak)
    if peak > SRAM_SAFE and strict and any(nsub[li] is None for li in convs):
        raise NotImplementedError("the plan needs %d B of SRAM per IDP (near 256 KiB) and weights.py cannot tell "
                                  "its partial sums" % peak)
    if _unflatten_tiles(Co, Hi, Wi, layers[-1]["kind"] == "conv") > 1:
        # V (final-chains f001, f061): the output read-back alone passes 256 KiB: NPUC splits it in rows as in a
        # tiled plan (plan_tiled keeps every layer in one single block then)
        return plan_tiled(low, wsize, fix_npuc, nsub)
    if any(_cmdq_estimate(L, shp) > CMDQ_LIMIT for L, shp in zip(layers, shapes)) and not _IN_STREAM:
        # (with a trim-and-spill copy too: its group has no DMA; V: mobilenet-v1-050-224, a copy before its 13th
        # depthwise layer and its classifier cut in three)
        copies = [c for c in placed if c.get("copy")]
        cut = [li for li, (L, shp) in enumerate(zip(layers, shapes)) if _cmdq_estimate(L, shp) > CMDQ_LIMIT]
        if copies and min(cut) < max(c["layer"] for c in copies):
            raise NotImplementedError("a copy group after a layer NPUC cuts (its group index)")
        p = _cut_plan(low, layers, shapes, placed, ops, dmas, wsize, v_in, v_im, v_w, av, mv, v_out, convs,
                      (Co, Ho, Wo, Hi, Wi), fix_npuc)
        _mark_copies(p["groups"], copies, 0, list(range(len(layers))))
        return p
    # the output group reads the stored output back (V: spread or gathered, _read_back)
    copies = [c for c in placed if c.get("copy")]
    gu = len(layers) + len(copies)
    ol = cell(Co, Hi, Wi)
    un_n, un_c = _read_back(dmas, gu, ol, Co, v_im, fix_npuc)
    io = _io(dmas)
    groups = []
    for g in range(gu + 1):
        has_io = any(x[2] == g for x in io)
        groups.append({"index": g, "kind": "output" if g == gu else ("dma" if has_io else "plain")})
    _mark_copies(groups, copies, 0, list(range(len(layers))))
    return {"layers": [L["kind"] for L in layers], "ops": ops, "chunks": placed, "peak": peak, "av": av, "mv": mv,
            "io": io, "groups": groups, "v_out": v_out, "v_w": [v_w[li] for li in convs],
            "shapes": shapes,
            "unflatten": {"n": un_n, "c": un_c, "C": Co, "H": Ho, "W": Wo, "Hi": Hi, "Wi": Wi, "Hp": ol["Hp"],
                          "Wp": ol["Wp"], "sram_off": 0, "group": gu}}


# ---- NPUC's "CMDQ seems too big" cut (docs/format/cmdq.md 7d; frame.md 6.9) ---------------------------------
CMDQ_LIMIT = 61440
_IN_STREAM = False      # set while frame asks stream.py for its DMAs (stream.Plan calls frame.plan back)


def copies_any(placed):
    """a plan with one of NPUC's trim-and-spill copies"""
    return any(c.get("copy") for c in placed)


def _cmdq_estimate(L, shp):
    """NPUC's "Heuristically predicted size of CMDQ" for a layer's group (stream.py's rule, V on 1619 blocks):
    16350 + 1020 per conv pass + 60 per padded plane slot + 102 per reformatted one + 150 per pooled one"""
    C = shp["in"][0]
    slots = cdiv(cell(C, 1, 1)["cpi"], 2)
    if L["kind"] == "conv":
        e = 16350 + 1020 * cdiv(L["cout"], 16)
        e += 60 * slots if any(L["pads"]) else 0
        e += 102 * slots if tuple(L["stride"]) != (1, 1) else 0
        return e
    if L["kind"] == "pool":
        return 16350 + 150 * slots
    return 0


def _range_store(dmas, g, lay, sram_off, v_dst, p0, p1, base=0):
    """the store of passes p0 .. p1 of a conv output (planes 2 p0 .. 2 p1 of each IDP; V: k1c64-ci64, k3c64-ci512):
    the IDPs holding as many as IDP 0 in one blob (h = planes x Hp rows of 2Wp), an IDP holding fewer on its own
    through its SRAM vector, 64 B wide. base: the tensor's offset in the DRAM vector (a graph's concat buffer)"""
    k = [max(0, min(2 * (p1 - p0), cdiv(min(lay["cpi"], lay["C"] - i * lay["cpi"]), 2) - 2 * p0))
         for i in range(lay["n"])]
    full = 0
    while full < lay["n"] and k[full] == k[0]:
        full += 1
    off = 2 * p0 * lay["plane"]
    out = []
    if k[0]:
        h, w = k[0] * lay["Hp"], 2 * lay["Wp"]
        out.append((g, 1, _blob(0, sram_off + off, 1, full, h, w, cs=IDP_STRIDE - h * w if full > 1 else 0),
                    _blob(v_dst, base + off, 1, full, h, w, cs=lay["per_idp"] - h * w if full > 1 else 0)))
    for i in range(full, lay["n"]):
        if k[i]:
            sz = k[i] * lay["plane"]
            out.append((g, 1, _blob(i, sram_off + off, 1, 1, sz // 64, 64),
                        _blob(v_dst, base + i * lay["per_idp"] + off, 1, 1, sz // 64, 64)))
    return out


def _unflatten_tiles(Co, Hi, Wi, after_conv=True):
    """row tiles of the output read-back: NPUC's policy on the output (unflatten) node (getBestProcessingParams,
    frame.md 6.10; V on the 1577 logged chain compiles: every output node's tile count and memory print). Whole when
    its input (16-rounded channels behind a conv) and its output, per IDP in cells, fit 256 KiB together; else the
    fewest tileable row counts n >= 2 whose largest (first) tile of rup(ceil(H / n), 4) rows fits twice: input and
    output both of the real channels (tiling3 tr-2nd 4 tiles, tr-320 6)"""
    Cr = rup(Co, 16) if after_conv else Co
    if _pol_blk(Cr, Hi, Wi) + _pol_blk(Co, Hi, Wi) <= SRAM_LIMIT:
        return 1
    for n in range(2, cdiv(Hi, 4) + 1):
        lens = _pol_lengths(Hi, n, 4)
        if lens is None:
            continue
        tot = 2 * _pol_blk(Co, lens[1], Wi)
        if _POL_TRACE is not None:
            _POL_TRACE.append([["out"], tot // 2, tot // 2, [], tot])
        if tot <= SRAM_LIMIT:
            return n
    raise NotImplementedError("no row tiling of the %d x %d x %d read-back" % (Co, Hi, Wi))


def _cut_plan(low, layers, shapes, placed, ops, dmas, wsize, v_in, v_im, v_w, av, mv, v_out, convs, oshape, fix_npuc):
    """a single-tile plan where NPUC cuts a layer's group in chunks (cmdq.md 7d): the groups and the DMA order, events
    and batches are stream.py's (stream.plan_dmas: per chunk its input slots, its passes' weights, its store);
    the descriptors are built here: input slot loads as the clustered loads (section 5), each pass's weight load (core
    by core, 4 per pass to the same slot, when the kernel passes 4 MiB: k3c64-ci512, s2c64), a store per chunk of its
    passes' planes (_range_store), the output read-back"""
    global _IN_STREAM
    _IN_STREAM = True
    try:
        dl = stream.plan_dmas(low)
    finally:
        _IN_STREAM = False
    Co, Ho, Wo, Hi, Wi = oshape
    C0, H0, W0 = low["input"]["chw"]
    inl = cell(C0, H0, W0)
    in_chunk = placed[0]
    names = {L["name"]: li for li, L in enumerate(layers)}
    last = len(layers) - 1
    oc = [c for c in placed if c.get("layer") == last and c["kind"] in ("CONV", "POOL")][0]
    ol = cell(Co, Hi, Wi)
    plain_in = [d for d in dmas if d[1] == 0 and d[2][0] == v_in]
    stores = [x for x in dl if x["kind"] == "store"]
    P_last = cdiv(layers[last].get("cout", Co), 16)
    rows = []
    un = (1, Co)
    k = 0
    while k < len(dl):
        x = dl[k]
        g = x["sgid"]
        if x["kind"] == "ifm":
            if x.get("slots"):
                s0, s1 = x["slots"]
                sz = (s1 - s0) * inl["plane"]
                idps, h, w, cs = _sram_dma(inl["n"], sz // (2 * inl["Wp"]), 2 * inl["Wp"])
                rows.append((g, 0, _blob(v_in, s0 * inl["plane"], 1, idps, h, w, cs=inl["per_idp"] - sz),
                             _blob(0, in_chunk["off"] + s0 * inl["plane"], 1, idps, h, w, cs=cs)))
            else:
                d = plain_in.pop(0)
                rows.append((g,) + d[1:])
        elif x["kind"] == "weight":
            li = names[x["layer"]]
            L = layers[li]
            nin = cell(*shapes[li]["in"])["n"]
            per = wsize[li] // nin
            nl = cdiv(L["cout"], 16)
            part = per // nl
            if x.get("core") is not None:
                c = x["core"]
                rows.append((g, 0, _blob(v_w[li], c * per + x["p"] * part, 1, 1, part // 64, 64),
                             _blob(c, x["sram"], 1, 1, part // 64, 64)))
            else:
                idps, h, w, cs = _sram_dma(nin, part // 64, 64)
                rows.append((g, 0, _blob(v_w[li], x["p"] * part, 1, idps, h, w,
                                         cs=(per - part) if nl > 1 and nin > 1 else 0),
                             _blob(0, x["sram"], 1, idps, h, w, cs=cs)))
        elif x["kind"] == "store":
            # the chunk's passes: from its SRAM offset to the next chunk's (or the last pass)
            run = [y for y in dl[k:] if y["kind"] == "store" and y["sgid"] == g]
            i = stores.index(x)
            nxt = next((y for y in stores[i + 1:] if y["sgid"] != g), None)
            step = 2 * ol["plane"]
            p0 = (x["sram"] - oc["off"]) // step
            p1 = (nxt["sram"] - oc["off"]) // step if nxt else P_last
            got = _range_store([], g, ol, oc["off"], v_im, p0, p1)
            rows += got[:len(run)]
            k += len(run)
            continue
        elif x["kind"] == "dram":
            un = _read_back(rows, g, ol, Co, v_im, fix_npuc)
        k += 1
    io = []
    for n_, (x, (g, d_, sb, db)) in enumerate(zip(dl, rows)):
        io.append([n_ + 1, x["bid"] or 0, g, 0, x["flag"], 2, 0, 1, 0] + sb + db)
    G = max(x["sgid"] for x in dl) + 1
    groups = [{"index": g, "kind": "output" if g == G - 1 else ("dma" if any(r[2] == g for r in io) else "plain")}
              for g in range(G)]
    return {"layers": [L["kind"] for L in layers], "ops": ops, "chunks": placed, "av": av, "mv": mv, "io": io,
            "groups": groups, "v_out": v_out, "v_w": [v_w[li] for li in convs], "shapes": shapes, "keep_bids": True,
            "cut": True,
            "unflatten": {"n": un[0], "c": un[1], "C": Co, "H": Ho, "W": Wo, "Hi": Hi, "Wi": Wi, "Hp": ol["Hp"],
                          "Wp": ol["Wp"], "sram_off": 0, "group": G - 1}}


# ---- graph networks (eltwise, concat, standalone ReLU; docs/format/cmdq.md 7c) -------------------------------
def plan_graph(low, weight_sizes=None):
    """the container of a graph network (docs/format/cmdq.md 7c; frame.md 6.8): the groups and DMAs are the
    stream part's (stream.Plan: NPUC's layer order, one group per layer but concats, several for a layer NPUC cuts
    for its CMDQ size, the unflatten last; each DMA op with its tensor and SRAM offset); this builds their descriptors
    and the vectors: IN, OT, an IM per tensor stored to DRAM (in the order of their first store), one more IM per
    concat, a WEIGHT per conv and ReLU layer and a META_DATA per eltwise in execution order, IM 8x8x2. weight_sizes:
    the sizes of the WEIGHT blocks in that vector order (weights.py's weight_blocks, META_DATA included:
    weights.eltwise_block)."""
    layers = low["layers"]
    by = {L["name"]: L for L in layers}
    src = low["input"]["name"]
    P = stream.Plan(low)
    for gi, ops in enumerate(P.groups):
        stream.emit_group(gi, ops, P, gi == len(P.groups) - 1, False)
    ins = {L["name"]: list(L["inputs"]) if "inputs" in L else [L["input"]] for L in layers}
    cons = {}
    for L in layers:
        for x in ins[L["name"]]:
            cons.setdefault(x, []).append(L["name"])
    outs = net_outputs(low)                     # several outputs: in NPUC's order (frame.md 12)
    order = exec_order(outs, ins, cons, src)
    gl = P.group_layers                         # the layer of each group, None for an output group (frame.md 6.8,
    #                                             6.9; one output: its read-back's row tiles last, 6.5; several: one
    #                                             group each, each ending its policy group, 12)
    nu = gl.count(None)                         # the output groups
    out = outs[0]
    ogroups = {}                                # output -> its output groups (several: row tiles of its read-back)
    for g_, ops_ in enumerate(P.groups):
        if gl[g_] is None:
            ogroups.setdefault(next(o_.tensor for o_ in ops_ if o_.kind == "load"), []).append(g_)
    inner = getattr(P, "graph_inner", set())    # a tiled stratum's layers after its head (6.11)
    if list(dict.fromkeys(x for x in gl if x is not None)) != \
            [n for n in getattr(P, "emit_order", order) if by[n]["kind"] != "concat" and n not in inner] or \
            (len(outs) > 1 and list(ogroups) != outs) or nu < 1:
        raise ValueError("the stream's groups do not follow the layer order")
    # the WEIGHT vectors, in the NCP's group order (NPUC's layer order for one output; frame.md 12)
    wl = [n for n in getattr(P, "emit_order", order) if by[n]["kind"] in ("conv", "relu", "eltwise")]
    if weight_sizes is None:
        # weights.py's sizes come in NPUC's layer order
        wsz = dict(zip([n for n in order if by[n]["kind"] in ("conv", "relu", "eltwise")], _weight_sizes(low)))
        weight_sizes = [wsz.get(n) for n in wl]
    if len(weight_sizes) != len(wl) or None in weight_sizes:
        raise ValueError("%d weight blocks for %d conv/ReLU/eltwise layers" % (len(weight_sizes), len(wl)))
    wsz = dict(zip(wl, weight_sizes))
    chw = {src: tuple(low["input"]["chw"])}
    for L in layers:
        chw[L["name"]] = tuple(L["out_chw"])
    av, mv = [], []

    def vec(t, size, shape=(0, 0, 0)):
        av.append([len(av), 0, 0, size])
        mv.append([t, 1, shape[0], shape[1], shape[2], 0, 0, len(av) - 1])
        return len(av) - 1

    for k in range(IDP):
        vec(MV_P0 + k, SRAM_VEC)
    C0, H0, W0 = chw[src]
    v_in = vec(MV_IN, cell(C0, H0, W0)["size"], (W0, H0, C0))

    def ot_vec(o):
        Co, Ho, Wo = chw[o]
        return vec(MV_OT, ofm_size(Co, Ho, Wo), (Wo, Ho, Co))
    v_outs = {outs[0]: ot_vec(outs[0])}
    dmas = [(op, i, g) for g, ops in enumerate(P.groups) for op in ops if op.kind in ("load", "store")
            for i in range(len(op.flags) if op.kind == "store" else 1)]
    vim = {}
    # an IM per stored tensor, in the order of its first store in NPUC's layer order (= the NCP's group order for one
    # output; several outputs may reorder the groups, not these: V outputs-random m004)
    xi = {n: j for j, n in enumerate(order)}
    sts = {}
    for k_, (op, i, g) in enumerate(dmas):
        if op.kind == "store":
            pos = (xi.get(gl[g], len(order)), k_)
            sts[op.tensor] = min(sts.get(op.tensor, pos), pos)
    oint = getattr(P, "out_internal", {})      # the outputs' NPUC shapes (a widened pool's one more column)
    for t in sorted(sts, key=sts.get):
        c_, h_, w_ = chw[t]
        hi, wi = oint.get(t, chw[t])[1:]
        im = _im_vector(c_, hi, wi, c_ * h_ * wi)   # an output's by its NPUC width, as a chain's (f036)
        vim[t] = vec(MV_IM, im["av"], (im["w"], im["h"], im["c"]))
    # V (the outputs sweep): the first output's OT_FMAP in its usual place, the others' after the stored tensors' IMs
    for o in outs[1:]:
        v_outs[o] = ot_vec(o)
    for n in order:
        if by[n]["kind"] == "concat":
            # V (cat*, g005, g007, g019): every concat also has a vector of its own, by the IM rule on its real
            # channel count (not rounded to 16)
            c_, h_, w_ = chw[n]
            im = _im_vector(c_, h_, w_, c_ * h_ * w_, False)
            vec(MV_IM, im["av"], (im["w"], im["h"], im["c"]))
    wvec = {n: vec(MV_WT, wsz[n]) for n in wl}
    vec(MV_IM, 138, (8, 8, 2))

    def tvec(t):
        return v_in if t == src else vim[t]

    def coff(t, kn):
        # a producer's place in a concat's DRAM buffer: after the earlier inputs' channels, each a half plane of the
        # concat's cells (V: cat2, g030; uneven splits 1+7, 3+5, 5+3, 20+12 of weights.py's cat sweep)
        o = 0
        for x in ins[kn]:
            if x == t:
                return o
            o += chw[x][0] * cell(*chw[x])["plane"] // 2
        raise ValueError("%s is not an input of %s" % (t, kn))

    rows = []           # one (group, direction, src blob, dst blob) per DMA
    un = {}             # output -> the output intrinsic's (n, c)
    k = 0
    while k < len(dmas):
        op, i, g = dmas[k]
        what = op.what if op.kind == "load" else "store"
        if what == "weight":
            nin = cell(*chw[ins[op.layer][0]])["n"]
            per = wsz[op.layer] // nin
            nl = cdiv(chw[op.layer][0], 16)
            part = per // nl
            if getattr(op, "core", None) is not None:       # a kernel over 4 MiB: core by core (section 6.9)
                rows.append((g, 0, _blob(wvec[op.layer], op.core * per + op.p * part, 1, 1, part // 64, 64),
                             _blob(op.core, op.sram, 1, 1, part // 64, 64)))
            else:
                idps, h, w, cs = _sram_dma(nin, part // 64, 64)
                rows.append((g, 0, _blob(wvec[op.layer], op.p * part, 1, idps, h, w,
                                         cs=(per - part) if nl > 1 and nin > 1 else 0),
                             _blob(0, op.sram, 1, idps, h, w, cs=cs)))
            k += 1
            continue
        if what == "eltwise_meta":
            n = gl[g]
            ncore = cell(*chw[n])["n"]
            # V: h = the block's whole rows over the cores: 0 for 32-byte blocks on several cores (res, rc12), 1 on
            # one (rc2), 1 for 64-byte blocks (res40)
            h, w = cdiv(wsz[n], 64) // ncore, 64
            rows.append((g, 0, _blob(wvec[n], 0, 1, ncore, h, w),
                         _blob(0, op.sram, 1, ncore, h, w, cs=IDP_STRIDE - h * w if ncore > 1 else 0)))
            k += 1
            continue
        if what == "dram":
            o = op.tensor if len(outs) > 1 else out
            tiles = ogroups[o]
            c_, h_, w_ = oint.get(o, chw[o])
            if len(tiles) == 1:
                un[o] = _read_back(rows, g, cell(chw[o][0], h_, w_), chw[o][0], tvec(o), False)
                k += 1
                continue
            # a row tile of a tiled read-back (6.5, plan_tiled's): its rows per IDP from the output's IM
            uc, uh, uw = oint.get(o, P.out_chw)
            u0, u1 = _tile_ranges(uh, len(tiles), 4)[tiles.index(g)]
            got = []
            _tile_dma(got, cell(uc, uh, uw), (u0, u1, 0, uw - 1), tvec(o), 0, True, g, ns0=True)
            j = k + 1
            while j < len(dmas) and dmas[j][2] == g and getattr(dmas[j][0], "what", None) == "dram":
                j += 1
            if len(got) != j - k:
                raise NotImplementedError("a read-back tile's %d DMAs, %d in the stream" % (len(got), j - k))
            rows += got
            k = j
            continue
        if getattr(op, "tile", None) is not None:
            # a tile of a stratum NPUC tiles inside the graph (frame.md 6.11): its input region loaded from the DRAM
            # tensor, or its output region stored there (a concat's buffer at the producer's offset), as a chain's
            # tiles (_tile_dma)
            got = []
            r0, r1, c0, c1 = op.region
            if op.kind == "load":
                j = k + 1
                while j < len(dmas) and dmas[j][2] == g and dmas[j][0].kind == "load" and \
                        getattr(dmas[j][0], "tile", None) == op.tile and getattr(dmas[j][0], "what", None) == "ifm":
                    j += 1
                lay = cell(*chw[op.tensor])
                _tile_dma(got, lay, (r0, r1 - 1, c0, c1 - 1), tvec(op.tensor), op.sram, True, g,
                          ns0=getattr(op, "ns0", False),
                          parts=op.slots[1] - op.slots[0] if getattr(op, "slots", None) else None)
            else:
                j = k + len(op.flags)
                prod = getattr(op, "prod", gl[g])
                t = op.tensor
                base = coff(prod, t) if by.get(t, {}).get("kind") == "concat" and t != prod else 0
                _tile_dma(got, cell(*chw[prod]), (r0, r1 - 1, c0, c1 - 1), vim[t], op.sram, False, g,
                          pool=getattr(op, "pool", False), base=base)
            if len(got) != j - k:
                raise NotImplementedError("a tile's %d DMAs, %d in the stream" % (len(got), j - k))
            rows += got
            k = j
            continue
        # a tensor moved between DRAM and SRAM: this DMA and the next ones of the same op / tensor in this group
        j = k + 1
        while j < len(dmas) and dmas[j][2] == g and dmas[j][0].kind == op.kind and \
                getattr(dmas[j][0], "what", None) == getattr(op, "what", None) and \
                getattr(dmas[j][0], "tensor", None) == op.tensor and (op.kind == "load" or dmas[j][0] is op):
            j += 1
        run = dmas[k:j]
        t = op.tensor
        if op.kind == "store" and hasattr(op, "srams"):
            # a tensor holding an SRAM concat's segments (a pool of it; cmdq.md 7c): each segment is stored as a
            # tensor of its own after the earlier ones, one DMA per core and segment (64 B wide), or per core with one
            # plane per segment when they are alike (V: g141, g169)
            prod = gl[g]
            kn = prod
            while by[kn]["kind"] != "concat":
                kn = ins[kn][0]
            segs = [stream.Dist(chw[x][0]).ppc for x in ins[kn]]
            lay = cell(*chw[prod])
            ncore = len(op.srams) // len(segs) if len(set(op.srams)) > 1 else len(op.srams)
            bases = [sum(ncore * segs[l] * lay["plane"] for l in range(sg)) for sg in range(len(segs))]
            for q, (op_, i_, _) in enumerate(run):
                if len(set(op.srams)) > 1:
                    core, sg = divmod(q, len(segs))
                    sz = segs[sg] * lay["plane"]
                    rows.append((g, 1, _blob(core, op_.srams[i_], 1, 1, sz // 64, 64),
                                 _blob(vim[t], bases[sg] + core * sz, 1, 1, sz // 64, 64)))
                else:
                    # SRAM n_stride 0x40000 - segments x h x w, 0 from SRAM offset 0 (as the tiles' and clustered
                    # loads', 6.5, 6.11; V: g141, g169 at 0, outputs-unseen m1015 at 3328)
                    h, w = segs[0] * lay["Hp"], 2 * lay["Wp"]
                    sr = op_.srams[i_]
                    rows.append((g, 1, _blob(q, sr, 1, len(segs), h, w, ns=IDP_STRIDE - len(segs) * h * w if sr else 0),
                                 _blob(vim[t], q * h * w, 1, len(segs), h, w, cs=(ncore - 1) * h * w)))
        elif op.kind == "store":
            prod = gl[g]
            lay = cell(*chw[prod])
            base = coff(prod, t) if by.get(t, {}).get("kind") == "concat" and t != prod else 0
            if hasattr(op, "passes"):
                # the planes of the passes of a chunk of a layer NPUC cuts, as in a chain (section 6.9)
                p0, p1 = op.passes
                got = _range_store([], g, lay, op.sram - 2 * p0 * lay["plane"], vim[t], p0, p1, base)
                if len(got) != len(run):
                    raise NotImplementedError("a store of passes %d..%d in %d DMAs, %d in the stream"
                                              % (p0, p1, len(got), len(run)))
                rows += got
            else:
                parts = _spread_dmas(lay, True, by[prod]["kind"] == "pool")
                for q, (op_, i_, _) in enumerate(run):
                    vec_, idps, h, w, cs, doff = parts[min(q, len(parts) - 1)]
                    sram = op_.srams[i_] if hasattr(op_, "srams") else op_.sram
                    rows.append((g, 1, _blob(vec_, sram, 1, idps, h, w, cs=cs),
                                 _blob(vim[t], base + doff, 1, idps, h, w)))
        elif hasattr(op, "slots"):
            # an input read slot by slot (its first op an NFU one, 16 plane slots per core or more; or a cut layer's
            # chunks): each load its slots of every IDP, as in a chain (section 6.9)
            lay = cell(*chw[t])
            for op_, _, _ in run:
                s0, s1 = op_.slots
                sz = (s1 - s0) * lay["plane"]
                idps, h, w, cs = _sram_dma(lay["n"], sz // (2 * lay["Wp"]), 2 * lay["Wp"])
                rows.append((g, 0, _blob(tvec(t), s0 * lay["plane"], 1, idps, h, w, cs=lay["per_idp"] - sz),
                             _blob(0, op_.sram, 1, idps, h, w, cs=cs)))
        else:
            lay = cell(*chw[t])
            parts = _spread_dmas(lay, len(run) > 1)
            for q, (op_, i_, _) in enumerate(run):
                vec_, idps, h, w, cs, doff = parts[min(q, len(parts) - 1)]
                rows.append((g, 0, _blob(tvec(t), doff, 1, idps, h, w), _blob(vec_, op_.sram, 1, idps, h, w, cs=cs)))
        k = j
    io = []
    for n_, ((op, i, _), (g, d_, sb, db)) in enumerate(zip(dmas, rows)):
        flag = op.flags[i] if op.kind == "store" else op.flag
        io.append([n_ + 1, getattr(op, "bid", 0) or 0, g, 0, flag, 2, 0, 1, 0] + sb + db)
    G = len(P.groups)
    groups = [{"index": g, "kind": "output" if gl[g] is None else ("dma" if any(x[2] == g for x in io) else "plain")}
              for g in range(G)]
    if len(outs) > 1:
        # one output group per output (or per row tile of its read-back), in NPUC's order, each writing the output's
        # own OT_FMAP (frame.md 12)
        ufs = []
        for o in outs:
            Co, Ho, Wo = chw[o]
            _, Hi, Wi = oint.get(o, chw[o])
            ol = cell(Co, Hi, Wi)
            tiles = ogroups[o]
            if len(tiles) == 1:
                ufs.append({"n": un[o][0], "c": un[o][1], "C": Co, "H": Ho, "W": Wo, "Hi": Hi, "Wi": Wi,
                            "Hp": ol["Hp"], "Wp": ol["Wp"], "sram_off": 0, "group": tiles[0], "v_out": v_outs[o],
                            "layer": o})
                continue
            for u, (u0, u1) in enumerate(_tile_ranges(Hi, len(tiles), 4)):
                r_ = u1 - u0 + 1
                ufs.append({"n": ol["n"], "c": ol["cpi"], "C": Co, "H": r_, "W": Wo, "Hi": r_, "Wi": Wi,
                            "Hp": rup(r_, 4), "Wp": ol["Wp"], "sram_off": 0, "group": tiles[u], "dst_off": u0 * Wo,
                            "dst_cs": (Ho - r_) * Wo, "v_out": v_outs[o], "layer": o})
        return {"graph": True, "keep_bids": True, "av": av, "mv": mv, "io": io, "groups": groups,
                "v_out": v_outs[out], "v_w": [wvec[n] for n in wl], "wlayers": wl,
                "chunks": [], "layers": [L["kind"] for L in layers], "unflatten": ufs[0], "outputs": ufs}
    Co, Ho, Wo = chw[out]
    _, Hi, Wi = oint.get(out, chw[out])         # a widened pool's one more column (as a chain's)
    ol = cell(Co, Hi, Wi)
    outs = [{"n": un[out][0], "c": un[out][1], "C": Co, "H": Ho, "W": Wo, "Hi": Hi, "Wi": Wi, "Hp": ol["Hp"],
             "Wp": ol["Wp"], "sram_off": 0, "group": G - 1}] if nu == 1 else []
    if nu > 1:
        uc, uh, uw = P.out_chw
        lu = cell(uc, uh, uw)
        for u, (u0, u1) in enumerate(_tile_ranges(uh, nu, 4)):
            r_ = u1 - u0 + 1
            outs.append({"n": lu["n"], "c": lu["cpi"], "C": Co, "H": r_, "W": Wo, "Hi": r_, "Wi": uw, "Hp": rup(r_, 4),
                         "Wp": lu["Wp"], "sram_off": 0, "group": G - nu + u, "dst_off": u0 * Wo,
                         "dst_cs": (Ho - r_) * Wo})
    return {"graph": True, "keep_bids": True, "av": av, "mv": mv, "io": io, "groups": groups, "v_out": v_outs[out],
            "v_w": [wvec[n] for n in wl], "wlayers": wl,
            "chunks": [], "layers": [L["kind"] for L in layers], "unflatten": outs[0], "outputs": outs}


# ---- tiled plans (NPUC's stratum tiling) ------------------------------------------------------------------------------
def _back(L, shp, o0, o1, axis):
    """the input range a layer reads for output range [o0, o1] along an axis (0 rows, 1 columns), clipped, and the
    padding the tile gets before/after (V: NPUC's TILE start/end and PAD_PARAM prints): a tile whose input reaches
    the tensor's edge gets the layer's whole pad on that side, even one that needs none of it (V: tiling2 b01, b02,
    a 5x5 conv's next-to-last 4-row tile ending on the last input row is padded 2 rows below)"""
    k = L["k"][axis]
    s = L["stride"][axis]
    pb, pa = (L["pads"][0], L["pads"][1]) if axis == 0 else (L["pads"][2], L["pads"][3])
    D = shp["in"][1 + axis]
    i0, i1 = o0 * s - pb, o1 * s + k - 1 - pb
    return max(i0, 0), min(i1, D - 1), pb if i0 <= 0 else 0, pa if i1 >= D - 1 else 0


def _tile_ranges(D, n, unit):
    """NPUC's tiles along an axis of size D cut into n: size rup(ceil(D/n), unit) (unit = cell: 4 rows, 8 columns),
    the last one the rest (V: 300 -> 152 + 148, 512 -> 172 * 2 + 168, 128 / 3 -> 44 + 44 + 40)"""
    t = rup(cdiv(D, n), unit)
    return [(a, min(D, a + t) - 1) for a in range(0, D, t)]


def _tile_geometry(layers, shapes, lis, rt, ct):
    """per tile: per layer of the stratum (back from the last) its output and input ranges and pads; the first layer's
    loaded range (start aligned down to the cell, 4 rows / 8 columns: V). A tile whose output reaches the stratum
    output's last row (column) computes every layer's output to that layer's last row (column), its input to the
    tensor's end with the layer's trailing pad (V: tiling2 e03, e04, 2x2 pools on 27 x 31 behind a 3x3 conv; NPUC's
    layouts add each layer's rounding left-over to the last tile, frame.md 6.10: stream.py's tile_to_end)"""
    geo = []
    end_h, end_w = shapes[lis[-1]]["out"][1] - 1, shapes[lis[-1]]["out"][2] - 1
    for r in rt:
        for c in ct:
            per = {}
            o = (r[0], r[1], c[0], c[1])
            to_h, to_w = r[1] >= end_h, c[1] >= end_w
            for li in reversed(lis):
                L, shp = layers[li], shapes[li]
                if to_h:
                    o = (o[0], shp["out"][1] - 1, o[2], o[3])
                if to_w:
                    o = (o[0], o[1], o[2], shp["out"][2] - 1)
                i0, i1, pt, pb = _back(L, shp, o[0], o[1], 0)
                j0, j1, pl, pr = _back(L, shp, o[2], o[3], 1)
                if to_h:
                    i1, pb = shp["in"][1] - 1, L["pads"][1]
                if to_w:
                    j1, pr = shp["in"][2] - 1, L["pads"][3]
                per[li] = {"out": o, "in": (i0, i1, j0, j1), "pads": (pt, pb, pl, pr)}
                o = (i0, i1, j0, j1)
            i0, i1, j0, j1 = o
            geo.append({"layers": per, "load": (i0 // 4 * 4, i1, j0 // 8 * 8, j1), "out": (r[0], r[1], c[0], c[1])})
    return geo


def _stratum_ops(layers, shapes, lis, g, trim):
    """the ops of one tile and their per-IDP SRAM sizes (V: TRIM when some tile's input starts inside a cell, then
    per layer PAD / REFORMAT / CONV or POOL)"""
    li0 = lis[0]
    C0 = shapes[li0]["in"][0]
    l0, l1, m0, m1 = g["load"]
    i0, i1, j0, j1 = g["layers"][li0]["in"]
    ops = [{"op": "SPLIT", "size": cell(C0, l1 - l0 + 1, m1 - m0 + 1)["per_idp"], "layer": li0}]
    if trim:
        ops.append({"op": "TRIM", "size": cell(C0, i1 - i0 + 1, j1 - j0 + 1)["per_idp"], "layer": li0})
    for li in lis:
        L, t = layers[li], g["layers"][li]
        C = shapes[li]["in"][0]
        Co = shapes[li]["out"][0]
        i0, i1, j0, j1 = t["in"]
        pt, pb, pl, pr = t["pads"]
        o0, o1, q0, q1 = t["out"]
        h, w = i1 - i0 + 1 + pt + pb, j1 - j0 + 1 + pl + pr
        if L["kind"] == "conv":
            sh, sw = L["stride"]
            if any(L["pads"]):
                ops.append({"op": "PAD", "size": cell(C, h, w)["per_idp"], "layer": li})
            if (sh, sw) != (1, 1):
                r_ = cell(C, cdiv(h, sh), cdiv(w, sw))
                ops.append({"op": "REFORMAT", "size": r_["cpi"] * sh * sw // 2 * r_["plane"], "layer": li})
            ops.append({"op": "CONV", "size": cell(rup(Co, 16), o1 - o0 + 1, q1 - q0 + 1)["per_idp"], "layer": li})
        else:
            ops.append({"op": "POOL", "size": cell(Co, o1 - o0 + 1, q1 - q0 + 1)["per_idp"], "layer": li})
    return ops


def _stratum_alloc(layers, lis, ops_t, wsizes, ps=None):
    """SRAM of one tiled stratum (V: NPUC's log). First citizens: the WEIGHT chunks, the first at 0, the next at the
    top of the SRAM (G beyond two); second citizens by group: the input tile (SPLIT), then tile 0's op chunks
    biggest first (ties: created last first), a conv's PARTIAL_SUM chunk last (ps: {layer: size}; V: c512); live
    ranges over tile 0's ops (orders 2..: SPLIT [2,2], each op [o, o+1], the last [o, o]; WEIGHT [2, last];
    PARTIAL_SUM its conv's [o, o])"""
    n = len(ops_t)
    ps = ps or {}
    first, top = 0, IDP_STRIDE
    placed = []
    if n <= 2:              # one op: the weights are an ordinary chunk of the op's group (V: tiled k1 / VALID convs)
        split = dict(ops_t[0], kind="SPLIT", live=(2, 2))
        grp2 = [{"kind": ops_t[1]["op"], "size": ops_t[1]["size"], "live": (2, 2), "layer": ops_t[1]["layer"]}]
        grp2 += [{"kind": "WEIGHT", "size": sz, "live": (2, 2)} for sz in wsizes]
        grp2 += [{"kind": "PARTIAL_SUM", "size": sz, "live": (2, 2), "layer": li, "tail": True}
                 for li, sz in ps.items()]
        return _first_fit([[split], grp2], cap=SRAM_LIMIT)
    for k, sz in enumerate(wsizes):
        # V (tiling2: 19 three-conv strata; G for the rule): each at the end of the SRAM that holds less so far, the
        # bottom on a tie (the first at 0, the second at the top, a third under the second or after the first)
        if first <= IDP_STRIDE - top:
            ch = {"kind": "WEIGHT", "size": sz, "live": (2, n), "off": first}
            first += sz
        else:
            top -= sz
            ch = {"kind": "WEIGHT", "size": sz, "live": (2, n), "off": top}
        placed.append(ch)
    split = dict(ops_t[0], kind="SPLIT", live=(2, 2))
    grp2 = []
    for k, o in enumerate(ops_t[1:]):
        order = k + 2
        grp2.append({"kind": o["op"], "size": o["size"], "live": (order, order if k == n - 2 else order + 1),
                     "layer": o["layer"]})
        if o["op"] == "CONV" and o["layer"] in ps:
            grp2.append({"kind": "PARTIAL_SUM", "size": ps[o["layer"]], "live": (order, order), "layer": o["layer"],
                         "tail": True})
    rest = _first_fit([[split], grp2], placed, cap=SRAM_LIMIT)
    return None if rest is None else placed + rest


# ---- NPUC's tiling policy, read from its code (frame.md 6.10; libnpuc_template, Ghidra addresses) ---------------------
# V (code): NewPolicyMaker::computeTilesNumber walks the nodes from the output back and asks getBestProcessingParams
# (0x3adc8c) for each node's processing: whole (all weights, or a weight ring), else tiles (nh, nw); then
# recalculateTilingForStratum (0x3ae948) grows chains of nodes from the tail and tiles each chain with one layout. The
# memory each decision tests is NPUC's own estimate (PNodeOperation::estimateFullMemoryRequirements 0x34481c for a
# whole node, estimateStratumMemoryRequirements 0x3b1364 for tiles: the "Memory requirements calculated on tiled
# nodes" print), not the allocator's chunks, so its blocks differ from the plan's chunks in places (a pad block of
# H + 2 max(pt, pb) rows, a partial-sum block whenever one weight buffer passes 511 bytes, ...).
POL_NZV = 511               # V: NPUDeviceNeus::getMaxNzvInBatch, the bound past which a conv needs partial sums
POL_PSUM_CELL = 896         # V (code + 741 log prints): 2 DRUs per IDP x ceil(32 / (32 // 6)) x 64 B per cell
POL_MAX_H = 256             # V: PTraits2020::getMaxPadInputShape's height (the pad unit; no width bound)
POL_INF = 1 << 40
_POL_TRACE = None           # a list to record every estimate (frame_sweep.py policy)


def _pol_lengths(D, n, unit):
    """TilingUtils::calcBasicTileLengths (0x3b309c): n tiles of rup(ceil(D / n), unit), the last the rest, as
    (count, first, middle, last, overlap); None when that does not give exactly n non-empty tiles (isTileable)"""
    if n == 1:
        return (1, D, 0, 0, 0)
    t = rup(cdiv(D, n), unit)
    if n == 2:
        return (2, t, 0, D - t, 0) if 0 < t < D else None
    if t * (n - 2) >= D:
        return None
    rest = D - t * (n - 2)
    return (n, t, t, rest - t, 0) if rest >= t and rest > t else None


def _pol_back_axis(lens, k, s, pb, pa, D, pool):
    """KernelNodeTileBackPropagator::calcInputTilesLayout (0x3b3868) along one axis: input tile lengths k + (n - 1) s,
    the first less the leading pad and the last less the trailing one, the overlap k - s (or k + (ov - 1) s behind a
    tiled successor), the rounding left-over added to the last tile; None where NPUC would assert"""
    cnt, f, m, l, ov = lens
    dim = (lambda x: k + (x - 1) * s if x else 0)
    f2, m2, l2 = dim(f), dim(m), dim(l)
    if cnt == 1:
        f2 -= pb + pa
    else:
        f2 -= pb
        l2 -= pa
    ov2 = k - s if ov == 0 else k + (ov - 1) * s
    tot = f2 if cnt == 1 else f2 - ov2 + l2 + (m2 - ov2) * (cnt - 2)
    diff = D - tot
    if (abs(diff) >= s) if pool else not (0 <= diff < s):
        return None
    if cnt == 1:
        f2 += diff
    else:
        l2 += diff
    return (cnt, f2, m2, l2, ov2)


def _pol_tiles(lens):
    """(start, length) of each tile along an axis: tile i starts (first - ov) + (i - 1)(middle - ov) (G: PTilesLayout::
    getTile; the TRIM test of 0x3b1364 uses exactly these starts)"""
    cnt, f, m, l, ov = lens
    if cnt == 1:
        return [(0, f)]
    out, a = [(0, f)], f - ov
    for _ in range(cnt - 2):
        out.append((a, m))
        a += m - ov
    return out + [(a, l)]


def _pol_blk(C, h, w, n=None):
    """PFeatureShape's per-IDP bytes as getRequiredMemory counts them: ceil(C / 2) planes rounded up to the IDPs used,
    each rup(h, 4) x 2 rup(w, 8) bytes"""
    if n is None:
        n = cell(C, 1, 1)["n"]
    return 2 * rup(w, 8) * rup(h, 4) * (rup(cdiv(C, 2), n) // n)


def _pol_largest(lay, C, aligned=False):
    """PTilesLayout::getLargestTileShape (0x3503d0): the tile with the most memory, the first such one in (W outer, H
    inner) order; aligned: its loaded shape, from the cell boundary at or before its start (getTileShape(.., true))"""
    best, mem = None, -1
    for ws, wl in _pol_tiles(lay["W"]):
        for hs, hl in _pol_tiles(lay["H"]):
            h, w = (hl + hs % 4, wl + ws % 8) if aligned else (hl, wl)
            m = _pol_blk(C, h, w)
            if m > mem:
                best, mem = (C, h, w), m
    return best


def _pol_maxin(L):
    """PTraits::getMaxNodeInputShape (0x34f7c0): the (H, W) a node's input must stay below. A conv: (s 256, s 256)
    without padding; with padding min(the pad unit's 256 rows, s 256) less pt + pb, and s 256 less pl + pr (the pad
    unit has no width bound). A pool: no bound without padding; with padding 256 rows less its pads. (V: PTraits2020
    with both configuration flags set, which the 1577 logged compiles confirm: 300-row pools stay whole, a 300-row
    stride-2 padded conv is tiled, a 300-wide one is not, 600 is)"""
    pt, pb, pl, pr = L["pads"]
    sh, sw = L["stride"]
    if L["kind"] == "conv":
        if not any(L["pads"]):
            return sh * 256, sw * 256
        return min(POL_MAX_H, sh * 256) - pt - pb, sw * 256 - pl - pr
    if not any(L["pads"]):
        return POL_INF, POL_INF
    return POL_MAX_H - pt - pb, POL_INF


def _pol_node(layers, shapes, li, wsize):
    """the per-node constants of NPUC's estimate"""
    L = layers[li]
    C, H, W = shapes[li]["in"]
    Cr = rup(C, 16) if li and layers[li - 1]["kind"] == "conv" else C     # V: "Convolution successor's ifm will be
    nd = {"L": L, "li": li, "C": C, "Cr": Cr, "in": (C, H, W), "out": shapes[li]["out"]}    # rounded to 16"
    if L["kind"] == "conv":
        nin = cell(C, H, W)["n"]
        nd["per"] = cdiv(wsize[li], nin)          # V: requiredWeights = ceil(compressed / IDPs of the true IFM)
        nd["co16"] = rup(L["cout"], 16)
        nd["P"] = nd["co16"] // 16
    return nd


def _pol_blocks(nd, ins, outs, nbuf=0):
    """PNodeOperation::calcRequiredMemory (0x343e04 over getRequiredMemory 0x342e64) for input shape ins and output
    shape outs (C, h, w): its IFM, OFM, working blocks (by the node's mode), weights (all, or nbuf buffers of 16
    channels: calcRequiredWeights), partial sums and whether its blocks may share memory"""
    L = nd["L"]
    _, h, w = ins
    _, oh, ow = outs
    C = nd["Cr"]
    nC = cell(C, 1, 1)["n"]
    pt, pb, pl, pr = L["pads"]
    sh, sw = L["stride"]
    ph, pw = max(pt, pb), max(pl, pr)
    b = {"ifm": _pol_blk(C, h, w, nC), "wk": [], "w": 0, "ps": 0, "nofms": False}
    if L["kind"] == "conv":
        co = nd["co16"]
        b["ofm"] = _pol_blk(co, oh, ow)
        p = min(ph, pw)
        mode = (0 if p == 0 else 11 if p < 4 else 20) if sw == 1 else (100 if p == 0 else 110 if p < 4 else 120)
        if mode == 11:
            b["wk"] = [_pol_blk(C, h + 2 * ph, w + 2 * pw, nC)]
        elif mode == 100:
            b["wk"] = [_pol_blk(sh * sw * rup(C, 2), cdiv(h, sh), cdiv(w, sw), nC)]
        elif mode == 110:
            b["wk"] = [_pol_blk(C, h + 2 * ph, w + 2 * pw, nC),
                       _pol_blk(sh * sw * C, cdiv(h + 2 * ph, sh), cdiv(w + 2 * pw, sw), nC)]
        b["w"] = nd["per"] if not nbuf else cdiv(nd["per"], co) * nbuf * 16
        # V (PNodeConvolution::getRequiredPartialSummationBlock 0x346070, requiredPartialSummationBlock 0x336220):
        # one weight buffer = s*s*p1 x 2 B x ceil(kh/s) ceil(kw/s), p1 = the IFM's planes per IDP (even for s > 1);
        # past 511 a block of 896 B per cell over ceil((w + pt + pb) / s) columns x ceil((h + pl + pr) / s) rows
        # (NPUC adds the vertical pads to the width and the horizontal ones to the height)
        p1 = rup(cdiv(C, 2), nC) // nC
        if sh > 1 or sw > 1:
            p1 = rup(p1, 2)
        kh, kw = L["k"]
        if sh * sw * p1 * 2 * cdiv(kh, sh) * cdiv(kw, sw) > POL_NZV:
            b["ps"] = cdiv(cdiv(w + pt + pb, sw), 8) * cdiv(cdiv(h + pl + pr, sh), 4) * POL_PSUM_CELL
    else:
        b["ofm"] = _pol_blk(nd["out"][0], oh, ow)
        if any(L["pads"]) and L.get("type") == "AveragePool":      # G: the pad-mode pools (2010/2110/2210)
            b["wk"] = [_pol_blk(C, h + 2 * ph, w + 2 * pw, nC)]
            b["nofms"] = True
    return b


def _pol_full(nd, nbuf=0, extended=True, ifm_pre=False, ofm_pre=False):
    """estimateFullMemoryRequirements (0x34481c): the whole node; extended = its weights live from its first op;
    ifm_pre, ofm_pre: its input (output) is held elsewhere and not counted (the layer-block forwarding's test)"""
    b = _pol_blocks(nd, nd["in"], nd["out"], nbuf)
    if ifm_pre:
        b["ifm"] = 0
    if ofm_pre:
        b["ofm"] = 0
    wk, x = b["wk"], b["ps"]
    if not wk:
        return b["ifm"] + b["ofm"] + b["w"] + x
    if b["nofms"]:
        return b["ifm"] + b["ofm"] + b["w"] + x + sum(wk)
    w3 = wk + [0, 0]
    pairs = max(b["ifm"] + w3[0], w3[0] + w3[1], w3[1] + w3[2])
    last = b["ofm"] + wk[-1] + x + (0 if extended else b["w"])
    return max(pairs, last) + (b["w"] if extended else 0)


def _pol_basic(shape, nh, nw):
    """calcBasicTilesLayout (0x3b31cc): a layout of nh x nw tiles over an output (C, H, W); None if not tileable"""
    C, H, W = shape
    a, b = _pol_lengths(H, nh, 4), _pol_lengths(W, nw, 8)
    return None if a is None or b is None else {"C": C, "H": a, "W": b}


def _pol_tile_ends(nds, tiles):
    """NPUC's calculateNodeCodegenParameters on a stratum's tiles (V: decompiled, PolicyMemory.cpp:297-300): every
    tile of the stratum's input (the split node's layout: tile i from (first - ov) + (i - 1)(middle - ov), first,
    middle or last long) must end inside the map and after its start, else NPUC asserts ("Tile end row exceeds FM
    shape" / "Tile end column ..." / "Negative tile length"; V: tiling2 b04, a middle row tile ending on row 9 of 9;
    posenet-mobilenet-v1-257, conv1 .. dw4 in 5 row tiles, the fourth ending on row 260 of 257). Raises
    NotImplementedError naming NPUC's assertion."""
    nh, nw = tiles
    lays = _pol_layouts(nds, _pol_basic(nds[-1]["out"], nh, nw) or {})
    if not lays:
        return
    _, H, W = nds[0]["in"]
    for ax, D, what in (("H", H, "row"), ("W", W, "column")):
        for a, n in _pol_tiles(lays[0][ax]):
            if a + n - 1 >= D:
                raise NotImplementedError('NPUC asserts: "Tile end %s exceeds FM shape" (layer %s: its %d x %d tiles, '
                                          'one from %s %d to %d of %d)' % (what, nds[0]["li"], nh, nw, what, a,
                                                                          a + n - 1, D))
            if n < 2:
                raise NotImplementedError('NPUC asserts: "Negative tile length" (layer %s: its %d x %d tiles)'
                                          % (nds[0]["li"], nh, nw))


def _pol_back(nd, lay):
    """backPropagateTiles: the input layout of a node for output layout lay"""
    L = nd["L"]
    C, H, W = nd["in"]
    pt, pb, pl, pr = L["pads"]
    pool = L["kind"] == "pool"
    a = _pol_back_axis(lay["H"], L["k"][0], L["stride"][0], pt, pb, H, pool)
    b = _pol_back_axis(lay["W"], L["k"][1], L["stride"][1], pl, pr, W, pool)
    return None if a is None or b is None else {"C": C, "H": a, "W": b}


def _pol_stratum(nds, lays, nbuf=0):
    """estimateStratumMemoryRequirements (0x3b1364): nodes nds, layouts lays (the head's input, then each node's
    output). Sum of the weights and partial sums + the largest pair of consecutive blocks along [IFM, (TRIM), per node:
    its working blocks, its output]; POL_INF when a tile passes a node's input bound"""
    l0 = lays[0]
    ifm_sh = _pol_largest(l0, l0["C"])
    ifm = _pol_blk(*ifm_sh)
    wk = []
    hc, wc = l0["H"], l0["W"]
    aligned = all([hc[0] < 2 or (hc[1] - hc[4]) % 4 == 0, wc[0] < 2 or (wc[1] - wc[4]) % 8 == 0,
                   hc[0] < 3 or (hc[2] - hc[4]) % 4 == 0, wc[0] < 3 or (wc[2] - wc[4]) % 8 == 0])
    if not aligned:             # a TRIM block (the tile's own input), the IFM its loaded (cell-aligned) shape
        wk.append(ifm)
        ifm = _pol_blk(*_pol_largest(l0, l0["C"], True))    # (getMaxPUInputShape bounds it: no bound here)
    base, ofm_last = 0, 0
    for i, nd in enumerate(nds):
        ins = _pol_largest(lays[i], lays[i]["C"])
        mh, mw = _pol_maxin(nd["L"])
        if ins[1] > mh or ins[2] > mw:
            return POL_INF
        outs = _pol_largest(lays[i + 1], lays[i + 1]["C"])
        b = _pol_blocks(nd, ins, outs, nbuf)
        base += b["w"] + b["ps"]
        wk += b["wk"]
        if i == len(nds) - 1:
            ofm_last = b["ofm"]
        else:
            wk.append(b["ofm"])
    tot = ofm_last + ifm + base if not wk else \
        base + max(a + b for a, b in zip([ifm] + wk + [ofm_last], wk + [ofm_last]))
    if _POL_TRACE is not None:  # NPUC's "Memory requirements calculated on tiled nodes" print (frame_sweep.py policy)
        _POL_TRACE.append([[nd["li"] for nd in nds], ifm, ofm_last, wk, tot])
    return tot


def _pol_layouts(nds, out_lay):
    """the layouts of a chain for its last output layout: [head input, ..., last output], or None"""
    lays = [out_lay]
    for nd in reversed(nds):
        lin = _pol_back(nd, lays[0])
        if lin is None:
            return None
        lays.insert(0, lin)
    return lays


def _pol_fit_h(nd, nw, nbuf):
    """FUN_003b0fe8 (0x3b0fe8): for nw columns, the fewest rows nh (isTileable, not 1 x 1) whose loaded input tile
    stays below the node's input bound and whose estimate fits 256 KiB; (nh, layouts) or None"""
    C, Ho, Wo = nd["out"]
    mh, mw = _pol_maxin(nd["L"])
    for nh in range(1, cdiv(Ho, 4) + 1):
        if nh == 1 and nw == 1:
            continue
        lo = _pol_basic(nd["out"], nh, nw)
        if lo is None:
            continue
        lays = _pol_layouts([nd], lo)
        if lays is None:
            continue
        _, h, w = _pol_largest(lays[0], lays[0]["C"], True)
        if h >= mh or w >= mw:
            continue
        if _pol_stratum([nd], lays, nbuf) <= SRAM_LIMIT:
            return nh, lays
    return None


def _pol_whole(nd, limit, ifm_pre=False, ofm_pre=False):
    """getBestProcessingParams' whole-node part (0x3adc8c) within limit: {"whole", "nbuf", "extended"}. All weights
    with an extended lifetime when they fit; else tryAllocateWeightsPartially's ring (FUN_003b0e2c: the largest
    ceil(P / k), k = 2.., that fits), kept with 5 or more buffers when extended, else retried with a short lifetime;
    all weights with a short lifetime when they fit and no ring of more than 4 buffers does. The weights' lifetime is
    a tag the call sets on the node ("weights_have_extended_lifetime"): cleared whenever the extended forms fail,
    also when the node is not kept whole (V: decompiled; the layer-block forwarding's side effect, forwarding())"""
    L = nd["L"]
    conv = L["kind"] == "conv"
    P = nd.get("P", 0)

    def ring(extended):
        prev = 0
        for k in range(2, P + 1):
            nb = cdiv(P, k)
            if nb != prev:
                if _pol_full(nd, nb, extended, ifm_pre, ofm_pre) <= limit:
                    return nb
                prev = nb
        return 0

    if _pol_full(nd, 0, True, ifm_pre, ofm_pre) <= limit:
        return {"whole": True, "nbuf": 0, "extended": True}
    all_short = _pol_full(nd, 0, False, ifm_pre, ofm_pre) <= limit
    nb1 = ring(True) if conv else 0
    if not all_short:
        if nb1 >= 5:
            return {"whole": True, "nbuf": nb1, "extended": True}
        nb2 = ring(False) if conv else 0
        return {"whole": bool(nb2), "nbuf": nb2, "extended": False}
    if nb1 > 4:
        return {"whole": True, "nbuf": nb1, "extended": True}
    return {"whole": True, "nbuf": 0, "extended": False}


def _pol_best(nd):
    """getBestProcessingParams (0x3adc8c) for one node: {"tiles": (nh, nw), "nbuf": weight buffers (0 = all),
    "extended": the weights' lifetime}. Whole when it fits (all weights with an extended lifetime; else a ring of
    ceil(P / k) buffers, k = 2.., kept only with 5 or more buffers when extended, else retried with a short lifetime);
    else tiles: nw = 1.. outer, nh = 1.. inner (_pol_fit_h), all weights first, then (P < 33) rings of ceil(P / k)
    buffers, k = P..1, grown while the row count does not rise; P >= 33: rings only"""
    L = nd["L"]
    C, H, W = nd["in"]
    mh, mw = _pol_maxin(L)
    conv = L["kind"] == "conv"
    P = nd.get("P", 0)

    if H < mh and W < mw:
        r = _pol_whole(nd, SRAM_LIMIT)
        if r["whole"]:
            return {"tiles": (1, 1), "nbuf": r["nbuf"], "extended": r["extended"]}
    Ho, Wo = nd["out"][1], nd["out"][2]
    for nw in range(1, cdiv(Wo, 8) + 1):
        if _pol_lengths(Wo, nw, 8) is None:
            continue
        if not conv or P < 33:
            r = _pol_fit_h(nd, nw, 0)
            if r:
                return {"tiles": (r[0], nw), "nbuf": 0, "extended": False}
            if not conv:
                continue
        best, best_h, prev = 0, None, 0
        for k in range(P, 0, -1):
            nb = cdiv(P, k)
            if nb == prev:
                continue
            prev = nb
            r = _pol_fit_h(nd, nw, nb)
            if r is None or (best_h is not None and best_h < r[0]):
                break
            best, best_h = nb, r[0]
        if best:
            _pol_fit_h(nd, nw, best)    # NPUC computes the chosen ring's tiles once more ("validTilesFoundAgain")
            return {"tiles": (best_h, nw), "nbuf": best, "extended": False}
    return None


def _pol_chain_layout(nds, tail_nw):
    """getStratumTilesLayout / findStratumNumTilesAlongH (in 0x3ae948): nw from the tail's own column count up, nh
    from 1 (isTileable, not 1 x 1): every node's input tile below its bound, the head's input tile longer than 3x its
    overlap along both axes, the chain's estimate within 256 KiB; (nh, nw) or None (chains of at most 10 nodes)"""
    if len(nds) > 10:
        return None
    _, Ho, Wo = nds[-1]["out"]
    for nw in range(tail_nw, cdiv(Wo, 8) + 1):
        if _pol_lengths(Wo, nw, 8) is None:
            continue
        for nh in range(1, cdiv(Ho, 4) + 1):
            if nh == 1 and nw == 1:
                continue
            lo = _pol_basic(nds[-1]["out"], nh, nw)
            if lo is None:
                continue
            lays = _pol_layouts(nds, lo)
            if lays is None:
                continue
            ok = True
            for i, nd in enumerate(nds):
                _, h, w = _pol_largest(lays[i], lays[i]["C"])
                mh, mw = _pol_maxin(nd["L"])
                if h >= mh or w >= mw:
                    ok = False
                    break
                if i == 0 and (h <= 3 * abs(lays[0]["H"][4]) or w <= 3 * abs(lays[0]["W"][4])):
                    ok = False
                    break
            if ok and _pol_stratum(nds, lays) <= SRAM_LIMIT:
                return nh, nw
    return None


def npuc_policy(low, wsize):
    """NPUC's tiling decisions for a chain network: per layer its own processing (_pol_best) and the strata
    (recalculateTilingForStratum): [{"layers": [..], "tiles": (nh, nw), "nbuf": ..}] for every tiled layer or chain,
    in layer order. A chain grows from its tail (a tiled layer with all its weights, or a layer behind one that is
    tiled and shrinks its data: IFM > OFM, a conv or pool) towards the input while _pol_chain_layout finds a layout;
    "readback": the output read-back's row tiles (_unflatten_tiles)."""
    layers = low["layers"]
    shapes = internal_shapes(low)
    nds = [_pol_node(layers, shapes, li, wsize) for li in range(len(layers))]
    Co, Ho, Wo = shapes[-1]["out"]
    readback = _unflatten_tiles(Co, Ho, Wo, layers[-1]["kind"] == "conv")     # the output node comes first
    best = [_pol_best(nd) for nd in reversed(nds)][::-1]      # V: computeTilesNumber goes from the output back
    if any(b is None for b in best):
        raise NotImplementedError("NPUC finds no processing for layer %d" % best.index(None))
    tiled = [b["tiles"] != (1, 1) for b in best]
    tnr = [t and not b["nbuf"] for t, b in zip(tiled, best)]
    chains, chain, chain_t = [], [], None
    for li in range(len(layers) - 1, -1, -1):
        cand = tnr[li]
        if li and tnr[li - 1] and not cand:
            if li + 1 < len(layers) and tnr[li + 1]:
                cand = True
            else:
                C, H, W = shapes[li]["in"]
                Co, Ho, Wo = shapes[li]["out"]
                cand = _pol_blk(C, H, W) > _pol_blk(Co, Ho, Wo)
        if not cand:
            continue
        if chain and chain[0] == li + 1:
            t = _pol_chain_layout([nds[x] for x in [li] + chain], best[chain[-1]]["tiles"][1])
            if t:
                chain, chain_t = [li] + chain, t
                continue
        if len(chain) > 1:
            chains.append((chain, chain_t))
        chain, chain_t = [li], None
    if len(chain) > 1:
        chains.append((chain, chain_t))
    inchain = {x: c for c in chains for x in c[0]}
    out = []
    for li in range(len(layers)):
        if li in inchain:
            c, t = inchain[li]
            if li == c[0]:
                out.append({"layers": c, "tiles": t, "nbuf": 0})
        elif tiled[li]:
            out.append({"layers": [li], "tiles": best[li]["tiles"], "nbuf": best[li]["nbuf"]})
    for st in out:
        _pol_tile_ends([nds[x] for x in st["layers"]], st["tiles"])
    return {"best": best, "strata": out, "readback": readback}


def graph_shapes(low):
    """(order, shapes) of a graph network: NPUC's layer order (common.exec_order) and each layer's backend input and
    output shapes {name: {"in": (C, H, W), "out": (Co, Ho, Wo)}}, by internal_shapes' rule (a VALID pool that drops
    columns keeps one more); a concat's input is its first input's shape"""
    ins, cons, by, src = topology(low)
    order = exec_order(net_outputs(low), ins, cons, src)
    chw = {src: tuple(low["input"]["chw"])}
    shp = {}
    for n in order:
        L = by[n]
        C, H, W = chw[ins[n][0]]
        if L["kind"] in ("conv", "pool"):
            kh, kw = L["k"]
            sh, sw = L["stride"]
            pt, pb, pl, pr = L["pads"]
            Ho, Wo = (H + pt + pb - kh) // sh + 1, (W + pl + pr - kw) // sw + 1
            if L["kind"] == "pool" and (Ho - 1) * sh + kh - pt - pb - H >= 0 > (Wo - 1) * sw + kw - pl - pr - W:
                Wo += 1
            Co = L["out_chw"][0]
        elif L["kind"] == "concat":
            Co, Ho, Wo = sum(chw[i][0] for i in ins[n]), H, W
        else:
            Co, Ho, Wo = C, H, W
        shp[n] = {"in": (C, H, W), "out": (Co, Ho, Wo)}
        chw[n] = (Co, Ho, Wo)
    return order, shp


def graph_policy(low, wsize=None):
    """(memoised by the network's content when wsize is weights.py's) _graph_policy"""
    if wsize is not None:
        return _graph_policy(low, wsize)
    return _GPOL.get(network_key(low), lambda: _graph_policy(low, None))


_GPOL = Memo(16)


def _graph_policy(low, wsize=None):
    """NPUC's tiling decisions for a graph network (computeTilesNumber over the tsorted graph from the output back,
    then recalculateTilingForStratum; frame.md 6.10, 6.11): per conv, pool and standalone ReLU node its own
    processing (_pol_best: a ReLU as its 1x1 dummy conv), its IFM rounded to 16 channels behind a conv; strata grow
    from a tiled node with all its weights towards the input along single links (the head's one predecessor has one
    successor) as npuc_policy's chains do. An eltwise is never tiled: NPUC asserts in its SingleTileBackPropagator
    when its operands and result pass 256 KiB (V: graph-split res96-64 .. res160-64; G for the estimate, 3 blocks).
    wsize: {layer: its WEIGHT block's size} (default weights.py's). Returns {"order", "shapes", "nodes", "best",
    "strata": [{"layers": [names, head first], "tiles": (nh, nw), "nbuf"}]}."""
    ins, cons, by, src = topology(low)
    order, shp = graph_shapes(low)
    if wsize is None:
        wl = [n for n in order if by[n]["kind"] in ("conv", "relu", "eltwise")]
        wsize = dict(zip(wl, _weight_sizes(low)))
    nds = {}
    for n in order:
        L, k = by[n], by[n]["kind"]
        C, H, W = shp[n]["in"]
        if k == "concat":
            continue
        if k == "eltwise":
            if 3 * _pol_blk(C, H, W) > SRAM_LIMIT:
                raise NotImplementedError("eltwise %s: %d x %d x %d operands need tiling, which NPUC asserts on "
                                          "('Trying to split an unsplittable layer', SingleTileBackPropagator)"
                                          % (n, C, H, W))
            continue
        i0 = ins[n][0]
        P = L if k != "relu" else {"kind": "conv", "name": n, "k": [1, 1], "stride": [1, 1], "pads": [0, 0, 0, 0],
                                   "cout": C}
        nd = {"L": P, "li": n, "C": C, "Cr": rup(C, 16) if i0 != src and by[i0]["kind"] == "conv" else C,
              "in": (C, H, W), "out": shp[n]["out"]}
        if P["kind"] == "conv":
            nd["per"] = cdiv(wsize[n], cell(C, H, W)["n"])
            nd["co16"] = rup(P["cout"], 16)
            nd["P"] = nd["co16"] // 16
        nds[n] = nd
    best = {}
    for n in reversed(order):
        if n in nds:
            best[n] = _pol_best(nds[n])
            if best[n] is None:
                raise NotImplementedError("NPUC finds no processing for layer %s" % n)
    tnr = {n: b["tiles"] != (1, 1) and not b["nbuf"] for n, b in best.items()}

    def pred1(n):           # the single predecessor of n when it is a node read by n alone
        p = ins[n]
        return p[0] if len(p) == 1 and p[0] in nds and len(cons.get(p[0], [])) == 1 else None

    def succ1(n):           # the single successor of n when n is its only input
        s = cons.get(n, [])
        return s[0] if len(s) == 1 and s[0] in nds and len(ins[s[0]]) == 1 else None
    chains, chain, chain_t = [], [], None
    for n in reversed(order):
        if n not in nds:
            continue
        cand = tnr[n]
        p = pred1(n)
        if any(by[r]["kind"] == "eltwise" for r in cons.get(n, [])):
            # a node feeding an eltwise neither starts nor joins a stratum (V: large random t1002, a tiled residual
            # block input left alone; G: the decompile's type test before a chain starts)
            continue
        if not cand and p is not None and tnr[p]:
            s = succ1(n)
            if s is not None and tnr[s]:
                cand = True
            else:
                cand = by[n]["kind"] in ("conv", "pool") and \
                    not any(by[r]["kind"] == "eltwise" for r in cons.get(n, [])) and \
                    _pol_blk(*shp[n]["in"]) > _pol_blk(*shp[n]["out"])
        if not cand:
            continue
        if chain and pred1(chain[0]) == n and succ1(n) == chain[0]:
            t = _pol_chain_layout([nds[x] for x in [n] + chain], best[chain[-1]]["tiles"][1])
            if t:
                chain, chain_t = [n] + chain, t
                continue
        if len(chain) > 1:
            chains.append((chain, chain_t))
        chain, chain_t = [n], None
    if len(chain) > 1:
        chains.append((chain, chain_t))
    inchain = {x: c for c in chains for x in c[0]}
    strata = []
    for n in order:
        if n in inchain:
            c, t = inchain[n]
            if n == c[0]:
                strata.append({"layers": c, "tiles": t, "nbuf": 0})
        elif n in best and best[n]["tiles"] != (1, 1):
            strata.append({"layers": [n], "tiles": best[n]["tiles"], "nbuf": best[n]["nbuf"]})
    for st in strata:
        _pol_tile_ends([nds[x] for x in st["layers"]], st["tiles"])
    return {"order": order, "shapes": shp, "nodes": nds, "best": best, "strata": strata, "wsize": wsize}


def forwarding(low, pol=None):
    """NPUC's layer-block forwarding tests (LayerBlockFMSharingCalculator::calculateOptimizationParameters, one pass
    per mode: input sharing, concat in SRAM, full forwarding; V: decompiled, oracle/ghidra/targets-graphtile2.txt):
    a block is a fork whose readers all lead, through layers of one input and one reader, to one concat that reads
    nothing else. Over the nodes in order, each mode holds its block's initial memory (input sharing: the head's
    output; concat in SRAM: the concat's ("ahead_concat_size"); full forwarding: both) and tests every untiled node
    of the block whose predecessor passed (or is the head) with getBestProcessingParams(node, 256 KiB - that - its
    predecessor's output when the predecessor is not the head and has several readers), the input not counted when
    the predecessor is the head (input sharing, full forwarding), the output not counted when a concat reads it
    (concat in SRAM, full forwarding); less than one byte left: "is too fat", no test. A node passes when it stays
    whole. Each test sets the node's weight lifetime (_pol_whole), so the last test wins (V: large random t1004,
    t1013: the branch convs' weights live on their conv only). A block succeeds in a mode when every input of its
    concat passed (or is the head: tryProcessLayerBlockEnd). Returns {"life": {node: extended} for the nodes tested,
    "ok": {mode: {head: succeeded}}, "nbuf": {mode: {node: weight buffers it passed with}}, "heads": {head: concat}}."""
    ins, cons, by, src = topology(low)
    pol = pol or graph_policy(low)
    nds, shp = pol["nodes"], pol["shapes"]
    tiled = {x for st in pol["strata"] for x in st["layers"]}
    order = pol["order"]

    def out_mem(n):
        if n == src:
            return _pol_blk(*low["input"]["chw"])
        C, H, W = shp[n]["out"]
        return _pol_blk(rup(C, 16) if by[n]["kind"] == "conv" else C, H, W)

    def root(t):
        while t != src and by[t]["kind"] in ("conv", "pool", "relu") and len(cons[t]) == 1 and len(ins[t]) == 1:
            t = ins[t][0]
        return t
    heads = {}                              # block head -> its concat
    for kn in order:
        if by[kn]["kind"] != "concat":
            continue
        rs = {root(i) for i in ins[kn]}
        if len(rs) != 1:
            continue
        r = rs.pop()
        if r != src and by[r]["kind"] == "concat" and r in ins[kn]:
            continue
        # every reader of the head on a branch to this concat
        branch = set()
        for i in ins[kn]:
            t = i
            while t != r:
                branch.add(t)
                t = ins[t][0]
        if all(x in branch or x == kn for x in cons.get(r, [])):
            heads[r] = kn
    life, res, nbuf = {}, {}, {}
    for mode in ("is", "cs", "ff"):
        cur, mem0, ok = None, 0, set()
        res[mode], nbuf[mode] = {}, {}
        for n in [src] + order:
            if n != src and by[n]["kind"] == "concat" and cur is not None:
                if heads.get(cur) == n:
                    res[mode][cur] = all(i in ok for i in ins[n])
                cur = None
            if n in heads:
                kn = heads[n]
                C, H, W = shp[kn]["out"]
                cat = _pol_blk(C, H, W)
                mem0 = {"is": out_mem(n), "cs": cat, "ff": out_mem(n) + cat}[mode]
                cur, ok = n, {n}
                continue
            if cur is None or n == src or by[n]["kind"] == "concat" or n not in nds or n in tiled:
                continue
            if len(ins[n]) != 1 or ins[n][0] not in ok:
                continue
            p = ins[n][0]
            more = out_mem(p) if p != cur and len(cons.get(p, [])) >= 2 else 0
            mem = SRAM_LIMIT - mem0 - more
            if mem < 1:
                continue
            rd = cons.get(n, [])
            r = _pol_whole(nds[n], mem, ifm_pre=mode != "cs" and p == cur,
                           ofm_pre=mode != "is" and bool(rd) and by[rd[0]]["kind"] == "concat")
            life[n] = r["extended"]
            if r["whole"]:
                ok.add(n)
                nbuf[mode][n] = r["nbuf"]
    return {"life": life, "ok": res, "nbuf": nbuf, "heads": heads}


def concat_ofm_shared(kn, cons, fwd):
    """NPUC keeps a concat in SRAM only while its OFM is shared (LivenessAnalysis::collectAllChunks: "Concat-node id#k
    was supposed to be collected in SRAM but its OFM is not shared so it is useless. ConcatInSram was turned off
    here!"; V: decompiled, large random t2006): read by one layer, or by several that share it, the input-sharing
    block it heads succeeding (forwarding's "is"); else it goes to DRAM, its readers' weights in the DRAM order"""
    return len(cons.get(kn, [])) < 2 or bool(fwd["ok"]["is"].get(kn))


def node_whole(nd, limit):
    """getBestProcessingParams(node, limit) keeps the node whole with all its weights (the test NPUC's layer-block
    forwarding makes of a block's nodes with the SRAM the block leaves, LayerBlockFMSharingCalculator::
    tryProcessInternalNode): its input below its bound and the whole node's estimate, extended or short weight
    lifetime, within limit (a weight ring within limit also passes NPUC's test: not modelled)"""
    mh, mw = _pol_maxin(nd["L"])
    _, H, W = nd["in"]
    if H >= mh or W >= mw:
        return False
    return _pol_full(nd, 0, True) <= limit or _pol_full(nd, 0, False) <= limit


def _tile_dma(dmas, lay, rng, dram_vec, sram_off, load, g, ns0=False, pool=False, base=0, parts=None):
    """one tile's DMAs between a DRAM tensor (IDP-major, real planes) and the SRAM (V: H tiles use h = rows, w = 2Wp;
    tiles narrower than the tensor use cell rows, w = cells * 64 and a DRAM w_stride; more than one plane per IDP: one
    DMA per IDP through its own SRAM vector, planes contiguous in SRAM). base: the tensor's offset in the DRAM vector
    (a graph's concat buffer, frame.md 6.8). parts: a per-IDP load in 4 parts of that many planes, part by part over
    the IDPs (NPUC's DMACluster; 6.11)"""
    r0, r1, c0, c1 = rng
    rows = rup(r1 - r0 + 1, 4)
    cells_row = lay["Wp"] // 8
    cells = rup(c1 + 1, 8) // 8 - c0 // 8
    full = cells == cells_row
    if full:
        h, w, ws = rows, 2 * lay["Wp"], 0
        doff = r0 * 2 * lay["Wp"]
        dspan = h * w
        if lay["n"] == 1 and lay["pp"] == 1:
            h, w = dspan // 64, 64      # V (final-chains f059, C = 1): one IDP's tile as one blob 64 B wide
    else:
        h, w, ws = rows // 4, cells * 64, (cells_row - cells) * 64
        doff = (r0 // 4) * cells_row * 64 + (c0 // 8) * 64
        dspan = h * (w + ws)
    doff += base
    pp, n = lay["pp"], lay["n"]
    if pp == 1 or n == 1 and pp == 1:
        sdram = _blob(dram_vec, doff, 1, n, h, w, ws=ws, cs=lay["per_idp"] - dspan if n > 1 else 0)
        ssram = _blob(0, sram_off, 1, n, h, w, cs=(IDP_STRIDE - h * w) if n > 1 else 0)
        dmas.append((g, 0, sdram, ssram) if load else (g, 1, ssram, sdram))
    elif parts and load:
        for s0 in range(0, pp, parts):
            q = min(pp, s0 + parts) - s0
            for i in range(n):
                sdram = _blob(dram_vec, i * lay["per_idp"] + s0 * lay["plane"] + doff, 1, q, h, w, ws=ws,
                              cs=lay["plane"] - dspan)
                # V: no n_stride for the part landing at SRAM offset 0 (large random t1016's first part; t1031's
                # parts at 7040.. all keep it); G: the rule behind the "direct" loads above
                o = sram_off + s0 * h * w
                ssram = _blob(i, o, 1, q, h, w, ns=0 if o == 0 else IDP_STRIDE - q * h * w)
                dmas.append((g, 0, sdram, ssram))
    else:
        for i in range(n):
            # V: loads move every IDP's pp planes; a store from an IDP holding fewer planes (the last one) moves
            # only those (a pool's: pp rounded down to even, as untiled); one plane: no n_stride
            pl_ = pp
            if not load:
                pl_ = cdiv(min(lay["cpi"], lay["C"] - i * lay["cpi"]), 2)
                if pool and pl_ < pp:
                    pl_ = pp - pp % 2
            sdram = _blob(dram_vec, i * lay["per_idp"] + doff, 1, pl_, h, w, ws=ws, cs=lay["plane"] - dspan)
            ssram = _blob(i, sram_off, 1, pl_, h, w, ns=0 if pl_ == 1 or sram_off == 0 else IDP_STRIDE - pl_ * h * w)
            dmas.append((g, 0, sdram, ssram) if load else (g, 1, ssram, sdram))


def plan_tiled(low, wsize, fix_npuc=False, nsub=None, policy=None):
    """NPUC's plan when layers must be tiled: strata of chained layers, each cut into tiles (one group per tile),
    joined through DRAM (IM vectors), then the output read-back (one or more output groups). Which layers are tiled,
    how they chain and into how many tiles (and a tiled conv's weight ring) are NPUC's policy, read from its code
    (npuc_policy, frame.md 6.10); the tiles' geometry, ops, SRAM chunks and DMAs are the rules of frame.md 6.5 (V on
    the tiled compiles of frame_sweep.py's 'tiled' family except where marked G)."""
    layers = low["layers"]
    shapes = internal_shapes(low)
    convs = [i for i, L in enumerate(layers) if L["kind"] == "conv"]
    if nsub is None:
        nsub = _sub_batches(low)
    for L in layers:
        if L["kind"] == "conv" and tuple(L["stride"]) not in ((1, 1), (2, 2)):
            raise NotImplementedError("tiled conv with stride %s" % (L["stride"],))
    if policy is None:
        policy = npuc_policy(low, wsize)
    # blocks: NPUC's tiled strata (a chain, or one tiled layer), the runs of other layers between them in one tile
    st_at = {st["layers"][0]: st for st in policy["strata"]}
    in_st = {x for st in policy["strata"] for x in st["layers"]}
    blocks = []
    li = 0
    while li < len(layers):
        if li in st_at:
            st = st_at[li]
            blocks.append({"kind": "tiled", "layers": st["layers"], "tiles": st["tiles"], "nbuf": st["nbuf"]})
            li = st["layers"][-1] + 1
            continue
        run = []
        while li < len(layers) and li not in in_st:
            run.append(li)
            li += 1
        blocks.append({"kind": "single", "layers": run})
    plan_strata = []
    for blk in blocks:
        if blk["kind"] == "single":
            plan_strata.append(blk)
            continue
        lis = blk["layers"]
        last = lis[-1]
        _, Ho, Wo = shapes[last]["out"]
        nh, nw = blk["tiles"]
        g = _tile_geometry(layers, shapes, lis, _tile_ranges(Ho, nh, 4), _tile_ranges(Wo, nw, 8))
        trim = any(t["load"][0] != t["layers"][lis[0]]["in"][0] or t["load"][2] != t["layers"][lis[0]]["in"][2]
                   for t in g)
        ops = [_stratum_ops(layers, shapes, lis, t, trim) for t in g]
        mx = [dict(o, size=max(x[k]["size"] for x in ops)) for k, o in enumerate(ops[0])]
        ws, ps, ring, step = [], {}, {}, {}
        for li in lis:
            if layers[li]["kind"] != "conv":
                continue
            L = layers[li]
            if (nsub.get(li) or 1) > 1:
                # the PARTIAL_SUM over the tile's MTC input (its padded input, or the space-to-depth input)
                sz = 0
                for t in g:
                    i0, i1, j0, j1 = t["layers"][li]["in"]
                    pt, pb, pl, pr = t["layers"][li]["pads"]
                    sz = max(sz, _ps_size({"in": (0, i1 - i0 + 1, j1 - j0 + 1)}, dict(L, pads=[pt, pb, pl, pr])))
                ps[li] = sz
            per = wsize[li] // cell(*shapes[li]["in"])["n"]
            passes = cdiv(L["cout"], 16)
            # V (NPUC's policy): a tiled conv's weights are all resident, or nbuf of its passes ring through as many
            # slots, every tile reloading every pass (c512 2 of 32, r_c384 6 of 24, th-256-h64 4 of 16, ...)
            ring[li] = blk["nbuf"] or passes
            wsz, step[li] = _ring_chunk(per, L["cout"], ring[li], passes)
            ws.append(wsz)
        ch = _stratum_alloc(layers, lis, mx, ws, ps)
        if ch is None:
            raise NotImplementedError("no SRAM allocation for the %d x %d tiles of layers %s" % (nh, nw, lis))
        plan_strata.append({"kind": "tiled", "layers": lis, "tiles": g, "ops": mx, "chunks": ch, "nh": nh, "nw": nw,
                            "rows": _tile_ranges(Ho, nh, 4), "cols": _tile_ranges(Wo, nw, 8), "ring": ring,
                            "wstep": step})
    # vectors: SRAMs, IN, OT, one IM per stratum (its output, in order), WEIGHT per conv, IM 8x8x2 (V: t3c)
    C0, H0, W0 = low["input"]["chw"]
    Co, Ho, Wo = layers[-1]["out_chw"]
    _, Hi, Wi = shapes[-1]["out"]
    av, mv = [], []

    def vec(t, size, shape=(0, 0, 0)):
        av.append([len(av), 0, 0, size])
        mv.append([t, 1, shape[0], shape[1], shape[2], 0, 0, len(av) - 1])
        return len(av) - 1

    for k in range(IDP):
        vec(MV_P0 + k, SRAM_VEC)
    inl = cell(C0, H0, W0)
    v_in = vec(MV_IN, inl["size"], (W0, H0, C0))
    v_out = vec(MV_OT, ofm_size(Co, Ho, Wo), (Wo, Ho, Co))
    v_ims = []
    for st in plan_strata:
        if st["kind"] == "single":      # its spilled tensors first (frame.md 6.12)
            st["spill_vec"] = _spill_vectors(layers, shapes, st["layers"], wsize, nsub, policy["best"], fix_npuc, vec)
        c_, h_, w_ = shapes[st["layers"][-1]]["out"]
        Lt = layers[st["layers"][-1]]["out_chw"]
        im = _im_vector(c_, h_, w_, Lt[0] * Lt[1] * Lt[2])
        v_ims.append(vec(MV_IM, im["av"], (im["w"], im["h"], im["c"])))
    v_w = {li: vec(MV_WT, wsize[li]) for li in convs}
    vec(MV_IM, 138, (8, 8, 2))

    dmas = []
    group = 0

    def tile_dma(lay, rng, dram_vec, sram_off, load, g, ns0=False, pool=False):
        _tile_dma(dmas, lay, rng, dram_vec, sram_off, load, g, ns0, pool)

    src_vec = v_in
    for si, st in enumerate(plan_strata):
        lis = st["layers"]
        if st["kind"] == "single":
            st["chunks"], st["peak"], _ = _single_block(layers, shapes, lis, wsize, src_vec, v_ims[si], v_w, group,
                                                        dmas, fix_npuc, nsub, policy["best"], st["spill_vec"])
            ng = len(lis) + sum(1 for c in st["chunks"] if c.get("copy"))
            st["groups"] = list(range(group, group + ng))
            group += ng
            src_vec = v_ims[si]
            continue
        lin = cell(*shapes[lis[0]]["in"])
        c_, h_, w_ = shapes[lis[-1]]["out"]
        lout = cell(c_, h_, w_)
        split_off = next(c["off"] for c in st["chunks"] if c["kind"] == "SPLIT")
        last_off = st["chunks"][-1]["off"] if False else None
        out_chunk = [c for c in st["chunks"] if c["kind"] in ("CONV", "POOL") and c.get("layer") == lis[-1]][0]
        # V: per-IDP tile loads carry no SRAM n_stride when the stratum is one conv reading the tile directly (k1,
        # no pad; ts128c32k1, nsA, nsD, nsE); a longer stratum starting so keeps it (nsC, m3-mnet-stem224)
        direct = st["ops"][1]["op"] == "CONV" and not any(layers[lis[0]]["pads"]) and \
            tuple(layers[lis[0]]["stride"]) == (1, 1) and len(lis) == 1
        wch = [c for c in st["chunks"] if c["kind"] == "WEIGHT"]
        st["groups"] = list(range(group, group + len(st["tiles"])))
        clus = lin["pp"] >= 16 and lin["n"] > 1 and st["ops"][1]["op"] in ("PAD", "POOL")
        for ti, t in enumerate(st["tiles"]):
            _tile_dma(dmas, lin, t["load"], src_vec, split_off, True, group, ns0=direct,
                      parts=cdiv(lin["pp"], 4) if clus else None)
            for k, li in enumerate([x for x in lis if layers[x]["kind"] == "conv"]):
                nin = cell(*shapes[li]["in"])["n"]
                per = wsize[li] // nin
                nl = cdiv(layers[li]["cout"], 16)
                part = per // nl
                r = st["ring"][li]
                if ti and r == nl:
                    continue                # V: the weights loaded by tile 0 stay; a ring reloads every pass per tile
                for q in range(nl):
                    _weight_load(dmas, group, v_w[li], per, part, q, nin, nl, wch[k]["off"] + (q % r) * st["wstep"][li])
            tile_dma(lout, t["out"], v_ims[si], out_chunk["off"], False, group, pool=layers[lis[-1]]["kind"] == "pool")
            group += 1
        src_vec = v_ims[si]
    # the output read-back: split in rows by NPUC's policy on the output node (_unflatten_tiles)
    ol = cell(Co, Hi, Wi)
    nu = _unflatten_tiles(Co, Hi, Wi, layers[-1]["kind"] == "conv")
    outs = []
    for (u0, u1) in _tile_ranges(Hi, nu, 4):
        if nu == 1:
            un_n, un_c = _read_back(dmas, group, ol, Co, v_ims[-1], fix_npuc)
        else:
            tile_dma(ol, (u0, u1, 0, Wi - 1), v_ims[-1], 0, True, group, ns0=True)
            un_n, un_c = ol["n"], ol["cpi"]
        rows = u1 - u0 + 1
        outs.append({"n": un_n, "c": un_c, "C": Co, "H": rows if nu > 1 else Ho, "W": Wo, "Hi": rows if nu > 1 else Hi,
                     "Wi": Wi, "Hp": rup(rows, 4) if nu > 1 else ol["Hp"], "Wp": ol["Wp"], "sram_off": 0,
                     "group": group, "dst_off": u0 * Wo if nu > 1 else 0,
                     "dst_cs": (Ho - rows) * Wo if nu > 1 else 0})
        group += 1
    io = _io(dmas)
    ogroups = {u["group"] for u in outs}
    groups = [{"index": g, "kind": "output" if g in ogroups else ("dma" if any(x[2] == g for x in io) else "plain")}
              for g in range(group)]
    for st in plan_strata:
        if st["kind"] == "single":
            _mark_copies(groups, [c for c in st["chunks"] if c.get("copy")], st["groups"][0], st["layers"])
    allch = [c for st in plan_strata for c in st["chunks"]]
    out = {"layers": [L["kind"] for L in layers], "chunks": allch,
           "peak": max(c["off"] + c["size"] for c in allch),
           "av": av, "mv": mv, "io": io, "groups": groups, "v_out": v_out, "v_w": [v_w[li] for li in convs],
           "shapes": shapes, "strata": plan_strata, "unflatten": outs[0], "outputs": outs, "tiled": True}
    cut = [li for st in plan_strata if st["kind"] == "single" for li in st["layers"]
           if _cmdq_estimate(layers[li], shapes[li]) > CMDQ_LIMIT]
    # a stratum whose tiles NPUC cuts for their CMDQ size (6.9; V: ssd-mobilenet-v1 at 300 px, 2 groups a tile)
    tcut = any(_stratum_cmdq_estimate(layers, shapes, st) > CMDQ_LIMIT for st in plan_strata if st["kind"] == "tiled")
    if (cut or tcut) and not _IN_STREAM:
        _tiled_cut(low, out, dmas, cut, v_ims, nu)
    return out


def _stratum_cmdq_estimate(layers, shapes, st):
    """NPUC's "Heuristically predicted size of CMDQ" for a tiled stratum (one tile's ops; V: ssd-mobilenet-v1-t99,
    66570 for conv1 .. conv8 with a trimmed 3-channel input): 16350 + per layer its _cmdq_estimate share + 90 per
    trimmed plane slot"""
    lis = st["layers"]
    e = 16350 + sum(_cmdq_estimate(layers[li], shapes[li]) - 16350 for li in lis)
    if any(o["op"] == "TRIM" for o in st["ops"]):
        e += 90 * cdiv(cell(shapes[lis[0]]["in"][0], 1, 1)["cpi"], 2)
    return e


def _tiled_cut(low, p, dmas, cut, v_ims, nu):
    """a tiled plan whose single blocks hold layers NPUC cuts for their CMDQ size (6.9; V: mobilenet-v1-100-224, its
    1024-channel layers and classifier in 3 groups each): the groups, the DMA order, events and batches are the
    stream part's (its plan of the same network, frame's chunks); our descriptors follow its DMAs one for one, but
    for the stores of a cut layer, one per chunk of its passes' planes (_range_store, as in an untiled plan's cut)"""
    global _IN_STREAM
    layers = low["layers"]
    shapes = p["shapes"]
    _IN_STREAM = True
    try:
        dl = stream.plan_dmas(low)
        P = stream.Plan(low)
    finally:
        _IN_STREAM = False
    # each stream group's layer, and our group's
    starts = sorted((g, li) for li, g in P.chain_groups.items())
    sgl = {}
    for (g, li), nxt in zip(starts, starts[1:] + [(len(P.groups), None)]):
        for x in range(g, nxt[0]):
            sgl[x] = li
    fgl = {}
    for st in p["strata"]:
        g = st["groups"][0]
        if st["kind"] == "tiled":
            for x in st["groups"]:
                fgl[x] = st["layers"][0]
            continue
        cps = {c["layer"] for c in st["chunks"] if c.get("copy")}
        for li in st["layers"]:
            g += li in cps
            fgl[g] = li
            g += 1
    rows, k, j = [], 0, 0
    stores = [x for x in dl if x["kind"] == "store"]
    while j < len(dl):
        x = dl[j]
        li = sgl.get(x["sgid"])
        if x["kind"] == "store" and li in cut:
            # the cut layer's stores: our DMAs of its whole output skipped, one store per chunk of its passes
            while k < len(dmas) and dmas[k][1] == 1 and fgl.get(dmas[k][0]) == li:
                k += 1
            g = x["sgid"]
            run = [y for y in dl[j:] if y["kind"] == "store" and y["sgid"] == g]
            Co, _, _ = layers[li]["out_chw"]
            _, Hi, Wi = shapes[li]["out"]
            ol = cell(Co, Hi, Wi)
            oc = next(c for c in p["chunks"] if c.get("layer") == li and c["kind"] in ("CONV", "POOL"))
            st = next(st for st in p["strata"] if li in st["layers"])
            if st["layers"][-1] != li:
                raise NotImplementedError("a cut layer storing its output inside a single block")
            i = stores.index(x)
            nxt = next((y for y in stores[i + 1:] if y["sgid"] != g), None)
            step = 2 * ol["plane"]
            p0 = (x["sram"] - oc["off"]) // step
            p1 = (nxt["sram"] - oc["off"]) // step if nxt and sgl.get(nxt["sgid"]) == li else cdiv(layers[li]["cout"], 16)
            got = _range_store([], g, ol, oc["off"], v_ims[p["strata"].index(st)], p0, p1)
            if len(got) < len(run):
                raise NotImplementedError("a cut layer's store in %d DMAs, %d in the stream" % (len(got), len(run)))
            rows += got[:len(run)]
            j += len(run)
            continue
        if k >= len(dmas) or (dmas[k][1] == 1) != (x["kind"] == "store"):
            raise NotImplementedError("the stream's DMAs do not follow ours (a cut in a tiled plan)")
        rows.append((x["sgid"],) + tuple(dmas[k][1:]))
        k += 1
        j += 1
    if k != len(dmas):
        raise NotImplementedError("the stream's DMAs do not follow ours (a cut in a tiled plan): %d of %d" % (k, len(dmas)))
    G = len(P.groups)
    p["io"] = [[n_ + 1, x["bid"] or 0, g, 0, x["flag"], 2, 0, 1, 0] + sb + db
               for n_, (x, (g, d_, sb, db)) in enumerate(zip(dl, rows))]
    for u, o in zip(range(G - nu, G), p["outputs"]):
        o["group"] = u
    p["groups"] = [{"index": g, "kind": "output" if g >= G - nu else ("dma" if any(r[2] == g for r in p["io"])
                                                                       else "plain")} for g in range(G)]
    p["keep_bids"] = True
    p["cut"] = True


def _sub_batches(low):
    """{conv layer index: its weight sub-batch count} from weights.py; None for a conv it cannot
    pack (then the PARTIAL_SUM chunk is unknown)"""
    out = {}
    for li, L in enumerate(low["layers"]):
        if L["kind"] != "conv":
            continue
        try:
            out[li] = weights.conv_block(L, low.get("quant_mode", 1))[0]["sub_batches"]
        except Exception:
            # a packing weights.py does not model (stride 2, kernel >= 4, quant 1): more than 511 weight positions
            # per unit means sub-batches all the same (G; the allocated PARTIAL_SUM chunk only: NPUC's tiling
            # decisions count partial sums by their own 511-byte rule, npuc_policy)
            try:
                out[li] = 2 if weights.conv_geometry(L)["nbits"] > 511 else None
            except Exception:
                out[li] = None
    return out


def _weight_sizes(low):
    try:
        return weights.weight_block_sizes(low)
    except Exception:   # a packing weights.py refuses: an estimate (sizes only steer the allocation)
        return [_dense_weight_estimate(L) for L in low["layers"] if L["kind"] == "conv"]


def _dense_weight_estimate(L):
    """a stand-in WEIGHT size when weights.py's blocks are not given (G; exact sizes come from weights.py)"""
    C = L["in_chw"][0]
    n = cell(*L["in_chw"])["n"]
    kh, kw = L["k"]
    per = rup(L["cout"] * cdiv(C, n) * kh * kw + 64, 128)
    return per * n


# ---- the ARM intrinsics: our own A32 encoder and NPUC's two templates -------------------------------------------------
def _imm12(v):
    """A32 modified immediate: imm8 rotated right by 2*rot"""
    for rot in range(16):
        imm = ((v << (2 * rot)) | (v >> (32 - 2 * rot))) & 0xFFFFFFFF if rot else v
        if imm < 256:
            return rot << 8 | imm
    raise ValueError("no A32 immediate for %#x" % v)


AL = 0xE << 28
SP, LR, PC = 13, 14, 15


def a_push(regs):            # STMDB sp!, {regs}
    return AL | 0x092D0000 | sum(1 << r for r in regs)


def a_pop(regs):             # LDMIA sp!, {regs}
    return AL | 0x08BD0000 | sum(1 << r for r in regs)


def a_sub_imm(rd, rn, v):    # SUB rd, rn, #v
    return AL | 0x02400000 | rn << 16 | rd << 12 | _imm12(v)


def a_add_imm(rd, rn, v):    # ADD rd, rn, #v
    return AL | 0x02800000 | rn << 16 | rd << 12 | _imm12(v)


def a_mov_imm(rd, v):        # MOV rd, #v
    return AL | 0x03A00000 | rd << 12 | _imm12(v)


def a_mov(rd, rm):           # MOV rd, rm
    return AL | 0x01A00000 | rd << 12 | rm


def a_movw(rd, v):           # MOVW rd, #v (16 bits)
    return AL | 0x03000000 | (v >> 12 & 0xF) << 16 | rd << 12 | (v & 0xFFF)


def a_movt(rd, v):           # MOVT rd, #v (upper 16 bits)
    return AL | 0x03400000 | (v >> 12 & 0xF) << 16 | rd << 12 | (v & 0xFFF)


def a_str(rt, rn, off):      # STR rt, [rn, #off]
    return AL | 0x05800000 | rn << 16 | rt << 12 | off


def a_ldr(rt, rn, off):      # LDR rt, [rn, #off]
    return AL | 0x05900000 | rn << 16 | rt << 12 | off


def a_blx(rm):               # BLX rm
    return AL | 0x012FFF30 | rm


def _load32(rd, v):
    """MOVW (+ MOVT when the value has upper bits; G for values > 0xffff in the output struct)"""
    v &= 0xFFFFFFFF
    return [a_movw(rd, v & 0xFFFF)] + ([a_movt(rd, v >> 16)] if v >> 16 else [])


FRAME = 140                  # V: the stack frame both templates reserve (an io descriptor fits at sp+12)
SAVED = [4, 5, 6, 7, 8, 9, 10, 11]


def _prologue():
    # push {r4-r11, lr}; sub sp, sp, #140; mov r4, #0 (the constant 0); mov r5, #1 (the constant 1); mov r7, pc
    return [a_push(SAVED + [LR]), a_sub_imm(SP, SP, FRAME), a_mov_imm(4, 0), a_mov_imm(5, 1), a_mov(7, PC)]


def _epilogue():
    # mov r0, r4 (return 0); add sp, sp, #140; pop {r4-r11, pc}
    return [a_mov(0, 4), a_add_imm(SP, SP, FRAME), a_pop(SAVED + [PC])]


def _fw_call(entry, arg_insns):
    """call the firmware's intrinsic table entry: mov r3, #0x1d000; ldr r3, [r3, #4*entry]; <r0, r1 set>; blx r3"""
    return [a_mov_imm(3, FW_TABLE), a_ldr(3, 3, 4 * entry)] + arg_insns + [a_blx(3)]


def stub_intrinsic():
    """V: groups that only carry DMAs (io descriptors) get this 32-byte function that returns 0"""
    return _pack(_prologue() + _epilogue())


def output_intrinsic(u, out_vector, group):
    """V (disassembled, all variants): the output group's ARM function. It calls table entry 5 with 0x10000|group,
    fills a 31-word blob descriptor at sp+12 (the io-descriptor layout: uid 0, bid 0, sgid = group, wait 0, trig 0,
    mode 2, pad/scale/bias from r4/r5, then the source blob = the SRAM in format 0x120 and the destination blob = the
    OT_FMAP in format 0x110), calls entry 8 (the formatter) with (&blob, 0), then entry 3 with 0xffff0000|group, and
    returns 0. Words are set with movw (and movt above 16 bits) through r0/r1 (header) and r0/r1/r2 (blobs)."""
    C, H, W, Hp, Wp = u["C"], u["H"], u["W"], u["Hp"], u["Wp"]
    Hi, Wi = u.get("Hi", H), u.get("Wi", W)     # NPUC's internal (possibly one column wider) output
    hdr = [0, 0, group, 0, 0, 2]
    src = [0, u["sram_off"], FMT_CELL, 64, u["n"], u["c"], Hi, Wi, Wp - Wi, (Hp - Hi) * Wp, 0]
    # a tiled output (V: several output groups): each writes its rows at dst offset r0 * W, skipping the others' rows
    dst = [out_vector, u.get("dst_off", 0), FMT_CHW, 64, 1, C, Hi, Wi, (W - Wi) & 0xFFFFFFFF, u.get("dst_cs", 0), 0]
    ins = _prologue()
    ins += _fw_call(5, _load32(0, 0x10000 | group))
    off = 12
    for k, v in enumerate(hdr):
        r = k % 2
        ins += _load32(r, v) + [a_str(r, SP, off)]
        off += 4
    for r in (4, 5, 4):                     # pad 0, scale 1, bias 0
        ins.append(a_str(r, SP, off))
        off += 4
    for blob in (src, dst):                 # each blob cycles r0, r1, r2 from r0
        for k, v in enumerate(blob):
            r = k % 3
            ins += _load32(r, v) + [a_str(r, SP, off)]
            off += 4
    ins += _fw_call(8, [a_add_imm(0, SP, 12), a_mov_imm(1, 0)])
    ins += _fw_call(3, _load32(0, 0xFFFF0000 | group))
    ins += _epilogue()
    return _pack(ins)


def _pack(ins):
    return b"".join(struct.pack("<I", i) for i in ins)


def intrinsics(low, p=None):
    """the ARM intrinsic of every group (V): 32-byte stub for groups with DMAs, nothing for groups without, the
    output function for the unflatten group"""
    p = p or plan(low)
    out = []
    outs = {u["group"]: u for u in p.get("outputs", [p["unflatten"]])}
    for g in p["groups"]:
        if g["kind"] == "output":
            u = outs[g["index"]]
            out.append(output_intrinsic(u, u.get("v_out", p["v_out"]), g["index"]))
        elif g["kind"] == "dma":
            out.append(stub_intrinsic())
        else:
            out.append(b"")
    return out


# ---- the container -----------------------------------------------------------------------------------------------------
EV2SET_OP = 0x8809         # V: CMDQ set_reg 0x401e2024 ("ev2set"): one write per DMA batch, mask = trigger bits << 16


def batches_from_streams(io, streams, capped=(), wvecs=()):
    """the io descriptors' batch ids as the CMDQ streams declare them (V on every variant): each group's stream writes
    ev2set once per batch with the batch's trigger bits << 16, in uid order; a batch of more than 16 DMAs (trigger
    bits wrap) takes a second write for the wrapped ones of the same direction, which continues it, except in a group
    of `capped` (a tile group with a TRIM or a weight ring) once the batch holds 15 weight loads (loads from a vector
    of `wvecs`): stream.py's cap, cmdq.md 7b (V: tiling3 tr-c8, the input + 15 of 16 passes, the 16th a batch of
    its own). Returns the bids, or None when the streams do not match the DMAs (then plan()'s own rule stands: runs of
    one direction in one group, at most 15 DMAs; the streams contradict it only for some batches of exactly 16
    loads)."""
    writes = []
    for g, st in enumerate(streams):
        for op, val in decode_isa(st):
            if op == EV2SET_OP and val >> 16:
                writes.append((g, val >> 16))
    bids, k, bid, prev = [], 0, 0, None
    for g, mask in writes:
        n = bin(mask).count("1")
        take = io[k:k + n]
        if len(take) != n or any(x[2] != g or not (x[4] & mask) for x in take):
            return None
        d = take[0][9] < IDP              # direction: a store reads an SRAM vector
        dst = set((x[20], x[21]) for x in take)
        # V: the write after a full mask continues its batch (bc-241, ws-167-512, r_c256h40), unless one of its
        # loads targets an SRAM slot a load of the batch fills: a weight ring's next round is a new batch
        # (ws-64-512k5, r3-056)
        if not (prev and prev[0] == g and prev[1] == 16 and prev[2] == d and not (dst & prev[3])
                and not (g in capped and prev[4] >= 15)):
            bid += 1
            cnt, slots, nw = 0, set(), 0
        cnt += n
        slots |= dst
        nw += sum(1 for x in take if x[9] in wvecs)
        bids += [bid] * n
        k += n
        prev = (g, cnt, d, slots, nw)
    return bids if k == len(io) else None


def frame(low, streams, intrinsics_, weights, fix_npuc=False):
    """the whole NCP (V for every rule):
      header (32 words) | address vectors (16 B) | memory vectors (32 B) | group vectors (40 B) | io descriptors (124 B)
      | zero pad to hdr_size = io end rounded up to 16 | body: the intrinsics back to back (each at the running offset),
      zero pad to 16, then the CMDQ streams back to back | zero pad to 64 | the WEIGHT blocks, each at a 64-aligned
      offset (its av m_addr) | ncp_end | zero tail of 360 B per io descriptor."""
    p = plan(low, [len(w) for w in weights], fix_npuc=fix_npuc)
    G = len(p["groups"])
    if len(streams) != G or len(intrinsics_) != G:
        raise ValueError("%d groups planned, %d streams and %d intrinsics given" % (G, len(streams), len(intrinsics_)))
    av = [list(a) for a in p["av"]]
    mv, io = p["mv"], [list(x) for x in p["io"]]
    # the batches the streams declare; a cut plan keeps stream.py's (a group with 64 loads or more has them all
    # in one batch while its triggers come in sub-batches: cmdq.md 7d)
    capped = set(g for st in p.get("strata", []) if st["kind"] == "tiled"
                 and (any(o["op"] == "TRIM" for o in st["ops"])
                      or any(r < cdiv(low["layers"][li]["cout"], 16) for li, r in st["ring"].items()))
                 for g in st["groups"])
    wvecs = set(i for i, m in enumerate(mv) if m[0] == MV_WT)
    bids = None if p.get("keep_bids") else batches_from_streams(io, streams, capped, wvecs)
    if bids:
        for x, b in zip(io, bids):
            x[1] = b
    avo = 32 * 4
    mvo = avo + AV_SIZE * len(av)
    gvo = mvo + MV_SIZE * len(mv)
    iov = gvo + GV_SIZE * G
    hdr_size = rup(iov + IO_SIZE * len(io), 16)
    # body
    ioff, intr = [], b""
    for b in intrinsics_:
        ioff.append(len(intr))
        intr += b
    intr += bytes(rup(len(intr), 16) - len(intr))
    soff, isa = [], b""
    for s in streams:
        soff.append(len(intr) + len(isa))
        isa += s
        isa += bytes(rup(len(isa), 16) - len(isa))
    body = intr + isa
    end = hdr_size + len(body)
    wbytes = b""
    for vi, w in zip(p["v_w"], weights):
        at = rup(end + len(wbytes), 64)
        wbytes += bytes(at - end - len(wbytes))
        av[vi][1] = at
        wbytes += w
    if weights:     # V (g038, a 96-byte META_DATA last): the weights end on a 64-byte boundary
        wbytes += bytes(rup(end + len(wbytes), 64) - end - len(wbytes))
    ncp_end = end + len(wbytes)
    tail = RQ_PER_IO * len(io)
    h = [0] * 32
    h[0], h[1], h[2], h[3], h[4] = MAGIC1, HDR_VERSION, hdr_size, INTRINSIC_VERSION, NET_ID
    h[10], h[11], h[12], h[13], h[14], h[15] = avo, len(av), mvo, len(mv), gvo, G
    h[17], h[18], h[19], h[20], h[21], h[22] = hdr_size, len(body), iov, len(io), ncp_end, tail
    h[31] = MAGIC2
    out = bytearray(struct.pack("<32I", *h))
    for a in av:
        out += struct.pack("<4I", *a)
    for m in mv:
        out += struct.pack("<8I", *m)
    for g in range(G):
        isz, ssz = len(intrinsics_[g]), len(streams[g])
        out += struct.pack("<10I", g, 0, isz + ssz, 0, 0, 1, ioff[g], isz, soff[g], ssz)
    for x in io:
        out += struct.pack("<31I", *x)
    out += bytes(hdr_size - len(out))
    out += body + wbytes + bytes(tail)
    return bytes(out)


# ---- check: our frame around the oracle's own streams and weights ------------------------------------------------------
def region(ncp, off):
    """name the NCP region an offset falls in (for diff reports)"""
    h = ncp_header(ncp)
    avo, avc, mvo, mvc, gvo, gvc = h[10:16]
    iov, ioc = h[19], h[20]
    if off < 128:
        return "header w%d" % (off // 4)
    for name, base, n, sz in (("av", avo, avc, 16), ("mv", mvo, mvc, 32), ("gv", gvo, gvc, 40), ("io", iov, ioc, 124)):
        if base <= off < base + n * sz:
            return "%s %d +%d" % (name, (off - base) // sz, (off - base) % sz)
    if off < h[2]:
        return "hdr pad"
    if off < h[17] + h[18]:
        return "body +%#x" % (off - h[17])
    if off < h[21]:
        return "weights"
    return "tail"


def ncp_layout(low, ncp):
    """the IFM and OFM formats of a compiled NCP, checked against the rules above: a drop-in for ncpgen/npunet.py's
    ncp_layout (same dict; its ifm_addr(c, y, x, lay) with plane_h = Hp and plane_w = 2*Wp is this module's
    ifm_addr). Raises ValueError when the NCP's IN_FMAP / OT_FMAP vectors or its IFM DMAs are not the ones plan()
    predicts for this network (then the layout is not the one modelled)."""
    h, av, mv, _ = ncp_vectors(ncp)
    io = [list(struct.unpack_from("<31I", ncp, h[19] + 124 * i)) for i in range(h[20])]
    parts = ncp_parts(ncp)
    p = plan(low, [len(w) for w in parts["weights"]], strict=False)
    C, H, W = low["input"]["chw"]
    Co, Ho, Wo = low["layers"][-1]["out_chw"]
    L = cell(C, H, W)
    ins = [m for m in mv if m[0] == MV_IN]
    outs = [m for m in mv if m[0] == MV_OT]
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("%d IN_FMAP and %d OT_FMAP vectors (one each modelled)" % (len(ins), len(outs)))
    if tuple(ins[0][2:5]) != (W, H, C) or av[ins[0][7]][3] != L["size"]:
        raise ValueError("IN_FMAP %s, %d B; the rule wants (%d, %d, %d), %d B" % (ins[0][2:5], av[ins[0][7]][3],
                                                                                 W, H, C, L["size"]))
    want = [x for x in p["io"] if x[9] == ins[0][7]]
    got = [x for x in io if x[9] == ins[0][7]]
    if [x[9:] for x in got] != [x[9:] for x in want]:
        raise ValueError("the IFM DMAs are %s; the rule wants %s" % ([x[9:20] for x in got], [x[9:20] for x in want]))
    if tuple(outs[0][2:5]) != (Wo, Ho, Co) or av[outs[0][7]][3] != Co * Ho * Wo:
        raise ValueError("OT_FMAP %s, %d B: not a dense CHW %dx%dx%d" % (outs[0][2:5], av[outs[0][7]][3], Co, Ho, Wo))
    return {"ifm": {"format": "0x120 cells", "size": L["size"], "planes": L["n"] * L["pp"], "plane_h": L["Hp"],
                    "plane_w": 2 * L["Wp"], "idps": L["n"], "channels_per_idp": L["cpi"],
                    "proven": (C, H, W) == (8, 16, 16),
                    "how": "byte (c//2)*Hp*2Wp + (y//4)*4*2Wp + (x//8)*64 + (c%2)*32 + (y%4)*8 + x%8 (frame.ifm_addr; "
                           "V on the NPU for 8x16x16, V for the descriptors of 700+ compiles, G for the byte order "
                           "of other shapes until probed)"},
            "ofm": {"format": "CHW", "size": Co * Ho * Wo, "proven": (Co, Ho, Wo) in ((8, 16, 16), (8, 8, 8)),
                    "how": "dense CHW (the output intrinsic's formatter, 0x120 -> 0x110)"},
            "quant_mode": low.get("quant_mode", 1)}


# V (dossier/npu-compiler.md 9.11): the 8x16x16 IFM read by the NPU, byte address bit -> CHW index bit
MEASURED_8x16x16 = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 8, 6: 3, 7: 6, 8: 7, 9: 9, 10: 10}


def selftest():
    """the IFM layout against the NPU-measured 8x16x16 map and injectivity on other shapes; the A32 encoder"""
    n = 0
    C, H, W = 8, 16, 16
    for c in range(C):
        for y in range(H):
            for x in range(W):
                t = ifm_addr(c, y, x, C, H, W)
                idx = sum(1 << MEASURED_8x16x16[b] for b in range(11) if t >> b & 1)
                assert idx == (c * H + y) * W + x, (c, y, x, t, idx)
                n += 1
    for C, H, W in ((1, 5, 3), (3, 7, 9), (9, 16, 16), (20, 13, 17), (33, 16, 16), (64, 40, 24)):
        seen = set()
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    t = ifm_addr(c, y, x, C, H, W)
                    assert 0 <= t < ifm_size(C, H, W) and t not in seen
                    seen.add(t)
                    n += 1
    st = struct.unpack("<8I", stub_intrinsic())
    assert st[0] >> 16 == 0xE92D and st[7] >> 16 == 0xE8BD and st[0] & 0xFFFF == st[7] & 0x7FFF | 0x4000, st  # push {..lr} / pop {..pc}
    assert _imm12(0x1D000) == 0xA1D and _imm12(140) == 0x8C   # rotated immediates (ARM ARM A5.2.4)
    print("frame selftest: PASS (%d element addresses: the measured 8x16x16 map and injective layouts; the stub "
          "encoding)" % n)


def check(dirs):
    """our container around NPUC's own streams and weight blocks, against each variant's net.ncp; 0 when all exact"""
    n = ok = iok = 0
    fails = {}
    for d in dirs:
        lp, np_ = os.path.join(d, "lowered.json"), os.path.join(d, "net.ncp")
        name = os.path.basename(d.rstrip("/"))
        if not (os.path.isfile(lp) and os.path.isfile(np_)):
            print("%-28s skipped (no lowered.json/net.ncp)" % name)
            continue
        n += 1
        low = json.load(open(lp))
        ref = open(np_, "rb").read()
        parts = ncp_parts(ref)
        try:
            p = plan(low, [len(w) for w in parts["weights"]])
            intr = intrinsics(low, p)
            mine = frame(low, parts["streams"], intr, parts["weights"])
        except Exception as e:
            print("%-28s differs: %s: %s" % (name, type(e).__name__, e))
            fails[type(e).__name__] = fails.get(type(e).__name__, 0) + 1
            continue
        ie = intr == parts["intrinsics"]
        iok += ie
        if mine == ref:
            ok += 1
            print("%-28s exact (%d B)" % (name, len(ref)))
            continue
        o = next((i for i in range(min(len(mine), len(ref))) if mine[i] != ref[i]), min(len(mine), len(ref)))
        where = region(ref, o)
        print("%-28s differs at %#x (%s): ours %s oracle %s; len %d/%d; intrinsics %s" % (
            name, o, where, mine[o:o + 8].hex(), ref[o:o + 8].hex(), len(mine), len(ref), "exact" if ie else "differ"))
        key = where.split(" +")[0]
        fails[key] = fails.get(key, 0) + 1
    print("frame check: %d variants, %d whole files exact, intrinsics exact in %d%s" % (
        n, ok, iok, ("; differing regions: " + ", ".join("%s %d" % kv for kv in sorted(fails.items()))) if fails else ""))
    return 0 if n and ok == n else 2


def main(argv=None):
    """the command line (module docstring)"""
    ap = argparse.ArgumentParser(prog="frame.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="our container around NPUC's own streams and weights")
    p.add_argument("dirs", nargs="*")
    sub.add_parser("selftest", help="the layout against the NPU-measured map, the A32 encoder")
    p = sub.add_parser("plan", help="print the plan")
    p.add_argument("src", help="a variant dir or a lowered.json")
    p.add_argument("sizes", nargs="?", help="the WEIGHT block sizes, comma-separated")
    p = sub.add_parser("layout", help="the IFM/OFM sizes, and the byte address of element (c, y, x)")
    p.add_argument("dims", nargs="+", type=int, metavar="C H W [c y x]")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return check(a.dirs)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "plan":
        src = a.src
        lp = os.path.join(src, "lowered.json") if os.path.isdir(src) else src
        low = json.load(open(lp))
        ws = None
        if os.path.isdir(src) and os.path.isfile(os.path.join(src, "net.ncp")):
            ws = [len(w) for w in ncp_parts(open(os.path.join(src, "net.ncp"), "rb").read())["weights"]]
        if a.sizes:
            ws = [int(x) for x in a.sizes.split(",")]
        p = plan(low, ws)
        for c in p["chunks"]:
            print("chunk %-8s layer %2d live %s off %6d size %6d" % (c["kind"], c.get("layer", -1), c["live"], c["off"],
                                                                     c["size"]))
        for st in p.get("strata", []):
            print("stratum %s layers %s %s" % (st["kind"], st["layers"], " ".join(
                "%s %s" % (k, st[k]) for k in ("nh", "nw", "rows", "cols", "ring") if k in st)))
        for i, (av, m) in enumerate(zip(p["av"], p["mv"])):
            print("vec %2d type %d shape %s av size %d" % (i, m[0], m[2:5], av[3]))
        for x in p["io"]:
            print("io u%d b%d sg%d trig %#x src %s dst %s" % (x[0], x[1], x[2], x[4], x[9:20], x[20:31]))
        print("groups", [g["kind"] for g in p["groups"]], "unflatten", p["unflatten"])
    else:
        C, H, W = a.dims[:3]
        L = cell(C, H, W)
        print("IFM %dx%dx%d: %d B (%d IDP x %d planes x %d x %d), %d channels per IDP; OFM %d B dense CHW" % (
            C, H, W, L["size"], L["n"], L["pp"], L["Hp"], 2 * L["Wp"], L["cpi"], ofm_size(C, H, W)))
        if len(a.dims) >= 6:
            c, y, x = a.dims[3:6]
            print("element (%d, %d, %d): IFM byte %d, OFM byte %d" % (c, y, x, ifm_addr(c, y, x, C, H, W),
                                                                    ofm_addr(c, y, x, C, H, W)))
    return 0


if __name__ == "__main__":
    from ncpgen import frame as _pkg    # the package's copy of this module (one copy of every module)
    sys.exit(_pkg.main())
