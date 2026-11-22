#!/usr/bin/env python3
"""The WEIGHT blocks of an NCP, generated from npunet.py's lowered form (notes in docs/format/weights.md). Pure Python,
no vendor bytes: the packing below was measured against NPUC 1.36.1.a (the byte-exact oracle, oracle/npuc-local.sh)
by differential compiles.

  weights.py check <variant dir>...    compare with the oracle's WEIGHT blocks (lowered.json + net.ncp)
  weights.py show <lowered.json>       the geometry and block sizes this module computes

Interface: weight_blocks(low) -> list[bytes], every WEIGHT block in the NCP's order (padding included): a chain, one per
conv (and standalone ReLU) in layer order; a graph (eltwise, concat, standalone ReLU), one per conv, eltwise and ReLU
in NPUC's execution order (exec_order). layer_blocks(low) -> [(layer, kind, block)]; weight_block_sizes(low);
conv_block(L, quant_mode) -> (geometry, block); weight_layer_info(low) (sizes and geometry for stream.py);
weight_geometry(low) -> per conv what the io descriptors need (cores, rows per load, loads...); weight_layers(low)
(convs and ReLU dummy convs, layer order).

Graph layers (V: graph sweeps 98 + 47 + 66, ops-check 13, npunet-graph 10, all exact): a conv reading a concat kept in
SRAM sees each core holding [input 1's core share, input 2's, ...] (_segments); an eltwise's block is its CU_BIN
(eltwise_block); a standalone ReLU is a 1x1 dummy conv, all at the zero point when it reads a conv or a concat, else
NPUC's depthwise identity through its slotted walk (relu_identity_block).

The format (V: byte-exact on the sweep; docs/format/weights.md has the evidence):
  conv weights, OIHW uint8 as lowered (quant 1 stores them as is; quant 3 stores value - 128, i.e. XOR 0x80; the
  compression's zero is the stored weight zero point), are split
  - over input channels into `ncores` cores of `cpc` channels: cpc = 2 * ceil(Cin / 8), ncores = ceil(Cin / cpc),
    core c holds channels c*cpc .. c*cpc+cpc-1 (those >= Cin are absent);
  - over output channels into 4 groups of gs = 2 * ceil(Cout / 8) channels and nloads = ceil(gs / 4) loads; load l
    of group g holds channels g*gs + 4l .. g*gs + 4l + 3 (inside the group and < Cout; else an empty slot);
  - stride 2 (NPUC supports 1 and 2, square, only) is a space-to-depth: kernel ceil(K / 2), each channel pair split
    into 4 phases: virtual channel j = (ci // 2) * 8 + phase * 2 + ci % 2, phase = (y % 2) * 2 + x % 2, tap
    (y // 2, x // 2); a stride-1 conv is the same with one phase (j = ci).
  - a core's virtual channels into sub-batches of the same channel count (the last one shorter) when one output
    channel's positions would exceed NPUC's counter (511 values "non-zero" for it, i.e. stored byte != 0, or 1016
    values): getMinNumberOfBatches/updateBatchesForFile below, data dependent.
  Each core's region holds, for each load, two batches (groups 0-1, then 2-3) of R rows of 64 bytes; a batch is its
  sub-batches back to back, zero padded to R * 64 (R = the largest batch of the conv, all cores and loads). A
  sub-batch is a lead byte (its rows, _lead) and 8 chunks, slot t = (g % 2) * 4 + q (group g, lane q). A chunk is one
  output channel's weights over the sub-batch's positions p = j * VKH * VKW + ky' * VKW + kx': a big-endian u16
  header (bitmap bytes << 9 | non-zero count), the bitmap (bit p, LSB first, set when the stored byte is not the
  zero) and the non-zero bytes in p order. Block = ncores x nloads x 2 x R x 64 bytes, core-major.
"""
import argparse
import json
import os
import struct
import sys

if not __package__:     # run as a script: the package is this file's directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncpgen.common import (Memo, cdiv, concat_in_sram, exec_order as _exec_names, has_graph_ops, ncp_parts,  # noqa
                           topology, weights_digest)

ROW = 64            # bytes per row of a batch (the io descriptor's w)
SLOTS = 8           # chunks per batch
GROUPS = 4          # output channel groups (one per core's OFM share)
LANES = 4           # output channels per group and load
MAX_CORES = 4
IL = 2              # interleaved channels (NPUDevice2020::getInterleavedChannelCount)
MAX_NZV = 511       # non-zero values per chunk (NPUDevice2020::getMaxNzvInBatch; header field 9 bits)
MAX_VALUES = 0x3F8  # positions per sub-batch (LIRKernelData::checkMaxValuesInBatch)


def conv_geometry(L):
    """The packing geometry of one lowered conv layer."""
    cout, cin = L["cout"], L["in_chw"][0]
    kh, kw = L["k"]
    sh, sw = L.get("stride", [1, 1])
    cpc = L.get("_cpc") or 2 * cdiv(cin, 2 * MAX_CORES)
    ncores = L.get("_ncores") or cdiv(cin, cpc)
    gs = 2 * cdiv(cout, 2 * GROUPS)
    nloads = cdiv(gs, LANES)
    vkh, vkw = cdiv(kh, sh), cdiv(kw, sw)
    vcpc = cpc * sh * sw
    nbits = vcpc * vkh * vkw
    return {"cout": cout, "cin": cin, "k": [kh, kw], "stride": [sh, sw], "cpc": cpc, "ncores": ncores, "gs": gs,
            "nloads": nloads, "vk": [vkh, vkw], "vcpc": vcpc, "nbits": nbits}


def _positions(geo):
    """chunk position p (over the core's virtual channels: p = j * VKH * VKW + ky' * VKW + kx') -> (channel offset in
    the core, ky, kx), or None for a tap the stride's space-to-depth leaves empty. Stride 1: j = the channel."""
    kh, kw = geo["k"]
    sh, sw = geo["stride"]
    vkh, vkw = geo["vk"]
    nph = sh * sw
    pos = [None] * geo["nbits"]
    for cj in range(geo["cpc"]):
        for y in range(kh):
            for x in range(kw):
                ph = (y % sh) * sw + (x % sw)
                j = (cj // IL) * (IL * nph) + ph * IL + cj % IL
                pos[(j * vkh + y // sh) * vkw + x // sw] = (cj, y, x)
    return pos


def _chunk(vals, zero):
    """vals: one stored byte (or None = absent) per position -> header + bitmap + non-zero values"""
    bm_bytes = cdiv(len(vals), 8)
    bm = bytearray(bm_bytes)            # bit p (LSB first) set for a stored byte that is not the zero
    nz = bytearray()
    for p, v in enumerate(vals):
        if v is not None and v != zero:
            bm[p >> 3] |= 1 << (p & 7)
            nz.append(v)
    if bm_bytes >= 0x80 or len(nz) > MAX_NZV:
        # NPUC asserts here ("(nzvSize & 0xFE00) == 0", "(bmSize & 0x80) == 0") and aborts the compile
        raise ValueError("chunk of %d positions with %d non-zero values: NPUC aborts (max %d)"
                         % (len(vals), len(nz), MAX_NZV))
    return struct.pack(">H", bm_bytes << 9 | len(nz)) + bytes(bm) + bytes(nz)


def _stored(L, quant_mode):
    """the kernel bytes as NPUC stores them (KernelONFiller: quant 3 subtracts 128) and the compression's zero"""
    w = L["weights_oihw_u8"]
    zp = L.get("w_zp", 128)
    if quant_mode == 3:
        return [(v - 128) & 0xFF for v in w], (zp - 128) & 0xFF
    return list(w), zp


def _slot_channel(geo, l, b, t):
    """the output channel of load l, batch b, slot t, or None for an empty slot"""
    g, q = 2 * b + t // LANES, t % LANES
    lane = l * LANES + q
    o = g * geo["gs"] + lane
    return o if lane < geo["gs"] and o < geo["cout"] else None


def _min_batches(vals, K, nch):
    """NPU2020::NPULIRKernelMemoryData::getMinNumberOfBatches for one (core, output channel) (decompiled, @0x3c4ef8):
    vals = the stored bytes in position order (0 for an absent one). checkMaxValuesInBatch counts every value and the
    non-zero ones (non-zero = stored byte != 0, not != the zero point: NPUC passes 0) until one sub-batch would exceed
    1016 values or 511 non-zero (maxrec = the shortest such run, restarting the walk when a later run is shorter).
    Then cpb0 = floor(maxrec / K / 2) * 2, nb = ceil(nch / cpb0), kept if nch % nb == 0 or
    even_ceil(ceil(nch / nb)) * K < maxrec, else raised to the next divisor of nch. nch = the IFM ordering's channel
    count / its IDPs (getChannelCount / IDPSet size), not the layout's channels per core: returns the count."""
    maxrec = 0
    while True:
        total = nz = 0
        restart = False
        for v in vals:
            old = total
            total += 1
            if v:
                nz += 1
            if total <= MAX_VALUES and nz <= MAX_NZV:
                continue
            if maxrec == 0:
                maxrec = old
            elif total <= maxrec:
                maxrec = old
                restart = True
                break
            total = nz = 0
        if not restart:
            break
    if maxrec == 0:
        return 1
    cpb = maxrec // K // IL * IL
    if cpb == 0:
        # no channel pair fits the counter's run (a global-average-pool conv's long kernel): NPUC divides by zero, ARM's
        # udiv gives 0, and updateBatchesForFile's maximum ignores a 0: one sub-batch per core (V: models/gap_sweep.py,
        # the 5 quant-1 GAP convs of $NCPGEN_WORK/sweep/gaplim, exact with this; docs/format/weights.md)
        return 0
    nb = cdiv(nch, cpb)
    if nch % nb and cdiv(cdiv(nch, nb), IL) * IL * K < maxrec:
        return nb
    while nch % nb:
        nb += 1
    return nb


def _max_batches(geo):
    """tryBuildData's maxPossibleBatches (decompiled, @0x3c5794; its debug line 'kernel VKH x VKW numInputChannels N
    maxPossibleBatches = M' is in the log): the smallest divisor of the channels per IDP (numInputChannels / IDPs =
    getNumChannelsPerIDP = vcpc) >= max(ceil(K * vcpc / 1016), ceil(K * vcpc / 511)); 1 when no divisor is that large
    (V: the 64x8 and 32x16 global-pool convs of $NCPGEN_WORK/sweep/gaplim, where NPUC logs "maxPossibleBatches = 1",
    then fails its bitmap-size assertion, which _chunk's refusal mirrors)"""
    K = geo["vk"][0] * geo["vk"][1]
    vc = geo["vcpc"]
    bound = max(cdiv(K * vc, MAX_NZV), cdiv(K * vc, MAX_VALUES))
    while bound <= vc and vc % bound:
        bound += 1
    return bound if bound <= vc else 1


def sub_batches(L, quant_mode=1, geo=None, planner=False):
    """The split of each core's (virtual) channels into sub-batches: (count, virtual channels per sub-batch).
    updateBatchesForFile (@0x3c7be0): the count is the largest getMinNumberOfBatches over every (core, real output
    channel); tryBuildData then sets channels per sub-batch = even_ceil(ceil(vcpc / count)) for every batch, and
    aborts when a count exceeds maxPossibleBatches. The counter sees a stride's empty taps as the weight zero point
    (q1: 128, non-zero for it) and a channel past Cin as 0. NPUC runs this twice: for its planner (a
    StridedChannelOrdering over s^2 * even(Cin) channels) and for the code (the IFM buffer's ordering, s^2 * Cin
    channels); each divides its own channel count by the IDPs (cdiv), and either can abort. planner=True: the
    planner's split (its pack's size is the log's compWeight; V: q7i255, Cin 255)."""
    geo = geo or conv_geometry(L)
    w, zero = _stored(L, quant_mode)
    cin, (kh, kw) = geo["cin"], geo["k"]
    pos = _positions(geo)
    K = geo["vk"][0] * geo["vk"][1]
    if len(pos) <= MAX_NZV:
        return 1, geo["vcpc"]   # the counter cannot overflow (at most 511 values per output channel and core)
    s2 = geo["stride"][0] * geo["stride"][1]
    if s2 == 1:
        cdivs = [geo["vcpc"]]
    else:
        cdivs = [s2 * (cin + cin % IL) // _compact_idps(s2 * (cin + cin % IL)), s2 * cin // geo["ncores"]]
    nbs = [1] * len(cdivs)
    for c in range(geo["ncores"]):
        for o in range(geo["cout"]):
            vals = []
            for e in pos:
                if e is None:
                    vals.append(zero)
                    continue
                i = c * geo["cpc"] + e[0]
                vals.append(w[((o * cin + i) * kh + e[1]) * kw + e[2]] if i < cin else 0)
            for k, nch in enumerate(cdivs):
                nbs[k] = max(nbs[k], _min_batches(vals, K, nch))
    nb = nbs[0] if planner else nbs[-1]
    if nb > 1 and s2 == 1:
        _planner_check(geo, w, zero, pos, K)
    bound = _max_batches(geo)
    if max(nbs) > bound:
        raise ValueError("%d sub-batches > maxPossibleBatches %d: NPUC aborts" % (max(nbs), bound))
    return nb, cdiv(cdiv(geo["vcpc"], nb), IL) * IL


def _compact_idps(c):
    """NPUIDPAllocator::assignForChannelCountCompact (decompiled, @0x207410): g = ceil(c / 2) granules,
    u = ceil(g / 4) per IDP, ceil(g / u) IDPs"""
    g = cdiv(c, IL)
    return cdiv(g, cdiv(g, MAX_CORES))


def _planner_check(geo, w, zero, pos, K):
    """NPUC packs every kernel twice: first for its planner (PolicyMaker::makeCompressedKernel, whose size is the log's
    compWeight), then for the code. The planner's counter sees floor(Cin / 4) channels per core, not the layout's cpc
    (V: Cin 81..83 k5 and 226 k3 abort, 84..122 k5 and 234 k3 do not; the same floor(C/4) as the slotted path's core
    base): when no (core, output channel) then reaches the counter's limit, it keeps one sub-batch and its chunk of cpc
    channels overflows GenBitmasks' 511 non-zero values: NPUC aborts (G for the exact channels it walks)."""
    cin, (kh, kw) = geo["cin"], geo["k"]
    q4 = cin // MAX_CORES
    if q4 * K > MAX_VALUES:
        return
    for c in range(geo["ncores"]):
        for o in range(geo["cout"]):
            vals = []
            for e in pos[:q4 * K]:
                i = c * geo["cpc"] + e[0]
                vals.append(w[((o * cin + i) * kh + e[1]) * kw + e[2]] if i < cin else 0)
            if _min_batches(vals, K, q4) > 1:
                return
    for c in range(geo["ncores"]):
        for o in range(geo["cout"]):
            nz = 0
            for e in pos:
                i = c * geo["cpc"] + e[0]
                if i < cin and w[((o * cin + i) * kh + e[1]) * kw + e[2]] != zero:
                    nz += 1
            if nz > MAX_NZV:
                raise ValueError("NPUC's planner keeps one sub-batch (floor(Cin/4) = %d channels per core reach no "
                                 "limit) and a chunk of %d non-zero values aborts it" % (q4, nz))


def _lead(pos, body):
    """The lead byte of a sub-batch whose lead byte sits at offset pos of its batch (batches start on 64-byte rows)
    and whose chunks take body bytes. NPULIRKernelPack::tryBuildData first writes (64 + body) / 64, then rewrites
    it as the rows the sub-batch reaches from its start, floor((size + pos % 64 - 1) / 64) (size = body + 1; plain
    ceil(size / 64) at a row start), except when the first value is a multiple of 64: that one is kept (a quirk:
    the rewrite tests the lead byte, not the offset, modulo 64)."""
    lead0 = (ROW + body) // ROW
    if lead0 % ROW == 0:
        return lead0
    size, s = body + 1, pos % ROW
    return cdiv(size, ROW) if s == 0 else (size + s - 1) // ROW


def conv_groups(L, quant_mode=1, planner=False):
    """-> geometry, groups[core][load][b] as bytes (the sub-batches of one batch, each a lead byte + 8 chunks,
    concatenated; unpadded)"""
    geo = conv_geometry(L)
    cout, cin = geo["cout"], geo["cin"]
    kh, kw = geo["k"]
    if len(L["weights_oihw_u8"]) != cout * cin * kh * kw:
        raise ValueError("weights_oihw_u8 has %d values, want %d" % (len(L["weights_oihw_u8"]), cout * cin * kh * kw))
    if geo["stride"][0] != geo["stride"][1] or geo["stride"][0] not in (1, 2):
        raise ValueError("stride %s: NPUC supports 1 or 2, square, only" % geo["stride"])
    w, zero = _stored(L, quant_mode)
    nb, cpb = sub_batches(L, quant_mode, geo, planner)
    pos = _positions(geo)
    K = geo["vk"][0] * geo["vk"][1]
    subs = []
    c0 = 0
    while c0 < geo["vcpc"]:
        n = min(cpb, geo["vcpc"] - c0)
        subs.append(pos[c0 * K:(c0 + n) * K])
        c0 += n
    geo.update({"sub_batches": len(subs), "sub_channels": [len(x) // K for x in subs]})
    out = []
    for c in range(geo["ncores"]):
        loads = []
        for l in range(geo["nloads"]):
            pair = []
            for b in range(2):
                grp = bytearray()
                for pos in subs:
                    body = bytearray()
                    for t in range(SLOTS):
                        o = _slot_channel(geo, l, b, t)
                        vals = [None] * len(pos)
                        if o is not None:
                            for p, e in enumerate(pos):
                                if e is None:
                                    continue
                                i = c * geo["cpc"] + e[0]
                                if i < cin:
                                    vals[p] = w[((o * cin + i) * kh + e[1]) * kw + e[2]]
                        body += _chunk(vals, zero)
                    lead = _lead(len(grp), len(body))
                    if not 0 < lead < 0x100:
                        raise ValueError("a sub-batch of %d rows does not fit its lead byte" % lead)
                    grp += bytes([lead]) + body
                pair.append(bytes(grp))
            loads.append(pair)
        out.append(loads)
    return geo, out


_PACKED = Memo(64)      # conv_block's results by _pack_key (the packing is pure)


def _pack_key(L, quant_mode, planner):
    """everything conv_block's result depends on"""
    w = L["weights_oihw_u8"]
    return (L["cout"], L["in_chw"][0], tuple(L["k"]), tuple(L.get("stride", [1, 1])), L.get("_cpc"), L.get("_ncores"),
            L.get("w_zp", 128), quant_mode, bool(planner), len(w), weights_digest(w))


def conv_block(L, quant_mode=1, planner=False):
    """-> (geometry, the conv's WEIGHT block): conv_groups' batches padded to R rows each, R = the largest (memoised:
    the planner, the stream generator and the checks ask for the same convs many times)"""
    def pack():
        geo, groups = conv_groups(L, quant_mode, planner)
        R = cdiv(max(len(x) for core in groups for load in core for x in load), ROW)
        blk = bytearray()
        for core in groups:
            for load in core:
                for x in load:
                    blk += x + bytes(R * ROW - len(x))
        geo["rows_per_batch"] = R
        return geo, bytes(blk)
    return _PACKED.get(_pack_key(L, quant_mode, planner), pack)


def weight_blocks(low):
    """The exact bytes of each WEIGHT block, in the NCP's order. A chain (convs and pools): one per conv, layer order.
    A graph (an eltwise, a concat or a standalone ReLU): one per conv, eltwise and ReLU layer, in NPUC's execution
    order (layer_blocks). Raises ValueError where NPUC aborts (the checks NPUC asserts on: sub-batch counts above
    maxPossibleBatches in either pass, chunks over 511 non-zero values, the stride-1 planner abort)."""
    return [blk for _, _, blk in layer_blocks(low)]


def _output(low, cons):
    """the network's output layer: its only sink, else npu_output or the last layer"""
    sinks = [L["name"] for L in low["layers"] if L["name"] not in cons]
    return sinks[0] if len(sinks) == 1 else (low.get("npu_output") or low["layers"][-1]["name"])


def exec_order(low):
    """the layers in NPUC's order (common.exec_order); the WEIGHT memory vectors follow it (V: every graph compile)"""
    ins, cons, by, src = topology(low)
    return [by[n] for n in _exec_names(_output(low, cons), ins, cons, src)]


def _channels(name, low, by):
    return low["input"]["chw"][0] if name not in by else by[name]["out_chw"][0]


def _segments(name, low, ins, cons, by):
    """the channel segments a conv reading tensor `name` sees per core: [(first channel, count), ...]. One segment,
    except a concat in SRAM (each input split over the cores on its own: V cat16, cat16-3, cat16-chain, cat-32-16,
    g161, g169, g141), also through a pool that has one reader (V: c16_16/c16_48/c32_16-catpool-k3); a pool of one
    with several readers is stored and loaded back: normal order (V: g169, g141)."""
    L = by.get(name)
    if L is not None and L["kind"] == "pool" and len(cons.get(name, [])) == 1:
        return _segments(L["input"], low, ins, cons, by)
    if L is not None and L["kind"] == "concat" and concat_in_sram(name, _output(low, cons), by, ins, cons) \
            and not _concat_tiled(name, low, ins, cons):
        segs, base = [], 0
        for i in ins[name]:
            c = _channels(i, low, by)
            segs.append((base, c))
            base += c
        return segs
    return [(0, _channels(name, low, by))]


_POLICY_DEPTH = [0]


def _concat_tiled(name, low, ins, cons):
    """a concat one of whose producers or readers NPUC tiles stays in DRAM (frame.graph_policy, frame.md 6.11; the
    stream part's rule). While the policy itself is computed (it asks for the weight sizes), the structural rule
    alone (G: the policy then sizes those readers' blocks in the SRAM order)"""
    if _POLICY_DEPTH[0]:
        return False
    from ncpgen import frame
    _POLICY_DEPTH[0] += 1
    try:
        pol = frame.graph_policy(low)
    except Exception:
        return False
    finally:
        _POLICY_DEPTH[0] -= 1
    tiled = {x for st in pol["strata"] for x in st["layers"]}
    return any(x in tiled for x in ins[name] + cons.get(name, []))


def _permuted_conv(L, segs):
    """a conv over a segmented input as an equivalent conv: per core the segments' own core splits back to back
    (absent channels at the zero point)"""
    cpcs = [2 * cdiv(c, 2 * MAX_CORES) for _, c in segs]
    ncores = max(cdiv(c, cp) for (_, c), cp in zip(segs, cpcs))
    cpc = sum(cpcs)
    order = []
    for core in range(ncores):
        for (b0, c), cp in zip(segs, cpcs):
            for j in range(cp):
                i = core * cp + j
                order.append(b0 + i if i < c else -1)
    cin, cout = L["in_chw"][0], L["cout"]
    kk = L["k"][0] * L["k"][1]
    w, zp = L["weights_oihw_u8"], L.get("w_zp", 128)
    w2 = []
    for o in range(cout):
        for i in order:
            w2.extend(w[(o * cin + i) * kk:(o * cin + i + 1) * kk] if i >= 0 else [zp] * kk)
    return dict(L, in_chw=[len(order)] + list(L["in_chw"][1:]), weights_oihw_u8=w2, _cpc=cpc, _ncores=ncores)


def _eltwise_regions(C):
    """(cores, channels per core, bytes per array, bytes per core region) of a C-channel eltwise's META_DATA"""
    planes = cdiv(C, 2)
    ppc = cdiv(planes, MAX_CORES)
    cores = cdiv(planes, ppc)
    n = cdiv(C, cores)
    arr = cdiv(n, 4) * 4
    return cores, n, arr, cdiv(3 * arr, 32) * 32


def eltwise_meta_size(C):
    """NPUC's META_DATA chunk of a C-channel eltwise, SRAM bytes per core: one core's region of its block (V: 32 up to
    32 channels, 64 for 40..64 (res40, res64), 96 for 128, 192 for 256, 384 for 512, 544 for 720: graph-split)"""
    return _eltwise_regions(C)[3]


def eltwise_block(L, dram=False):
    """An eltwise add's WEIGHT block (NPUC's CU_BIN / META_DATA, V on every graph compile): per core of the input's
    layout (ceil(C/2) planes, ceil(planes/4) per core; 4 regions for exactly 4 channels when an operand is loaded
    from DRAM (dram: the network input or a concat, V: rc4-*, g120, g134; both in SRAM: 2, V: g513)) three arrays, the fractional
    lengths of input 1, input 2 and the output (lowered "fl"), one byte per channel of the core (C split evenly:
    ceil(C / cores) each, the rest on the last), each array padded to 4 bytes, the core's region to 32."""
    C = L["in_chw"][0]
    cores, n, arr, reg = _eltwise_regions(C)
    regions = MAX_CORES if C == 4 and dram else cores
    out = bytearray()
    for c in range(regions):
        cnt = max(0, min(n, C - c * n))
        r = bytearray()
        for f in L["fl"][:3]:
            r += bytes([f & 0xFF]) * cnt + bytes(arr - cnt)
        out += r + bytes(reg - len(r))
    return bytes(out)


def _umod(a, b):
    return a if b == 0 else a % b      # ARM udiv by 0 gives 0, so a % 0 = a


def relu_identity_block(C, quant_mode=1):
    """A standalone ReLU's dummy conv when it reads a pool, an eltwise or a ReLU (V: 196 compiles, C 1..64): NPUC packs
    it as a depthwise kernel through IterateOverAllValues' slotted path (@0x2c48b8): for output o (o = -1 for a lane
    past the group size) and core c, the walk visits channels w = c * floor(C/4) + 2 * pos + ci + 4 * ((o mod
    floor(C/4)) // 4) (pos over the core's channel pairs) and emits a position for each w with
    (w & ~3) mod floor(C/4) == (o & ~3) mod floor(C/4): the value 1 where w == o (a real output), else the zero point.
    (NPUC's bug: floor(C/4) is not the core split, so for C not a multiple of 8 windows overlap or miss channels.)
    The batches are a normal conv's (Cin = Cout = C, 1x1)."""
    geo = conv_geometry({"cout": C, "in_chw": [C, 1, 1], "k": [1, 1], "stride": [1, 1]})
    q4 = C // MAX_CORES
    zero = 0 if quant_mode == 3 else 128
    one = 1     # the stored byte 1 in both modes (V: g-pool-relu-q1, where 1 is u8 1, i.e. -127 against zp 128)
    groups = []
    for c in range(geo["ncores"]):
        for l in range(geo["nloads"]):
            for b in range(2):
                body = bytearray()
                for t in range(SLOTS):
                    g, q = 2 * b + t // LANES, t % LANES
                    lane = l * LANES + q
                    o = g * geo["gs"] + lane
                    real = lane < geo["gs"] and o < C
                    if lane >= geo["gs"]:
                        o = 0xFFFFFFFF
                    add = 4 * (_umod(o, q4) // 4)
                    key = _umod(o & 0xFFFFFFFC, q4)
                    vals = []
                    for pos in range(cdiv(geo["cpc"], IL)):
                        for ci in range(IL):
                            w = c * q4 + IL * pos + ci + add
                            if real and w == o and w < C:
                                vals.append(one)
                            elif _umod(w & 0xFFFFFFFC, q4) == key:
                                vals.append(zero)
                    body += _chunk(vals, zero)
                groups.append(bytes([_lead(0, len(body))]) + bytes(body))
    R = cdiv(max(len(x) for x in groups), ROW)
    return b"".join(x + bytes(R * ROW - len(x)) for x in groups)


def _relu_zero_conv(L):
    c = L["cout"]
    return {"kind": "conv", "name": L.get("name", "relu"), "cout": c, "in_chw": [c, 1, 1], "k": [1, 1],
            "stride": [1, 1], "w_zp": 128, "weights_oihw_u8": [128] * (c * c)}


def layer_blocks(low):
    """[(layer name, kind, WEIGHT block)] in the NCP's order"""
    q = low.get("quant_mode", 1)
    if not has_graph_ops(low):
        return [(L["name"], L["kind"], conv_block(L if L["kind"] == "conv" else _relu_zero_conv(L), q)[1])
                for L in low["layers"] if L["kind"] in ("conv", "relu")]
    ins, cons, by, src = topology(low)
    res = []
    for L in exec_order(low):
        k = L["kind"]
        if k == "conv":
            segs = _segments(L["input"], low, ins, cons, by)
            res.append((L["name"], k, conv_block(L if len(segs) == 1 else _permuted_conv(L, segs), q)[1]))
        elif k == "eltwise":
            # an operand its group loads from DRAM: the network input, or a concat (one an eltwise reads stays in DRAM)
            dram = any(i == src or (i in by and by[i]["kind"] == "concat") for i in ins[L["name"]])
            res.append((L["name"], k, eltwise_block(L, dram)))
        elif k == "relu":
            res.append((L["name"], k, relu_identity_block(L["cout"], q) if _relu_identity(L, by)
                        else conv_block(_relu_zero_conv(L), q)[1]))
    return res


def _relu_source(L, by):
    """the first producer before a ReLU that is not itself a ReLU (None: the network input)"""
    p = by.get(L["input"])
    while p is not None and p["kind"] == "relu":
        p = by.get(p["input"])
    return p


def _relu_identity(L, by):
    """a standalone ReLU's block in a graph: the identity (relu_identity_block) or every weight at the zero point. A
    ReLU reading what a conv wrote (a conv, a concat): the zero point (V: grelu r6c8/16, rlc8, cat-relu, cat16-relu,
    g008, g029, g147, g161); reading a pool or an eltwise: the identity through the depthwise path; a ReLU of a ReLU
    takes the inner one's form (V: relu<concat zero, g555; relu<eltwise, relu<pool identity, g002, g016); the network
    input: G (taken as the identity)"""
    p = _relu_source(L, by)
    return not (p is not None and p["kind"] in ("conv", "concat"))


def relu_rows(L, by, quant_mode=1):
    """rows of 64 B per weight batch of the block a graph's standalone ReLU L gets (layer_blocks): the rows its
    conv ops and its WEIGHT chunk count (V: the identity's 1 where the zero-point conv has 4: 720 and 768 channels,
    graph-split cut-elt and cut-relu; the same 1 up to 128 channels)"""
    if not _relu_identity(L, by):
        return conv_block(_relu_zero_conv(L), quant_mode)[0]["rows_per_batch"]
    C = L["cout"]
    geo = conv_geometry({"cout": C, "in_chw": [C, 1, 1], "k": [1, 1], "stride": [1, 1]})
    return len(relu_identity_block(C, quant_mode)) // (geo["ncores"] * geo["nloads"] * 2 * ROW)


def _mf_conv(L, chans):
    """The planner's second ("MF") kernel of a conv reading a concat: the channel table that
    DynamicBuffer::setChannelMFAccIndex builds (decompiled, @0x2237d0): concat input t = (a_t planes before it, b_t =
    ceil(C_t/2) planes); every core gets the rows ceil(a_t/n) .. ceil((a_t+b_t)/n) - 1 of each input, 2 channel slots a
    row, filled with the kernel channels 2 a_t, 2 a_t + 1, ... core after core (the channel index is used as is: it
    runs into the next input's channels when an input does not fill its rows); n = the IDPs assignForChannelCountCompact
    gives the whole concat (P = ceil(C/2) planes: ceil(P / ceil(P/4)))."""
    C = sum(chans)
    P = cdiv(C, 2)
    n = cdiv(P, cdiv(P, MAX_CORES))
    per = [[] for _ in range(n)]
    a = 0
    for c in chans:
        b = cdiv(c, 2)
        v = 2 * a
        for core in range(n):
            for _ in range(cdiv(a, n), cdiv(a + b, n)):
                for _e in range(IL):
                    per[core].append(v if v < C else -1)
                    v += 1
        a += b
    rows = max(len(x) for x in per)
    order = []
    for core in range(n):
        order += per[core] + [-1] * (rows - len(per[core]))
    cin, cout = L["in_chw"][0], L["cout"]
    kk = L["k"][0] * L["k"][1]
    w, zp = L["weights_oihw_u8"], L.get("w_zp", 128)
    w2 = []
    for o in range(cout):
        for i in order:
            w2.extend(w[(o * cin + i) * kk:(o * cin + i + 1) * kk] if i >= 0 else [zp] * kk)
    return dict(L, in_chw=[len(order)] + list(L["in_chw"][1:]), weights_oihw_u8=w2, _cpc=rows, _ncores=n)


def _planner_concat(L, by, ins, low):
    """the concat input channel counts when makeCompressedKernel sees a concat before this conv (its first input is a
    Concat, or a Pool of one: one level), else None"""
    p = by.get(L["input"])
    if p is not None and p["kind"] == "pool":
        p = by.get(p["input"])
    if p is None or p["kind"] != "concat":
        return None
    return [_channels(i, low, by) for i in ins[p["name"]]]


def weight_planning_sizes(low):
    """One per WEIGHT block, in the NCP's order: the size NPUC's planner computes for it (the log's compWeight, which
    sets the WEIGHT SRAM chunk the allocator asks for; weights.md section 7). Equal to the block except:
    - a conv whose first input is a concat (or a pool of one): the larger of the normal channel order's pack and the
      planner's MF pack (_mf_conv), the MF one only when every concat input has an even channel count (V: 224 of 224
      concat readers, incl. g103, g161, g167);
    - a standalone ReLU's identity dummy conv: its block counted on 4 cores (C <= 16: 512 B);
    - an eltwise: no compWeight (its CU_BIN is a fixed 128 B in SRAM): its block size;
    - stride 2: the planner's own sub-batch split (sub_batches(planner=True); differs for odd Cin)."""
    q = low.get("quant_mode", 1)

    def plan_len(L, blk):
        if L.get("stride", [1, 1]) == [1, 1]:
            return len(blk)
        return len(conv_block(L, q, True)[1])

    if not has_graph_ops(low):
        convs = [L for L in low["layers"] if L["kind"] == "conv"]
        return [plan_len(L, b) for L, b in zip(convs, weight_blocks(low))]
    ins, cons, by, src = topology(low)
    out = []
    for name, kind, blk in layer_blocks(low):
        L = by[name]
        if kind == "conv":
            segs = _segments(L["input"], low, ins, cons, by)
            normal = plan_len(L, blk) if len(segs) == 1 else len(conv_block(L, q, True)[1])
            chans = _planner_concat(L, by, ins, low)
            if chans is not None and all(c % 2 == 0 for c in chans):
                normal = max(normal, len(conv_block(_mf_conv(L, chans), q)[1]))
            out.append(normal)
        elif kind == "relu":
            p = _relu_source(L, by)
            if p is not None and p["kind"] in ("conv", "concat"):
                out.append(len(blk))
            else:
                geo = conv_geometry({"cout": L["cout"], "in_chw": [L["cout"], 1, 1], "k": [1, 1], "stride": [1, 1]})
                out.append(len(blk) // geo["ncores"] * MAX_CORES)
        else:
            out.append(len(blk))
    return out


def weight_layer_info(low):
    """Per WEIGHT block in the NCP's order: {name, kind, size (the block), planning_size (the planner's, compWeight),
    and for a conv or ReLU the emitted block's geometry: ncores, nloads, rows_per_batch (R: io h = 2R), sub_batches,
    sub_channels, plus planning_rows = planning_size / (planning cores x nloads x 2 x 64)}."""
    q = low.get("quant_mode", 1)
    lb = layer_blocks(low)
    ps = weight_planning_sizes(low)
    by = {L["name"]: L for L in low["layers"]}
    graph = has_graph_ops(low)
    if graph:
        ins, cons, _, src = topology(low)
    out = []
    for (name, kind, blk), psz in zip(lb, ps):
        d = {"name": name, "kind": kind, "size": len(blk), "planning_size": psz}
        if kind in ("conv", "relu"):
            L = by[name]
            if kind == "relu":
                G = _relu_zero_conv(L)
            elif graph:
                segs = _segments(L["input"], low, ins, cons, by)
                G = L if len(segs) == 1 else _permuted_conv(L, segs)
            else:
                G = L
            geo = conv_block(G, q)[0]
            pc = MAX_CORES if kind == "relu" and psz != len(blk) else geo["ncores"]
            d.update({k: geo[k] for k in ("ncores", "nloads", "rows_per_batch", "sub_batches", "sub_channels")})
            d["planning_rows"] = psz // (pc * geo["nloads"] * 2 * ROW)
        out.append(d)
    return out


def weight_layers(low):
    """The layers that own a WEIGHT block, as convs (sizes; layer order): every conv, and a standalone ReLU/ReLU6 as
    its 1x1 dummy conv with every weight at the zero point (the same size as the identity form). Kept for the stream
    part; the bytes come from layer_blocks."""
    out = []
    for L in low["layers"]:
        if L["kind"] == "conv":
            out.append(L)
        elif L["kind"] == "relu":
            out.append(_relu_zero_conv(L))
    return out


def weight_block_sizes(low):
    """the byte size of each WEIGHT block, in the NCP's order"""
    return [len(b) for b in weight_blocks(low)]


def weight_geometry(low):
    """Per conv: the packing geometry plus what the io descriptors that load it show (V on the sweep): one load per
    output group set, `fmt 288 bpp 128 n1 c<ncores> h<2 * rows_per_batch> w64` at src offset l * 2 * R * 64, the
    per-core stride in the block = nloads * 2 * R * 64."""
    q = low.get("quant_mode", 1)
    res = []
    for L in weight_layers(low):
        geo, blk = conv_block(L, q)
        R = geo["rows_per_batch"]
        geo.update({"name": L["name"], "size": len(blk), "load_h": 2 * R, "load_bytes_per_core": 2 * R * ROW,
                    "core_stride": geo["nloads"] * 2 * R * ROW,
                    "load_src_offsets": [l * 2 * R * ROW for l in range(geo["nloads"])]})
        res.append(geo)
    return res


# --- the check against NPUC's compiles ---
WEIGHT_ASSERTS = ("NPULIRKernelPack", "NPULIRKernelMemoryData", "_strideWidth == _strideHeight")


def _oracle_refused(vd):
    """(the compile log's assertion, whether it is the weight packer's), when NPUC aborted (no net.ncp)"""
    try:
        for line in open(os.path.join(vd, "log.txt"), errors="replace"):
            if "assertion" in line and "failed" in line:
                return line.strip().split("assertion", 1)[1][:120], any(k in line for k in WEIGHT_ASSERTS)
    except OSError:
        pass
    return None, False


def check(dirs):
    """Compare weight_blocks(lowered.json) with the WEIGHT blocks of each variant's net.ncp (no compiling here).
    A variant NPUC aborted on (no net.ncp, an assertion in log.txt) counts as matching when we refuse it too."""
    ok = bad = refused = skipped = 0
    for vd in dirs:
        name = os.path.basename(os.path.normpath(vd))
        try:
            low = json.load(open(os.path.join(vd, "lowered.json")))
        except (OSError, ValueError) as e:
            print("%-28s skip (no lowered.json)" % name)
            skipped += 1
            continue
        ncp = os.path.join(vd, "net.ncp")
        try:
            gen, err = weight_blocks(low), None
        except ValueError as e:
            gen, err = None, str(e)
        if not os.path.exists(ncp):
            why, ours = _oracle_refused(vd)
            if why and ours and err:
                refused += 1
                print("%-28s refused, as NPUC (%s | NPUC: %s)" % (name, err, why))
            elif why and ours:
                bad += 1
                print("%-28s differs: NPUC's weight packer aborted (%s), we generate" % (name, why))
            else:
                skipped += 1
                print("%-28s skip (no net.ncp: NPUC failed outside the weights%s)" % (name, ": " + why if why else ""))
            continue
        ref = ncp_parts(open(ncp, "rb").read())["weights"]
        if err:
            bad += 1
            print("%-28s differs: we refuse (%s), NPUC compiled it" % (name, err))
            continue
        if gen == ref:
            ok += 1
            print("%-28s exact  %d block(s) %s B" % (name, len(gen), "+".join(str(len(b)) for b in gen)))
            continue
        bad += 1
        why = []
        if len(gen) != len(ref):
            why.append("%d blocks, oracle %d" % (len(gen), len(ref)))
        for k, (g, r) in enumerate(zip(gen, ref)):
            if g != r:
                first = next((i for i in range(min(len(g), len(r))) if g[i] != r[i]), min(len(g), len(r)))
                why.append("block %d: %d B vs oracle %d B, first diff at %#x" % (k, len(g), len(r), first))
        print("%-28s differs: %s" % (name, "; ".join(why)))
    print("total: %d/%d exact, %d differ, %d refused as NPUC refuses, %d skipped"
          % (ok, ok + bad, bad, refused, skipped))
    return bad == 0


def main(argv=None):
    """the command line (module docstring)"""
    ap = argparse.ArgumentParser(prog="weights.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="our WEIGHT blocks against NPUC's compiles")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--loose", action="store_true", help=argparse.SUPPRESS)  # no effect any more (old command lines)
    p = sub.add_parser("show", help="the geometry and block sizes of each conv")
    p.add_argument("lowered")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return 0 if check(a.dirs) else 1
    for g in weight_geometry(json.load(open(a.lowered))):
        print(json.dumps(g))
    return 0


if __name__ == "__main__":
    from ncpgen import weights as _pkg    # the package's copy of this module (one copy of every module)
    sys.exit(_pkg.main())
