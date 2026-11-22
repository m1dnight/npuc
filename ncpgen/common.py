"""What the parts of ncpgen share: NPUC's integer and float32 arithmetic, a network's topology as NPUC reads it, the
CMDQ instruction encoding, the sections of an NCP (docs/format/), and memoisation of the pure planners."""
import collections
import copy
import hashlib
import heapq
import json
import struct


# ---- arithmetic -----------------------------------------------------------------------------------------------------
def cdiv(a, b):
    """ceil(a / b) for integers"""
    return -(-a // b)


def rup(a, b):
    """a rounded up to a multiple of b"""
    return cdiv(a, b) * b


def f32(x):
    """x rounded to float32 (NPUC computes its scales in float32)"""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_hex(x):
    """x's float32 bits as lowered.json writes a scale ("0x3c000000")"""
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


def hex_f32(s):
    """a lowered.json scale ("0x3c000000") -> its float32 value"""
    return struct.unpack("<f", struct.pack("<I", int(s, 16)))[0]


def scale_value(v):
    """a scale given by its float32 bits ("0x3c000000") or as a number -> the float32 value"""
    return hex_f32(v) if isinstance(v, str) else f32(v)


def multiplier(in_s, w_s, out_s):
    """NPUC's real multiplier f32(f32(in * w) / out), every operand and operation in float32 (V,
    NNGraphNodeScaleONFiller::setScale, libnpuc_frontend 0x9da64)"""
    return f32(f32(f32(in_s) * f32(w_s)) / f32(out_s))


def requant(m):
    """(m', slope, shift) from a multiplier m > 0 as setScale goes on (V): m doubled while < 0.5, at most 7 times, into
    m'; slope = round half away (m' * 256) (fcvtas, 0x9dd40: m' * 256 is exact in float32, + 0.5 exact in double);
    shift = doublings + 8"""
    n = 0
    while m < 0.5 and n < 7:
        m *= 2
        n += 1
    return m, int(m * 256.0 + 0.5), n + 8


# ---- a network's topology (lowered.json) -----------------------------------------------------------------------------
GRAPH_KINDS = ("eltwise", "concat", "relu")      # the layer kinds of npuc-net 2 (docs/harness-ops.md)


def has_graph_ops(low):
    """an eltwise, a concat or a standalone ReLU among the layers"""
    return any(L["kind"] in GRAPH_KINDS for L in low["layers"])


def is_graph(low):
    """a network the graph planners take: one with graph ops, or whose layers do not form a chain (npuc-net 2)"""
    prev = low["input"]["name"]
    for L in low["layers"]:
        if L["kind"] in GRAPH_KINDS or L.get("input") != prev or len(L.get("inputs", [prev])) != 1:
            return True
        prev = L["name"]
    return False


def topology(low):
    """(ins, cons, by, src): each layer's inputs and each tensor's readers (layer names), the layers by name, the
    network input's name"""
    src = low["input"]["name"]
    by = {L["name"]: L for L in low["layers"]}
    ins = {L["name"]: list(L["inputs"]) if "inputs" in L else [L["input"]] for L in low["layers"]}
    cons = {}
    for L in low["layers"]:
        for i in ins[L["name"]]:
            cons.setdefault(i, []).append(L["name"])
    return ins, cons, by, src


def net_outputs(low):
    """the network's output layers in NPUC's order: lowered.json's "npu_outputs" when it has several (every layer no
    layer reads, in execution order: docs/format/frame.md 12), else its one output (npu_output, or the last layer)"""
    if low.get("npu_outputs"):
        return list(low["npu_outputs"])
    return [low.get("npu_output") or low["layers"][-1]["name"]]


def exec_order(out, ins, cons, src):
    """NPUC's layer order, the names (V: its NODE ORDER logs, every graph compile of the sweeps with one, 278; resnet8
    and its variants: an eltwise's input order does not change it, the layer list's order does). NPUC numbers its
    graph's nodes breadth first from the network input, each tensor's readers in layer-list order (cons as topology()
    builds it; run/ir_graph_init.txt), then runs, of the layers whose inputs are all done, the one with the highest
    number. Only the layers the output `out` depends on are run (out: a layer name, or a list of the outputs' names
    when the network has several: V, the outputs sweep, frame.md 12)."""
    outs = [out] if isinstance(out, str) else list(out)
    nid, q, seen = {}, collections.deque([src]), {src}
    while q:
        for c in cons.get(q.popleft(), []):
            if c not in seen:
                seen.add(c)
                nid[c] = len(nid) + 1
                q.append(c)
    need, stack = set(outs), list(outs)
    while stack:
        for t in ins.get(stack.pop(), []):
            if t != src and t not in need:
                need.add(t)
                stack.append(t)
    done, order = {src}, []
    ready = [(-nid[n], n) for n in need if all(x in done for x in ins[n])]
    heapq.heapify(ready)
    queued = {n for _, n in ready}
    while ready:
        _, n = heapq.heappop(ready)
        order.append(n)
        done.add(n)
        for c in cons.get(n, []):
            if c in need and c not in queued and all(x in done for x in ins[c]):
                queued.add(c)
                heapq.heappush(ready, (-nid[c], c))
    return order


SRAM_PER_IDP = 0x40000    # V: one IDP's SRAM, the bound of every NPUC memory estimate


def fm_memory(chw):
    """a feature map's per-IDP memory as NPUC's estimates count it (PFeatureShape::requiredMemory, frame.md 6.10):
    ceil(C / 2) planes rounded up to the IDPs used, each rup(H, 4) x 2 rup(W, 8) bytes"""
    C, H, W = chw
    planes = cdiv(C, 2)
    n = cdiv(planes, cdiv(planes, 4))
    return 2 * rup(W, 8) * rup(H, 4) * (rup(planes, n) // n)


def feeds_eltwise(t, by, cons):
    """an eltwise reads tensor t, or reads one of the one-input layers (conv, pool, ReLU) below it"""
    for r in cons.get(t, []):
        if by[r]["kind"] == "eltwise" or (by[r]["kind"] in ("conv", "pool", "relu") and feeds_eltwise(r, by, cons)):
            return True
    return False


def concat_in_sram(name, out, by, ins, cons):
    """NPUC builds concat `name` in SRAM, its producers writing their channels straight into one CONCAT_STORAGE chunk
    at the top of SRAM, each core holding [input 1's planes, input 2's, ...] (V: cat16, cat16-3, cat16-chain,
    cat16-relu, cat-32-16, g021, g118; in DRAM: cat2, cat-8-16, cat-4-12, cat-16-8, cat16-pool, cat16-out, g029,
    g144), when every input is a pool or a conv with a multiple of 16 channels (a conv writes 16-channel OFMs), the
    concat is not the network output `out` (a name, or the outputs' names), no eltwise reads it and it does not fork
    into branches an eltwise joins (V: g012, a concat whose one reader forks, stays in SRAM)"""
    if (name == out if isinstance(out, str) else name in out) or any(i not in by for i in ins[name]):
        return False
    if not all(by[i]["kind"] == "pool" or (by[i]["kind"] == "conv" and by[i]["out_chw"][0] % 16 == 0)
               for i in ins[name]):
        return False
    if fm_memory(by[name]["out_chw"]) >= SRAM_PER_IDP:
        # NPUC's layer-block forwarding ("Concat in SRAM", LayerBlockFMSharingCalculator::tryProcessInternalNode):
        # the concat's own memory is held for the whole block, so a block node finds SRAM - that < 1 left: "is too
        # fat", the concat goes to DRAM (V: graph-split cat96-64, 128 channels at 96 x 96)
        return False
    readers = cons.get(name, [])
    if any(by[r]["kind"] == "eltwise" for r in readers):
        return False
    if len(readers) > 1 and feeds_eltwise(name, by, cons):
        return False
    # read by several layers, the concat stays in SRAM only as the root of a later concat's branches that all its
    # readers start (a layer-block its forwarding keeps in SRAM: V g021, g141, g169, g172, t1024, t1029, t2023, t2032);
    # readers that never meet again (detection heads, each a network output) load it from DRAM (V: outputs-random
    # m004, m016, m039), as when one of them leaves the later concat's branches (V: outputs-unseen2 m2028)
    return len(readers) < 2 or any(all(bulb_root(i, by, cons) == name for i in ins[k]) and
                                   set(readers) <= branch_heads(k, name, by, cons)
                                   for k in by if by[k]["kind"] == "concat" and k != name)


def branch_heads(k, root, by, cons):
    """the readers of root that start one of concat k's branches (the layers on the way from root to k: one input,
    one reader each), and k itself when it reads root directly"""
    heads = set()
    for i in by[k].get("inputs", [by[k].get("input")]):
        t = i
        while t in by and by[t]["kind"] in ("conv", "pool", "relu") and len(cons.get(t, [])) == 1:
            if by[t]["input"] == root:
                heads.add(t)
                break
            t = by[t]["input"]
        if i == root:
            heads.add(k)
    return heads


def bulb_root(t, by, cons):
    """where a concat's branch through tensor t starts: back through one-input layers (conv, pool, ReLU) read by one
    layer each, to a tensor several layers read (or the network input)"""
    while t in by and by[t]["kind"] in ("conv", "pool", "relu") and len(cons.get(t, [])) == 1:
        t = by[t]["input"]
    return t


# ---- CMDQ instructions (docs/format/cmdq.md 1) ----------------------------------------------------------------------
CMDQ, MTC, NFU, INTC = 0x401E0000, 0x40120000, 0x40160000, 0x40100000
REG_BLOCKS = {0x8: CMDQ, 0x9: MTC, 0xA: NFU, 0xB: INTC}     # a set_reg opcode's block -> its register window


def set_reg_op(addr):
    """the set_reg opcode writing register addr: block << 12 | word index (V, NPUC's compile log)"""
    for blk, base in REG_BLOCKS.items():
        if base <= addr < base + 0x4000:
            return blk << 12 | (addr - base) >> 2
    raise ValueError("no block for %#x" % addr)


def set_reg_addr(op):
    """the register a set_reg opcode writes, or None for another instruction (nop, waiti3 0x3a.., wait3 0x2b..)"""
    if op >> 12 in REG_BLOCKS and op >> 8 not in (0x3A, 0x2B):
        return REG_BLOCKS[op >> 12] + (op & 0xFFF) * 4
    return None


def encode_isa(ins):
    """[(opcode, value)] -> the stream's bytes: 6 per instruction, u32 value then u16 opcode (V)"""
    return b"".join(struct.pack("<IH", v, op) for op, v in ins)


def decode_isa(b):
    """a stream's bytes -> [(opcode, value)]"""
    return [struct.unpack_from("<IH", b, 6 * i)[::-1] for i in range(len(b) // 6)]


# ---- an NCP's sections (docs/format/frame.md 2) ---------------------------------------------------------------------
MV_IN, MV_OT, MV_IM, MV_P0, MV_WT = 0, 1, 2, 3, 7     # memory vector types: IN_FMAP, OT_FMAP, IM_FMAP, OT_PIX0, WEIGHT


def ncp_header(ncp):
    """the 32 header words"""
    return struct.unpack_from("<32I", ncp, 0)


def ncp_vectors(ncp):
    """(header, address vectors, memory vectors, group vectors) as tuples of words"""
    h = ncp_header(ncp)
    avo, avc, mvo, mvc, gvo, gvc = h[10:16]
    av = [struct.unpack_from("<4I", ncp, avo + 16 * i) for i in range(avc)]
    mv = [struct.unpack_from("<8I", ncp, mvo + 32 * i) for i in range(mvc)]
    gv = [struct.unpack_from("<10I", ncp, gvo + 40 * i) for i in range(gvc)]
    return h, av, mv, gv


def ncp_parts(ncp):
    """the parts ncpgen generates, cut out of an NCP: {"streams": isa bytes per group, "intrinsics": bytes per group,
    "weights": the WEIGHT blocks in memory vector order}"""
    h, av, mv, gv = ncp_vectors(ncp)
    body = h[17]
    return {"streams": [ncp[body + g[8]:body + g[8] + g[9]] for g in gv],
            "intrinsics": [ncp[body + g[6]:body + g[6] + g[7]] for g in gv],
            "weights": [ncp[av[m[7]][1]:av[m[7]][1] + av[m[7]][3]] for m in mv if m[0] == MV_WT]}


def first_diff(a, b):
    """the first index where two sequences differ (the shorter one's length when one is a prefix), None if equal"""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


# ---- memoisation --------------------------------------------------------------------------------------------------
class Memo:
    """the latest `size` results of a pure computation by a hashable key, handed out as deep copies (callers may change
    them); an exception it raised is raised again, same type and arguments. The planners, the stream generator and the
    container ask each other for the same plans and weight blocks many times per network."""

    def __init__(self, size):
        self.size = size
        self.d = collections.OrderedDict()

    def get(self, key, compute):
        hit = self.d.get(key)
        if hit is None:
            try:
                hit = (compute(), None)
            except Exception as e:
                hit = (None, (type(e), e.args))
            self.d[key] = hit
            while len(self.d) > self.size:
                self.d.popitem(last=False)
        else:
            self.d.move_to_end(key)
        if hit[1] is not None:
            raise hit[1][0](*hit[1][1])
        return copy.deepcopy(hit[0])


def weights_digest(w):
    """a hashable digest of a weight list (uint8 values: through bytes; anything else as a tuple)"""
    try:
        return hashlib.sha256(bytes(w)).digest()
    except (TypeError, ValueError):
        return tuple(w)


def network_key(low):
    """a hashable digest of everything in a lowered network (weights by weights_digest)"""
    def weightless(L):
        L = dict(L)
        if "weights_oihw_u8" in L:
            d = weights_digest(L["weights_oihw_u8"])
            L["weights_oihw_u8"] = d.hex() if isinstance(d, bytes) else repr(d)
        return L
    rest = {k: v for k, v in low.items() if k != "layers"}
    text = json.dumps([rest, [weightless(L) for L in low.get("layers", [])]], sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).digest()
