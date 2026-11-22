"""What the parts of ncpgen share: NPUC's integer and float32 arithmetic, a network's topology as NPUC reads it, the
CMDQ instruction encoding, the sections of an NCP (docs/format/), and memoisation of the pure planners."""
import collections
import copy
import hashlib
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


def exec_order(out, ins, cons, src):
    """NPUC's layer order (V: its NODE ORDER logs; add2, cat3, cat-pool, cat-chain, res-two and every graph compile):
    from the output back, a layer is entered once every layer reading it is, its inputs in order (depth first); the
    names, run in reverse"""
    seen, lst, stack = {out}, [out], [(out, 0)]
    while stack:
        n, i = stack.pop()
        if i < len(ins[n]):
            stack.append((n, i + 1))
            t = ins[n][i]
            if t != src and t not in seen and all(r in seen for r in cons[t]):
                seen.add(t)
                lst.append(t)
                stack.append((t, 0))
    return lst[::-1]


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
    concat is not the network output `out`, no eltwise reads it and it does not fork into branches an eltwise joins
    (V: g012, a concat whose one reader forks, stays in SRAM)"""
    if name == out or any(i not in by for i in ins[name]):
        return False
    if not all(by[i]["kind"] == "pool" or (by[i]["kind"] == "conv" and by[i]["out_chw"][0] % 16 == 0)
               for i in ins[name]):
        return False
    readers = cons.get(name, [])
    if any(by[r]["kind"] == "eltwise" for r in readers):
        return False
    return not (len(readers) > 1 and feeds_eltwise(name, by, cons))


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
