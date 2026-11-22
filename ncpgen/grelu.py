#!/usr/bin/env python3
"""The GReLU activation table of an NPU2020/Neus convolution (docs/format/grelu.md).

Our own model of what Samsung's NPUC 1.36.1.a writes (formulas, register addresses and observed constants only; no
vendor bytes), and of the corrected table.

The block (V: libnpuc_backend NPU2020::NcpMtcConvolution::ncp_mtc_set_file @0x1422d8, when the conv's GReLU mode is 2
or 3; every block in every oracle NCP of the sweep and in the stock NFD.ncp):
    CMDQ  0x401e0050 = 1
    NFUBR 0x40163800 = 0xff
    NFUBR 0x40163044 + 4k = (int16 A[k] sign-extended) | (uint16 B[k]) << 16      k = 0..15
    NFUBR 0x40163084 = T1, 0x40163088 = T2                                        (mode 3 only)
    NFUBR 0x40163800 = 0
    CMDQ  0x401e0050 = 0
One block per output-channel pass of the conv (one MTC convolution per pass).

What NPUC puts in A/B (V: libnpuc_template LIRGenExecutor2020::convCreateOperation @0x1bbe0c and its per-DRU lambda
@0x1bf3d0; the oracle sweep): on device (CompileOptions +0x138 = 1, forced by setCompileOptions) the lambda stores each
populated DRU's requant slope and output zero point at index d = core * 4 + lane: B[d] = slope, A[d] = out zp byte.
The activation then edits the array as if it were a piece table: no activation (GRELU_BYPASS_ONDEVICE, 11) only sets
T1 = T2 = 0; ReLU (GRELU_RELU_ONDEVICE, 10) also zeroes B[0] and B[1]. Both are mode 3.

The hardware (V from the NPU runs, dossier/npu-compiler.md 9.10-9.16): one table for every channel, group 0 (entries
0..3); entry 0 for acc < T1, entry 2 for acc >= T2 (acc = T1 = T2 = 0 took entry 2), entry 1 unreachable when
T1 = T2; out = A + round_half_away(acc * B / 2^shift). So NPUC's table is right only when DRU 2 (core 0, lane 2) is
populated, i.e. when every pass gives core 0 at least 3 channels: Cout a multiple of 16 always, Cout <= 8 never.

Channel -> DRU (V, sweep of Cout 1..255): chunk = 2 * ceil(Cout / 8) channels per core, core k takes channels
[k * chunk, (k + 1) * chunk); pass p takes lanes [4p, 4p + 4) of every core's chunk; DRU = core * 4 + lane - 4p.

Slope (V: libnpuc_frontend NNGraphNodeScaleONFiller::setScale @0x9da64, fcvtas): m = f32(f32(in * w) / out), doubled
while < 0.5 (at most 7 times), slope = round_half_away(m * 256), shift = doublings + 8 (common.requant).

  grelu.py check <variant dir>...      fixed=False vs the oracle's GReLU writes, fixed=True vs ncp-grelu-patch.py auto
  grelu.py table <ncp>...              print every GReLU block of an NCP (A/B per entry, T1/T2)
  grelu.py regs <lowered.json> [--fixed]   print grelu_regs() for each conv layer
  grelu.py selftest                    the model against facts read in the sweep (no compile)
"""
import argparse
import importlib.util
import json
import os
import struct
import sys

if not __package__:     # run as a script: the package is this file's directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncpgen.common import CMDQ, NFU, decode_isa, f32, multiplier, ncp_vectors, requant, scale_value, set_reg_addr  # noqa

GRELU_EN = CMDQ + 0x50  # 1 before the table, 0 after (G: grants the CMDQ access to the NFU's GReLU registers)
NFU_SEL = NFU + 0x3800  # NFUBR: 0xff before, 0 after (G: a broadcast mask over 8 NFU units)
TABLE = tuple(NFU + 0x3044 + 4 * k for k in range(16))
T1, T2 = NFU + 0x3084, NFU + 0x3088
CORES, LANES = 4, 4

MODE_NONE, MODE_RELU = 11, 10  # NPUC's NNActivationType for a conv without activation / with ReLU (soc 1..3)


def npuc_slope(in_s, w_s, out_s):
    """(slope, shift) exactly as NPUC's ScaleONFiller::setScale (common.requant); ValueError where NPUC asserts"""
    m = multiplier(in_s, w_s, out_s)
    if not m > 0:
        raise ValueError("multiplier %r not > 0" % m)
    _, slope, shift = requant(m)
    if slope > 0xFFFF:
        raise ValueError("slope %d > 65535 (NPUC asserts)" % slope)
    return slope, shift


def layer_slope(layer):
    """(slope, shift) of a lowered conv layer"""
    return npuc_slope(scale_value(layer["in_scale"]), scale_value(layer["w_scale"]), scale_value(layer["out_scale"]))


def zp_byte(layer, quant_mode):
    """NPUC's ncp_mtc_conv +0x109 (the A of every populated DRU): the output zero point; 0 in quasi-symmetric mode
    whatever the layer's zero point (V: stream.py's zp-0 graph nets; npunet always writes 128 there)"""
    return 0 if quant_mode == 3 else int(layer["out_zp"]) & 0xFF


def channel_passes(cout):
    """[{dru: channel}] per output-channel pass (one MTC convolution, one GReLU block each), in emission order"""
    chunk = 2 * -(-cout // 8)
    npass = -(-chunk // LANES)
    out = []
    for p in range(npass):
        m = {}
        for core in range(CORES):
            for lane in range(LANES):
                c = core * chunk + p * LANES + lane
                if p * LANES + lane < chunk and c < cout:
                    m[core * LANES + lane] = c
        out.append(m)
    return out


def act_code(layer):
    """NPUC's NNActivationType of a lowered conv layer: ReLU or none"""
    return MODE_RELU if int(layer.get("act", 0)) == 1 else MODE_NONE


def npuc_blocks(layer, quant_mode):
    """NPUC's tables for a lowered conv layer: [(A[16], B[16], t1, t2, mode)] per pass"""
    slope, _ = layer_slope(layer)
    zp = zp_byte(layer, quant_mode)
    code = act_code(layer)
    blocks = []
    for drus in channel_passes(int(layer["cout"])):
        a, b = [0] * 16, [0] * 16
        for d in drus:  # the per-DRU lambda (CompileOptions +0x138 == 1)
            b[d], a[d] = slope, zp
        if code == MODE_RELU:  # convCreateOperation, case 10: *(u32 *)(conv + 0x1a6) = 0
            b[0] = b[1] = 0
        blocks.append((a, b, 0, 0, 3))  # cases 10 and 11: T1 = T2 = 0, mode 3
    return blocks


def npuc_relu6_blocks(cout):
    """NPUC's table for a standalone ReLU6 layer (npuc-net 2 "relu <name> <in> ReLU6"; V on the oracle with the
    harness-ops dev build): a dummy conv with NNActivationType 2. Its per-DRU fill is slope 1, zero point 0 (both
    quant modes); case 2 then writes B[0..2] = 0, 1, 0, A[0..2] = 0, 0, 0x60, T1 = 0, T2 = 0x60 (the stock NFD.ncp's
    table: a clamp at the fixed accumulator value 96, whatever the scales). A standalone ReLU gets no block (mode 1)."""
    blocks = []
    for drus in channel_passes(int(cout)):
        a, b = [0] * 16, [0] * 16
        for d in drus:
            b[d], a[d] = 1, 0
        b[0:3], a[0:3] = [0, 1, 0], [0, 0, 0x60]
        blocks.append((a, b, 0, 0x60, 3))
    return blocks


def entry(a, b):
    """one table register as NPUC encodes it: (int)(int16)A | (uint16)B << 16"""
    a16 = a & 0xFFFF
    v = (a16 - 0x10000 if a16 & 0x8000 else a16) | (b & 0xFFFF) << 16
    return v & 0xFFFFFFFF


def block_writes(vals, t1, t2, mode):
    """the writes of one block, in NPUC's order; vals = the 16 register values"""
    w = [(GRELU_EN, 1), (NFU_SEL, 0xFF)] + list(zip(TABLE, vals))
    if mode == 3:
        w += [(T1, t1 & 0xFFFFFFFF), (T2, t2 & 0xFFFFFFFF)]
    return w + [(NFU_SEL, 0), (GRELU_EN, 0)]


# ---- the correction ----------------------------------------------------------------------------------------------
def auto_kind(vals):
    """ncp-grelu-patch.py's classification of a block (kept identical)"""
    g = [vals[4 * i:4 * i + 4] for i in range(4)]
    if g[0][0] >> 16 == 0 and g[1][0] >> 16 != 0 and g[0][2] == 0:
        return "relu"
    if g[0][0] >> 16 != 0 and g[0][2] == 0:
        return "none"
    return "other"


def auto_fix(vals):
    """ncp-grelu-patch.py auto on one block's 16 values (relu -> "relu", none -> "piece2", other untouched)"""
    v = list(vals)
    k = auto_kind(v)
    if k == "none":
        for gi in range(4):
            if v[4 * gi] and v[4 * gi + 2] == 0:
                v[4 * gi + 2] = v[4 * gi]
    elif k == "relu":
        g0 = [v[0], v[4], v[4], v[3]]
        v = g0 * 4
    return v, k


def fixed_vals(vals, layer, quant_mode):
    """the corrected 16 values: ncp-grelu-patch.py auto, plus the one case auto cannot repair (V reasoning below)

    A ReLU pass with <= 2 channels (Cout <= 2) carries no slope anywhere in NPUC's table (B[0] = B[1] = 0, no other
    DRU): auto classifies it "other" and leaves acc >= 0 -> entry 2 = (0, 0), the measured defect. We know the slope
    from the layer: group 0 = (zp, 0) (zp, s) (zp, s) entry 3, groups 1-3 = group 0 (auto's own relu layout)."""
    v, k = auto_fix(vals)
    extra = None
    if act_code(layer) == MODE_RELU and v[2] == 0:
        s, _ = layer_slope(layer)
        zp = zp_byte(layer, quant_mode)
        g0 = [entry(zp, 0), entry(zp, s), entry(zp, s), v[3]]
        v = g0 * 4
        extra = "relu pass without a slope in NPUC's table (auto leaves it broken)"
    return v, k, extra


def robust_vals(layer, quant_mode):
    """the canonical table: group 0 = (zp, s) (zp, s) (zp, s) (0, 0) without activation, (zp, 0) (zp, s) (zp, s)
    (0, 0) with ReLU; groups 1-3 = group 0. It is byte for byte every table that ran bit-exact on the NPU (N3d piece2,
    npu-run v2/v4/v5 and net-current, all Cout = 8), and it does not depend on which group lanes 2-3 read (never run:
    Cout >= 9), where auto keeps NPUC's groups 1-3 = other DRUs' (zp, s), not ReLU pieces."""
    s, _ = layer_slope(layer)
    zp = zp_byte(layer, quant_mode)
    lo = 0 if act_code(layer) == MODE_RELU else s
    return [entry(zp, lo), entry(zp, s), entry(zp, s), 0] * 4


def grelu_blocks(layer, quant_mode, fixed=False):
    """[(writes, info)] per pass; info = {"kind": auto's class, "extra": the extra correction or None}"""
    out = []
    if layer.get("kind") == "relu":  # a standalone activation layer: NPUC's own blocks only (no correction modelled)
        if layer.get("type") == "ReLU6":
            for a, b, t1, t2, mode in npuc_relu6_blocks(layer["cout"]):
                vals = [entry(a[k], b[k]) for k in range(16)]
                out.append((block_writes(vals, t1, t2, mode), {"kind": "relu6-dummy", "extra": None}))
        return out
    for a, b, t1, t2, mode in npuc_blocks(layer, quant_mode):
        vals = [entry(a[k], b[k]) for k in range(16)]
        info = {"kind": auto_kind(vals), "extra": None}
        if fixed == "robust":
            vals = robust_vals(layer, quant_mode)
        elif fixed:
            vals, info["kind"], info["extra"] = fixed_vals(vals, layer, quant_mode)
        out.append((block_writes(vals, t1, t2, mode), info))
    return out


def grelu_regs(layer, quant_mode, fixed=False):
    """every GReLU write of a lowered conv layer (lowered.json's layer dict), in order, for all its passes.
    fixed=False: NPUC's exact writes; fixed=True: the corrected table (= ncp-grelu-patch.py auto + fixed_vals());
    fixed="robust": robust_vals() in every block (the recommended table for generation until runs E1/E2 of grelu.md)."""
    return [w for ws, _ in grelu_blocks(layer, quant_mode, fixed) for w in ws]


# ---- a general piecewise-linear table (G until run on the NPU: see grelu.md, "Proposed NPU runs") -----------------
def pwl_regs(pieces, t1=0, t2=0, copy_groups=True):
    """one block for an arbitrary 3-piece function: pieces = [(A0, B0), (A1, B1), (A2, B2)] for acc < t1,
    t1 <= acc < t2, acc >= t2; out = A + round_half_away(acc * B / 2^shift) with the conv's own shift registers.
    Entry 3 = (0, 0); groups 1-3 = group 0 when copy_groups (V: harmless; G: unused)."""
    g0 = [entry(a, b) for a, b in pieces] + [0]
    vals = g0 * 4 if copy_groups else g0 + [0] * 12
    return block_writes(vals, t1, t2, 3)


def relu6_pieces(layer, quant_mode, six=6.0):
    """(pieces, t1, t2) for a bounded ReLU (clamp at round(six / out_scale) above the zero point, TFLite's rule)"""
    s, sh = layer_slope(layer)
    zp = zp_byte(layer, quant_mode)
    qmax = 127 if quant_mode == 3 else 255
    top = min(qmax, zp + int(f32(six) / scale_value(layer["out_scale"]) + 0.5))
    k = top - zp  # first acc whose output reaches top: A + rha(acc s / 2^sh) >= top  <=>  acc s >= (k - 1/2) 2^sh
    t2 = -(-((2 * k - 1) << (sh - 1)) // s) if k > 0 else 0
    return [(zp, 0), (zp, s), (top, 0)], 0, t2


# ---- reading NCPs and checking ------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "oracle")  # ncp-grelu-patch.py, ncp-isa.py


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ncp_writes(data):
    """every set_reg of every group's CMDQ stream: [(reg, value, file offset)]"""
    h, _, _, gv = ncp_vectors(data)
    out = []
    for g in gv:
        off = h[17] + g[8]
        for i, (op, val) in enumerate(decode_isa(data[off:off + g[9]])):
            a = set_reg_addr(op)
            if a is not None:
                out.append((a, val, off + 6 * i))
    return out


GRELU_REGS = {GRELU_EN, NFU_SEL, T1, T2} | set(TABLE)


def ncp_grelu(data):
    """the NCP's GReLU writes, in stream order: [(reg, value)]"""
    return [(r, v) for r, v, _ in ncp_writes(data) if r in GRELU_REGS]


def split_blocks(writes):
    """GReLU writes [(reg, value)] -> one list per block (CMDQ 0x401e0050 = 1 ... = 0)"""
    blocks, cur = [], None
    for r, v in writes:
        if r == GRELU_EN and v == 1:
            cur = [(r, v)]
        elif cur is not None:
            cur.append((r, v))
            if r == GRELU_EN and v == 0:
                blocks.append(cur)
                cur = None
    return blocks


def auto_patched(data):
    """the NCP after ncp-grelu-patch.py auto (its own functions, in memory)"""
    gp = _load("ncp_grelu_patch", "ncp-grelu-patch.py")
    d = bytearray(data)
    s = gp.slots(d)
    if not any(r in s for r in gp.TABLE):
        return bytes(d)
    nb = len(s[gp.TABLE[0]])
    rd = lambda o: struct.unpack_from("<I", d, o)[0]
    for b in range(nb):
        vals = [rd(s[r][b]) for r in gp.TABLE]
        sp = {"relu": "relu", "none": "piece2"}.get(gp.kind(vals))
        if sp:
            for i, v in sorted(gp.block_want(vals, sp).items()):
                struct.pack_into("<I", d, s[gp.TABLE[i]][b], v)
    return bytes(d)


def conv_layers(low):
    """the layers that carry GReLU blocks: convs, and standalone "relu" layers (type ReLU6 / ReLU, cout)"""
    return [l for l in low["layers"] if l.get("kind") in ("conv", "relu")]


def model_writes(low, fixed):
    """(every GReLU write of a lowered network in order, the extra corrections fixed_vals made)"""
    q = int(low.get("quant_mode", 1))
    ws, extras = [], []
    for l in conv_layers(low):
        for w, info in grelu_blocks(l, q, fixed):
            ws += w
            if info["extra"]:
                extras.append("%s: %s" % (l.get("name", "?"), info["extra"]))
    return ws, extras


def _write_diff(a, b):
    """where two GReLU write lists first differ, as text"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "write %d: model %#010x=%#010x, oracle %#010x=%#010x" % (i, x[0], x[1], y[0], y[1])
    return "lengths: model %d writes, oracle %d" % (len(a), len(b))


def check(dirs):
    """our tables against each variant's net.ncp (NPUC's) and ncp-grelu-patch.py auto's; 0 when as expected"""
    tot = {"npuc": [0, 0], "fixed": [0, 0, 0]}
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        try:
            low = json.load(open(os.path.join(d, "lowered.json")))
            data = open(os.path.join(d, "net.ncp"), "rb").read()
        except OSError as e:
            print("%-14s skipped (%s)" % (name, e.strerror))
            continue
        oracle = ncp_grelu(data)
        mine, _ = model_writes(low, False)
        ok = mine == oracle
        tot["npuc"][0 if ok else 1] += 1
        nb = len(split_blocks(oracle))
        line = "%-14s npuc  %s (%d blocks, %d writes)" % (name, "exact" if ok else "DIFFERS: " + _write_diff(mine, oracle),
                                                        nb, len(oracle))
        fixed, extras = model_writes(low, True)
        auto = ncp_grelu(auto_patched(data))
        if fixed == auto:
            tot["fixed"][0] += 1
            line += " | fixed exact (= auto)"
        elif extras and len(fixed) == len(auto):
            tot["fixed"][1] += 1
            line += " | fixed differs from auto as intended: " + "; ".join(sorted(set(extras)))
        else:
            tot["fixed"][2] += 1
            line += " | fixed DIFFERS from auto: " + _write_diff(fixed, auto)
        robust, _ = model_writes(low, "robust")
        nd = sum(1 for x, y in zip(robust, auto) if x != y) if len(robust) == len(auto) else -1
        tot["robust"] = tot.get("robust", 0) + (nd == 0)
        line += " | robust %s" % ("= auto" if nd == 0 else "%d entries differ" % nd)
        print(line)
    n = sum(tot["npuc"])
    print("total: %d variants; npuc exact %d, differs %d; fixed = auto %d, intended deviation %d, differs %d; "
          "robust = auto %d" % (n, tot["npuc"][0], tot["npuc"][1], tot["fixed"][0], tot["fixed"][1], tot["fixed"][2],
                                tot.get("robust", 0)))
    return 0 if not tot["npuc"][1] and not tot["fixed"][2] else 1


def show_table(path):
    """print every GReLU block of an NCP"""
    data = open(path, "rb").read()
    for i, b in enumerate(split_blocks(ncp_grelu(data))):
        t = dict(b)
        vals = [t.get(r, 0) for r in TABLE]
        groups = []
        for g in range(4):
            groups.append(" ".join("(%d,%d)" % (struct.unpack("<h", struct.pack("<H", v & 0xFFFF))[0], v >> 16)
                                   for v in vals[4 * g:4 * g + 4]))
        print("%s block %d: [%s] T1 %s T2 %s  (A,B) per entry, groups | %s" % (
            os.path.basename(path), i, auto_kind(vals), hex(t[T1]) if T1 in t else "-", hex(t[T2]) if T2 in t else "-",
            " | ".join(groups)))


def selftest():
    """facts read in the oracle sweep (docs/format/grelu.md), re-derived by the model; no compile"""
    used = lambda c: [sorted(m) for m in channel_passes(c)]
    ok = [
        ("Cout 3 -> DRUs 0,1,4", used(3) == [[0, 1, 4]]),
        ("Cout 8 -> two lanes per core", used(8) == [[0, 1, 4, 5, 8, 9, 12, 13]]),
        ("Cout 9 -> DRUs 0..8", used(9) == [list(range(9))]),
        ("Cout 17 -> 12 + 5", used(17) == [list(range(12)), [0, 1, 4, 5, 8]]),
        ("Cout 33 -> 15 + 12 + 6", [len(m) for m in channel_passes(33)] == [15, 12, 6]),
        ("Cout 100 -> 7 passes, last 6", [len(m) for m in channel_passes(100)] == [16] * 5 + [14, 6]),
        ("Cout 255 -> 16 passes, last 15", [len(m) for m in channel_passes(255)] == [16] * 15 + [15]),
        ("slope 1/128 x 1/256 / 1/128 = 128 / 2^15", npuc_slope(1 / 128, 1 / 256, 1 / 128) == (128, 15)),
        ("slope rounds half away (s6: 162 / 2^10, npunet says 161)", npuc_slope(0.02, 0.0071, 0.0009) == (162, 10)),
        ("slope rounds half away (s7: 195 / 2^11)", npuc_slope(0.0078125, 0.0377, 0.0031) == (195, 11)),
        ("entry encodes A | B << 16", entry(0x80, 0x80) == 0x00800080 and entry(0, 1) == 0x10000),
    ]
    lay = {"cout": 2, "act": 1, "in_scale": "0x3c000000", "w_scale": "0x3b800000", "out_scale": "0x3c000000",
           "out_zp": 128}
    npuc = grelu_regs(lay, 1)
    fix = grelu_regs(lay, 1, True)
    ok.append(("Cout 2 ReLU: NPUC writes (128,0) (128,0) and nothing else", [v for r, v in npuc if r in TABLE][:3] == [0x80, 0x80, 0]))
    ok.append(("Cout 2 ReLU fixed: group 0 = (128,0) (128,128) (128,128) (0,0)",
               [v for r, v in fix if r in TABLE][:4] == [0x80, 0x00800080, 0x00800080, 0]))
    ok.append(("block = 22 writes in mode 3", len(npuc) == 22 and npuc[0] == (GRELU_EN, 1) and npuc[-1] == (GRELU_EN, 0)))
    p, t1, t2 = relu6_pieces({"in_scale": "0x3c000000", "w_scale": "0x3b800000", "out_scale": "0x3d000000",
                              "out_zp": 0x80}, 1)  # out scale 1/32: six = 192 steps -> top 255, slope 32 / 2^15
    s, sh = npuc_slope(1 / 128, 1 / 256, 1 / 32)
    rha = lambda acc: int(acc * s / 2 ** sh + 0.5)
    ok.append(("relu6 T2 is the first acc reaching the top", 0x80 + rha(t2) >= 255 > 0x80 + rha(t2 - 1)))
    bad = [n for n, v in ok if not v]
    for n, v in ok:
        print("%s %s" % ("ok  " if v else "FAIL", n))
    print("grelu selftest: %s (%d checks)" % ("PASS" if not bad else "FAIL", len(ok)))
    return 1 if bad else 0


def main(argv=None):
    """the command line (module docstring)"""
    ap = argparse.ArgumentParser(prog="grelu.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="the model against facts read in the sweep")
    p = sub.add_parser("check", help="our tables against NPUC's compiles")
    p.add_argument("dirs", nargs="+")
    p = sub.add_parser("table", help="every GReLU block of NCPs")
    p.add_argument("ncps", nargs="+")
    p = sub.add_parser("regs", help="grelu_regs() of each conv layer")
    p.add_argument("lowered")
    p.add_argument("--fixed", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "check":
        return check(a.dirs)
    if a.cmd == "table":
        for p in a.ncps:
            show_table(p)
        return 0
    low = json.load(open(a.lowered))
    for l in conv_layers(low):
        for r, v in grelu_regs(l, int(low.get("quant_mode", 1)), a.fixed):
            print("%s %#010x = %#010x" % (l.get("name", "?"), r, v))
    return 0


if __name__ == "__main__":
    from ncpgen import grelu as _pkg    # the package's copy of this module (one copy of every module)
    sys.exit(_pkg.main())
