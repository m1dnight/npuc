#!/usr/bin/env python3
"""npunet: the host half of "run a network on the Exynos 990 NPU" (PORT-32; npu/README.md). Pure Python 3: no numpy,
no flatbuffers package (a minimal FlatBuffer reader and writer are below).

  npunet.py lower <model.tflite|net.json> <dir> [--quant 1|3] [--fold-per-channel]
        the model as the NPU will compute it: <dir>/lowered.json (every parameter explicit) and <dir>/net.npuc (the
        phone's compiler harness input, oracle/harness/npuc-net.cpp)
  npunet.py ref <lowered.json> <input.bin> <output.bin>        the CPU reference: the model's output bytes
  npunet.py layout <lowered.json> <net.ncp> <layout.json>      the NPU's IFM/OFM formats from the NCP's descriptors
  npunet.py to-npu <lowered.json> <layout.json> <input.bin> <npu-in.bin>
  npunet.py from-npu <lowered.json> <layout.json> <ofm.bin> <output.bin>
  npunet.py compare <lowered.json> <layout.json> <input.bin>:<ofm.bin>...   the NPU's output against the CPU
  npunet.py example <dir>        the tiny example: tiny-net.json, tiny.tflite, input.bin, expected.bin
  npunet.py example-residual <dir>   a residual block (conv+relu, conv, add with the block input, relu):
        residual-net.json, residual.tflite, input.bin, input2.bin, expected.bin, expected2.bin (--quant 3 --allow-none)
  npunet.py variants <dir>       the example taken apart (v1-pool ... v5-tiny), for journal/plans/npu-run-0218/nr-variants.sh
  npunet.py selftest

Model formats. (1) A quantized TFLite file (int8 or uint8, one subgraph, one input, one output; ops in graph order,
a tensor may feed several ops). (2) Our graph JSON "npu-net/1" (npu/README.md; npu/example/tiny-net.json): the same
information, written by hand or by a script. Each layer reads the previous layer's output unless it names its
input(s): "input": "<tensor>" or "inputs": ["<a>", "<b>", ...] (a tensor = the network input's or a layer's "name").
input.bin / output.bin are the model's own tensors: NHWC, in the model's dtype (int8 or uint8), as a TFLite
interpreter takes and gives them.

What runs on the NPU (the ops nn-eden-drv's NNNetworkBuilder itself hands NPUC on this firmware, V: its strings
"operator type is conv2d / average_pool_2d / max_pool_2d / reshape / concat", else "not yet supported"):
CONV_2D (fused NONE or RELU; RELU6 accepted when its clamp is a no-op), MAX_POOL_2D, AVERAGE_POOL_2D,
FULLY_CONNECTED (lowered here to a CONV_2D whose kernel covers its input). RESHAPE before a FULLY_CONNECTED or at the
end, and SOFTMAX at the end, run on the host's CPU. Graph ops through the harness's "npuc-net 2" layers
(docs/harness-ops.md; NPUC compiles them, nothing of them has run on the NPU yet):
  ADD -> eltwise Add (--quant 3 only, every scale 2^-n, no broadcasting; a fused RELU -> a relu layer after it);
  CONCATENATION -> concat (channels only, every input already in the output's quantisation, not the network input);
  RELU (RELU6 when its clamp is a no-op) -> folded into the conv before it when it alone reads that conv, else relu.
Anything else (DEPTHWISE_CONV_2D, MUL...) is refused with its name. A chain of conv/pool layers is written as
"npuc-net 1", exactly as before; "npuc-net 2" only when a graph op is present.

NPUC's arithmetic, as far as it is known (dossier/npu-compiler.md section 9.12):
  - the multiplier (V, ScaleONFiller::setScale @libnpuc_frontend 0x9da64): m = float32(in_scale * w_scale) /
    out_scale in float32; doubled up to 7 times while < 0.5; slope = round half away (m * 256) (fcvtas @0x9dd40;
    <= 65535), shift = doublings + 8;
    ONE multiplier per layer (wScale[0]): per-channel weight scales are not supported (--fold-per-channel rescales the
    weights to one scale, a lossy change of the model, reported);
  - out = zp_out + R(acc * slope / 2^shift), R a half rounding (V for slope 16, shift 15: section 9.11), clamped to
    0..255; acc = bias + sum (x - zp_in) * (w - zp_w), the border = the zero point;
  - RELU: max(out, zp_out) (G: under test). "No activation" compiles to GRELU_BYPASS_ONDEVICE, which writes the byte 0
    wherever acc >= 0 (V, section 9.11): such a layer is refused unless --allow-none (the comparison then also tries
    that defect and names it);
  - quant mode 1 (asymmetric u8, the HAL's default) for any zero point; mode 3 (native s8: bytes XOR 0x80) only when
    every zero point is 128 (int8 0).
  - Eltwise Add (G, docs/harness-ops.md 3.1: it only shifts by powers of two, no zero-point term):
    out = sat_s8(R(in1 * 2^(flo - fl1) + in2 * 2^(flo - fl2))) on the s8 values, scale = 2^-fl; R half away like the
    convs (compare also tries floor, the other roundings, and rounding each term before the sum);
  - concat: the inputs' bytes placed as they are, channel slices in input order (V: NPUC's ChannelSlicedFeatureNode);
  - standalone ReLU (G): max(x, zp). NPUC makes it a 1x1 dummy conv with all-zero weights, GReLU mode 1 and no table
    (docs/format/grelu.md 7, weights.md); compare also tries the identity and a constant zero point.
"""
import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
from fractions import Fraction

if not __package__:     # run as a script, or loaded from its path (the porting kit's npu/npunet.py): the package is
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # this file's directory

from ncpgen.common import f32, f32_hex, has_graph_ops, hex_f32, multiplier, ncp_header, requant  # noqa: E402

ROUNDINGS = ("half_away", "half_even", "half_up")
HERE = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    print("npunet: FAIL: " + msg)
    sys.exit(1)


class Refused(Exception):
    pass


# ---- float32 ------------------------------------------------------------------------------------------------------
def npuc_multiplier(in_s, w_s, out_s):
    """(slope, shift) as NPUC's NNGraphNodeScaleONFiller::setScale computes them (V, decompiled; common.requant);
    Refused where NPUC asserts, or where the slope would be 0"""
    m = multiplier(in_s, w_s, out_s)
    if not m > 0:
        raise Refused("multiplier %r is not > 0 (NPUC asserts real_multiplier > 0)" % m)
    m, slope, shift = requant(m)
    if slope > 65535:
        raise Refused("multiplier %r: slope %d > 65535 (NPUC asserts q <= uint16 max)" % (m, slope))
    if slope == 0:
        raise Refused("multiplier in*w/out = %.3g is below 2^-16: NPUC's slope would be 0" % (f32(in_s) * f32(w_s) / f32(out_s)))
    return slope, shift


def pow2_fl(s):
    """a float32 scale 2^-n (n 0..31) -> n (its fractional length, as npuc-net reads it from the bits), else None"""
    u = struct.unpack("<I", struct.pack("<f", s))[0]
    n = 127 - ((u >> 23) & 0xff)
    return n if not u & 0x807fffff and 0 <= n <= 31 else None


def rnd(v, mode):
    """v: Fraction -> int"""
    f = v.numerator // v.denominator
    if mode == "floor":
        return f
    r = v - f
    if r != Fraction(1, 2):
        return f + (1 if r > Fraction(1, 2) else 0)
    if mode == "half_up":
        return f + 1
    if mode == "half_even":
        return f + (f & 1)
    return f + 1 if v > 0 else f  # half_away


# ---- a minimal FlatBuffer reader (enough for TFLite's schema) ----------------------------------------------------
class FTable:
    def __init__(self, b, pos):
        self.b, self.pos = b, pos
        self.vt = pos - struct.unpack_from("<i", b, pos)[0]
        self.vtlen = struct.unpack_from("<H", b, self.vt)[0]

    def _off(self, field):
        o = 4 + 2 * field
        return struct.unpack_from("<H", self.b, self.vt + o)[0] if o < self.vtlen else 0

    def scalar(self, field, fmt, default=0):
        o = self._off(field)
        return struct.unpack_from("<" + fmt, self.b, self.pos + o)[0] if o else default

    def _ref(self, field):
        o = self._off(field)
        if not o:
            return None
        p = self.pos + o
        return p + struct.unpack_from("<I", self.b, p)[0]

    def table(self, field):
        p = self._ref(field)
        return FTable(self.b, p) if p is not None else None

    def vector(self, field):
        p = self._ref(field)
        if p is None:
            return None, 0
        return p + 4, struct.unpack_from("<I", self.b, p)[0]

    def vec_scalars(self, field, fmt):
        start, n = self.vector(field)
        if start is None:
            return []
        sz = struct.calcsize("<" + fmt)
        return list(struct.unpack_from("<%d%s" % (n, fmt), self.b, start)) if n else []

    def vec_bytes(self, field):
        start, n = self.vector(field)
        return bytes(self.b[start:start + n]) if start is not None else b""

    def vec_tables(self, field):
        start, n = self.vector(field)
        out = []
        for i in range(n if start is not None else 0):
            p = start + 4 * i
            out.append(FTable(self.b, p + struct.unpack_from("<I", self.b, p)[0]))
        return out

    def string(self, field):
        b = self.vec_bytes(field)
        return b.decode("utf-8", "replace")


# ---- a minimal FlatBuffer writer: front to back, children after their parent (every uoffset forward) ----------
class FWriter:
    def __init__(self, ident=b"TFL3"):
        self.b = bytearray(8)
        self.b[4:8] = ident

    def _align(self, n):
        while len(self.b) % n:
            self.b.append(0)

    def finish(self, root):
        struct.pack_into("<I", self.b, 0, root)
        return bytes(self.b)

    def table(self, fields):
        """fields: [(field index, kind, value)]; kind 'b','B','h','i','I','q','Q','f' (a scalar) or 'ref' (value: a
        callable that writes the child and returns its position). Returns the table's position."""
        fields = sorted(fields, key=lambda f: f[0])
        nf = (max(f[0] for f in fields) + 1) if fields else 0
        # inline layout: soffset (4) then the fields, biggest first, each aligned to its size
        sizes = {"b": 1, "B": 1, "h": 2, "i": 4, "I": 4, "f": 4, "q": 8, "Q": 8, "ref": 4}
        order = sorted(fields, key=lambda f: -sizes[f[1]])
        layout, off = {}, 4
        for idx, kind, _ in order:
            s = sizes[kind]
            off = (off + s - 1) // s * s
            layout[idx] = off
            off += s
        tsize = (off + 3) // 4 * 4
        vt = struct.pack("<HH", 4 + 2 * nf, tsize) + b"".join(struct.pack("<H", layout.get(i, 0)) for i in range(nf))
        self._align(2)
        vtpos = len(self.b)
        self.b += vt
        self._align(8)
        tpos = len(self.b)
        self.b += bytes(tsize)
        struct.pack_into("<i", self.b, tpos, tpos - vtpos)
        refs = []
        for idx, kind, val in fields:
            if kind == "ref":
                refs.append((tpos + layout[idx], val))
            else:
                struct.pack_into("<" + kind, self.b, tpos + layout[idx], val)
        for slot, writer in refs:
            p = writer()
            struct.pack_into("<I", self.b, slot, p - slot)
        return tpos

    def vec_scalars(self, fmt, values, align=4):
        sz = struct.calcsize("<" + fmt)
        a = max(4, sz, align)
        while (len(self.b) + 4) % a:
            self.b.append(0)
        p = len(self.b)
        self.b += struct.pack("<I", len(values))
        if values:
            self.b += struct.pack("<%d%s" % (len(values), fmt), *values)
        return p

    def vec_bytes(self, data, align=16):
        while (len(self.b) + 4) % align:
            self.b.append(0)
        p = len(self.b)
        self.b += struct.pack("<I", len(data)) + bytes(data)
        return p

    def string(self, s):
        p = self.vec_bytes(s.encode() + b"\0", align=4)
        struct.pack_into("<I", self.b, p, len(s.encode()))
        return p

    def vec_refs(self, writers):
        self._align(4)
        p = len(self.b)
        self.b += struct.pack("<I", len(writers)) + bytes(4 * len(writers))
        for i, w in enumerate(writers):
            q = w()
            struct.pack_into("<I", self.b, p + 4 + 4 * i, q - (p + 4 + 4 * i))
        return p


# ---- TFLite: the subset we read ----------------------------------------------------------------------------------
OPS = {0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D", 6: "DEQUANTIZE",
       9: "FULLY_CONNECTED", 14: "LOGISTIC", 17: "MAX_POOL_2D", 18: "MUL", 19: "RELU", 21: "RELU6", 22: "RESHAPE",
       25: "SOFTMAX", 114: "QUANTIZE"}
OPCODE = {v: k for k, v in OPS.items()}
TTYPES = {0: "float32", 2: "int32", 3: "uint8", 9: "int8", 7: "int16", 4: "int64"}
TTYPE = {v: k for k, v in TTYPES.items()}
ACTS = {0: "none", 1: "relu", 2: "relu_n1_to_1", 3: "relu6", 4: "tanh", 5: "sign_bit"}
PADS = {0: "same", 1: "valid"}


def read_tflite(data):
    """-> a dict: tensors (shape, dtype, data, scale, zero_point), inputs, outputs, operators (op, inputs, outputs,
    options) of subgraph 0"""
    try:
        return _read_tflite(data)
    except (struct.error, IndexError, KeyError) as e:
        raise Refused("not a readable TFLite flatbuffer (%s)" % e)


def is_tflite(data):
    """the TFL3 identifier, or (older converters write none) a root table whose version field is 3"""
    if len(data) < 16:
        return False
    if data[4:8] == b"TFL3":
        return True
    try:
        return FTable(data, struct.unpack_from("<I", data, 0)[0]).scalar(0, "I", 0) == 3
    except (struct.error, IndexError):
        return False


def _read_tflite(data):
    if not is_tflite(data):
        raise Refused("not a TFLite file (no TFL3 identifier, no version-3 root)")
    m = FTable(data, struct.unpack_from("<I", data, 0)[0])
    buffers = []
    for b in m.vec_tables(4):
        d = b.vec_bytes(0)
        off, size = b.scalar(1, "Q"), b.scalar(2, "Q")
        if not d and off > 1 and size:
            d = data[off:off + size]
        buffers.append(d)
    codes = []
    for c in m.vec_tables(1):
        codes.append(max(c.scalar(0, "b", 0), c.scalar(3, "i", 0)))
    sgs = m.vec_tables(2)
    if len(sgs) != 1:
        raise Refused("%d subgraphs (one is supported)" % len(sgs))
    sg = sgs[0]
    tensors = []
    for t in sg.vec_tables(0):
        q = t.table(4)
        tensors.append({
            "shape": t.vec_scalars(0, "i"), "dtype": TTYPES.get(t.scalar(1, "b", 0), "type%d" % t.scalar(1, "b", 0)),
            "buffer": t.scalar(2, "I", 0), "name": t.string(3),
            "scale": q.vec_scalars(2, "f") if q else [], "zero_point": q.vec_scalars(3, "q") if q else [],
            "qdim": q.scalar(6, "i", 0) if q else 0})
    for t in tensors:
        t["data"] = buffers[t["buffer"]] if t["buffer"] < len(buffers) else b""
    ops = []
    for o in sg.vec_tables(3):
        code = codes[o.scalar(0, "I", 0)]
        name = OPS.get(code, "builtin %d" % code)
        opt = o.table(4)
        otype = o.scalar(3, "B", 0)
        d = {"op": name, "inputs": o.vec_scalars(1, "i"), "outputs": o.vec_scalars(2, "i"), "opt": {}}
        if opt is not None:
            if name == "CONV_2D" and otype == 1:
                d["opt"] = {"padding": PADS[opt.scalar(0, "b", 0)], "stride_w": opt.scalar(1, "i", 0),
                            "stride_h": opt.scalar(2, "i", 0), "act": ACTS.get(opt.scalar(3, "b", 0), "?"),
                            "dil_w": opt.scalar(4, "i", 1), "dil_h": opt.scalar(5, "i", 1)}
            elif name == "DEPTHWISE_CONV_2D" and otype == 2:
                d["opt"] = {"padding": PADS[opt.scalar(0, "b", 0)], "act": ACTS.get(opt.scalar(4, "b", 0), "?")}
            elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D") and otype == 5:
                d["opt"] = {"padding": PADS[opt.scalar(0, "b", 0)], "stride_w": opt.scalar(1, "i", 0),
                            "stride_h": opt.scalar(2, "i", 0), "filter_w": opt.scalar(3, "i", 0),
                            "filter_h": opt.scalar(4, "i", 0), "act": ACTS.get(opt.scalar(5, "b", 0), "?")}
            elif name == "FULLY_CONNECTED" and otype == 8:
                d["opt"] = {"act": ACTS.get(opt.scalar(0, "b", 0), "?"), "weights_format": opt.scalar(1, "b", 0),
                            "keep_num_dims": opt.scalar(2, "B", 0)}
            elif name == "SOFTMAX" and otype == 9:
                d["opt"] = {"beta": opt.scalar(0, "f", 1.0)}
            elif name == "ADD" and otype == 11:
                d["opt"] = {"act": ACTS.get(opt.scalar(0, "b", 0), "?")}
            elif name == "CONCATENATION" and otype == 10:
                d["opt"] = {"axis": opt.scalar(0, "i", 0), "act": ACTS.get(opt.scalar(1, "b", 0), "?")}
        ops.append(d)
    return {"tensors": tensors, "inputs": sg.vec_scalars(1, "i"), "outputs": sg.vec_scalars(2, "i"), "ops": ops,
            "description": m.string(3)}


def tflite_summary(tf):
    lines = []
    for o in tf["ops"]:
        t = tf["tensors"][o["outputs"][0]] if o["outputs"] else {}
        lines.append("%s -> %s %s" % (o["op"], t.get("shape"), t.get("dtype")))
    return lines


def _q1(t, what):
    if len(t["scale"]) != 1 or len(t["zero_point"]) != 1:
        raise Refused("%s %r: %d scales / %d zero points (one per tensor wanted)" % (what, t["name"], len(t["scale"]), len(t["zero_point"])))
    return float(t["scale"][0]), int(t["zero_point"][0])


def _ints(data, dtype):
    fmt = {"int8": "b", "uint8": "B", "int32": "i"}[dtype]
    n = len(data) // struct.calcsize(fmt)
    return list(struct.unpack("<%d%s" % (n, fmt), data[:n * struct.calcsize(fmt)]))


def tflite_to_graph(tf, src):
    """TFLite (subgraph 0) -> our graph JSON ("npu-net/1"), unchanged values. Every layer gets "id" = "t<index>" of
    its output tensor and "inputs" = the ids it reads (TFLite's own tensor names may be empty or repeat); its "name"
    stays the tensor's name (the lowered layer's name comes from it)."""
    T = tf["tensors"]
    if len(tf["inputs"]) != 1 or len(tf["outputs"]) != 1:
        raise Refused("%d inputs, %d outputs (one each is supported)" % (len(tf["inputs"]), len(tf["outputs"])))
    tin = T[tf["inputs"][0]]
    if tin["dtype"] not in ("int8", "uint8"):
        raise Refused("input %r is %s (a quantized int8 or uint8 input is needed; float models are not supported)" % (tin["name"], tin["dtype"]))
    if len(tin["shape"]) != 4 or tin["shape"][0] != 1:
        raise Refused("input shape %s (NHWC with batch 1 wanted)" % tin["shape"])
    s, z = _q1(tin, "input")
    g = {"format": "npu-net/1", "name": os.path.splitext(os.path.basename(src))[0], "source": os.path.basename(src),
         "input": {"name": tin["name"], "id": "t%d" % tf["inputs"][0], "shape": tin["shape"], "dtype": tin["dtype"],
                   "scale": s, "zero_point": z},
         "layers": []}
    known = {tf["inputs"][0]}
    last = None
    for o in tf["ops"]:
        out = T[o["outputs"][0]]
        name = out["name"] or "t%d" % o["outputs"][0]
        op = o["op"]
        # the data inputs (the others are constants: weights, bias, a reshape's shape)
        data = o["inputs"][:{"ADD": 2, "CONCATENATION": len(o["inputs"])}.get(op, 1)]
        for t in data:
            if t not in known:
                raise Refused("%s (%r) reads tensor %d (%r), which is neither the model's input nor an earlier op's output "
                              "(a constant operand is not supported)" % (op, name, t, T[t]["name"] if 0 <= t < len(T) else "?"))
        if op in ("CONV_2D", "FULLY_CONNECTED"):
            w, b = T[o["inputs"][1]], (T[o["inputs"][2]] if len(o["inputs"]) > 2 and o["inputs"][2] >= 0 else None)
            if w["dtype"] not in ("int8", "uint8"):
                raise Refused("%s %r: weights %s (int8/uint8 wanted)" % (op, name, w["dtype"]))
            os_, oz = _q1(out, op + " output")
            L = {"op": "conv2d" if op == "CONV_2D" else "fully_connected", "name": name,
                 "activation": o["opt"].get("act", "none"),
                 "weights": {"dtype": w["dtype"], "layout": "OHWI" if op == "CONV_2D" else "OI", "shape": w["shape"],
                             "scale": [float(v) for v in w["scale"]], "zero_point": [int(v) for v in w["zero_point"]],
                             "data": _ints(w["data"], w["dtype"])},
                 "bias": _ints(b["data"], "int32") if b is not None and b["data"] else [0] * w["shape"][0],
                 "output": {"scale": os_, "zero_point": oz}}
            if b is not None and b["dtype"] != "int32":
                raise Refused("%s %r: bias %s (int32 wanted)" % (op, name, b["dtype"]))
            if op == "CONV_2D":
                if o["opt"].get("dil_w", 1) != 1 or o["opt"].get("dil_h", 1) != 1:
                    raise Refused("CONV_2D %r: dilation (not supported)" % name)
                L.update({"filters": w["shape"][0], "kernel": [w["shape"][1], w["shape"][2]],
                          "strides": [o["opt"]["stride_h"], o["opt"]["stride_w"]], "padding": o["opt"]["padding"]})
            else:
                L["units"] = w["shape"][0]
        elif op in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            s_, z_ = _q1(out, op + " output")
            L = {"op": "max_pool2d" if op == "MAX_POOL_2D" else "average_pool2d", "name": name,
                 "pool": [o["opt"]["filter_h"], o["opt"]["filter_w"]],
                 "strides": [o["opt"]["stride_h"], o["opt"]["stride_w"]], "padding": o["opt"]["padding"],
                 "activation": o["opt"]["act"], "output": {"scale": s_, "zero_point": z_}}
        elif op == "RESHAPE":
            L = {"op": "reshape", "name": name, "shape": out["shape"]}
        elif op == "SOFTMAX":
            s_, z_ = _q1(out, "SOFTMAX output")
            L = {"op": "softmax", "name": name, "beta": o["opt"].get("beta", 1.0), "output": {"scale": s_, "zero_point": z_}}
        elif op == "ADD":
            s_, z_ = _q1(out, "ADD output")
            L = {"op": "add", "name": name, "activation": o["opt"].get("act", "none"), "output": {"scale": s_, "zero_point": z_}}
        elif op == "CONCATENATION":
            s_, z_ = _q1(out, "CONCATENATION output")
            L = {"op": "concat", "name": name, "axis": o["opt"].get("axis", 0), "activation": o["opt"].get("act", "none"),
                 "output": {"scale": s_, "zero_point": z_}}
        elif op in ("RELU", "RELU6"):
            s_, z_ = _q1(out, op + " output")
            L = {"op": op.lower(), "name": name, "output": {"scale": s_, "zero_point": z_}}
        else:
            raise Refused("op %s (%r) is not supported: the NPU path takes CONV_2D, MAX/AVERAGE_POOL_2D, FULLY_CONNECTED, "
                          "ADD, CONCATENATION and RELU; RESHAPE and SOFTMAX run on the CPU at the end" % (op, name))
        L["id"] = "t%d" % o["outputs"][0]
        L["inputs"] = ["t%d" % t for t in data]
        g["layers"].append(L)
        known.add(o["outputs"][0])
        last = o["outputs"][0]
    if last != tf["outputs"][0]:
        raise Refused("the last op's output is not the model's output")
    return g


# ---- graph JSON -> lowered (what the NPU computes; uint8 asymmetric throughout, OIHW weights) -----------------
def _u8zp(dtype, zp):
    if dtype == "int8":
        if not -128 <= zp <= 127:
            raise Refused("int8 zero point %d" % zp)
        return zp + 128
    if dtype == "uint8":
        if not 0 <= zp <= 255:
            raise Refused("uint8 zero point %d" % zp)
        return zp
    raise Refused("dtype %s (int8 or uint8)" % dtype)


def _out_shape(h, k, s, padding):
    """TFLite's (= NPUC's, V: NNConvLayer::computeOutputShape) output size and pads (top/left, bottom/right)"""
    if padding == "valid":
        o = (h - k) // s + 1
        return o, 0, 0
    o = (h + s - 1) // s
    tot = (o - 1) * s + k - h
    if tot < 0:
        raise Refused("SAME padding with stride %d > kernel %d leaves a negative pad (NPUC would compute %d)" % (s, k, tot))
    return o, tot // 2, tot - tot // 2


def _clean(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:40]


def _resolve(g):
    """the graph's layers with explicit inputs: [[layer, key, input keys]], the network input's key, warnings, and the
    names to show for the keys. Keys are internal ("#<position>", "#in"); a layer names a tensor by the network input's
    or a layer's "id" (tflite_to_graph) or "name" (the latest layer of that name); without "input"/"inputs" a layer
    reads the previous layer's output (a chain). A standalone RELU/RELU6 whose input is a CONV_2D/FULLY_CONNECTED
    without activation, read by that RELU alone, in the same quantisation, is folded into the conv's fused activation
    (NPUC merges nothing on device: docs/harness-ops.md 3.3)."""
    io = g["input"]
    names = {str(io.get("id") or io.get("name") or "input"): "#in"}
    show = {"#in": str(io.get("name") or io.get("id") or "input")}
    ents, prev = [], "#in"
    for i, L in enumerate(g["layers"]):
        key = "#%d" % (i + 1)
        refs = [str(v) for v in L["inputs"]] if "inputs" in L else [str(L["input"])] if "input" in L else None
        if refs is None:
            ins = [prev]
        else:
            for r in refs:
                if r not in names:
                    raise Refused("layer %r (%s) reads %r, which is neither the network input nor an earlier layer's output"
                                  % (L.get("name", key), L.get("op"), r))
            ins = [names[r] for r in refs]
        ents.append([L, key, ins])
        for r in (L.get("name"), L.get("id")):
            if r is not None:
                names[str(r)] = key
        show[key] = str(L.get("name") or L.get("id") or key)
        prev = key
    readers = {}
    for _, _, ins in ents:
        for k in ins:
            readers[k] = readers.get(k, 0) + 1
    byk = {e[1]: e for e in ents}
    alias, out, warns = {}, [], []
    for e in ents:
        L, key, ins = e
        e[2] = ins = [alias.get(k, k) for k in ins]
        P = byk[ins[0]][0] if len(ins) == 1 and ins[0] in byk else None
        if L["op"] in ("relu", "relu6") and P is not None and P["op"] in ("conv2d", "fully_connected") \
                and P.get("activation", "none") == "none" and readers.get(ins[0]) == 1:
            q, pq = L.get("output") or {}, P.get("output") or {}
            if q.get("scale") is None or (float(q["scale"]), int(q["zero_point"])) == (float(pq["scale"]), int(pq["zero_point"])):
                byk[ins[0]][0] = dict(P, activation=L["op"])
                alias[key] = ins[0]
                warns.append("%s: standalone %s folded into the fused activation of %s (NPUC does not merge them on device)"
                             % (show[key], L["op"].upper(), show[ins[0]]))
                continue
        out.append(e)
    return out, "#in", warns, show


def _fused_act(name, what, act, zp, scale, low):
    """a fused activation on an op whose activation becomes a relu layer: none or relu (RELU6 when its clamp is a no-op)"""
    if act == "relu6":
        if zp + 6.0 / scale >= 255:
            low["warnings"].append("%s: RELU6 taken as RELU (6.0 is above the output range: the clamp is a no-op)" % name)
            return "relu"
        raise Refused("%s: %s with RELU6 clamping inside the output range: NPUC's standalone ReLU6 clamps at a fixed "
                      "accumulator value (0x60) whatever the scales (docs/format/grelu.md 7)" % (name, what))
    if act not in ("none", "relu"):
        raise Refused("%s: %s activation %s (none or relu)" % (name, what, act))
    return act


def _nonneg(low, blob):
    """True when every value of this lowered tensor is >= its zero point: the output of a ReLU conv or ReLU layer, or
    a pool of such a tensor"""
    for l in low["layers"]:
        if l["name"] == blob:
            if l["kind"] == "pool":
                return _nonneg(low, l["input"])
            return l.get("act") == 1
    return False


def _global_avg_as_conv(L, cur, dtype, low):
    """A global AVERAGE_POOL_2D (VALID, its kernel the whole map) as a conv NPUC accepts: NPUC's pooling takes strides
    up to 3 only (V: 4 refused, 8 aborts), but a VALID conv with kernel up to 14x14 compiles (V, the oracle: 5..14 on
    16 channels; 16x16 trips "(bmSize & 0x80) == 0"). Identity weights (w[o][o] = 1), bias 0, weight scale
    1/(kh*kw): acc = the sum, NPUC's multiplier 1/(kh*kw) (exact when kh*kw is a power of two). ReLU when the input
    is already >= its zero point (a no-op there, and it avoids NPUC's 'no activation' table), else no activation.
    None when the pool is not of that kind (the caller lowers it as a pool, or refuses it)."""
    kh, kw = L["pool"]
    C, H, W = cur["chw"]
    sh, sw = L.get("strides", [kh, kw])
    outq = L.get("output") or {}
    zpm = cur["zp"] - 128 if dtype == "int8" else cur["zp"]
    same_q = (outq.get("scale") in (None, cur["scale"]) or f32(float(outq["scale"])) == cur["scale"]) and \
        outq.get("zero_point") in (None, zpm)
    if not ((kh, kw) == (H, W) and L.get("padding", "valid").lower() == "valid" and max(sh, sw) > 3 and same_q):
        return None
    if max(kh, kw) > 14:
        raise Refused("%s: a global average pool over %dx%d: NPUC's pooling takes strides up to 3, and as a conv its "
                      "kernel may be up to 14x14 (V); pool in steps first" % (L.get("name", "pool"), kh, kw))
    if C * C * kh * kw > 1 << 24:
        raise Refused("%s: a global average pool over %d channels as a conv needs %d x %d x %d x %d weights (NPUC "
                      "compiles 512 x 512 x 7 x 7 in seconds; 1024 channels did not finish): reduce the channels first"
                      % (L.get("name", "pool"), C, C, C, kh, kw))
    act = "relu" if _nonneg(low, cur["name"]) else "none"
    data = [1 if c == o else 0 for o in range(C) for i in range(kh) for j in range(kw) for c in range(C)]
    low["warnings"].append("%s: global average pool %dx%d lowered to a %dx%d VALID conv with identity weights and "
                           "multiplier 1/%d (%s; activation %s)" % (L.get("name", "pool"), kh, kw, kh, kw, kh * kw,
                                                                    "exact" if kh * kw & (kh * kw - 1) == 0 else
                                                                    "NPUC's slope approximates it", act))
    low["unproven"].append("%s: global average pool as a conv (not run on the NPU)" % L.get("name", "pool"))
    return {"op": "conv2d", "name": L.get("name", "gap"), "filters": C, "kernel": [kh, kw], "strides": [1, 1],
            "padding": "valid", "activation": act,
            "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1.0 / (kh * kw), "zero_point": 0, "data": data},
            "bias": [0] * C, "output": {"scale": cur["scale"], "zero_point": zpm}}


class _Lowering:
    """lower()'s state: the lowered network being built, the options, and the NPU blob of each graph tensor (its
    lowered name, CHW, float32 scale and u8 zero point)"""

    def __init__(self, low, dtype, quant, fold, allow_none, ents, show):
        self.low, self.dtype, self.quant, self.fold, self.allow_none = low, dtype, quant, fold, allow_none
        self.ents, self.show = ents, show
        self.blobs = {}
        self.tail = None        # the key of the last CPU-tail op

    def u8zp(self, zp):
        return _u8zp(self.dtype, zp)

    def relu_layer(self, name, src, frm, src_name):
        """a standalone ReLU layer (npuc-net 2 "relu"): NPUC's 1x1 dummy conv, the input's quantisation kept"""
        low = self.low
        low["layers"].append({"kind": "relu", "name": name, "type": "ReLU", "from": frm, "src_name": src_name,
                              "input": src["name"], "in_chw": list(src["chw"]), "cout": src["chw"][0],
                              "in_scale": f32_hex(src["scale"]), "in_zp": src["zp"], "out_scale": f32_hex(src["scale"]),
                              "out_zp": src["zp"], "act": 1, "pads": [0, 0, 0, 0], "out_chw": list(src["chw"])})
        low["unproven"].append("%s: standalone ReLU (NPUC: a 1x1 dummy conv with all-zero weights, GReLU mode 1, no table; "
                               "not run on the NPU): the reference is max(x, zero point); compare also tries the identity "
                               "and a constant zero point" % name)
        return {"name": name, "chw": list(src["chw"]), "scale": src["scale"], "zp": src["zp"]}


# lower()'s steps, one per op: each appends its layer(s) to lw.low and returns the blob of its output
def _lower_conv(lw, idx, L, op, name, key, ins, srcs):
    """CONV_2D, or FULLY_CONNECTED as a conv whose kernel covers its input: OIHW uint8 weights with one scale and
    zero point 128, NPUC's multiplier, TFLite's output size and pads"""
    cur = srcs[0]
    wt = L["weights"]
    wd = wt["dtype"]
    wsc = [f32(float(v)) for v in (wt["scale"] if isinstance(wt["scale"], list) else [wt["scale"]])]
    wzp = [int(v) for v in (wt.get("zero_point", [0]) if isinstance(wt.get("zero_point", [0]), list) else [wt["zero_point"]])]
    data = list(wt["data"])
    bias = [int(v) for v in L["bias"]]
    Cc, Hc, Wc = cur["chw"]
    if op == "conv2d":
        cout = int(L["filters"])
        kh, kw = L["kernel"]
        sh, sw = L.get("strides", [1, 1])
        pad = L.get("padding", "same").lower()
        if wt.get("layout", "OHWI") != "OHWI":
            raise Refused("conv %s: weights layout %s (OHWI wanted)" % (name, wt.get("layout")))
        if len(data) != cout * kh * kw * Cc:
            raise Refused("conv %s: %d weights for %d x %d x %d x %d" % (name, len(data), cout, kh, kw, Cc))
        # OHWI -> OIHW
        oihw = [data[((o * kh + i) * kw + j) * Cc + c] for o in range(cout) for c in range(Cc) for i in range(kh) for j in range(kw)]
    else:
        cout = int(L["units"])
        K = Cc * Hc * Wc
        if len(data) != cout * K:
            raise Refused("fully_connected %s: %d weights for %d x %d" % (name, len(data), cout, K))
        # TFLite flattens NHWC: k = (h * W + w) * C + c; the lowered conv's kernel covers H x W (VALID)
        kh, kw, sh, sw, pad = Hc, Wc, 1, 1, "valid"
        oihw = [data[o * K + (i * Wc + j) * Cc + c] for o in range(cout) for c in range(Cc) for i in range(kh) for j in range(kw)]
        lw.low["unproven"].append("%s: FULLY_CONNECTED lowered to a %dx%d VALID conv (not yet run on the NPU)" % (name, kh, kw))
    if len(bias) != cout:
        raise Refused("%s: %d biases for %d outputs" % (name, len(bias), cout))
    # weights to uint8 with one zero point 128 and one scale
    if wd == "int8":
        if any(z != 0 for z in wzp):
            raise Refused("%s: int8 weights with zero point %s (symmetric 0 wanted)" % (name, wzp))
        wu8 = [v + 128 for v in oihw]
    elif wd == "uint8":
        if any(z != 128 for z in wzp):
            raise Refused("%s: uint8 weights with zero point %s: NPUC's on-device path is validated with 128 only" % (name, wzp))
        wu8 = list(oihw)
    else:
        raise Refused("%s: weights dtype %s" % (name, wd))
    if len(set(wsc)) > 1:
        if not lw.fold:
            raise Refused("%s: %d per-channel weight scales: NPUC's on-device filler uses one scale per layer "
                          "(wScale[0]); --fold-per-channel rescales the weights to the largest (lossy)" % (name, len(wsc)))
        smax = max(wsc)
        kper = len(wu8) // cout
        worst = 0
        for o in range(cout):
            r = Fraction(wsc[o]) / Fraction(smax)
            for t in range(o * kper, (o + 1) * kper):
                nv = rnd((wu8[t] - 128) * r, "half_away")
                worst = max(worst, abs(Fraction(wu8[t] - 128) * r - nv))
                wu8[t] = nv + 128
            bias[o] = rnd(bias[o] * r, "half_away")
        lw.low["warnings"].append("%s: per-channel weight scales folded into %g (rounding error up to %.2f LSB of the new scale)" % (name, smax, float(worst)))
        wsc = [smax]
    elif len(wsc) != 1 and len(wsc) != cout:
        raise Refused("%s: %d weight scales" % (name, len(wsc)))
    ws = wsc[0]
    outq = L["output"]
    os_ = f32(float(outq["scale"]))
    oz = lw.u8zp(int(outq["zero_point"]))
    act = L.get("activation", "none")
    if act == "relu6":
        q6 = oz + 6.0 / os_
        if q6 >= 255:
            lw.low["warnings"].append("%s: RELU6 taken as RELU (6.0 is above the output range: the clamp is a no-op)" % name)
            act = "relu"
        else:
            raise Refused("%s: RELU6 clamps inside the output range (NPUC: 'Unsupported FusedActivation type' 3)" % name)
    if act == "none" and not lw.allow_none:
        raise Refused("%s: no fused activation: NPUC compiles it to GRELU_BYPASS_ONDEVICE, which writes the byte 0 "
                      "wherever the accumulator is >= 0 (dossier/npu-compiler.md section 9.11); use RELU, or "
                      "--allow-none to compile it anyway" % name)
    if act not in ("none", "relu"):
        raise Refused("%s: activation %s (none or relu)" % (name, act))
    slope, shift = npuc_multiplier(cur["scale"], ws, os_)
    ho, pt, pb = _out_shape(Hc, kh, sh, pad)
    wo, pl, pr = _out_shape(Wc, kw, sw, pad)
    if ho < 1 or wo < 1:
        raise Refused("%s: output %dx%d" % (name, ho, wo))
    for b in bias:
        if not -2 ** 31 <= b < 2 ** 31:
            raise Refused("%s: bias %d out of int32" % (name, b))
    lay = {"kind": "conv", "name": name, "from": op, "src_name": L.get("name", ""), "input": cur["name"],
           "in_chw": [Cc, Hc, Wc], "cout": cout, "k": [kh, kw], "stride": [sh, sw], "padding": pad.upper(),
           "pads": [pt, pb, pl, pr], "act": 1 if act == "relu" else 0,
           "in_scale": f32_hex(cur["scale"]), "in_zp": cur["zp"], "w_scale": f32_hex(ws), "w_zp": 128,
           "out_scale": f32_hex(os_), "out_zp": oz, "slope": slope, "shift": shift,
           "weights_oihw_u8": wu8, "bias": bias, "out_chw": [cout, ho, wo]}
    if act == "none":
        lw.low["warnings"].append("%s: no activation (GRELU_BYPASS_ONDEVICE: the byte 0 wherever acc >= 0, V)" % name)
    if slope & (slope - 1):
        lw.low["unproven"].append("%s: multiplier %d/2^%d is not a power of two: NPUC's slope rounds it (%.4g vs %.4g asked); "
                               "the rounding of acc x slope is proven only for a power of two" % (
                                   name, slope, shift, slope / 2 ** shift, f32(cur["scale"]) * ws / os_))
    if (sh, sw) != (1, 1):
        lw.low["unproven"].append("%s: stride %dx%d (only stride 1 run on the NPU so far)" % (name, sh, sw))
    lw.low["layers"].append(lay)
    return {"name": name, "chw": [cout, ho, wo], "scale": os_, "zp": oz}


def _lower_pool(lw, idx, L, op, name, key, ins, srcs):
    """MAX/AVERAGE_POOL_2D (NFU strides up to 3, the input's quantisation kept)"""
    cur = srcs[0]
    Cc, Hc, Wc = cur["chw"]
    kh, kw = L["pool"]
    sh, sw = L.get("strides", [kh, kw])
    if max(sh, sw) > 3:
        # V (the oracle, 2026-11-21): stride 4 -> "ncp_nfu_pooling: StrideH (4) exceed limited value", stride 8
        # -> assertion "IFM.valid()" in PolicyMaker; kernels up to 5 with strides 1..3 compile
        raise Refused("%s: pooling stride %dx%d: NPUC's NFU pooling takes a stride of at most 3 (it refuses 4, "
                      "aborts on 8); a global average pool can be split into pools of stride <= 3" % (name, sh, sw))
    pad = L.get("padding", "valid").lower()
    act = L.get("activation", "none")
    outq = L.get("output", {"scale": None, "zero_point": None})
    os_ = f32(float(outq["scale"])) if outq.get("scale") is not None else cur["scale"]
    oz = lw.u8zp(int(outq["zero_point"])) if outq.get("zero_point") is not None else cur["zp"]
    if (os_, oz) != (cur["scale"], cur["zp"]):
        raise Refused("%s: pooling with a different output quantisation (%g, %d) than its input (%g, %d)" % (name, os_, oz, cur["scale"], cur["zp"]))
    if act not in ("none", "relu"):
        raise Refused("%s: pooling activation %s" % (name, act))
    ho, pt, pb = _out_shape(Hc, kh, sh, pad)
    wo, pl, pr = _out_shape(Wc, kw, sw, pad)
    ptype = "MaxPool" if op == "max_pool2d" else "AvgPool"
    if ptype == "AvgPool":
        lw.low["unproven"].append("%s: AVERAGE_POOL_2D (the NPU's division and rounding are not measured yet; the "
                               "reference uses TFLite's: round half up of sum / count of valid inputs)" % name)
    if pad == "same" and (pt or pb or pl or pr):
        lw.low["unproven"].append("%s: padded pooling (not run on the NPU yet)" % name)
    lw.low["layers"].append({"kind": "pool", "name": name, "type": ptype, "src_name": L.get("name", ""),
                          "input": cur["name"], "in_chw": [Cc, Hc, Wc], "k": [kh, kw], "stride": [sh, sw],
                          "padding": pad.upper(), "pads": [pt, pb, pl, pr], "act": 1 if act == "relu" else 0,
                          "in_scale": f32_hex(cur["scale"]), "in_zp": cur["zp"], "out_scale": f32_hex(os_),
                          "out_zp": oz, "out_chw": [Cc, ho, wo]})
    return {"name": name, "chw": [Cc, ho, wo], "scale": os_, "zp": oz}


def _lower_add(lw, idx, L, op, name, key, ins, srcs):
    """ADD as NPUC's Eltwise: quant 3, every scale 2^-n (its fractional lengths), no broadcasting; a fused RELU
    becomes a relu layer"""
    if len(ins) != 2:
        raise Refused("add %s: %d inputs (two)" % (name, len(ins)))
    a, b = srcs
    if a["name"] == b["name"]:
        raise Refused("add %s: both inputs are %r: x + x is not supported (NPUC's policy maker asserts, PGraph::clone "
                      "\"this->equals(graph)\": docs/harness-ops.md 3.1)" % (name, lw.show[ins[0]]))
    if a["chw"] != b["chw"]:
        raise Refused("add %s: inputs %s and %s (C x H x W): broadcasting is not supported (NPUC's Eltwise adds two "
                      "tensors of the same shape)" % (name, "x".join(map(str, a["chw"])), "x".join(map(str, b["chw"]))))
    if lw.quant != 3:
        raise Refused("add %s: ADD needs --quant 3 (native s8, every zero point int8 0): NPUC's Eltwise has no "
                      "zero-point term (docs/harness-ops.md 3.1)" % name)
    outq = L["output"]
    os_ = f32(float(outq["scale"]))
    oz = lw.u8zp(int(outq["zero_point"]))
    fl = []
    for what, s_ in (("input 1 (%s)" % lw.show[ins[0]], a["scale"]), ("input 2 (%s)" % lw.show[ins[1]], b["scale"]), ("output", os_)):
        n_ = pow2_fl(s_)
        if n_ is None:
            raise Refused("add %s: %s scale %.9g is not 2^-n (n 0..31): NPUC's Eltwise aligns its inputs by powers "
                          "of two only and cannot rescale (docs/harness-ops.md 3.1); quantise the model with "
                          "power-of-two scales for the ADD's inputs and output" % (name, what, s_))
        fl.append(n_)
    if not a["zp"] == b["zp"] == oz:
        raise Refused("add %s: zero points %d, %d, %d (u8) differ: NPUC's Eltwise has no zero-point term" % (name, a["zp"], b["zp"], oz))
    act = _fused_act(name, "ADD", L.get("activation", "none"), oz, os_, lw.low)
    lw.low["layers"].append({"kind": "eltwise", "name": name, "op": "Add", "from": "add", "src_name": L.get("name", ""),
                          "input": a["name"], "inputs": [a["name"], b["name"]], "in_chw": list(a["chw"]),
                          "in_scales": [f32_hex(a["scale"]), f32_hex(b["scale"])], "in_zps": [a["zp"], b["zp"]],
                          "fl": fl, "out_scale": f32_hex(os_), "out_zp": oz, "act": 0, "pads": [0, 0, 0, 0],
                          "out_chw": list(a["chw"])})
    lw.low["unproven"].append("%s: ADD as NPUC's Eltwise (compiles; not run on the NPU): the reference is the harness's "
                           "guess sat_s8(round(in1 x 2^%d + in2 x 2^%d)) (fractional lengths %d, %d -> %d), rounding "
                           "half away; compare also tries floor, half even/up and rounding each term" % (
                               name, fl[2] - fl[0], fl[2] - fl[1], fl[0], fl[1], fl[2]))
    blob = {"name": name, "chw": list(a["chw"]), "scale": os_, "zp": oz}
    if act == "relu":
        blob = lw.relu_layer("l%d_%s" % (len(lw.low["layers"]) + 1, _clean(L.get("name", op) + "_relu")), blob, "add",
                          L.get("name", ""))
    return blob


def _lower_concat(lw, idx, L, op, name, key, ins, srcs):
    """CONCATENATION of channels, every input already in the output's quantisation"""
    axis = int(L.get("axis", 3))
    if axis not in (3, -1):
        raise Refused("concat %s: axis %d of NHWC: NPUC concatenates channels only (axis 3 / -1)" % (name, axis))
    if not 2 <= len(ins) <= 64:
        raise Refused("concat %s: %d inputs (2 to 64)" % (name, len(ins)))
    if len(set(b["name"] for b in srcs)) != len(srcs):
        raise Refused("concat %s: the same tensor twice (not supported)" % name)
    outq = L["output"]
    os_ = f32(float(outq["scale"]))
    oz = lw.u8zp(int(outq["zero_point"]))
    for k, b in zip(ins, srcs):
        if b["name"] == "data0":
            raise Refused("concat %s: input %r is the network input: NPUC writes it straight into a channel slice "
                          "and its policy maker asserts (\"DataNode id isn't found in the NNGraph inputs\", "
                          "docs/harness-ops.md 3.2); a layer must come before it" % (name, lw.show[k]))
        if (b["scale"], b["zp"]) != (os_, oz):
            raise Refused("concat %s: input %r is scale %.9g zp %d, the output scale %.9g zp %d: NPUC places the bytes "
                          "as they are (no rescale on this path); every input must already be in the output's "
                          "quantisation" % (name, lw.show[k], b["scale"], b["zp"], os_, oz))
        if b["chw"][1:] != srcs[0]["chw"][1:]:
            raise Refused("concat %s: input %r is %dx%d (H x W), %r is %dx%d" % (
                name, lw.show[k], b["chw"][1], b["chw"][2], lw.show[ins[0]], srcs[0]["chw"][1], srcs[0]["chw"][2]))
    act = _fused_act(name, "CONCATENATION", L.get("activation", "none"), oz, os_, lw.low)
    chw = [sum(b["chw"][0] for b in srcs)] + list(srcs[0]["chw"][1:])
    lw.low["layers"].append({"kind": "concat", "name": name, "from": "concat", "src_name": L.get("name", ""),
                          "input": srcs[0]["name"], "inputs": [b["name"] for b in srcs], "axis": 1,
                          "in_chws": [list(b["chw"]) for b in srcs], "out_scale": f32_hex(os_), "out_zp": oz,
                          "act": 0, "pads": [0, 0, 0, 0], "out_chw": chw})
    lw.low["unproven"].append("%s: CONCATENATION of channels (compiles; not run on the NPU): the reference places the "
                           "inputs' channels in order" % name)
    blob = {"name": name, "chw": chw, "scale": os_, "zp": oz}
    if act == "relu":
        blob = lw.relu_layer("l%d_%s" % (len(lw.low["layers"]) + 1, _clean(L.get("name", op) + "_relu")), blob, "concat",
                          L.get("name", ""))
    return blob


def _lower_relu(lw, idx, L, op, name, key, ins, srcs):
    """a standalone RELU (RELU6 when its clamp is a no-op): NPUC's relu layer"""
    cur = srcs[0]
    outq = L.get("output") or {}
    if outq.get("scale") is not None:
        os_ = f32(float(outq["scale"]))
        oz = lw.u8zp(int(outq["zero_point"]))
        if (os_, oz) != (cur["scale"], cur["zp"]):
            raise Refused("%s %s: output scale %.9g zp %d, its input scale %.9g zp %d: NPUC's ReLU layer keeps its "
                          "input's quantisation" % (op.upper(), name, os_, oz, cur["scale"], cur["zp"]))
    if op == "relu6":
        _fused_act(name, "RELU6", "relu6", cur["zp"], cur["scale"], lw.low)
    return lw.relu_layer(name, cur, op, L.get("name", ""))


def _lower_reshape(lw, idx, L, op, name, key, ins, srcs):
    """RESHAPE: kept for a FULLY_CONNECTED that reads it, else a CPU-tail op"""
    cur = srcs[0]
    n = 1
    for v in L["shape"]:
        n *= v
    if n != cur["chw"][0] * cur["chw"][1] * cur["chw"][2]:
        raise Refused("reshape %s to %s: %d elements, the tensor has %d" % (name, L["shape"], n, cur["chw"][0] * cur["chw"][1] * cur["chw"][2]))
    nxt = lw.ents[idx + 1] if idx + 1 < len(lw.ents) else None
    if nxt is not None and nxt[0]["op"] == "fully_connected" and nxt[2] == [key]:
        return dict(cur, flat=True)
    else:
        if lw.tail is None and lw.low["layers"] and cur["name"] != lw.low["layers"][-1]["name"]:
            raise Refused("reshape %s reads %r: the CPU part (RESHAPE, SOFTMAX) must read the NPU part's last output" % (name, lw.show[ins[0]]))
        if lw.tail is not None and ins[0] != lw.tail:
            raise Refused("reshape %s reads %r: the CPU part must be a chain at the end" % (name, lw.show[ins[0]]))
        lw.tail = key
        lw.low["cpu_tail"].append({"op": "reshape", "shape": L["shape"]})
        return dict(cur, tail=True)


def _lower_softmax(lw, idx, L, op, name, key, ins, srcs):
    """SOFTMAX: a CPU-tail op"""
    cur = srcs[0]
    if lw.tail is None and lw.low["layers"] and cur["name"] != lw.low["layers"][-1]["name"]:
        raise Refused("softmax %s reads %r: the CPU part (RESHAPE, SOFTMAX) must read the NPU part's last output" % (name, lw.show[ins[0]]))
    if lw.tail is not None and ins[0] != lw.tail:
        raise Refused("softmax %s reads %r: the CPU part must be a chain at the end" % (name, lw.show[ins[0]]))
    lw.tail = key
    outq = L["output"]
    lw.low["cpu_tail"].append({"op": "softmax", "beta": float(L.get("beta", 1.0)), "in_scale": f32_hex(cur["scale"]),
                            "in_zp": cur["zp"], "out_scale": f32_hex(f32(float(outq["scale"]))),
                            "out_zp": lw.u8zp(int(outq["zero_point"]))})
    return dict(cur, tail=True)


LOWER_OPS = {"conv2d": _lower_conv, "fully_connected": _lower_conv, "max_pool2d": _lower_pool,
             "average_pool2d": _lower_pool, "add": _lower_add, "concat": _lower_concat, "relu": _lower_relu,
             "relu6": _lower_relu, "reshape": _lower_reshape, "softmax": _lower_softmax}


def lower(g, quant=1, fold=False, allow_none=False, soc=2):
    """an npu-net/1 graph -> lowered.json's dict: every layer as NPUC computes it (uint8 asymmetric, OIHW weights,
    NPUC's multipliers and pads), the CPU tail, the warnings and what is not proven on the NPU; Refused for what NPUC
    or this path cannot take (the module docstring)"""
    if g.get("format") != "npu-net/1":
        raise Refused("not an npu-net/1 graph (format %r)" % g.get("format"))
    io = g["input"]
    dtype = io["dtype"]
    shp = io["shape"]
    if len(shp) != 4 or shp[0] != 1:
        raise Refused("input shape %s: [1, H, W, C] wanted" % shp)
    H, W, C = shp[1], shp[2], shp[3]
    zp_in = _u8zp(dtype, int(io["zero_point"]))
    sc = f32(float(io["scale"]))
    low = {"format": "npu-lowered/1", "name": g.get("name", "net"), "quant_mode": quant, "soc": soc,
           "io_dtype": dtype, "input": {"name": "data0", "chw": [C, H, W], "scale": f32_hex(sc), "zp": zp_in,
                                        "nhwc": [1, H, W, C]},
           "layers": [], "cpu_tail": [], "warnings": [], "unproven": []}
    ents, inkey, fwarn, show = _resolve(g)
    low["warnings"] += fwarn
    lw = _Lowering(low, dtype, quant, fold, allow_none, ents, show)
    lw.blobs[inkey] = {"name": "data0", "chw": [C, H, W], "scale": sc, "zp": zp_in}
    for idx, (L, key, ins) in enumerate(ents):
        op = L["op"]
        name = "l%d_%s" % (len(low["layers"]) + 1, _clean(L.get("name", op)))
        if lw.tail is not None and op not in ("reshape", "softmax"):
            raise Refused("%s after a CPU op: only RESHAPE and SOFTMAX may follow the NPU part" % op)
        if op not in ("add", "concat") and len(ins) != 1:
            raise Refused("%s %s: %d inputs (one)" % (op, name, len(ins)))
        srcs = [lw.blobs[k] for k in ins]
        for k, b in zip(ins, srcs):
            if b.get("flat") and op != "fully_connected":
                raise Refused("%s %s reads %r, a RESHAPE kept for a FULLY_CONNECTED: only a FULLY_CONNECTED may read it"
                              % (op, name, show[k]))
        if op == "average_pool2d":
            gap = _global_avg_as_conv(L, srcs[0], dtype, low)
            if gap is not None:
                L, op = gap, "conv2d"
        if op not in LOWER_OPS:
            raise Refused("op %s is not supported (conv2d, fully_connected, max_pool2d, average_pool2d, add, concat, relu; "
                          "reshape and softmax at the end)" % op)
        lw.blobs[key] = LOWER_OPS[op](lw, idx, L, op, name, key, ins, srcs)
    if not low["layers"]:
        raise Refused("no layer runs on the NPU")
    if not any(l["kind"] == "conv" for l in low["layers"]):
        raise Refused("no conv or fully-connected layer: NPUC compiles it (no WEIGHT vector), but N2's npu-lite refuses an NCP "
                      "without weights at ncp map (\"weights end 0\", npu-run v1-pool 2026-11-21; the driver change is in "
                      "journal/plans/npu-run-0218/README.md)")
    # one network output: NPUC makes every blob no layer reads an output of the network
    read = set(n_ for l in low["layers"] for n_ in l.get("inputs", [l["input"]]))
    for l in low["layers"][:-1]:
        if l["name"] not in read:
            raise Refused("%s (%s) is read by no later layer: NPUC would make it a second network output (one output "
                          "supported; the network's output is its last layer)" % (l["name"], l.get("src_name") or l["kind"]))
    # two suspected NPUC defects (G until the NPU runs P9 / P11 of docs/format/frame.md 8.5): named, not refused
    lastl = low["layers"][-1]
    if lastl["out_chw"][0] in (3, 5, 7):
        low["unproven"].append("%s: an NPU output with %d channels: NPUC reads it back with a negative plane stride (IM "
                               "c_stride -2 x plane, docs/format/frame.md 9.1): channels 2.. are likely wrong"
                               % (lastl["name"], lastl["out_chw"][0]))
    for i, l in enumerate(low["layers"]):
        if l["kind"] == "pool" and not any(l["pads"]):
            (_, Hc, Wc), (kh, kw), (sh, sw), (_, ho, wo) = l["in_chw"], l["k"], l["stride"], l["out_chw"]
            if (wo - 1) * sw + kw - Wc < 0 <= (ho - 1) * sh + kh - Hc:
                rd = [m["name"] for m in low["layers"][i + 1:] if l["name"] in m.get("inputs", [m["input"]])]
                if rd:
                    low["unproven"].append(
                        "%s: VALID pool on width %d with its height covered: NPUC computes one more output column (%d, "
                        "not %d) and %s reads that wider map (docs/format/cmdq.md 7, frame.md 9.2): its last "
                        "column is likely wrong (planes overlap when the extra column pushes the width past a multiple "
                        "of 8)" % (l["name"], Wc, wo + 1, wo, ", ".join(rd)))
    if quant == 3:
        zps = [low["input"]["zp"]] + [l["out_zp"] for l in low["layers"]]
        if any(z != 128 for z in zps):
            raise Refused("quant mode 3 (native s8) ignores zero points: every zero point must be 128 (int8 0); got %s: use --quant 1" % sorted(set(zps)))
    elif quant != 1:
        raise Refused("quant mode %d (1 or 3)" % quant)
    last = low["layers"][-1]
    low["npu_output"] = last["name"]
    low["npu_out_chw"] = last["out_chw"]
    C2, H2, W2 = last["out_chw"]
    shape = [1, H2, W2, C2]
    for t in low["cpu_tail"]:
        if t["op"] == "reshape":
            shape = t["shape"]
    low["output"] = {"shape": shape, "dtype": dtype, "bytes": C2 * H2 * W2}
    if is_graph(low):
        low["npuc_net"] = 2
    return low


is_graph = has_graph_ops     # a network with a layer only "npuc-net 2" files carry


def npuc_text(low):
    """the input of oracle/harness/npuc-net.cpp: "npuc-net 1" for conv/pool layers (as always), "npuc-net 2" when
    the network has an eltwise, concat or relu layer (docs/harness-ops.md 4)"""
    out = ["npuc-net %d" % (2 if is_graph(low) else 1), "# %s (npu/npunet.py lower)" % low["name"], "soc %d" % low["soc"],
           "quant %d" % low["quant_mode"], "input data0 %d %d %d" % tuple(low["input"]["chw"])]
    for l in low["layers"]:
        if l["kind"] == "conv":
            out.append("conv %s %s %d %d %d %d %d %s %d %s %d %s %d %s %d %s %d %s" % (
                l["name"], l["input"], l["cout"], l["k"][0], l["k"][1], l["stride"][0], l["stride"][1], l["padding"],
                l["act"], l["in_scale"], l["in_zp"], l["w_scale"], l["w_zp"], l["out_scale"], l["out_zp"],
                bytes(l["weights_oihw_u8"]).hex(), len(l["bias"]), " ".join(str(b) for b in l["bias"])))
        elif l["kind"] == "pool":
            out.append("pool %s %s %s %d %d %d %d %s %d %s %d %s %d" % (
                l["name"], l["input"], l["type"], l["k"][0], l["k"][1], l["stride"][0], l["stride"][1], l["padding"],
                l["act"], l["in_scale"], l["in_zp"], l["out_scale"], l["out_zp"]))
        elif l["kind"] == "eltwise":
            out.append("eltwise %s %s %s %s %s %d %s %d %s %d" % (
                l["name"], l["inputs"][0], l["inputs"][1], l["op"], l["in_scales"][0], l["in_zps"][0], l["in_scales"][1],
                l["in_zps"][1], l["out_scale"], l["out_zp"]))
        elif l["kind"] == "concat":
            out.append("concat %s %d %d %s %s %d" % (l["name"], l["axis"], len(l["inputs"]), " ".join(l["inputs"]),
                                                     l["out_scale"], l["out_zp"]))
        elif l["kind"] == "relu":
            out.append("relu %s %s %s" % (l["name"], l["input"], l["type"]))
        else:
            raise Refused("layer %s: kind %s" % (l["name"], l["kind"]))
    out.append("end")
    return "\n".join(out) + "\n"


def load_model(path, quant=1, fold=False, allow_none=False):
    data = open(path, "rb").read()
    tfl = is_tflite(data) and not data.lstrip()[:1] == b"{"
    if tfl:
        g = tflite_to_graph(read_tflite(data), path)
    else:
        try:
            g = json.loads(data)
        except ValueError:
            raise Refused("%s is neither a TFLite file nor JSON" % path)
    low = lower(g, quant=quant, fold=fold, allow_none=allow_none)
    low["source"] = {"path": os.path.basename(path), "sha256": hashlib.sha256(data).hexdigest(),
                     "kind": "tflite" if tfl else "npu-net/1 json"}
    return low


# ---- the CPU reference -------------------------------------------------------------------------------------------
def ref_conv(l, x, rounding, none_defect, quant):
    C, H, W = l["in_chw"]
    cout, (kh, kw), (sh, sw) = l["cout"], l["k"], l["stride"]
    pt, pb, pl, pr = l["pads"]
    _, ho, wo = l["out_chw"]
    zi, zw, zo = l["in_zp"], l["w_zp"], l["out_zp"]
    Hp, Wp = H + pt + pb, W + pl + pr
    # centred input with the border (= the zero point: 0 once centred)
    xp = [[[0] * Wp for _ in range(Hp)] for _ in range(C)]
    for c in range(C):
        for y in range(H):
            row, dst = x[(c * H + y) * W:(c * H + y + 1) * W], xp[c][y + pt]
            for xx in range(W):
                dst[xx + pl] = row[xx] - zi
    wb = l["weights_oihw_u8"]
    den = 1 << l["shift"]
    out = bytearray(cout * ho * wo)
    for o in range(cout):
        wk = [[[wb[((o * C + c) * kh + i) * kw + j] - zw for j in range(kw)] for i in range(kh)] for c in range(C)]
        for y in range(ho):
            for xx in range(wo):
                s = l["bias"][o]
                for c in range(C):
                    pc, wc = xp[c], wk[c]
                    for i in range(kh):
                        row, wr = pc[y * sh + i], wc[i]
                        base = xx * sw
                        for j in range(kw):
                            s += row[base + j] * wr[j]
                q = zo + rnd(Fraction(s * l["slope"], den), rounding)
                q = min(255, max(0, q))
                if l["act"] == 1:
                    q = max(q, zo)
                elif none_defect and s >= 0:
                    q = 0 if quant == 1 else 128  # the byte 0: u8 0 in mode 1, s8 0 (= u8 128) in mode 3
                out[(o * ho + y) * wo + xx] = q
    return bytes(out)


def ref_pool(l, x):
    C, H, W = l["in_chw"]
    (kh, kw), (sh, sw) = l["k"], l["stride"]
    pt, _, pl, _ = l["pads"]
    _, ho, wo = l["out_chw"]
    out = bytearray(C * ho * wo)
    for c in range(C):
        for y in range(ho):
            for xx in range(wo):
                vals = [x[(c * H + yy) * W + xw] for yy in range(y * sh - pt, y * sh - pt + kh) if 0 <= yy < H
                        for xw in range(xx * sw - pl, xx * sw - pl + kw) if 0 <= xw < W]
                if l["type"] == "MaxPool":
                    q = max(vals)
                else:
                    q = (sum(vals) + len(vals) // 2) // len(vals)
                if l["act"] == 1:
                    q = max(q, l["out_zp"])
                out[(c * ho + y) * wo + xx] = q
    return bytes(out)


ADD_VARIANTS = ("sum", "sum-floor", "each", "each-floor")  # the Eltwise's arithmetic tried by compare; the first: the reference
RELU_VARIANTS = ("max", "identity", "zp")  # a standalone ReLU layer read as max(x, zp) (the reference), x, or zp


def rshift_round(n, d, mode):
    """n / 2^d rounded (d >= 0), as rnd() on Fraction(n, 2^d), in integers"""
    if d <= 0:
        return n << -d
    f = n >> d
    if mode == "floor":
        return f
    r, half = n - (f << d), 1 << (d - 1)
    if r != half:
        return f + (1 if r > half else 0)
    if mode == "half_up":
        return f + 1
    if mode == "half_even":
        return f + (f & 1)
    return f + 1 if n > 0 else f  # half_away


def ref_eltwise(l, a, b, rounding="half_away", variant="sum"):
    """NPUC's Eltwise Add as the harness guesses it (docs/harness-ops.md 3.1, G): on the s8 values (quant 3:
    the byte XOR 0x80 = u8 - 128), out = sat_s8(R(a 2^(flo - fl1) + b 2^(flo - fl2))). variant "sum": R of the exact
    sum with `rounding`; "each": each term rounded, then added; "-floor": floor (an arithmetic shift) for R"""
    fl1, fl2, flo = l["fl"]
    mode = "floor" if variant.endswith("floor") else rounding
    e1, e2 = flo - fl1, flo - fl2
    D = max(0, -e1, -e2)
    out = bytearray(len(a))
    for i in range(len(a)):
        p, q = a[i] - 128, b[i] - 128
        if variant.startswith("each"):
            v = rshift_round(p, -e1, mode) + rshift_round(q, -e2, mode)
        else:
            v = rshift_round((p << (e1 + D)) + (q << (e2 + D)), D, mode)
        out[i] = min(127, max(-128, v)) + 128
    return bytes(out)


def ref_relu(l, x, variant="max"):
    zp = l["out_zp"]
    if variant == "identity":
        return bytes(x)
    if variant == "zp":
        return bytes([zp]) * len(x)
    return bytes(v if v > zp else zp for v in x)


def layer_deps(low):
    """per blob: the hypotheses its value depends on (its own layer and every layer before it in the graph): "r" the
    rounding (a conv, an eltwise), "d" the 'none' defect (a conv without activation), "a" the add variant, "u" the relu
    variant"""
    deps = {"data0": frozenset()}
    for l in low["layers"]:
        s = set()
        for n in l.get("inputs", [l["input"]]):
            s |= deps[n]
        k = l["kind"]
        if k == "conv":
            s.add("r")
            if l["act"] == 0:
                s.add("d")
        elif k == "eltwise":
            s |= {"r", "a"}
        elif k == "relu":
            s.add("u")
        deps[l["name"]] = frozenset(s)
    return deps


def ref_npu(low, x_chw, rounding="half_away", none_defect=False, add="sum", relu="max", memo=None):
    """the NPU part on the CPU: CHW uint8 in -> CHW uint8 out (the lowered network's last layer); the layers in their
    order, each reading its input blob(s). rounding, none_defect, add and relu are the hypotheses compare tries (the
    defaults: the reference); memo (a dict per input) shares the blobs whose hypotheses agree"""
    vals = {"data0": x_chw}
    deps = layer_deps(low) if memo is not None else None
    hyp = {"r": rounding, "d": none_defect, "a": add, "u": relu}
    for l in low["layers"]:
        if memo is not None:
            mk = (l["name"],) + tuple(hyp[c] if c in deps[l["name"]] else None for c in "rdau")
            if mk in memo:
                vals[l["name"]] = memo[mk]
                continue
        ins = [vals[n] for n in l.get("inputs", [l["input"]])]
        k = l["kind"]
        if k == "conv":
            y = ref_conv(l, ins[0], rounding, none_defect, low["quant_mode"])
        elif k == "pool":
            y = ref_pool(l, ins[0])
        elif k == "eltwise":
            y = ref_eltwise(l, ins[0], ins[1], rounding, add)
        elif k == "concat":
            y = b"".join(ins)  # CHW: channel concatenation = the inputs' bytes one after the other
        elif k == "relu":
            y = ref_relu(l, ins[0], relu)
        else:
            raise Refused("layer %s: kind %s" % (l["name"], k))
        vals[l["name"]] = y
        if memo is not None:
            memo[mk] = y
    return vals[low["layers"][-1]["name"]]


def cpu_tail(low, y_chw):
    """the NPU part's CHW uint8 output -> the model's output, NHWC (or flat) uint8"""
    C, H, W = low["npu_out_chw"]
    y = bytes(y_chw[(c * H + h) * W + w] for h in range(H) for w in range(W) for c in range(C))
    for t in low["cpu_tail"]:
        if t["op"] == "softmax":
            si, zi = hex_f32(t["in_scale"]), t["in_zp"]
            so, zo = hex_f32(t["out_scale"]), t["out_zp"]
            n = C if (H, W) != (1, 1) else len(y)
            out = bytearray()
            for b in range(0, len(y), n):
                v = y[b:b + n]
                m = max(v)
                e = [math.exp(t["beta"] * si * (q - m)) for q in v]
                s = sum(e)
                out += bytes(min(255, max(0, zo + int(math.floor(p / s / so + 0.5)))) for p in e)
            y = bytes(out)
    return y


# ---- IO: the model's bytes <-> CHW uint8 -------------------------------------------------------------------------
def model_in_to_chw(low, data):
    C, H, W = low["input"]["chw"]
    if len(data) != C * H * W:
        raise Refused("input is %d B; the model takes %d (NHWC 1x%dx%dx%d %s)" % (len(data), C * H * W, H, W, C, low["io_dtype"]))
    if low["io_dtype"] == "int8":
        data = bytes(b ^ 0x80 for b in data)
    return bytes(data[(h * W + w) * C + c] for c in range(C) for h in range(H) for w in range(W))


def out_to_model(low, y):
    return bytes(b ^ 0x80 for b in y) if low["io_dtype"] == "int8" else bytes(y)


def reference(low, data, rounding="half_away", none_defect=False):
    return out_to_model(low, cpu_tail(low, ref_npu(low, model_in_to_chw(low, data), rounding, none_defect)))


# ---- the NCP: descriptors -> the NPU's IFM and OFM formats -------------------------------------------------------
MV_TYPES = {0: "IN_FMAP", 1: "OT_FMAP", 2: "IM_FMAP", 7: "WEIGHT"}


def ncp_parse(d):
    if len(d) < 0x80:
        raise Refused("not an NCP (%d B)" % len(d))
    h = struct.unpack_from("<32I", d, 0)
    if h[0] != 0x0C0FFEE0 or h[1] != 22 or h[31] != 0xC0DEC0DE:
        raise Refused("not an NCP v22 (magic %#x version %d magic2 %#x)" % (h[0], h[1], h[31]))
    avo, avc, mvo, mvc = h[10:14]
    iov, ioc = h[19], h[20]
    av = [struct.unpack_from("<4I", d, avo + 16 * i) for i in range(avc)]
    mv = [struct.unpack_from("<8I", d, mvo + 32 * i) for i in range(mvc)]
    io = []
    for i in range(ioc):
        w = struct.unpack_from("<31I", d, iov + 124 * i)
        io.append({"src": w[9:20], "dst": w[20:31], "trig": w[4], "mode": w[5]})
    return av, mv, io


def ncp_layout(low, ncp):
    """{"ifm": {...}, "ofm": {...}} from the NCP; Refused when the format is not one this tool knows how to write.
    A conv/pool chain: frame.ncp_layout (the NCP's IN_FMAP/OT_FMAP vectors and IFM DMAs checked against frame.plan():
    every single-tile shape, any C, NPUC's split IFM loads; frame.md 8.4). A graph network, a plan frame.py does not
    model (tiled, streamed weights), or frame.py missing: ncp_layout_basic (one IFM format, 2 channels per plane)."""
    ncp_parse(ncp)  # an NCP v22, or Refused
    why = "a graph network"
    if not is_graph(low):
        from ncpgen import frame as fr     # imported here: lowering and the reference do not need the generator
        try:
            lay = fr.ncp_layout(low, ncp)
            lay["checked_by"] = "ncpgen/frame.py"
            return lay
        except ValueError as e:
            raise Refused("the NCP's input/output formats are not the ones modelled for this network "
                          "(ncpgen/frame.py): %s" % e)
        except NotImplementedError as e:
            why = "frame.py does not model this plan (%s)" % e
        except Exception as e:  # a failure inside frame.py (it is still being extended): the basic check decides
            why = "frame.py failed (%s: %s)" % (type(e).__name__, e)
    lay = ncp_layout_basic(low, ncp)
    lay["checked_by"] = "npunet basic check (%s)" % why
    return lay


def ncp_layout_basic(low, ncp):
    """{"ifm": {...}, "ofm": {...}} from the NCP's memory vectors and the IFM's io descriptor(s): one IFM format
    (format 0x120, n 1, ceil(C/2) planes of Hp x 2Wp), read by one or several layers; Refused otherwise"""
    av, mv, io = ncp_parse(ncp)
    C, H, W = low["input"]["chw"]
    ins = [(i, m) for i, m in enumerate(mv) if m[0] == 0]
    outs = [(i, m) for i, m in enumerate(mv) if m[0] == 1]
    if len(ins) != 1 or len(outs) != 1:
        raise Refused("the NCP has %d IN_FMAP and %d OT_FMAP memory vectors (one each supported)" % (len(ins), len(outs)))
    _, (t, pf, mw, mh, mc, ws, cs, ai) = ins[0]
    if (mw, mh, mc) != (W, H, C):
        raise Refused("IN_FMAP is %dx%dx%d (w h c), the model's input %dx%dx%d" % (mw, mh, mc, W, H, C))
    ifm_size = av[ai][3]
    rd = [x for x in io if x["src"][0] == ai]
    # the input read by several layers (a residual from the input): one descriptor per reader, the same format (V,
    # add-res-q3: two identical "fmt 288 n1 c4 h16 w32" reads); differing reads (tiles, offsets) are not described here
    if not rd or len(set(tuple(x["src"]) for x in rd)) != 1:
        raise Refused("%d io descriptors read the IFM%s (one format supported)" % (
            len(rd), ", in different formats: " + "; ".join(str(tuple(x["src"])) for x in rd) if rd else ""))
    s = rd[0]["src"]
    fmt, bpp, bn, bc, bh, bw, bws, bcs, bns = s[2:11]
    Hp, Wp = (H + 3) // 4 * 4, (W + 7) // 8 * 8
    # the descriptor of one DMA reading the whole IFM: ceil(C/2) planes as one block (the rule first seen), or, the
    # general form (docs/format/frame.md 8.1), n IDP blocks of pp planes: cpi = 2 ceil(C/8) channels per IDP,
    # n = ceil(C/cpi), pp = cpi/2 (the same for C <= 8); either way channel pair c/2 is plane c/2 of Hp x 2Wp bytes
    cpi = 2 * -(-C // 8)
    n_, pp = -(-C // cpi), cpi // 2
    wants = [(288, 1, (C + 1) // 2, Hp, 2 * Wp), (288, 1, n_, pp * Hp, 2 * Wp)]
    got = (fmt, bn, bc, bh, bw)
    if got not in wants or (bws, bcs, bns) != (0, 0, 0) or bc * bh * bw != ifm_size:
        raise Refused("the IFM descriptor is fmt %d n%d c%d h%d w%d ws %d cs %d ns %d for %d B; the known rules (format "
                      "0x120, 2 channels per plane in 4x8 cells) want fmt 288 n1 c%d h%d w%d or c%d h%d w%d: run the "
                      "impulse probe (journal/plans/npu-n3b-0218, probe mode) for this shape" % (
                          fmt, bn, bc, bh, bw, bws, bcs, bns, ifm_size, wants[0][2], wants[0][3], wants[0][4],
                          wants[1][2], wants[1][3], wants[1][4]))
    proven = (C, H, W) == (8, 16, 16)
    ifm = {"format": "0x120 cells", "size": ifm_size, "planes": bc * bh // Hp, "plane_h": Hp, "plane_w": 2 * Wp,
           "proven": proven,
           "how": "byte (c//2)*h*w + (y//4)*4*w + (x//8)*64 + (c%2)*32 + (y%4)*8 + x%8 (V for 8x16x16: the impulse "
                  "probe's npu-map.json, run 20261121-123201; other shapes: the same rule, sized by the NCP's descriptor, G)"}
    if len(rd) > 1:
        ifm["readers"] = len(rd)
    _, (t, pf, ow, oh, oc, ows, ocs, oai) = outs[0]
    C2, H2, W2 = low["npu_out_chw"]
    if (ow, oh, oc) != (W2, H2, C2):
        raise Refused("OT_FMAP is %dx%dx%d (w h c), the network's output %dx%dx%d" % (ow, oh, oc, W2, H2, C2))
    if av[oai][3] != C2 * H2 * W2 or (ows, ocs) != (0, 0):
        raise Refused("OT_FMAP holds %d B (wstride %d cstride %d) for %d values: not a dense CHW map" % (av[oai][3], ows, ocs, C2 * H2 * W2))
    last = low["layers"][-1]["kind"]
    ofm = {"format": "CHW", "size": C2 * H2 * W2, "proven": (C2, H2, W2) in ((8, 16, 16), (8, 8, 8)) and last in ("conv", "pool"),
           "how": "dense CHW (V for 8x16x16, run 20261121-121349, and 8x8x8, npu-run v4/v5 20261121: the output intrinsic "
                  "converts 0x120 to 0x110; other shapes: the OT_FMAP memory vector, G)"}
    if last not in ("conv", "pool"):
        ofm["how"] += "; written by an %s layer here: the same OT_FMAP descriptor, never run (G)" % last
    return {"ifm": ifm, "ofm": ofm, "quant_mode": low["quant_mode"]}


def ifm_addr(c, y, x, lay):
    w = lay["ifm"]["plane_w"]
    return (c // 2) * lay["ifm"]["plane_h"] * w + (y // 4) * 4 * w + (x // 8) * 64 + (c % 2) * 32 + (y % 4) * 8 + (x % 8)


def to_npu(low, lay, chw):
    C, H, W = low["input"]["chw"]
    pad = low["input"]["zp"]
    buf = bytearray([pad]) * lay["ifm"]["size"]
    for c in range(C):
        for y in range(H):
            for x in range(W):
                buf[ifm_addr(c, y, x, lay)] = chw[(c * H + y) * W + x]
    if low["quant_mode"] == 3:
        buf = bytearray(b ^ 0x80 for b in buf)
    return bytes(buf)


def from_npu(low, lay, ofm, layout="CHW"):
    C, H, W = low["npu_out_chw"]
    n = C * H * W
    if len(ofm) < n:
        raise Refused("the OFM is %d B; the network's output is %d" % (len(ofm), n))
    y = ofm[:n]
    if low["quant_mode"] == 3:
        y = bytes(b ^ 0x80 for b in y)
    if layout == "HWC":
        y = bytes(y[(h * W + w) * C + c] for c in range(C) for h in range(H) for w in range(W))
    elif layout == "CELL":
        # the IFM's 0x120 cell order applied to the output (a hypothesis for shapes whose OFM is not measured yet)
        cl = {"ifm": {"plane_h": (H + 3) // 4 * 4, "plane_w": 2 * ((W + 7) // 8 * 8)}}
        full = ofm if low["quant_mode"] != 3 else bytes(b ^ 0x80 for b in ofm)
        need = (C + 1) // 2 * cl["ifm"]["plane_h"] * cl["ifm"]["plane_w"]
        if len(full) < need:
            return bytes(n)  # not enough bytes for this reading: it can only mismatch
        y = bytes(full[ifm_addr(c, h, w, cl)] for c in range(C) for h in range(H) for w in range(W))
    return bytes(y)


def compare(low, lay, pairs, out=print):
    """pairs: [(model input bytes, OFM bytes)] -> (verdict, detail). Tries the rounding modes, the 'none' defect, the
    OFM read as HWC or in cells, and for graph networks the Eltwise's arithmetic (ADD_VARIANTS) and the standalone
    ReLU's (RELU_VARIANTS); the verdict comes from the best. Exact under any rounding and any add variant is
    "bit-exact" (they are the open questions of the reference); a defect, another OFM reading or a ReLU layer read
    otherwise than max(x, zp) is "exact-other"."""
    elt = any(l["kind"] == "eltwise" for l in low["layers"])
    rel = any(l["kind"] == "relu" for l in low["layers"])
    avs = ADD_VARIANTS if elt else ADD_VARIANTS[:1]
    uvs = RELU_VARIANTS if rel else RELU_VARIANTS[:1]
    hyps = [(r, d, ol, a, u) for ol in ("CHW", "HWC", "CELL") for d in (False, True) for a in avs for u in uvs for r in ROUNDINGS]
    if not any(l["kind"] == "conv" and l["act"] == 0 for l in low["layers"]):
        hyps = [h for h in hyps if not h[1]]
    stats = {}
    for k, (x, ofm) in enumerate(pairs):
        chw = model_in_to_chw(low, x)
        refs, memo = {}, {}
        for r, d, ol, a, u in hyps:
            key = (r, d, a, u)
            if key not in refs:
                refs[key] = ref_npu(low, chw, r, d, a, u, memo)
            got = from_npu(low, lay, ofm, ol)
            ref = refs[key]
            diff = [abs(p - q) for p, q in zip(ref, got) if p != q]
            s = stats.setdefault((r, d, ol, a, u), [0, 0, 0])
            s[0] += len(diff)
            s[1] = max(s[1], max(diff) if diff else 0)
            s[2] += 0 if diff else 1
    # exact first (the clean CHW reading before the defect or HWC), then the smallest max |diff|, then the fewest
    ranked = sorted(stats.items(), key=lambda kv: (kv[1][0] > 0, kv[1][1], kv[1][0], kv[0][1], kv[0][2] != "CHW",
                                                   kv[0][4] != "max", kv[0][3] != "sum"))
    n = len(from_npu(low, lay, pairs[0][1])) * len(pairs) if pairs else 0
    for (r, d, ol, a, u), s in ranked[:6]:
        out("compare: rounding %-9s 'none' defect %-5s OFM %s%s: %d of %d bytes differ, max |diff| %d, inputs exact %d/%d"
            % (r, d, ol, (" add %s" % a if elt else "") + (" ReLU %s" % u if rel else ""), s[0], n, s[1], s[2], len(pairs)))
    m = n // len(pairs) if pairs else 0
    cvals = [o[0] if m and len(o) >= m and o[:m] == bytes([o[0]]) * m else None for _, o in pairs]
    if pairs and all(v is not None for v in cvals):
        out("compare: every OFM is constant (byte(s) %s): the activation table of the compiled network writes one value "
            "(dossier/npu-compiler.md 9.11; the GReLU fix under study, oracle/ncp-grelu-patch.py)"
            % ", ".join("%#04x" % v for v in sorted(set(cvals))))
    (r, d, ol, a, u), s = ranked[0]
    how = "rounding %s%s%s%s%s" % (r, ", add %s" % a if elt else "", ", ReLU layer as %s" % u if u != "max" else "",
                                   ", with the known 'none' activation defect" if d else "",
                                   ", OFM read as %s" % ol if ol != "CHW" else "")
    clean = not d and ol == "CHW" and u == "max"
    if s[0] == 0 and clean:
        exact = [h for h, t in stats.items() if t[0] == 0 and not h[1] and h[2] == "CHW" and h[4] == "max"]
        return "bit-exact", "%s; exact under %s" % (how, "/".join(h[0] + (" add " + h[3] if elt else "") for h in exact))
    if s[0] == 0:
        return "exact-other", how
    if s[1] <= 1 and clean:
        return "within 1 LSB", "%s, %d of %d bytes off by 1" % (how, s[0], n)
    return "mismatch", "best: %s, %d of %d bytes differ, max |diff| %d" % (how, s[0], n, s[1])


# ---- the example ---------------------------------------------------------------------------------------------------
def example_graph():
    """a 3-layer int8 network on an 8x16x16 input (the IFM shape whose byte order was measured): conv 3x3 8->8 relu,
    max pool 2x2, conv 3x3 8->8 relu -> 8x8x8. Power-of-two scales, so NPUC's multipliers are exact (2^-11, 2^-11)."""
    r = random.Random(0x4E505531)
    C = 8

    def conv(name, cin, cout, wspan, bspan, out_scale):
        return {"op": "conv2d", "name": name, "filters": cout, "kernel": [3, 3], "strides": [1, 1], "padding": "same",
                "activation": "relu",
                "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 256, "zero_point": 0,
                            "data": [r.randint(-wspan, wspan) for _ in range(cout * 3 * 3 * cin)]},
                "bias": [r.randint(-bspan, bspan) for _ in range(cout)],
                "output": {"scale": out_scale, "zero_point": 0}}
    return {"format": "npu-net/1", "name": "tiny",
            "description": "ncpgen/npunet.py example: conv3x3+relu, maxpool2x2, conv3x3+relu on 16x16x8 int8",
            "input": {"name": "image", "shape": [1, 16, 16, C], "dtype": "int8", "scale": 1 / 128, "zero_point": 0},
            "layers": [conv("conv1", C, C, 24, 1500, 1 / 128),
                       {"op": "max_pool2d", "name": "pool1", "pool": [2, 2], "strides": [2, 2], "padding": "valid"},
                       conv("conv2", C, C, 16, 800, 1 / 512)]}


def graph_to_tflite(g):
    """our graph JSON -> a TFLite flatbuffer (the subset above: conv2d, max_pool2d, add, concat, relu; inputs by name
    as lower() reads them), written by FWriter"""
    tensors, buffers = [], [b""]  # buffer 0: empty, by convention
    ops, codes = [], []
    acts = {"none": 0, "relu": 1, "relu6": 3}

    def tensor(name, shape, dtype, scale=None, zp=None, data=None):
        bi = 0
        if data is not None:
            buffers.append(bytes(data))
            bi = len(buffers) - 1
        tensors.append({"name": name, "shape": shape, "dtype": dtype, "scale": scale, "zp": zp, "buffer": bi})
        return len(tensors) - 1

    def code(name):
        if OPCODE[name] not in codes:
            codes.append(OPCODE[name])
        return codes.index(OPCODE[name])
    io = g["input"]
    dt = io["dtype"]
    cur = tensor(io["name"], io["shape"], dt, [io["scale"]], [io["zero_point"]])
    first = cur
    tmap = {str(io.get("id") or io.get("name") or "input"): first}
    for L in g["layers"]:
        refs = L["inputs"] if "inputs" in L else [L["input"]] if "input" in L else None
        ins = [tmap[str(k)] for k in refs] if refs is not None else [cur]
        x = tensors[ins[0]]
        shape, sc, zp = x["shape"], x["scale"][0], x["zp"][0]
        if L["op"] == "conv2d":
            w = L["weights"]
            kh, kw = L["kernel"]
            cin = shape[3]
            wt = tensor(L["name"] + "/weights", [L["filters"], kh, kw, cin], "int8", [w["scale"]], [0],
                        struct.pack("<%db" % len(w["data"]), *w["data"]))
            bt = tensor(L["name"] + "/bias", [L["filters"]], "int32", [sc * w["scale"]], [0],
                        struct.pack("<%di" % len(L["bias"]), *L["bias"]))
            if L["padding"] == "same":
                ho = (shape[1] + L["strides"][0] - 1) // L["strides"][0]
                wo = (shape[2] + L["strides"][1] - 1) // L["strides"][1]
            else:
                ho = (shape[1] - kh) // L["strides"][0] + 1
                wo = (shape[2] - kw) // L["strides"][1] + 1
            o = tensor(L["name"], [1, ho, wo, L["filters"]], dt, [L["output"]["scale"]], [L["output"]["zero_point"]])
            ops.append(("CONV_2D", [ins[0], wt, bt], [o], 1, [(0, "b", 0 if L["padding"] == "same" else 1),
                                                               (1, "i", L["strides"][1]), (2, "i", L["strides"][0]),
                                                               (3, "b", acts[L["activation"]]),
                                                               (4, "i", 1), (5, "i", 1)]))
        elif L["op"] == "max_pool2d":
            kh, kw = L["pool"]
            sh, sw = L["strides"]
            o = tensor(L["name"], [1, (shape[1] - kh) // sh + 1, (shape[2] - kw) // sw + 1, shape[3]], dt, [sc], [zp])
            ops.append(("MAX_POOL_2D", [ins[0]], [o], 5, [(0, "b", 1), (1, "i", sw), (2, "i", sh), (3, "i", kw), (4, "i", kh), (5, "b", 0)]))
        elif L["op"] == "add":
            o = tensor(L["name"], shape, dt, [L["output"]["scale"]], [L["output"]["zero_point"]])
            ops.append(("ADD", ins, [o], 11, [(0, "b", acts[L.get("activation", "none")])]))
        elif L["op"] == "concat":
            cs = sum(tensors[t]["shape"][3] for t in ins)
            o = tensor(L["name"], shape[:3] + [cs], dt, [L["output"]["scale"]], [L["output"]["zero_point"]])
            ops.append(("CONCATENATION", ins, [o], 10, [(0, "i", L.get("axis", 3)), (1, "b", acts[L.get("activation", "none")])]))
        elif L["op"] in ("relu", "relu6"):
            q = L.get("output") or {"scale": sc, "zero_point": zp}
            o = tensor(L["name"], shape, dt, [q["scale"]], [q["zero_point"]])
            ops.append((L["op"].upper(), ins, [o], 0, []))
        else:
            raise Refused("graph_to_tflite: %s" % L["op"])
        tmap[str(L.get("id") or L["name"])] = o
        cur = o
    fb = FWriter()

    def wq(t):
        return lambda: fb.table([(2, "ref", lambda: fb.vec_scalars("f", t["scale"])),
                                 (3, "ref", lambda: fb.vec_scalars("q", t["zp"], 8))])

    def wtensor(t):
        f = [(0, "ref", lambda: fb.vec_scalars("i", t["shape"])), (1, "b", TTYPE[t["dtype"]]), (2, "I", t["buffer"]),
             (3, "ref", lambda: fb.string(t["name"]))]
        if t["scale"] is not None:
            f.append((4, "ref", wq(t)))
        return lambda: fb.table(f)

    def wop(o):
        name, ins, outs, otype, opts = o
        f = [(0, "I", codes.index(OPCODE[name])), (1, "ref", lambda: fb.vec_scalars("i", ins)),
             (2, "ref", lambda: fb.vec_scalars("i", outs))]
        if otype:
            f += [(3, "B", otype), (4, "ref", lambda: fb.table(opts))]
        return lambda: fb.table(f)
    for o in ops:
        code(o[0])

    def subgraph():
        return fb.table([(0, "ref", lambda: fb.vec_refs([wtensor(t) for t in tensors])),
                         (1, "ref", lambda: fb.vec_scalars("i", [first])), (2, "ref", lambda: fb.vec_scalars("i", [cur])),
                         (3, "ref", lambda: fb.vec_refs([wop(o) for o in ops])), (4, "ref", lambda: fb.string("main"))])

    def opcode(c):
        return lambda: fb.table([(0, "b", min(c, 127)), (2, "i", 1), (3, "i", c)])

    def buffer(b):
        return lambda: fb.table([(0, "ref", lambda: fb.vec_bytes(b))] if b else [])
    root = fb.table([(0, "I", 3), (1, "ref", lambda: fb.vec_refs([opcode(c) for c in codes])),
                     (2, "ref", lambda: fb.vec_refs([subgraph])),
                     (3, "ref", lambda: fb.string(g.get("description", g.get("name", "")))),
                     (4, "ref", lambda: fb.vec_refs([buffer(b) for b in buffers]))])
    return fb.finish(root)


VARIANTS = ("v2-conv", "v3-conv-conv", "v4-conv-pool", "v5-tiny")  # v1-pool (pool alone) dropped: npu-lite needs weights


def cmd_variants(d):
    """the example taken apart, one new piece per variant, for the phone (journal/plans/npu-run-0218/nr-variants.sh):
    v2 conv1+ReLU alone (16x16x8, the measured
    layout: the ReLU table), v3 conv1+ReLU, conv2+ReLU at 16x16 (two layers chained), v4 conv1+ReLU, max pool (pooling
    after a conv, 8x8x8), v5 the whole example. Same weights as the example; inputs: the example's and a second one."""
    os.makedirs(d, exist_ok=True)
    g = example_graph()
    c1, pool, c2 = g["layers"]
    for name, layers in zip(VARIANTS, ([c1], [c1, c2], [c1, pool], [c1, pool, c2])):
        v = dict(g, name=name, layers=layers, description="npunet.py variants: %s" % name)
        json.dump(v, open(os.path.join(d, name + ".json"), "w"), indent=1)
        low = lower(v)
        print("variants: %-13s %s -> %s" % (name, " + ".join(l["name"] for l in low["layers"]), "x".join(map(str, low["npu_out_chw"]))))
    open(os.path.join(d, "input.bin"), "wb").write(example_input())
    open(os.path.join(d, "input2.bin"), "wb").write(example_input(0x4E505533))
    print("variants: %d models, input.bin (the example's), input2.bin in %s" % (len(VARIANTS), d))
    return 0


def example_input(seed=0x4E505532, n=2048):
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


def cmd_example(d):
    os.makedirs(d, exist_ok=True)
    g = example_graph()
    json.dump(g, open(os.path.join(d, "tiny-net.json"), "w"), indent=1)
    tfl = graph_to_tflite(g)
    open(os.path.join(d, "tiny.tflite"), "wb").write(tfl)
    x = example_input()
    open(os.path.join(d, "input.bin"), "wb").write(x)
    low = lower(g)
    y = reference(low, x)
    open(os.path.join(d, "expected.bin"), "wb").write(y)
    low2 = lower(tflite_to_graph(read_tflite(tfl), "tiny.tflite"))
    same = reference(low2, x) == y
    for f in ("tiny-net.json", "tiny.tflite", "input.bin", "expected.bin"):
        b = open(os.path.join(d, f), "rb").read()
        print("example: %-14s %6d B sha256 %s" % (f, len(b), hashlib.sha256(b).hexdigest()))
    sat = sum(1 for v in y if v in (0x7f,)) / len(y)
    zero = sum(1 for v in y if v == 0) / len(y)
    print("example: output 1x8x8x8 int8: %.0f %% at 0 (relu), %.0f %% at 127 (saturated); the TFLite form gives the same "
          "output: %s" % (100 * zero, 100 * sat, same))
    return 0 if same else 1


def residual_graph():
    """a residual block (ResNet's basic block) on the measured 16x16x8 int8 input: conv 3x3 8->8 + ReLU, conv 3x3 8->8
    without activation, ADD of the block's input with a fused ReLU (-> eltwise + relu layers). Power-of-two scales,
    so every multiplier is exact (128 / 2^15, 128 / 2^14): input and conv1 2^-7, conv2 2^-8, the add's output 2^-6, i.e.
    out = conv2 / 4 + input / 2 (fractional lengths 8, 7 -> 6: two different shifts, nothing saturates anywhere, and the
    rounded bits name the Eltwise's rounding and its per-input order). Lower with --quant 3 --allow-none."""
    r = random.Random(0x4E505534)
    C = 8

    def conv(name, act, wspan, bspan, out_scale):
        return {"op": "conv2d", "name": name, "filters": C, "kernel": [3, 3], "strides": [1, 1], "padding": "same",
                "activation": act,
                "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 256, "zero_point": 0,
                            "data": [r.randint(-wspan, wspan) for _ in range(C * 3 * 3 * C)]},
                "bias": [r.randint(-bspan, bspan) for _ in range(C)],
                "output": {"scale": out_scale, "zero_point": 0}}
    return {"format": "npu-net/1", "name": "residual",
            "description": "ncpgen/npunet.py example-residual: conv3x3+relu, conv3x3, add(block input)+relu on 16x16x8 int8",
            "input": {"name": "image", "shape": [1, 16, 16, C], "dtype": "int8", "scale": 1 / 128, "zero_point": 0},
            "layers": [conv("conv1", "relu", 20, 1500, 1 / 128), conv("conv2", "none", 20, 800, 1 / 256),
                       {"op": "add", "name": "add1", "inputs": ["conv2", "image"], "activation": "relu",
                        "output": {"scale": 1 / 64, "zero_point": 0}}]}


def cmd_example_residual(d):
    os.makedirs(d, exist_ok=True)
    g = residual_graph()
    json.dump(g, open(os.path.join(d, "residual-net.json"), "w"), indent=1)
    tfl = graph_to_tflite(g)
    open(os.path.join(d, "residual.tflite"), "wb").write(tfl)
    low = lower(g, quant=3, allow_none=True)
    low2 = lower(tflite_to_graph(read_tflite(tfl), "residual.tflite"), quant=3, allow_none=True)
    same = True
    files = ["residual-net.json", "residual.tflite"]
    for k, seed in enumerate((0x4E505532, 0x4E505533)):
        sfx = "" if k == 0 else str(k + 1)
        x = example_input(seed)
        y = reference(low, x)
        same = same and reference(low2, x) == y
        open(os.path.join(d, "input%s.bin" % sfx), "wb").write(x)
        open(os.path.join(d, "expected%s.bin" % sfx), "wb").write(y)
        files += ["input%s.bin" % sfx, "expected%s.bin" % sfx]
        if k == 0:
            chw = model_in_to_chw(low, x)
            vals = {v: ref_npu(low, chw, add=v) for v in ADD_VARIANTS}
            ties = sum(1 for a_, b_ in zip(vals["sum"], vals["sum-floor"]) if a_ != b_)
    for f in files:
        b = open(os.path.join(d, f), "rb").read()
        print("example-residual: %-17s %6d B sha256 %s" % (f, len(b), hashlib.sha256(b).hexdigest()))
    print("example-residual: layers %s; output 1x16x16x8 int8: %.0f %% at 0 (relu), %.0f %% at 127; the add's rounding shows "
          "on %d of 2048 bytes of input.bin (half away vs floor); the TFLite form gives the same outputs: %s" % (
              ", ".join("%s (%s)" % (l["name"], l["kind"]) for l in low["layers"]), 100 * y.count(0) / len(y),
              100 * y.count(0x7f) / len(y), ties, same))
    print("example-residual: compile: NPU_ROOT=$NCPGEN_WORK/n3/root-current oracle/npuc-local.sh "
          "%s/residual-net.json <outdir> --quant 3 --allow-none" % d)
    return 0 if same else 1


# ---- self-test -----------------------------------------------------------------------------------------------------
def selftest():
    ok = []
    # 1. NPUC's multiplier on the values its log printed (npu-compiler.md 9.8: slope 16, shift 15)
    ok.append(("multiplier 1/128 x 1/256 / (1/16) = 16 / 2^15", npuc_multiplier(1 / 128, 1 / 256, 1 / 16) == (16, 15)))
    ok.append(("multiplier 0.7 -> 179 / 2^8", npuc_multiplier(0.7, 1.0, 1.0) == (179, 8)))
    ok.append(("multiplier 0.6 -> 154 / 2^8 (153.6 rounded half away, as NPUC's fcvtas; the oracle wrote 154)",
               npuc_multiplier(0.6, 1.0, 1.0) == (154, 8)))
    ok.append(("multiplier 2^-20 -> 2^-13 after 7 doublings: slope 0 refused", _refuses(lambda: npuc_multiplier(2 ** -20, 1, 1))))
    # 2. the IFM rule = the measured map (if the probe's map is on this host)
    mp = os.path.join(os.environ.get("NCPGEN_WORK") or os.path.join(HERE, "..", "work"), "n3b",
                      "set-d6cb7f2a-map60bdcf7f", "npu-map.json")
    lay = {"ifm": {"plane_h": 16, "plane_w": 32, "size": 2048}}
    rule = [ifm_addr(i // 256, (i // 16) % 16, i % 16, lay) for i in range(2048)]
    if os.path.exists(mp):
        m = json.load(open(mp))["chw_of_byte"]
        ok.append(("IFM rule = the impulse probe's npu-map.json (2048 bytes)", all(rule[i] == t for t, i in enumerate(m))))
    ok.append(("IFM rule is a bijection on 8x16x16", sorted(rule) == list(range(2048))))
    # 3. the example: graph -> tflite -> graph gives the same lowered network and output
    g = example_graph()
    tfl = graph_to_tflite(g)
    low, low2 = lower(g), lower(tflite_to_graph(read_tflite(tfl), "x.tflite"))
    strip = lambda l: [{k: v for k, v in x.items() if k not in ("src_name",)} for x in l["layers"]]
    ok.append(("example: TFLite round trip gives the same lowered layers", strip(low) == strip(low2)))
    x = example_input()
    y = reference(low, x)
    ok.append(("example: output 512 B", len(y) == 512))
    # 4. conv reference against a direct float formula on a small case (exact multipliers)
    small = {"format": "npu-net/1", "input": {"shape": [1, 4, 4, 2], "dtype": "uint8", "scale": 0.5, "zero_point": 128},
             "layers": [{"op": "conv2d", "name": "c", "filters": 1, "kernel": [3, 3], "strides": [1, 1], "padding": "same",
                         "activation": "relu", "weights": {"dtype": "int8", "layout": "OHWI", "scale": 0.25, "data": list(range(-9, 9))},
                         "bias": [3], "output": {"scale": 1.0, "zero_point": 128}}]}
    ls = lower(small)
    xin = bytes(range(100, 132))
    ys = reference(ls, xin)
    exp = []
    for yy in range(4):
        for xx in range(4):
            acc = 3
            for i in range(3):
                for j in range(3):
                    for c in range(2):
                        Y, X = yy + i - 1, xx + j - 1
                        if 0 <= Y < 4 and 0 <= X < 4:
                            acc += (xin[(Y * 4 + X) * 2 + c] - 128) * list(range(-9, 9))[(i * 3 + j) * 2 + c]
            v = acc * 0.5 * 0.25 / 1.0
            q = 128 + (math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5))
            exp.append(max(128, min(255, max(0, q))))
    ok.append(("conv reference = a direct formula (OHWI, SAME, relu, half away)", list(ys) == exp))
    # 5. compare: a planted NPU output is found; one byte off -> within 1 LSB; garbage -> mismatch
    lay8 = {"ifm": {"plane_h": 16, "plane_w": 32, "size": 2048}, "ofm": {"size": 512}}
    chw = model_in_to_chw(low, x)
    ofm = ref_npu(low, chw, "half_even")
    v1 = compare(low, lay8, [(x, ofm)], out=lambda s: None)
    b = bytearray(ofm)
    b[3] = (b[3] + 1) & 255
    v2 = compare(low, lay8, [(x, bytes(b))], out=lambda s: None)
    v3 = compare(low, lay8, [(x, bytes(512))], out=lambda s: None)
    ok.append(("compare: the reference itself is bit-exact", v1[0] == "bit-exact"))
    ok.append(("compare: one byte +1 -> within 1 LSB", v2[0] == "within 1 LSB"))
    msgs = []
    compare(low, lay8, [(x, bytes(512))], out=msgs.append)
    ok.append(("compare: zeros -> mismatch, the constant OFM named", v3[0] == "mismatch" and any("every OFM is constant (byte(s) 0x00)" in m for m in msgs)))
    # 5a. an OFM in the IFM's cell order (the 8x8x8 hypothesis) is found and named
    y8 = ref_npu(low, chw, "half_away")
    cl = {"ifm": {"plane_h": 8, "plane_w": 16}}
    cell = bytearray(512)
    for c in range(8):
        for h in range(8):
            for w in range(8):
                cell[ifm_addr(c, h, w, cl)] = y8[(c * 8 + h) * 8 + w]
    v5 = compare(low, lay8, [(x, bytes(cell))], out=lambda s: None)
    ok.append(("compare: an 8x8x8 OFM in cell order -> exact-other, 'OFM read as CELL'", v5[0] == "exact-other" and "CELL" in v5[1]))
    # 5b. the known 'none' activation defect is named, not passed
    gn = json.loads(json.dumps(g))
    gn["layers"][2]["activation"] = "none"
    ln_ = lower(gn, allow_none=True)
    ofd = ref_npu(ln_, chw, "half_away", none_defect=True)
    v4 = compare(ln_, lay8, [(x, ofd)], out=lambda s: None)
    ok.append(("compare: a planted 'none' defect -> exact-other, named", v4[0] == "exact-other" and "'none' activation defect" in v4[1]))
    # 6. to_npu / from_npu: mode 3 XOR, the IFM rule
    l3 = lower(g, quant=3)
    npu = to_npu(l3, lay8, chw)
    ok.append(("to_npu mode 3: byte (c1,y5,x9) at the rule's address, XOR 0x80", npu[ifm_addr(1, 5, 9, lay8)] == chw[(1 * 16 + 5) * 16 + 9] ^ 0x80))
    # 7. refusals
    bad = json.loads(json.dumps(g))
    bad["layers"][0]["activation"] = "none"
    ok.append(("'none' activation refused without --allow-none", _refuses(lambda: lower(bad))))
    bad2 = json.loads(json.dumps(g))
    bad2["layers"][0]["weights"]["scale"] = [1 / 256] * 7 + [1 / 128]
    ok.append(("per-channel scales refused without --fold-per-channel", _refuses(lambda: lower(bad2))))
    ok.append(("per-channel scales folded with it", not _refuses(lambda: lower(bad2, fold=True))))
    bad3 = json.loads(json.dumps(g))
    pl = [L for L in bad3["layers"] if L["op"] == "max_pool2d"][0]
    pl["pool"], pl["strides"] = [4, 4], [4, 4]
    ok.append(("pooling stride 4 refused (NPUC: 'StrideH (4) exceed limited value')", _refuses(lambda: lower(bad3))))
    gp = {"format": "npu-net/1", "name": "gap", "input": {"shape": [1, 8, 8, 4], "dtype": "int8", "scale": 1 / 128, "zero_point": 0},
          "layers": [{"op": "conv2d", "name": "c", "filters": 4, "kernel": [1, 1], "strides": [1, 1], "padding": "same",
                      "activation": "relu", "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 256, "zero_point": 0,
                                                        "data": [1] * 16}, "bias": [0] * 4, "output": {"scale": 1 / 128, "zero_point": 0}},
                     {"op": "average_pool2d", "name": "gap", "pool": [8, 8], "strides": [8, 8], "padding": "valid"}]}
    lg = lower(gp)
    gl = lg["layers"][-1]
    ok.append(("global average pool 8x8 -> an 8x8 VALID conv, identity weights, multiplier 1/64 (128/2^13), ReLU",
               gl["kind"] == "conv" and gl["k"] == [8, 8] and gl["padding"] == "VALID" and gl["act"] == 1 and
               (gl["slope"], gl["shift"]) == (128, 13) and gl["out_chw"] == [4, 1, 1] and
               sum(1 for v in gl["weights_oihw_u8"] if v != 128) == 4 * 64))
    x = bytes(random.Random(5).randrange(256) for _ in range(256))
    y = reference(lg, x)
    xc = model_in_to_chw(lg, x)
    c1 = ref_npu({**lg, "layers": lg["layers"][:1], "npu_out_chw": lg["layers"][0]["out_chw"]}, xc)
    want = bytes((lambda s_: (s_ + 32) // 64 if s_ >= 0 else -((-s_ + 32) // 64))(sum(c1[c * 64 + k] - 128 for k in range(64))) + 0 for c in range(4))
    ok.append(("global average pool reference = the mean of the conv's outputs, rounded half away",
               bytes((b + 128) & 0xFF for b in y) == bytes((w + 128) & 0xFF for w in want)))
    bad4 = json.loads(json.dumps(g))
    bad4["layers"] = [bad4["layers"][1]]
    ok.append(("a network without a conv refused (npu-lite needs a WEIGHT vector)", _refuses(lambda: lower(bad4))))
    bad3 = json.loads(json.dumps(g))
    bad3["input"]["zero_point"] = -5
    ok.append(("quant mode 3 with a zero point != 0 (int8) refused", _refuses(lambda: lower(bad3, quant=3))))
    ok += selftest_graph()
    for name, v in ok:
        print("selftest: %-70s %s" % (name, "ok" if v else "FAIL"))
    good = all(v for _, v in ok)
    print("npunet selftest: " + ("PASS" if good else "FAIL"))
    return 0 if good else 1


def _refuses(f):
    try:
        f()
    except Refused:
        return True
    return False


def _sconv(name, cin, cout, r, act="relu", out_scale=1 / 128, inp=None, k=3, wspan=8):
    L = {"op": "conv2d", "name": name, "filters": cout, "kernel": [k, k], "strides": [1, 1], "padding": "same",
         "activation": act, "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 256, "zero_point": 0,
                                        "data": [r.randint(-wspan, wspan) for _ in range(cout * k * k * cin)]},
         "bias": [r.randint(-300, 300) for _ in range(cout)], "output": {"scale": out_scale, "zero_point": 0}}
    if inp is not None:
        L["input"] = inp
    return L


def _sgraph(layers, shape=(1, 8, 8, 8)):
    return {"format": "npu-net/1", "name": "t", "layers": layers,
            "input": {"name": "x", "shape": list(shape), "dtype": "int8", "scale": 1 / 128, "zero_point": 0}}


def _refusal(f):
    try:
        f()
    except Refused as e:
        return str(e)
    return ""


def _synthetic_ncp(C, H, W, srcs):
    """a minimal NCP v22: IN_FMAP (av 0) read by one io descriptor per entry of srcs, OT_FMAP (av 1), CHW"""
    size = (C + 1) // 2 * ((H + 3) // 4 * 4) * 2 * ((W + 7) // 8 * 8)
    av = [(0, 0, 0, size), (1, 0, 0, C * H * W)]
    mv = [(0, 1, W, H, C, 0, 0, 0), (1, 1, W, H, C, 0, 0, 1)]
    avo, mvo = 0x80, 0x80 + 16 * len(av)
    iov = mvo + 32 * len(mv)
    h = [0] * 32
    h[0], h[1], h[31] = 0x0C0FFEE0, 22, 0xC0DEC0DE
    h[10:14] = [avo, len(av), mvo, len(mv)]
    h[19], h[20] = iov, len(srcs)
    b = struct.pack("<32I", *h) + b"".join(struct.pack("<4I", *a) for a in av) + b"".join(struct.pack("<8I", *m) for m in mv)
    for k, s in enumerate(srcs):
        b += struct.pack("<31I", *([0, 0, 0, 0, 1 << k, 2, 0, 0, 0] + list(s) + [0] * 11))
    return b


def selftest_graph():
    """the graph ops (ADD, CONCATENATION, RELU): lowering, refusals, the reference, compare, the NCP's IFM readers"""
    ok = []
    r = random.Random(5)
    # chains are lowered exactly as before ("npuc-net 1", same bytes: the sha256 of the example's net.npuc before graphs)
    t = npuc_text(lower(example_graph()))
    ok.append(("chain: the example's net.npuc unchanged (npuc-net 1, sha256 ac9feba2)", t.startswith("npuc-net 1\n") and
               hashlib.sha256(t.encode()).hexdigest().startswith("ac9feba20005632e")))
    # the residual example: conv, conv, eltwise (conv2 + the network input), relu; npuc-net 2
    rg = residual_graph()
    low = lower(rg, quant=3, allow_none=True)
    t = npuc_text(low)
    ok.append(("residual: conv, conv, eltwise, relu in an npuc-net 2 file", [l["kind"] for l in low["layers"]] ==
               ["conv", "conv", "eltwise", "relu"] and t.startswith("npuc-net 2\n") and low.get("npuc_net") == 2))
    ok.append(("residual: eltwise line reads conv2 and the input, fl 8 7 -> 6", "eltwise l3_add1 l2_conv2 data0 Add "
               "0x3b800000 128 0x3c000000 128 0x3c800000 128\n" in t and "relu l4_add1_relu l3_add1 ReLU\n" in t))
    tfl = graph_to_tflite(rg)
    low2 = lower(tflite_to_graph(read_tflite(tfl), "r.tflite"), quant=3, allow_none=True)
    strip = lambda l: [{k: v for k, v in x.items() if k != "src_name"} for x in l["layers"]]
    ok.append(("residual: the TFLite form (ADD with fused RELU) lowers to the same layers", strip(low) == strip(low2)))
    # the Eltwise reference: integer rounding = rnd() on fractions, every variant; saturation
    good = True
    for fl in ((8, 7, 6), (7, 7, 7), (6, 7, 8), (7, 5, 9), (9, 9, 6)):
        l = {"fl": list(fl), "out_zp": 128}
        a = bytes(range(0, 256, 5)) * 52
        b = bytes(v for v in range(0, 256, 5) for _ in range(52))
        for rr in ROUNDINGS + ("floor",):
            want = bytes(min(127, max(-128, rnd(Fraction(p - 128) * Fraction(2) ** (fl[2] - fl[0]) + Fraction(q - 128) *
                                               Fraction(2) ** (fl[2] - fl[1]), rr))) + 128 for p, q in zip(a, b))
            good = good and ref_eltwise(l, a, b, rr, "sum" if rr != "floor" else "sum-floor") == want
    ok.append(("eltwise reference = sat_s8(round(a 2^(flo-fl1) + b 2^(flo-fl2))) on 2704 pairs x 5 x 4", good))
    l = {"fl": [8, 7, 6], "out_zp": 128}
    ok.append(("eltwise variants: 5/4 + 3/2 -> 3 2 3 2 (sum, sum-floor, each, each-floor); -2/4 - 1/2 -> -1 (sum), -2 (each)",
               [ref_eltwise(l, bytes([133]), bytes([131]), "half_away", v)[0] - 128 for v in ADD_VARIANTS] == [3, 2, 3, 2]
               and ref_eltwise(l, bytes([126]), bytes([127]), "half_away", "sum")[0] - 128 == -1
               and ref_eltwise(l, bytes([126]), bytes([127]), "half_away", "each")[0] - 128 == -2))
    l7 = {"fl": [7, 7, 7], "out_zp": 128}
    ok.append(("eltwise saturates at 127 / -128", ref_eltwise(l7, bytes([255, 0]), bytes([255, 0])) == bytes([255, 0])))
    # concat: lowered, the reference = the inputs' channels in order, TFLite round trip
    cg = _sgraph([_sconv("a", 8, 8, r, inp="x"), _sconv("b", 8, 8, r, inp="x"),
                  {"op": "concat", "name": "cat", "inputs": ["a", "b"], "axis": 3, "output": {"scale": 1 / 128, "zero_point": 0}},
                  _sconv("c", 16, 8, r)])
    cl = lower(cg)
    ok.append(("concat: conv, conv, concat (16 channels), conv; line 'concat l3_cat 1 2 l1_a l2_b'",
               [l["kind"] for l in cl["layers"]] == ["conv", "conv", "concat", "conv"] and cl["layers"][2]["out_chw"] == [16, 8, 8]
               and "concat l3_cat 1 2 l1_a l2_b 0x3c000000 128\n" in npuc_text(cl)))
    xs = bytes(r.randrange(256) for _ in range(512))
    c3 = dict(cl, layers=cl["layers"][:3])
    ok.append(("concat reference = conv a's bytes then conv b's (CHW)", ref_npu(c3, xs) ==
               ref_conv(cl["layers"][0], xs, "half_away", False, 1) + ref_conv(cl["layers"][1], xs, "half_away", False, 1)))
    cl2 = lower(tflite_to_graph(read_tflite(graph_to_tflite(cg)), "c.tflite"))
    ok.append(("concat: the TFLite form (CONCATENATION axis 3) lowers to the same layers", strip(cl) == strip(cl2)))
    # relu: standalone after a pool; folded into the conv before it
    pg = _sgraph([_sconv("a", 8, 8, r), {"op": "max_pool2d", "name": "p", "pool": [2, 2], "strides": [2, 2], "padding": "valid"},
                  {"op": "relu", "name": "r"}, _sconv("c", 8, 8, r)])
    pl = lower(pg)
    ok.append(("relu after a pool: a relu layer ('relu l3_r l2_p ReLU')", [l["kind"] for l in pl["layers"]] ==
               ["conv", "pool", "relu", "conv"] and "relu l3_r l2_p ReLU\n" in npuc_text(pl)))
    ok.append(("relu reference = max(x, zp)", ref_relu({"out_zp": 128}, bytes([0, 127, 128, 129, 255])) == bytes([128, 128, 128, 129, 255])))
    c_none = _sconv("a", 8, 8, r, act="none")
    fl_ = lower(_sgraph([c_none, {"op": "relu", "name": "r"}]))
    ok.append(("RELU after a conv without activation: folded into it (the same net.npuc as a fused RELU)",
               len(fl_["layers"]) == 1 and npuc_text(fl_) == npuc_text(lower(_sgraph([dict(c_none, activation="relu")])))
               and any("folded" in w for w in fl_["warnings"])))
    r6 = _sgraph([_sconv("a", 8, 8, r), {"op": "max_pool2d", "name": "p", "pool": [2, 2], "strides": [2, 2]},
                  {"op": "relu6", "name": "r"}, _sconv("c", 8, 8, r)])
    ok.append(("RELU6 whose clamp is a no-op (scale 1/128): a ReLU layer", [l["kind"] for l in lower(r6)["layers"]][2] == "relu"))
    r6b = json.loads(json.dumps(r6))
    r6b["layers"][0]["output"]["scale"] = 0.1
    r6b["layers"][3]["output"]["scale"] = 0.1
    ok.append(("RELU6 clamping inside the range refused (NPUC's fixed clamp at 0x60)", "0x60" in _refusal(lambda: lower(r6b))))
    rq = json.loads(json.dumps(pg))
    rq["layers"][2]["output"] = {"scale": 1 / 64, "zero_point": 0}
    ok.append(("RELU with another output quantisation refused", "keeps its input's quantisation" in _refusal(lambda: lower(rq))))
    # refusals
    ok.append(("ADD in quant mode 1 refused (no zero-point term)", "--quant 3" in _refusal(lambda: lower(rg, allow_none=True))))
    b1 = json.loads(json.dumps(rg))
    b1["layers"][2]["output"]["scale"] = 0.01
    ok.append(("ADD with a scale not 2^-n refused", "is not 2^-n" in _refusal(lambda: lower(b1, quant=3, allow_none=True))))
    b2 = json.loads(json.dumps(rg))
    b2["layers"][2]["inputs"] = ["conv2", "conv2"]
    ok.append(("ADD x + x refused", "x + x" in _refusal(lambda: lower(b2, quant=3, allow_none=True))))
    b3 = _sgraph([_sconv("a", 8, 8, r, inp="x"), _sconv("b", 8, 4, r, inp="x"),
                  {"op": "add", "name": "s", "inputs": ["a", "b"], "output": {"scale": 1 / 128, "zero_point": 0}}])
    ok.append(("ADD of different shapes refused (no broadcasting)", "broadcasting" in _refusal(lambda: lower(b3, quant=3))))
    b4 = json.loads(json.dumps(cg))
    b4["layers"][2]["axis"] = 1
    ok.append(("CONCATENATION on another axis than channels refused", "channels only" in _refusal(lambda: lower(b4))))
    b5 = json.loads(json.dumps(cg))
    b5["layers"][2]["inputs"] = ["x", "a"]
    b5["layers"][1]["input"] = "a"
    ok.append(("CONCATENATION of the network input refused", "network input" in _refusal(lambda: lower(b5))))
    b6 = json.loads(json.dumps(cg))
    b6["layers"][1]["output"]["scale"] = 1 / 64
    ok.append(("CONCATENATION of an input in another quantisation refused", "no rescale" in _refusal(lambda: lower(b6))))
    b7 = _sgraph([_sconv("a", 8, 8, r, inp="x"), _sconv("b", 8, 8, r, inp="x")])
    ok.append(("a tensor read by no layer (a second output) refused", "read by no later layer" in _refusal(lambda: lower(b7))))
    b8 = _sgraph([_sconv("a", 8, 8, r, inp="nope")])
    ok.append(("a layer reading an unknown tensor refused", "neither the network input" in _refusal(lambda: lower(b8))))
    ok.append(("an unsupported op refused with its name (npu-net/1)",
               "depthwise_conv2d" in _refusal(lambda: lower(_sgraph([_sconv("a", 8, 8, r), {"op": "depthwise_conv2d"}])))))
    tq = {"name": "x", "shape": [1, 4, 4, 2], "dtype": "int8", "scale": [0.5], "zero_point": [0], "data": b""}
    tf = {"tensors": [tq, dict(tq, name="m"), dict(tq, name="k")], "inputs": [0], "outputs": [1],
          "ops": [{"op": "MUL", "inputs": [0, 0], "outputs": [1], "opt": {}}]}
    ok.append(("an unsupported TFLite op refused with its name (MUL)", "MUL" in _refusal(lambda: tflite_to_graph(tf, "m"))))
    tf["ops"] = [{"op": "ADD", "inputs": [0, 2], "outputs": [1], "opt": {}}]
    ok.append(("a TFLite ADD with a constant operand refused", "constant operand" in _refusal(lambda: tflite_to_graph(tf, "m"))))
    # compare on the residual: the add variant and the ReLU reading are found and named
    lay = {"ifm": {"plane_h": 16, "plane_w": 32, "size": 2048}, "ofm": {"size": 2048}}
    x = example_input()
    chw = model_in_to_chw(low, x)
    planted = lambda **kw: bytes(v ^ 0x80 for v in ref_npu(low, chw, **kw))  # quant 3: the NPU's bytes are s8
    v1 = compare(low, lay, [(x, planted())], out=lambda s: None)
    v2 = compare(low, lay, [(x, planted(add="each-floor"))], out=lambda s: None)
    v3 = compare(low, lay, [(x, planted(relu="identity"))], out=lambda s: None)
    ok.append(("compare residual: the reference -> bit-exact, add sum", v1[0] == "bit-exact" and "add sum" in v1[1]))
    ok.append(("compare residual: per-term floor planted -> bit-exact, 'add each-floor' named", v2[0] == "bit-exact" and "add each-floor" in v2[1]))
    ok.append(("compare residual: ReLU layer not applied -> exact-other, named", v3[0] == "exact-other" and "ReLU layer as identity" in v3[1]))
    # the NCP side: the input read by two layers (two identical IFM descriptors, as NPUC's add-res-q3), or refused
    s0 = (0, 0, 288, 128, 1, 4, 16, 32, 0, 0, 0)
    l2 = ncp_layout(low, _synthetic_ncp(8, 16, 16, [s0, s0]))
    ok.append(("layout: the IFM read twice in one format accepted (residual)", l2["ifm"].get("readers") == 2 and
               (l2["ifm"]["plane_h"], l2["ifm"]["plane_w"]) == (16, 32) and not l2["ofm"]["proven"]))
    ok.append(("layout: the IFM read in two formats refused", _refuses(lambda: ncp_layout(low, _synthetic_ncp(
        8, 16, 16, [s0, (0, 512) + s0[2:]])))))
    # suspected NPUC defects named (frame.md 9.1, 9.2)
    u3 = lower(_sgraph([_sconv("a", 8, 3, r)]))["unproven"]
    ok.append(("an NPU output with 3 channels: named unproven (frame.md 9.1)", any("frame.md 9.1" in u for u in u3)))
    wg = [_sconv("a", 8, 8, r), {"op": "max_pool2d", "name": "p", "pool": [2, 2], "strides": [2, 2], "padding": "valid"}]
    u4 = lower(_sgraph(wg + [_sconv("c", 8, 8, r)], shape=(1, 16, 17, 8)))["unproven"]
    u5 = lower(_sgraph(wg, shape=(1, 16, 17, 8)))["unproven"]
    ok.append(("a VALID pool widened by NPUC, read by a conv: named unproven (cmdq.md 7); last layer: not",
               any("cmdq.md 7" in u for u in u4) and not any("cmdq.md 7" in u for u in u5)))
    return ok


# ---- commands -------------------------------------------------------------------------------------------------------
def cmd_lower(a):
    """lowered.json and net.npuc for a model, and a line per layer"""
    low = load_model(a.model, a.quant, a.fold_per_channel, a.allow_none)
    os.makedirs(a.dir, exist_ok=True)
    json.dump(low, open(os.path.join(a.dir, "lowered.json"), "w"), indent=1)
    open(os.path.join(a.dir, "net.npuc"), "w").write(npuc_text(low))
    for l in low["layers"]:
        if l["kind"] == "conv":
            print("lower: %-22s conv %dx%d s%d %s %d->%d act %s, multiplier %d/2^%d, out %s" % (
                l["name"], l["k"][0], l["k"][1], l["stride"][0], l["padding"], l["in_chw"][0], l["cout"],
                "relu" if l["act"] else "none", l["slope"], l["shift"], "x".join(map(str, l["out_chw"]))))
        elif l["kind"] == "pool":
            print("lower: %-22s %s %dx%d s%d %s, out %s" % (l["name"], l["type"], l["k"][0], l["k"][1],
                                                             l["stride"][0], l["padding"], "x".join(map(str, l["out_chw"]))))
        elif l["kind"] == "eltwise":
            print("lower: %-22s eltwise %s %s + %s, fractional lengths %d %d -> %d, out %s" % (
                l["name"], l["op"], l["inputs"][0], l["inputs"][1], l["fl"][0], l["fl"][1], l["fl"][2],
                "x".join(map(str, l["out_chw"]))))
        elif l["kind"] == "concat":
            print("lower: %-22s concat (channels) %s, out %s" % (l["name"], " + ".join(l["inputs"]),
                                                                 "x".join(map(str, l["out_chw"]))))
        else:
            print("lower: %-22s %s of %s, out %s" % (l["name"], l["type"], l["input"], "x".join(map(str, l["out_chw"]))))
    for t in low["cpu_tail"]:
        print("lower: CPU tail: %s" % t["op"])
    for w in low["warnings"]:
        print("lower: warning: " + w)
    for u in low["unproven"]:
        print("lower: unproven: " + u)
    print("lower: PASS: %s (%s %s): input NHWC %s %s, %d NPU layers, output %s %s, quant mode %d -> %s/lowered.json, net.npuc%s"
          % (low["name"], low["source"]["kind"], low["source"]["sha256"][:8], low["input"]["nhwc"], low["io_dtype"],
             len(low["layers"]), low["output"]["shape"], low["io_dtype"], a.quant, a.dir,
             " (npuc-net 2: a graph; compile with a harness that reads it, e.g. $NCPGEN_WORK/n3/root-current)"
             if is_graph(low) else ""))


def cmd_ref(a):
    low = json.load(open(a.lowered))
    y = reference(low, open(a.input, "rb").read())
    open(a.output, "wb").write(y)
    print("ref: %s -> %s: %d B sha256 %s" % (a.input, a.output, len(y), hashlib.sha256(y).hexdigest()))


def cmd_layout(a):
    low = json.load(open(a.lowered))
    lay = ncp_layout(low, open(a.ncp, "rb").read())
    json.dump(lay, open(a.layout, "w"), indent=1)
    print("layout: IFM %s, %d B (%d planes of %dx%d)%s; OFM %s %d B%s" % (
        lay["ifm"]["format"], lay["ifm"]["size"], lay["ifm"]["planes"], lay["ifm"]["plane_h"], lay["ifm"]["plane_w"],
        "" if lay["ifm"]["proven"] else " (rule extrapolated to this shape)", lay["ofm"]["format"], lay["ofm"]["size"],
        "" if lay["ofm"]["proven"] else " (from the descriptor, this shape not measured yet)") +
          ("" if "frame.py" in lay.get("checked_by", "") else "; %s" % lay.get("checked_by", "")))


def cmd_to_npu(a):
    low, lay = json.load(open(a.lowered)), json.load(open(a.layout))
    b = to_npu(low, lay, model_in_to_chw(low, open(a.input, "rb").read()))
    open(a.out, "wb").write(b)
    print("to-npu: %s -> %s: %d B sha256 %s" % (a.input, a.out, len(b), hashlib.sha256(b).hexdigest()))


def cmd_from_npu(a):
    low, lay = json.load(open(a.lowered)), json.load(open(a.layout))
    y = out_to_model(low, cpu_tail(low, from_npu(low, lay, open(a.ofm, "rb").read())))
    open(a.out, "wb").write(y)
    print("from-npu: %s -> %s: %d B sha256 %s" % (a.ofm, a.out, len(y), hashlib.sha256(y).hexdigest()))


def cmd_compare(a):
    low, lay = json.load(open(a.lowered)), json.load(open(a.layout))
    pairs = [(open(p.split(":", 1)[0], "rb").read(), open(p.split(":", 1)[1], "rb").read()) for p in a.pairs]
    v, why = compare(low, lay, pairs)
    print("compare: %s (%s)" % (v, why))
    return 0 if v in ("bit-exact", "within 1 LSB") else 2


def main(argv=None):
    """the commands of the module docstring. Exits (SystemExit) with the command's status: the porting kit's
    npu/npunet.py loads this file and calls main() as its whole program."""
    ap = argparse.ArgumentParser(prog="npunet.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lower", help="lowered.json and net.npuc for a model")
    p.add_argument("model")
    p.add_argument("dir")
    p.add_argument("--quant", type=int, default=1)
    p.add_argument("--fold-per-channel", action="store_true")
    p.add_argument("--allow-none", action="store_true")
    p.set_defaults(fn=cmd_lower)
    for name, fn, args in (("ref", cmd_ref, ("lowered", "input", "output")),
                           ("layout", cmd_layout, ("lowered", "ncp", "layout")),
                           ("to-npu", cmd_to_npu, ("lowered", "layout", "input", "out")),
                           ("from-npu", cmd_from_npu, ("lowered", "layout", "ofm", "out"))):
        p = sub.add_parser(name)
        for x in args:
            p.add_argument(x)
        p.set_defaults(fn=fn)
    p = sub.add_parser("compare", help="the NPU's output against the CPU reference")
    p.add_argument("lowered")
    p.add_argument("layout")
    p.add_argument("pairs", nargs="+", metavar="input.bin:ofm.bin")
    p.set_defaults(fn=cmd_compare)
    for name, fn in (("example", cmd_example), ("example-residual", cmd_example_residual), ("variants", cmd_variants)):
        p = sub.add_parser(name)
        p.add_argument("dir")
        p.set_defaults(fn=lambda a, fn=fn: fn(a.dir))
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    a = ap.parse_args(argv)
    try:
        rc = a.fn(a)
    except Refused as e:
        die(str(e))
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
