#!/usr/bin/env python3
"""The stride rewrites compute the original model: npunet's CPU reference of the rewritten network (npunet.lower,
which rewrites the strides NPUC refuses, then npunet.reference = `npunet.py ref`) against a direct reference of the
original npu-net/1 graph written here, independently of npunet's lowering: NHWC, the model's own strides, padding and
weights, TFLite's output sizes and pads, and the reference arithmetic npunet documents (acc = bias + sum (x - zp) w,
NPUC's multiplier slope / 2^shift, rounding half away, the clamp, ReLU as max(q, zp); an average pool rounds the mean
half up; SOFTMAX as npunet's CPU tail). Random inputs, seeded. No vendor file.

  tests/rewrite_ref.py [--inputs N] [--jobs N] [case...]     exit 0 when every output is equal byte for byte
  tests/rewrite_ref.py --write <dir> [case...]     the small networks as <dir>/<case>.json (npu-net/1) and
        <dir>/specs.txt (case|lowering options): the sweep $NCPGEN_WORK/sweep/restride (oracle/npuc-local.sh)
Cases: the published zoo models npunet rewrites (kws-cnn, magic-wand, alexnet-96) and small networks for each
rewrite form (s-*).
"""
import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from operator import mul

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "models"))
from ncpgen import npunet  # noqa: E402
import zoo  # noqa: E402


# ---- the direct reference of an npu-net/1 graph ---------------------------------------------------------------------
def out_pad(n, k, s, pad):
    """TFLite's output size and leading pad (SAME's total pad clamped at 0)"""
    if pad == "valid":
        return (n - k) // s + 1, 0
    o = (n + s - 1) // s
    return o, max(0, (o - 1) * s + k - n) // 2


def requant(acc, slope, shift, zp, lo, hi, relu):
    """zp + round half away (acc slope / 2^shift), clamped, then ReLU"""
    v, d = acc * slope, 1 << shift
    q = (abs(v) + d // 2) // d
    q = zp + (q if v >= 0 else -q)
    q = min(hi, max(lo, q))
    return max(q, zp) if relu else q


class T:
    """a tensor: NHWC values in the model's dtype (a flat list), its shape, scale and zero point"""
    def __init__(self, v, h, w, c, scale, zp):
        self.v, self.h, self.w, self.c, self.scale, self.zp = v, h, w, c, scale, zp


def conv(x, L, lo, hi, depthwise=False):
    kh, kw = L["kernel"]
    sh, sw = L.get("strides", [1, 1])
    pad = L.get("padding", "same").lower()
    wt = L["weights"]
    ws = wt["scale"] if isinstance(wt["scale"], list) else [wt["scale"]]
    if len(set(ws)) != 1:
        raise ValueError("per-channel scales: not in this reference")
    C = x.c
    co = C if depthwise else L["filters"]
    ho, pt = out_pad(x.h, kh, sh, pad)
    wo, pl = out_pad(x.w, kw, sw, pad)
    oq = L["output"]
    slope, shift = npunet.npuc_multiplier(x.scale, ws[0], oq["scale"])
    # the input centred and padded with zeros (the border is the zero point)
    Hp, Wp = max(x.h + pt, (ho - 1) * sh + kh), max(x.w + pl, (wo - 1) * sw + kw)
    rows = [[0] * (Wp * C) for _ in range(Hp)]
    for y in range(x.h):
        rows[y + pt][pl * C:(pl + x.w) * C] = [v - x.zp for v in x.v[y * x.w * C:(y + 1) * x.w * C]]
    d = wt["data"]
    if depthwise:   # 1HWO: tap (i, j) of channel o at (i kw + j) C + o
        wk = [[d[t * C + o] for t in range(kh * kw)] for o in range(C)]
    else:           # OHWI: (o, i, j, c)
        K = kh * kw * C
        wk = [d[o * K:(o + 1) * K] for o in range(co)]
    relu = L.get("activation", "none") == "relu"
    out = []
    for y in range(ho):
        for xx in range(wo):
            patch = []
            for i in range(kh):
                patch += rows[y * sh + i][xx * sw * C:(xx * sw + kw) * C]
            for o in range(co):
                acc = L["bias"][o] + (sum(map(mul, patch[o::C], wk[o])) if depthwise else sum(map(mul, patch, wk[o])))
                out.append(requant(acc, slope, shift, oq["zero_point"], lo, hi, relu))
    return T(out, ho, wo, co, oq["scale"], oq["zero_point"])


def pool(x, L):
    kh, kw = L["pool"]
    sh, sw = L.get("strides", [kh, kw])
    pad = L.get("padding", "valid").lower()
    ho, pt = out_pad(x.h, kh, sh, pad)
    wo, pl = out_pad(x.w, kw, sw, pad)
    C, out = x.c, []
    for y in range(ho):
        for xx in range(wo):
            win = [(yy, xw) for yy in range(y * sh - pt, y * sh - pt + kh) if 0 <= yy < x.h
                   for xw in range(xx * sw - pl, xx * sw - pl + kw) if 0 <= xw < x.w]
            for c in range(C):
                vals = [x.v[(yy * x.w + xw) * C + c] for yy, xw in win]
                if L["op"] == "max_pool2d":
                    out.append(max(vals))
                else:       # the mean rounded half up (npunet's reference), floor((sum + n / 2) / n)
                    out.append((sum(vals) + len(vals) // 2) // len(vals))
    return T(out, ho, wo, C, x.scale, x.zp)


def fully_connected(x, L, lo, hi):
    K = x.h * x.w * x.c
    wt = L["weights"]
    oq = L["output"]
    slope, shift = npunet.npuc_multiplier(x.scale, wt["scale"], oq["scale"])
    xc = [v - x.zp for v in x.v]
    relu = L.get("activation", "none") == "relu"
    out = [requant(L["bias"][o] + sum(map(mul, xc, wt["data"][o * K:(o + 1) * K])), slope, shift, oq["zero_point"], lo, hi,
                   relu) for o in range(L["units"])]
    return T(out, 1, 1, L["units"], oq["scale"], oq["zero_point"])


def softmax(x, L, lo, hi):
    m = max(x.v)
    e = [math.exp(L.get("beta", 1.0) * npunet.f32(x.scale) * (q - m)) for q in x.v]
    s = sum(e)
    so, zo = npunet.f32(L["output"]["scale"]), L["output"]["zero_point"]
    return T([min(hi, max(lo, zo + int(math.floor(p / s / so + 0.5)))) for p in e], 1, 1, x.c, L["output"]["scale"], zo)


def direct(g, data):
    """the original graph (a chain: each layer reads the previous one) on the model's input bytes -> its output bytes"""
    io = g["input"]
    lo, hi = (-128, 127) if io["dtype"] == "int8" else (0, 255)
    shp = io["shape"] if len(io["shape"]) == 4 else [1, 1, 1, io["shape"][1]]
    vals = list(data) if io["dtype"] == "uint8" else [b - 256 if b > 127 else b for b in data]
    x = T(vals, shp[1], shp[2], shp[3], io["scale"], io["zero_point"])
    for L in g["layers"]:
        if "input" in L or "inputs" in L:
            raise ValueError("a chain only")
        op = L["op"]
        if op in ("conv2d", "depthwise_conv2d"):
            x = conv(x, L, lo, hi, depthwise=op == "depthwise_conv2d")
        elif op in ("max_pool2d", "average_pool2d"):
            x = pool(x, L)
        elif op == "fully_connected":
            x = fully_connected(x, L, lo, hi)
        elif op == "reshape":
            x = T(x.v, 1, 1, len(x.v), x.scale, x.zp)
        elif op == "softmax":
            x = softmax(x, L, lo, hi)
        else:
            raise ValueError("op %s: not in this reference" % op)
    return bytes(v & 0xFF for v in x.v)


# ---- the cases ------------------------------------------------------------------------------------------------------
def small(name):
    """a small network for each rewrite form, seeded: (graph, lower options)"""
    r = random.Random(name)

    def cv(cin, cout, k, s, pad, act="relu", nm=None, out=(1 / 64, -128)):
        return {"op": "conv2d", "name": nm or "c%d" % r.randrange(10 ** 6), "filters": cout, "kernel": list(k),
                "strides": list(s), "padding": pad, "activation": act,
                "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 300, "zero_point": 0,
                            "data": [r.randint(-90, 90) for _ in range(cout * k[0] * k[1] * cin)]},
                "bias": [r.randint(-900, 900) for _ in range(cout)], "output": {"scale": out[0], "zero_point": out[1]}}

    def fc(K, units, act="none"):
        return {"op": "fully_connected", "name": "fc%d" % r.randrange(10 ** 6), "units": units, "activation": act,
                "weights": {"dtype": "int8", "layout": "OI", "scale": 1 / 900, "zero_point": 0,
                            "data": [r.randint(-90, 90) for _ in range(K * units)]},
                "bias": [r.randint(-900, 900) for _ in range(units)], "output": {"scale": 1 / 16, "zero_point": 5}}

    def pl(typ, k, s, pad="valid"):
        return {"op": typ + "_pool2d", "name": "p%d" % r.randrange(10 ** 6), "pool": list(k), "strides": list(s),
                "padding": pad}

    def net(shape, layers):
        return {"format": "npu-net/1", "name": name, "layers": layers,
                "input": {"name": "x", "shape": [1] + list(shape), "dtype": "int8", "scale": 1 / 50, "zero_point": -3}}
    flat = lambda n: {"op": "reshape", "name": "flat", "shape": [1, n]}
    sm = {"op": "softmax", "name": "sm", "beta": 1.0, "output": {"scale": 1 / 256, "zero_point": -128}}
    if name == "s-conv2x1-fc":       # unequal strides, FC through a RESHAPE (kws-cnn's form, SAME)
        return net((20, 9, 3), [cv(3, 8, (5, 3), (2, 1), "same"), flat(8 * 10 * 9), fc(8 * 10 * 9, 6), sm]), {"allow_none": True}
    if name == "s-conv1x3-fc":       # unequal strides 1x3, VALID, FC read directly
        return net((12, 17, 2), [cv(2, 6, (3, 3), (1, 3), "valid"), fc(6 * 10 * 5, 6)]), {"allow_none": True}
    if name == "s-conv3-fc":         # stride 3 SAME: at stride 1 SAME, an FC on rows/columns 0, 3, ..
        return net((16, 16, 3), [cv(3, 6, (3, 3), (3, 3), "same"), flat(6 * 36), fc(6 * 36, 6), sm]), {"allow_none": True}
    if name == "s-conv4same-fc":     # stride 4 SAME: at stride 1 SAME, an FC on rows/columns 2, 6, 10, 14
        return net((16, 16, 3), [cv(3, 6, (5, 5), (4, 4), "same"), flat(6 * 16), fc(6 * 16, 6)]), {"allow_none": True}
    if name == "s-pool3x1-fc":       # magic-wand's pooling, average this time
        return net((30, 1, 4), [cv(4, 8, (3, 1), (1, 1), "same"), pl("average", (3, 1), (3, 1), "same"), flat(8 * 10),
                                fc(8 * 10, 6), sm]), {"allow_none": True}
    if name == "s-pool2s3-fc":       # pooling stride larger than its kernel
        return net((16, 16, 3), [cv(3, 6, (3, 3), (1, 1), "same"), pl("max", (2, 2), (3, 3)), flat(6 * 25), fc(6 * 25, 6)]), \
            {"allow_none": True}
    if name == "s-conv4-conv":       # stride 4 VALID, read by a conv: stride 2 + the identity conv (alexnet's form)
        return net((23, 23, 3), [cv(3, 8, (3, 3), (4, 4), "valid"), cv(8, 6, (3, 3), (1, 1), "same"), flat(6 * 36),
                                 fc(6 * 36, 6), sm]), {"allow_none": True}
    if name == "s-pool4-conv":       # pooling 2x2 stride 4 on the input (no ReLU before it): the identity conv, no activation
        return net((16, 16, 3), [pl("max", (2, 2), (4, 4)), cv(3, 6, (3, 3), (1, 1), "same")]), {"allow_none": True}
    if name == "s-pool1s2-conv":     # TF-slim's subsample: a 1x1 stride-2 max pool
        return net((16, 16, 3), [cv(3, 8, (3, 3), (1, 1), "same"), pl("max", (1, 1), (2, 2)), cv(8, 6, (1, 1), (1, 1), "same")]), {}
    raise KeyError(name)


SMALL = ["s-conv2x1-fc", "s-conv1x3-fc", "s-conv3-fc", "s-conv4same-fc", "s-pool3x1-fc", "s-pool2s3-fc", "s-conv4-conv",
         "s-pool4-conv", "s-pool1s2-conv"]
ZOO = ["kws-cnn", "magic-wand", "alexnet-96"]


def run_input(args):
    """case, input number -> (equal, the rewrites' descriptions)"""
    name, k = args
    if name in zoo.MODELS:
        net = zoo.build(name)
        g, kw = net.graph(name), {"quant": net.quant, "allow_none": net.none}
    else:
        g, kw = small(name)
    low = npunet.lower(g, **kw)
    rewritten = [w.split(", rewritten exactly into ")[1].split(",")[0] for w in low["warnings"] if "rewritten exactly" in w]
    C, H, W = low["input"]["chw"]
    x = random.Random("rewrite_ref/%s/%d" % (name, k)).randbytes(C * H * W)
    return bool(rewritten) and npunet.reference(low, x) == direct(g, x), rewritten


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--write", metavar="DIR")
    a = ap.parse_args()
    if a.write:
        os.makedirs(a.write, exist_ok=True)
        with open(os.path.join(a.write, "specs.txt"), "w") as f:
            for c in a.cases or SMALL:
                g, kw = small(c)
                json.dump(g, open(os.path.join(a.write, c + ".json"), "w"))
                f.write("%s|%s\n" % (c, "--allow-none" if kw.get("allow_none") else ""))
        print("rewrite_ref: %d networks in %s" % (len(a.cases or SMALL), a.write))
        return 0
    cases = a.cases or ZOO + SMALL
    ok = True
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_input, [(c, k) for c in cases for k in range(a.inputs)]))
    for i, name in enumerate(cases):
        rs = res[i * a.inputs:(i + 1) * a.inputs]
        n = sum(1 for e, _ in rs if e)
        print("rewrite_ref: %-16s %s  %d of %d random inputs equal; rewritten: %s" % (
            name, "ok" if n == a.inputs else "FAIL", n, a.inputs, "; ".join(rs[0][1]) or "nothing"))
        ok = ok and n == a.inputs
    print("rewrite_ref: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
