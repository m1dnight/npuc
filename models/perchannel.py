#!/usr/bin/env python3
"""models/perchannel.py: what --fold-per-channel costs. TFLite quantises every CONV_2D and DEPTHWISE_CONV_2D with one
weight scale per output channel (FULLY_CONNECTED per tensor, in its int8 spec); NPUC's on-device filler takes one
scale per layer, so npunet folds them (each channel's weights and bias rescaled to the largest scale, rounded: a
channel whose scale is s_max / g keeps log2(127 / g) bits of its weights). This measures the output bytes that changes,
on the zoo's models made per-channel, against the per-channel model's own reference. Nothing here changes what
ncpgen generates.

  models/perchannel.py [--spreads 1,4,16,64] [--inputs N] [--jobs N] [--md] [model...]

The per-channel form of a zoo model (perchannel_graph): each CONV_2D / DEPTHWISE_CONV_2D output channel o gets a gain
g_o in [1/R, 1] (log-uniform, deterministic: shake_256 of the layer and the channel; the largest 1, the smallest 1/R,
so R is the ratio of the largest to the smallest weight scale in the layer) on its weight scale, its int8 weights and
bias unchanged: TFLite's per-channel quantisation of the float weights w s g_o, full-range int8 in every channel.
The output scales stay (the largest channel keeps the range the zoo calibrated, the others use less of it), and each
later conv / FC's weight scale is divided by the rms gain of its input so every activation keeps its range. R = 1 is
the zoo's model itself (one scale: nothing folded).

Three references of the NPU part's output (the lowered network's last layer, before the CPU tail), on random inputs:
  A  the per-channel model, every multiplier in * w_o / out exact (to 2^-48: TFLite's own is a 31-bit fixed point)
  B  the folded model (npunet lower --fold-per-channel), multipliers exact: the cost of folding alone
  C  the folded model as npunet's reference computes it (NPUC's 16-bit slope, shift <= 15): what ncpgen's NCP computes
Each column: the share of output bytes that differ from A, the mean and the largest |difference| (LSB of the output),
and for the classifiers the share of inputs whose top-1 (the largest logit) is A's.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ncpgen import npunet  # noqa: E402
from ncpgen.common import f32, hex_f32  # noqa: E402
import zoo  # noqa: E402

# the published zoo models with a conv or depthwise layer (the -npu and -dense variants compute the same functions; the
# FC-only models have no per-channel layer)
MODELS = ["lenet5", "keras-mnist-cnn", "tf-cifar-cnn", "vgg-cifar", "resnet8", "kws-cnn", "micro-speech", "magic-wand",
          "squeezenet11-96", "alexnet-96", "vgg11-64", "yolov2-tiny-128", "mobilenet-v1-025", "mobilenet-v2-035", "ds-cnn"]
EXACT_SHIFT = 48


def gains(key, n, R):
    """n per-channel gains in [1/R, 1], log-uniform, the largest 1 and the smallest 1/R (n >= 2)"""
    if R == 1 or n < 2:
        return [1.0] * n
    u = [int.from_bytes(hashlib.shake_256(("ncpgen-perchannel/%s/%d" % (key, o)).encode()).digest(4), "little")
         for o in range(n)]
    lo, hi = min(u), max(u)
    return [R ** -((v - lo) / (hi - lo)) for v in u]


def perchannel_graph(g, R):
    """the npu-net/1 graph with per-channel weight scales (the module docstring); a chain or a graph"""
    g = copy.deepcopy(g)
    io = g["input"]
    shrink = {str(io.get("id") or io.get("name")): 1.0}
    prev = str(io.get("id") or io.get("name"))
    for L in g["layers"]:
        ins = [str(v) for v in L["inputs"]] if "inputs" in L else [str(L["input"])] if "input" in L else [prev]
        f_in = math.sqrt(sum(shrink[i] ** 2 for i in ins) / len(ins))
        f = f_in
        if L["op"] in ("conv2d", "depthwise_conv2d", "fully_connected"):
            wt = L["weights"]
            base = wt["scale"][0] if isinstance(wt["scale"], list) else wt["scale"]
            base /= f_in                            # the activation's range restored
            if L["op"] == "fully_connected":
                wt["scale"] = f32(base)
                f = 1.0
            else:
                n = L["filters"] if L["op"] == "conv2d" else len(wt["data"]) // (L["kernel"][0] * L["kernel"][1])
                gs = gains(g["name"] + "/" + L["name"], n, R)
                wt["scale"] = [f32(base * x) for x in gs] if R != 1 else f32(base)
                f = math.sqrt(sum(x * x for x in gs) / n)
        shrink[str(L["name"])] = f
        if "id" in L:
            shrink[str(L["id"])] = f
        prev = str(L["name"])
    return g


def _exact(m):
    """a multiplier as (slope, shift) to 2^-48: ref_conv's arithmetic with an exact multiplier"""
    return int(m * (1 << EXACT_SHIFT) + Fraction(1, 2)), EXACT_SHIFT


def conv_exact(l, x, scales=None):
    """a lowered conv with exact multipliers: one (its w_scale), or one per output channel (scales)"""
    i, o = Fraction(hex_f32(l["in_scale"])), Fraction(hex_f32(l["out_scale"]))
    if scales is None:
        s, sh = _exact(i * Fraction(hex_f32(l["w_scale"])) / o)
        return npunet.ref_conv(dict(l, slope=s, shift=sh), x, "half_away", False, 1)
    cout, (_, ho, wo) = l["cout"], l["out_chw"]
    K = len(l["weights_oihw_u8"]) // cout
    out = b""
    for c in range(cout):
        s, sh = _exact(i * Fraction(f32(scales[c])) / o)
        out += npunet.ref_conv(dict(l, cout=1, weights_oihw_u8=l["weights_oihw_u8"][c * K:(c + 1) * K],
                                    bias=[l["bias"][c]], slope=s, shift=sh, out_chw=[1, ho, wo]), x, "half_away", False, 1)
    return out


def ref_with(low, x, conv):
    """the NPU part of a lowered network on CHW input bytes, each conv by conv(l, x)"""
    vals = {"data0": x}
    for l in low["layers"]:
        ins = [vals[n] for n in l.get("inputs", [l["input"]])]
        k = l["kind"]
        vals[l["name"]] = (conv(l, ins[0]) if k == "conv" else npunet.ref_pool(l, ins[0]) if k == "pool" else
                           npunet.ref_eltwise(l, ins[0], ins[1]) if k == "eltwise" else b"".join(ins) if k == "concat"
                           else npunet.ref_relu(l, ins[0]))
    return vals[low["layers"][-1]["name"]]


def lowerings(name, R):
    """(A's network: unfolded weights, its per-channel scales by layer; the folded network) for model name at spread R"""
    net = zoo.build(name)
    g = perchannel_graph(net.graph(name), R)
    kw = {"quant": net.quant, "allow_none": net.none}
    pcs, gmax = {}, copy.deepcopy(g)
    for L in gmax["layers"]:
        s = L.get("weights", {}).get("scale")
        if isinstance(s, list):
            pcs[L["name"]] = s
            L["weights"]["scale"] = max(s)      # one scale: npunet keeps the int8 weights as they are
    return npunet.lower(gmax, **kw), pcs, npunet.lower(g, fold=True, **kw)


def run(task):
    name, R, k = task
    unf, pcs, fold = lowerings(name, R)
    C, H, W = unf["input"]["chw"]
    x = random.Random("perchannel/%s/%d" % (name, k)).randbytes(C * H * W)
    a = ref_with(unf, x, lambda l, v: conv_exact(l, v, pcs.get(l["src_name"])))
    b = ref_with(fold, x, conv_exact)
    c = npunet.ref_npu(fold, x)
    return name, R, a, b, c


def stats(pairs):
    """[(A, X) per input] -> (share of differing bytes, mean |diff|, max |diff|, share of top-1 equal)"""
    n = sum(len(a) for a, _ in pairs)
    d = [abs(p - q) for a, x in pairs for p, q in zip(a, x)]
    top = sum(1 for a, x in pairs if a.index(max(a)) == x.index(max(x))) / len(pairs)
    return sum(1 for v in d if v) / n, sum(d) / n, max(d), top


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--spreads", default="1,4,16,64")
    ap.add_argument("--inputs", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    models = a.models or MODELS
    spreads = [int(v) for v in a.spreads.split(",")]
    tasks = [(m, R, k) for m in models for R in spreads for k in range(a.inputs)]
    tasks.sort(key=lambda t: -zoo.build(t[0], shapes_only=True).nweights)
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, R, A, B, C in ex.map(run, tasks):
            res.setdefault((name, R), []).append((A, B, C))
    rows = {}
    for (name, R), v in res.items():
        clf = zoo.build(name, shapes_only=True).layers[-1]["op"] == "softmax"
        rows[(name, R)] = (stats([(A, B) for A, B, _ in v]), stats([(A, C) for A, _, C in v]), clf, len(v[0][0]))
    cell = lambda s, clf: "%.1f %% / %.2f / %d%s" % (100 * s[0], s[1], s[2], " / %d %%" % round(100 * s[3]) if clf else "")
    for m in models:
        for R in spreads:
            (b, c, clf, nb) = rows[(m, R)]
            print("perchannel: %-18s R %-3d (%d inputs x %d B)  B folding: %s   C ncpgen: %s" % (
                m, R, a.inputs, nb, cell(b, clf), cell(c, clf)), flush=True)
    if a.md:        # the table of models/README.md, after the lines
        print("\n| model (output bytes) | " + " | ".join("R = %d: B (folding) | R = %d: C (ncpgen)" % (R, R)
                                                         for R in spreads) + " |")
        print("|---|" + "---|---|" * len(spreads))
        for m in models:
            print("| %s (%d) | %s |" % (m, rows[(m, spreads[0])][3], " | ".join(
                "%s | %s" % (cell(rows[(m, R)][0], rows[(m, R)][2]), cell(rows[(m, R)][1], rows[(m, R)][2])) for R in spreads)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
