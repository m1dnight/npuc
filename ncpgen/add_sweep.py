#!/usr/bin/env python3
"""The sweep of ADD with any scales (npunet --add-as-conv, docs/format/add.md): npu-net/1 networks whose ADDs have
arbitrary scales and zero points, lowered by npunet to a mixed-quantisation channel concat and a 1x1 conv over it,
compiled by the NPUC oracle with a harness that reads "concat ... mixed" (root-81d28366 or later). Our own seeded
weights and quantisations; uint8 models (quant 1, any zero point) and int8 ones (quant 3, zero points 0).

  add_sweep.py write <dir> [--set probe|probe2|random] [--seed S] [--count N]     <dir>/<name>.json per network
  add_sweep.py compile <dir> [name...] [--set probe|probe2|random] [--seed S] [--count N]
        write, lower each with --add-as-conv into <dir>/<name>/ (lowered.json, net.npuc), then npuc-local.sh it there
        (NPU_ROOT: the harness run directory)
Check: python3 -m ncpgen check <dir>/*/ ; the CPU reference against TFLite's ADD: models/camera.py addcheck.
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if not __package__:
    sys.path.insert(0, os.path.dirname(HERE))
from ncpgen import npunet  # noqa: E402

ORACLE = os.path.join(os.path.dirname(HERE), "oracle")
ROOT = os.environ.get("NPU_ROOT") or os.path.join(os.environ.get("NCPGEN_WORK") or os.path.join(HERE, "..", "work"),
                                                  "n3", "root-current")


class Net:
    """an npu-net/1 network with quantised tensors of any scale and zero point (dtype uint8: quant 1; int8 with zero
    points 0: quant 3)"""
    def __init__(self, name, h, w, c, seed, dtype="uint8"):
        self.r = random.Random(seed)
        self.dtype = dtype
        self.g = {"format": "npu-net/1", "name": name,
                  "input": {"name": "image", "shape": [1, h, w, c], "dtype": dtype, "scale": self.scale(),
                            "zero_point": self.zp()},
                  "layers": []}
        self.t = {"image": (c, h, w, self.g["input"]["scale"])}

    def scale(self):
        return round(0.004 * 2 ** self.r.uniform(0, 7), 7)       # 0.004 .. 0.5, not powers of two

    def zp(self):
        return self.r.randint(0, 255) if self.dtype == "uint8" else 0

    def conv(self, name, inp, cout, k=3, act="none", stride=1):
        cin, h, w, s_in = self.t[inp]
        out = self.scale()
        # a weight scale whose multiplier in x w / out is near 1/64 (NPUC: 8-bit slope, shift up to 15)
        ws = round(out / (s_in * 64) * 2 ** self.r.uniform(-1, 1), 9)
        ho, wo = -(-h // stride), -(-w // stride)
        self.g["layers"].append({
            "op": "conv2d", "name": name, "input": inp, "filters": cout, "kernel": [k, k], "strides": [stride, stride],
            "padding": "same", "activation": act,
            "weights": {"dtype": "int8", "layout": "OHWI", "scale": ws, "zero_point": 0,
                        "data": [self.r.randint(-40, 40) for _ in range(cout * k * k * cin)]},
            "bias": [self.r.randint(-3000, 3000) for _ in range(cout)],
            "output": {"scale": out, "zero_point": self.zp()}})
        self.t[name] = (cout, ho, wo, out)
        return name

    def pool(self, name, inp, kind="max", k=2, s=2):
        c, h, w, sc = self.t[inp]
        self.g["layers"].append({"op": "max_pool2d" if kind == "max" else "average_pool2d", "name": name, "input": inp,
                                 "pool": [k, k], "strides": [s, s], "padding": "valid"})
        self.t[name] = (c, (h - k) // s + 1, (w - k) // s + 1, sc)
        return name

    def add(self, name, a, b, act="none"):
        # the output scale near the inputs' (the camera models: s_in / s_out 0.15 .. 9)
        out = max(self.t[a][3], self.t[b][3]) * 2 ** self.r.uniform(-1.5, 0.7)
        zp = self.zp()
        if act == "relu6":      # RELU6 whose clamp is a no-op: 6.0 at or above the top of the output range
            out = min(out, 5.9 / (255 - zp)) if self.dtype == "uint8" else min(out, 5.9 / 127)
        self.g["layers"].append({"op": "add", "name": name, "inputs": [a, b], "activation": act,
                                 "output": {"scale": round(out, 7), "zero_point": zp}})
        self.t[name] = self.t[a][:3] + (round(out, 7),)
        return name


def probe_variants():
    """hand-made networks: each ADD shape the camera models have (a residual over a computed tensor, two branches, a
    chain of residuals, ADD of pools, ADD then conv, x + x), channel counts 4..64, a fused ReLU, quant 3"""
    V = {}

    def net(name, h, w, c, seed, dtype="uint8"):
        V[name] = Net(name, h, w, c, seed, dtype)
        return V[name]

    for C in (4, 8, 16, 24, 32, 48, 64):
        n = net("res-c%d" % C, 16, 16, 8, 100 + C)
        s = n.conv("stem", "image", C, act="relu")
        n.add("add", n.conv("proj", s, C), s)
    for C in (8, 16, 40):
        n = net("two-c%d" % C, 12, 20, 8, 200 + C)
        s = n.conv("stem", "image", 16, act="relu")
        n.add("add", n.conv("a", s, C), n.conv("b", s, C, k=1))
    n = net("chain3", 16, 16, 8, 300)
    t = n.conv("stem", "image", 24, act="relu")
    for i in range(3):
        e = n.conv("exp%d" % i, t, 48, k=1, act="relu")
        t = n.add("add%d" % i, n.conv("proj%d" % i, e, 24, k=1), t)
    n = net("chain-then-conv", 16, 16, 8, 301)
    t = n.conv("stem", "image", 16, act="relu")
    t = n.add("add0", n.conv("proj0", t, 16), t)
    t = n.add("add1", n.conv("proj1", t, 16), t)
    n.conv("head", t, 8, act="relu")
    n = net("pools", 16, 16, 8, 302)
    s = n.conv("stem", "image", 16, act="relu")
    n.add("add", n.pool("pa", n.conv("a", s, 16)), n.pool("pb", n.conv("b", s, 16), kind="avg"))
    n = net("relu", 16, 16, 8, 303)
    s = n.conv("stem", "image", 16, act="relu")
    n.add("add", n.conv("proj", s, 16), s, act="relu")
    n = net("relu6", 16, 16, 8, 304)
    s = n.conv("stem", "image", 16, act="relu")
    n.add("add", n.conv("proj", s, 16), s, act="relu6")
    n = net("xx", 16, 16, 8, 305)
    s = n.conv("stem", "image", 16, act="relu")
    n.conv("head", n.add("add", s, s), 8, act="relu")
    n = net("big-c24", 40, 40, 8, 306)
    s = n.conv("stem", "image", 24, act="relu")
    n.add("add", n.conv("proj", s, 24, k=1), s)
    n = net("strided", 32, 32, 8, 307)
    s = n.conv("stem", "image", 32, act="relu", stride=2)
    n.add("add", n.conv("proj", s, 32, k=1), s)
    for C in (8, 32):
        n = net("q3-c%d" % C, 16, 16, 8, 400 + C, dtype="int8")
        s = n.conv("stem", "image", C, act="relu")
        n.add("add", n.conv("proj", s, C), s)
    return V


def probe2_variants():
    """residuals at 40..80 px (NPUC's layer-block forwarding): a plain one (fw-), and one whose block expands to 64 or
    128 channels with a 3x3 (fx-, as MobileDet's fused blocks); at 80x80 with 128 channels the forwarding test fails
    and NPUC builds the ADD's concat in DRAM, which ncpgen refuses (docs/format/add.md 7)"""
    V = {}
    for hw, C, k in ((80, 16, 3), (64, 16, 3), (48, 16, 3), (80, 16, 1), (40, 32, 3)):
        name = "fw-%d-c%d-k%d" % (hw, C, k)
        n = V[name] = Net(name, hw, hw, 8, 500 + hw + C + k)
        s = n.conv("stem", "image", C, act="relu")
        n.conv("head", n.add("add", n.conv("proj", s, C, k=k), s), 8, act="relu")
    for hw, E, root1 in ((80, 64, False), (64, 64, False), (40, 64, False), (80, 128, False), (80, 128, True),
                         (64, 128, False)):
        name = "fx-%d-c16-e%d%s" % (hw, E, "-r1" if root1 else "")
        n = V[name] = Net(name, hw, hw, 8, (700 if E == 128 else 600) + hw + E + (16 if E != 128 else 0) + root1)
        s = n.conv("stem", "image", 16, act="relu")
        if root1:                   # the residual input a one-node layer (a 1x1 conv), as MobileDet's ADD outputs
            s = n.conv("s1", s, 16, k=1, act="relu")
        e = n.conv("exp", s, E, k=3, act="relu")
        n.conv("head", n.add("add", n.conv("proj", e, 16, k=1), s), 8, act="relu")
    return V


def random_net(name, seed):
    """a seeded random network: a stem conv, then 1 to 3 blocks (a residual over the current tensor, an ADD of two
    branches, an inverted-residual chain), pools and convs between; uint8 with any zero points (1 in 5: int8, quant 3)"""
    r = random.Random(seed)
    dtype = "int8" if r.random() < 0.2 else "uint8"
    n = Net(name, r.choice([8, 12, 16, 20, 24, 32, 40]), r.choice([8, 12, 16, 20, 24, 28, 40]),
            r.choice([3, 4, 8, 16]), seed, dtype)
    k = [0]

    def nm(p):
        k[0] += 1
        return "%s%d" % (p, k[0])

    t = n.conv(nm("c"), "image", r.choice([4, 8, 16, 24, 32, 48]), k=r.choice([1, 3]), act="relu")
    for _ in range(r.randint(1, 3)):
        C = n.t[t][0]
        kind = r.choice(["res", "res", "two", "chain"])
        act = r.choice(["none", "none", "relu"])
        if kind == "res":
            t = n.add(nm("a"), n.conv(nm("c"), t, C, k=r.choice([1, 3])), t, act=act)
        elif kind == "two":
            cc = r.choice([4, 8, 16, 24, 32, 40, 64])
            t = n.add(nm("a"), n.conv(nm("c"), t, cc, k=r.choice([1, 3])), n.conv(nm("c"), t, cc, k=1), act=act)
        else:
            for _ in range(r.randint(2, 3)):
                e = n.conv(nm("c"), t, 2 * C, k=1, act="relu")
                t = n.add(nm("a"), n.conv(nm("c"), e, C, k=1), t, act=act)
        if r.random() < 0.25 and min(n.t[t][1:3]) >= 8:
            t = n.pool(nm("p"), t, kind=r.choice(["max", "avg"]))
        if r.random() < 0.3:
            t = n.conv(nm("c"), t, r.choice([8, 16, 32]), act="relu")
    return n


def networks(a):
    if a.set == "random":
        return {"r%03d" % s: random_net("r%03d" % s, s) for s in range(a.seed, a.seed + a.count)}
    return probe2_variants() if a.set == "probe2" else probe_variants()


def lower_into(js, out, quant):
    """npunet's lowering with --add-as-conv --allow-none into out/ (lowered.json, net.npuc); None or the refusal"""
    try:
        low = npunet.load_model(js, quant, False, True, add_conv=True)
    except npunet.Refused as e:
        return str(e)
    os.makedirs(out, exist_ok=True)
    json.dump(low, open(os.path.join(out, "lowered.json"), "w"), indent=1)
    open(os.path.join(out, "net.npuc"), "w").write(npunet.npuc_text(low))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="add_sweep.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("write", "compile"))
    ap.add_argument("dir")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--set", choices=("probe", "probe2", "random"), default="probe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=40)
    a = ap.parse_args(argv)
    V = networks(a)
    names = a.names or list(V)
    os.makedirs(a.dir, exist_ok=True)
    for k in names:
        json.dump(V[k].g, open(os.path.join(a.dir, k + ".json"), "w"))
    if a.cmd == "write":
        return 0
    env = dict(os.environ, NPU_ROOT=ROOT)
    procs = []
    for k in names:
        out = os.path.join(a.dir, k)
        if os.path.exists(os.path.join(out, "net.ncp")) and os.environ.get("FORCE") != "1":
            continue
        q = 3 if V[k].dtype == "int8" else 1
        why = lower_into(os.path.join(a.dir, k + ".json"), out, q)
        if why:
            print("%-16s lowering refused: %s" % (k, why[:150]))
            continue
        procs.append((k, subprocess.Popen(["bash", os.path.join(ORACLE, "npuc-local.sh"), os.path.join(out, "net.npuc"),
                                           out], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        while len([p for _, p in procs if p.poll() is None]) >= 6:
            next(p for _, p in procs if p.poll() is None).wait()
    for k, p in procs:
        o = p.communicate()[0]
        print("%-16s %s" % (k, (o.strip().splitlines() or ["?"])[-1][:150]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
