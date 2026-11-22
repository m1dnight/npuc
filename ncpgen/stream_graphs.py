#!/usr/bin/env python3
"""The graph-op sweep of ncpgen/stream.py: npu-net/1 networks with ADD (eltwise), CONCATENATION and standalone RELU
layers (ncpgen/npunet.py lowers them, npuc-net 2; docs/harness-ops.md), compiled by the NPUC oracle with the
harness run directory that supports them. Our own seeded weights; quant 3, zero points 0, power-of-two scales (what
NPUC's Eltwise needs).

  stream_graphs.py write <dir>       <dir>/<name>.json for every variant
  stream_graphs.py compile <dir> [name...]   write, then npuc-local.sh each into <dir>/<name>/ (NPU_ROOT below)
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(os.path.dirname(HERE), "oracle")
# graph ops need a harness that reads "npuc-net 2" (harness/build.sh's run directory); NPU_ROOT overrides
ROOT = os.environ.get("NPU_ROOT") or os.path.join(os.environ.get("NCPGEN_WORK") or os.path.join(HERE, "..", "work"),
                                                  "n3", "root-current")


class Net:
    def __init__(self, name, h, w, c, seed=1):
        self.r = random.Random(seed)
        self.g = {"format": "npu-net/1", "name": name,
                  "input": {"name": "image", "shape": [1, h, w, c], "dtype": "int8", "scale": 1 / 128, "zero_point": 0},
                  "layers": []}
        self.ch = {"image": c}
        self.hw = {"image": (h, w)}
        self.sc = {"image": 1 / 128}

    def conv(self, name, inp, cout, k=3, act="relu", out=1 / 128, stride=1, pad="same"):
        cin = self.ch[inp]
        h, w = self.hw[inp]
        if pad == "same":
            ho, wo = -(-h // stride), -(-w // stride)
        else:
            ho, wo = (h - k) // stride + 1, (w - k) // stride + 1
        self.g["layers"].append({
            "op": "conv2d", "name": name, "input": inp, "filters": cout, "kernel": [k, k], "strides": [stride, stride],
            "padding": pad, "activation": act,
            "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 256, "zero_point": 0,
                        "data": [self.r.randint(-24, 24) for _ in range(cout * k * k * cin)]},
            "bias": [self.r.randint(-1500, 1500) for _ in range(cout)],
            "output": {"scale": out, "zero_point": 0}})
        self.ch[name], self.hw[name], self.sc[name] = cout, (ho, wo), out
        return name

    def pool(self, name, inp, kind="max", k=2, s=2, pad="valid"):
        h, w = self.hw[inp]
        ho, wo = ((h - k) // s + 1, (w - k) // s + 1) if pad == "valid" else (-(-h // s), -(-w // s))
        self.g["layers"].append({"op": "max_pool2d" if kind == "max" else "average_pool2d", "name": name, "input": inp,
                                 "pool": [k, k], "strides": [s, s], "padding": pad})
        self.ch[name], self.hw[name], self.sc[name] = self.ch[inp], (ho, wo), self.sc[inp]
        return name

    def add(self, name, a, b, act="none", out=1 / 64):
        self.g["layers"].append({"op": "add", "name": name, "inputs": [a, b], "activation": act,
                                 "output": {"scale": out, "zero_point": 0}})
        self.ch[name], self.hw[name], self.sc[name] = self.ch[a], self.hw[a], out
        return name

    def concat(self, name, ins, act="none"):
        out = self.sc[ins[0]]
        self.g["layers"].append({"op": "concat", "name": name, "inputs": list(ins), "axis": 3, "activation": act,
                                 "output": {"scale": out, "zero_point": 0}})
        self.ch[name], self.hw[name], self.sc[name] = sum(self.ch[i] for i in ins), self.hw[ins[0]], out
        return name

    def relu(self, name, inp):
        self.g["layers"].append({"op": "relu", "name": name, "input": inp})
        self.ch[name], self.hw[name], self.sc[name] = self.ch[inp], self.hw[inp], self.sc[inp]
        return name


def variants():
    V = {}

    def mk(name, h=16, w=16, c=8, seed=1):
        n = Net(name, h, w, c, seed)
        V[name] = n
        return n

    # residual blocks (conv, conv, add of the block input [, relu])
    n = mk("res")
    n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-norelu")
    n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image")
    n = mk("res-conv")                   # a conv after the add
    n.conv("c3", n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image",
                       act="relu"), 8, out=1 / 128)
    n = mk("res16", c=16)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 16), 16, act="none", out=1 / 256), "image", act="relu")
    n = mk("res32", h=32, w=32, c=16, seed=2)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 16), 16, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-hw12", h=12, w=20, c=8, seed=3)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-c3", c=3, seed=4)        # 3 channels: uneven planes
    n.add("a1", n.conv("c2", n.conv("c1", "image", 3), 3, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-pool-before")
    p = n.pool("p0", n.conv("c0", "image", 8))
    n.add("a1", n.conv("c2", n.conv("c1", p, 8), 8, act="none", out=1 / 256), p, act="relu")
    n = mk("res-pool-after")
    n.pool("p1", n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image",
                       act="relu"))
    n = mk("res-two")                    # two residual blocks
    a = n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image", act="relu",
              out=1 / 128)
    n.add("a2", n.conv("c4", n.conv("c3", a, 8), 8, act="none", out=1 / 256), a, act="relu")
    # adds of two convs
    n = mk("add2")
    n.add("a1", n.conv("c1", "image", 8, out=1 / 128), n.conv("c2", "image", 8, out=1 / 128))
    n = mk("add2-relu")
    n.add("a1", n.conv("c1", "image", 8, act="none", out=1 / 128), n.conv("c2", "image", 8, act="none", out=1 / 128),
          act="relu")
    n = mk("add2-k1")
    n.add("a1", n.conv("c1", "image", 16, k=1, out=1 / 128), n.conv("c2", "image", 16, k=1, out=1 / 128))
    n = mk("add-scales")                 # different fractional lengths
    n.add("a1", n.conv("c1", "image", 8, out=1 / 32), "image", out=1 / 16)
    # concat
    n = mk("cat2")
    n.conv("c3", n.concat("k1", [n.conv("c1", "image", 8), n.conv("c2", "image", 8)]), 8)
    n = mk("cat2-out")
    n.concat("k1", [n.conv("c1", "image", 8), n.conv("c2", "image", 8)])
    n = mk("cat3")
    n.conv("c4", n.concat("k1", [n.conv("c1", "image", 8), n.conv("c2", "image", 8), n.conv("c3", "image", 8)]), 8)
    n = mk("cat16")
    n.conv("c3", n.concat("k1", [n.conv("c1", "image", 16), n.conv("c2", "image", 16)]), 16)
    n = mk("cat-8-16")
    n.conv("c3", n.concat("k1", [n.conv("c1", "image", 8), n.conv("c2", "image", 16)]), 8)
    n = mk("cat-4-12")
    n.concat("k1", [n.conv("c1", "image", 4), n.conv("c2", "image", 12)])
    n = mk("cat-relu")
    n.conv("c3", n.concat("k1", [n.conv("c1", "image", 8, act="none"), n.conv("c2", "image", 8, act="none")],
                          act="relu"), 8)
    n = mk("cat-pool")
    n.concat("k1", [n.pool("p1", n.conv("c1", "image", 8)), n.pool("p2", n.conv("c2", "image", 8))])
    n = mk("cat-chain")                  # a conv on one branch before the concat
    c1 = n.conv("c1", "image", 8)
    n.concat("k1", [n.conv("c2", c1, 8), n.conv("c3", "image", 8)])
    n = mk("cat-k1-k3")
    n.concat("k1", [n.conv("c1", "image", 8, k=1), n.conv("c2", "image", 8, k=3)])
    # standalone relu
    n = mk("relu-pool")
    n.relu("r1", n.pool("p1", n.conv("c1", "image", 8, act="none")))
    n = mk("relu-pool-conv")
    n.conv("c2", n.relu("r1", n.pool("p1", n.conv("c1", "image", 8, act="none"))), 8)
    n = mk("relu-avg16", c=16)
    n.relu("r1", n.pool("p1", n.conv("c1", "image", 16, act="none"), kind="avg"))
    # second round: the input read twice, wider tensors, concat in SRAM, more relu shapes
    n = mk("in-two")                     # the network input read by a conv after another branch ran
    n.add("a1", n.conv("c2", n.conv("c1", "image", 8, k=1), 8, out=1 / 128), n.conv("c3", "image", 8, out=1 / 128))
    n = mk("add-chain")
    a = n.add("a1", n.conv("c1", "image", 8, out=1 / 256), "image", act="relu", out=1 / 128)
    n.add("a2", n.conv("c2", a, 8, out=1 / 256), a, act="relu")
    n = mk("res-k1-32", h=8, w=8, c=32, seed=5)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 32, k=1), 32, k=1, act="none", out=1 / 256), "image", act="relu")
    n = mk("res64", h=8, w=8, c=64, seed=6)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 64), 64, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-big", h=64, w=64, c=16, seed=7)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 16), 16, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-wide", h=8, w=136, c=8, seed=8)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), "image", act="relu")
    n = mk("add2-c24", c=8, seed=9)
    n.add("a1", n.conv("c1", "image", 24, out=1 / 128), n.conv("c2", "image", 24, out=1 / 128), act="relu")
    n = mk("add2-big", h=40, w=40, c=8, seed=10)
    n.add("a1", n.conv("c1", "image", 32, out=1 / 128), n.conv("c2", "image", 32, out=1 / 128))
    n = mk("res-pp", seed=11)            # pool before and after
    p = n.pool("p0", n.conv("c0", "image", 16))
    n.pool("p1", n.add("a1", n.conv("c2", n.conv("c1", p, 16), 16, act="none", out=1 / 256), p, act="relu"))
    n = mk("cat-16-8", seed=12)
    n.concat("k1", [n.conv("c1", "image", 16), n.conv("c2", "image", 8)])
    n = mk("cat-32-16", seed=13)
    n.conv("c3", n.concat("k1", [n.conv("c1", "image", 32), n.conv("c2", "image", 16)]), 8)
    n = mk("cat16-out", seed=14)
    n.concat("k1", [n.conv("c1", "image", 16), n.conv("c2", "image", 16)])
    n = mk("cat16-3", seed=15)
    n.conv("c4", n.concat("k1", [n.conv("c1", "image", 16), n.conv("c2", "image", 16), n.conv("c3", "image", 16)]),
           8)
    n = mk("cat16-relu", seed=16)
    n.conv("c3", n.concat("k1", [n.conv("c1", "image", 16, act="none"), n.conv("c2", "image", 16, act="none")],
                          act="relu"), 8)
    n = mk("cat16-pool", seed=17)
    n.concat("k1", [n.pool("p1", n.conv("c1", "image", 16)), n.pool("p2", n.conv("c2", "image", 16))])
    n = mk("cat16-chain", seed=18)
    n.conv("c4", n.concat("k1", [n.conv("c2", n.conv("c1", "image", 16), 16), n.conv("c3", "image", 16)]), 8)
    for c in (4, 24, 32, 64):
        n = mk("relu-c%d" % c, c=c, seed=20 + c)
        n.relu("r1", n.pool("p1", n.conv("c1", "image", c, act="none")))
    n = mk("relu-hw", h=20, w=28, c=8, seed=30)
    n.relu("r1", n.pool("p1", n.conv("c1", "image", 8, act="none"), k=3, s=2))
    n = mk("relu-big", h=96, w=96, c=8, seed=31)
    n.conv("c2", n.relu("r1", n.pool("p1", n.conv("c1", "image", 8, act="none"))), 8)
    # third round: event flags near 16 DMAs, the input read by separated branches
    n = mk("res64-norelu", h=8, w=8, c=64, seed=40)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 64), 64, act="none", out=1 / 256), "image")
    n = mk("res64-conv", h=8, w=8, c=64, seed=41)
    n.conv("c3", n.add("a1", n.conv("c2", n.conv("c1", "image", 64), 64, act="none", out=1 / 256), "image",
                       act="relu"), 64)
    n = mk("res48", h=8, w=8, c=48, seed=42)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 48), 48, act="none", out=1 / 256), "image", act="relu")
    n = mk("res64-deep", h=8, w=8, c=64, seed=43)
    n.add("a1", n.conv("c3", n.conv("c2", n.conv("c1", "image", 64), 64), 64, act="none", out=1 / 256), "image",
          act="relu")
    n = mk("add2-64-relu", h=8, w=8, c=64, seed=44)
    n.add("a1", n.conv("c1", "image", 64, out=1 / 128), n.conv("c2", "image", 64, out=1 / 128), act="relu")
    n = mk("add2-64-conv", h=8, w=8, c=64, seed=45)
    n.conv("c3", n.add("a1", n.conv("c1", "image", 64, out=1 / 128), n.conv("c2", "image", 64, out=1 / 128),
                       act="relu"), 64)
    n = mk("res32-k1", h=8, w=8, c=32, seed=46)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 32), 32, act="none", out=1 / 256), "image", act="relu")
    n = mk("add-branches", seed=47)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 8), 8, out=1 / 128),
          n.conv("c4", n.conv("c3", "image", 8), 8, out=1 / 128))
    n = mk("cat-branches", seed=48)
    n.concat("k1", [n.conv("c2", n.conv("c1", "image", 8), 8), n.conv("c4", n.conv("c3", "image", 8), 8)])
    # fourth round: what decides the eltwise META flag (channels, or the DMAs of the group before)
    n = mk("add-pool48", seed=50)
    n.add("a1", n.pool("p1", n.conv("c1", "image", 48, out=1 / 128)), n.pool("p2", n.conv("c2", "image", 48,
                                                                                         out=1 / 128)))
    for c in (20, 24, 40):
        n = mk("res%d" % c, h=8, w=8, c=c, seed=50 + c)
        n.add("a1", n.conv("c2", n.conv("c1", "image", c), c, act="none", out=1 / 256), "image", act="relu")
    n = mk("res-pool48", seed=51)
    p = n.pool("p0", n.conv("c0", "image", 48))
    n.add("a1", n.conv("c2", n.conv("c1", p, 48), 48, act="none", out=1 / 256), p, act="relu")
    n = mk("res48-4x4", h=4, w=4, c=48, seed=52)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 48), 48, act="none", out=1 / 256), "image", act="relu")
    n = mk("res32-big", h=32, w=32, c=32, seed=53)
    n.add("a1", n.conv("c2", n.conv("c1", "image", 32), 32, act="none", out=1 / 256), "image", act="relu")
    n = mk("add-in-first", seed=54)      # the network input as the eltwise's first operand
    n.add("a1", "image", n.conv("c2", n.conv("c1", "image", 8), 8, act="none", out=1 / 256), act="relu")
    # sixth round: the eltwise's DRAM operand loads in one or two parts (few channels)
    for c in (2, 3, 4, 6, 12):
        for h, w in ((16, 16), (24, 12), (8, 8), (16, 8), (8, 24)):
            n = mk("rc%d-%dx%d" % (c, h, w), h=h, w=w, c=c, seed=70 + c + h + w)
            n.add("a1", n.conv("c1", "image", c, act="none", out=1 / 256), "image", act="relu")
    # fifth round: the ReLU's weight chunk on 1, 2, 3 cores
    for c in (1, 2, 6, 10, 12, 18, 3):
        n = mk("relu-c%d" % c, c=c, seed=60 + c)
        n.relu("r1", n.pool("p1", n.conv("c1", "image", c, act="none")))
    return V


def random_net(name, seed):
    """a seeded random graph: a stem, then 1 to 4 blocks (residual from the input or a computed tensor, add of two
    branches, concat of 2 or 3 branches, a standalone ReLU or ReLU6 after a pool, add or concat, pools between)"""
    r = random.Random(seed)
    h = r.choice([8, 12, 16, 20, 24, 32, 40])
    w = r.choice([8, 12, 16, 20, 24, 28, 40])
    c0 = r.choice([3, 4, 8, 12, 16, 24, 32])
    n = Net(name, h, w, c0, seed)
    k = [0]

    def nm(p):
        k[0] += 1
        return "%s%d" % (p, k[0])

    def conv(t, c, act="relu", out=1 / 128):
        return n.conv(nm("c"), t, c, k=r.choice([1, 3, 3, 5]), act=act, out=out)

    t = "image"
    if r.random() < 0.6:
        t = conv(t, r.choice([8, 16, 24, 32, 48]))
        if r.random() < 0.3 and min(n.hw[t]) >= 8:
            t = n.pool(nm("p"), t, kind=r.choice(["max", "avg"]))
    for _ in range(r.randint(1, 4)):
        c = n.ch[t]
        kind = r.choice(["res", "res", "add", "cat", "cat", "relu"])
        if kind == "res":
            b = conv(t, c)
            if r.random() < 0.5:
                b = conv(b, c, act="none", out=1 / 256)
            t = n.add(nm("a"), b, t, act=r.choice(["relu", "none"]), out=1 / 128)
        elif kind == "add":
            cc = r.choice([8, 16, 24, 32, 40, 64])
            t = n.add(nm("a"), conv(t, cc), conv(t, cc), act=r.choice(["relu", "none"]), out=1 / 64)
        elif kind == "cat":
            br = []
            for _ in range(r.choice([2, 2, 3])):
                cc = r.choice([4, 8, 16, 16, 32])
                b = conv(t, cc)
                if r.random() < 0.3:
                    b = conv(b, cc)
                br.append(b)
            if r.random() < 0.2 and min(n.hw[t]) >= 8:
                br = [n.pool(nm("p"), b) for b in br]
            t = n.concat(nm("k"), br)
        else:
            if t == "image" or n.g["layers"][-1]["op"] == "conv2d":
                t = n.pool(nm("p"), t, kind=r.choice(["max", "avg"])) if min(n.hw[t]) >= 8 else \
                    n.add(nm("a"), conv(t, n.ch[t]), conv(t, n.ch[t]), out=1 / 64)
            t = n.relu(nm("r"), t)
            if r.random() < 0.3:
                n.g["layers"][-1]["op"] = "relu6"
        if r.random() < 0.25 and min(n.hw[t]) >= 8:
            t = n.pool(nm("p"), t, kind=r.choice(["max", "avg"]))
        if r.random() < 0.3:
            t = conv(t, r.choice([8, 16, 32]))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stream_graphs.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write", "compile"):
        p = sub.add_parser(name)
        p.add_argument("dir")
        p.add_argument("names", nargs="*")
    p = sub.add_parser("random", help="random graph networks g<seed>, compiled")
    p.add_argument("dir")
    p.add_argument("seed", type=int)
    p.add_argument("count", type=int, nargs="?", default=40)
    a = ap.parse_args(argv)
    if a.cmd == "random":
        V = {"g%03d" % s: random_net("g%03d" % s, s) for s in range(a.seed, a.seed + a.count)}
        names = list(V)
    else:
        V = variants()
        names = a.names or list(V)
    d = a.dir
    os.makedirs(d, exist_ok=True)
    for k in names:
        json.dump(V[k].g, open(os.path.join(d, k + ".json"), "w"))
    if a.cmd == "write":
        return 0
    env = dict(os.environ, NPU_ROOT=ROOT)
    procs = []
    for k in names:
        out = os.path.join(d, k)
        if os.path.exists(os.path.join(out, "net.ncp")) and os.environ.get("FORCE") != "1":
            continue
        procs.append((k, subprocess.Popen(["bash", os.path.join(ORACLE, "npuc-local.sh"),
                                           os.path.join(d, k + ".json"), out, "--quant", "3", "--allow-none"],
                                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        while len([p for _, p in procs if p.poll() is None]) >= 6:
            procs[0][1].wait()
    for k, p in procs:
        o = p.communicate()[0]
        print("%-16s %s" % (k, (o.strip().splitlines() or ["?"])[-1][:150]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
