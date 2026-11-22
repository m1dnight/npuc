#!/usr/bin/env python3
"""Depthwise convolutions through Samsung's NPUC on this host (docs/format/depthwise.md): each network is written three
ways and compiled with the local oracle (npuc-local.sh), to see what NPUC makes of a depthwise layer.

  python3 -I oracle/ops/dwsweep.py <set> <outdir> [--root <run dir>] [-j N] [--only <name>...]
  sets: grid (single depthwise layers: C, stride, padding, activation, quant, size), blocks (MobileNet blocks:
        dw 3x3 + pw 1x1, chains of them), mnv1 (MobileNet v1 0.25 on 96x96x1, as TFLM's person detection), random
        (an unseen seeded set: single layers and dw/pw chains), all (the first three)

For every network <outdir>/<name>/:
  net.json    npu-net/1 (ncpgen/npunet.py's input), every depthwise layer written as a regular conv2d whose OHWI kernel
              is the dense C x C diagonal one (the channel's taps where o == i, zero elsewhere): npunet lowers it as is
  dense/      npuc-local.sh of net.json: lowered.json, net.npuc, NPUC's net.ncp (ncpgen can check this directory)
  stock/      dense/net.npuc with each depthwise layer's conv line turned into "dwconv <name> <input> stock ..." (the
              diagonal taps only; npuc-net expands them back and sets ConvAttr.group = C): what NPUC does on the device
  dwopt/      the same with "dwconv ... dwopt": NPUC's depthwise path, which its on-device frontend never takes
              (NNGraphAttribute::DwOptSupported wants the framework "scaffe"; npuc-net's hook lets it see "scaffe")
and one line per network in <outdir>/report.tsv: name, the three NCP sha256 prefixes (or the failure), stock == dense,
DwOpt applied (NPUC's log: "Group is <C>" and the hook's answer). Our own seeded weights; the outputs are
vendor-generated (keep <outdir> in the work folder).
"""
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "..", "npuc-local.sh")


# ---- the networks -------------------------------------------------------------------------------------------------
# a layer: ("dw", k, s, pad, act) depthwise k x k, multiplier 1 | ("pw", cout, act) 1x1 conv |
#          ("conv", cout, k, s, pad, act) | ("gap",) global average pool | ("fc", units)
def grid():
    nets = []
    for c in (8, 16, 24, 32, 48, 64, 96, 128, 256):
        for s in (1, 2):
            for pad in ("same", "valid"):
                nets.append(("g-c%d-s%d-%s" % (c, s, pad), 16, 16, c, 3, [("dw", 3, s, pad, 1)]))
    for c in (16, 64):
        nets.append(("g-c%d-noact" % c, 16, 16, c, 3, [("dw", 3, 1, "same", 0)]))
        nets.append(("g-c%d-q1" % c, 16, 16, c, 1, [("dw", 3, 1, "same", 1)]))
        nets.append(("g-c%d-32x32" % c, 32, 32, c, 3, [("dw", 3, 1, "same", 1)]))
        nets.append(("g-c%d-k5" % c, 16, 16, c, 3, [("dw", 5, 1, "same", 1)]))
        nets.append(("g-c%d-7x7-s2" % c, 7, 7, c, 3, [("dw", 3, 2, "same", 1)]))
    for c in (2, 3, 4, 12, 20, 40, 100):
        nets.append(("g-c%d" % c, 16, 16, c, 3, [("dw", 3, 1, "same", 1)]))
    nets.append(("g-c512", 8, 8, 512, 3, [("dw", 3, 1, "same", 1)]))
    nets.append(("g-c32-48x48-s2", 48, 48, 32, 3, [("dw", 3, 2, "same", 1)]))
    return nets


def blocks():
    nets = []
    for c, cout, s in ((8, 16, 1), (16, 32, 2), (32, 32, 1), (32, 64, 2), (64, 64, 1), (64, 128, 2), (128, 128, 1),
                       (128, 256, 2), (256, 256, 1)):
        hw = 24 if c <= 64 else 12
        nets.append(("b-c%d-%d-s%d" % (c, cout, s), hw, hw, c, 3, [("dw", 3, s, "same", 1), ("pw", cout, 1)]))
    nets.append(("b-conv-dw-pw", 24, 24, 3, 3, [("conv", 16, 3, 2, "same", 1), ("dw", 3, 1, "same", 1), ("pw", 32, 1)]))
    nets.append(("b-2blocks", 24, 24, 16, 3, [("dw", 3, 1, "same", 1), ("pw", 32, 1), ("dw", 3, 2, "same", 1),
                                               ("pw", 64, 1)]))
    nets.append(("b-c32-q1", 24, 24, 32, 1, [("dw", 3, 1, "same", 1), ("pw", 32, 1)]))
    return nets


def mnv1(alpha=0.25, hw=96, cin=1, classes=2):
    """MobileNet v1 (alpha, 96x96x1): conv 3x3 s2, 13 dw/pw blocks, global average pool, a 1x1 classifier (TFLM's
    person detection: 0.25, 96x96 grayscale, 2 classes). Every dw/pw/conv with ReLU (ReLU6 in the original: NPUC's
    conv has ReLU only, G for the clip)"""
    def ch(n):
        return max(8, int(n * alpha))
    L = [("conv", ch(32), 3, 2, "same", 1)]
    for cout, s in ((64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1), (512, 1), (512, 1), (512, 1),
                    (512, 1), (1024, 2), (1024, 1)):
        L += [("dw", 3, s, "same", 1), ("pw", ch(cout), 1)]
    L += [("gap",), ("fc", classes)]
    name = "mnv1-%03d-%d" % (int(alpha * 100), hw)
    return [(name, hw, hw, cin, 3, L), (name + "-q1", hw, hw, cin, 1, L)] + \
        [(name + "-first%d" % n, hw, hw, cin, 3, L[:n]) for n in (3, 5, 7, 9, 11, 15, 19, 23, 27)]


def rand(n=40, seed=20261122):
    """an unseen random set: single depthwise layers (C 2..300, sizes 3..40, k 1/3/5/7, stride 1/2, SAME/VALID, ReLU
    or not, quant 1/3) and random dw/pw chains"""
    R = random.Random(seed)
    nets = []
    for i in range(n):
        q = R.choice((1, 3))
        if i % 2 == 0:
            c = R.choice((R.randint(2, 64), R.randint(2, 300), 16 * R.randint(1, 16)))
            k = R.choice((1, 3, 3, 3, 5, 7))
            hw = R.randint(max(3, k), 40)
            L = [("dw", k, R.choice((1, 2)), R.choice(("same", "valid")), R.choice((0, 1, 1)))]
        else:
            c = R.choice((8, 16, 24, 32, 48, 64, R.randint(3, 96)))
            hw = R.randint(8, 40)
            L, cc = [], c
            for _ in range(R.randint(1, 4)):
                cc = R.choice((cc, 2 * cc, 16 * R.randint(1, 8), R.randint(4, 128)))
                L += [("dw", 3, R.choice((1, 1, 2)), "same", 1), ("pw", cc, 1)]
        nets.append(("r%02d-c%d-%dx%d-q%d" % (i, c, hw, hw, q), hw, hw, c, q, L))
    return nets


SETS = {"grid": grid, "blocks": blocks, "mnv1": mnv1, "random": rand}


def out_hw(h, k, s, pad):
    return -(-h // s) if pad == "same" else (h - k) // s + 1


def build_json(name, H, W, C, quant, layers, seed):
    """the npu-net/1 graph, depthwise layers as dense diagonal conv2d; returns (graph, names of the depthwise layers)"""
    R = random.Random(seed)
    if quant == 3:
        inp = {"name": "image", "shape": [1, H, W, C], "dtype": "int8", "scale": 1 / 128, "zero_point": 0}
    else:
        inp = {"name": "image", "shape": [1, H, W, C], "dtype": "uint8", "scale": 1 / 128, "zero_point": 128}
    zp = inp["zero_point"]
    g = {"format": "npu-net/1", "name": name, "input": inp, "layers": []}
    dws = []
    h, w, c = H, W, C
    for i, L in enumerate(layers, 1):
        kind = L[0]
        if kind in ("dw", "pw", "conv", "fc"):
            if kind == "dw":
                _, k, s, pad, act = L
                cout = c
                taps = [R.randint(-24, 24) for _ in range(c * k * k)]
                data = [0] * (cout * k * k * c)
                for o in range(c):  # OHWI: [o][y][x][i], nonzero only where i == o
                    for t in range(k * k):
                        data[(o * k * k + t) * c + o] = taps[o * k * k + t]
                lname = "dw%d" % i
                dws.append(lname)
            elif kind == "pw":
                _, cout, act = L
                k, s, pad = 1, 1, "same"
                data = [R.randint(-24, 24) for _ in range(cout * c)]
                lname = "pw%d" % i
            elif kind == "conv":
                _, cout, k, s, pad, act = L
                data = [R.randint(-24, 24) for _ in range(cout * k * k * c)]
                lname = "conv%d" % i
            else:  # a classifier over the pooled 1 x 1 x C: a 1x1 conv (npunet lowers FULLY_CONNECTED the same way)
                _, cout = L
                k, s, pad, act = 1, 1, "same", 0
                data = [R.randint(-24, 24) for _ in range(cout * c)]
                lname = "fc%d" % i
            g["layers"].append({"op": "conv2d", "name": lname, "filters": cout, "kernel": [k, k], "strides": [s, s],
                                "padding": pad, "activation": "relu" if act else "none",
                                "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 256, "zero_point": 0,
                                            "data": data},
                                "bias": [R.randint(-2000, 2000) for _ in range(cout)],
                                "output": {"scale": 1 / 128, "zero_point": zp}})
            h, w, c = out_hw(h, k, s, pad), out_hw(w, k, s, pad), cout
        elif kind == "gap":
            g["layers"].append({"op": "average_pool2d", "name": "gap%d" % i, "pool": [h, w], "strides": [1, 1],
                                "padding": "valid"})
            h = w = 1
        else:
            raise ValueError(kind)
    return g, dws


CONV = re.compile(r"^conv (\S+) (\S+) (\d+) (\d+) (\d+) (\d+) (\d+) (\S+) ([01]) (\S+) (\d+) (\S+) (\d+) (\S+) (\d+) "
                  r"([0-9a-f]+) (\d+)((?: -?\d+)*)\s*$")


def derive(npuc_text, dws, mode):
    """dense net.npuc -> the same network with each depthwise layer as a dwconv line (diagonal taps only)"""
    lines = npuc_text.split("\n")
    if lines[0] not in ("npuc-net 1", "npuc-net 2"):
        raise ValueError("not an npuc-net file")
    lines[0] = "npuc-net 2"
    want = set(dws)
    done = set()
    for i, ln in enumerate(lines):
        m = CONV.match(ln)
        if not m:
            continue
        name = m.group(1)
        base = re.sub(r"^l\d+_", "", name)
        if base not in want:
            continue
        cout, kh, kw = int(m.group(3)), int(m.group(4)), int(m.group(5))
        wz = int(m.group(13))
        wb = bytes.fromhex(m.group(16))
        c = cout
        kk = kh * kw
        if len(wb) != c * c * kk:
            raise ValueError("%s: %d weight bytes for a %d x %d x %d x %d diagonal" % (name, len(wb), c, c, kh, kw))
        diag = bytearray()
        for o in range(c):
            for ci in range(c):
                blk = wb[(o * c + ci) * kk:(o * c + ci + 1) * kk]
                if ci == o:
                    diag += blk
                elif any(b != wz for b in blk):
                    raise ValueError("%s: off-diagonal weight != the zero point %d" % (name, wz))
        g = list(m.groups())
        lines[i] = " ".join(["dwconv", g[0], g[1], mode] + g[3:15] + [bytes(diag).hex(), g[16]]) + g[17]
        done.add(base)
    if done != want:
        raise ValueError("depthwise layers not found in net.npuc: %s" % sorted(want - done))
    return "\n".join(lines)


def run_local(src, out, root, quant=None, allow_none=False):
    env = dict(os.environ)
    if root:
        env["NPU_ROOT"] = root
    cmd = ["bash", LOCAL, src, out] + (["--quant", str(quant)] if quant else []) + (["--allow-none"] if allow_none else [])
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = p.stdout.strip().split("\n")
    m = re.search(r"npuc-local: PASS \S+ ([0-9a-f]{64})", p.stdout)
    if m:
        return m.group(1)[:8], ""
    why = [ln for ln in tail if re.search(r"Assert|assert|abort|crash: (signal|pc)|npuc-net: FAIL|Refused|error", ln)]
    return None, " / ".join(why[-3:]) or tail[-1]


def one(net, outdir, root):
    name, H, W, C, quant, layers = net
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    g, dws = build_json(name, H, W, C, quant, layers, seed=hash_seed(name))
    with open(os.path.join(d, "net.json"), "w") as f:
        json.dump(g, f)
    res = {"name": name}
    # a layer without ReLU (a classifier, a "noact" variant) needs npunet's --allow-none (NPUC's GRELU bypass)
    none = any(L["activation"] == "none" for L in g["layers"] if L["op"] == "conv2d")
    res["dense"], res["dense_why"] = run_local(os.path.join(d, "net.json"), os.path.join(d, "dense"), root, quant, none)
    npuc = os.path.join(d, "dense", "net.npuc")
    if not os.path.exists(npuc):
        res["stock"] = res["dwopt"] = None
        res["stock_why"] = res["dwopt_why"] = "no dense net.npuc"
        return res
    text = open(npuc).read()
    for mode in ("stock", "dwopt"):
        try:
            t = derive(text, dws, mode)
        except ValueError as e:
            res[mode], res[mode + "_why"] = None, "derive: %s" % e
            continue
        p = os.path.join(d, mode + ".npuc")
        with open(p, "w") as f:
            f.write(t)
        res[mode], res[mode + "_why"] = run_local(p, os.path.join(d, mode), root)
    log = os.path.join(d, "dwopt", "log.txt")
    res["dwopt_applied"] = "?"
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        ans = re.findall(r'DwOptSupported with framework "scaffe" \(was "\w+"\) = (\d)', txt)
        res["dwopt_applied"] = ",".join(ans) or "-"
    return res


def hash_seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    which, outdir = a[0], a[1]
    root, jobs, only = None, 4, []
    i = 2
    while i < len(a):
        if a[i] == "--root":
            root = os.path.abspath(a[i + 1]); i += 2
        elif a[i] == "-j":
            jobs = int(a[i + 1]); i += 2
        elif a[i] == "--only":
            only = a[i + 1:]; break
        else:
            sys.exit("dwsweep: unknown option %s" % a[i])
    nets = []
    for s in (("grid", "blocks", "mnv1") if which == "all" else [which]):
        if s not in SETS:
            sys.exit("dwsweep: unknown set %s" % s)
        nets += SETS[s]()
    if only:
        nets = [n for n in nets if n[0] in only]
    os.makedirs(outdir, exist_ok=True)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(lambda n: one(n, outdir, root), nets):
            same = "yes" if r["stock"] and r["stock"] == r["dense"] else ("NO" if r["stock"] and r["dense"] else "-")
            row = [r["name"], r["dense"] or "FAIL", r["stock"] or "FAIL", r["dwopt"] or "FAIL", same,
                   r.get("dwopt_applied", "?"),
                   " | ".join(x for x in (r.get("dense_why"), r.get("stock_why"), r.get("dwopt_why")) if x)]
            rows.append(row)
            print("\t".join(row), flush=True)
    with open(os.path.join(outdir, "report.tsv"), "a") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    n = len(rows)
    print("dwsweep: %d networks, dense compiled %d, stock compiled %d (= dense %d), dwopt compiled %d" % (
        n, sum(r[1] != "FAIL" for r in rows), sum(r[2] != "FAIL" for r in rows), sum(r[4] == "yes" for r in rows),
        sum(r[3] != "FAIL" for r in rows)))


if __name__ == "__main__":
    main()
