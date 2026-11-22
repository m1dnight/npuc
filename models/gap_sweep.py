#!/usr/bin/env python3
"""models/gap_sweep.py: the sweep that sets npunet's limit for a global average pool lowered as a conv (npunet.py
_global_avg_as_conv). Each variant: an input kh x kw x C (int8, scale 1/128, zero point 0), a 1x1 conv C -> C with ReLU,
then the global average pool written as the conv npunet lowers it to (identity weights, weight scale 1/(kh*kw), bias
0, the input's quantisation, ReLU), in quant mode 1 or 3. Our own seeded weights; NPUC (oracle/npuc-local.sh) compiles
each into $NCPGEN_WORK/sweep/gaplim/<name>/, regression data for tests/regress.sh (DEFECT: NPUC's broken NCPs, into
$NCPGEN_WORK/zoo-evidence/gaplim-defect/).

  gap_sweep.py list                      the variants: name, quant, C, kernel
  gap_sweep.py write <dir> [name...]     <dir>/<name>.json (npu-net/1)
  gap_sweep.py compile [--jobs N] [name...]   write + npuc-local.sh into $NCPGEN_WORK/sweep/gaplim/<name>/
"""
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("NCPGEN_WORK") or os.path.join(REPO, "work")
OUT = os.path.join(WORK, "sweep", "gaplim")

# (quant, C, kh, kw): the kernel areas around the limits weights.py's packing predicts (quant 1: 255, 256; quant 3:
# 504..512), tall and wide kernels (our harness takes kernel sides up to 64), 1 to 128 channels, and the quant-1 channel
# counts whose sub-batches exceed tryBuildData's maxPossibleBatches (10x10 over 33..40, 80, 120 channels)
GRID = [(1, 8, 15, 17), (1, 8, 16, 16), (1, 8, 20, 20), (1, 8, 63, 8), (1, 8, 64, 8), (1, 8, 30, 17), (1, 8, 64, 4),
        (1, 16, 15, 17), (1, 16, 16, 16), (1, 64, 15, 17), (1, 128, 15, 17), (1, 1, 64, 7), (1, 1, 64, 8),
        (1, 1, 30, 17), (1, 2, 20, 13), (1, 24, 14, 18), (1, 64, 25, 5), (1, 8, 1, 64), (1, 8, 64, 1),
        (1, 32, 10, 10), (1, 40, 9, 9), (1, 40, 10, 10), (1, 36, 5, 20), (1, 80, 10, 10), (1, 120, 10, 10),
        (3, 8, 25, 20), (3, 8, 63, 8), (3, 8, 8, 63), (3, 8, 64, 8), (3, 8, 30, 17), (3, 8, 32, 16), (3, 16, 16, 16),
        (3, 64, 63, 8), (3, 64, 30, 17), (3, 120, 17, 17), (3, 1, 63, 8), (3, 1, 64, 8), (3, 64, 16, 16),
        (3, 96, 25, 20), (3, 40, 10, 10)]
# NPUC's defect: a per-core chunk of 256 or more bitmap bytes passes its "(bmSize & 0x80) == 0" test and the chunk
# header (bitmap bytes << 9, a u16) cannot hold it: NPUC writes a broken NCP. Evidence, not regression data:
# compiled into $NCPGEN_WORK/zoo-evidence/gaplim-defect/
DEFECT = [(1, 64, 16, 16), (1, 32, 16, 16)]


def name_of(q, C, kh, kw):
    return "q%d-c%d-%dx%d" % (q, C, kh, kw)


VARIANTS = {name_of(*v): v for v in GRID + DEFECT}


def graph(name):
    q, C, kh, kw = VARIANTS[name]
    r = random.Random(name)
    pw = {"op": "conv2d", "name": "pw", "filters": C, "kernel": [1, 1], "strides": [1, 1], "padding": "same",
          "activation": "relu", "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1 / 64, "zero_point": 0,
                                            "data": [r.randint(-40, 40) for _ in range(C * C)]},
          "bias": [r.randint(-200, 200) for _ in range(C)], "output": {"scale": 1 / 128, "zero_point": 0}}
    gap = {"op": "conv2d", "name": "gap", "filters": C, "kernel": [kh, kw], "strides": [1, 1], "padding": "valid",
           "activation": "relu", "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1.0 / (kh * kw), "zero_point": 0,
                                             "data": [1 if c == o else 0 for o in range(C) for i in range(kh)
                                                      for j in range(kw) for c in range(C)]},
           "bias": [0] * C, "output": {"scale": 1 / 128, "zero_point": 0}}
    return {"format": "npu-net/1", "name": name, "description": "models/gap_sweep.py %s: a global average pool %dx%d over "
            "%d channels as npunet's identity conv, quant %d" % (name, kh, kw, C, q),
            "input": {"name": "x", "shape": [1, kh, kw, C], "dtype": "int8", "scale": 1 / 128, "zero_point": 0},
            "layers": [pw, gap]}


def write(d, name, base=None):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, (base or name) + ".json")
    with open(p, "w") as f:
        json.dump(graph(name), f)
    return p


def compile_one(name):
    d = os.path.join(OUT if VARIANTS[name] in GRID else os.path.join(WORK, "zoo-evidence", "gaplim-defect"), name)
    os.makedirs(d, exist_ok=True)
    src = write(d, name, "model")
    p = subprocess.run(["bash", os.path.join(REPO, "oracle", "npuc-local.sh"), src, d, "--quant", str(VARIANTS[name][0])],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return "%-18s %s" % (name, last.replace(d + "/", ""))


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("list", "write", "compile"):
        sys.exit(__doc__)
    cmd = a.pop(0)
    jobs = 4
    if "--jobs" in a:
        i = a.index("--jobs")
        jobs = int(a[i + 1])
        del a[i:i + 2]
    if cmd == "write":
        d = a.pop(0)
    names = a or list(VARIANTS)
    for n in names:
        if n not in VARIANTS:
            sys.exit("gap_sweep: unknown variant %s" % n)
    if cmd == "list":
        for n in names:
            q, C, kh, kw = VARIANTS[n]
            print("%-18s quant %d  C %-4d kernel %dx%d (area %d)" % (n, q, C, kh, kw, kh * kw))
    elif cmd == "write":
        for n in names:
            print(write(d, n))
    else:
        with ThreadPoolExecutor(jobs) as ex:
            for line in ex.map(compile_one, names):
                print(line, flush=True)


if __name__ == "__main__":
    main()
