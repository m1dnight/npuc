#!/usr/bin/env python3
"""Write a small npu-net/1 network (npu/README.md) from a compact spec: the variants the NPUC oracle compiles to map
the NCP format (dossier/npu-compiler.md section 10). Our own weights (seeded), never vendor data.

  npuc-variant.py <out.json> --in H,W,C [--seed N] [--in-scale S] [--in-zp Z]
        [--conv COUT,K[,S[,same|valid[,relu|none]]] [--kw K2] [--wscale S] [--oscale S] [--ozp Z]
                [--weights random|zero|ones|impulse:O,I,Y,X,V] [--bias random|zero|V|list:a/b/c]]...
        [--pool max|avg,K[,S[,valid|same]]]...

Layers are taken in command-line order; the options after a --conv apply to that conv (defaults: stride 1, same,
relu, weight scale 1/256, output scale = input scale, output zero point = input zero point, random int8 weights in
[-24, 24], random bias in [-2000, 2000]). --kw makes the kernel K x K2 (rectangular). Weights are OHWI int8, as
npunet.py lower expects. "impulse:O,I,Y,X,V" = all zero except weight [O][Y][X][I] = V (O output, I input channel).
"""
import json
import random
import sys


def die(msg):
    sys.exit("npuc-variant: " + msg)


def main():
    a = sys.argv[1:]
    if len(a) < 3 or a[1] != "--in":
        sys.exit(__doc__)
    out = a[0]
    try:
        H, W, C = (int(v) for v in a[2].split(","))
    except ValueError:
        die("--in H,W,C")
    rest = a[3:]
    g = {"format": "npu-net/1", "name": "variant", "input": {"name": "image", "shape": [1, H, W, C], "dtype": "int8",
                                                           "scale": 0.0078125, "zero_point": 0}, "layers": []}
    seed = 1
    layers = []  # (kind, dict of options)
    i = 0
    while i < len(rest):
        k, v = rest[i], rest[i + 1] if i + 1 < len(rest) else None
        if v is None:
            die("%s wants a value" % k)
        if k == "--seed":
            seed = int(v)
        elif k == "--in-scale":
            g["input"]["scale"] = float(v)
        elif k == "--in-zp":
            g["input"]["zero_point"] = int(v)
        elif k in ("--conv", "--pool"):
            layers.append([k[2:], v, {}])
        elif k in ("--kw", "--wscale", "--oscale", "--ozp", "--weights", "--bias"):
            if not layers or layers[-1][0] != "conv":
                die("%s must follow a --conv" % k)
            layers[-1][2][k[2:]] = v
        else:
            die("unknown option %s" % k)
        i += 2
    if not layers:
        die("no layer")
    rng = random.Random(seed)
    c, h, w = C, H, W
    scale, zp = g["input"]["scale"], g["input"]["zero_point"]
    names = {}
    for kind, spec, o in layers:
        names[kind] = names.get(kind, 0) + 1
        name = "%s%d" % (kind, names[kind])
        f = spec.split(",")
        if kind == "conv":
            cout, kh = int(f[0]), int(f[1])
            kw = int(o.get("kw", kh))
            s = int(f[2]) if len(f) > 2 else 1
            pad = f[3] if len(f) > 3 else "same"
            act = f[4] if len(f) > 4 else "relu"
            n = cout * kh * kw * c
            wm = o.get("weights", "random")
            if wm == "random":
                wd = [rng.randint(-24, 24) for _ in range(n)]
            elif wm == "zero":
                wd = [0] * n
            elif wm == "ones":
                wd = [1] * n
            elif wm.startswith("impulse:"):
                oo, ii, yy, xx, vv = (int(t) for t in wm[8:].split(","))
                wd = [0] * n
                wd[((oo * kh + yy) * kw + xx) * c + ii] = vv
            else:
                die("weights %s" % wm)
            bm = o.get("bias", "random")
            if bm == "random":
                bias = [rng.randint(-2000, 2000) for _ in range(cout)]
            elif bm == "zero":
                bias = [0] * cout
            elif bm.startswith("list:"):
                bias = [int(t) for t in bm[5:].split("/")]
            else:
                bias = [int(bm)] * cout
            osc = float(o.get("oscale", scale))
            ozp = int(o.get("ozp", zp))
            g["layers"].append({"op": "conv2d", "name": name, "filters": cout, "kernel": [kh, kw], "strides": [s, s],
                                "padding": pad, "activation": act,
                                "weights": {"dtype": "int8", "layout": "OHWI", "scale": float(o.get("wscale", 1 / 256)),
                                            "zero_point": 0, "data": wd},
                                "bias": bias, "output": {"scale": osc, "zero_point": ozp}})
            scale, zp, c = osc, ozp, cout
        else:
            typ, k = f[0], int(f[1])
            s = int(f[2]) if len(f) > 2 else k
            pad = f[3] if len(f) > 3 else "valid"
            g["layers"].append({"op": "max_pool2d" if typ == "max" else "average_pool2d", "name": name,
                                "pool": [k, k], "strides": [s, s], "padding": pad})
    g["name"] = "v-" + "-".join(x[0] + x[1].replace(",", "_") for x in layers)
    json.dump(g, open(out, "w"), indent=1)
    print("npuc-variant: %s: %s, in %dx%dx%d" % (out, g["name"], H, W, C))


if __name__ == "__main__":
    main()
