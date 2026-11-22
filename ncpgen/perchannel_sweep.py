#!/usr/bin/env python3
"""perchannel_sweep: the probes of docs/format/perchannel.md (per-channel slopes and shifts through the harness's
setScale hook, a per-channel weight scale vector) and of the weight zero points other than 128 (quant mode 1).

  perchannel_sweep.py write <dir>      <dir>/perchannel-probe/<probe>/ and <dir>/wzp-probe/<probe>/: net.npuc +
                                       lowered.json (the per-channel values or the zero point in both)
  perchannel_sweep.py compile <dir> [--jobs N]   NPUC on each (oracle/npuc-local.sh; a harness with the setScale hook:
                                       NPU_ROOT=$NCPGEN_WORK/n3/root-40bb3e3f or later)
then: python3 -m ncpgen check <dir>/perchannel-probe/*/ <dir>/wzp-probe/*/   (34 exact)
Our own seeded weights (oracle/npuc-variant.py). <dir> holds vendor-generated NCPs after compile: the work folder.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ncpgen import npunet  # noqa: E402

ORACLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle")

# base convs: name -> npuc-variant.py arguments, npunet options
BASES = {"c16": ("--in 16,16,8 --conv 16,3 --seed 7", {}),
         "c40": ("--in 16,16,8 --conv 40,3 --seed 8", {}),
         "c40q3": ("--in 16,16,8 --conv 40,3 --seed 8", {"quant": 3}),
         "c5n": ("--in 12,12,6 --conv 5,3,1,same,none --seed 9", {"allow_none": True}),
         "c24s2": ("--in 32,32,16 --conv 24,3,2 --seed 10", {})}
SL = [128 + 3 * c for c in range(40)]
SH = [8 + (5 * c) % 8 for c in range(40)]
# probe -> (base, kind, values): "slopes" (slopes, shifts), "wscales" (scales), "wzp" zero point, None (the base)
PROBES = {"c16": ("c16", None, None),
          "c16-ws": ("c16", "wscales", [2 ** -8 * 2 ** (-c / 4) for c in range(16)]),
          "c16-same": ("c16", "slopes", ([128] * 16, [15] * 16)),
          "c16-sl": ("c16", "slopes", ([128, 129, 130, 131, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
                                       [15] * 16)),
          "c16-sh": ("c16", "slopes", ([128] * 16, [8 + c % 8 for c in range(16)])),
          "c40": ("c40", None, None), "c40-pc": ("c40", "slopes", (SL, SH)),
          "c40-sh": ("c40", "slopes", ([128] * 40, SH)),
          "c40q3": ("c40q3", None, None), "c40q3-pc": ("c40q3", "slopes", (SL, SH)),
          "c5n": ("c5n", None, None), "c5n-pc": ("c5n", "slopes", ([140, 150, 160, 170, 180], [9, 10, 11, 12, 13])),
          "c24s2": ("c24s2", None, None), "c24s2-pc": ("c24s2", "slopes", (SL[:24], SH[:24]))}
WZP = [(b, z) for b in ("c16", "c40", "c24s2", "c5n") for z in (163, 132, 100, 0, 255)]


def _hex(f):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", f))[0]


def base_net(tmp, name):
    args, opt = BASES[name]
    js = os.path.join(tmp, name + ".json")
    subprocess.run([sys.executable, "-I", os.path.join(ORACLE, "npuc-variant.py"), js] + args.split(),
                   check=True, capture_output=True)
    low = npunet.load_model(js, opt.get("quant", 1), False, opt.get("allow_none", False))
    return low


def write(d):
    tmp = os.path.join(d, ".perchannel-src")
    os.makedirs(tmp, exist_ok=True)
    bases = {n: base_net(tmp, n) for n in BASES}
    for p, (b, kind, vals) in PROBES.items():
        low = json.loads(json.dumps(bases[b]))
        L = low["layers"][0]
        txt = npunet.npuc_text(low).rstrip("\n")
        assert txt.endswith("end")
        txt = txt[:-3]
        if kind == "slopes":
            L["slopes"], L["shifts"] = list(vals[0]), list(vals[1])
            txt = npunet.npuc_text(low).rstrip("\n")[:-3]        # npuc_text writes the slopes line itself
        elif kind == "wscales":
            txt += "wscales %s %d %s\n" % (L["name"], len(vals), " ".join(_hex(v) for v in vals))
        out = os.path.join(d, "perchannel-probe", p)
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "net.npuc"), "w").write(txt + "end\n")
        json.dump(low, open(os.path.join(out, "lowered.json"), "w"), indent=1)   # wscales: NPUC keeps wScale[0]
    for b, z in WZP:
        low = json.loads(json.dumps(bases[b]))
        low["layers"][0]["w_zp"] = z
        out = os.path.join(d, "wzp-probe", "%s-wzp%d" % (b, z))
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "net.npuc"), "w").write(npunet.npuc_text(low))
        json.dump(low, open(os.path.join(out, "lowered.json"), "w"), indent=1)
    print("perchannel_sweep: %d per-channel probes, %d zero-point probes in %s" % (len(PROBES), len(WZP), d))


def compile_all(d, jobs):
    dirs = [os.path.join(d, s, p) for s in ("perchannel-probe", "wzp-probe")
            for p in sorted(os.listdir(os.path.join(d, s)))]

    def one(v):
        r = subprocess.run([os.path.join(ORACLE, "npuc-local.sh"), os.path.join(v, "net.npuc"), v],
                           capture_output=True, text=True)
        return v, (r.stdout.strip().splitlines() or ["?"])[-1]
    with ThreadPoolExecutor(jobs) as ex:
        for v, line in ex.map(one, dirs):
            print("%-40s %s" % (os.path.relpath(v, d), line[:60]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write").add_argument("dir")
    p = sub.add_parser("compile")
    p.add_argument("dir")
    p.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.dir)
    else:
        compile_all(a.dir, a.jobs)


if __name__ == "__main__":
    main()
