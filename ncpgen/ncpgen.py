#!/usr/bin/env python3
"""ncpgen: an open NCP generator for the Exynos 990 NPU (docs/format/). It writes the NCP that Samsung's NPUC 1.36.1.a
writes for the same lowered network, part by part, checked byte for byte against NPUC run as an oracle
(oracle/npuc-local.sh).

  ncpgen.py gen <net.json|model.tflite|lowered.json> <out.ncp> [--quant 1|3] [--allow-none]
                [--grelu fixed|robust|npuc] [--fix-npuc]
        the NCP for a model (npunet.py lowers it first unless it is already lowered). The activation table: fixed
        (default: the corrected one, = oracle/ncp-grelu-patch.py auto, the tables that ran bit-exact on the NPU),
        robust (group 0 copied to the four groups), npuc (NPUC's own bytes, with its defect: docs/format/grelu.md).
        --fix-npuc: consistent DMA strides where NPUC's are suspect (outputs of 3/5/7 channels, a conv after a widened
        pool: docs/format/frame.md section 9; G until the NPU runs them); otherwise NPUC's bytes.
  ncpgen.py check [--grelu npuc|fixed|robust] <variant dir>...
        each dir as written by npuc-local.sh (lowered.json + NPUC's net.ncp): every part generated from lowered.json
        and compared with NPUC's: streams, intrinsics, weight blocks, then the whole file (with the parts the generator
        refuses taken from NPUC's file, so that each part is judged alone). Default: NPUC's GReLU bytes; one line per
        dir and a total.
  ncpgen.py corpus <dir> [--jobs N]
        compile the regression corpus (CORPUS below) with NPUC into <dir>/<name>/ (npuc-local.sh). Vendor-generated
        bytes: <dir> must be in the work folder or a scratch folder.

The parts (modules of this package):
  stream.py   group_streams(low, fixed) -> [isa bytes per group]     (the CMDQ instruction streams)
  grelu.py    grelu_regs(layer, quant_mode, fixed) -> [(addr, value)] (the activation table, used by stream.py)
  weights.py  weight_blocks(low) -> [WEIGHT block bytes]
  frame.py    intrinsics(low) -> [bytes per group]; frame(low, streams, intrinsics, weights) -> the NCP
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

if not __package__:     # run as a script: the package is this file's directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncpgen import frame, npunet, stream, weights  # noqa: E402
from ncpgen.common import first_diff, ncp_parts  # noqa: E402

ORACLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle")

# The regression corpus: (name, npuc-variant.py arguments, npuc-local.sh options). Family F1 (one conv, optionally a
# pool) first, then chains. Our own seeded weights.
CORPUS = ([("f1-k%d" % k, "--in 16,16,8 --conv 8,%d" % k, "") for k in (1, 3, 5, 7)]
          + [("f1-cout%d" % co, "--in 16,16,8 --conv %d,3" % co, "") for co in (1, 2, 4, 16, 32)]
          + [("f1-cin%d" % ci, "--in 16,16,%d --conv 8,3" % ci, "") for ci in (1, 3, 4, 16, 32)]
          + [("f1-hw%dx%d" % (h, w), "--in %d,%d,8 --conv 8,3" % (h, w), "")
             for h, w in ((8, 8), (12, 12), (16, 24), (20, 20), (32, 32), (7, 9), (5, 13))]
          + [("f1-s2", "--in 16,16,8 --conv 8,3,2", ""),
             ("f1-valid", "--in 16,16,8 --conv 8,3,1,valid", ""),
             ("f1-s2valid", "--in 16,16,8 --conv 8,3,2,valid", ""),
             ("f1-k1x3", "--in 16,16,8 --conv 8,1 --kw 3", ""),
             ("f1-k3x1", "--in 16,16,8 --conv 8,3 --kw 1", ""),
             ("f1-none", "--in 16,16,8 --conv 8,3,1,same,none", "--allow-none"),
             ("f1-q3", "--in 16,16,8 --conv 8,3", "--quant 3"),
             ("f1-q3-none", "--in 16,16,8 --conv 8,3,1,same,none", "--quant 3 --allow-none"),
             ("f1-oscale", "--in 16,16,8 --conv 8,3 --oscale 0.01 --ozp 5", ""),
             ("f1-zero-w", "--in 16,16,8 --conv 8,3 --weights zero", ""),
             ("f1-seed7", "--in 16,16,8 --conv 8,3 --seed 7", ""),
             ("f1-maxpool", "--in 16,16,8 --conv 8,3 --pool max,2", ""),
             ("f1-avgpool", "--in 16,16,8 --conv 8,3 --pool avg,2", ""),
             ("f1-q3-maxpool", "--in 16,16,8 --conv 8,3 --pool max,2", "--quant 3"),
             ("f2-convconv", "--in 16,16,8 --conv 8,3 --conv 8,3", ""),
             ("f2-tiny", "--in 16,16,8 --conv 8,3 --pool max,2 --conv 8,3 --oscale 0.001953125", ""),
             ("f2-convconv-c16", "--in 16,16,8 --conv 16,3 --conv 8,1", ""),
             ("f3-hw64", "--in 64,64,8 --conv 8,3", ""),
             ("f3-c64", "--in 16,16,64 --conv 64,3", "")])

# --grelu: grelu.grelu_regs' `fixed` argument
GRELU = {"fixed": True, "robust": "robust", "npuc": False}


def lower_model(path, quant=1, allow_none=False):
    """lowered.json's dict for a model file: a lowered.json as it is, else npunet's lowering of the JSON or TFLite"""
    d = json.load(open(path)) if path.endswith(".json") else None
    if d is not None and d.get("format") == "npu-lowered/1":
        return d
    return npunet.load_model(path, quant, False, allow_none)


def generate(low, grelu="fixed", ref_parts=None, fix_npuc=False):
    """(parts, whole, notes): the NCP's parts from the part modules, the whole NCP (None when a part is missing) and a
    note per refusal. A part refused for this network is taken from ref_parts when given (check mode: NPUC's own
    parts, ncp_parts()), else the whole file is not made. grelu: "fixed" (the corrected table, = what
    ncp-grelu-patch.py auto writes: the NCPs that ran bit-exact), "robust" (group 0 copied to all four groups) or
    "npuc" (NPUC's own bytes, defect included)."""
    fixed = GRELU[grelu]
    parts, notes = {}, []
    for key, make in (("streams", lambda: stream.group_streams(low, fixed=fixed)),
                      ("intrinsics", lambda: frame.intrinsics(low)),
                      ("weights", lambda: weights.weight_blocks(low))):
        try:
            parts[key] = make()
        except Exception as e:  # a part that cannot handle this network says so
            parts[key] = None
            notes.append("%s: %s: %s" % (key, type(e).__name__, e))
    use = {k: (v if v is not None else (ref_parts or {}).get(k)) for k, v in parts.items()}
    whole = None
    if all(v is not None for v in use.values()):
        try:
            whole = frame.frame(low, use["streams"], use["intrinsics"], use["weights"], fix_npuc=fix_npuc)
        except Exception as e:
            notes.append("frame: %s: %s" % (type(e).__name__, e))
    return parts, whole, notes


def _cmp_parts(mine, ref, unit):
    """(summary, equal) of a generated part list against NPUC's"""
    if mine is None:
        return "-", False
    if len(mine) != len(ref):
        return "count %d/%d" % (len(mine), len(ref)), False
    bad = []
    for i, (m, r) in enumerate(zip(mine, ref)):
        o = first_diff(m, r)
        if o is not None:
            bad.append("%s%d@%#x%s" % (unit, i, o, "" if len(m) == len(r) else "(len %d/%d)" % (len(m), len(r))))
    return ("ok" if not bad else " ".join(bad[:3])), not bad


def cmd_check(dirs, grelu="npuc"):
    """compare every part, then the whole file, with NPUC's compile in each variant dir; 0 when all are exact"""
    tot = {"streams": 0, "intrinsics": 0, "weights": 0, "file": 0, "pure": 0}
    n = 0
    for d in dirs:
        lp, np_ = os.path.join(d, "lowered.json"), os.path.join(d, "net.ncp")
        if not (os.path.isfile(lp) and os.path.isfile(np_)):
            print("%-24s skipped (no lowered.json/net.ncp)" % os.path.basename(d.rstrip("/")))
            continue
        n += 1
        low = json.load(open(lp))
        ref = open(np_, "rb").read()
        rp = ncp_parts(ref)
        parts, whole, notes = generate(low, grelu=grelu, ref_parts=rp)
        res = []
        pure = True
        for key, unit in (("streams", "g"), ("intrinsics", "g"), ("weights", "w")):
            s, ok = _cmp_parts(parts[key], rp[key], unit)
            tot[key] += ok
            pure &= ok
            res.append("%s %s" % (key[:5], s))
        if whole is None:
            res.append("file -")
            pure = False
        else:
            o = first_diff(whole, ref)
            tot["file"] += o is None
            pure &= o is None
            res.append("file %s" % ("ok" if o is None else "@%#x (len %d/%d)" % (o, len(whole), len(ref))))
        tot["pure"] += pure
        print("%-24s %s%s" % (os.path.basename(d.rstrip("/")), " | ".join(res), "  EXACT" if pure else ""))
        for x in notes:
            print("    note: " + x)
    print("ncpgen check: %d variants: streams %d, intrinsics %d, weights %d, whole file %d (with NPUC's parts where "
          "ours are missing), generated entirely by us and exact: %d" % (n, tot["streams"], tot["intrinsics"],
                                                                        tot["weights"], tot["file"], tot["pure"]))
    return 0 if n and tot["pure"] == n else 2


def cmd_gen(src, out, quant, allow_none, grelu, fix_npuc=False):
    """write the NCP of a model; 0, or 2 when a part refuses it"""
    low = lower_model(src, quant, allow_none)
    parts, whole, notes = generate(low, grelu=grelu, fix_npuc=fix_npuc)
    for x in notes:
        print("ncpgen: " + x)
    if whole is None:
        missing = [k for k, v in parts.items() if v is None]
        print("ncpgen: FAIL: cannot generate this network yet (missing: %s)" % ", ".join(missing))
        return 2
    open(out, "wb").write(whole)
    print("ncpgen: PASS %s %d B sha256 %s (GReLU table: %s)" % (out, len(whole), hashlib.sha256(whole).hexdigest(),
                                                             grelu + (", strides fixed" if fix_npuc else "")))
    return 0


def cmd_corpus(d, jobs):
    """compile CORPUS with NPUC into d/<name>/, jobs at a time"""
    os.makedirs(d, exist_ok=True)
    procs, fails = [], []
    todo = list(CORPUS)

    def start(name, spec, opts):
        vj = os.path.join(d, name + ".json")
        subprocess.run([sys.executable, "-I", os.path.join(ORACLE, "npuc-variant.py"), vj] + spec.split(), check=True,
                       stdout=subprocess.DEVNULL)
        return subprocess.Popen(["bash", os.path.join(ORACLE, "npuc-local.sh"), vj, os.path.join(d, name)]
                                + opts.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while todo or procs:
        while todo and len(procs) < jobs:
            name, spec, opts = todo.pop(0)
            procs.append((name, start(name, spec, opts)))
        name, p = procs.pop(0)
        out = p.communicate()[0]
        last = out.strip().splitlines()[-1] if out.strip() else ""
        print("%-24s %s" % (name, last.replace(os.path.abspath(os.path.join(d, name)) + "/", "")))
        if p.returncode:
            fails.append(name)
    print("ncpgen corpus: %d variants, %d failed%s" % (len(CORPUS), len(fails),
                                                     (": " + " ".join(fails)) if fails else ""))
    return 0


def main(argv=None):
    """the command line (module docstring)"""
    ap = argparse.ArgumentParser(prog="ncpgen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gen", help="the NCP of a model")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--quant", type=int, default=1)
    p.add_argument("--allow-none", action="store_true")
    p.add_argument("--grelu", choices=GRELU, default="fixed")
    p.add_argument("--fix-npuc", action="store_true")
    p = sub.add_parser("check", help="every part against NPUC's compiles")
    p.add_argument("dirs", nargs="*")
    p.add_argument("--grelu", choices=GRELU, default="npuc")   # NPUC's own bytes are compared: its own table
    p = sub.add_parser("corpus", help="compile the regression corpus with NPUC")
    p.add_argument("dir")
    p.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        return cmd_gen(a.model, a.out, a.quant, a.allow_none, a.grelu, a.fix_npuc)
    if a.cmd == "check":
        return cmd_check(a.dirs, grelu=a.grelu)
    return cmd_corpus(a.dir, a.jobs)


if __name__ == "__main__":
    from ncpgen import ncpgen as _pkg    # the package's copy of this module (one copy of every module)
    sys.exit(_pkg.main())
