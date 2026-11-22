#!/usr/bin/env python3
"""Proof that a change to ncpgen's code changes none of its outputs: a snapshot of everything the code makes of the
compiled corpus in the work folder ($NCPGEN_WORK/sweep), recorded once with a trusted version and compared after.

  tests/snapshot.py record <snapshot.json.gz> [--jobs N] [--only KIND,...] [--sample K] [--code <repo>]
  tests/snapshot.py compare <snapshot.json.gz> [--jobs N] [--only KIND,...] [--sample K] [--code <repo>]

Kinds (all by default):
  gen    every variant directory with a lowered.json: the generated parts and whole NCP (sha256) with NPUC's and the
         fixed activation table, and the fixed one with --fix-npuc, or the exact refusal messages (in process, the
         generate() API of ncpgen/ncpgen.py; the robust table is grelu.py check's)
  check  every part checker and the generator's check on every compiled set, in chunks (the command lines)
  lower  npunet.py lower of every npu-net/1 JSON and TFLite file in the sweeps, with three option sets
  ref    npunet.py ref, layout, to-npu, from-npu and compare on the small networks
  misc   selftests, examples, listings (stream.py show, frame.py plan, weights.py show, grelu.py regs/table,
         frame_sweep.py list/policy, stream_graphs.py write) and the gen command line on a sample
The snapshot holds hashes, our own messages and our own listings; it belongs in the work folder (it names vendor
compiles). --code runs another checkout (default: this one); --sample K takes every K-th task only (a quick check;
compare then reads only those entries). Exit 0 when compare finds no difference.
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("NCPGEN_WORK") or os.path.join(REPO, "work")
SWEEP = os.path.join(WORK, "sweep")
KINDS = ("gen", "check", "lower", "ref", "misc")
CODE = REPO
GEN = None


def h(b):
    return hashlib.sha256(b).hexdigest()[:20] if b is not None else None


def variant_dirs():
    """every directory under the sweeps holding a lowered.json, relative, sorted"""
    out = []
    for root, dirs, files in os.walk(SWEEP):
        dirs.sort()
        if "lowered.json" in files:
            out.append(os.path.relpath(root, SWEEP))
    return sorted(out)


def sets(vds):
    """the compiled sets: (parent, [variant dirs]) for every parent holding variant dirs"""
    by = {}
    for v in vds:
        p = os.path.dirname(v)
        if p:
            by.setdefault(p, []).append(v)
    return sorted(by.items())


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def macs(low):
    n = 0
    for L in low["layers"]:
        if L["kind"] == "conv":
            co, ho, wo = L["out_chw"]
            n += co * ho * wo * L["in_chw"][0] * L["k"][0] * L["k"][1]
        else:
            n += 4 * L["out_chw"][0] * L["out_chw"][1] * L["out_chw"][2]
    return n


def det_bytes(n, seed):
    out, i = bytearray(), 0
    while len(out) < n:
        out += hashlib.sha256(b"%s:%d" % (seed.encode(), i)).digest()
        i += 1
    return bytes(out[:n])


# ---- running the code under test ----------------------------------------------------------------------------------
def init(code):
    global CODE, GEN
    CODE = code
    if os.path.isfile(os.path.join(code, "ncpgen", "__init__.py")):
        sys.path.insert(0, code)
        from ncpgen import ncpgen as gen  # noqa: the package's generator module
    else:
        sys.path.insert(0, os.path.join(code, "ncpgen"))
        import ncpgen as gen  # noqa: the flat layout: ncpgen/ncpgen.py
    GEN = gen


def run(tool, args, tmp=None):
    """a command line of the code under test: exit code, stdout and the last stderr line, paths neutralised"""
    p = subprocess.run([sys.executable, "-I", os.path.join(CODE, "ncpgen", tool)] + args, capture_output=True,
                       text=True, errors="replace", cwd=tmp or CODE, env=dict(os.environ, NCPGEN_WORK=WORK))

    def norm(s):
        # the work folder first: a baseline recorded from another checkout (--code) must compare equal
        s = s.replace(os.path.realpath(WORK), "<work>").replace(WORK, "<work>").replace(CODE, "<code>")
        return s.replace(tmp, "<tmp>") if tmp else s
    err = p.stderr.strip().splitlines()
    last = re.sub(r"^[\w.]+\.(\w+): ", r"\1: ", norm(err[-1])) if err else ""   # npunet.Refused = ncpgen.npunet.Refused
    return {"rc": p.returncode, "out": norm(p.stdout), "err": last}


def files_of(d):
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = h(open(p, "rb").read())
    return dict(sorted(out.items()))


def summary(parts, whole, notes):
    return {"s": None if parts["streams"] is None else [h(x) for x in parts["streams"]],
            "i": None if parts["intrinsics"] is None else [h(x) for x in parts["intrinsics"]],
            "w": None if parts["weights"] is None else [h(x) for x in parts["weights"]],
            "f": h(whole), "n": notes}


def task_gen(v):
    text = open(os.path.join(SWEEP, v, "lowered.json")).read()
    out = {}
    for g, fix in (("npuc", False), ("fixed", False), ("fixed", True)):
        out[g + ("+fix" if fix else "")] = summary(*GEN.generate(json.loads(text), grelu=g, fix_npuc=fix))
    return out


def task_cmd(tool, args):
    return run(tool, args)


def task_lower(src):
    out = {}
    for opts in ([], ["--allow-none"], ["--quant", "3", "--allow-none"]):
        tmp = tempfile.mkdtemp()
        try:
            r = run("npunet.py", ["lower", os.path.join(SWEEP, src), os.path.join(tmp, "o")] + opts, tmp)
            r["files"] = files_of(os.path.join(tmp, "o")) if os.path.isdir(os.path.join(tmp, "o")) else {}
            out[" ".join(opts) or "-"] = r
        finally:
            shutil.rmtree(tmp)
    return out


def task_ref(v):
    d = os.path.join(SWEEP, v)
    lp = os.path.join(d, "lowered.json")
    low = json.load(open(lp))
    tmp = tempfile.mkdtemp()
    try:
        n = 1
        for x in low["input"]["nhwc"]:
            n *= x
        inp = os.path.join(tmp, "in.bin")
        open(inp, "wb").write(det_bytes(n, v))
        out = {"ref": run("npunet.py", ["ref", lp, inp, os.path.join(tmp, "out.bin")], tmp)}
        ncp = os.path.join(d, "net.ncp")
        if os.path.isfile(ncp):
            lay = os.path.join(tmp, "lay.json")
            out["layout"] = run("npunet.py", ["layout", lp, ncp, lay], tmp)
            if os.path.isfile(lay):
                out["layout"]["file"] = h(open(lay, "rb").read())
                co, ho, wo = low["layers"][-1]["out_chw"]
                ofm = os.path.join(tmp, "ofm.bin")
                open(ofm, "wb").write(det_bytes(co * ho * wo, v + "ofm"))
                out["to"] = run("npunet.py", ["to-npu", lp, lay, inp, os.path.join(tmp, "npu-in.bin")], tmp)
                out["from"] = run("npunet.py", ["from-npu", lp, lay, ofm, os.path.join(tmp, "o.bin")], tmp)
                out["cmp"] = run("npunet.py", ["compare", lp, lay, inp + ":" + ofm], tmp)
        return out
    finally:
        shutil.rmtree(tmp)


def task_dir(tool, args):
    """a command writing into a fresh directory: its output and the files it wrote"""
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "d")
        r = run(tool, [a.replace("@D", d) for a in args], tmp)
        r["files"] = files_of(d) if os.path.isdir(d) else {}
        return r
    finally:
        shutil.rmtree(tmp)


def task_gencli(v, opts):
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(SWEEP, v)
        r = run("ncpgen.py", ["gen", src, os.path.join(tmp, "o.ncp")] + opts, tmp)
        r["file"] = h(open(os.path.join(tmp, "o.ncp"), "rb").read()) if os.path.exists(os.path.join(tmp, "o.ncp")) \
            else None
        return r
    finally:
        shutil.rmtree(tmp)


def changed(rel, since):
    """a variant dir or source gone, or a file of it modified after `since` (another sweep writing there)"""
    p = os.path.join(SWEEP, rel)
    if not os.path.exists(p):
        return True
    fs = [os.path.join(p, f) for f in os.listdir(p)] if os.path.isdir(p) else [p]
    return any(os.path.getmtime(f) > since for f in fs if os.path.isfile(f))


def touched(t):
    """the variant dirs and sources (relative) a task reads"""
    out = set()
    for a in t[2]:
        for x in (a if isinstance(a, list) else [a]):
            if isinstance(x, str):
                r = os.path.relpath(x, SWEEP) if os.path.isabs(x) else x
                out.add(r)
                out.add(os.path.dirname(r))
    return out


def cost(t):
    """a rough cost (bytes of lowered.json behind the task: the weights dominate) to start the heavy tasks first"""
    paths = [a for a in t[2] if isinstance(a, str)] + [x for a in t[2] if isinstance(a, list) for x in a]
    n = 0
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(SWEEP, p)
        q = os.path.join(p, "lowered.json") if os.path.isdir(p) else p
        n += os.path.getsize(q) if os.path.isfile(q) else 0
    return n


def call(t):
    key, fn, args = t
    try:
        return key, globals()[fn](*args)
    except Exception as e:  # the snapshot itself must not die on one task
        return key, {"snapshot-error": "%s: %s" % (type(e).__name__, e)}


# ---- the tasks ----------------------------------------------------------------------------------------------------
def sources():
    """every network source in the sweeps: TFLite files and npu-net/1 JSONs, relative, sorted"""
    srcs = []
    for root, dirs, files in os.walk(SWEEP):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.endswith(".tflite") or (f.endswith(".json") and f not in ("lowered.json", "net.json")
                                         and '"npu-net/1"' in open(p, errors="replace").read(400)):
                srcs.append(os.path.relpath(p, SWEEP))
    return srcs


def tasks(kinds, vds, srcs):
    """(key, worker function, arguments) of every entry, over the variant dirs vds and the sources srcs"""
    S = lambda v: os.path.join(SWEEP, v)
    out = []
    if "gen" in kinds:
        out += [("gen:" + v, "task_gen", (v,)) for v in vds]
    if "check" in kinds:
        for s, vs in sets(vds):
            for i, ch in enumerate(chunks(vs, 25)):
                dirs = [S(v) for v in ch]
                k = "%s#%d" % (s, i)
                out += [("check:ncpgen:" + k, "task_cmd", ("ncpgen.py", ["check"] + dirs)),
                        ("check:stream:" + k, "task_cmd", ("stream.py", ["check"] + dirs)),
                        ("check:events:" + k, "task_cmd", ("stream.py", ["events"] + dirs)),
                        ("check:frame:" + k, "task_cmd", ("frame.py", ["check"] + dirs)),
                        ("check:weights:" + k, "task_cmd", ("weights.py", ["check"] + dirs)),
                        ("check:weights-loose:" + k, "task_cmd", ("weights.py", ["check", "--loose"] + dirs)),
                        ("check:grelu:" + k, "task_cmd", ("grelu.py", ["check"] + dirs))]
                if s.startswith("frame/"):
                    out.append(("check:policy:" + k, "task_cmd", ("frame_sweep.py", ["policy"] + dirs)))
                if i == 0:
                    out.append(("check:ncpgen-fixed:" + k, "task_cmd", ("ncpgen.py", ["check", "--grelu", "fixed"]
                                                                        + dirs)))
    if "lower" in kinds:
        out += [("lower:" + s, "task_lower", (s,)) for s in srcs]
    if "ref" in kinds:
        for v in vds:
            try:
                small = macs(json.load(open(os.path.join(S(v), "lowered.json")))) <= 1500000
            except (OSError, ValueError, KeyError):
                small = False
            if small:
                out.append(("ref:" + v, "task_ref", (v,)))
    if "misc" in kinds:
        out += [("misc:selftest:" + t, "task_cmd", (t, ["selftest"])) for t in ("npunet.py", "frame.py", "grelu.py")]
        out += [("misc:example", "task_dir", ("npunet.py", ["example", "@D"])),
                ("misc:example-residual", "task_dir", ("npunet.py", ["example-residual", "@D"])),
                ("misc:variants", "task_dir", ("npunet.py", ["variants", "@D"])),
                ("misc:stream-graphs", "task_dir", ("stream_graphs.py", ["write", "@D"])),
                ("misc:frame-sweep-list", "task_cmd", ("frame_sweep.py", ["list"]))]
        for args in (["8", "16", "16"], ["3", "7", "9", "2", "5", "6"], ["33", "40", "24", "31", "39", "23"]):
            out.append(("misc:layout:" + ",".join(args), "task_cmd", ("frame.py", ["layout"] + args)))
        sample = vds[::23]
        for v in sample:
            lp = os.path.join(S(v), "lowered.json")
            out += [("misc:show:" + v, "task_cmd", ("stream.py", ["show", S(v), "--diff"])),
                    ("misc:plan:" + v, "task_cmd", ("frame.py", ["plan", S(v)])),
                    ("misc:wshow:" + v, "task_cmd", ("weights.py", ["show", lp])),
                    ("misc:regs:" + v, "task_cmd", ("grelu.py", ["regs", lp])),
                    ("misc:regs-fixed:" + v, "task_cmd", ("grelu.py", ["regs", lp, "--fixed"]))]
            if os.path.isfile(os.path.join(S(v), "net.ncp")):
                out.append(("misc:table:" + v, "task_cmd", ("grelu.py", ["table", os.path.join(S(v), "net.ncp")])))
        for v in vds[::97]:
            out.append(("misc:show-plain:" + v, "task_cmd", ("stream.py", ["show", S(v)])))
            for opts in ([], ["--grelu", "npuc"], ["--grelu", "robust"], ["--fix-npuc"]):
                out.append(("misc:gen:%s:%s" % (v, " ".join(opts)), "task_gencli",
                            (os.path.join(v, "lowered.json"), opts)))
        for src in ("graph/v/res/residual.tflite", "weights/res/ex/residual.tflite"):
            if os.path.exists(S(src)):
                for opts in ([], ["--quant", "3", "--allow-none"]):
                    out.append(("misc:gen:%s:%s" % (src, " ".join(opts)), "task_gencli", (src, opts)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("record", "compare"))
    ap.add_argument("file")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default=",".join(KINDS))
    ap.add_argument("--code", default=REPO)
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--since", type=float, help="compare: skip dirs changed after this time (default: the baseline's)")
    a = ap.parse_args()
    kinds = set(a.only.split(","))
    code = os.path.abspath(a.code)
    t0 = time.time()
    ref, skip = None, set()
    if a.mode == "compare":
        # the work folder is shared and grows: the baseline's own lists of dirs and sources, minus what changed since
        ref = json.load(gzip.open(a.file, "rt"))
        if "meta:dirs" in ref:
            vds, srcs = ref["meta:dirs"], ref["meta:sources"]
        elif a.sample == 1:     # an older snapshot: its complete entries list them
            vds = sorted(k[4:] for k in ref if k.startswith("gen:"))
            srcs = sorted(k[6:] for k in ref if k.startswith("lower:"))
        else:
            vds, srcs = variant_dirs(), sources()
        since = a.since or ref.get("meta:t0") or os.path.getmtime(a.file)
        skip = {x for x in vds + srcs if changed(x, since)}
    else:
        vds, srcs = variant_dirs(), sources()
    todo = tasks(kinds, vds, srcs)[::a.sample]
    dropped = [t[0] for t in todo if touched(t) & skip]
    todo = [t for t in todo if not touched(t) & skip]
    todo.sort(key=lambda t: (-cost(t), t[0]))
    res = {}
    with multiprocessing.Pool(a.jobs, initializer=init, initargs=(code,)) as pool:
        for i, (k, v) in enumerate(pool.imap_unordered(call, todo, chunksize=1)):
            res[k] = v
            if (i + 1) % 500 == 0:
                print("snapshot: %d of %d" % (i + 1, len(todo)), file=sys.stderr, flush=True)
    if a.mode == "record":
        old = {}
        if os.path.exists(a.file):
            old = json.load(gzip.open(a.file, "rt"))
            old = {k: v for k, v in old.items() if k.split(":", 1)[0] not in kinds}
        old.update(res)
        old.update({"meta:dirs": vds, "meta:sources": srcs, "meta:t0": t0})
        with gzip.open(a.file, "wt") as f:
            json.dump(dict(sorted(old.items())), f, sort_keys=True)
        errs = [k for k, v in res.items() if isinstance(v, dict) and "snapshot-error" in v]
        print("snapshot: recorded %d entries (%s) in %s%s" % (len(res), ",".join(sorted(kinds)), a.file,
                                                            "; snapshot errors: %s" % errs[:5] if errs else ""))
        return 0
    ref = {k: v for k, v in ref.items() if k.split(":", 1)[0] in kinds and (a.sample == 1 or k in res)
           and k not in dropped and not k.startswith("meta:") and not any(":" + x in k for x in skip)}
    diff = sorted(k for k in set(ref) | set(res) if ref.get(k) != res.get(k))
    for k in diff[:40]:
        print("DIFF %s\n  was %s\n  now %s" % (k, json.dumps(ref.get(k))[:600], json.dumps(res.get(k))[:600]))
    print("snapshot: %d entries compared (%s)%s: %s" % (
        len(res), ",".join(sorted(kinds)), ", %d skipped (their dirs changed since the baseline)" % len(dropped)
        if dropped else "", "IDENTICAL" if not diff else "%d DIFFER" % len(diff)))
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
