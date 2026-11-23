# ncpgen: instructions for agents

An open compiler for the Exynos 990 NPU (README.md): a network in, the NCP the NPU firmware runs out. Developed against
Samsung's own compiler, NPUC 1.36.1.a, used as a byte-exact **oracle** on this host. There is no phone here: the work is
the compiler, and it is tested on the host against NPUC.

## Rules

1. **No vendor byte in git.** NPUC, the stock Bionic runtime, compiled NCPs, NPUC's logs and IR dumps, and Ghidra's
   output live only in the work folder (`work/`, git-ignored; `oracle/env.sh`). Committed code holds formulas, register
   addresses, field layouts and observed constants, never a copied table, blob or decompiled code. Hashes of NCPs are
   fine (`tests/golden.txt`). Check `git status` before every commit.
2. **Measure, don't guess.** Every rule in `ncpgen/` comes from the oracle: compile variants (`oracle/npuc-variant.py`,
   `oracle/npuc-local.sh`), diff them (`oracle/ncp-dump.py --log`), read NPUC's code when diffs are not enough
   (`oracle/ghidra/run-ghidra.sh`). Notes in `docs/format/` tag facts **V** (seen in bytes, logs or code) or **G** (guess).
3. **Refuse rather than guess.** A network the generator cannot model raises (Unsupported / NotImplementedError / Refused);
   it must never emit bytes it has not matched against NPUC.
4. **Test before every commit.** `tests/run.sh` (no vendor file; must PASS) for any change. For a change to a part
   (`stream.py`, `frame.py`, `weights.py`, `grelu.py`, `npunet.py`): also `tests/regress.sh`, which checks every compiled
   set in `work/sweep` (only the known misses allowed: graph 93/98, graph-random2 64/66, NPUC's own uninitialised
   bytes; zoo 1 (efficientdet-lite0-320-p4) and graph-camera 1 (v3003) of the camera-resolution work: models/README.md
   "Camera resolutions", frame.md 6.11).
   A change meant to change no output (a refactor, a speed-up): `tests/snapshot.py compare` against a snapshot the code
   before it recorded must say IDENTICAL. Say the numbers in the commit message.
5. **New networks become regression data.** A new sweep goes to `work/sweep/<name>/` (one folder per variant, as
   npuc-local.sh writes them) with its spec list committed next to the part (`ncpgen/stream_variants_*.txt`,
   `frame_sweep.py` families); networks NPUC compiled may extend `tests/golden.txt` (`tests/make-golden.sh`).
6. **Unseen data is the real test.** After fitting a rule, compile a fresh random set nobody has looked at and report the
   whole-file rate on it, misses included.

## Commands

| Task | Command |
|---|---|
| Tests (no vendor file) | `tests/run.sh -j 6` |
| Regression against NPUC | `tests/regress.sh [set...]` |
| Compile with NPUC | `oracle/npuc-local.sh net.json out/ [--quant 1\|3] [--allow-none]` (1.3 s; `out/log.txt`, `out/run/ir_graph_*`) |
| A variant | `oracle/npuc-variant.py out.json --in H,W,C --conv COUT,K[,S[,same\|valid[,relu\|none]]] --pool max\|avg,K[,S] --seed N` |
| Generate | `python3 -m ncpgen gen model out.ncp [--grelu npuc\|fixed]` (= `ncpgen/ncpgen.py gen ...`); check a part: `ncpgen/<part>.py check <dirs>` (`python3 -m ncpgen -h` lists every command) |
| Snapshot (refactors) | `tests/snapshot.py record $NCPGEN_WORK/<name>.json.gz [--code <checkout>]`, then `compare` after the change (`--sample K` for a quick look) |
| Dump / diff an NCP | `oracle/ncp-dump.py net.ncp --log log.txt [--section isa\|io\|...]` |
| Ghidra | `oracle/ghidra/run-ghidra.sh decompile <targets.txt>` (one process at a time; JAVA_HOME unset) |
| Rebuild the harness | `docker run --rm --platform linux/amd64 -v "$PWD":"$PWD" -w "$PWD" python:3.12-slim-bookworm oracle/harness/build.sh` → root-fd659af2 (with `dwconv`; `work/n3/root-current`) |

Python: `python3 -I`, standard library only. The host has docker (arm64 containers run NPUC; amd64 ones run the NDK).

## Where things stand (2026-11-23)

Byte-exact with NPUC on every compiled network (`tests/regress.sh`) except NPUC's 7 uninitialised-byte eltwises, one
camera backbone (efficientdet-lite0-320-p4) and one of the fresh camera-size random graphs (graph-camera v3003; frame.md
6.11). Camera resolutions (192..320 px): SSD MobileNet v1's heads and the SSD v1/v2, MoveNet and MobileDet (to C4)
backbones exact; NPUC crashes or asserts on MobileDet's and EfficientDet-Lite0's full backbones and PoseNet at 257 px,
and ncpgen refuses them the same way (models/README.md "Camera resolutions"). Tiling inside a graph follows NPUC's
policy (frame.graph_policy), NPUC's trim and spill is modelled from its GlobalMemoryAllocator in graphs and chains
(frame.md 6.12), its layer-block forwarding from its calculators (frame.forwarding), the CMDQ cut in a tiled plan's
single blocks too (6.9). Common models: `models/` (31 of 32 zoo entries exact, the 32nd fails in NPUC; every standard-
resolution model as published). Depthwise is lowered to the dense block-diagonal conv NPUC itself compiles
(docs/format/depthwise.md; NPUC's own depthwise-optimised path is off on the device and unproven on the NPU). Camera
models: models/camera.md (person_detect exact as published; per-channel scales as per-channel shifts,
docs/format/perchannel.md; the harness's setScale hook needs root-40bb3e3f). ADD with any scales: `--add-as-conv`
(opt-in, a concat with the harness's `concat ... mixed` and a 1x1 conv; docs/format/add.md; NPUC itself has no form of
it). Not supported yet: channel split/slice, other activations (leaky ReLU, SiLU; the NCP has a LUT memory type), upsampling
(NPUC's deconvolution crashed in the harness), an output another layer also reads (NPUC crashes on it), strata joined
to each other. Several outputs (detection heads) are supported (docs/format/frame.md
12; the harness's `output` line needs root-38925a6d or later). Nothing beyond one example has been run on the NPU; the NPU runs are done elsewhere (the nervesinator
porting kit).
