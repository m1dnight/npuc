# ncpgen

An open compiler for the NPU of Samsung's Exynos 990 (Galaxy S20 family): it turns a small quantized neural network into
an **NCP**, the binary the NPU's firmware loads and runs. Pure Python 3, no dependencies, no Samsung code.

ncpgen was written by using Samsung's own on-device compiler (NPUC 1.36.1.a, from the phone's stock firmware) as an
oracle: thousands of networks were compiled with it, their NCPs diffed, and NPUC's code read where the diffs were not
enough. ncpgen now writes **the same bytes as NPUC** for every network NPUC compiles on this path, and corrects what NPUC
gets wrong (its activation table, below).

## Status

| | |
|---|---|
| Byte-exact with NPUC | about 3150 of about 3250 compiled networks, every developed set at its maximum; the misses are 7 networks whose NCPs NPUC fills with uninitialised bytes (it differs from itself) and part of a sweep built to stress short weight rings (in progress) |
| Without any vendor file | `tests/run.sh`: 109 of our own networks regenerated, each NCP equal by sha256 to NPUC's |
| On the NPU | the example network (`ncpgen gen` of npunet's example writes the exact NCP of a bit-exact NPU run); the rest is predicted, with a run plan in the porting kit |

Operations: CONV_2D (kernels 1..13, VALID up to 14, stride 1/2, SAME/VALID, ReLU or none), MAX/AVERAGE_POOL_2D (strides up to
3), global average pooling (as a conv), FULLY_CONNECTED (as a conv), ADD (quant mode 3, power-of-two scales), CONCATENATION
(channels), standalone RELU; RESHAPE and SOFTMAX after the NPU part run on the CPU. Quant modes 1 (uint8) and 3 (int8),
one weight scale per layer. Large layers: NPUC's tiling policy (read from its code), weight rings, its command-queue
split. Not yet: depthwise convolutions, other activations (leaky ReLU, SiLU), upsampling, channel split, several outputs.

NPUC's defects ncpgen knows (`docs/format/grelu.md`, `docs/format/frame.md` section 9): its activation (GReLU) table is
the DRUs' slope and zero-point slots read as one piece table, right only when ceil(Cout/8) is even; ncpgen writes the
corrected table by default (`--grelu npuc` gives NPUC's bytes). Two DMA stride defects are suspected (3/5/7-channel
outputs; a conv after a widened pool): `--fix-npuc` corrects them, unproven on the NPU.

## Use

```bash
# a model -> an NCP (TFLite int8/uint8 or the npu-net/1 JSON of ncpgen/npunet.py)
python3 -I ncpgen/ncpgen.py gen model.tflite net.ncp [--quant 1|3] [--allow-none] [--grelu fixed|robust|npuc] [--fix-npuc]

# the CPU reference: what the NPU must output for an input (the model's own NHWC tensor)
python3 -I ncpgen/npunet.py lower model.tflite low/        # low/lowered.json, low/net.npuc
python3 -I ncpgen/npunet.py ref low/lowered.json input.bin expected.bin

tests/run.sh                                               # the tests, no vendor file needed
```

Running an NCP needs the NPU, its firmware (`NPU.bin`, from the phone's own stock firmware; not distributable) and a
driver: the [nervesinator](../nervesinator) porting kit has a test driver and the scripts (`ports/samsung-x1s/npu/`).

## Layout

| Path | What |
|---|---|
| `ncpgen/ncpgen.py` | the generator's entry: `gen`, `check` (against compiled networks), `corpus` |
| `ncpgen/stream.py` | the CMDQ instruction streams (`docs/format/cmdq.md`) |
| `ncpgen/frame.py` | the container: header, vectors, DMA descriptors, intrinsics, NPUC's plan and tiling policy, the input layout (`docs/format/frame.md`) |
| `ncpgen/weights.py` | the weight packing (`docs/format/weights.md`) |
| `ncpgen/grelu.py` | the activation table (`docs/format/grelu.md`) |
| `ncpgen/npunet.py` | models in: TFLite or JSON lowered to NPUC's arithmetic; the exact CPU reference; input/output conversion |
| `ncpgen/*_sweep*`, `stream_variants*.txt`, `stream_graphs.py`, `frame_sweep.py` | the sweeps each rule was found and checked on |
| `oracle/` | NPUC as an oracle: our harness around it (`harness/`), the host runner, `npuc-local.sh`, `npuc-variant.py`, `ncp-dump.py`, Ghidra scripts (`ghidra/`), op tests (`ops/`); `docs/oracle.md` |
| `tests/` | `run.sh` and `golden.txt` |

## Vendor files

Nothing from Samsung is in this repository: no library, no firmware, no compiled NCP, no decompiled code. NPUC, the
stock Bionic runtime, compiled NCPs, NPUC's logs and Ghidra's output live in the work folder (`NCPGEN_WORK`, default
`work/`, git-ignored; `oracle/env.sh`). The generator holds formulas, register addresses, field layouts and constants
observed on the oracle; the two ARM intrinsics are written by our own A32 encoder field by field. The work was done for
interoperability (EU Directive 2009/24/EC, article 6); this is not legal advice.

## Licence

To be decided before anything is published.
