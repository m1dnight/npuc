# NPUC harness: reproducible build on the Mac, and more layer types (PORT-32, 2026-11-21)

Host only (the Apple-silicon Mac); no phone, nothing ran on the NPU. Tags: **V** verified (run, offset or code read),
**G** guess. Evidence: compile runs under `output/npu/n3/ops-check/`, `output/npu/n3/ops-evidence/` (git-ignored: NCPs
and logs are vendor-generated), offsets in the stock G980FXXSNHYB1 libraries (`$NCPGEN_WORK/vendor-libs/`). Read with
`/usr/bin/objdump` (Apple LLVM 21, aarch64 ELF, `-C`) and the earlier Ghidra decompiles in `$NCPGEN_WORK/n3/ghidra/decomp/`.
Background: `npu-compiler.md` §1.4-1.5, §9.

## 1. The harness build reproduced byte for byte on the Mac (V)

| Input | What was done |
|---|---|
| NDK r27d | `https://dl.google.com/android/repository/android-ndk-r27d-linux.zip` into `cache/android-ndk-dl/`: 663 956 036 B, sha256 `60124608…5d9`, sha1 `22105e41…121` (= §9.1), unpacked to `cache/android-ndk-r27d/` (`Pkg.ReleaseName = r27d`). APFS is case-insensitive: 8 netfilter headers collide (`xt_CONNMARK.h`/`xt_connmark.h`…); none is used by the harness |
| Bionic | `output/npu/bionic/` recreated from `output/npu/n3/root-7517804f/` (linker64, ld-android, libc, libm, libdl, liblog, libc++) + `SHA256SUMS`: the 7 hashes equal §9.1's |
| Vendor libraries | `$NCPGEN_WORK/vendor-libs/libnpuc_*.so` = the run directory's (sha256 compared) |

`build-n3.sh`, unmodified, in an amd64 Debian container (Rosetta; the docker engine is socktainer), the kit mounted at
the main machine's path so every absolute path in the debug info is the same:

```bash
K=<this host's kit>; M=<the main machine's kit path>
docker run --rm --platform linux/amd64 -v "$K":"$M" -w "$M" python:3.12-slim-bookworm \
  bash "$M/ports/samsung-x1s/kernel/test-npu-n3/build-n3.sh"
```

(That was the porting kit's layout. In this repository the build is `oracle/harness/build.sh`, reproducible from any
path; it gives root-314e61a6: `docs/oracle.md` section 2.)

Result (V): **npuc-probe-482be94e, npuc-conv-6595c839, npuc-net-1a1f733b and root-7517804f reproduced byte for byte**
(the script found the three binaries and the run directory already present and `cmp` equal), check-symbols PASS and its
self-test. Image `python:3.12-slim-bookworm`, amd64, `sha256:313cf8f1…` (pulled 2026-11-21).

## 2. How NPUC's frontend turns an NNLayer into graph nodes (V)

- **Type string -> NodeLayerType.** `NNGraphAttribute::NNGraphAttribute(const shared_ptr<NNLayer>&)` @libnpuc_frontend
  0x4d8e0 compares `NNLayer.type` case-insensitively (`NPUC::iequals`) with 64 Caffe names in Caffe's order and stores
  the index at NNGraphAttribute +0x30 (= NNGraphNode +0xa0); no match -> 0x43. Recovered by emulating the string
  building of each compare: 0 ImageData, 1 Data, … 4 Input, 8 Convolution, 9 Pooling, 0xc Crop, 0xd Deconvolution,
  0x12 InnerProduct, 0x17 BatchNorm, 0x18 ReLU, 0x19 ReLU6, 0x1a PReLU, 0x1c Sigmoid, 0x1d TanH, 0x1f Power, 0x26
  Flatten, 0x27 Reshape, 0x2a Concat, 0x2b Slice, 0x2c Eltwise, 0x31 ArgMax, 0x32 Softmax, 0x3c PriorBox, 0x3d Permute,
  0x3e/0x3f Convolution/DeconvolutionRistretto, 0x40 FcRistretto (enum in `npuc.h`).
- **Dispatch.** `NNGraphCgObject::addGraphNode` @0x65540: jump table at 0xbe940 on (type - 4): Input 4/7, Convolution 8,
  Pooling 9/10, Crop, Deconvolution, InnerProduct, ReLU 0x18/0x19/0x1a (`addReLUGraphNode(node, type)`; ReLU6 goes to
  addActivationGraphNode when soc == 0), Sigmoid/TanH/Power (addActivationGraphNode), Flatten, Reshape, Concat, Slice,
  Eltwise, PriorBox, Permute; anything else: "not supported" (returns 4). The builders return void; addGraphNode
  returns 0.
- **Attributes.** The `NNGraphAttribute::get*Param()` accessors for the "stensorflow" framework (what setCompileOptions
  sets): Convolution, Pooling, Eltwise and Reshape are implemented; InnerProduct, ReLU, Concat, ArgMax, Flatten,
  Permute, Crop, Slice throw "doesn't support STF … param" (0x50d38, 0x52030, 0x5245c…). The builders for Concat,
  InnerProduct and ReLU do not call theirs.
- **On-device fillers** (CompileOptions +0x138 = 1: BiasON, KernelON, ScaleON, ShiftON, ZeroPoint, FusedActivation,
  Data): Bias/Kernel/Scale/Shift act only when `NNLayer.type == "Convolution"` (memcmp with 0xc0df0), ZeroPoint and
  FusedActivation only on node types 8 and 9 (ZeroPoint also needs CompileOptions +0xec == 2). No MergeActivation or
  AddDummyConv filter runs on this path (none in any log), and the off-device NNGraphNodeMultipleInputFiller (Eltwise)
  is not in the chain. **ConcatScaleFoldingFilter does run** (V, 2026-11-22, docs/format/add.md 3): a concat whose
  conv predecessors' output scales differ from its own gives them its scale (their multipliers recomputed, zero points
  kept), a pool predecessor gets a scale conv; it was a no-op on every concat before (equal scales), hence "Graph
  BEFORE/AFTER ConcatScaleFoldingFilter" with nothing changed in their logs.

## 3. The ops

### 3.1 Eltwise Add: compiles (V), with one gap filled by the harness

| Fact | Evidence |
|---|---|
| Layer: `NNLayer` (C2), type "Eltwise", inputs {in1, in2}, outputs {name}, shapes of all three blobs set | npuc-net |
| Attribute: our own `EltwiseAttr` (no class exists: no typeinfo in NPUC or nn-eden-drv); the only reader, `getEltwiseParam` @0x50f5c, copies the std::string at attr +0x08: "Add" -> 0, "Multiply" -> 1, "Maximum" -> 2, else 3 (addElemtWiseGraphNode asserts) | V, disassembly |
| addElemtWiseGraphNode @0x6a8c0: 2 input data nodes (node +0x160), 1 output (+0x178), op, and **node +0xe8 copied into NNGraphCgEltwiseParam** | V |
| Node +0xe8 = `shared_ptr<NPUMultipleInputData>`, set only by NNGraphNodeMultipleInputFiller::setMultipleInput @0xa46e4 (off-device, from the node's fractional lengths). Empty on device -> **SIGSEGV in EltwiseNode2020::buildMetaDataDram** (libnpuc_template 0x1c2ba4) | V, first run (crash locator) |
| `NPUMultipleInputMemoryData` (0x1a0 B, vtable exported: [~D2, ~D0, getInputData(i) = +0x08 + 0x18 i, getFractionalLengthData(i) = +0xc8 + 0x18 i]): fl[0] = input 1, fl[1] = input 2, fl[8] = output, one byte per channel | V, setMultipleInput, vtable relocations @0xed5f8 |
| **The harness hook**: npuc-net defines and exports `NPUC::NNGraphCgObject::addElemtWiseGraphNode(NNGraphNode*)`; addGraphNode calls it through the PLT, so Bionic binds it to the executable (first in the lookup scope). For an Eltwise node of the net file it stores an NPUMultipleInputMemoryData with the fractional lengths (scale = 2^-fl), then calls NPUC's own function (`dlsym(RTLD_NEXT)`). Networks without Eltwise never reach it | V: the log's "eltwise add1: multiple-input data set … in1 7 in2 7 out 7", then a normal compile; check-symbols PASS |
| Result: NCP, ISA `ncp_eltwise_op` ×4 (one per core), plan "ELTWISE op: SRC1, SRC2, WM1, WM2, CU_bin (128 B), DST" + a 32 B META_DATA block per SRAM; residual (conv + network input), two convs, a 16-channel add, add then ReLU all compile | V, `ops/check-ops.sh` |
| The META_DATA (a WEIGHT address vector, 128 B, DMA'd to SRAM +0x880): per core `[fl1 c0, fl1 c1, 0, 0] [fl2 c0, fl2 c1, 0, 0] [flo c0, flo c1, 0, 0]` + 20 zero bytes (8 channels, 2 per core). Changing the scales (7/8/6) changes only these 16 bytes of the 11 112 B NCP; the instruction stream is identical | V, add-res-q3 vs add-res-fl-q3 (`ops-evidence/`) |
| Multiply: backend assert "Unsupported eltwise op!" (ncp_eltwise_op.cpp:81); Maximum: assert in EltwiseNode2020::getEltwiseOpcode; the same blob twice (x + x): PolicyMaker PGraph::clone "this->equals(graph)" | V (scratch runs before the refusals were added; max-q3's log kept in `ops-evidence/`; npuc-net now refuses all three) |
| Semantics on the NPU: the eltwise only shifts by powers of two and has no zero-point term. **G**: out = sat(round(in1·2^(flo−fl1) + in2·2^(flo−fl2))) in the signed domain; meaningful in quant 3 (s8, zero point 0); quant 1 compiles but is expected to be off by the zero point | G until an NPU run (§5) |

### 3.2 Concat (channels): compiles (V)

- Layer: `NNLayer` type "Concat" with nn-eden-drv's `ConcatAttr` layout (0x28 B: int axis +0x08, vector<float> output
  scales +0x10; loadConcatAttrForConcatLayer @0x175378), shapes of every input and the output set by npuc-net (the
  HAL's NNConcatLayer::computeOutputShape is its own, not in NPUC). After appending, npuc-net calls NPUC's exported
  `NNNetwork::mergeConcatLayer()` as nn-eden-drv's buildNNNetwork does last (it only merges Concat chains).
- NPUC's graph code never reads ConcatAttr on this path (getConcatParam throws). The data filler makes each input a
  `ChannelSlicedFeatureNode` of the output: **channels only, in input order** (V: swapping the inputs swaps the two
  convs' DRAMOUTSET offsets 0 / 0x800 = channel 8 of a CHW 16x16 map; concat-q3 vs concat-swap-q3). The frontend's
  ConcatScaleFoldingFilter reads its output scale (ConcatAttr +0x10, element 0) and gives conv predecessors with
  another scale that scale (a change of their multipliers), pool predecessors a scale conv (docs/format/add.md 3): so
  every input must already be in the output's scale; zero points are not compared (`concat ... mixed`, section 4).
- Placement: "Concat in SRAM" needs producer convs with a multiple of 16 output channels (log: "not supported yet for
  2020" otherwise); else a DRAM concatenation ("output DRAM is concatenated in …"). 8+8, 8+8+8 and 16+16 compile.
- The network input as a concat operand: NPUC writes the Input op straight into a channel slice and its policy maker
  asserts "DataNode id isn't found in the NNGraph inputs" (V; also needed `shapes["ROOT"]` on the input layer, now set
  for "npuc-net 2" files as the HAL's createDummyInputLayer does through computeOutputShape). Refused by npuc-net.

### 3.3 Standalone ReLU / ReLU6: compiles (V)

Type "ReLU"/"ReLU6", an empty attribute (`PlainAttr`; addReLUGraphNode reads only the input shape). NPUC makes it a
**1x1 convolution** (an extra `ncp_mtc_convolution`, 128 B of weights, no PAD) and emits **no GReLU block** for it (the
two blocks in relu-q3 belong to the two real convs). Since no MergeActivation filter runs on device, a ReLU right after
a conv is not merged: use the conv's fused activation there; the layer is for a ReLU after an Eltwise, a Concat or a
pool. What its 1x1 conv computes (identity weights? which activation register?) is G until an NPU run.

### 3.4 Depthwise convolution (`dwconv`): compiles (V; 2026-11-22, docs/format/depthwise.md)

- Layer: NPUC's own `NNConvLayer` (type "Convolution") with ConvAttr group = numInput = numOutput = C, as nn-eden-drv's
  ConvAttr::setKernelInfo sets it for a non-"Conv2D" attribute (multiplier 1). The kernel must be the dense C x C x kh
  x kw one (KernelONFiller copies numOutput x numInput x kh x kw bytes, unchecked): the taps on the diagonal, the
  weight zero point elsewhere, as NPUC's off-device KernelFiller expands a depthwise kernel.
- **Stock** (what the device's NPUC does): NNGraphAttribute::isDwConvolution is true, but DwOptSupported is false (it
  wants the framework "scaffe"; the on-device frontend's attribute says "stensorflow"), so the op's group stays 1 and
  the NCP is byte for byte the NCP of a regular conv with the dense diagonal kernel: 118 of 118 networks (one-layer
  grid, MobileNet blocks, MobileNet v1 0.25 on 96x96x1, 40 unseen random ones). No NPUC abort.
- **DW opt** (NPUC's depthwise path, off the device only): the harness defines `NPUC::NNGraphAttribute::DwOptSupported()`
  (called through the PLTs of libnpuc_frontend and libnpuc_template, so the executable's definition binds first) and,
  for a `dwconv ... dwopt` layer, runs NPUC's function with the framework string set to "scaffe". Taken for C a
  multiple of 16 and stride <= 2: a 16-input-channel weight pack per pass and three conv register families change
  (depthwise.md section 5). Every other node gets NPUC's function unchanged.

### 3.5 Not available from NNNetwork on this path (V)

| Op | What happened | Consequence |
|---|---|---|
| InnerProduct (type "InnerProduct" + ConvAttr) | the frontend builds the FC node (addInnerProductGraphNode: FCParam + Unflatten), then SIGSEGV in the policy maker (`PGraph::fillNNGraphToPGraphMap`, BFS lambda; fc-x). G: the kernel/bias/shift data nodes it reads (node +0xa8/+0xc8/+0xd8) are never filled, the ON fillers only handle "Convolution" | keep npunet's lowering of FULLY_CONNECTED to a conv whose kernel covers the input (fc-conv-ref compiles) |
| Deconvolution (type "Deconvolution" + ConvAttr) | SIGSEGV in `NNGraphAttribute::getBlobParam` called by addDeconvolutionGraphNode (deconv-x) | not supported |

Not tried: Sigmoid/TanH/Power (addActivationGraphNode), Reshape (the HAL supports it; its ReshapeAttr is not
reconstructed), Flatten/Permute/Crop/Slice (their STF param getters throw).

## 4. npuc-net: what was added (`oracle/harness/npuc-net.cpp`, `npuc.h`)

"npuc-net 1" files are read exactly as before (the new kinds are refused there). An **"npuc-net 2"** file may also hold:

```
eltwise <name> <in1> <in2> Add <in1 scale hex> <in1 zp> <in2 scale hex> <in2 zp> <out scale hex> <out zp>
concat  <name> 1 <n> <in1> ... <in_n> <out scale hex> <out zp> [mixed]
relu    <name> <input> <ReLU|ReLU6>
dwconv  <name> <input> <stock|dwopt> <kh> <kw> <sh> <sw> <SAME|VALID> <act 0|1> <in scale hex> <in zp> <w scale hex> <w zp>
        <out scale hex> <out zp> <weights: C x kh x kw uint8, hex> <C> <bias x C>
```

Inputs may be any earlier blob (graphs, not only chains). Checks before any NPUC call (exit 1): eltwise Add only, two
different blobs of the same shape, every scale 2^-n (n 0..31), equal zero points; concat channels only, equal H x W, not
on the network input; **every reader (conv, pool, eltwise, concat) must declare the scale and zero point its input
blob already has** (the producer's output; the network input's from its first reader); a blob nobody reads prints a
note. net.json gets an `"inputs": [...]` list for the new layers (`"input"` = the first). `dwconv` (added 2026-11-22):
C = the input's channels >= 2, the weights per channel [c][ky][kx]; npuc-net expands them to the dense diagonal kernel;
refused in an "npuc-net 1" file, for C = 1, an unknown mode or a wrong weight count. Layouts in npuc.h with
static_asserts: EltwiseAttr (+0x08, 0x20), ConcatAttr (+0x08, +0x10, 0x28), NPUMultipleInputMemoryData (+0x08, +0xc8,
0x1a0), NodeLayerType, NNGraphNode offsets. The hook is compiled for aarch64 only (the host test's x86_64 stub build
has no NPUC; it still builds and runs, V).

`concat ... mixed` (added 2026-11-22, root-81d28366; with the `output` line: root-3504c01f): the concat's inputs keep their own zero points (the check that
each input has the concat's quantisation is skipped; each input must still be a layer's output). For npunet's
`--add-as-conv` (docs/format/add.md): a 1x1 conv over the concat weighs each input on its own and takes the zero-point
difference into its bias. The scales must still agree (NPUC's ConcatScaleFoldingFilter compares them): npunet gives
both inputs one scale label without changing a multiplier.

**Run directory `$NCPGEN_WORK/n3/root-current`** (built by build-n3.sh in the container): npuc-net-1e694f20, npuc-conv-409e9b04
(npuc.h changed; its NCP for `--quant 3` is d6cb7f2a, the same as npuc-conv-6595c839's, V), npuc-probe-482be94e;
check-symbols PASS. (`output/npu/n3/dev/root` and `dev2/root` are scratch builds of the same source; the GReLU sweep
already uses `dev/root`.)

Regression (V): 53 distinct earlier inputs (the 6 example/phone nets + 47 sampled from `$NCPGEN_WORK/sweep/`) give
byte-identical NCPs with root-7517804f and root-b53daf10 (net-current -> 5c080ec1); 14 of them recompiled with an
"npuc-net 2" header give the same NCPs too. `oracle/ops/check-ops.sh [root]` (networks from `ops/mknets.py`,
our own seeded weights): **20 PASS** (10 compiles with the op in the log, 8 refusals, 2 regressions); a second run gave
the same 10 NCP hashes (add-res-q3 f4005cdd, concat-q3 aaf6f04e, relu-q3 8480fe7e…).

**Run directory with `dwconv` (2026-11-22)**: `oracle/harness/build.sh` gives root-fd659af2 (npuc-net-acc7a456,
npuc-conv-6e2aa54d: npuc.h changed, npuc-probe-85fe33b0); `LINK_CURRENT=0` builds it without moving n3/root-current.
Same NCPs as root-314e61a6 on every earlier input tried: `oracle/ops/check-ops.sh` 31 PASS (the 20 earlier cases with
their NCP hashes unchanged, plus 3 stock/dense pairs with equal NCPs, a dwopt case and 4 refusals), and 66 of 66
stored NCPs (40 corpus, 10 graph-random2, 10 random, 6 tiling3) recompiled byte for byte.

**Several outputs (2026-11-22, docs/format/frame.md 12)**: an "npuc-net 2" file may end with
`output <n> <blob1> ... <blob_n>`, the network's outputs in the model's order. The HAL marks nothing: NPUC's frontend
makes every node without successors an output (NNGraphCgObject::appendNewOutData), so the line only declares them and
npuc-net checks NPUC's rule before compiling (exit 1): every blob no layer reads is listed, no listed blob is read by
a layer, no name twice, not the input. net.json then has `"outputs": [{"name", "out_shape"}...]` in the file's order
(NPUC's own order is its log's "generateGraph Network Output:" list, its execution order). Without the line, as
before. `--intermediate <layer>` (npuc-local.sh `--intermediate`) sets CompileOptions +0xb8, NPUC's off-device way to
output a blob a layer also reads (isIntermediateOutput): a probe only; NPUC's policy maker crashes on it
(PBuilder::IDram::build2ndStage, frame.md 12). Build: **root-f7b580ef** (npuc-net-3c235038; npuc-probe-85fe33b0 and
npuc-conv-6e2aa54d unchanged), `LINK_CURRENT=0`. The same NCPs as root-fd659af2: the 12 outputs probes recompiled with
the line (and "npuc-net 2" instead of 1 for the conv-only ones) byte for byte, and 37 of 38 stored NCPs sampled from
corpus, graph, graph-random, graph-random2, random, graph-split and tiling2 (the other: graph rc4-24x12, NPUC's own
uninitialised descriptor words, which root-fd659af2 does not reproduce either). Merged with main's per-channel lines
(`wscales`, `slopes`, the setScale hook): **root-38925a6d** (npuc-net-627f953e), the run directory for both.

## 5. What ncpgen/npunet.py needs to lower TFLite to it (for the orchestrator)

- Write "npuc-net 2" (v1 content gives the same NCPs under it) and keep a blob table: every reader must quote its
  input's exact scale/zp.
- **ADD** -> `eltwise … Add`: requires quant 3 (zero points 0) and power-of-two scales for both inputs and the output.
  Where a conv produces an input, set that conv's output scale to the power of two (requantising in the conv is exact
  up to its rounding); a network-input operand fixes its own scale. A fused ReLU on ADD -> a following `relu` line.
  Otherwise refuse (NPUC cannot rescale an Eltwise input).
- **CONCATENATION** -> `concat … 1 …` for the channel axis (TFLite NHWC axis 3 / -1) only; give each producing conv
  the concat's output scale/zp (no rescale exists); refuse an operand that is the network input (or insert a 1x1
  identity conv before it, G).
- **FULLY_CONNECTED**: keep today's lowering to a conv (kernel = the input's H x W, VALID); native InnerProduct does not
  compile on device.
- Standalone RELU/RELU6 after a conv: fold into the conv's `act`; elsewhere `relu`.
- The CPU reference: Add as G in §3.1 until measured; ReLU's 1x1 conv as identity + ReLU (G).
- **DEPTHWISE_CONV_2D** (multiplier 1) -> a plain `conv` with the dense diagonal kernel (no harness syntax needed: the
  NCP is the stock one, and ncpgen already generates it), or `dwconv ... stock` (same NCP); depthwise.md section 6.

NPU run to propose on the main machine (no phone here): `npu/npu-run.sh` with an add-res-q3-like net (conv + ReLU,
residual Add of the input at 2^-7, out 2^-7, then a conv) and a concat-q3-like net; compare with the G formula; also
dump the eltwise's META_DATA/CU_BIN as loaded. Expect the GReLU table issue on the convs (`--ncp-patch auto`).
