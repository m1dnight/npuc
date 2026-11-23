# ADD with any scales (NCP format, parts "npunet", "stream", "frame")

Exynos 990 NPU, NCPs compiled by Samsung's NPUC 1.36.1.a (the oracle). What Samsung's software does with an Android
quantised ADD whose inputs and output have arbitrary scales and zero points (every ADD of the camera models,
models/camera.md), and how npunet lowers one into what NPUC compiles and ncpgen generates exactly. Code:
`ncpgen/npunet.py` (`_lower_add_conv`, `add_fit`, `--add-as-conv`), `ncpgen/stream.py` and `ncpgen/frame.py` (a concat
input other layers read), the harness's `concat ... mixed` (`oracle/harness/npuc-net.cpp`, root-81d28366), the sweep
`ncpgen/add_sweep.py`, the measurement `models/camera.py addcheck`. Ghidra targets: `oracle/ghidra/targets-add.txt`,
`targets-add2.txt`.

Tags: **V** verified (bytes, log, code), **G** guess. Addresses are the libraries' own (Ghidra +0x100000).

## 1. Summary

- **Samsung's stock software never hands NPUC an ADD** (V, nn-eden-drv). The NNAPI HAL's per-operation test puts ADD
  outside the NPU partition, its whole-model test requires a model of exactly CONV_2D, AVERAGE_POOL_2D, MAX_POOL_2D,
  CONCATENATION and RESHAPE (SOFTMAX allowed), and its network builder has no ADD case. An ADD runs on the CPU (or GPU)
  in the Eden runtime, the model split around it. There is no power-of-two approximation, no multiplier and no
  inserted rescaling layer to copy: NPUC's handling of an arbitrary-scale ADD is **absent**.
- **NPUC's own Eltwise only shifts** (V): its data are three per-channel fractional lengths (input 1, input 2, output),
  the firmware's activation unit interface has no multiplier and no zero point, and in quant mode 1 the eltwise group
  and its META_DATA are byte for byte those of quant mode 3. It cannot take any of the camera models' 124 ADDs (every
  one has zero points other than u8 128, i.e. needs quant mode 1).
- **NPUC rescales a concat's inputs** (V; this corrects docs/harness-ops.md 2): its ConcatScaleFoldingFilter, on the
  device chain, gives a concat's conv predecessors the concat's output scale (zero point kept) and inserts a scale conv
  after a pool predecessor. It was a no-op on every earlier concat (equal scales).
- **npunet's lowering (`--add-as-conv`, opt-in: a change of the model)**: the ADD becomes a channel concat of its two
  inputs, each keeping its own zero point, and a 1x1 conv over the 2C channels: output o weighs input a's channel o by
  an integer wa and input b's by wb, its bias holds the zero-point difference, so the accumulator is wa (a - z_a) +
  wb (b - z_b) exactly and one requantisation (NPUC's 8-bit multiplier) gives the output. wa, wb and the multiplier are
  fitted to s_a / s_out and s_b / s_out (worst case 0.03 to 0.11 LSB on average per model, 0.36 at most).
  Against TFLite's ADD on the models' own activations: **0.15 to 0.38 % of each ADD's output values differ, by 1 LSB,
  never more** (section 6). NPUC compiles it and ncpgen generates it exactly: 20 of 20 probes, **40 of 40 unseen random
  networks** at first sight, and 40 of 40 more in a second unseen set.

## 2. What Samsung does (V)

| Where | What |
|---|---|
| nn-eden-drv `ModelConverter::supportedByNPU` @0x156bc8 | per NNAPI operation: type < 0x17 with bit set in 0x420006 (AVERAGE_POOL_2D 1, CONCATENATION 2, MAX_POOL_2D 17, RESHAPE 22) is supported, CONV_2D 3 with its input TENSOR_QUANT8_ASYMM and kernel, stride and padding limits; every other type, ADD (0) and DEPTHWISE_CONV_2D (4) included, logs "opType=%d is not supported by NPU!" and returns 0 |
| `ModelConverter::divideOperationList` @0x155814, `getOperationsExceptNPU` @0x155a20, `convert` @0x154fc8 | split the model's operations into NPU and non-NPU sub-lists ("vecSupportedByNPU", "Create for NPU!!" / "Create for GPU!!"): the ADD goes to the CPU/GPU side (nn-eden-drv carries TFLite's CPU kernels: strings `optimized_integer_ops::Add`, `reference_integer_ops::BroadcastAdd4DSlow`) |
| `CompilerManager::isSupportedModelByNPUC` @0x169058 | ORs a bit per operation (CONV_2D 1, MAX_POOL_2D 2, AVERAGE_POOL_2D 4, RESHAPE 8, CONCATENATION 0x10, SOFTMAX 0x20), returns 0 for any other type, and 1 only when (bits \| 0x20) == 0x3f: all five present |
| `NNNetworkBuilder::buildNNNetwork` @0x169f7c | switch on the NNAPI type: 1 average_pool_2d, 2 concat, 3 conv2d, 0x11 max_pool_2d, 0x16 reshape; else "not yet supported : %d" |

So on this firmware no NNAPI ADD reaches NPUC in any form. NPUC's Eltwise is reachable only through its Caffe-style
graph layer (Samsung's offline flow; our harness's `eltwise` line, docs/harness-ops.md 3.1):

| Where | What |
|---|---|
| frontend `NNGraphNodeMultipleInputFiller::setMultipleInput` @0x1a46e4 (off-device filler) | builds the NPUMultipleInputMemoryData from the node's fractional-length vectors: one byte per channel for input 1, input 2 and the output (fl[0], fl[1], fl[8]); nothing else |
| template `EltwiseNode2020::buildMetaDataDram` @0x2c2ad4 | the META_DATA block: per core, the three fractional-length arrays (weights.eltwise_block); no multiplier, no zero point |
| backend `NcpEltwiseOp::ncp_eltwise_add` @0x1f60bc, its intrinsic header text | `struct act_tensor { vector, offset, fractional_vec, fractional_offset, w/c/n_stride }`, `op.proc_type = ACT_PROC_ELTADD`, `op.data_type = ACT_DATA_S8 / S16 / U8` (asserts "Unsupported data type!" otherwise): the firmware's activation unit gets a fractional-length vector per tensor and nothing to rescale or offset with |
| oracle: add-res-q1 vs add-res-q3 (`$NCPGEN_WORK/n3/ops-evidence/`) | the same network in quant 1 (zero points 128) and quant 3: 1922 bytes differ, all in the two convs' weight blocks and GReLU registers; the eltwise group (isa 0x6f0+240) and its META_DATA are identical. The zero point never reaches the Eltwise (in quant 1 it is off by the zero point, G until an NPU run) |

Approximating each scale by a power of two (what an Eltwise-based lowering would need) moves a scale by up to a factor
sqrt(2) (up to 41 %), and still needs zero points 0, which none of the camera models has; section 6 measures the best
such scheme.

## 3. NPUC's concat scale folding (V)

`ConcatScaleFoldingFilter::needsFilter` @frontend 0x183138: for a Concat node (type 0x2a) when the framework is not
"scaffe" (the device's "stensorflow"), compares each conv predecessor's output scale (ConvAttr +0x160, element 0) and
each pool predecessor's (PoolAttr +0xb0) with the concat's (ConcatAttr +0x10); any difference: the filter runs.
`doFilter` @0x186fbc: a conv predecessor's output scale vector is replaced by the concat's (the zero point is kept; its
multiplier is then recomputed by the scale filler); a pool predecessor gets `addScaleConvLayer` @0x18383c (a scale
conv after it), which asserts "poolAttr->getInputScale()[0] <= concatAttr->getOutputScale()[0]" (V: add-probe "pools"
before the relabelling below, SIGABRT). The log says "LAYER(Concat/...) will update the scale value of predecessors".
V: add-probe res-c8 compiled with a concat declaring proj's scale: the stem's multiplier became 246/2^10 (its output
scale replaced) instead of 218/2^14. Every concat of the earlier sweeps had equal scales, so the filter never acted.

## 4. npunet's lowering (`--add-as-conv`)

An ADD NPUC's Eltwise cannot take (any scale not 2^-n, any zero points, quant 1, x + x) with `--add-as-conv`; without
the flag it is refused as before, the message naming the flag. Power-of-two quant 3 ADDs keep the Eltwise.

- **The conv.** ADD(a, b) -> `concat [a, b]` (channels) -> 1x1 VALID conv 2C -> C: weights w[o][o] = wa, w[o][C + o]
  = wb, every other weight at the weight zero point; input zero point z_a, bias wb (z_a - z_b) per channel; so acc =
  wa (a - z_a) + wb (b - z_b), exact. out = z_out + R(acc slope / 2^shift), clamped (NPUC's conv, the reference's
  rounding half away). Quant 1: uint8 weights, weight zero point 0, wa and wb in 1..255 (both ratios are positive);
  quant 3: int8 weights (zero point 128), 1..127. A fused RELU is the conv's; RELU6 only where its clamp is a no-op
  (or `--relu6-range`). x + x: one 1x1 conv over x with ratio 2 s_x / s_out.
- **The fit** (`add_fit`): over NPUC's multipliers below 1 (slope 128..255, shift 8..15: NPUC's setScale gives 8 bits,
  docs/format/grelu.md 5) and the two integer weights, minimise the worst output error |wa M - r_a| R_a + |wb M - r_b|
  R_b (M = slope / 2^shift, r = s / s_out, R = the input's largest |x - z|). The conv's declared weight scale is the
  float32 for which NPUC's setScale gives exactly that (slope, shift) (`_w_scale_for`, checked with npuc_multiplier).
  Camera ADDs (all 124, quant 1 weights up to 255): worst case 0.03 to 0.11 LSB on average per model, 0.36 at most
  (EfficientDet's ratios reach 9).
- **One scale label for the concat** (section 3): NPUC sees a scale only through multipliers, so the operand whose
  producer is a conv read by this ADD alone gets the other operand's scale as its declared output scale, with its
  declared weight scale changed so that NPUC's (slope, shift) is the same (`_relabel`; its bytes do not change; the
  layer keeps its real scales in `scale_label`). When neither operand can (two pools, a tensor read by several
  layers), operand b goes through an identity 1x1 conv (weights 1, bias 0, multiplier exactly 256/2^8, output zero
  point = input's: the same bytes) declared at a's scale (`_copy_conv`). The network input, which NPUC refuses as a
  concat operand (harness-ops.md 3.2), is copied the same way first.
- **The zero points stay different**: the harness's `concat ... mixed` (npuc-net 2) skips its check that every concat
  input has the concat's quantisation; NPUC reads no concat zero point on this path.

The CPU reference is that conv's arithmetic (ref_conv), as for any conv: npunet's reference and ncpgen's NCP agree with
NPUC by construction; what changes against the model is the ADD's arithmetic (section 6).

## 5. What NPUC makes of it, and ncpgen's rules (V)

The concat of an ADD's operands is a concat whose input is often read by another layer too (the residual tensor: the
block input feeds the next block's first conv and the ADD). Rules found on `$NCPGEN_WORK/sweep/add-probe` (stream.py
`graph`, frame.py `plan_graph`; cmdq.md 7c):

- **Concat in SRAM** (NPUC's usual conditions, common.concat_in_sram): a producer read by other layers writes its
  planes into the CONCAT_STORAGE chunk all the same, and its other readers read them there (res-c16..c64, strided,
  relu, relu6, q3-c32: the stem's OFM MBLOCK at the storage's second segment, proj's IFM "parent <- 2"). A branch root
  (the input-sharing rule below) that is itself an input of the SRAM concat is never stored (chain-then-conv: the first
  ADD's conv writes into the second concat's storage at the top of SRAM, the next proj reads it there).
- **Concat in DRAM** (an input with channels not a multiple of 16): every input is stored into the concat's buffer, as
  before. When a stored input's layer is more than one node (a padded or strided conv) and another layer reads it, that
  reader's group loads it back from the concat's buffer at its channel offset, into the input's own chunk (NPUC:
  "Node 10, input DRAM is output of node 7", "disable forwarding"; res-c4, res-c8, res-c24, big-c24, chain3's stem), and
  NPUC still allocates the input a DRAM of its own: an IM vector no DMA uses, before the concat's, at its first store.
  A one-node layer (a 1x1 conv: chain3's ADD convs) is read in SRAM by the others ("enable forwarding"); as a branch
  root it is a first citizen at the top of SRAM, neither stored nor reloaded. (G for the rule's form: NPUC decides it
  in `Assembly::setForwardingFlag` from a "DRAM" tag on the edge, whose setting was not read; the mirror image of the
  stored branch root rule, which reloads a one-node root and shares a padded one.)
- **No concat scale folding**: with the labels of section 4, none of the 111 compiles logs "will update the scale".

| Set | Networks | ncpgen exact |
|---|---|---|
| `add-probe` (`add_sweep.py compile <dir>`): residuals over a computed tensor C = 4..64, two branches, chains of 3, ADD of pools, ReLU, ReLU6, x + x, 40x40, stride 2, quant 3 | 20 | 20 |
| `add-random` (`--set random --seed 1 --count 40`, unseen when the rules were fixed): 1..3 blocks of residuals, branch pairs and inverted-residual chains, 1 in 5 int8 (quant 3) | 40 | **40** at first sight |
| `add-random2` (`--seed 101 --count 40`, compiled after everything above) | 40 | **40** at first sight |
| `add-probe2` (`--set probe2`): residuals at 40..80 px, blocks expanding to 64 or 128 channels | 11 | 11 (the 2 NPUC's forwarding keeps out of SRAM since main's concat rules: section 7) |

`tests/golden.txt` holds the 100 NCP hashes of add-probe, add-random and add-random2 (rows `add:probe|random:<name>`);
`tests/regress.sh add-probe add-random add-random2 add-probe2`.

## 6. Error against TFLite's ADD, options ranked (`models/camera.py addcheck --all`)

TFLite's ADD (reference kernels, add.cc): each input shifted left 20, scaled by its own 31-bit multiplier, summed, then
rescaled: in practice the exact real sum rounded once (the exact sum differs from it on no value below). Measured on
the models' own activations (TFLM's person image resized, and seeded noise), each ADD alone (its TFLite inputs, the
three arithmetics' outputs against TFLite's) and at the cut tensor (the arithmetic's outputs propagated through the
rest of the model, TFLite's otherwise).

| model (cut: its last ADD reachable with tflref's ops) | ADDs | conv: each ADD alone | conv: at the cut, mean (max) | power-of-two + Eltwise: alone | its cut, mean (max) | NPUC's 8-bit conv multipliers alone, at the cut |
|---|---|---|---|---|---|---|
| SSD MobileNet v2 (t188, 10x10x160) | 10 of 10 | 0.296 % differ, max 1 | 58.6 % differ, 0.72 LSB (5) | 65.7 %, max 4 | 91.0 %, 3.60 LSB (20) | 67.3 %, 0.95 LSB (5) |
| SSDLite MobileDet (t308, 10x10x120) | 15 of 15 | 0.378 %, max 1 | 67.6 %, 1.03 LSB (9) | 62.6 %, max 3 | 93.1 %, 4.70 LSB (29) | 73.6 %, 1.33 LSB (10) |
| face SSD MobileNet v2 (t206, 10x10x160) | 10 of 10 | 0.300 %, max 1 | 42.3 %, 0.47 LSB (3) | 57.8 %, max 3 | 87.4 %, 2.44 LSB (20) | 60.7 %, 0.87 LSB (13) |
| SSDLite MobileNet v2, PINTO (from the QUANTIZE output, t243) | 10 of 10 | 0.219 %, max 1 | 69.5 %, 1.06 LSB (8) | 56.4 %, max 4 | 90.2 %, 3.36 LSB (21) | per-channel (not measured) |
| EfficientDet-Lite0 (from the QUANTIZE output, the backbone: t386) | 9 of 42 | 0.196 %, max 1 | 43.4 %, 0.49 LSB (4) | 63.4 %, max 3 | 87.6 %, 2.62 LSB (16) | per-channel |
| MoveNet Lightning (from the first conv's input, the backbone: t239) | 10 of 19 | 0.354 %, max 1 | 70.6 %, 1.11 LSB (9) | 64.3 %, max 3 | 92.8 %, 4.56 LSB (36) | per-channel |
| YOLO-Fastest 192 face v4 (t266, 6x6x48) | 18 of 18 | 0.149 %, max 1 | 63.0 %, 0.83 LSB (4) | 51.5 %, max 3 | 83.4 %, 2.09 LSB (18) | per-channel |

The exact real sum (real_add) equals TFLite's ADD on every value of every row (0 of 5.7 million). Inputs: the person
image and 1 seeded noise input (YOLO-Fastest, MoveNet: 2). "At the cut": the ADD arithmetic's outputs propagated
through every later layer: a 1-LSB change at 0.3 % of an ADD's values reaches most of a feature map a few blocks on
(each later conv mixes many inputs and flips some roundings); the same holds for NPUC's own 8-bit conv multipliers,
which move the cut further than the ADD lowering does (last column: every conv with NPUC's slope, the ADDs TFLite's;
per-tensor models only). Outputs: `$NCPGEN_WORK/camera-add/addcheck/`.

Options, by error against TFLite's ADD:

1. **Exact real sum, one rounding**: what TFLite computes; not available on the NPU (no multiplier with 31 bits).
2. **The conv over the concat (`--add-as-conv`)**: one requantisation of an exact accumulator, the ratios fitted to
   8-bit slope x 8-bit weights: 0.15 to 0.38 % of values off by 1 per ADD. Costs a concat (a DRAM round trip when the
   channels are not a multiple of 16) and a 1x1 conv over 2C channels per ADD.
3. **Power-of-two requantisation and NPUC's Eltwise** (each operand requantised by a conv to zero point 0 at a common
   2^-n that holds both ranges, the Eltwise at 2^-m, a conv back to the ADD's quantisation; three roundings, measured
   with exact multipliers, which flatters it): 51 to 66 % of each ADD's values differ, by up to 4; at the cut 2.1 to
   4.7 LSB on average, up to 36. It also needs the Eltwise's G arithmetic (no NPU run) and two more convs per ADD.
4. **CPU**: what Samsung does; exact, but splits the NPU part at every ADD.

Not done: an NPU run of the conv form (its arithmetic is a conv's, on the NPU so far for power-of-two multipliers
only: npunet's unproven list); the multiplier rounding of NPUC's 8-bit slope applies to it as to every conv.

## 7. The camera models with ADDs (`$NCPGEN_WORK/sweep/camera-add`, models/camera.md)

SSD MobileNet v2 and SSDLite MobileDet are the two whose NPU part starts with a conv: with `--add-as-conv` (and
`--output` per head, `--allow-none --relu6-range`) every head of both lowers, and so does each whole NPU part with its 12
heads as outputs (`--output` repeated, after main's several outputs: 92 and 123 layers). NPUC compiles SSD v2's box head
0 (t166, 57 layers), its last class head (t253, 81 layers, 10 ADDs) and the 12-head network (92 layers, 219 s, 15.5 MB,
root-3504c01f), and MobileDet's box head 0 (t0, 68 layers); MobileDet's later heads and its last ADD crash NPUC itself in its weight packer (NPULIRKernelPack::
tryBuildData, a memset: the dense depthwise 3x3 stride 2 over 768 channels, "numInputChannels 3072"), not in anything
of the ADD. ncpgen's weights are exact on the SSD v2 cuts; its streams refuse them ("a layer-block whose node only fits
with a weight ring": in a residual block's layer-block the dense 3x3 depthwise at 19x19x576 fits only with 18 weight
buffers; taking the forwarding test's ring there gets t166 generated, but groups 15 to 17 still differ: not modelled) and MobileDet's t0 ("a concat in SRAM NPUC's
forwarding refuses": the second ADD's concat at 80x80, whose block's 3x3 expansion to 128 channels leaves no room; the
case of graph-large2 t2022, frame.md 6.11). The other five models stop before their first ADD (PAD, a CPU head).

That last case reproduces on add-probe2 fx-80-c16-e128 and -r1 (NPUC: "Concat in SRAM: the optimization cannot be
performed for the block containing the node ... l2_exp"; at 64x64 it succeeds). **Exact since the merge with main's
camera-resolution work** (frame.md 6.11): such a concat is a DRAM concat (in stream.py and in weights.py's segment
rule, `frame.concat_cs_ok`), its reader's chunk a second citizen, and the expansion conv keeps the 2-of-8 weight ring
its input-sharing test passed with though the later modes fail on it (forwarding's "last"). SSD v2's t166 then gets
past the ring (no longer refused for it) but its groups 15 to 17 still differ and frame refuses a load of a tensor no
layer stores (not modelled).
