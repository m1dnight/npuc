# Depthwise convolutions (NCP format, part "depthwise")

How Samsung's NPUC 1.36.1.a (the host oracle, `oracle/npuc-local.sh`) compiles a depthwise convolution, which the
stock Android HAL never handed it ("not yet supported" in nn-eden-drv), and what ncpgen needs to emit the same NCP.
Measured with the harness's `dwconv` line (`oracle/harness/npuc-net.cpp`, "npuc-net 2", section 2) and read in
NPUC's code (Ghidra 12.2, `oracle/ghidra/targets-depthwise.txt`; addresses are objdump's, Ghidra's are + 0x100000).

Tags: **V** verified (bytes, NPUC's log or code read), **G** guess. Sweep tool: `oracle/ops/dwsweep.py` (outputs in
`$NCPGEN_WORK/n3/dw/s1` (grid, blocks, mnv1) and `s2` (random), git-ignored); harness root root-fd659af2
(npuc-net-acc7a456), which gives the same NCPs as root-314e61a6 on every older input tried (`oracle/ops/check-ops.sh`
31/31 with the 20 earlier cases' hashes unchanged; 66 of 66 stored NCPs of corpus, graph-random2, random and tiling3).

## 1. Summary

- **NPUC compiles depthwise convolutions** (multiplier 1): a Caffe "Convolution" whose ConvAttr has group = C (V).
- **On the device path ("stock") it compiles them as a regular convolution with the dense C x C diagonal kernel**: the
  NCP is byte for byte the NCP of the same layer written as a regular conv whose kernel holds the channel's taps where
  output = input channel and the weight zero point elsewhere (V: 78 of 78 networks of the grid, block and MobileNet
  sets, and 40 of 40 unseen random networks; section 4). NPUC's weight format already skips zeros (a bitmap per
  chunk, weights.md "The layout" item 7), so the diagonal costs bitmap bytes, not C^2 weights.
- **ncpgen already writes those NCPs**: `ncpgen.py check` on the dense form: 78/78 whole files exact, the full
  MobileNet v1 0.25 on 96x96x1 (quant 1 and 3) included, and 39/40 on the unseen random set (V). The miss is a gap in
  weights.py's existing sub-batch rule, not a depthwise matter (section 4). Supporting depthwise in ncpgen is a
  lowering job in npunet.py (section 6), no format work: **done** (`npunet.depthwise_dense`, TFLite DEPTHWISE_CONV_2D
  and npu-net/1 `depthwise_conv2d`, multiplier 1). The zoo's MobileNet v1 0.25 and v2 0.35 (models/zoo.py) compiled
  by NPUC from their depthwise form give their `-dense` NCPs byte for byte; v1 is exact in ncpgen (V).
- **NPUC also has a real depthwise path ("DW opt")**, used for Caffe models off the device: the frontend takes it only
  when the layer's framework string is "scaffe", and the on-device frontend's is always "stensorflow". Through the
  harness's hook it compiles: mostly the same instruction count, a smaller weight block (a 16-input-channel conv per
  pass) and three conv register families changed (section 5). An optimisation for later; never produced by the stock
  NPUC, never run on the NPU (G that the firmware accepts it).

## 2. The harness line (V)

```
dwconv <name> <input> <stock|dwopt> <kh> <kw> <sh> <sw> <SAME|VALID> <act 0|1>
       <in scale hex> <in zp> <w scale hex> <w zp> <out scale hex> <out zp> <weights: C x kh x kw uint8, hex> <C> <bias x C>
```

"npuc-net 2" only. C = the input's channels = the output's (multiplier 1), C >= 2. The weights are per channel,
[c][ky][kx]. npuc-net builds what nn-eden-drv's ConvAttr::setKernelInfo would for a non-"Conv2D" attribute (group =
numInput = numOutput = C, attrType "DepthwiseConv2dNative", which NPUC never reads) and hands KernelONFiller the kernel
NPUC's own off-device filler makes of a depthwise one: dense C x C x kh x kw, the taps on the diagonal, the weight zero
point elsewhere (a real 0; quant 3 stores it as 0). Mode `stock` = what NPUC does on the device; `dwopt` = npuc-net
defines NNGraphAttribute::DwOptSupported (frontend and template call it through their PLTs, so the executable's
definition binds first, as for the Eltwise hook) and lets NPUC's own function run with the framework string set to
"scaffe" for that layer only (NPUC's other conditions still decide). Refused before NPUC: an "npuc-net 1" file, C = 1,
an unknown mode, a wrong weight count. net.json gets `"group"` and `"dw_mode"` for these layers.
`oracle/ops/check-ops.sh` runs 3 stock/dense pairs (the NCPs must be equal), a dwopt case and 4 refusals.

## 3. NPUC's code (V, read; described, not copied)

| Where | What it does with group |
|---|---|
| nn-eden-drv `ConvAttr::setKernelInfo` @0x17330c | attrType "Conv2D" or "Conv2DBackpropInput": group 1, numOutput = the filter's count; anything else: group = numOutput = numInput (multiplier 1 only). The HAL's builder only makes conv2d layers |
| `NNGraphAttribute::getConvolutionParam` @frontend 0x5049c | copies ConvAttr.group (+0x50) to the param (+0x2c) |
| `NNGraphAttribute::isDwConvolution` @0x53724 | asserts a Convolution node; true when kernel shape [0] == [1] (numOutput == numInput, KernelONFiller's {numOutput, numInput, kh, kw}), group == C out, C in == shape [1], group > 1 and C in == C out |
| `NNGraphNodeKernelONFiller::setKernel` @0x9cbb4 (on device) | copies the numOutput x numInput x kh x kw bytes as given (any length, no check): the caller must give the dense kernel |
| `NNGraphNodeKernelFiller::setKernel` @0x9fa20 (off device) | Caffe groups (asserts "ifmDepth * groupVal == input C", "ofmDepth * groupVal == output C"); for group == C out > 1 quantises with doQuantization_dwconv and **expands the kernel to dense C x C x kh x kw**, the tap where o == i, 0 elsewhere |
| `NNGraphCgObject::addConvolutionGraphNode` @0x66dd0 | the op's DW flag (param object +0x79) = isDwConvolution(); its group (+0x84) = param.group only when SocSupportsDwOpt(), DwOptSupported() and not (stride h > 2 and stride w > 2), else 1 |
| `SocSupportsDwOpt` @common 0x4f920 | CompileOptions.soc in 1..3 (2, Neus: true) |
| `NNGraphAttribute::DwOptSupported` @0x5333c | group > 1, framework string (NNGraphAttribute +0x18) == "scaffe", node type 8, kernel shape [0] == [1], both multiples of 16. The constructor (@0x4d8e0) sets "stensorflow"; no on-device code changes it: **always false on the device** |
| `NPU::Op::Convolution::getGroupVal` @template 0x159280 | returns 1 (a constant); isDwConvolution reads +0x79; no template code reads +0x79 directly |
| `PolicyMaker::makeCompressedKernel` @template 0x29c618 | with stride < 3, SocSupportsDwOpt and DwOptSupported: asserts a Convolution ("Deconvolution DW support is not supported yet!") and sets the compressed kernel's DW size to stride^2 x getChannelsPerDwBunch() (NPUDeviceNeus/2020: 16, @common 0x4ede4) |
| libnpuc_backend | nothing depthwise (no string, no ISA mode); the NCP's instructions are the conv's |

So the stock path sets only the DW flag, which changes nothing in the NCP (section 4); the group that would select
NPUC's depthwise packing stays 1.

## 4. Stock: a regular conv with the dense diagonal kernel (V)

`dwsweep.py` writes each network as npu-net/1 with every depthwise layer as a conv2d whose OHWI kernel is the dense
diagonal, lowers and compiles it (dense), then compiles the same net.npuc with those conv lines turned into
`dwconv ... stock` (and `dwopt`). **stock == dense, byte for byte, on every network that compiled:**

| Set | Networks | What |
|---|---|---|
| grid | 56/56 | one 3x3 layer, C 8, 16, 24, 32, 48, 64, 96, 128, 256 x stride 1/2 x SAME/VALID on 16x16; C 2, 3, 4, 12, 20, 40, 100, 512; for C 16/64: no ReLU, quant 1, 32x32, 5x5, 7x7 input with stride 2; C 32 on 48x48 stride 2 |
| blocks | 12/12 | dw 3x3 + pw 1x1 (C 8..256, stride 1/2, 12x12 / 24x24), conv + dw + pw, two blocks, quant 1 |
| mnv1 | 11/11 | MobileNet v1 0.25 on 96x96x1 (conv 3x3 s2 to 8, 13 dw/pw blocks to 256 channels, global average pool, 1x1 classifier to 2; ReLU, not ReLU6), quant 3 and 1, and 9 prefixes |
| random (unseen) | 40/40 | seed 20261122: single layers (C 2..300, sizes 3..40, k 1/3/5/7, stride 1/2, SAME/VALID, ReLU or not, quant 1/3) and 1..4-block dw/pw chains |

No NPUC abort in any of them. (npunet refuses a layer without fused ReLU unless `--allow-none`: its rule, not NPUC's.)

**ncpgen on the dense form** (`ncpgen.py check <set>/*/dense`, today's main branch): grid, blocks and mnv1 78/78
(streams, intrinsics, weights, whole files); random 39/40. The miss, r00-c42-31x31-q1 (7x7 SAME, C 42, quant 1): NPUC
packs one sub-batch of 12 channels per core (chunks of 588 positions, bitmap 74 B, lead byte 0x0d), weights.py two of
6 (lead 0x08), and the stream and frame follow. In quant 1 every off-diagonal byte is the zero point 128, which the
counter calls non-zero (weights.md "The layout" item 6), so a 12 x 49 walk passes 512 and weights.py cuts; NPUC does
not. G: NPUC's counter walks floor(C / 4) = 10 channels per core here (490 positions, no cut), as weights.md section
7 found for the planner's pass. A regular conv with such weights would miss the same way: a weights.py rule to fix
with a targeted sweep (large kernels, quant 1, mostly zero-point weights, C not a multiple of 8).

Cost of the dense form (V, weight bytes): C 64: 8192 (DW opt 4096); C 128: 24576 (8192); C 256: 81920 (16384); C 512:
327680 (32768); the whole MobileNet: 520448 (333056), NCP 800768 (613376). What the NPU spends on the zero taps
(cycles) is not known (G: the bitmap lets the DRUs skip them).

## 5. DW opt: what differs from the regular conv (V: grid, blocks, mnv1, random through `dwopt`)

Taken when C is a multiple of 16 (DwOptSupported) and stride <= 2. For C = 16 the weight block and registers are the
dense ones (one bunch), and so is the NCP except where the stride-2 planner sizing (item 4) moves SRAM chunks
(g-c16-7x7-s2). Compared with the dense NCP of the same network (`dwsweep.py` stock/dense vs dwopt):

1. **Weight block** = weights.py's packing of a *virtual* conv with 16 input channels (cpc 4 on 4 cores), C outputs,
   the same kernel size and stride, whose kernel gives output channel o its own taps at virtual input channel
   4 g + q, with g = o div gs, q = (o mod gs) mod 4, gs = 2 ceil(C / 8) (the output group size), and the zero point
   everywhere else. Exact on **162 of 162** DW-opt layers (every layer with C a multiple of 16 in the grid, blocks,
   mnv1 and random sets: C 16..512, stride 1/2, SAME/VALID, k 1/3/5/7 (157 of them 3x3), quant 1/3; the sub-batch
   split and R come out of the same packer). NPUC's log shows it: `kernel 3 x 3 numInputChannels 16` instead of C,
   chunks of 4 x 9 positions (bmSize 5).
2. **CMDQ, per output pass p** (the pass count 0x40120784 is unchanged):
   - MTC 0x40120240 + 4k (input channels per core) = the virtual conv's: 4, stride 2: 16 (dense: 2 ppc_in, 8 ppc_in);
   - MTC 0x40120300 + 8k + 4j = R << 16 | (WEIGHT + (2p + j) R) with the virtual block's R (2 for 3x3, 4 for 5x5);
   - MTC 0x401200a0 = ceil(Win / 8) << 16 | (input addr + p x (channels per core / 2) x input cells), input cells =
     MTC 0x40120180 (dense: the same address for every pass). E.g. 16x16 SAME: + 0x1e per pass (2 pairs x 15 cells);
     VALID: + 0x10; 32x32: + 0x5a; stride 2 (phase planes of 6 cells): + 0x30 (8 x 6).
   - unchanged: MTC 0x40120200 (sub-batches << 16 | channels: still the dense conv's, e.g. 0x10010 for C 64), the
     GReLU block, biases, shifts, the input core mask 0x40120044.
3. **Why it is right** (G, consistent with every byte): pass p's output lanes are g gs + 4p + q (weights.md "The layout"
   item 3) on output core g; input core g holds channels g cpc .. g cpc + cpc - 1 and cpc = gs, so the inputs those
   outputs need are core g's channels 4p .. 4p + 3, i.e. 2 channel pairs at pair offset 2p: hence the base step and 4
   channels per core, each output core reading its own input core.
4. **The plan**: NPUC's planner logs `InputBunchSize=16` (stride 1) or `64` (stride 2) for these layers (0 for
   a regular conv). Its compWeight (the WEIGHT chunk is a quarter of it) is the virtual block's size at stride 1 (C
   64: 4096, 128: 8192, 512: 32768; 5x5 C 64: 8192), but larger at stride 2 (C 16: 3072 for a 1024 B block, 32: 6144
   for 2048, 64: 10240 for 4096, 256: 40960 for 16384; not decoded: a 64-virtual-channel pack fits some of these, no
   single mapping all). The chunk sizes move the SRAM allocations: OFM addresses (0x40120700..) of the layer itself
   (g-c256-s1-valid, g-c16-7x7-s2) and of later layers, NFU 0x4016020c/0x40160210/0x40160304/0x40160308 in
   multi-layer networks; the weight io descriptors follow the block (h = 2R, offsets l x 2R x 64: weights.md "what the
   other parts need"); for C 256 with stride 2 and C 512 the instruction count changes too (4802 -> 4716, 6540 ->
   6380: the tiling/CMDQ split sees the smaller kernel; not decoded).

## 6. What ncpgen and npunet.py need

**Part 1, stock depthwise (small; the only one needed to run MobileNet-class models): done 2026-11-22**
- npunet.py: lower TFLite DEPTHWISE_CONV_2D (filter [1, kh, kw, C x m], depth_multiplier m, padding, strides,
  dilation 1, fused activation) and npu-net/1's own depthwise op to a lowered `conv` with cout = C and the dense
  diagonal OIHW kernel: w[o][i][y][x] = tap(o, y, x) when i == o, else the weight zero point (128 in npunet's u8
  form). Everything downstream (lowered.json, net.npuc's `conv` line, `ncpgen gen`) stays as it is: the NCP is NPUC's
  (V, section 4). Writing `dwconv ... stock` into net.npuc instead gives the same NCP (and needs "npuc-net 2"); the
  plain conv line keeps chains in "npuc-net 1". Refuse m > 1 (G: a dense kernel with o's taps at input o div m is
  the natural form, not compiled yet), dilation > 1, per-channel weight scales (as for CONV_2D: refuse or
  --fold-per-channel). ReLU6 (MobileNet's) is a separate question: the conv's fused activation is ReLU or none on
  this path (a standalone `relu ... ReLU6` layer compiles: harness-ops.md section 3.3).
- The CPU reference: the lowered layer is a regular conv, so `npunet.py ref` is already exact for it; a depthwise
  shortcut (skip i != o) only saves time (C x C x k^2 MACs per pixel otherwise).
- Tests: a selftest case (a TFLite-like depthwise graph -> the dense lowered conv), golden hashes for a few
  `dwsweep.py` networks (`tests/make-golden.sh` takes npuc-variant specs: it needs a depthwise spec form, or the
  hashes come from the dense nets directly).
- Effort: about half a day.

**Part 2, DW opt (optional, an optimisation; larger)**
- weights.py: the virtual 16-channel conv of section 5.1 (one function on top of conv_block).
- stream.py: the three register rules of section 5.2; frame.py: the plan with DW opt's compWeight (section 5.4:
  stride 1 known, stride 2 to decode), the io descriptors, and whatever tiling the smaller kernel changes for big C
  (not decoded: C 256 s2, C 512).
- Oracle: `dwconv ... dwopt`; the regressions are the dwsweep sets.
- A gate first: the stock NPUC never emits this form, so an NPU run must show the firmware computes it (G).
- Effort: 2 to 4 days plus the NPU run; worth it only if the dense form is too slow or too big on the NPU.
