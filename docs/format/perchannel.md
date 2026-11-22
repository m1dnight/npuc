# Per-channel requantisation (NCP format, parts "grelu" and "stream")

Exynos 990 NPU, NCPs compiled by Samsung's NPUC 1.36.1.a (the oracle). Can NPUC compile a convolution whose weight
scales differ per output channel (TFLite's default for every int8 CONV_2D and DEPTHWISE_CONV_2D), where do per-channel
multipliers go in the NCP, and what can the NPU do with them. Code: `ncpgen/grelu.py` (`channel_requant`,
`table_slope`), `ncpgen/stream.py` (`emit_conv`: the shift words), `ncpgen/npunet.py` (`--per-channel-shifts`), the
harness's `wscales`/`slopes` lines and setScale hook (`oracle/harness/npuc-net.cpp`). Probes:
`$NCPGEN_WORK/sweep/perchannel-probe/` (`ncpgen/perchannel_sweep.py`).

Tags: **V** verified (bytes, log, code or an NPU run), **G** guess. Addresses are the libraries' own (Ghidra +0x100000).

## 1. Summary

- **NPUC compiles one multiplier per convolution** (V, code and oracle). Its on-device scale filler reads element 0 of
  the input, output and weight scale vectors and nothing else; given a per-channel weight scale vector it writes the
  NCP of the per-tensor scale `wScale[0]`, byte for byte (probe c16-ws = c16). The HAL in front of it reads only the
  operand's scalar `scale` (0 for a per-channel operand in NNAPI), so on the phone a per-channel model does not reach
  NPUC with its scales (V for that function, G for the rest of the HAL). **There is no per-channel path to turn on.**
- **NPUC's data model is per channel all the same** (V): the filler stores its one slope and shift C out times in per-
  channel vectors, and the backend gives channel c's slope to its DRU's slot of the GReLU table and its shift to its
  DRU's MTC shift word. With other values in those vectors (the harness's setScale hook), NPUC compiles them there, and
  ncpgen generates those NCPs exactly (14 of 14 probes: 1..3 passes, quant 1 and 3, ReLU and none, stride 2).
- **On the NPU the 16 slope slots are one piece table per pass** (grelu.md 6, V on the NPU): channels of different
  slopes cannot share a pass. Per-channel slopes are therefore not usable, which is also why NPUC's own table is wrong
  for small layers (its per-DRU design meets a per-pass table).
- **Each DRU's shift word is its own** (V in bytes): Samsung's stock NFD.ncp writes different shift words to the DRUs
  of 824 of its 840 conv passes (its FL shifts, bits 0..5). Whether a DRU applies its own bits 6..10 / 11..15 (the
  fields NPUC's on-device path writes) to its channel: **G**, no NPU run yet (section 6).
- So per-channel scales can be kept up to a power of two: one slope per layer, each channel's power of two in its
  DRU's shift word, the rest (a factor in (1/2, 1]) folded into its weights and bias. That is npunet's
  `--per-channel-shifts`: less than a bit of folding per channel instead of log2(R) (R = a layer's largest to smallest
  weight scale). person_detect: exact against NPUC both ways; against TFLite's own arithmetic, the logits differ by
  8.5 LSB on average with shifts and 87 with plain folding, which loses the person on TFLM's person image (section 5).

## 2. NPUC's code path (V)

| Step | Where | What |
|---|---|---|
| HAL -> ConvAttr | nn-eden-drv `NNNetworkBuilder::getScaleAndZeroPoint` @0x719fc | pushes `Operand.scale` into the vector when it is not 0, and `zeroPoint`; it reads no per-channel parameters (the rest of the HAL was not searched for them, G). A TENSOR_QUANT8_SYMM_PER_CHANNEL operand has scale 0 (NNAPI): wScale stays empty |
| the multiplier | frontend `NNGraphNodeScaleONFiller::setScale` @0x9da64 | `inScale.at(0)`, `outScale.at(0)`, `wScale.at(0)` (ConvAttr +0x130, +0x160, +0x190; `at` throws on an empty vector); m = f32(in x w) / out, doubled while < 0.5 (at most 7 times), slope = fcvtas(m x 256) (asserts <= 0xffff), shift = doublings + 8 (grelu.md 5) |
| per-channel storage | same function | a new NPUSlopeMemoryData (node +0x108 object, +0x110 control block; vector<uint32> at object +8) and a shift object (node +0x118 / +0x120; vector<uint8> at object +8), each filled with the one value numOutput (ConvAttr +0x58) times |
| the call | `NNGraphNodeScaleONFiller::run` @0x9d8e4 | `b setScale@plt` (@0x9d908): an executable defining the symbol replaces it (the harness hook) |
| to the DRUs | template, the per-DRU lambda (grelu.md 3) | for each populated DRU d of a pass: B[d] = its channel's slope, the shift word = (s << 11) \| (s << 6) of its channel's shift |

## 3. Where per-channel values land (V, the oracle)

The harness's `slopes <layer> <n> <slope> x n <shift> x n` line runs NPUC's setScale, then writes those values into the
two vectors (`npuc-net.cpp`, the setScale hook); `wscales <layer> <n> <scale> x n` gives the ConvAttr a per-channel
weight scale vector. Probes (`$NCPGEN_WORK/sweep/perchannel-probe/`, from `npuc-variant.py` convs on 16x16x8,
12x12x6 and 32x32x16):

| Probe | Per-channel input | NCP |
|---|---|---|
| c16-ws | wscales: 16 scales w0 x 2^(-c/4) | byte-identical to c16 (wScale[0] only) |
| c16-same | slopes: NPUC's own (128, 15) x 16 | byte-identical to c16 (the hook changes nothing by itself) |
| c16-sl | slopes 128..250, shift 15 | GReLU entries B[d] = channel d's slope (channel c -> DRU c for C out 16); slots 0, 1 zeroed by ReLU as always |
| c16-sh | slope 128, shifts 8..15 | MTC 0x40120900 + 4j: halves (s_c << 11) \| (s_c << 6) per DRU |
| c40-pc, c40q3-pc, c24s2-pc | 40 / 40 / 24 channels, slopes 128 + 3c, shifts 8 + (5c mod 8); quant 1 and 3, stride 2 | every pass's table and shift words by the channel -> DRU map of grelu.md 4 |
| c5n-pc | 5 channels, no activation | idem (the BYPASS table: no slot zeroed) |
| c40-sh | slope 128, shifts per channel | shift words only |

`ncpgen check` on all 14 (with their lowered.json carrying `slopes`/`shifts`): 14 exact.

## 4. ncpgen

- A lowered conv may carry `"slopes"` and `"shifts"` (C out each): NPUC's per-channel vectors. `grelu.channel_requant`
  gives them per channel (else the layer's one (slope, shift) from its scales).
- `--grelu npuc`: NPUC's per-DRU table, each DRU its channel's slope. `fixed` and `robust` (the corrected tables)
  need one slope per layer: `grelu.table_slope` refuses different slopes ("no table serves them").
- The shift words (`stream.emit_conv`): each populated DRU its channel's shift, `(s << 11) | (s << 6)`.

## 5. npunet: `--per-channel-shifts` and what it costs (person_detect)

For a conv or depthwise layer with per-channel weight scales w_c: w_max = max w_c; the layer's one (slope, shift) is
NPUC's for w_max; channel c gets shift + k_c with k_c = the largest k such that r_c = w_c 2^k / w_max > 1/2, and its
int8 weights and bias are multiplied by r_c and rounded (half away). Scales a power of two apart need no rounding at
all (the selftest). A shift over 31 (the 5-bit field) is refused; NPUC's own shifts stop at 15, ours reach 19 in
person_detect (G for the NPU). `--fold-per-channel` stays: r_c = w_c / w_max, every channel folded to w_max.

person_detect (TFLM, MobileNet v1 0.25, 96x96x1 int8, 28 per-channel layers, R median 3.7, max 121), against a
direct implementation of TFLite's integer arithmetic (`models/camera.py tflref`: QuantizeMultiplier per channel,
SaturatingRoundingDoublingHighMul, RoundingDivideByPOT; equal to TFLite's BUILTIN_REF kernels, ai-edge-litert 1.4.0,
on all 8 inputs tried), on TFLM's two test images (fed as its test feeds them: the pixel bytes as int8) and 4 random
inputs; the logits (the NPU part's output, int8, zero point -1; index 0 = no person, 1 = person):

| input | TFLite | folded | per-channel shifts |
|---|---|---|---|
| person.bmp | (-112, 110): person | (102, -102): **no person** | (-121, 119): person |
| no_person.bmp | (38, -39) | (64, -64) | (31, -32) |
| random 0..3 | (94, -93) (88, -87) (98, -97) (95, -94) | (12, -13) (19, -19) (23, -23) (37, -35) | (96, -94) (103, -101) (107, -105) (106, -104) |
| mean abs difference | | 86.8 LSB | 8.5 LSB |

The remaining 8.5 LSB are NPUC's 8-bit slopes (TFLite's multipliers have 31 bits) and the sub-bit folding, compounded
over 28 layers. (TFLite's optimised kernels differ from its reference kernels by up to 2 LSB on the same inputs.)

## 6. Proposed NPU run (settles the G above)

| Run | Net | Prediction | Settles |
|---|---|---|---|
| E6 | `npuc-variant.py --in 16,16,8 --conv 16,3` with per-channel shifts 12..15 (`slopes l1_conv1 16 128 x 16 12 13 14 15 ...`), ncpgen's robust table | bit-exact with npunet's reference (each channel its shift) | each DRU applies its own bits 6..15; then `--per-channel-shifts` is right on the NPU |
| E6b | if E6 fails: the extra shift in bits 0..5 instead (as Samsung's FL path writes per DRU), bits 6..15 the layer's shift | bit-exact with a double-rounding reference (G) | the FL pre-shift's rounding |
| E7 | person_detect `--per-channel-shifts` on TFLM's two images | npunet's reference byte for byte; person / no person as TFLite | the whole model |
