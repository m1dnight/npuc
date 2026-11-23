# Camera models

The public int8/uint8 TFLite models people run on cameras (person detection, object detectors, face, pose), as
published, and what ncpgen does with each. Nothing third-party is in the repository: `models/camera.py` holds the URLs,
sha256 and licences, fetches the files into `$NCPGEN_WORK/models/<name>/` and checks them; the compiled sets are
`$NCPGEN_WORK/sweep/camera/` (`tests/regress.sh camera`).

```bash
python3 -I models/camera.py list | fetch [model...]          # the catalogue; download + sha256 check
python3 -I models/camera.py tflref model.tflite in.bin out.bin [--stop TENSOR]   # TFLite's own integer arithmetic
python3 -I models/camera.py person                           # person_detect: npunet's reference against tflref
python3 -m ncpgen lower <model> <dir> [--per-channel-shifts | --fold-per-channel] [--allow-none] [--output TENSOR] [--relu6-range] [--add-as-conv]
python3 -I models/camera.py addcheck model.tflite [--input T] [--output T] [--all]  # ADD as a conv against TFLite's ADD
```

## The models (fetched 2026-11-22)

sha256 in `models/camera.py` (all URLs re-downloaded and compared). "R": a per-channel layer's largest to smallest
weight scale (median / max over the model). "NPU outputs": the tensors the NPU part hands over when everything from
the first CPU-only op on (RESHAPE or CONCATENATION across anchors, LOGISTIC, QUANTIZE/DEQUANTIZE, custom ops) stays
on the CPU. Op counts from the files.

| model | source, licence | input | ops (count) | weights | NPU outputs | `ncpgen lower` as published |
|---|---|---|---|---|---|---|
| **person_detect** (VWW, MobileNet v1 0.25) | tflite-micro `models/person_detect.tflite`, Apache-2.0 | 96x96x1 int8 | DW 14, CONV 14, AVG_POOL 1, RESHAPE 1, SOFTMAX 1; RELU6 (no-op clamps) | int8 per-channel, R 3.7 / 121; first DW multiplier 8 on 1 channel | 1 | **PASS** (29 NPU layers) with `--per-channel-shifts` or `--fold-per-channel`, and `--allow-none` (the logits) |
| SSD MobileNet v1 COCO quant | google-coral/test_data, Apache-2.0 | 300x300x3 uint8 | CONV 34, DW 13, RESHAPE 13, QUANTIZE 10, CONCAT 2 (axis 1), DEQUANTIZE 2, LOGISTIC 1, TFLite_Detection_PostProcess 1 | uint8 per-tensor, 46 of 47 zero points != 128 | 12 | 4 outputs (one supported); with `--output`: every head lowers (`--relu6-range --allow-none`) |
| SSD MobileNet v2 COCO quant | google-coral/test_data, Apache-2.0 | 300x300x3 uint8 | CONV 55, DW 17, RESHAPE 13, ADD 10, QUANTIZE 10, CONCAT 2, DEQUANTIZE 2, LOGISTIC 1, PostProcess 1 | uint8 per-tensor, zero points != 128 | 12 | 4 outputs; with `--output` and `--add-as-conv` every head lowers (ADD below) |
| SSDLite MobileDet COCO (Frigate's default CPU model) | test_data release-frogfish, Apache-2.0 (Frigate's `docker/main/Dockerfile`: `cpu_model.tflite`) | 320x320x3 uint8 | CONV 66, DW 27, ADD 15, RESHAPE 13, CONCAT 2, LOGISTIC 1, PostProcess 1; fused NONE everywhere (ReLU6 in the output ranges), 5x5 kernels | uint8 per-tensor, 91 of 93 zero points != 128 | 12 | 4 outputs; with `--output` and `--add-as-conv` every head lowers |
| SSDLite MobileNet v2 (no official quantised release: PINTO_model_zoo 006, MIT) | PINTO 006 tarball (`ssdlite_mobilenet_v2_coco_300_full_integer_quant.tflite`, sha256 2ece88ad...) | 300x300x3 **float32** (QUANTIZE inside) | CONV 55, DW 33, RESHAPE 13, QUANTIZE 11, ADD 10, CONCAT 2, DEQUANTIZE 2, LOGISTIC 1 | int8 per-channel, R 15 / 5e8 (dead channels) | 12 | 2 outputs; float input |
| EfficientDet-Lite0 | Coral `efficientdet_lite0_320_ptq.tflite` and MediaPipe int8/1 (same graph), Apache-2.0 | 320x320x3 uint8 -> QUANTIZE int8 | CONV 102, DW 80, ADD 42, MAX_POOL 14, RESIZE_NEAREST_NEIGHBOR 12, RESHAPE 10, CONCAT 2, DEQUANTIZE 2, LOGISTIC 1 (+ PostProcess in Coral's) | int8 per-channel, R 7.1 / 2e5 | 10 (largest 40x40x810) | 4 / 2 outputs; then RESIZE_NEAREST_NEIGHBOR |
| BlazeFace short range | MediaPipe float16/1, Apache-2.0 | 128x128x3 float32 | float16 weights (74 DEQUANTIZE), PAD 11, RELU 17, ADD 16 | **float only** (no quantised release) | 2 | 2 outputs; a float model |
| face SSD MobileNet v2 | google-coral/test_data, Apache-2.0 | 320x320x3 uint8 | CONV 55, PAD 22, DW 17, RESHAPE 13, ADD 10, QUANTIZE 10, CONCAT 2, DEQUANTIZE 2, LOGISTIC 1, PostProcess 1 | uint8 per-tensor, zero points != 128 | 12 | 4 outputs; then PAD |
| PoseNet MobileNet v1 0.75 | google-coral/project-posenet (`..._353_481_quant_decoder.tflite`), Apache-2.0 | 353x481x3 uint8 | CONV 18, DW 13, CONCAT 1 (channels), PosenetDecoderOp 1; fused NONE (ReLU6 in the output ranges) | uint8 per-tensor, 7 of 31 zero points != 128 | 3 (23x31x17/34/64) | 4 outputs |
| MoveNet Lightning | google-coral/test_data (`movenet_single_pose_lightning_ptq.tflite`), Apache-2.0 | 192x192x3 uint8 without quantisation (CAST, QUANTIZE, SUB, MUL) | CONV 50, DW 24, ADD 19, RESIZE_BILINEAR 3, MUL 7, SUB 6, GATHER_ND 4, ARG_MAX 2, ... | int8 per-channel, R 8.3 / 6e4 | 4 (48x48) | the input has no quantisation |
| YOLO-Fastest face 192 v4 | emza-vs/ModelZoo, Apache-2.0 | 192x192x1 int8 | CONV 56, DW 28, ADD 18, PAD 5, MAX_POOL 3, CONCAT 2 (channels), RESIZE_NEAREST_NEIGHBOR 1; RELU6 or none (no leaky ReLU, no LOGISTIC) | int8 per-channel, R 4.7 / 2389 | 2 (6x6x18, 12x12x18) | 2 outputs; then PAD |
| Tiny YOLO v2 224 (leaky ReLU) | STMicroelectronics stm32ai-modelzoo (Git LFS), **SLA0044** (redistribution restricted to ST devices) | 224x224x3 uint8 -> QUANTIZE | CONV 9, LEAKY_RELU 8 (alpha 0.2), MAX_POOL 6, QUANTIZE 1, DEQUANTIZE 1 | int8 per-channel, R 9.8 / 2e5 | 1 | QUANTIZE (a CPU head); then LEAKY_RELU |

No quantised YOLO with leaky ReLU and a permissive licence was found as a published .tflite (Arm ML-zoo's tiny YOLO
v3 is float; Ultralytics' exports use SiLU); YOLO-Fastest (ReLU6) is the YOLO-family model that fits best.

## What they need, ranked

Counted over the 11 quantised models (EfficientDet-Lite0's two copies once; BlazeFace, float, aside). Done items
first in their rank.

| # | feature | models needing it | status |
|---|---|---|---|
| 1 | **several outputs** (the detection heads) | 9: SSD v1, SSD v2, MobileDet, SSDLite, EfficientDet, face SSD, PoseNet, MoveNet, YOLO-Fastest | the other agent's work. Each head alone now lowers and compiles (`--output`) |
| 2 | **the CPU tail** (stop before RESHAPE/CONCAT across anchors, LOGISTIC, DEQUANTIZE, the post-process op) | 9: the 8 with a post-process or decoder + Tiny YOLO v2's DEQUANTIZE | **partly done**: `--output TENSOR` cuts the model at one tensor (tflite_cut). What remains is lowering, not hardware: cut at the set of tensors that feed the first CPU-only op (with several outputs) and keep the rest as a CPU tail graph (npunet's cpu_tail runs only RESHAPE and SOFTMAX today; LOGISTIC, CONCATENATION along an axis, DEQUANTIZE and TFLite_Detection_PostProcess would be host code) |
| 3 | **ADD with any scales and zero points** | 7: SSD v2, MobileDet, SSDLite, EfficientDet, face SSD, MoveNet, YOLO-Fastest (none of their 124 ADDs has power-of-two scales or zero points NPUC's Eltwise could take; 2 of MoveNet's broadcast) | **done, opt-in** (`--add-as-conv`, a change of the model): NPUC has no form of it (the stock HAL never hands it an ADD; its Eltwise only shifts); npunet lowers it to a concat of the two inputs and a 1x1 conv over it (exact accumulator, one requantisation): **0.15 to 0.38 % of each ADD's values off by 1 LSB** against TFLite's ADD, never more (section "ADD with any scales"; docs/format/add.md). NPUC compiles it, ncpgen exact on 20 probes and 80 of 80 unseen random networks |
| 3 | **resolution over 224** (tiling inside a graph) | 7: SSD v1/v2 (300), MobileDet, EfficientDet, face SSD (320), SSDLite (300), PoseNet (353x481) | ncpgen follows NPUC's graph tiling; SSD v1's heads: section "SSD detectors" |
| 5 | **per-channel weight scales** | 6: person_detect, SSDLite, EfficientDet, MoveNet, YOLO-Fastest, Tiny YOLO v2 | **done**: NPUC has no per-channel multiplier and the NPU no per-channel slope; `--per-channel-shifts` keeps each channel's power of two in its DRU's shift word (unproven on the NPU), `--fold-per-channel` folds all (docs/format/perchannel.md) |
| 6 | **uint8 weight zero points other than 128** | 5: SSD v1, SSD v2, MobileDet, face SSD, PoseNet | **done**: NPUC compiles any (MTC 0x1b0's low half), ncpgen exact on 20 probes (`$NCPGEN_WORK/sweep/wzp-probe`, quant 1); npunet takes them in quant 1 (unproven on the NPU: every run had 128) |
| 7 | a CPU head (QUANTIZE of the input, float input, CAST/SUB/MUL preprocessing) | 4: SSDLite, EfficientDet, MoveNet, Tiny YOLO v2 | not done: a requantisation of the input on the host before the NPU part |
| 8 | upsampling (RESIZE_NEAREST_NEIGHBOR, RESIZE_BILINEAR) | 3: EfficientDet (12 x2 nearest), MoveNet (3 bilinear), YOLO-Fastest (1 nearest) | not done (NPUC's deconvolution crashed in the harness; a nearest x2 could be a stride-1 conv on a zero-stuffed map: G) |
| 9 | PAD | 2: face SSD (22, pads 1 1 1 1), YOLO-Fastest (5, bottom/right) | not done: a PAD whose only reader is a VALID conv is that conv with explicit pads (NPUC's ConvAttr knows "EXPLICIT" pads; the harness takes SAME/VALID only) |
| 9 | RELU6 clamping inside the output range | 2: SSD v1 (30 of 35 layers: TF1 QAT ranges like [-25, 31]), YOLO-Fastest (1) | **adaptation**: `--relu6-range` requantises such an output to [0, 6] (scale 6/255), the clamp becoming the saturation (a change of the model's rounding); NPUC's on-device path has no bounded ReLU (grelu.md 7) |
| 11 | leaky ReLU | 1: Tiny YOLO v2 | not done (grelu.md 6: a fractional slope on piece 0, needs NPU runs) |
| 11 | depthwise multiplier > 1 | 1: person_detect (TFLite's form of a 1-channel CONV_2D) | **done**: npunet lowers any multiplier to the dense conv |
| - | LOGISTIC, HARD_SWISH on the NPU | 0: every LOGISTIC is in a CPU tail, no HARD_SWISH | - |

What each model still lacks after this work: person_detect nothing; **SSD v1, PoseNet and (with `--add-as-conv`) SSD
v2 only several outputs and the CPU tail** (and, for ncpgen's exactness, the graph work at their resolution: below);
MobileDet the same, and past its 20x20 stage NPUC itself crashes (below); face SSD PAD; SSDLite a CPU head;
EfficientDet upsampling, a CPU head; MoveNet upsampling, a CPU head; YOLO-Fastest upsampling, PAD; Tiny YOLO v2 a CPU
head and leaky ReLU.

## person_detect: exact as published

`python3 -m ncpgen lower person_detect.tflite <dir> --per-channel-shifts --allow-none` (or `--fold-per-channel`), then
`python3 -m ncpgen gen <dir>/lowered.json person_detect.ncp` (815744 B, the corrected GReLU table; `--grelu npuc` gives
NPUC's bytes, sha256 ca24eec5...):
29 NPU layers (14 depthwise as dense convs, the first with multiplier 8; 13 pointwise; the 3x3 average pool; the 1x1
logits conv without activation), RESHAPE and SOFTMAX on the CPU. NPUC compiles both forms (the shifts form with the
harness's setScale hook, root-40bb3e3f) and `ncpgen check` is **exact** on both (`$NCPGEN_WORK/sweep/camera/
person_detect-{fold,shifts}`). The CPU reference against TFLite's own arithmetic (`camera.py person`; tflref equals
TFLite's BUILTIN_REF kernels on these inputs):

| input | TFLite logits | folded | per-channel shifts |
|---|---|---|---|
| person.bmp | (-112, 110): person | (102, -102): **no person** | (-121, 119): person |
| no_person.bmp | (38, -39) | (64, -64) | (31, -32) |
| 4 random | (94, -93) (88, -87) (98, -97) (95, -94) | (12, -13) (19, -19) (23, -23) (37, -35) | (96, -94) (103, -101) (107, -105) (106, -104) |
| mean abs difference | | 86.8 LSB, top-1 5 of 6 | 8.5 LSB, top-1 6 of 6 |

Folding loses up to 6 bits in the layers whose R reaches 121 (the first conv and the first depthwise) and with them
the person; per-channel shifts lose less than a bit per channel, and the remaining difference is NPUC's 8-bit slope
over 28 layers. Use `--per-channel-shifts` (G on the NPU until perchannel.md's run E6; then E7 runs the model).

## SSD detectors

SSD MobileNet v1 (Coral's, TF1 quantisation-aware, uint8) is the closest: its NPU part is the MobileNet v1 trunk at
300x300, four extra feature layers and 12 head convs without activation (box encodings, 4 per anchor, and class
scores, 91 per anchor, at 19x19, 10x10, 5x5, 3x3, 2x2 and 1x1); everything after them is CPU work: a RESHAPE per
head, CONCATENATION across the six levels (axis 1), LOGISTIC on the scores and TFLite_Detection_PostProcess (anchor
decoding and NMS).

- **Each head lowers alone**: `ncpgen lower ssd_mobilenet_v1_coco_quant_postprocess.tflite <dir> --output <head>
  --relu6-range --allow-none` (`--relu6-range`: 30 of its 35 RELU6 clamp inside their ranges, e.g. at 140 with zero
  point 113; its weight zero points 163, 132, ... now pass). All 12 heads lower (24 to 36 NPU layers).
- **NPUC compiles them** (root-40bb3e3f; 137 to 204 s each alone, so run.sh's 300 s limit bites when several run at
  once): `$NCPGEN_WORK/sweep/camera/ssd-mobilenet-v1-t99` (box head 0, 24 layers), `-t102` (class head 0, 273
  channels), `-t168` (the last class head: the whole trunk and the extra layers, 36 layers, 11.8 MB NCP). NPUC's logs are gzipped there (675 MB for
  t168).
- **ncpgen**: the weight blocks are exact on all three (uint8 weight zero points, the 1024-channel depthwise layers as
  dense convs); the streams and frame refuse them: NPUC tiles the 150x150 and 75x75 layers (nodes 1..8; node 7 in 2
  row tiles, node 8 in 3) where ncpgen's plan keeps them whole ("NeedsTiling: SRAM 278528 B per core > 256 KiB";
  29 groups planned against NPUC's 37 for t99, 44 against 58 for t168): the tiling of a chain at 300 px, the
  big-models work (frame.md 6.11). `tests/regress.sh camera` lists them as its 3 known misses.
- **For the whole detector**: several outputs (the 12 heads in one NCP: the other agent's work) and a lowering change
  for the tail: `tflite_cut` takes one tensor; it should take the set of tensors read by the first CPU-only ops (here
  the 12 heads), lowered.json then carrying several NPU outputs and a CPU tail graph (RESHAPE, CONCATENATION on an
  axis, LOGISTIC, DEQUANTIZE and the post-process with its options and anchors) run on the host (npunet's cpu_tail
  runs RESHAPE and SOFTMAX only today).

SSD MobileNet v2, the face SSD and Frigate's MobileDet add residual ADDs (rank 3 above; and PAD for the face SSD);
PoseNet (353x481, 3 outputs, a decoder op on the CPU) needs exactly what SSD v1 needs.

## ADD with any scales

None of the 124 ADDs of the seven models can be NPUC's Eltwise (it only shifts by powers of two and has no zero-point
term; every one needs zero points other than u8 128), and Samsung's own HAL never gives NPUC an ADD: it runs them on
the CPU (docs/format/add.md 2). `--add-as-conv` (opt-in) lowers each to a channel concat of its inputs and a 1x1 conv
over it that weighs them by fitted integers and takes the zero-point difference into its bias: one requantisation of
an exact accumulator.

```bash
python3 -m ncpgen lower ssd_mobilenet_v2_coco_quant_postprocess.tflite <dir> --output 253 --add-as-conv --allow-none --relu6-range
python3 -I models/camera.py addcheck <model.tflite> [--input T] --output T --all      # the error against TFLite's ADD
```

| model | lowers with `--add-as-conv` | NPUC (root-81d28366) | `ncpgen check` |
|---|---|---|---|
| SSD MobileNet v2 | every head alone (`--output` 166..253; 57 to 81 NPU layers, all 10 ADDs from t188 on) | compiles: t166 (box head 0, 57 layers, 39 s), t253 (the last class head, the whole network, 81 layers, 111 s, 13.5 MB) | not yet: "a layer-block whose node only fits with a weight ring" (t166; the big-models work, frame.md 6.11) |
| SSDLite MobileDet | every head alone (68 to 101 layers), the last ADD (t308, 84 layers) | compiles t0 (box head 0, 68 layers, 51 s); t16, t80, t308 crash NPUC itself (SIGSEGV in NPULIRKernelPack::tryBuildData: the dense depthwise 3x3 stride 2 over 768 channels, "numInputChannels 3072") | not yet: "a concat in SRAM NPUC's forwarding refuses" (an ADD's concat at 80x80 NPUC keeps in DRAM; graph-large2 t2022's case, frame.md 6.11) |
| face SSD, YOLO-Fastest | refused at their first PAD | - | - |
| SSDLite (PINTO), EfficientDet-Lite0, MoveNet | refused at their CPU head (QUANTIZE, CAST) | - | - |

The ADD's arithmetic against TFLite's on the models' own activations (person image, noise; each ADD alone, and at the
cut with the conv's outputs propagated; `$NCPGEN_WORK/camera-add/addcheck/`; docs/format/add.md 6 has the power-of-two
alternative and NPUC's own conv multipliers for scale):

| model (ADDs measured) | each ADD alone | at the cut: values differing, mean (max) |
|---|---|---|
| SSD MobileNet v2 (10 of 10) | 0.296 % off by 1 | 58.6 %, 0.72 LSB (5) |
| SSDLite MobileDet (15 of 15) | 0.378 % off by 1 | 67.6 %, 1.03 LSB (9) |
| face SSD MobileNet v2 (10 of 10) | 0.300 % off by 1 | 42.3 %, 0.47 LSB (3) |
| SSDLite MobileNet v2 PINTO (10 of 10) | 0.219 % off by 1 | 69.5 %, 1.06 LSB (8) |
| EfficientDet-Lite0 (9 of 42, the backbone) | 0.196 % off by 1 | 43.4 %, 0.49 LSB (4) |
| MoveNet Lightning (10 of 19, the backbone) | 0.354 % off by 1 | 70.6 %, 1.11 LSB (9) |
| YOLO-Fastest 192 face v4 (18 of 18) | 0.149 % off by 1 | 63.0 %, 0.83 LSB (4) |

TFLite's ADD equals the exact real sum rounded once on every one of these values, so these are the lowering's own
differences. A 1-LSB change at 0.3 % of an ADD's values spreads through the later convs; NPUC's 8-bit conv multipliers
alone move the same cuts more (SSD v2 0.95, MobileDet 1.33, face SSD 0.87 LSB mean). The best power-of-two scheme
through NPUC's Eltwise would differ on 51 to 66 % of each ADD's values (up to 4 LSB; 2.1 to 4.7 LSB mean at the cut).
