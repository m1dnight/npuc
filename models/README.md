# The model zoo

Common small networks, written as ncpgen's input by `models/zoo.py`, and what ncpgen does with each: does
`npunet.py lower` accept it, does Samsung's NPUC (the oracle, `oracle/npuc-local.sh`) compile it, does
`ncpgen.py check` generate NPUC's whole NCP byte for byte, and does `npunet.py ref` give the CPU reference on a
random input. Nothing here has run on the NPU.

```bash
python3 -I models/zoo.py list [--md] [model...]          # the models
python3 -I models/zoo.py write <dir> [model...]          # <dir>/<model>.json (npu-net/1) [+ .tflite]
python3 -I models/zoo.py compile [--jobs N] [model...]   # lower + NPUC + reference -> $NCPGEN_WORK/sweep/zoo/<model>/
python3 -I models/zoo.py check [--jobs N] [--md] [--cached] [model...]   # ncpgen check on each: the table below
python3 -I models/zoo.py golden                          # tests/golden.txt rows (tests/make-golden.sh)
```

## How the models are made

- **Architectures** follow the published ones layer for layer (sources per model below), at the input sizes they are
  used at. **Weights are ours**: deterministic noise (shake_256 of the model family and the layer name; integer
  arithmetic only, so every host writes the same bytes) shaped like per-tensor int8 weights (-127..127, std ~37,
  ~1 % zeros). No pretrained or third-party weight file is used or needed.
- **Quantisation** is TFLite's int8 scheme: per-tensor weight scales chosen so every pre-activation has a float rms of 1
  (He-like), output scales covering about 4 sigma with a per-layer factor 0.75..1.5, ReLU outputs at zero point -128,
  ReLU6 at TFLite's 6/255. Activations therefore use their range on a random input (see the reference column).
- **Adaptations** are what someone deploying on this NPU would do, and each is listed with its model: batch norm folded,
  dropout removed, power-of-two activation scales and zero point 0 for the models with residual ADDs (quant 3: NPUC's
  Eltwise aligns by shifts and has no zero-point term), ReLU for activations the NPU path lacks (tanh, leaky ReLU),
  `--allow-none` for layers without activation (ncpgen's corrected GReLU table, its default, makes them right on the
  NPU; NPUC's own table is right only when ceil(Cout/8) is even: docs/format/grelu.md).
- **Variants.** `<model>-npu`: the published model rewritten by hand so NPUC compiles it, computing the **same
  function**. npunet now makes these rewrites itself (its stride rewrites, item 7): kws-cnn, magic-wand and alexnet-96
  lower to their `-npu` variant's network, which NPUC compiles to the same NCP, so the `-npu` variants remain as the
  cross-check (their CPU reference is the published network's byte for byte, `tests/rewrite_ref.py`). `<model>-dense`: each depthwise convolution written as the dense convolution with block-diagonal
  weights (the same function). NPUC itself compiles a depthwise layer exactly this way on the device path, its weight
  format skipping the zeros with bitmaps (docs/format/depthwise.md), and npunet now lowers DEPTHWISE_CONV_2D to it:
  the `-dense` variants remain as the cross-check (their NCP is their depthwise model's, byte for byte).
- **TFLite**: `write` adds `<model>.tflite` (`npunet.graph_to_tflite`: CONV_2D, DEPTHWISE_CONV_2D, MAX/AVERAGE_POOL_2D,
  FULLY_CONNECTED, ADD, CONCATENATION, RELU, RESHAPE, SOFTMAX and MEAN with their quantisation, as TFLite's converter
  writes them), kept when npunet lowers it to the same lowered network as the JSON (or refuses both alike): every one of
  the 24 (`tests/run.sh` checks it). That is the path a converted model takes.

## Status (2026-11-22, NPUC root-314e61a6)

**24 models: 23 exact, 1 fails in NPUC.** kws-cnn, magic-wand and alexnet-96 are exact as published since npunet
rewrites the strides NPUC refuses (item 7; compiled with root-fd659af2): each lowers to its `-npu` variant's network,
NPUC compiles it to that variant's NCP byte for byte, and its CPU reference is the published network's
(`tests/rewrite_ref.py`, against a direct reference of the original graph on random inputs). Every model has its CPU
reference (`npunet.py ref`, 0.1 to 18 s); the depthwise models give the reference of their `-dense` form. Every NCP
NPUC compiled is regression data in `$NCPGEN_WORK/sweep/zoo/` (`tests/regress.sh zoo`: every one exact), and the 23
exact ones are rows of `tests/golden.txt`, so `tests/run.sh` regenerates them with no vendor file. The NPUC column's
time includes npunet's lowering and was measured under a parallel load.

| model | input | options | lowering | NPUC | ncpgen check | CPU reference |
|---|---|---|---|---|---|---|
| lenet5 | 28x28x1 | --allow-none | ok, 7 NPU layers | PASS, 109 KB, 4.5 s | **EXACT** (0 s) | 895225fe26b4, 10 B, 0.1 s |
| keras-mnist-cnn | 28x28x1 | --allow-none | ok, 5 NPU layers | PASS, 67 KB, 4.4 s | **EXACT** (0 s) | bee31a89bef9, 10 B, 0.4 s |
| mnist-mlp | 1x1x784 | --allow-none | ok, 3 NPU layers | PASS, 852 KB, 4.7 s | **EXACT** (0 s) | cbfaca9fb6f4, 10 B, 0.4 s |
| tf-cifar-cnn | 32x32x3 | --allow-none | ok, 7 NPU layers | PASS, 168 KB, 5.5 s | **EXACT** (0 s) | e78c8338eb6e, 10 B, 0.5 s |
| vgg-cifar | 32x32x3 | --allow-none | ok, 11 NPU layers | PASS, 685 KB, 24.5 s | **EXACT** (3 s) | c5c997d7131b, 10 B, 3.8 s |
| resnet8 | 32x32x3 | --quant 3 --allow-none | ok, 17 NPU layers | PASS, 202 KB, 2.5 s | **EXACT** (0 s) | 29f02bf6832a, 10 B, 1.1 s |
| ad-autoencoder | 1x1x640 | --allow-none | ok, 10 NPU layers | PASS, 463 KB, 4.5 s | **EXACT** (0 s) | 533c72a18436, 640 B, 0.2 s |
| kws-cnn | 49x10x1 | --allow-none | ok, 5 NPU layers (stride 2x1 rewritten) | PASS, 116 KB, 10.6 s | **EXACT** (0 s) | 7a44e7c3827a, 12 B, 0.5 s |
| kws-cnn-npu | 49x10x1 | --allow-none | ok, 5 NPU layers | PASS, 116 KB, 3.2 s | **EXACT** (0 s) | 7a44e7c3827a, 12 B, 0.4 s |
| micro-speech | 49x40x1 | --allow-none | ok, 2 NPU layers | FAIL: (nzvSize & 0xFE00) == 0 | - | 0336eae2a362, 4 B, 0.1 s |
| magic-wand | 128x3x1 | --allow-none | ok, 6 NPU layers (pooling stride 3x1 rewritten) | PASS, 17 KB, 11.2 s | **EXACT** (0 s) | ddc82277175a, 4 B, 0.1 s |
| magic-wand-npu | 128x3x1 | --allow-none | ok, 6 NPU layers | PASS, 17 KB, 10.0 s | **EXACT** (0 s) | ddc82277175a, 4 B, 0.1 s |
| hello-world | 1x1x1 | --allow-none | ok, 3 NPU layers | PASS, 9 KB, 1.6 s | **EXACT** (0 s) | 2b4c342f5433, 1 B, 0.1 s |
| squeezenet11-96 | 96x96x3 | - | ok, 38 NPU layers | PASS, 1162 KB, 5.9 s | **EXACT** (1 s) | 4951801caa0a, 10 B, 4.5 s |
| alexnet-96 | 96x96x3 | --allow-none | ok, 12 NPU layers (stride 4 rewritten) | PASS, 4576 KB, 16.5 s | **EXACT** (3 s) | 6d9704d8711f, 10 B, 9.5 s |
| alexnet-96-npu | 96x96x3 | --allow-none | ok, 12 NPU layers | PASS, 4576 KB, 17.9 s | **EXACT** (43 s) | 6d9704d8711f, 10 B, 9.5 s |
| vgg11-64 | 64x64x3 | --allow-none | ok, 16 NPU layers | PASS, 3838 KB, 122.5 s | **EXACT** (14 s) | 5456870cc263, 200 B, 16.0 s |
| yolov2-tiny-128 | 128x128x3 | --allow-none | ok, 15 NPU layers | PASS, 4607 KB, 99.8 s | **EXACT** (20 s) | b4805006f2b0, 2000 B, 9.0 s |
| mobilenet-v1-025 | 96x96x1 | --allow-none | ok, 29 NPU layers | PASS, 797 KB, 45.6 s | **EXACT** (1 s) | 5af11588254e, 2 B, 5.1 s |
| mobilenet-v1-025-dense | 96x96x1 | --allow-none | ok, 29 NPU layers | PASS, 797 KB, 6.3 s | **EXACT** (1 s) | 5af11588254e, 2 B, 4.9 s |
| mobilenet-v2-035 | 96x96x3 | --quant 3 --allow-none | ok, 64 NPU layers | PASS, 3537 KB, 23.5 s | **EXACT** (9 s) | 0cc597e5b755, 1000 B, 11.9 s |
| mobilenet-v2-035-dense | 96x96x3 | --quant 3 --allow-none | ok, 64 NPU layers | PASS, 3537 KB, 24.4 s | **EXACT** (8 s) | 0cc597e5b755, 1000 B, 11.9 s |
| ds-cnn | 49x10x1 | --allow-none | ok, 11 NPU layers | PASS, 209 KB, 2.7 s | **EXACT** (0 s) | ad2c9fdec999, 12 B, 1.7 s |
| ds-cnn-dense | 49x10x1 | --allow-none | ok, 11 NPU layers | PASS, 209 KB, 22.2 s | **EXACT** (3 s) | ad2c9fdec999, 12 B, 2.0 s |

Per published model: LeNet-5, the Keras MNIST convnet and MLP, the TensorFlow CIFAR CNN, a VGG-style CIFAR CNN, the
DCASE/MLPerf Tiny autoencoder, the TFLM sine model, SqueezeNet 1.1, VGG-11, Tiny YOLOv2 and ResNet-8 are **exact as
published** (with the generic adaptations above; ResNet-8's SAME 1x1 stride-2 shortcuts, the vector models' `[1, N]`
inputs and SqueezeNet's MEAN as Keras writes them). The Hello Edge KWS CNN, magic wand and AlexNet are **exact as
published** too, through npunet's exact stride rewrites (item 7: their `-npu` networks, made automatically).
MobileNet v1 is **exact as published** (DEPTHWISE_CONV_2D and TFLite's ReLU6 quantisation: npunet lowers each depthwise
layer to the dense block-diagonal conv, and NPUC's NCP for the depthwise form is the `-dense` form's byte for byte);
MobileNet v2 lowers and compiles the same way and is exact too (with its adaptations; its MEAN as Keras writes it);
DS-CNN is **exact as published** too (its 25x5 global average pool as the identity conv, item 5). micro_speech **fails
in NPUC**.

## What ncpgen and npunet need, ranked

What unblocks the most models for the least work comes first. Items marked done have been applied since.

1. **Done (2026-11-22): ncpgen: NPUC's layer order for graphs** (`common.exec_order`; resnet8 exact, `tests/regress.sh`
   3330 of 3342 with only the known misses).
   resnet8 differs only because of it: NPUC runs each stage's shortcut 1x1 conv before the block's two 3x3 convs, and
   ncpgen runs it after them. NPUC's order depends on the layer order of the net file, **not** on the order of an
   add's inputs (V: swapping the add's inputs gives the same NCP; moving the shortcut line before the block changes
   it). The rule, **V on all 278 graph compiles of `$NCPGEN_WORK/sweep` with a NODE ORDER log** (the current rule: 276):
   number the layers breadth first from the network input over their readers, readers in layer-list order (this is
   how NPUC numbers its graph's nodes, `run/ir_graph_init.txt`), then repeatedly run the ready layer (all inputs done)
   with the highest number.
   ```python
   def exec_order(out, ins, cons, src):          # ins/cons in layer-list order, as stream.py and frame.py build them
       nid, q, seen = {}, [src], {src}
       while q:
           n = q.pop(0)
           for c in cons.get(n, []):
               if c not in seen:
                   seen.add(c); nid[c] = len(nid) + 1; q.append(c)
       done, order = {src}, []
       ready = [(-nid[n], n) for n in ins if n in nid and all(x in done for x in ins[n])]
       heapq.heapify(ready); queued = {n for _, n in ready}
       while ready:
           _, n = heapq.heappop(ready); order.append(n); done.add(n)
           for c in cons.get(n, []):
               if c not in queued and all(x in done for x in ins[c]):
                   queued.add(c); heapq.heappush(ready, (-nid[c], c))
       return order
   ```
   With both functions replaced in memory (no file changed): resnet8 is exact, and `ncpgen check` over all 392 graph
   compiles in `$NCPGEN_WORK/sweep` gives 385, the 7 misses being the known NPUC uninitialised-byte ones (graph rc4-*,
   graph-random2 g120/g134). **Effort: an hour** (the code above plus the regression run).
2. **Done (2026-11-22): npunet: RELU6 the way TFLite rounds it** (`npunet.relu6_top`: TFLite's `zp + round(6.0f /
   scale)`, float32 division rounded half away, from kernel_util.cc `Quantize` and activations.cc `QuantizedReluX`).
   TFLite's int8 ReLU6 output is scale 6/255
   with zero point -128, and TFLite clamps at `zp + round(6 / scale)` = 127, the saturation (a no-op). npunet tests
   `zp + 6 / scale >= 255` in float32, gets 254.999995 and refuses it. So every TFLite MobileNet (and EfficientNet-lite,
   MobileNet-SSD...) is refused at its first layer, before its depthwise layers. Fix: `zp + round(6 / scale) >= 255`.
   **Effort: minutes.**
3. **Done (2026-11-22): npunet: DEPTHWISE_CONV_2D** (`npunet.depthwise_dense`, TFLite and npu-net/1, multiplier 1,
   per-channel scales by `--fold-per-channel`; mobilenet-v1-025 exact, mobilenet-v2-035 = its `-dense` NCP, ds-cnn
   then stops at 5). Lower it to the dense convolution with
   block-diagonal weights, which is what NPUC itself does on the device path (the depthwise study under way,
   docs/format/depthwise.md). The `-dense` models show ncpgen exact on that form for MobileNet v1 and DS-CNN. Needed:
   a `depthwise_conv2d` op in npu-net/1 (the zoo writes it as TFLite stores it: layout 1HWO, `depth_multiplier` 1), the
   op in `tflite_to_graph`, and per-channel weight scales, which TFLite uses by default for depthwise weights. **Effort:
   half a day** without per-channel scales (folding them is lossy; see 9).
4. **Done (2026-11-22): frame.py: graphs whose large layers NPUC splits into several groups** (mobilenet-v2-035-dense:
   NPUC cuts the 112->1280 conv and the 1280->1000 FC into three groups each, cmdq.md 7d). `plan_graph` follows the
   stream part's groups (`stream.Plan.group_layers`) with the chain's cut descriptors (frame.md 6.8), and the stream
   part's graph planner took four more of NPUC's rules: the FC's weight ring slot when Cout is not a multiple of 16
   (what g65-g67 missed), META_DATA's chunk, a wide standalone ReLU's weight rows, no pass-by-pass store in a graph.
   Checked on `$NCPGEN_WORK/sweep/graph-split` (`stream_graphs.py --set split`: a cut layer in each place a graph puts
   one, a kernel over 4 MiB, 128..720-channel eltwise and ReLU layers; concat blocks at 96..160 px, which NPUC tiles
   and ncpgen refuses, and residual blocks there, which NPUC cannot compile: 15 of 18 compiled, the 3 tiled refused)
   and on unseen random wide graphs (`stream_graphs.py random <dir> 1 30 --set split`: 25 of 25 compiled exact; 5 more
   stop at npunet's limit on a wide global pool).
5. **Done (2026-11-22): npunet: a global average pool larger than 14** (ds-cnn: exact as published). The 14x14 limit
   is replaced by the measured one, `npunet.gap_conv_fits` (`models/gap_sweep.py`: 42 identity convs through NPUC, 26
   in `$NCPGEN_WORK/sweep/gaplim/`; the selftest holds every outcome). NPUC's packer cuts each core's cpc = 2 ceil(C/8)
   channels into sub-batches before 1017 positions or 512 values its counter calls non-zero (stored byte != 0: in
   quant 1 the zero point 128 counts, so every byte; in quant 3 only the diagonal's 1), each of an even number of
   channels, floor(cut / K / 2) * 2 for a kernel area K:
   - **quant 3**: areas up to 508 (V: 504 over 1, 8 and 64 channels, 500 over 96, 289 over 120 compile and are exact;
     510 and 512 assert `(bmSize & 0x80) == 0`);
   - **quant 1**: areas up to 255 (one channel pair per 511 values), except where the sub-batches exceed tryBuildData's
     maxPossibleBatches, which NPUC asserts (`_numBatches[file] <= maxPossibleBatches`: 10x10 over 36..40, 80, 120
     channels, 5x20 over 36; 10x10 over 32 and 9x9 over 40 compile; the old 14x14 rule lowered these). Past 255 the
     sub-batch would hold no channel pair: NPUC divides by zero, and its 0 sub-batches drop out of its maximum, so it
     packs one sub-batch per core. That compiles while a chunk's bitmap stays under 128 bytes (cpc x K <= 1016: 8
     channels or fewer, areas up to 508; V: 16x16, 20x20, 64x4, 63x8 over 8, 20x13 over 2) and asserts
     `(bmSize & 0x80) == 0` from 128 bytes (16x16 over 16). **NPUC's defect**: the assertion tests one bit, so from
     256 bitmap bytes (16x16 over 32 or 64 channels) it compiles a broken NCP, the size overflowing the chunk header's
     7-bit field (`$NCPGEN_WORK/zoo-evidence/gaplim-defect/`). npunet lowers the fallback cases NPUC compiles right;
     **ncpgen's weights.py refuses them** for now ("a sub-batch would hold no channel pair"): with `_min_batches`
     returning 0 there instead of raising (NPUC's division by zero), ncpgen is exact on all 5 and on the whole sweep
     (26 of 26, checked in memory, weights.py unchanged here). `tests/regress.sh` lists them (gaplim 5).
   - kernel sides up to 64: our harness (`npuc-net`) takes 1..64, so nothing larger is measured.
   micro-speech's FC (8) is a conv of random weights, not this identity conv: still lowered, still NPUC's assertion.
6. **Done (2026-11-22): npunet: TFLite patterns real converters write** (MEAN over H, W: `_mean_as_pool`, as the NFU
   pooling or, when that cannot take it (stride > 3, unequal strides, another output scale), as the identity conv
   whose multiplier rescales; `[1, N]` inputs; SAME with a negative pad: VALID, `_padding`, V on the 37 networks of
   the sweeps npunet refused for it before: 36 compile in NPUC and ncpgen is exact on every one, NPUC aborts on the
   last, frame/s18 r3-110, "numTiles > 0" on a 2-row map; they are regression data now. The zoo now writes
   mnist-mlp, ad-autoencoder and hello-world with `[1, N]` inputs, resnet8 with its SAME shortcuts, SqueezeNet and
   MobileNet v2 with MEAN: the same NCPs). None of these was in the zoo's JSON, but a converted model hits each
   one. MEAN over H and W (Keras' GlobalAveragePooling2D) should lower like a global AVERAGE_POOL_2D. A 2-D input
   `[1, N]` (mnist-mlp, ad-autoencoder, hello-world as Keras writes them) is the same tensor as `[1, 1, 1, N]`. SAME
   padding whose pad would be negative (resnet8's 1x1 stride-2 shortcuts: "SAME padding with stride 2 > kernel 1 leaves
   a negative pad") is VALID in TFLite, which clamps the pad at 0. **Effort: an hour or two for the three.**
7. **Done (2026-11-22): npunet: rewrite the strides NPUC asserts on, exactly, or refuse them** (`npunet.restride_plan`,
   `_lower_restrided`; kws-cnn, magic-wand and alexnet-96 exact as published). The strides NPUC refuses (`stride_issue`,
   V): conv strides over 2 or unequal; pooling strides over 3, larger than the kernel, or unequal (a larger vertical
   one asserts `absDiff < stride`; a larger horizontal one compiles but ncpgen's streams do not model it, 2 probes in
   `$NCPGEN_WORK/zoo-evidence/probe-pool/`). Such a conv, depthwise conv or pooling runs instead at the largest equal
   stride t NPUC takes that divides both strides, VALID or else SAME, whichever puts a window on every original one
   (output i of stride s and pad p reads the inputs from s i - p, output j of stride t and pad p' from t j - p', so
   i is j = (s / t) i + (p' - p) / t; windows over the same inputs give the same value, the border being the zero point
   in both and a pooling's window clipped alike). Then the positions the original keeps are selected:
   - **by the FULLY_CONNECTED that alone reads the layer** (directly or through a RESHAPE): its weights on the kept rows
     and columns, the weight zero point elsewhere. Every accumulator is the original's, so this is **exact for any
     quantisation, bias and weight scales** (per-channel too). kws-cnn (conv 2x1), magic-wand (pool 3x1), and in the
     sweep: conv 2x1 SAME, conv 1x3, conv 3 SAME, conv 4 SAME (rows 2, 6, 10, 14), pool 2x2 stride 3, average pool 3x1;
   - **for strides 2t, whatever reads it**: a 1x1 stride-2 VALID conv with identity weights, weight scale 1, bias 0 and
     its input's quantisation. NPUC's multiplier is then exactly 256 / 2^8 = 1 (npunet checks it and refuses
     otherwise), so out = zp + (x - zp) = x, and the clamp changes nothing. **Exact under ReLU only when the input is >=
     its zero point** (the output of a ReLU conv, or a pooling of one: ReLU, a no-op there); otherwise it needs 'none'
     (refused without `--allow-none`; right on the NPU with ncpgen's corrected GReLU table, NPUC's own only for
     ceil(C/8) even). alexnet-96 (conv 4), and in the sweeps: conv 3x3 / 4x4 stride 4, max and average pools 4x4
     stride 4, a 2x2 stride-4 pool on the network input (none), TF-slim's 1x1 stride-2 max pool (no pooling at all: a
     1x1 pooling at stride 1 is its input).

   NPUC refuses every other subsampling conv (V, `$NCPGEN_WORK/zoo-evidence/probe-stride`: a 1x1 identity conv after a
   3x3 conv on 16x16x8 asserts `_strideWidth == _strideHeight` at strides 2x1, 1x2 and 3x1 and "Stride > 2 is not
   supported here!" at 3 and 4; at 2 it compiles, ncpgen exact, ReLU or none), and a 1x1 pooling cannot subsample
   (stride > kernel). So npunet refuses, naming why: unequal strides not read by an FC alone (a 2x1 conv read by a
   conv), strides whose remaining step is not 2 (3, 5, a pool 8x8 stride 8: steps 4 and 8), and SAME strides 4 whose
   stride-2 windows do not start at the first position. A max pool with kernel = stride could also be split into a
   cascade of smaller max pools (exact: a max of maxes; not for average pools, rounded twice): not done.
   Proofs: npunet's selftest (each form's reference against the network lowered with its strides as they are,
   `lower(..., npuc_strides=False)`, on random inputs; the refusals), `tests/rewrite_ref.py` in `tests/run.sh` (the
   three published models and a network per form against a direct NHWC reference of the original graph written in the
   test, random inputs), and NPUC: the three models compile to the `-npu` variants' NCPs byte for byte, the 9 networks
   of `$NCPGEN_WORK/sweep/restride` (`tests/rewrite_ref.py --write`) and the 5 sweep networks npunet refused before and
   now rewrites (`$NCPGEN_WORK/sweep/restride-unseen`: poollim's 4x4 stride-4 pools, the weights sweep's 4x4 stride-4
   conv; nobody had looked at them for this) compile, and ncpgen is exact on all 17 (`tests/regress.sh`; the 12 of
   the zoo and `restride` are golden rows).
8. **micro-speech** (the FC over 25x20x8 as a 25x20 conv: `(nzvSize & 0xFE00) == 0`). Covering it needs the bitmask
   limit of 5 and then a split of the FC into partial convs whose partial sums are requantised (lossy). Low value for
   the effort (**a day**); refusing it cleanly comes with 5.
9. **Beyond the zoo**: leaky ReLU (Tiny YOLOv2's activation; the GReLU table could hold the slope, docs/format/grelu.md
   section 6, needs NPU experiments: days), per-channel weight scales (TFLite's default for every conv: NPUC's
   on-device filler takes one scale per layer, so this needs the per-DRU slopes NPUC fills anyway, studied on the
   NPU: days). Done (2026-11-22): `graph_to_tflite` for every op npunet lowers, so the zoo writes TFLite and the
   TFLite path is tested on real topologies (all 24 lower as their JSON; `tests/run.sh`).

## NPUC limits met here (V, the oracle)

| Layer | NPUC | Seen in |
|---|---|---|
| conv stride 4 (11x11, SAME or VALID) | asserts "Stride > 2 is not supported here!" | alexnet-96, probe 64x64x3 |
| conv stride 2x1 (10x4 VALID) | asserts `_strideWidth == _strideHeight` (40x7x28) or "Couldn't find a correct processing strategy for a single node" | kws-cnn |
| max pool 3x1 stride 3x1 (VALID or SAME) | asserts `absDiff < stride` | magic-wand |
| max pool 3x3 stride 3 SAME on a 42x1 map | asserts "Tried to split a tensor's shape into zero tiles" | probe |
| max pool 3x1 stride 3x3 on a 42x1 map | fails | probe |
| max pool 1x1 stride 2 | "The stride value should be equal or less than the size of kernel" | alexnet-96-npu, first try |
| 1x1 identity conv at strides 2x1, 1x2, 3x1 (after a 3x3 conv, 16x16x8) | asserts `_strideWidth == _strideHeight` (PNodeConvolution::toAtomicChain) | `probe-stride/` |
| 1x1 identity conv at stride 3 or 4 | asserts "Stride > 2 is not supported here!" | `probe-stride/` |
| 1x1 identity conv at stride 2, ReLU or none | compiles; ncpgen exact | `probe-stride/`, alexnet-96, `sweep/restride*` |
| conv 25x20 VALID over 8 (or 4) channels, 13x20 over 8 | asserts `(nzvSize & 0xFE00) == 0` | micro-speech, probes |
| conv 25x5/64, 16x4/30, 15x4/16, 14x14/16, 15x15/8, 25x10/8, 12x20/8, 14x18/8, 25x20/1, 49x1/8, 1x40/8 VALID | compile; ncpgen exact | ds-cnn-dense, kws-cnn-npu, probes |
| conv 16x16/8, 20x13/8 VALID (random weights) | compile; ncpgen refuses ("a sub-batch would hold no channel pair") | probes |
| global average pool as the identity conv: quant 3 area <= 508, quant 1 area <= 255 within maxPossibleBatches, or one sub-batch per core up to cpc x K = 1016 | compile; ncpgen exact but for quant 1's one-sub-batch cases (it refuses them) | gap_sweep (item 5) |
| identity conv 16x16 over 32 or 64 channels, quant 1 | compiles a **broken** NCP: a 256- or 512-byte bitmap passes `(bmSize & 0x80) == 0` and overflows the chunk header | gap_sweep |
| max pool 2x2 / 3x3 at strides 2x1, 3x1, 3x2 (16x16) | asserts `absDiff < stride` | probes (`probe-pool/`) |
| max pool 2x2 / 3x3 at strides 1x2 (16x16) | compiles; ncpgen's streams differ (unequal pool strides not modelled; npunet refuses them) | probes (`probe-pool/`) |
| max pool 2x2 stride 3 | refuses ("The stride value should be equal or less than the size of kernel") | 28 frame-sweep variants, probe |
| conv kernel side > 64 | not measured (our harness takes 1..64) | gap_sweep |
| max pool 2x2 stride 1 SAME, 3x3 stride 3 SAME/VALID on 128x3/129x3, 3x1 stride 1 on 42x1 | compile; ncpgen exact | yolov2-tiny-128, magic-wand-npu, probes |

The probes' networks (`probe-src/`, npu-net/1 JSON, our weights; `probe-stride/<probe>/model.json`) and NPUC's outputs
(`probe/`, `probe3/`, `probe-pool/`, `probe-stride/`, `gaplim-defect/`) are in `$NCPGEN_WORK/zoo-evidence/` (outside `sweep/`, so `regress.sh` does not take
them), with the resnet8 ordering variants (`order/`: add inputs swapped, shortcut first, both).

## The models

### lenet5

- **Source**: LeNet-5 (LeCun et al. 1998, Gradient-based learning applied to document recognition)
- **Task / input**: MNIST digits 28x28x1, 10 classes; input 28x28x1 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 5x5 x2, average_pool2d x2, reshape, fully_connected x3, softmax; 0.06 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: tanh/sigmoid -> ReLU; trainable subsampling -> 2x2 average pool; C3's partial connection table -> fully connected; the 32x32 padded input -> 28x28 with SAME on C1; C5 (a 5x5 conv over 5x5) and F6 as FULLY_CONNECTED; RBF output -> FC + softmax
- **Status**: exact (lowering: ok, 7 NPU layers; NPUC: PASS, 109 KB, 4.5 s; ncpgen check: EXACT (0 s))

### keras-mnist-cnn

- **Source**: Keras example 'Simple MNIST convnet' (keras.io/examples/vision/mnist_convnet)
- **Task / input**: MNIST digits 28x28x1, 10 classes; input 28x28x1 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3 x2, max_pool2d x2, reshape, fully_connected, softmax; 0.03 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: dropout removed (inference)
- **Status**: exact (lowering: ok, 5 NPU layers; NPUC: PASS, 67 KB, 4.4 s; ncpgen check: EXACT (0 s))

### mnist-mlp

- **Source**: Keras example mnist_mlp (Dense 512-512-10)
- **Task / input**: MNIST digits as a 784 vector, 10 classes; input 1x1x784 int8 (scale 0.00392157, zero point -128)
- **Ops**: fully_connected x3, softmax; 0.67 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: dropout removed (inference)
- **Status**: exact (lowering: ok, 3 NPU layers; NPUC: PASS, 852 KB, 4.7 s; ncpgen check: EXACT (0 s))

### tf-cifar-cnn

- **Source**: TensorFlow tutorial 'Convolutional Neural Network (CNN)' (tensorflow.org/tutorials/images/cnn)
- **Task / input**: CIFAR-10 32x32x3, 10 classes; input 32x32x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3 x3, max_pool2d x2, reshape, fully_connected x2, softmax; 0.12 M weights; output 1x1x10; `--allow-none`
- **Status**: exact (lowering: ok, 7 NPU layers; NPUC: PASS, 168 KB, 5.5 s; ncpgen check: EXACT (0 s))

### vgg-cifar

- **Source**: VGG-style CIFAR-10 CNN, three blocks of two 3x3 convs + max pool (Simonyan & Zisserman 2014 pattern; the common 'VGG 3-block' tutorial baseline)
- **Task / input**: CIFAR-10 32x32x3, 10 classes; input 32x32x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3 x6, max_pool2d x3, reshape, fully_connected x2, softmax; 0.55 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); dropout removed (inference)
- **Status**: exact (lowering: ok, 11 NPU layers; NPUC: PASS, 685 KB, 24.5 s; ncpgen check: EXACT (3 s))

### resnet8

- **Source**: MLPerf Tiny image classification ResNet-8 (resnet_v1_eembc, mlcommons/tiny)
- **Task / input**: CIFAR-10 32x32x3, 10 classes; input 32x32x3 int8 (scale 0.0078125, zero point 0)
- **Ops**: conv2d 3x3 x5, add x3, conv2d 3x3/2 x2, conv2d 1x1/2 x2, average_pool2d, reshape, fully_connected, softmax; 0.08 M weights; output 1x1x10; `--quant 3 --allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); every activation scale a power of two and every zero point 0 (quant 3): NPUC's Eltwise add aligns by shifts only and has no zero-point term; no activation on the second conv of each block and on the shortcuts: --allow-none (ncpgen's corrected GReLU table makes 'none' right on the NPU; NPUC's own table only for ceil(Cout/8) even)
- **Status**: exact (lowering: ok, 17 NPU layers; NPUC: PASS, 202 KB, 2.5 s; ncpgen check: EXACT (0 s))

### ad-autoencoder

- **Source**: MLPerf Tiny anomaly detection: the DCASE 2020 task 2 baseline autoencoder (FC 640-128-128-128-128-8-128-128-128-128-640)
- **Task / input**: 5 frames x 128 log-mel bins (a 640 vector), reconstruction; input 1x1x640 int8 (scale 0.4, zero point 10)
- **Ops**: fully_connected x10; 0.26 M weights; output 1x1x640; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); the linear output layer: --allow-none
- **Status**: exact (lowering: ok, 10 NPU layers; NPUC: PASS, 463 KB, 4.5 s; ncpgen check: EXACT (0 s))

### kws-cnn

- **Source**: Hello Edge CNN (Zhang et al. 2017, 'Hello Edge: Keyword Spotting on Microcontrollers'; ARM ML-KWS-for-MCU CNN-S: conv 28 10x4 s1, conv 30 10x4 s2x1, linear 16, FC 128, 12 classes)
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4, conv2d 10x4/2x1, reshape, fully_connected x3, softmax; 0.07 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: dropout removed (inference); the linear layers (no activation): --allow-none; npunet rewrites conv2's stride 2x1 (NPUC fails on unequal strides: 'Couldn't find a correct processing strategy for a single node' here, '_strideWidth == _strideHeight' on a 40x7x28 probe) exactly and by itself: conv2 at stride 1, the linear layer's weights on its even rows and zeros on the odd ones (kws-cnn-npu's network, the same NCP)
- **Status**: exact (lowering: ok, 5 NPU layers; NPUC: PASS, 116 KB, 10.6 s; ncpgen check: EXACT (0 s))

### kws-cnn-npu

- **Source**: Hello Edge CNN (Zhang et al. 2017, 'Hello Edge: Keyword Spotting on Microcontrollers'; ARM ML-KWS-for-MCU CNN-S: conv 28 10x4 s1, conv 30 10x4 s2x1, linear 16, FC 128, 12 classes)
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4 x2, reshape, fully_connected x3, softmax; 0.10 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: dropout removed (inference); conv2 at stride 1 (31x4x30) and the linear layer's weights on its even rows, zeros on the odd ones: exactly the published stride-2x1 network (conv2 does twice the work); the linear layers (no activation): --allow-none
- **Status**: exact (lowering: ok, 5 NPU layers; NPUC: PASS, 116 KB, 3.2 s; ncpgen check: EXACT (0 s))

### micro-speech

- **Source**: TensorFlow Lite Micro micro_speech 'tiny_conv' (tensorflow/examples/speech_commands models.py: conv 8 10x8 s2 SAME, FC 4)
- **Task / input**: keyword spotting, 49x40 spectrogram, 4 classes; input 49x40x1 int8 (scale 0.101961, zero point -128)
- **Ops**: conv2d 10x8/2, reshape, fully_connected, softmax; 0.02 M weights; output 1x1x4; `--allow-none`
- **Adaptations**: dropout removed (inference)
- **Missing for the published form**: the FC over 25x20x8 (as a 25x20 VALID conv): NPUC asserts '(nzvSize & 0xFE00) == 0' (NPULIRKernelPack::GenBitmasks2020): too many non-zero weights in one kernel slice; npunet should refuse it or split the FC (lossy: partial sums requantised)
- **Status**: npuc-fails (lowering: ok, 2 NPU layers; NPUC: FAIL: (nzvSize & 0xFE00) == 0; ncpgen check: -)

### magic-wand

- **Source**: TensorFlow Lite Micro magic_wand gesture CNN (conv 8 4x3, max pool 3x3, conv 16 4x1, max pool 3x1, FC 16, FC 4)
- **Task / input**: gestures, 128 accelerometer samples x 3 axes, 4 classes; input 128x3x1 int8 (scale 15.748, zero point 0)
- **Ops**: conv2d 4x3, max_pool2d x2, conv2d 4x1, reshape, fully_connected x2, softmax; 0.00 M weights; output 1x1x4; `--allow-none`
- **Adaptations**: dropout removed (inference); the output FC: --allow-none; npunet rewrites the 3x1 stride-3x1 max pool (NPUC asserts 'absDiff < stride', PoolNodeTileBackPropagator, on a larger vertical stride) exactly and by itself: the pool at stride 1 VALID (40x1), the first FC's weights on rows 0, 3, ..., 39 and zeros elsewhere (magic-wand-npu's network, the same NCP)
- **Status**: exact (lowering: ok, 6 NPU layers; NPUC: PASS, 17 KB, 11.2 s; ncpgen check: EXACT (0 s))

### magic-wand-npu

- **Source**: TensorFlow Lite Micro magic_wand gesture CNN (conv 8 4x3, max pool 3x3, conv 16 4x1, max pool 3x1, FC 16, FC 4)
- **Task / input**: gestures, 128 accelerometer samples x 3 axes, 4 classes; input 128x3x1 int8 (scale 15.748, zero point 0)
- **Ops**: conv2d 4x3, max_pool2d x2, conv2d 4x1, reshape, fully_connected x2, softmax; 0.01 M weights; output 1x1x4; `--allow-none`
- **Adaptations**: dropout removed (inference); the 3x1 stride-3x1 max pool on the 42x1 map -> a 3x1 stride-1 VALID pool (40x1) and the first FC's weights on rows 0, 3, ..., 39, zeros elsewhere: exactly the published network (NPUC refuses every stride-3 pool on a width-1 map: 3x1/3x3 strides, VALID or SAME; 3x3 stride 3 SAME asserts 'Tried to split a tensor's shape into zero tiles'); the output FC: --allow-none
- **Status**: exact (lowering: ok, 6 NPU layers; NPUC: PASS, 17 KB, 10.0 s; ncpgen check: EXACT (0 s))

### hello-world

- **Source**: TensorFlow Lite Micro hello_world sine model (FC 16, FC 16, FC 1)
- **Task / input**: sine regression, 1 input; input 1x1x1 int8 (scale 0.0246399, zero point -128)
- **Ops**: fully_connected x3; 0.00 M weights; output 1x1x1; `--allow-none`
- **Adaptations**: the linear output: --allow-none
- **Status**: exact (lowering: ok, 3 NPU layers; NPUC: PASS, 9 KB, 1.6 s; ncpgen check: EXACT (0 s))

### squeezenet11-96

- **Source**: SqueezeNet 1.1 (Iandola et al. 2016; the v1.1 reference: conv1 3x3/2 64, fire2-9, conv10, global average pool)
- **Task / input**: 96x96x3 images, 10 classes; input 96x96x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3/2, max_pool2d x3, conv2d 1x1 x17, conv2d 3x3 x8, concat x8, mean, softmax; 0.72 M weights; output 1x1x10; `--quant 1`
- **Adaptations**: dropout removed (inference); the two expand convs of a fire module share one output quantisation (NPUC's concat copies bytes; TFLite's converter does the same for CONCATENATION); 1000 ImageNet classes -> 10
- **Status**: exact (lowering: ok, 38 NPU layers; NPUC: PASS, 1162 KB, 5.9 s; ncpgen check: EXACT (1 s))

### alexnet-96

- **Source**: AlexNet (Krizhevsky et al. 2012), torchvision widths 64-192-384-256-256: conv 11x11/4, max pools 3/2, three FC layers
- **Task / input**: 96x96x3 images (STL-10), 10 classes; input 96x96x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 11x11/4, max_pool2d x3, conv2d 5x5, conv2d 3x3 x3, reshape, fully_connected x3, softmax; 3.79 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: dropout removed (inference); LRN dropped (torchvision's AlexNet has none); FC 4096-4096 -> 1024-1024 for 10 classes (STL-10); explicit padding -> TFLite VALID/SAME (no PAD op); the output FC: --allow-none; npunet rewrites conv1's stride 4 (NPUC asserts 'Stride > 2 is not supported here!', NPULIRKernelMemoryData::IterateOverAllValues) exactly and by itself: conv1 at stride 2 VALID, then a 1x1 stride-2 VALID conv with identity weights and multiplier 1 keeping every other row and column (alexnet-96-npu's network, the same NCP)
- **Status**: exact (lowering: ok, 12 NPU layers; NPUC: PASS, 4576 KB, 16.5 s; ncpgen check: EXACT (3 s))

### alexnet-96-npu

- **Source**: AlexNet (Krizhevsky et al. 2012), torchvision widths 64-192-384-256-256: conv 11x11/4, max pools 3/2, three FC layers
- **Task / input**: 96x96x3 images (STL-10), 10 classes; input 96x96x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 11x11/2, conv2d 1x1/2, max_pool2d x3, conv2d 5x5, conv2d 3x3 x3, reshape, fully_connected x3, softmax; 3.79 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: dropout removed (inference); LRN dropped (torchvision's AlexNet has none); FC 4096-4096 -> 1024-1024 for 10 classes (STL-10); explicit padding -> TFLite VALID/SAME (no PAD op); conv1 11x11 stride 4 VALID -> stride 2 VALID, then every other row and column kept by a 1x1 stride-2 VALID conv with identity weights and multiplier 1: exactly the same output (conv1 computes 4x the positions; a 1x1 stride-2 max pool would do it too, but NPUC's pooling refuses a stride larger than its kernel); the output FC: --allow-none
- **Status**: exact (lowering: ok, 12 NPU layers; NPUC: PASS, 4576 KB, 17.9 s; ncpgen check: EXACT (43 s))

### vgg11-64

- **Source**: VGG-11 (configuration A, Simonyan & Zisserman 2014) at half width: 32-M-64-M-128-128-M-256-256-M-256-256-M, FC 512-512-200
- **Task / input**: 64x64x3 images (Tiny ImageNet), 200 classes; input 64x64x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3 x8, max_pool2d x5, reshape, fully_connected x3, softmax; 3.19 M weights; output 1x1x200; `--allow-none`
- **Adaptations**: dropout removed (inference); every width halved (64..512 -> 32..256), FC 4096 -> 512, 1000 -> 200 classes
- **Status**: exact (lowering: ok, 16 NPU layers; NPUC: PASS, 3838 KB, 122.5 s; ncpgen check: EXACT (14 s))

### yolov2-tiny-128

- **Source**: Tiny YOLOv2 (VOC) detector at half width: 3x3 convs 8-16-32-64-128-256 with 2x2 max pools, a 2x2 stride-1 SAME max pool, 512, 512, a 1x1 conv to 5 anchors x (5 + 20) = 125
- **Task / input**: 128x128x3 images, a 4x4 grid of VOC detections; input 128x128x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3 x8, max_pool2d x6, conv2d 1x1; 4.00 M weights; output 4x4x125; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); leaky ReLU (0.1) -> ReLU (NPUC's GReLU table could hold a leaky slope, not mapped yet); every width halved (16..1024 -> 8..512), 416x416 -> 128x128; the linear detection head: --allow-none
- **Missing for the published form**: leaky ReLU (the published activation): not supported (docs/format/grelu.md: a fractional slope on piece 0)
- **Status**: exact (lowering: ok, 15 NPU layers; NPUC: PASS, 4607 KB, 99.8 s; ncpgen check: EXACT (20 s))

### mobilenet-v1-025

- **Source**: MobileNet v1 0.25 (Howard et al. 2017), the TensorFlow Lite Micro person_detection model: 13 depthwise-separable blocks, 8..256 channels, average pool 3x3/2, 1x1 conv to 2 classes
- **Task / input**: person detection, 96x96x1 grayscale, 2 classes; input 96x96x1 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3/2, depthwise 3x3 x13, conv2d 1x1 x14, average_pool2d, reshape, softmax; 0.21 M weights; output 1x1x2; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); the logits conv (no activation): --allow-none
- **Status**: exact (lowering: ok, 29 NPU layers; NPUC: PASS, 797 KB, 45.6 s; ncpgen check: EXACT (1 s))

### mobilenet-v1-025-dense

- **Source**: MobileNet v1 0.25 (Howard et al. 2017), the TensorFlow Lite Micro person_detection model: 13 depthwise-separable blocks, 8..256 channels, average pool 3x3/2, 1x1 conv to 2 classes
- **Task / input**: person detection, 96x96x1 grayscale, 2 classes; input 96x96x1 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3/2 x5, conv2d 3x3 x9, conv2d 1x1 x14, average_pool2d, reshape, softmax; 1.77 M weights; output 1x1x2; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function; C times the weights of that layer counted dense, the zeros skipped by NPUC's weight format); the logits conv: --allow-none
- **Status**: exact (lowering: ok, 29 NPU layers; NPUC: PASS, 797 KB, 6.3 s; ncpgen check: EXACT (1 s))

### mobilenet-v2-035

- **Source**: MobileNet v2 0.35 (Sandler et al. 2018; keras.applications MobileNetV2 alpha 0.35 at 96x96: 17 inverted residual blocks, 8..112 channels, conv 1280, 1000 classes)
- **Task / input**: 96x96x3 images, 1000 classes; input 96x96x3 int8 (scale 0.0078125, zero point 0)
- **Ops**: conv2d 3x3/2, depthwise 3x3 x17, conv2d 1x1 x34, add x10, mean, fully_connected, softmax; 1.66 M weights; output 1x1x1000; `--quant 3 --allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); every activation scale a power of two except ReLU6's, every zero point 0 (quant 3) for the 10 residual ADDs; ReLU6 quantised 6/127 zp 0 (its clamp is then the int8 saturation); the linear bottlenecks and the classifier: --allow-none
- **Status**: exact (lowering: ok, 64 NPU layers; NPUC: PASS, 3537 KB, 23.5 s; ncpgen check: EXACT (9 s))

### mobilenet-v2-035-dense

- **Source**: MobileNet v2 0.35 (Sandler et al. 2018; keras.applications MobileNetV2 alpha 0.35 at 96x96: 17 inverted residual blocks, 8..112 channels, conv 1280, 1000 classes)
- **Task / input**: 96x96x3 images, 1000 classes; input 96x96x3 int8 (scale 0.0078125, zero point 0)
- **Ops**: conv2d 3x3/2 x5, conv2d 3x3 x13, conv2d 1x1 x34, add x10, mean, fully_connected, softmax; 6.74 M weights; output 1x1x1000; `--quant 3 --allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); every activation scale a power of two except ReLU6's, every zero point 0 (quant 3) for the 10 residual ADDs; ReLU6 quantised 6/127 zp 0 (its clamp is then the int8 saturation); the linear bottlenecks and the classifier: --allow-none; each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function; 1.66 M -> 6.74 M weights counted dense, but NPUC's weight format skips the zeros with bitmaps)
- **Status**: exact (lowering: ok, 64 NPU layers; NPUC: PASS, 3537 KB, 24.4 s; ncpgen check: EXACT (8 s))

### ds-cnn

- **Source**: DS-CNN (Zhang et al. 2017, Hello Edge), the MLPerf Tiny keyword-spotting model: conv 64 10x4/2, 4 x (depthwise 3x3 + pointwise 64), average pool 25x5, FC 12
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4/2, depthwise 3x3 x4, conv2d 1x1 x4, average_pool2d, reshape, fully_connected, softmax; 0.02 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); dropout removed (inference); the output FC: --allow-none
- **Status**: exact (lowering: ok, 11 NPU layers; NPUC: PASS, 209 KB, 2.7 s; ncpgen check: EXACT (0 s))

### ds-cnn-dense

- **Source**: DS-CNN (Zhang et al. 2017, Hello Edge), the MLPerf Tiny keyword-spotting model: conv 64 10x4/2, 4 x (depthwise 3x3 + pointwise 64), average pool 25x5, FC 12
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4/2, conv2d 3x3 x4, conv2d 1x1 x4, conv2d 25x5, reshape, fully_connected, softmax; 0.68 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); dropout removed (inference); each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function); the 25x5 global average pool written as the identity conv npunet lowers it to; the output FC: --allow-none
- **Status**: exact (lowering: ok, 11 NPU layers; NPUC: PASS, 209 KB, 22.2 s; ncpgen check: EXACT (3 s))
