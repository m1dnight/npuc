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
- **Variants.** `<model>-npu`: the published model rewritten so NPUC compiles it, computing the **same function** (its
  CPU reference equals the published model's byte for byte wherever npunet lowers both: kws-cnn, magic-wand,
  alexnet-96). `<model>-dense`: each depthwise convolution written as the dense convolution with block-diagonal
  weights (the same function). The depthwise study under way (docs/format/depthwise.md) finds that NPUC itself compiles
  a depthwise layer exactly this way on the device path, its weight format skipping the zeros with bitmaps.
- **TFLite**: `write` adds `<model>.tflite` when `npunet.graph_to_tflite` can write the model and the TFLite lowers to
  the same network as the JSON. It can write none of the zoo today (it writes conv2d, VALID max_pool2d, add, concat and
  relu only; every model here has a fully_connected, average_pool2d, reshape, softmax, a SAME pool or a depthwise conv).

## Status (2026-11-22, NPUC root-314e61a6)

**24 models: 15 exact, 2 compile but differ, 4 fail in NPUC, 3 refused by npunet.** Every model npunet lowers has its
CPU reference (`npunet.py ref`, 0.1 to 18 s). The three `-npu` rewrites give byte for byte the reference of the
published model. Every NCP NPUC compiled is regression data in `$NCPGEN_WORK/sweep/zoo/` (`tests/regress.sh zoo`: 15 of
17, the two known misses below), and the 15 exact ones are rows of `tests/golden.txt`, so `tests/run.sh` regenerates
them with no vendor file. The NPUC column's time includes npunet's lowering and was measured under a parallel load.

| model | input | options | lowering | NPUC | ncpgen check | CPU reference |
|---|---|---|---|---|---|---|
| lenet5 | 28x28x1 | --allow-none | ok, 7 NPU layers | PASS, 109 KB, 4.5 s | **EXACT** (0 s) | 895225fe26b4, 10 B, 0.1 s |
| keras-mnist-cnn | 28x28x1 | --allow-none | ok, 5 NPU layers | PASS, 67 KB, 4.4 s | **EXACT** (0 s) | bee31a89bef9, 10 B, 0.4 s |
| mnist-mlp | 1x1x784 | --allow-none | ok, 3 NPU layers | PASS, 852 KB, 8.1 s | **EXACT** (2 s) | cbfaca9fb6f4, 10 B, 0.4 s |
| tf-cifar-cnn | 32x32x3 | --allow-none | ok, 7 NPU layers | PASS, 168 KB, 5.5 s | **EXACT** (0 s) | e78c8338eb6e, 10 B, 0.5 s |
| vgg-cifar | 32x32x3 | --allow-none | ok, 11 NPU layers | PASS, 685 KB, 24.5 s | **EXACT** (3 s) | c5c997d7131b, 10 B, 3.8 s |
| resnet8 | 32x32x3 | --quant 3 --allow-none | ok, 17 NPU layers | PASS, 202 KB, 7.3 s | **differs**: groups 5-7 and weight blocks 5-7 in another order (ncpgen's layer order: need 1) | 29f02bf6832a, 10 B, 1.4 s |
| ad-autoencoder | 1x1x640 | --allow-none | ok, 10 NPU layers | PASS, 463 KB, 8.7 s | **EXACT** (1 s) | 533c72a18436, 640 B, 0.2 s |
| kws-cnn | 49x10x1 | --allow-none | ok, 5 NPU layers | FAIL: solutionFound && "Couldn't find a correct processing strategy for a single node" | - | 7a44e7c3827a, 12 B, 0.3 s |
| kws-cnn-npu | 49x10x1 | --allow-none | ok, 5 NPU layers | PASS, 116 KB, 3.2 s | **EXACT** (0 s) | 7a44e7c3827a, 12 B, 0.4 s |
| micro-speech | 49x40x1 | --allow-none | ok, 2 NPU layers | FAIL: (nzvSize & 0xFE00) == 0 | - | 0336eae2a362, 4 B, 0.1 s |
| magic-wand | 128x3x1 | --allow-none | ok, 6 NPU layers | FAIL: absDiff < stride && "Something went wrong during shape calculation" | - | ddc82277175a, 4 B, 0.1 s |
| magic-wand-npu | 128x3x1 | --allow-none | ok, 6 NPU layers | PASS, 17 KB, 10.0 s | **EXACT** (0 s) | ddc82277175a, 4 B, 0.1 s |
| hello-world | 1x1x1 | --allow-none | ok, 3 NPU layers | PASS, 9 KB, 2.9 s | **EXACT** (0 s) | 2b4c342f5433, 1 B, 0.0 s |
| squeezenet11-96 | 96x96x3 | - | ok, 38 NPU layers | PASS, 1162 KB, 76.2 s | **EXACT** (3 s) | 4951801caa0a, 10 B, 6.3 s |
| alexnet-96 | 96x96x3 | --allow-none | ok, 11 NPU layers | FAIL: "Stride > 2 is not supported here!" && (ifmOrder.getSlotSize() <= 4) | - | 6d9704d8711f, 10 B, 7.5 s |
| alexnet-96-npu | 96x96x3 | --allow-none | ok, 12 NPU layers | PASS, 4576 KB, 17.9 s | **EXACT** (43 s) | 6d9704d8711f, 10 B, 9.5 s |
| vgg11-64 | 64x64x3 | --allow-none | ok, 16 NPU layers | PASS, 3838 KB, 122.5 s | **EXACT** (14 s) | 5456870cc263, 200 B, 16.0 s |
| yolov2-tiny-128 | 128x128x3 | --allow-none | ok, 15 NPU layers | PASS, 4607 KB, 99.8 s | **EXACT** (20 s) | b4805006f2b0, 2000 B, 9.0 s |
| mobilenet-v1-025 | 96x96x1 | --allow-none | refused: l1_conv1: RELU6 clamps inside the output range (NPUC: 'Unsupported FusedActivation type' 3) | - | - | - |
| mobilenet-v1-025-dense | 96x96x1 | --allow-none | ok, 29 NPU layers | PASS, 797 KB, 62.8 s | **EXACT** (6 s) | 5af11588254e, 2 B, 5.7 s |
| mobilenet-v2-035 | 96x96x3 | --quant 3 --allow-none | refused: op depthwise_conv2d is not supported (conv2d, fully_connected, max_pool2d, average_pool2d, add, concat, relu; reshape and softmax at the end) | - | - | - |
| mobilenet-v2-035-dense | 96x96x3 | --quant 3 --allow-none | ok, 64 NPU layers | PASS, 3537 KB, 211.2 s | **differs**: 69 groups for 65 layers in a graph; frame.py raises (need 4) | 0cc597e5b755, 1000 B, 17.6 s |
| ds-cnn | 49x10x1 | --allow-none | refused: op depthwise_conv2d is not supported (conv2d, fully_connected, max_pool2d, average_pool2d, add, concat, relu; reshape and softmax at the end) | - | - | - |
| ds-cnn-dense | 49x10x1 | --allow-none | ok, 11 NPU layers | PASS, 209 KB, 22.2 s | **EXACT** (3 s) | ad2c9fdec999, 12 B, 2.0 s |

Per published model: LeNet-5, the Keras MNIST convnet and MLP, the TensorFlow CIFAR CNN, a VGG-style CIFAR CNN, the
DCASE/MLPerf Tiny autoencoder, the TFLM sine model, SqueezeNet 1.1, VGG-11 and Tiny YOLOv2 are **exact as published**
(with the generic adaptations above). The Hello Edge KWS CNN, magic wand and AlexNet are **exact after an exact rewrite**
(`-npu`). MobileNet v1, DS-CNN (and MobileNet v2, but for the frame gap below) are **exact with depthwise written as
dense** (`-dense`). ResNet-8 compiles but **differs** (ncpgen's layer order). micro_speech **fails in NPUC**.

## What ncpgen and npunet need, ranked

What unblocks the most models for the least work comes first. None of it is applied here (ncpgen/*.py is being
refactored elsewhere).

1. **ncpgen: NPUC's layer order for graphs** (`weights.exec_order`, `stream.exec_order`; frame.py calls the latter).
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
2. **npunet: RELU6 the way TFLite rounds it** (`lower()` and `_fused_act`). TFLite's int8 ReLU6 output is scale 6/255
   with zero point -128, and TFLite clamps at `zp + round(6 / scale)` = 127, the saturation (a no-op). npunet tests
   `zp + 6 / scale >= 255` in float32, gets 254.999995 and refuses it. So every TFLite MobileNet (and EfficientNet-lite,
   MobileNet-SSD...) is refused at its first layer, before its depthwise layers. Fix: `zp + round(6 / scale) >= 255`.
   **Effort: minutes.**
3. **npunet: DEPTHWISE_CONV_2D** (mobilenet-v1-025, mobilenet-v2-035, ds-cnn): lower it to the dense convolution with
   block-diagonal weights, which is what NPUC itself does on the device path (the depthwise study under way,
   docs/format/depthwise.md). The `-dense` models show ncpgen exact on that form for MobileNet v1 and DS-CNN. Needed:
   a `depthwise_conv2d` op in npu-net/1 (the zoo writes it as TFLite stores it: layout 1HWO, `depth_multiplier` 1), the
   op in `tflite_to_graph`, and per-channel weight scales, which TFLite uses by default for depthwise weights. **Effort:
   half a day** without per-channel scales (folding them is lossy; see 9).
4. **frame.py: graphs whose large layers NPUC splits into several groups** (mobilenet-v2-035-dense). NPUC gives the
   112->1280 conv and the 1280->1000 FC three groups each. ncpgen's stream part already makes the same 69 groups, with
   the same sizes (`stream.Plan`), but `frame.intrinsics` in graph mode raises "65 layer groups, 69 stream groups", and
   streams g65-g67 (the FC) differ at offsets 0x120/0x1f2 (likely its DMA addresses, which the frame plan supplies).
   Chains with layers as large as these are exact (vgg11-64, yolov2-tiny-128, alexnet-96-npu). **Effort: a day**
   (port the chain plan's multi-group layers to the graph plan).
5. **npunet: a global average pool larger than 14** (ds-cnn). `_global_avg_as_conv` refuses kernels over 14, but NPUC
   compiles the 25x5 identity conv over 64 channels and ncpgen is exact on it (ds-cnn-dense writes that conv itself;
   probes: 25x5, 25x8, 25x10, 8x25, 10x25, 12x20, 14x18, 18x14, 15x15 over 8 channels, 25x20 over 1 are exact too).
   Two limits stand past it: NPUC's weight bitmask (it asserts `(nzvSize & 0xFE00) == 0` in
   `NPULIRKernelPack::GenBitmasks2020` on 13x20 and 25x20 over 8 channels, 25x20 over 4: kernel area, channels and
   non-zero weights together), and ncpgen's own refusal of 16x16 and 20x13 over 8 channels, which NPUC compiles
   ("weights.py: a sub-batch would hold no channel pair"). Map both, then lower up to them. **Effort: half a day** (a
   sweep plus reading GenBitmasks2020); it also gives micro-speech (8) its honest refusal.
6. **npunet: TFLite patterns real converters write.** None of these is in the zoo's JSON, but a converted model hits each
   one. MEAN over H and W (Keras' GlobalAveragePooling2D) should lower like a global AVERAGE_POOL_2D. A 2-D input
   `[1, N]` (mnist-mlp, ad-autoencoder, hello-world as Keras writes them) is the same tensor as `[1, 1, 1, N]`. SAME
   padding whose pad would be negative (resnet8's 1x1 stride-2 shortcuts: "SAME padding with stride 2 > kernel 1 leaves
   a negative pad") is VALID in TFLite, which clamps the pad at 0. **Effort: an hour or two for the three.**
7. **npunet: refuse what NPUC asserts on, or rewrite it exactly.** npunet lowers these and NPUC then aborts:
   conv stride > 2 (alexnet-96: "Stride > 2 is not supported here!"), unequal conv strides (kws-cnn's 2x1), unequal
   pool strides (magic-wand's 3x1), any stride-3 pool on a width-1 map, a pool stride larger than its kernel. Two exact
   rewrites, applied by hand in the zoo, could be automatic:
   - conv stride 4 VALID -> stride 2 VALID, then a 1x1 stride-2 VALID identity conv (multiplier exactly 1) that keeps
     every other row and column (alexnet-96-npu). A 1x1 stride-2 max pool would do the same, but NPUC refuses it
     ("ncp_nfu_pooling: Invalid parameters! The stride value should be equal or less than the size of kernel").
   - a stride-(s, 1) conv or pool read only by an FC -> stride 1, with the FC's weights on every s-th row and zeros
     elsewhere (kws-cnn-npu, magic-wand-npu).

   **Effort: the refusals an hour, the rewrites half a day.**
8. **micro-speech** (the FC over 25x20x8 as a 25x20 conv: `(nzvSize & 0xFE00) == 0`). Covering it needs the bitmask
   limit of 5 and then a split of the FC into partial convs whose partial sums are requantised (lossy). Low value for
   the effort (**a day**); refusing it cleanly comes with 5.
9. **Beyond the zoo**: leaky ReLU (Tiny YOLOv2's activation; the GReLU table could hold the slope, docs/format/grelu.md
   section 6, needs NPU experiments: days), per-channel weight scales (TFLite's default for every conv: NPUC's
   on-device filler takes one scale per layer, so this needs the per-DRU slopes NPUC fills anyway, studied on the
   NPU: days), and `graph_to_tflite` for FULLY_CONNECTED, AVERAGE_POOL_2D, padded pools, RESHAPE, SOFTMAX and
   DEPTHWISE_CONV_2D, so the zoo writes TFLite and the TFLite path is tested on real topologies (**two hours**).

## NPUC limits met here (V, the oracle)

| Layer | NPUC | Seen in |
|---|---|---|
| conv stride 4 (11x11, SAME or VALID) | asserts "Stride > 2 is not supported here!" | alexnet-96, probe 64x64x3 |
| conv stride 2x1 (10x4 VALID) | asserts `_strideWidth == _strideHeight` (40x7x28) or "Couldn't find a correct processing strategy for a single node" | kws-cnn |
| max pool 3x1 stride 3x1 (VALID or SAME) | asserts `absDiff < stride` | magic-wand |
| max pool 3x3 stride 3 SAME on a 42x1 map | asserts "Tried to split a tensor's shape into zero tiles" | probe |
| max pool 3x1 stride 3x3 on a 42x1 map | fails | probe |
| max pool 1x1 stride 2 | "The stride value should be equal or less than the size of kernel" | alexnet-96-npu, first try |
| conv 25x20 VALID over 8 (or 4) channels, 13x20 over 8 | asserts `(nzvSize & 0xFE00) == 0` | micro-speech, probes |
| conv 25x5/64, 16x4/30, 15x4/16, 14x14/16, 15x15/8, 25x10/8, 12x20/8, 14x18/8, 25x20/1, 49x1/8, 1x40/8 VALID | compile; ncpgen exact | ds-cnn-dense, kws-cnn-npu, probes |
| conv 16x16/8, 20x13/8 VALID | compile; ncpgen refuses ("a sub-batch would hold no channel pair") | probes |
| max pool 2x2 stride 1 SAME, 3x3 stride 3 SAME/VALID on 128x3/129x3, 3x1 stride 1 on 42x1 | compile; ncpgen exact | yolov2-tiny-128, magic-wand-npu, probes |

The probes' networks (`probe-src/`, npu-net/1 JSON, our weights) and NPUC's outputs (`probe/`, `probe3/`) are in
`$NCPGEN_WORK/zoo-evidence/` (outside `sweep/`, so `regress.sh` does not take them), with the resnet8 ordering variants
(`order/`: add inputs swapped, shortcut first, both).

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
- **Adaptations**: dropout removed (inference); input [1, 784] written as [1, 1, 1, 784] (the same bytes): npunet wants a 4-D NHWC input
- **Status**: exact (lowering: ok, 3 NPU layers; NPUC: PASS, 852 KB, 8.1 s; ncpgen check: EXACT (2 s))

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
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); every activation scale a power of two and every zero point 0 (quant 3): NPUC's Eltwise add aligns by shifts only and has no zero-point term; the 1x1 stride-2 shortcut convs: SAME -> VALID (the same output; npunet refuses SAME when the pad would be negative, where TFLite clamps it to 0); no activation on the second conv of each block and on the shortcuts: --allow-none (ncpgen's corrected GReLU table makes 'none' right on the NPU; NPUC's own table only for ceil(Cout/8) even)
- **Missing for the published form**: SAME 1x1 stride-2 conv (negative pad): npunet should clamp the pad at 0 like TFLite
- **Status**: differs (lowering: ok, 17 NPU layers; NPUC: PASS, 202 KB, 7.3 s; ncpgen check: NPUC runs the shortcut conv of each stage first, ncpgen last: need 1)

### ad-autoencoder

- **Source**: MLPerf Tiny anomaly detection: the DCASE 2020 task 2 baseline autoencoder (FC 640-128-128-128-128-8-128-128-128-128-640)
- **Task / input**: 5 frames x 128 log-mel bins (a 640 vector), reconstruction; input 1x1x640 int8 (scale 0.4, zero point 10)
- **Ops**: fully_connected x10; 0.26 M weights; output 1x1x640; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); input [1, 640] written as [1, 1, 1, 640] (the same bytes); the linear output layer: --allow-none
- **Missing for the published form**: a 2-D [1, N] TFLite input: npunet refuses it (4-D NHWC wanted), [1, 1, 1, N] is the same tensor
- **Status**: exact (lowering: ok, 10 NPU layers; NPUC: PASS, 463 KB, 8.7 s; ncpgen check: EXACT (1 s))

### kws-cnn

- **Source**: Hello Edge CNN (Zhang et al. 2017, 'Hello Edge: Keyword Spotting on Microcontrollers'; ARM ML-KWS-for-MCU CNN-S: conv 28 10x4 s1, conv 30 10x4 s2x1, linear 16, FC 128, 12 classes)
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4, conv2d 10x4/2x1, reshape, fully_connected x3, softmax; 0.07 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: dropout removed (inference)
- **Missing for the published form**: conv stride 2x1: NPUC fails ('Couldn't find a correct processing strategy for a single node' here; '_strideWidth == _strideHeight' in PNodeConvolution::toAtomicChain on a 40x7x28 probe); npunet lowers it without complaint (should refuse, or rewrite it as kws-cnn-npu when an FC reads the conv)
- **Status**: npuc-fails (lowering: ok, 5 NPU layers; NPUC: FAIL: solutionFound && "Couldn't find a correct processing strategy for a single node"; ncpgen check: -)

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
- **Adaptations**: dropout removed (inference)
- **Missing for the published form**: max pool 3x1 stride 3x1: NPUC asserts 'absDiff < stride' (PoolNodeTileBackPropagator) on unequal pool strides; npunet lowers it without complaint (should refuse, or rewrite it as magic-wand-npu when an FC reads the pool)
- **Status**: npuc-fails (lowering: ok, 6 NPU layers; NPUC: FAIL: absDiff < stride && "Something went wrong during shape calculation"; ncpgen check: -)

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
- **Adaptations**: input [1, 1] written as [1, 1, 1, 1]; the linear output: --allow-none
- **Status**: exact (lowering: ok, 3 NPU layers; NPUC: PASS, 9 KB, 2.9 s; ncpgen check: EXACT (0 s))

### squeezenet11-96

- **Source**: SqueezeNet 1.1 (Iandola et al. 2016; the v1.1 reference: conv1 3x3/2 64, fire2-9, conv10, global average pool)
- **Task / input**: 96x96x3 images, 10 classes; input 96x96x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3/2, max_pool2d x3, conv2d 1x1 x17, conv2d 3x3 x8, concat x8, average_pool2d, reshape, softmax; 0.72 M weights; output 1x1x10; `--quant 1`
- **Adaptations**: dropout removed (inference); the two expand convs of a fire module share one output quantisation (NPUC's concat copies bytes; TFLite's converter does the same for CONCATENATION); 1000 ImageNet classes -> 10; global average pool as AVERAGE_POOL_2D 5x5 (a Keras GlobalAveragePooling2D converts to MEAN, which npunet does not take)
- **Status**: exact (lowering: ok, 38 NPU layers; NPUC: PASS, 1162 KB, 76.2 s; ncpgen check: EXACT (3 s))

### alexnet-96

- **Source**: AlexNet (Krizhevsky et al. 2012), torchvision widths 64-192-384-256-256: conv 11x11/4, max pools 3/2, three FC layers
- **Task / input**: 96x96x3 images (STL-10), 10 classes; input 96x96x3 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 11x11/4, max_pool2d x3, conv2d 5x5, conv2d 3x3 x3, reshape, fully_connected x3, softmax; 3.79 M weights; output 1x1x10; `--allow-none`
- **Adaptations**: dropout removed (inference); LRN dropped (torchvision's AlexNet has none); FC 4096-4096 -> 1024-1024 for 10 classes (STL-10); explicit padding -> TFLite VALID/SAME (no PAD op)
- **Missing for the published form**: conv stride 4: NPUC asserts 'Stride > 2 is not supported here!' (NPULIRKernelMemoryData::IterateOverAllValues); npunet lowers it without complaint (should refuse, or rewrite it as below)
- **Status**: npuc-fails (lowering: ok, 11 NPU layers; NPUC: FAIL: "Stride > 2 is not supported here!" && (ifmOrder.getSlotSize() <= 4); ncpgen check: -)

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
- **Missing for the published form**: DEPTHWISE_CONV_2D (13 layers): not supported (npunet refuses the op; NPUC's ConvAttr has a group field); fused RELU6 with TFLite's output quantisation (6/255, zp -128): the clamp is the saturation (TFLite: zp + round(6/scale) = 127), but npunet tests zp + 6/scale >= 255 in float32 (254.999995) and refuses it: use TFLite's rounding
- **Status**: refused (lowering: refused: l1_conv1: RELU6 clamps inside the output range (NPUC: 'Unsupported FusedActivation type' 3); NPUC: -; ncpgen check: -)

### mobilenet-v1-025-dense

- **Source**: MobileNet v1 0.25 (Howard et al. 2017), the TensorFlow Lite Micro person_detection model: 13 depthwise-separable blocks, 8..256 channels, average pool 3x3/2, 1x1 conv to 2 classes
- **Task / input**: person detection, 96x96x1 grayscale, 2 classes; input 96x96x1 int8 (scale 0.00392157, zero point -128)
- **Ops**: conv2d 3x3/2 x5, conv2d 3x3 x9, conv2d 1x1 x14, average_pool2d, reshape, softmax; 1.77 M weights; output 1x1x2; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function; C times the weights of that layer counted dense, the zeros skipped by NPUC's weight format); RELU6 written as RELU with ReLU6's quantisation (6/255, zp -128): the same function, since the clamp at 6.0 is the int8 saturation (npunet's float test refuses RELU6 there); the logits conv: --allow-none
- **Status**: exact (lowering: ok, 29 NPU layers; NPUC: PASS, 797 KB, 62.8 s; ncpgen check: EXACT (6 s))

### mobilenet-v2-035

- **Source**: MobileNet v2 0.35 (Sandler et al. 2018; keras.applications MobileNetV2 alpha 0.35 at 96x96: 17 inverted residual blocks, 8..112 channels, conv 1280, 1000 classes)
- **Task / input**: 96x96x3 images, 1000 classes; input 96x96x3 int8 (scale 0.0078125, zero point 0)
- **Ops**: conv2d 3x3/2, depthwise 3x3 x17, conv2d 1x1 x34, add x10, average_pool2d, reshape, fully_connected, softmax; 1.66 M weights; output 1x1x1000; `--quant 3 --allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); every activation scale a power of two except ReLU6's, every zero point 0 (quant 3) for the 10 residual ADDs; ReLU6 quantised 6/127 zp 0 (its clamp is then the int8 saturation); global average pool as AVERAGE_POOL_2D 3x3 (Keras writes MEAN, which npunet does not take); the linear bottlenecks and the classifier: --allow-none
- **Missing for the published form**: DEPTHWISE_CONV_2D (17 layers); MEAN (Keras' GlobalAveragePooling2D) is not taken by npunet's TFLite reader: lower MEAN over H, W like a global AVERAGE_POOL_2D
- **Status**: refused (lowering: refused: op depthwise_conv2d is not supported (conv2d, fully_connected, max_pool2d, average_pool2d, add, concat, relu; reshape and softmax at the end); NPUC: -; ncpgen check: -)

### mobilenet-v2-035-dense

- **Source**: MobileNet v2 0.35 (Sandler et al. 2018; keras.applications MobileNetV2 alpha 0.35 at 96x96: 17 inverted residual blocks, 8..112 channels, conv 1280, 1000 classes)
- **Task / input**: 96x96x3 images, 1000 classes; input 96x96x3 int8 (scale 0.0078125, zero point 0)
- **Ops**: conv2d 3x3/2 x5, conv2d 3x3 x13, conv2d 1x1 x34, add x10, average_pool2d, reshape, fully_connected, softmax; 6.74 M weights; output 1x1x1000; `--quant 3 --allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); every activation scale a power of two except ReLU6's, every zero point 0 (quant 3) for the 10 residual ADDs; ReLU6 quantised 6/127 zp 0 (its clamp is then the int8 saturation); global average pool as AVERAGE_POOL_2D 3x3 (Keras writes MEAN, which npunet does not take); the linear bottlenecks and the classifier: --allow-none; each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function; 1.66 M -> 6.74 M weights counted dense, but NPUC's weight format skips the zeros with bitmaps)
- **Status**: differs (lowering: ok, 64 NPU layers; NPUC: PASS, 3537 KB, 211.2 s; ncpgen check: streams equal in number and size, frame.py's graph plan takes one group per layer: need 4)

### ds-cnn

- **Source**: DS-CNN (Zhang et al. 2017, Hello Edge), the MLPerf Tiny keyword-spotting model: conv 64 10x4/2, 4 x (depthwise 3x3 + pointwise 64), average pool 25x5, FC 12
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4/2, depthwise 3x3 x4, conv2d 1x1 x4, average_pool2d, reshape, fully_connected, softmax; 0.02 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); dropout removed (inference)
- **Missing for the published form**: DEPTHWISE_CONV_2D (4 layers); the 25x5 global average pool: npunet refuses a pool kernel > 14 ('as a conv its kernel may be up to 14x14 (V)'), but NPUC compiles a 25x5 VALID conv over 64 channels, byte-exact in ncpgen (probe 2026-11-22): relax the limit to what NPUC takes
- **Status**: refused (lowering: refused: op depthwise_conv2d is not supported (conv2d, fully_connected, max_pool2d, average_pool2d, add, concat, relu; reshape and softmax at the end); NPUC: -; ncpgen check: -)

### ds-cnn-dense

- **Source**: DS-CNN (Zhang et al. 2017, Hello Edge), the MLPerf Tiny keyword-spotting model: conv 64 10x4/2, 4 x (depthwise 3x3 + pointwise 64), average pool 25x5, FC 12
- **Task / input**: keyword spotting, 49x10 MFCC, 12 classes; input 49x10x1 int8 (scale 0.6, zero point 80)
- **Ops**: conv2d 10x4/2, conv2d 3x3 x4, conv2d 1x1 x4, conv2d 25x5, reshape, fully_connected, softmax; 0.68 M weights; output 1x1x12; `--allow-none`
- **Adaptations**: batch normalisation folded into the conv/FC before it (as the TFLite converter does); dropout removed (inference); each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function); the 25x5 global average pool written as the identity conv npunet lowers smaller ones to (npunet refuses a pooling kernel > 14; NPUC compiles this one); the output FC: --allow-none
- **Status**: exact (lowering: ok, 11 NPU layers; NPUC: PASS, 209 KB, 22.2 s; ncpgen check: EXACT (3 s))
