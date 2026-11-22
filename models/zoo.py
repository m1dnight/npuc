#!/usr/bin/env python3
"""models/zoo.py: a zoo of common small models, written as ncpgen's input, and what ncpgen does with each
(models/README.md). Standard library only; our own seeded weights (never a pretrained or third-party weight file):
the architectures follow the published ones layer for layer, the weights are deterministic noise with a realistic
int8 distribution and per-layer scales that keep every activation inside its range.

  zoo.py list [--md] [model...]              the models: task, input, ops, quantisation, adaptations
  zoo.py write <dir> [--no-tflite] [model...]
        <dir>/<model>.json (npu-net/1, ncpgen/npunet.py) and <dir>/<model>.tflite (npunet's graph_to_tflite; kept
        only when npunet lowers the TFLite to the same lowered network as the JSON: "tflite: ok")
  zoo.py compile [--jobs N] [model...]
        each model into $NCPGEN_WORK/sweep/zoo/<model>/ (default work/sweep/zoo): model.json, npunet lower + NPUC
        (oracle/npuc-local.sh: lowered.json, net.npuc, net.ncp, log.txt), then npunet.py ref on a seeded random
        input (input.bin -> expected.bin, ref.txt)
  zoo.py check [--jobs N] [--md] [--cached] [model...]
        ncpgen check (whole file, every part ours) on each compiled model: a table of lowering, NPUC, ncpgen and the
        CPU reference; writes <model>/check.json (read by `golden`; --cached reuses it while its NCP is unchanged)
  zoo.py golden                               tests/golden.txt rows (tests/make-golden.sh) for the models whose
                                              NCP ncpgen generates exactly: name|zoo:<model>|options|sha256 of NPUC's NCP

Every model is a function below (MODELS, in order). Variants named <model>-npu are the published model rewritten by
hand to something NPUC compiles, computing the same function (their CPU references equal the published model's byte
for byte); npunet now makes those rewrites itself (its stride rewrites: the published model lowers to the -npu
variant's network), and the variants remain as the cross-check. -dense variants replace each depthwise convolution
by the dense convolution with block-diagonal weights (the same function, C times the weights). Weights, biases and
the reference input depend only on the model family and the layer's name (shake_256), so a variant shares them with its published model.
"""
import array
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from operator import add as _add

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "ncpgen"))
import npunet  # noqa: E402

WORK = os.environ.get("NCPGEN_WORK") or os.path.join(REPO, "work")
ZOO = os.path.join(WORK, "sweep", "zoo")
NPUC_LOCAL = os.path.join(REPO, "oracle", "npuc-local.sh")
NPUNET = os.path.join(REPO, "ncpgen", "npunet.py")
NCPGEN = os.path.join(REPO, "ncpgen", "ncpgen.py")


# ---- deterministic numbers (no dependence on Python's random module or on libm) ----------------------------------
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _stream(key, n):
    return hashlib.shake_256(("ncpgen-zoo/" + key).encode()).digest(n)


_PAIR = [(v & 0xff) + (v >> 8) for v in range(65536)]
_W8 = [max(-127, (s - 510) // 4) for s in range(1021)]
W_RMS = 36.95  # rms of weights8: (sum of 4 uniform bytes - 510) / 4, std sqrt(4 (256^2 - 1) / 12) / 4


def _sum4(key, n):
    """n sums of four uniform bytes (0..1020, bell-shaped around 510); independent of the host's byte order"""
    a = array.array("H", _stream(key, 4 * n))
    return list(map(_add, map(_PAIR.__getitem__, a[0::2]), map(_PAIR.__getitem__, a[1::2])))


def weights8(key, n):
    """n int8 weights, symmetric (-127..127, std ~37, ~1 % zeros): what per-tensor TFLite quantisation gives"""
    return list(map(_W8.__getitem__, _sum4(key, n)))


def bias32(key, n, sigma):
    sg = max(1, int(sigma))
    return [((s - 510) * sg) // 148 for s in _sum4(key, n)]


def unit(key):
    """a deterministic number in [0, 1)"""
    return int.from_bytes(_stream(key, 4), "little") / 2 ** 32


def pow2(x):
    """the power of two nearest x (in log scale)"""
    m, e = math.frexp(x)
    return math.ldexp(1.0, e if m * m >= 0.5 else e - 1)


def osize(h, k, s, pad):
    return (h - k) // s + 1 if pad == "valid" else (h + s - 1) // s


# ---- the model builder ----------------------------------------------------------------------------------------------
class Net:
    """An npu-net/1 graph built layer by layer. Scales: every conv/FC's weight scale makes its pre-activation's float
    rms 1 (He-like), its output scale then covers about 4 sigma (ReLU: [0, 4], else [-4, 4]) with a per-layer factor
    0.75..1.5; ReLU6 outputs use TFLite's (6/255, zp -128; quant 3: 6/127, zp 0). pow2=True rounds every activation
    scale to a power of two (what NPUC's Eltwise add needs). quant 3 (native s8) puts every zero point at 0."""

    def __init__(self, family, hwc, scale, zp, quant=1, pow2=False, dense=False, shapes_only=False, desc=""):
        if quant == 3 and zp != 0:
            raise ValueError("quant 3: input zero point 0")
        self.family, self.quant, self.pow2, self.dense = family, quant, pow2, dense
        self.shapes_only, self.desc = shapes_only, desc
        rms = math.sqrt(sum((v - zp) ** 2 for v in range(-128, 128)) / 256)  # a uniform random input, centred
        shape = [1] + list(hwc)       # (N,): a [1, N] input, as Keras writes a vector model's (= [1, 1, 1, N])
        hwc = [1, 1] + list(hwc) if len(hwc) == 1 else list(hwc)
        self.T = {"input": {"hwc": hwc, "scale": f32(scale), "zp": zp, "rms": rms, "nonneg": False}}
        self.inp = {"name": "input", "shape": shape, "dtype": "int8", "scale": f32(scale), "zero_point": zp}
        self.layers, self.cur, self.cnt = [], "input", {}
        self.none = False  # a conv/FC without activation (npunet: --allow-none)
        self.nweights = 0

    # -- helpers --
    def _name(self, kind, name):
        if name:
            return name
        self.cnt[kind] = self.cnt.get(kind, 0) + 1
        return "%s%d" % (kind, self.cnt[kind])

    def _key(self, name):
        return self.family + "/" + name

    def _w(self, name, n):
        self.nweights += n
        return [] if self.shapes_only else weights8(self._key(name) + "/w", n)

    def _b(self, name, n, sig):
        return [] if self.shapes_only else bias32(self._key(name) + "/b", n, 0.1 * sig)

    def _act(self, act):
        if act == "none":
            self.none = True
        return act

    def _emit(self, L, t, inp):
        if L["name"] in self.T:
            raise ValueError("layer name %s used twice" % L["name"])
        if inp is not None and inp != self.cur:
            L["input"] = inp
        self.layers.append(L)
        self.T[L["name"]] = t
        self.cur = L["name"]
        return L["name"]

    def qout(self, act, name):
        j = 0.75 + 0.75 * unit(self._key(name) + "/j")
        if act == "relu6":
            s = 6 / 255 if self.quant == 1 else 6 / 127
        elif act == "relu":
            s = (4 / 255 if self.quant == 1 else 4 / 127) * j
        else:
            s = 4 / 127 * j
        if self.pow2 and act != "relu6":
            s = pow2(s)
        if self.quant == 3:
            zp = 0
        else:
            zp = -128 if act in ("relu", "relu6") else int(unit(self._key(name) + "/zp") * 9) - 4
        return f32(s), zp

    @staticmethod
    def _t(hwc, q, act):
        return {"hwc": hwc, "scale": q[0], "zp": q[1], "rms": (0.7071 if act != "none" else 1.0) / q[0],
                "nonneg": act != "none"}

    # -- layers --
    def conv(self, cout, k, s=1, pad="same", act="relu", inp=None, name=None, out=None):
        kh, kw = (k, k) if isinstance(k, int) else k
        sh, sw = (s, s) if isinstance(s, int) else s
        x = self.T[inp or self.cur]
        H, W, C = x["hwc"]
        name = self._name("conv", name)
        K = kh * kw * C
        sig = math.sqrt(K) * x["rms"] * W_RMS  # rms of the accumulator (integer units)
        q = out or self.qout(act, name)
        L = {"op": "conv2d", "name": name, "filters": cout, "kernel": [kh, kw], "strides": [sh, sw], "padding": pad,
             "activation": self._act(act),
             "weights": {"dtype": "int8", "layout": "OHWI", "scale": f32(1.0 / (sig * x["scale"])), "zero_point": 0,
                         "data": self._w(name, cout * K)},
             "bias": self._b(name, cout, sig), "output": {"scale": q[0], "zero_point": q[1]}}
        return self._emit(L, self._t([osize(H, kh, sh, pad), osize(W, kw, sw, pad), cout], q, act), inp)

    def dw(self, k=3, s=1, pad="same", act="relu", inp=None, name=None):
        """DEPTHWISE_CONV_2D (depth multiplier 1); with dense=True the same function as a CONV_2D whose weights
        are zero off the diagonal (w[o][i][j][c] = 0 for c != o)"""
        kh, kw = (k, k) if isinstance(k, int) else k
        x = self.T[inp or self.cur]
        H, W, C = x["hwc"]
        name = self._name("dw", name)
        sig = math.sqrt(kh * kw) * x["rms"] * W_RMS
        q = self.qout(act, name)
        ws = f32(1.0 / (sig * x["scale"]))
        d = self._w(name, kh * kw * C)  # 1HWO: (i * kw + j) * C + c
        b = self._b(name, C, sig)
        if self.dense:
            self.nweights += C * kh * kw * C - kh * kw * C
            data = []
            if not self.shapes_only:
                data = [0] * (C * kh * kw * C)
                for o in range(C):
                    for ij in range(kh * kw):
                        data[(o * kh * kw + ij) * C + o] = d[ij * C + o]
            L = {"op": "conv2d", "name": name, "filters": C, "kernel": [kh, kw], "strides": [s, s], "padding": pad,
                 "activation": self._act(act),
                 "weights": {"dtype": "int8", "layout": "OHWI", "scale": ws, "zero_point": 0, "data": data},
                 "bias": b, "output": {"scale": q[0], "zero_point": q[1]}}
        else:
            L = {"op": "depthwise_conv2d", "name": name, "kernel": [kh, kw], "strides": [s, s], "padding": pad,
                 "depth_multiplier": 1, "activation": self._act(act),
                 "weights": {"dtype": "int8", "layout": "1HWO", "scale": ws, "zero_point": 0, "data": d},
                 "bias": b, "output": {"scale": q[0], "zero_point": q[1]}}
        return self._emit(L, self._t([osize(H, kh, s, pad), osize(W, kw, s, pad), C], q, act), inp)

    def fc(self, units, act="relu", name=None, rows_every=1):
        """FULLY_CONNECTED over the flattened (NHWC) input. rows_every=r: the published FC read only every r-th row
        of this input (a stride-(r, 1) conv before it, rewritten to stride 1): its weights go to those rows, zeros
        elsewhere (the same function)"""
        x = self.T[self.cur]
        H, W, C = x["hwc"]
        name = self._name("fc", name)
        Hp = (H - 1) // rows_every + 1
        K = Hp * W * C
        sig = math.sqrt(K) * x["rms"] * W_RMS
        q = self.qout(act, name)
        d = self._w(name, units * K)
        if rows_every > 1:
            self.nweights += units * (H * W * C - K)
            if not self.shapes_only:
                full, row = [0] * (units * H * W * C), W * C
                for o in range(units):
                    for hp in range(Hp):
                        dst = o * H * row + hp * rows_every * row
                        full[dst:dst + row] = d[o * K + hp * row:o * K + (hp + 1) * row]
                d = full
        L = {"op": "fully_connected", "name": name, "units": units, "activation": self._act(act),
             "weights": {"dtype": "int8", "layout": "OI", "scale": f32(1.0 / (sig * x["scale"])), "zero_point": 0,
                         "data": d},
             "bias": self._b(name, units, sig), "output": {"scale": q[0], "zero_point": q[1]}}
        return self._emit(L, self._t([1, 1, units], q, act), None)

    def pool(self, typ, k, s=None, pad="valid", inp=None, name=None):
        kh, kw = (k, k) if isinstance(k, int) else k
        sh, sw = ((s, s) if isinstance(s, int) else s) if s is not None else (kh, kw)
        x = self.T[inp or self.cur]
        H, W, C = x["hwc"]
        name = self._name("pool", name)
        L = {"op": "max_pool2d" if typ == "max" else "average_pool2d", "name": name, "pool": [kh, kw],
             "strides": [sh, sw], "padding": pad}
        f = 1.0 if kh * kw == 1 else 1.2 if typ == "max" else 0.85
        t = dict(x, hwc=[osize(H, kh, sh, pad), osize(W, kw, sw, pad), C], rms=x["rms"] * f)
        return self._emit(L, t, inp)

    def gap(self, inp=None, name=None):
        """a global average pool as TFLite's AVERAGE_POOL_2D writes it (kernel = stride = the map)"""
        H, W, _ = self.T[inp or self.cur]["hwc"]
        return self.pool("avg", (H, W), (H, W), "valid", inp, name)

    def gap1(self, inp=None, name=None):
        """a global average pool as an AVERAGE_POOL_2D of the map's size at stride 1, VALID: one output, the same
        function as kernel = stride = the map (NPUC's NFU pooling takes strides up to 3 and kernels up to 8, V:
        $NCPGEN_WORK/zoo-evidence/probe-gap7; npunet lowers a larger stride as an identity conv, too big over 1024
        channels)"""
        H, W, _ = self.T[inp or self.cur]["hwc"]
        return self.pool("avg", (H, W), 1, "valid", inp, name)

    def mean(self, name=None):
        """a global average pool as Keras' GlobalAveragePooling2D converts: MEAN over H and W, keep_dims false (an
        [1, C] output), the input's quantisation"""
        x = self.T[self.cur]
        name = self._name("pool", name)
        L = {"op": "mean", "name": name, "axis": [1, 2], "keep_dims": False,
             "output": {"scale": x["scale"], "zero_point": x["zp"]}}
        return self._emit(L, dict(x, hwc=[1, 1, x["hwc"][2]], rms=x["rms"] * 0.85), None)

    def gap_conv(self, name="gap"):
        """a global average pool written as the conv npunet lowers it to (identity weights, scale 1/(H*W), bias 0,
        the input's quantisation, ReLU when the input is already >= its zero point)"""
        x = self.T[self.cur]
        H, W, C = x["hwc"]
        self.nweights += C * H * W * C
        data = [] if self.shapes_only else \
            [1 if c == o else 0 for o in range(C) for i in range(H) for j in range(W) for c in range(C)]
        act = "relu" if x["nonneg"] else "none"
        L = {"op": "conv2d", "name": name, "filters": C, "kernel": [H, W], "strides": [1, 1], "padding": "valid",
             "activation": self._act(act),
             "weights": {"dtype": "int8", "layout": "OHWI", "scale": f32(1.0 / (H * W)), "zero_point": 0, "data": data},
             "bias": [0] * C, "output": {"scale": x["scale"], "zero_point": x["zp"]}}
        return self._emit(L, dict(x, hwc=[1, 1, C], rms=x["rms"] * 0.85), None)

    def subsample(self, s, name=None):
        """keep every s-th row and column: a 1x1 stride-s VALID conv with identity weights, weight scale 1 and the
        input's quantisation (multiplier exactly 1: NPUC's slope 256 / 2^8), ReLU when the input is already >= its zero
        point: the same values (NPUC's pooling refuses a stride larger than its kernel, so a 1x1 pool cannot do it)"""
        x = self.T[self.cur]
        H, W, C = x["hwc"]
        name = self._name("subsample", name)
        self.nweights += C * C
        act = "relu" if x["nonneg"] else "none"
        L = {"op": "conv2d", "name": name, "filters": C, "kernel": [1, 1], "strides": [s, s], "padding": "valid",
             "activation": self._act(act),
             "weights": {"dtype": "int8", "layout": "OHWI", "scale": 1.0, "zero_point": 0,
                         "data": [] if self.shapes_only else [1 if c == o else 0 for o in range(C) for c in range(C)]},
             "bias": [0] * C, "output": {"scale": x["scale"], "zero_point": x["zp"]}}
        return self._emit(L, dict(x, hwc=[osize(H, 1, s, "valid"), osize(W, 1, s, "valid"), C]), None)

    def add(self, a, b, act="none", name=None):
        A, B = self.T[a], self.T[b]
        sig = math.hypot(A["rms"] * A["scale"], B["rms"] * B["scale"])  # float rms of the sum
        q = (f32(pow2(4 * sig / 127)), 0)  # NPUC's Eltwise: power-of-two scales only
        name = self._name("add", name)
        L = {"op": "add", "name": name, "inputs": [a, b], "activation": act,
             "output": {"scale": q[0], "zero_point": q[1]}}
        t = {"hwc": list(A["hwc"]), "scale": q[0], "zp": 0, "rms": (0.7071 if act == "relu" else 1.0) * sig / q[0],
             "nonneg": act == "relu"}
        return self._emit(L, t, None)

    def concat(self, ins, name=None):
        ts = [self.T[i] for i in ins]
        if len({(t["scale"], t["zp"]) for t in ts}) != 1:
            raise ValueError("concat: inputs in different quantisations")
        name = self._name("concat", name)
        L = {"op": "concat", "name": name, "inputs": list(ins), "axis": 3,
             "output": {"scale": ts[0]["scale"], "zero_point": ts[0]["zp"]}}
        t = dict(ts[0], hwc=ts[0]["hwc"][:2] + [sum(t["hwc"][2] for t in ts)],
                 rms=sum(t["rms"] for t in ts) / len(ts), nonneg=all(t["nonneg"] for t in ts))
        return self._emit(L, t, None)

    def flatten(self, name=None):
        x = self.T[self.cur]
        name = self._name("reshape", name)
        return self._emit({"op": "reshape", "name": name, "shape": [1, x["hwc"][0] * x["hwc"][1] * x["hwc"][2]]},
                          dict(x), None)

    def softmax(self, name=None):
        name = self._name("softmax", name)
        return self._emit({"op": "softmax", "name": name, "beta": 1.0,
                           "output": {"scale": f32(1 / 256), "zero_point": -128}}, dict(self.T[self.cur]), None)

    # -- results --
    def graph(self, name):
        return {"format": "npu-net/1", "name": name, "description": self.desc, "input": dict(self.inp),
                "layers": self.layers}

    def opts(self):
        return (["--quant", "3"] if self.quant == 3 else []) + (["--allow-none"] if self.none else [])

    def out_hwc(self):
        return self.T[self.cur]["hwc"]


# ---- the models -------------------------------------------------------------------------------------------------------
MODELS = {}


def model(name, family, source, task, adapt=(), gaps=(), expect="exact"):
    """register a model: source = the published architecture, adapt = what was changed for the NPU path (and why),
    gaps = what ncpgen/npunet/NPUC lacks for the published form, expect = refused | npuc-fails | exact"""
    def deco(f):
        MODELS[name] = {"name": name, "family": family, "build": f, "source": source, "task": task,
                        "adapt": list(adapt), "gaps": list(gaps), "expect": expect}
        return f
    return deco


COMMON = ["batch normalisation folded into the conv/FC before it (as the TFLite converter does)",
          "dropout removed (inference)"]


@model("lenet5", "lenet5", "LeNet-5 (LeCun et al. 1998, Gradient-based learning applied to document recognition)",
       "MNIST digits 28x28x1, 10 classes",
       adapt=["tanh/sigmoid -> ReLU", "trainable subsampling -> 2x2 average pool",
              "C3's partial connection table -> fully connected", "the 32x32 padded input -> 28x28 with SAME on C1",
              "C5 (a 5x5 conv over 5x5) and F6 as FULLY_CONNECTED", "RBF output -> FC + softmax"])
def _lenet5(n):
    n.conv(6, 5, pad="same")
    n.pool("avg", 2)
    n.conv(16, 5, pad="valid")
    n.pool("avg", 2)
    n.flatten()
    n.fc(120)
    n.fc(84)
    n.fc(10, act="none")
    n.softmax()


@model("keras-mnist-cnn", "keras-mnist-cnn", "Keras example 'Simple MNIST convnet' (keras.io/examples/vision/mnist_convnet)",
       "MNIST digits 28x28x1, 10 classes", adapt=COMMON[1:])
def _keras_mnist(n):
    n.conv(32, 3, pad="valid")
    n.pool("max", 2)
    n.conv(64, 3, pad="valid")
    n.pool("max", 2)
    n.flatten()
    n.fc(10, act="none")
    n.softmax()


@model("mnist-mlp", "mnist-mlp", "Keras example mnist_mlp (Dense 512-512-10)", "MNIST digits as a 784 vector, 10 classes",
       adapt=COMMON[1:])
def _mnist_mlp(n):
    n.fc(512)
    n.fc(512)
    n.fc(10, act="none")
    n.softmax()


@model("tf-cifar-cnn", "tf-cifar-cnn", "TensorFlow tutorial 'Convolutional Neural Network (CNN)' (tensorflow.org/tutorials/images/cnn)",
       "CIFAR-10 32x32x3, 10 classes")
def _tf_cifar(n):
    n.conv(32, 3, pad="valid")
    n.pool("max", 2)
    n.conv(64, 3, pad="valid")
    n.pool("max", 2)
    n.conv(64, 3, pad="valid")
    n.flatten()
    n.fc(64)
    n.fc(10, act="none")
    n.softmax()


@model("vgg-cifar", "vgg-cifar", "VGG-style CIFAR-10 CNN, three blocks of two 3x3 convs + max pool (Simonyan & Zisserman "
       "2014 pattern; the common 'VGG 3-block' tutorial baseline)", "CIFAR-10 32x32x3, 10 classes", adapt=COMMON)
def _vgg_cifar(n):
    for c in (32, 64, 128):
        n.conv(c, 3)
        n.conv(c, 3)
        n.pool("max", 2)
    n.flatten()
    n.fc(128)
    n.fc(10, act="none")
    n.softmax()


@model("resnet8", "resnet8", "MLPerf Tiny image classification ResNet-8 (resnet_v1_eembc, mlcommons/tiny)",
       "CIFAR-10 32x32x3, 10 classes",
       adapt=COMMON[:1] + ["every activation scale a power of two and every zero point 0 (quant 3): NPUC's Eltwise add "
                           "aligns by shifts only and has no zero-point term",
                           "no activation on the second conv of each block and on the shortcuts: --allow-none (ncpgen's "
                           "corrected GReLU table makes 'none' right on the NPU; NPUC's own table only for "
                           "ceil(Cout/8) even)"])
def _resnet8(n):
    x = n.conv(16, 3)
    y = n.conv(16, 3, inp=x)
    y = n.conv(16, 3, act="none", inp=y)
    x = n.add(x, y, act="relu")
    for c in (32, 64):
        y = n.conv(c, 3, s=2, inp=x)
        y = n.conv(c, 3, act="none", inp=y)
        sc = n.conv(c, 1, s=2, pad="same", act="none", inp=x)   # TFLite clamps the negative pad: VALID
        x = n.add(sc, y, act="relu")
    n.gap(inp=x)
    n.flatten()
    n.fc(10, act="none")
    n.softmax()


@model("ad-autoencoder", "ad-autoencoder", "MLPerf Tiny anomaly detection: the DCASE 2020 task 2 baseline autoencoder "
       "(FC 640-128-128-128-128-8-128-128-128-128-640)", "5 frames x 128 log-mel bins (a 640 vector), reconstruction",
       adapt=COMMON[:1] + ["the linear output layer: --allow-none"])
def _ad(n):
    for u in (128, 128, 128, 128, 8, 128, 128, 128, 128):
        n.fc(u)
    n.fc(640, act="none")


def _kws_cnn(n, npu):
    n.conv(28, (10, 4), pad="valid")
    if npu:
        n.conv(30, (10, 4), s=1, pad="valid")
        n.flatten()
        n.fc(16, act="none", rows_every=2)
    else:
        n.conv(30, (10, 4), s=(2, 1), pad="valid")
        n.flatten()
        n.fc(16, act="none")
    n.fc(128)
    n.fc(12, act="none")
    n.softmax()


KWS_SRC = ("Hello Edge CNN (Zhang et al. 2017, 'Hello Edge: Keyword Spotting on Microcontrollers'; ARM ML-KWS-for-MCU "
           "CNN-S: conv 28 10x4 s1, conv 30 10x4 s2x1, linear 16, FC 128, 12 classes)")


@model("kws-cnn", "kws-cnn", KWS_SRC, "keyword spotting, 49x10 MFCC, 12 classes",
       adapt=COMMON[1:] + ["the linear layers (no activation): --allow-none",
                           "npunet rewrites conv2's stride 2x1 (NPUC fails on unequal strides: 'Couldn't find a correct "
                           "processing strategy for a single node' here, '_strideWidth == _strideHeight' on a 40x7x28 "
                           "probe) exactly and by itself: conv2 at stride 1, the linear layer's weights on its even rows "
                           "and zeros on the odd ones (kws-cnn-npu's network, the same NCP)"])
def _kws(n):
    _kws_cnn(n, False)


@model("kws-cnn-npu", "kws-cnn", KWS_SRC, "keyword spotting, 49x10 MFCC, 12 classes",
       adapt=COMMON[1:] + ["conv2 at stride 1 (31x4x30) and the linear layer's weights on its even rows, zeros on the "
                           "odd ones: exactly the published stride-2x1 network (conv2 does twice the work)",
                           "the linear layers (no activation): --allow-none"])
def _kws_npu(n):
    _kws_cnn(n, True)


@model("micro-speech", "micro-speech", "TensorFlow Lite Micro micro_speech 'tiny_conv' (tensorflow/examples/speech_commands "
       "models.py: conv 8 10x8 s2 SAME, FC 4)", "keyword spotting, 49x40 spectrogram, 4 classes", adapt=COMMON[1:],
       gaps=["the FC over 25x20x8 (as a 25x20 VALID conv): NPUC asserts '(nzvSize & 0xFE00) == 0' "
             "(NPULIRKernelPack::GenBitmasks2020): too many non-zero weights in one kernel slice; npunet should refuse "
             "it or split the FC (lossy: partial sums requantised)"], expect="npuc-fails")
def _micro(n):
    n.conv(8, (10, 8), s=2, pad="same")
    n.flatten()
    n.fc(4, act="none")
    n.softmax()


def _wand(n, npu):
    n.conv(8, (4, 3), pad="same")
    n.pool("max", 3)
    n.conv(16, (4, 1), pad="same")
    if npu:
        n.pool("max", (3, 1), 1)
        n.flatten()
        n.fc(16, rows_every=3)
    else:
        n.pool("max", (3, 1), pad="same")
        n.flatten()
        n.fc(16)
    n.fc(4, act="none")
    n.softmax()


WAND_SRC = "TensorFlow Lite Micro magic_wand gesture CNN (conv 8 4x3, max pool 3x3, conv 16 4x1, max pool 3x1, FC 16, FC 4)"


@model("magic-wand", "magic-wand", WAND_SRC, "gestures, 128 accelerometer samples x 3 axes, 4 classes",
       adapt=COMMON[1:] + ["the output FC: --allow-none",
                           "npunet rewrites the 3x1 stride-3x1 max pool (NPUC asserts 'absDiff < stride', "
                           "PoolNodeTileBackPropagator, on a larger vertical stride) exactly and by itself: the pool at "
                           "stride 1 VALID (40x1), the first FC's weights on rows 0, 3, ..., 39 and zeros elsewhere "
                           "(magic-wand-npu's network, the same NCP)"])
def _wand_pub(n):
    _wand(n, False)


@model("magic-wand-npu", "magic-wand", WAND_SRC, "gestures, 128 accelerometer samples x 3 axes, 4 classes",
       adapt=COMMON[1:] + ["the 3x1 stride-3x1 max pool on the 42x1 map -> a 3x1 stride-1 VALID pool (40x1) and the "
                           "first FC's weights on rows 0, 3, ..., 39, zeros elsewhere: exactly the published network "
                           "(NPUC refuses every stride-3 pool on a width-1 map: 3x1/3x3 strides, VALID or SAME; 3x3 "
                           "stride 3 SAME asserts 'Tried to split a tensor's shape into zero tiles')",
                           "the output FC: --allow-none"])
def _wand_npu(n):
    _wand(n, True)


@model("hello-world", "hello-world", "TensorFlow Lite Micro hello_world sine model (FC 16, FC 16, FC 1)", "sine regression, 1 input",
       adapt=["the linear output: --allow-none"])
def _hello(n):
    n.fc(16)
    n.fc(16)
    n.fc(1, act="none")


def _fire(n, s, e):
    sq = n.conv(s, 1)
    q = n.qout("relu", "expand%d" % len(n.layers))
    a = n.conv(e, 1, inp=sq, out=q)
    b = n.conv(e, 3, inp=sq, out=q)
    return n.concat([a, b])


@model("squeezenet11-96", "squeezenet11-96", "SqueezeNet 1.1 (Iandola et al. 2016; the v1.1 reference: conv1 3x3/2 64, "
       "fire2-9, conv10, global average pool)", "96x96x3 images, 10 classes",
       adapt=COMMON[1:] + ["the two expand convs of a fire module share one output quantisation (NPUC's concat copies "
                           "bytes; TFLite's converter does the same for CONCATENATION)",
                           "1000 ImageNet classes -> 10"])
def _squeeze(n, classes=10, npu_tail=False):
    n.conv(64, 3, s=2, pad="valid")
    n.pool("max", 3, 2)
    _fire(n, 16, 64)
    _fire(n, 16, 64)
    n.pool("max", 3, 2)
    _fire(n, 32, 128)
    _fire(n, 32, 128)
    n.pool("max", 3, 2)
    _fire(n, 48, 192)
    _fire(n, 48, 192)
    _fire(n, 64, 256)
    _fire(n, 64, 256)
    n.conv(classes, 1, name="classifier")
    if npu_tail:
        return                  # the 13x13 global average pool and the softmax on the CPU
    n.mean()
    n.softmax()


def _alex(n, npu):
    if npu:
        n.conv(64, 11, s=2, pad="valid")
        n.subsample(2)
    else:
        n.conv(64, 11, s=4, pad="valid")
    n.pool("max", 3, 2)
    n.conv(192, 5)
    n.pool("max", 3, 2)
    n.conv(384, 3)
    n.conv(256, 3)
    n.conv(256, 3)
    n.pool("max", 3, 2)
    n.flatten()
    n.fc(1024)
    n.fc(1024)
    n.fc(10, act="none")
    n.softmax()


ALEX_SRC = ("AlexNet (Krizhevsky et al. 2012), torchvision widths 64-192-384-256-256: conv 11x11/4, max pools 3/2, three "
            "FC layers")
ALEX_AD = COMMON[1:] + ["LRN dropped (torchvision's AlexNet has none)", "FC 4096-4096 -> 1024-1024 for 10 classes (STL-10)",
                        "explicit padding -> TFLite VALID/SAME (no PAD op)"]


@model("alexnet-96", "alexnet-96", ALEX_SRC, "96x96x3 images (STL-10), 10 classes",
       adapt=ALEX_AD + ["the output FC: --allow-none",
                        "npunet rewrites conv1's stride 4 (NPUC asserts 'Stride > 2 is not supported here!', "
                        "NPULIRKernelMemoryData::IterateOverAllValues) exactly and by itself: conv1 at stride 2 VALID, "
                        "then a 1x1 stride-2 VALID conv with identity weights and multiplier 1 keeping every other row "
                        "and column (alexnet-96-npu's network, the same NCP)"])
def _alex_pub(n):
    _alex(n, False)


@model("alexnet-96-npu", "alexnet-96", ALEX_SRC, "96x96x3 images (STL-10), 10 classes",
       adapt=ALEX_AD + ["conv1 11x11 stride 4 VALID -> stride 2 VALID, then every other row and column kept by a 1x1 "
                        "stride-2 VALID conv with identity weights and multiplier 1: exactly the same output (conv1 "
                        "computes 4x the positions; a 1x1 stride-2 max pool would do it too, but NPUC's pooling refuses "
                        "a stride larger than its kernel)", "the output FC: --allow-none"])
def _alex_npu(n):
    _alex(n, True)


@model("vgg11-64", "vgg11-64", "VGG-11 (configuration A, Simonyan & Zisserman 2014) at half width: 32-M-64-M-128-128-M-"
       "256-256-M-256-256-M, FC 512-512-200", "64x64x3 images (Tiny ImageNet), 200 classes",
       adapt=COMMON[1:] + ["every width halved (64..512 -> 32..256), FC 4096 -> 512, 1000 -> 200 classes"])
def _vgg11(n):
    for c in (32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"):
        if c == "M":
            n.pool("max", 2)
        else:
            n.conv(c, 3)
    n.flatten()
    n.fc(512)
    n.fc(512)
    n.fc(200, act="none")
    n.softmax()


@model("yolov2-tiny-128", "yolov2-tiny-128", "Tiny YOLOv2 (VOC) detector at half width: 3x3 convs 8-16-32-64-128-256 with "
       "2x2 max pools, a 2x2 stride-1 SAME max pool, 512, 512, a 1x1 conv to 5 anchors x (5 + 20) = 125",
       "128x128x3 images, a 4x4 grid of VOC detections",
       adapt=COMMON[:1] + ["leaky ReLU (0.1) -> ReLU (NPUC's GReLU table could hold a leaky slope, not mapped yet)",
                           "every width halved (16..1024 -> 8..512), 416x416 -> 128x128",
                           "the linear detection head: --allow-none"],
       gaps=["leaky ReLU (the published activation): not supported (docs/format/grelu.md: a fractional slope on piece 0)"])
def _yolo(n):
    for c in (8, 16, 32, 64, 128):
        n.conv(c, 3)
        n.pool("max", 2)
    n.conv(256, 3)
    n.pool("max", 2, 1, pad="same")
    n.conv(512, 3)
    n.conv(512, 3)
    n.conv(125, 1, act="none")


def _mnv1(n):
    n.conv(8, 3, s=2, act="relu6")
    for c, s in ((16, 1), (32, 2), (32, 1), (64, 2), (64, 1), (128, 2), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1),
                 (256, 2), (256, 1)):
        n.dw(3, s, act="relu6")
        n.conv(c, 1, act="relu6")
    n.pool("avg", 3, 2)
    n.conv(2, 1, act="none", name="logits")
    n.flatten()
    n.softmax()


MNV1_SRC = ("MobileNet v1 0.25 (Howard et al. 2017), the TensorFlow Lite Micro person_detection model: 13 depthwise-"
            "separable blocks, 8..256 channels, average pool 3x3/2, 1x1 conv to 2 classes")
@model("mobilenet-v1-025", "mobilenet-v1-025", MNV1_SRC, "person detection, 96x96x1 grayscale, 2 classes",
       adapt=COMMON[:1] + ["the logits conv (no activation): --allow-none"])
def _mnv1_pub(n):
    _mnv1(n)


@model("mobilenet-v1-025-dense", "mobilenet-v1-025", MNV1_SRC, "person detection, 96x96x1 grayscale, 2 classes",
       adapt=COMMON[:1] + ["each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function; "
                           "C times the weights of that layer counted dense, the zeros skipped by NPUC's weight format)",
                           "the logits conv: --allow-none"])
def _mnv1_dense(n):
    _mnv1(n)


# MobileNet v2 widths per alpha (keras.applications: _make_divisible by 8): first conv, first block, then the
# inverted residual stages (expansion t, channels c, repeats, stride of the first)
MNV2_035 = (16, 8, ((6, 8, 2, 2), (6, 16, 3, 2), (6, 24, 4, 2), (6, 32, 3, 1), (6, 56, 3, 2), (6, 112, 1, 1)))
MNV2_100 = (32, 16, ((6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)))


def _mnv2(n, widths=MNV2_035, gap1=False):
    first, b0, blocks = widths
    n.conv(first, 3, s=2, act="relu6")
    n.dw(3, 1, act="relu6")
    n.conv(b0, 1, act="none")
    cin = b0
    for t, c, rep, s in blocks:
        for i in range(rep):
            x = n.cur
            n.conv(cin * t, 1, act="relu6", inp=x)
            n.dw(3, s if i == 0 else 1, act="relu6")
            y = n.conv(c, 1, act="none")
            if i > 0:
                n.add(x, y)
            cin = c
    n.conv(1280, 1, act="relu6")
    if gap1:
        n.gap1()
        n.flatten()
    else:
        n.mean()
    n.fc(1000, act="none")
    n.softmax()


MNV2_SRC = ("MobileNet v2 0.35 (Sandler et al. 2018; keras.applications MobileNetV2 alpha 0.35 at 96x96: 17 inverted "
            "residual blocks, 8..112 channels, conv 1280, 1000 classes)")
MNV2_AD = COMMON[:1] + ["every activation scale a power of two except ReLU6's, every zero point 0 (quant 3) for the 10 "
                        "residual ADDs", "ReLU6 quantised 6/127 zp 0 (its clamp is then the int8 saturation)",
                        "the linear bottlenecks and the classifier: --allow-none"]


@model("mobilenet-v2-035", "mobilenet-v2-035", MNV2_SRC, "96x96x3 images, 1000 classes", adapt=MNV2_AD)
def _mnv2_pub(n):
    _mnv2(n)


@model("mobilenet-v2-035-dense", "mobilenet-v2-035", MNV2_SRC, "96x96x3 images, 1000 classes",
       adapt=MNV2_AD + ["each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function; "
                        "1.66 M -> 6.74 M weights counted dense, but NPUC's weight format skips the zeros with bitmaps)"])
def _mnv2_dense(n):
    _mnv2(n)


def _dscnn(n):
    n.conv(64, (10, 4), s=2, pad="same")
    for _ in range(4):
        n.dw(3, 1)
        n.conv(64, 1)
    if n.dense:
        n.gap_conv()
    else:
        n.gap()
    n.flatten()
    n.fc(12, act="none")
    n.softmax()


DSCNN_SRC = ("DS-CNN (Zhang et al. 2017, Hello Edge), the MLPerf Tiny keyword-spotting model: conv 64 10x4/2, 4 x "
             "(depthwise 3x3 + pointwise 64), average pool 25x5, FC 12")


@model("ds-cnn", "ds-cnn", DSCNN_SRC, "keyword spotting, 49x10 MFCC, 12 classes",
       adapt=COMMON + ["the output FC: --allow-none"])
def _dscnn_pub(n):
    _dscnn(n)


@model("ds-cnn-dense", "ds-cnn", DSCNN_SRC, "keyword spotting, 49x10 MFCC, 12 classes",
       adapt=COMMON + ["each depthwise 3x3 conv -> a dense 3x3 conv with block-diagonal weights (the same function)",
                       "the 25x5 global average pool written as the identity conv npunet lowers it to",
                       "the output FC: --allow-none"])
def _dscnn_dense(n):
    _dscnn(n)


# ---- standard resolutions: the common ImageNet models at the input sizes they are published at ----------------------
def _mnv1_std(n, alpha):
    n.conv(int(32 * alpha), 3, s=2, act="relu6")
    for c, s in ((64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)) + ((512, 1),) * 5 + ((1024, 2), (1024, 1)):
        n.dw(3, s, act="relu6")
        n.conv(int(c * alpha), 1, act="relu6")
    n.gap1()
    n.conv(1000, 1, act="none", name="logits")
    n.flatten()
    n.softmax()


MNV1_STD_SRC = ("MobileNet v1 %s (Howard et al. 2017; keras.applications MobileNet alpha %s at 224x224: conv 3x3/2, 13 "
                "depthwise-separable blocks, %d..%d channels, global average pool, 1x1 conv to 1000 classes)")
GAP1_AD = ("the 7x7 global average pool as a 7x7 stride-1 VALID average pool (one output: the same function; NPUC's "
           "NFU pools take strides up to 3, and npunet's identity-conv form of a stride-7 pool is %d x %d x 7 x 7 "
           "weights)")
MNV1_STD_AD = COMMON + ["ReLU6 at TFLite's 6/255 (zp -128)", GAP1_AD % (1024, 1024), "the logits conv: --allow-none"]


@model("mobilenet-v1-100-224", "mobilenet-v1-100-224", MNV1_STD_SRC % ("1.0", "1.0", 32, 1024),
       "ImageNet 224x224x3, 1000 classes", adapt=MNV1_STD_AD)
def _mnv1_100_224(n):
    _mnv1_std(n, 1.0)


@model("mobilenet-v1-050-224", "mobilenet-v1-050-224", MNV1_STD_SRC % ("0.5", "0.5", 16, 512),
       "ImageNet 224x224x3, 1000 classes", adapt=MNV1_STD_AD[:3] + [GAP1_AD % (512, 512)] + MNV1_STD_AD[4:])
def _mnv1_050_224(n):
    _mnv1_std(n, 0.5)


MNV2_STD_SRC = ("MobileNet v2 %s (Sandler et al. 2018; keras.applications MobileNetV2 alpha %s at 224x224: 17 inverted "
                "residual blocks, %s channels, conv 1280, 1000 classes)")


@model("mobilenet-v2-100-224", "mobilenet-v2-100-224", MNV2_STD_SRC % ("1.0", "1.0", "16..320"),
       "ImageNet 224x224x3, 1000 classes", adapt=MNV2_AD[:3] + [GAP1_AD % (1280, 1280)] + MNV2_AD[3:])
def _mnv2_100_224(n):
    _mnv2(n, MNV2_100, gap1=True)


@model("mobilenet-v2-035-224", "mobilenet-v2-035-224", MNV2_STD_SRC % ("0.35", "0.35", "8..112"),
       "ImageNet 224x224x3, 1000 classes", adapt=MNV2_AD[:3] + [GAP1_AD % (1280, 1280)] + MNV2_AD[3:])
def _mnv2_035_224(n):
    _mnv2(n, MNV2_035, gap1=True)


def _resnet18(n):
    n.conv(64, 7, s=2)
    x = n.pool("max", 3, 2, pad="same")
    for c, s in ((64, 1), (128, 2), (256, 2), (512, 2)):
        for i in range(2):
            st = s if i == 0 else 1
            y = n.conv(c, 3, s=st, inp=x)
            y = n.conv(c, 3, act="none", inp=y)
            sc = n.conv(c, 1, s=st, pad="same", act="none", inp=x) if st != 1 else x   # TFLite: VALID (negative pad)
            x = n.add(sc, y, act="relu")
    n.gap(inp=x)
    n.flatten()
    n.fc(1000, act="none")
    n.softmax()


RESNET18_SRC = ("ResNet-18 (He et al. 2016; torchvision resnet18: conv 7x7/2 64, max pool 3x3/2, 4 stages of 2 basic "
                "blocks, 64..512 channels, 1x1 stride-2 projection shortcuts, global average pool, FC 1000)")
RESNET18_AD = COMMON[:1] + ["every activation scale a power of two and every zero point 0 (quant 3): NPUC's Eltwise add "
                            "aligns by shifts only and has no zero-point term",
                            "explicit padding -> TFLite SAME (pad bottom/right for the stride-2 layers)",
                            "no activation on the second conv of each block and on the shortcuts: --allow-none"]


@model("resnet18-224", "resnet18-224", RESNET18_SRC, "ImageNet 224x224x3, 1000 classes", adapt=RESNET18_AD)
def _resnet18_224(n):
    _resnet18(n)


@model("resnet18-96", "resnet18-96", RESNET18_SRC, "96x96x3 images, 1000 classes", adapt=RESNET18_AD)
def _resnet18_96(n):
    _resnet18(n)


@model("squeezenet11-224", "squeezenet11-224", "SqueezeNet 1.1 (Iandola et al. 2016; the v1.1 reference: conv1 3x3/2 64, "
       "fire2-9, conv10 1000, global average pool)", "ImageNet 224x224x3, 1000 classes",
       adapt=COMMON[1:] + ["the two expand convs of a fire module share one output quantisation (NPUC's concat copies "
                           "bytes; TFLite's converter does the same for CONCATENATION)"])
def _squeeze224(n):
    _squeeze(n, 1000)


@model("squeezenet11-224-npu", "squeezenet11-224", "SqueezeNet 1.1 (Iandola et al. 2016; the v1.1 reference: conv1 3x3/2 "
       "64, fire2-9, conv10 1000, global average pool)", "ImageNet 224x224x3, 1000 classes",
       adapt=COMMON[1:] + ["the two expand convs of a fire module share one output quantisation (NPUC's concat copies "
                           "bytes; TFLite's converter does the same for CONCATENATION)",
                           "the NPU part ends at conv10 (13x13x1000); the global average pool and the softmax run on "
                           "the CPU: NPUC's NFU pools take kernels up to 8 (V: 8x8 compiles, 9..13 'ncp_nfu_pooling: "
                           "Invalid parameters!') and the identity-conv form is 1000 x 1000 x 13 x 13 weights"],
       expect="exact")
def _squeeze224_npu(n):
    _squeeze(n, 1000, npu_tail=True)


# the input of each family: shape, quantisation (scale, zero point), quant mode, power-of-two activations
INPUTS = {
    "lenet5": ((28, 28, 1), 1 / 255, -128, 1, False), "keras-mnist-cnn": ((28, 28, 1), 1 / 255, -128, 1, False),
    "mnist-mlp": ((784,), 1 / 255, -128, 1, False), "tf-cifar-cnn": ((32, 32, 3), 1 / 255, -128, 1, False),
    "vgg-cifar": ((32, 32, 3), 1 / 255, -128, 1, False), "resnet8": ((32, 32, 3), 1 / 128, 0, 3, True),
    "ad-autoencoder": ((640,), 0.4, 10, 1, False), "kws-cnn": ((49, 10, 1), 0.6, 80, 1, False),
    "micro-speech": ((49, 40, 1), 26 / 255, -128, 1, False), "magic-wand": ((128, 3, 1), 2000 / 127, 0, 1, False),
    "hello-world": ((1,), 2 * math.pi / 255, -128, 1, False),
    "squeezenet11-96": ((96, 96, 3), 1 / 255, -128, 1, False), "alexnet-96": ((96, 96, 3), 1 / 255, -128, 1, False),
    "vgg11-64": ((64, 64, 3), 1 / 255, -128, 1, False), "yolov2-tiny-128": ((128, 128, 3), 1 / 255, -128, 1, False),
    "mobilenet-v1-025": ((96, 96, 1), 1 / 255, -128, 1, False), "mobilenet-v2-035": ((96, 96, 3), 1 / 128, 0, 3, True),
    "ds-cnn": ((49, 10, 1), 0.6, 80, 1, False),
    "mobilenet-v1-100-224": ((224, 224, 3), 1 / 255, -128, 1, False),
    "mobilenet-v1-050-224": ((224, 224, 3), 1 / 255, -128, 1, False),
    "mobilenet-v2-100-224": ((224, 224, 3), 1 / 128, 0, 3, True),
    "mobilenet-v2-035-224": ((224, 224, 3), 1 / 128, 0, 3, True),
    "resnet18-224": ((224, 224, 3), 1 / 128, 0, 3, True), "resnet18-96": ((96, 96, 3), 1 / 128, 0, 3, True),
    "squeezenet11-224": ((224, 224, 3), 1 / 255, -128, 1, False),
}


def build(name, shapes_only=False):
    m = MODELS[name]
    hwc, sc, zp, q, p2 = INPUTS[m["family"]]
    n = Net(m["family"], hwc, sc, zp, quant=q, pow2=p2, dense=name.endswith("-dense"), shapes_only=shapes_only,
            desc="models/zoo.py %s: %s" % (name, m["source"]))
    m["build"](n)
    return n


def ops_summary(g):
    c = {}
    for L in g["layers"]:
        k = L["op"]
        if k == "conv2d":
            k = "conv2d %dx%d%s" % (L["kernel"][0], L["kernel"][1], "/%d" % L["strides"][0] if L["strides"] != [1, 1] else "")
            if L["strides"][0] != L["strides"][1]:
                k = "conv2d %dx%d/%dx%d" % (L["kernel"][0], L["kernel"][1], L["strides"][0], L["strides"][1])
        elif k == "depthwise_conv2d":
            k = "depthwise %dx%d" % tuple(L["kernel"])
        c[k] = c.get(k, 0) + 1
    return ", ".join("%s x%d" % (k, v) if v > 1 else k for k, v in c.items())


# ---- write --------------------------------------------------------------------------------------------------------
def _same_lowering(a, b):
    """two lowerings of one model (lower() of its JSON and of its TFLite) are the same network: every field but the
    network's name"""
    return {k: v for k, v in a.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}


def tflite_for(g, n):
    """the model as TFLite through npunet.graph_to_tflite, or (None, why): kept only when npunet lowers the TFLite to
    the same lowered network as the JSON (or refuses both the same way)"""
    try:
        tfl = npunet.graph_to_tflite(g)
    except npunet.Refused as e:
        return None, "graph_to_tflite: %s" % e
    q = n.quant
    res = []
    for src in (lambda: g, lambda: npunet.tflite_to_graph(npunet.read_tflite(tfl), "model.tflite")):
        try:
            res.append(npunet.lower(src(), quant=q, allow_none=n.none))
        except npunet.Refused as e:
            res.append("refused: %s" % e)
    if isinstance(res[0], str) or isinstance(res[1], str):
        return (tfl, "ok (both refused)") if res[0] == res[1] else (None, "round trip: %s / %s" % tuple(
            r if isinstance(r, str) else "lowered" for r in res))
    if not _same_lowering(*res):
        return None, "round trip: the TFLite lowers to another network"
    return tfl, "ok"


def write_model(name, d, tflite=True, base=None):
    n = build(name)
    g = n.graph(name)
    os.makedirs(d, exist_ok=True)
    jp = os.path.join(d, (base or name) + ".json")
    with open(jp, "w") as f:
        json.dump(g, f)
    why = "skipped"
    if tflite:
        tfl, why = tflite_for(g, n)
        if tfl is not None:
            open(os.path.join(d, (base or name) + ".tflite"), "wb").write(tfl)
    return n, g, jp, why


def cmd_write(d, names, tflite):
    for name in names:
        n, g, jp, why = write_model(name, d, tflite)
        print("zoo: %-24s %s (%d layers, %.2f M weights, %s); tflite: %s" % (
            name, jp, len(g["layers"]), n.nweights / 1e6, " ".join(n.opts()) or "quant 1", why))
    return 0


# ---- compile + reference ---------------------------------------------------------------------------------------------
def ref_input(name, low):
    C, H, W = low["input"]["chw"]
    return _stream(MODELS[name]["family"] + "/input", C * H * W)


def compile_one(name):
    d = os.path.join(ZOO, name)
    shutil.rmtree(d, ignore_errors=True)
    n, g, jp, why = write_model(name, d, tflite=True, base="model")
    t0 = time.time()
    p = subprocess.run(["bash", NPUC_LOCAL, jp, d] + n.opts(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    with open(os.path.join(d, "npuc.txt"), "w") as f:
        f.write(p.stdout + "npuc-local exit %d, %.1f s\n" % (p.returncode, dt))
    info = {"model": name, "opts": n.opts(), "weights": n.nweights, "tflite": why, "npuc_s": round(dt, 1)}
    low_p = os.path.join(d, "lowered.json")
    if os.path.exists(low_p):
        low = json.load(open(low_p))
        open(os.path.join(d, "input.bin"), "wb").write(ref_input(name, low))
        t0 = time.time()
        r = subprocess.run([sys.executable, "-I", NPUNET, "ref", low_p, os.path.join(d, "input.bin"),
                            os.path.join(d, "expected.bin")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        info["ref_s"] = round(time.time() - t0, 1)
        with open(os.path.join(d, "ref.txt"), "w") as f:
            f.write(r.stdout + "npunet ref exit %d, %.1f s\n" % (r.returncode, info["ref_s"]))
    json.dump(info, open(os.path.join(d, "compile.json"), "w"), indent=1)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    return "%-24s %s" % (name, last.replace(d + "/", ""))


def cmd_compile(names, jobs):
    os.makedirs(ZOO, exist_ok=True)
    with ThreadPoolExecutor(jobs) as ex:
        for line in ex.map(compile_one, names):
            print(line, flush=True)
    return 0


# ---- check ------------------------------------------------------------------------------------------------------------
def _read(p):
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def status(name, run_check=True):
    """lowering, NPUC, ncpgen check and the reference for a compiled model (its folder under ZOO); run_check=False
    reuses the last check (check.json) while the NCP it judged is still there"""
    d = os.path.join(ZOO, name)
    cj, ncp = os.path.join(d, "check.json"), os.path.join(d, "net.ncp")
    if not run_check and os.path.exists(cj) and os.path.exists(ncp):
        s = json.load(open(cj))
        if s.get("sha256") == _sha(ncp):
            return s
    s = {"model": name, "lower": "-", "npuc": "-", "ncpgen": "-", "ref": "-", "verdict": "not compiled"}
    if not os.path.isdir(d):
        return s
    info = json.loads(_read(os.path.join(d, "compile.json")) or "{}")
    s["opts"] = info.get("opts", [])
    s["weights"] = info.get("weights")
    s["tflite"] = info.get("tflite", "-")
    low_txt = _read(os.path.join(d, "lower.txt"))
    if os.path.exists(os.path.join(d, "lowered.json")):
        npl = [l for l in low_txt.splitlines() if l.startswith("lower: PASS")]
        m = re.search(r"(\d+) NPU layers", npl[0]) if npl else None
        s["lower"] = "ok, %s NPU layers" % (m.group(1) if m else "?")
    else:
        msg = [l for l in low_txt.splitlines() if l.startswith("npunet: FAIL: ")]
        s["lower"] = "refused: " + (msg[0][len("npunet: FAIL: "):] if msg else low_txt.strip()[-200:])
        s["verdict"] = "refused"
        return s
    ncp = os.path.join(d, "net.ncp")
    if os.path.exists(ncp):
        s["npuc"] = "PASS, %d KB, %s s" % ((os.path.getsize(ncp) + 1023) // 1024, info.get("npuc_s", "?"))
        s["sha256"] = _sha(ncp)
    else:
        log = _read(os.path.join(d, "npuc.txt")) + _read(os.path.join(d, "log.txt"))
        runs = os.path.join(d, ".runs")
        for r in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
            log += _read(os.path.join(runs, r, "log.txt"))
        a = re.findall(r'assertion "(.*)" failed', log)
        inv = re.findall(r"(\w+: Invalid parameters!)\s*\n\s*\n?([^\n]+)", log)
        s["npuc"] = "FAIL: " + (a[0] if a else "%s %s" % inv[0] if inv else
                                (log.strip().splitlines() or ["?"])[-1][:200])
        s["verdict"] = "npuc-fails"
    ref = _read(os.path.join(d, "ref.txt"))
    if ref.startswith("ref: "):
        y = open(os.path.join(d, "expected.bin"), "rb").read()
        s["ref"] = "%s, %d B, %s s" % (_sha(os.path.join(d, "expected.bin"))[:12], len(y), info.get("ref_s", "?"))
        s["ref_sha256"] = _sha(os.path.join(d, "expected.bin"))
    elif ref:
        s["ref"] = "FAIL: " + ref.strip().splitlines()[0][:200]
    if os.path.exists(ncp) and run_check:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", NCPGEN, "check", d + "/"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        out = p.stdout.strip().splitlines()
        row = [l for l in out if l.startswith(name + " ") or l.startswith(name + "\t")]
        line = row[0][len(name):].strip() if row else (out[-1] if out else "?")
        s["ncpgen"] = ("EXACT" if line.endswith("EXACT") else line) + " (%.0f s)" % (time.time() - t0)
        s["ncpgen_notes"] = [l.strip() for l in out if l.strip().startswith("note:")]
        s["verdict"] = "exact" if line.endswith("EXACT") else "differs"
        json.dump(s, open(os.path.join(d, "check.json"), "w"), indent=1)
    return s


def cmd_check(names, jobs, md, cached=False):
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(lambda x: status(x, not cached), names))
    if md:
        print("| model | input | options | lowering | NPUC | ncpgen check | CPU reference |")
        print("|---|---|---|---|---|---|---|")
        for s in rows:
            n = build(s["model"], shapes_only=True)
            print("| %s | %s | %s | %s | %s | %s | %s |" % (
                s["model"], "x".join(map(str, n.T["input"]["hwc"])), " ".join(s.get("opts") or []) or "-",
                s["lower"].replace("|", "/"), s["npuc"].replace("|", "/"), s["ncpgen"].replace("|", "/"),
                s["ref"]))
    else:
        for s in rows:
            print("%-24s %-12s lower %s | npuc %s | ncpgen %s | ref %s" % (
                s["model"], s["verdict"], s["lower"], s["npuc"], s["ncpgen"], s["ref"]))
            for x in s.get("ncpgen_notes", []):
                print("    " + x)
    v = [s["verdict"] for s in rows]
    print("zoo check: %d models: exact %d, differs %d, NPUC fails %d, refused by npunet %d, not compiled %d" % (
        len(v), v.count("exact"), v.count("differs"), v.count("npuc-fails"), v.count("refused"), v.count("not compiled")))
    return 1 if "differs" in v else 0


# ---- golden ------------------------------------------------------------------------------------------------------------
def golden_rows(work=WORK):
    """(name, spec, options, sha256) for every zoo model whose last `check` was EXACT on the NCP still there"""
    rows = []
    for name in MODELS:
        d = os.path.join(work, "sweep", "zoo", name)
        c, ncp = os.path.join(d, "check.json"), os.path.join(d, "net.ncp")
        if not (os.path.exists(c) and os.path.exists(ncp)):
            continue
        s = json.load(open(c))
        sha = _sha(ncp)
        if s.get("verdict") == "exact" and s.get("sha256") == sha:
            rows.append(("zoo-" + name, "zoo:" + name, " ".join(s.get("opts") or []), sha))
    return rows


# ---- list ---------------------------------------------------------------------------------------------------------------
def cmd_list(names, md):
    for name in names:
        m = MODELS[name]
        n = build(name, shapes_only=True)
        g = n.graph(name)
        if md:
            print("### %s\n" % name)
            print("- **Source**: %s" % m["source"])
            print("- **Task / input**: %s; input %s int8 (scale %.6g, zero point %d)" % (
                m["task"], "x".join(map(str, n.T["input"]["hwc"])), n.T["input"]["scale"], n.T["input"]["zp"]))
            print("- **Ops**: %s; %.2f M weights; output %s; `%s`" % (
                ops_summary(g), n.nweights / 1e6, "x".join(map(str, n.out_hwc())), " ".join(n.opts()) or "--quant 1"))
            if m["adapt"]:
                print("- **Adaptations**: " + "; ".join(m["adapt"]))
            if m["gaps"]:
                print("- **Missing for the published form**: " + "; ".join(m["gaps"]))
            st = status(name, run_check=False)
            if st["verdict"] != "not compiled":
                print("- **Status**: %s (lowering: %s; NPUC: %s; ncpgen check: %s)" % (
                    st["verdict"], st["lower"], st["npuc"], st["ncpgen"]))
            print()
        else:
            print("%-24s %-14s %-9s %5.2f M weights  %s" % (name, "x".join(map(str, n.T["input"]["hwc"])),
                                                           " ".join(n.opts()) or "-", n.nweights / 1e6, ops_summary(g)))
    return 0


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a.pop(0)

    def flag(f):
        if f in a:
            a.remove(f)
            return True
        return False

    def opt(f, d):
        if f in a:
            i = a.index(f)
            v = a[i + 1]
            del a[i:i + 2]
            return v
        return d

    if cmd == "write":
        if not a:
            sys.exit(__doc__)
        d = a.pop(0)
        tfl = not flag("--no-tflite")
        names = a or list(MODELS)
    else:
        jobs = int(opt("--jobs", 4))
        md = flag("--md")
        cached = flag("--cached")
        names = a or list(MODELS)
    for x in names:
        if x not in MODELS:
            sys.exit("zoo: unknown model %s (zoo.py list)" % x)
    if cmd == "list":
        sys.exit(cmd_list(names, md))
    if cmd == "write":
        sys.exit(cmd_write(d, names, tfl))
    if cmd == "compile":
        sys.exit(cmd_compile(names, jobs))
    if cmd == "check":
        sys.exit(cmd_check(names, jobs, md, cached))
    if cmd == "golden":
        for r in golden_rows():
            print("|".join(r))
        sys.exit(0)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
