#!/usr/bin/env python3
"""camera: the published camera models (person detection, detectors, face, pose) and what ncpgen does with them
(models/camera.md). Nothing third-party is stored in the repository: the models are fetched into the work folder
($NCPGEN_WORK/models/<name>/) and checked against the sha256 recorded here.

  camera.py list                                   the models: URL, sha256, licence
  camera.py fetch [model...]                       download into $NCPGEN_WORK/models/<name>/ (sha256 checked)
  camera.py tflref <model.tflite> <input.bin> <out.bin> [--stop TENSOR]
        TFLite's own integer arithmetic on the model, op by op (reference kernels: per-channel QuantizeMultiplier,
        SaturatingRoundingDoublingHighMul, RoundingDivideByPOT): the output tensor's bytes, or tensor TENSOR's
  camera.py person [--inputs N]                    person_detect: npunet's CPU reference (folded, and with
        per-channel shifts) against tflref on TFLM's two test images and N seeded random inputs
  camera.py addcheck <model.tflite> [--input TENSOR] [--output TENSOR] [--inputs N] [--all]
        ADD with any scales: npunet's --add-as-conv arithmetic (conv_add) against TFLite's ADD (tfl_add), the rest of
        the model TFLite's: per ADD on the activations TFLite computes, and at the output (the conv's error
        propagated) on the person image and N seeded random inputs; --all: also the exact real sum (real_add) and
        the power-of-two alternative through NPUC's Eltwise (pow2_add)

tflref covers what the camera models' NPU parts use: CONV_2D, DEPTHWISE_CONV_2D (any multiplier), AVERAGE_POOL_2D,
MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, ADD (8-bit, no broadcasting), PAD (H and W); int8 (per-channel or per-tensor weights) and uint8 (per-tensor). Anything else
stops it with the op's name (run it with --stop before that op).
"""
import argparse
import hashlib
import math
import os
import random
import struct
import sys
import urllib.request
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from ncpgen import npunet  # noqa: E402

WORK = os.environ.get("NCPGEN_WORK", os.path.join(ROOT, "work"))
CORAL = "https://github.com/google-coral/test_data/raw/"

# name: (url, sha256, licence, what). The models people run on cameras, as published (models/camera.md).
MODELS = {
    "person_detect": (
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/models/person_detect.tflite",
        "808cfdfc0cf3a6fa6f6fa26bfa379ea97c16d5db7334637766e39c3408502e9d", "Apache-2.0 (tflite-micro)",
        "Visual Wake Words person detector, MobileNet v1 0.25, 96x96x1 int8"),
    "person_detect.person.bmp": (
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp",
        "b50a4d51e86551ea13038c96c222b9c3aaf653f5946211e5e5fd3e60b029e4d2", "Apache-2.0 (tflite-micro)",
        "TFLM's test image with a person (96x96 8-bit BMP)"),
    "person_detect.no_person.bmp": (
        "https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/testdata/no_person.bmp",
        "2322df94e6788b05e4051e531f7a3a95b6db54624d170ebc9af2f1d5a73e9f79", "Apache-2.0 (tflite-micro)",
        "TFLM's test image without a person"),
    "ssd_mobilenet_v1_coco_quant": (
        CORAL + "master/ssd_mobilenet_v1_coco_quant_postprocess.tflite",
        "020d9f0a3e8b84e13cb1dbe70a756ef4dd5884bfc50e77f43c7bac84e791dd9d", "Apache-2.0 (google-coral/test_data)",
        "SSD MobileNet v1 COCO, TF1 quantisation-aware, 300x300 uint8, TFLite_Detection_PostProcess"),
    "ssd_mobilenet_v2_coco_quant": (
        CORAL + "master/ssd_mobilenet_v2_coco_quant_postprocess.tflite",
        "42fb3d70ffb7bb37dd518f730f7be784b831c2078f30c497d0019cc2e987fa26", "Apache-2.0 (google-coral/test_data)",
        "SSD MobileNet v2 COCO, TF1 quantisation-aware, 300x300 uint8, post-process"),
    "ssdlite_mobiledet_coco_qat": (
        CORAL + "release-frogfish/ssdlite_mobiledet_coco_qat_postprocess.tflite",
        "90bb33a634e041914cc1819aa5df99818e6c396c4d2db952c0fd7a9cffc4724f", "Apache-2.0 (google-coral/test_data)",
        "SSDLite MobileDet COCO, 320x320 uint8: Frigate's default CPU model (docker/main/Dockerfile, cpu_model.tflite)"),
    "efficientdet_lite0_coral": (
        CORAL + "master/efficientdet_lite0_320_ptq.tflite",
        "33a3b622c7cac0762f96089353cd61495f3e993968d133af7871bfc2d5396704", "Apache-2.0 (google-coral/test_data)",
        "EfficientDet-Lite0, 320x320, int8 per-channel (uint8 input + QUANTIZE), post-process"),
    "efficientdet_lite0_mediapipe": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
        "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb", "Apache-2.0 (MediaPipe model card)",
        "EfficientDet-Lite0 int8, MediaPipe's object detector (no post-process op: 2 float outputs)"),
    "blazeface_short_range": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f", "Apache-2.0 (MediaPipe model card)",
        "BlazeFace short range, 128x128: float16 weights, float32 I/O (no quantised release)"),
    "ssd_mobilenet_v2_face_quant": (
        CORAL + "master/ssd_mobilenet_v2_face_quant_postprocess.tflite",
        "4c33a81508aed53a2b55b91693957d777f051e24cd62f69270f512957d422aa2", "Apache-2.0 (google-coral/test_data)",
        "face SSD MobileNet v2, 320x320 uint8, post-process"),
    "posenet_mobilenet_v1_075_353_481": (
        "https://github.com/google-coral/project-posenet/raw/master/models/mobilenet/"
        "posenet_mobilenet_v1_075_353_481_quant_decoder.tflite",
        "7519aebd2c9052089bdcfaf222b03db1212f05f5942660fc076446a13708b042", "Apache-2.0 (google-coral/project-posenet)",
        "PoseNet MobileNet v1 0.75, 353x481 uint8, PosenetDecoderOp"),
    "movenet_lightning_coral": (
        CORAL + "master/movenet_single_pose_lightning_ptq.tflite",
        "cd7cc22fa946e5d146a7b98d496853e1923e22828d3972d579973f27f91bb105", "Apache-2.0 (google-coral/test_data)",
        "MoveNet single-pose Lightning, 192x192, int8 per-channel"),
    "yolo_fastest_192_face_v4_emza": (
        "https://media.githubusercontent.com/media/emza-vs/ModelZoo/master/object_detection/yolo-fastest_192_face_v4.tflite",
        "e94bcdb011784bead70ab0c0e9d2dae1a9ea5f103b43e1e6fac3019302cf71ab", "Apache-2.0 (emza-vs/ModelZoo)",
        "YOLO-Fastest face detector, 192x192x1 int8, 2 output heads"),
}


def model_path(name):
    """$NCPGEN_WORK/models/<model>/<file>: a test file "<model>.<file>" sits in <model>/testdata/"""
    url = MODELS[name][0]
    base = name.split(".")[0]
    sub = os.path.join(base, "testdata") if "." in name else base
    return os.path.join(WORK, "models", sub, os.path.basename(url))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def cmd_list(_a):
    for n, (url, h, lic, what) in MODELS.items():
        print("%-28s %s\n  %s\n  sha256 %s, %s" % (n, what, url, h or "-", lic))


def cmd_fetch(a):
    bad = 0
    for n in a.models or list(MODELS):
        url, h = MODELS[n][:2]
        p = model_path(n)
        if not os.path.isfile(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "ncpgen-camera"})
            data = urllib.request.urlopen(req, timeout=120).read()
            open(p + ".part", "wb").write(data)
            os.replace(p + ".part", p)
        got = sha256(p)
        ok = not h or got == h
        bad += not ok
        print("%s %s %s%s" % ("ok " if ok else "BAD", p, got, "" if ok else " (expected %s)" % h))
    return 1 if bad else 0


# ---- TFLite's integer arithmetic (tensorflow/lite/kernels/internal/common.h, quantization_util.cc) --------------------
INT32_MIN, INT32_MAX = -2 ** 31, 2 ** 31 - 1


def quantize_multiplier(d):
    """QuantizeMultiplier(double): (q31 multiplier, shift), d = m * 2^(shift - 31)"""
    if d == 0:
        return 0, 0
    q, e = math.frexp(d)
    qf = int(math.floor(abs(q) * (1 << 31) + 0.5)) * (1 if q > 0 else -1)   # TfLiteRound: half away from zero
    if qf == 1 << 31:
        qf //= 2
        e += 1
    if e < -31:
        return 0, 0
    return qf, e


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul"""
    if a == b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    v = ab + nudge
    return int(v / (1 << 31)) if v >= 0 else -((-v) // (1 << 31))     # C's truncating division


def rdbypot(x, e):
    """RoundingDivideByPOT"""
    mask = (1 << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)


def mbqm(x, m, shift):
    """MultiplyByQuantizedMultiplier (TFLITE_SINGLE_ROUNDING off: the reference kernels' default)"""
    left, right = max(shift, 0), max(-shift, 0)
    return rdbypot(srdhm(x * (1 << left), m), right)


def _f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _pads(n, k, s, padding):
    """(output size, pad before) as TFLite's ComputePaddingHeightWidth (dilation 1)"""
    if padding == "same":
        o = (n + s - 1) // s
        return o, max((o - 1) * s + k - n, 0) // 2
    return (n - k + s) // s, 0


def _act_range(act, zp, scale, dtype):
    lo, hi = (-128, 127) if dtype == "int8" else (0, 255)
    if act == "relu":
        lo = max(lo, zp)
    elif act == "relu6":
        lo, hi = max(lo, zp), min(hi, zp + int(math.floor(_f32(6.0 / scale) + 0.5)))
    elif act == "relu_n1_to_1":
        lo, hi = max(lo, zp + int(math.floor(-1.0 / scale + 0.5))), min(hi, zp + int(math.floor(1.0 / scale + 0.5)))
    elif act != "none":
        raise ValueError("fused activation %s" % act)
    return lo, hi


def _vals(t):
    fmt = {"int8": "b", "uint8": "B", "int32": "i"}[t["dtype"]]
    n = len(t["data"]) // struct.calcsize(fmt)
    return list(struct.unpack("<%d%s" % (n, fmt), t["data"][:n * struct.calcsize(fmt)]))


def _conv(tf, o, x, depthwise, npuc=False):
    """CONV_2D / DEPTHWISE_CONV_2D, reference_integer_ops (per-channel) or reference_ops (uint8 per-tensor): NHWC.
    npuc: NPUC's multiplier instead (one 8-bit slope from float32 scales, round half away: npunet's reference),
    per-tensor weight scales only"""
    T = tf["tensors"]
    ti, tw, to = T[o["inputs"][0]], T[o["inputs"][1]], T[o["outputs"][0]]
    tb = T[o["inputs"][2]] if len(o["inputs"]) > 2 and o["inputs"][2] >= 0 else None
    _, H, W, C = ti["shape"]
    _, Ho, Wo, Co = to["shape"]
    opt = o["opt"]
    if opt.get("dil_h", 1) != 1 or opt.get("dil_w", 1) != 1:
        raise ValueError("dilation")
    w = _vals(tw)
    b = _vals(tb) if tb is not None and tb["data"] else [0] * Co
    if depthwise:
        _, kh, kw, _ = tw["shape"]
        M = Co // C
    else:
        _, kh, kw, _ = tw["shape"]
    sh, sw = opt["stride_h"], opt["stride_w"]
    ho, pt = _pads(H, kh, sh, opt["padding"])
    wo, pl = _pads(W, kw, sw, opt["padding"])
    assert (ho, wo) == (Ho, Wo), (ho, wo, Ho, Wo)
    zi, zo = ti["zero_point"][0], to["zero_point"][0]
    si, so = ti["scale"][0], to["scale"][0]
    wsc = tw["scale"] if len(tw["scale"]) > 1 else tw["scale"] * Co
    zw = tw["zero_point"][0] if tw["zero_point"] else 0
    if len(set(tw["zero_point"] or [0])) > 1:
        raise ValueError("per-channel weight zero points")
    mult = [quantize_multiplier(float(si) * float(ws) / float(so)) for ws in wsc]   # in double, as TFLite
    if npuc:
        if len(set(wsc)) > 1:
            raise ValueError("NPUC's multiplier: per-channel weight scales")
        sl, sf = npunet.npuc_multiplier(_f32(float(si)), _f32(float(wsc[0])), _f32(float(so)))
    lo, hi = _act_range(opt.get("act", "none"), zo, so, to["dtype"])
    y = [0] * (Ho * Wo * Co)
    for oc in range(Co):
        m, s = mult[oc]
        ic = oc // M if depthwise else None
        for oy in range(Ho):
            for ox in range(Wo):
                acc = 0
                for ky in range(kh):
                    iy = oy * sh - pt + ky
                    if not 0 <= iy < H:
                        continue
                    for kx in range(kw):
                        ix = ox * sw - pl + kx
                        if not 0 <= ix < W:
                            continue
                        base = (iy * W + ix) * C
                        if depthwise:
                            acc += (x[base + ic] - zi) * (w[(ky * kw + kx) * Co + oc] - zw)
                        else:
                            wb = ((oc * kh + ky) * kw + kx) * C
                            for c in range(C):
                                acc += (x[base + c] - zi) * (w[wb + c] - zw)
                acc += b[oc]
                v = (npunet.rnd(Fraction(acc * sl, 1 << sf), "half_away") if npuc else mbqm(acc, m, s)) + zo
                y[(oy * Wo + ox) * Co + oc] = min(hi, max(lo, v))
    return y


def _fc(tf, o, x):
    T = tf["tensors"]
    ti, tw, to = T[o["inputs"][0]], T[o["inputs"][1]], T[o["outputs"][0]]
    tb = T[o["inputs"][2]] if len(o["inputs"]) > 2 and o["inputs"][2] >= 0 else None
    n, K = tw["shape"]
    w = _vals(tw)
    b = _vals(tb) if tb is not None and tb["data"] else [0] * n
    zi, zo, zw = ti["zero_point"][0], to["zero_point"][0], (tw["zero_point"] or [0])[0]
    wsc = tw["scale"] if len(tw["scale"]) > 1 else tw["scale"] * n
    lo, hi = _act_range(o["opt"].get("act", "none"), zo, to["scale"][0], to["dtype"])
    y = []
    for k in range(n):
        m, s = quantize_multiplier(float(ti["scale"][0]) * float(wsc[k]) / float(to["scale"][0]))
        acc = sum((x[i] - zi) * (w[k * K + i] - zw) for i in range(K)) + b[k]
        y.append(min(hi, max(lo, mbqm(acc, m, s) + zo)))
    return y


def _pool(tf, o, x, avg):
    T = tf["tensors"]
    ti, to = T[o["inputs"][0]], T[o["outputs"][0]]
    _, H, W, C = ti["shape"]
    _, Ho, Wo, _ = to["shape"]
    opt = o["opt"]
    kh, kw, sh, sw = opt["filter_h"], opt["filter_w"], opt["stride_h"], opt["stride_w"]
    _, pt = _pads(H, kh, sh, opt["padding"])
    _, pl = _pads(W, kw, sw, opt["padding"])
    if (ti["scale"], ti["zero_point"]) != (to["scale"], to["zero_point"]):
        raise ValueError("pooling with a rescale")
    lo, hi = _act_range(opt.get("act", "none"), to["zero_point"][0], to["scale"][0], to["dtype"])
    y = [0] * (Ho * Wo * C)
    for oy in range(Ho):
        for ox in range(Wo):
            for c in range(C):
                vs = [x[(iy * W + ix) * C + c] for iy in range(max(0, oy * sh - pt), min(H, oy * sh - pt + kh))
                      for ix in range(max(0, ox * sw - pl), min(W, ox * sw - pl + kw))]
                if avg:
                    acc, n = sum(vs), len(vs)
                    v = int((acc + n // 2) / n) if acc > 0 else -int((-acc + n // 2) / n)   # C truncation
                    if to["dtype"] == "uint8":                     # reference_ops: (acc + n / 2) / n, acc >= 0
                        v = (acc + n // 2) // n
                else:
                    v = max(vs)
                y[(oy * Wo + ox) * C + c] = min(hi, max(lo, v))
    return y


def _add_quant(T, o):
    """an ADD's quantisation: (u8 zero points a, b, out; float32 scales a, b, out; dtype)"""
    ta, tb, to = (T[t] for t in (o["inputs"][0], o["inputs"][1], o["outputs"][0]))
    if ta["shape"] != tb["shape"]:
        raise ValueError("ADD with broadcasting")
    u = 128 if to["dtype"] == "int8" else 0
    return ([t["zero_point"][0] + u for t in (ta, tb, to)], [t["scale"][0] for t in (ta, tb, to)], to["dtype"])


def tfl_add(tf, o, a, b):
    """ADD as TFLite's reference kernels (add.cc PrepareGeneralSubOp, reference_ops / reference_integer_ops::Add,
    8-bit): each input shifted left 20, scaled by QuantizeMultiplierSmallerThanOneExp(s_x / 2 max(s_a, s_b)), summed,
    then by twice_max / (2^20 s_out), every step SaturatingRoundingDoublingHighMul + RoundingDivideByPOT"""
    T = tf["tensors"]
    to = T[o["outputs"][0]]
    sa, sb, so = (T[t]["scale"][0] for t in (o["inputs"][0], o["inputs"][1], o["outputs"][0]))
    za, zb, zo = (T[t]["zero_point"][0] for t in (o["inputs"][0], o["inputs"][1], o["outputs"][0]))
    if T[o["inputs"][0]]["shape"] != T[o["inputs"][1]]["shape"]:
        raise ValueError("ADD with broadcasting")
    tm = 2 * max(float(sa), float(sb))
    ma, ea = quantize_multiplier(float(sa) / tm)
    mb, eb = quantize_multiplier(float(sb) / tm)
    mo, eo = quantize_multiplier(tm / ((1 << 20) * float(so)))
    lo, hi = _act_range(o["opt"].get("act", "none"), zo, so, to["dtype"])
    y = []
    for p, q in zip(a, b):
        s_ = mbqm((p - za) << 20, ma, ea) + mbqm((q - zb) << 20, mb, eb)
        y.append(min(hi, max(lo, mbqm(s_, mo, eo) + zo)))
    return y


def conv_add(tf, o, a, b):
    """ADD as npunet --add-as-conv computes it (ncpgen/npunet.py _lower_add_conv, docs/format/add.md): the same
    add_fit weights and NPUC multiplier, acc = wa (a - z_a) + wb (b - z_b), out = z_out + round half away (acc slope /
    2^shift), clamped; values in the model's dtype"""
    T = tf["tensors"]
    (za, zb, zo), (sa, sb, so), dt = _add_quant(T, o)
    u = 128 if dt == "int8" else 0
    fa, fb, fo = (Fraction(_f32(float(v))) for v in (sa, sb, so))
    same = o["inputs"][0] == o["inputs"][1]
    rng = lambda z: max(z, 255 - z)
    ra, rb = (2 * fa / fo, None) if same else (fa / fo, fb / fo)
    wa, wb, slope, shift, _ = npunet.add_fit(ra, rb, rng(za), rng(zb), 255)
    act = o["opt"].get("act", "none")
    lo, hi = _act_range(act, zo - u, so, dt)
    if act == "relu6" and hi + u < 255:
        raise ValueError("ADD with RELU6 inside the range (npunet: --relu6-range)")
    den = 1 << shift
    y = []
    for p, q in zip(a, b):
        acc = wa * (p + u - za) + (0 if same else wb * (q + u - zb))
        v = npunet.rnd(Fraction(acc * slope, den), "half_away") + zo
        y.append(min(hi, max(lo, min(255, max(0, v)) - u)))
    return y


def _pad(tf, o, x):
    """PAD of an NHWC tensor with its zero point (TFLite's quantised PAD: the same quantisation in and out)"""
    T = tf["tensors"]
    ti, to = T[o["inputs"][0]], T[o["outputs"][0]]
    pd = struct.unpack("<8i", T[o["inputs"][1]]["data"][:32])
    if pd[0] or pd[1] or pd[6] or pd[7] or (ti["scale"], ti["zero_point"]) != (to["scale"], to["zero_point"]):
        raise ValueError("PAD other than of H and W, or with a rescale")
    _, H, W, C = ti["shape"]
    _, Ho, Wo, _ = to["shape"]
    z = to["zero_point"][0]
    y = [z] * (Ho * Wo * C)
    for r in range(H):
        d0 = ((r + pd[2]) * Wo + pd[4]) * C
        y[d0:d0 + W * C] = x[r * W * C:(r + 1) * W * C]
    return y


def from_tensor(tf, start):
    """the model from tensor `start` on: start as its input, the ops that read only it, constants and their outputs
    (a CPU head such as QUANTIZE or CAST left out)"""
    T = tf["tensors"]
    have, ops = {start}, []
    for o in tf["ops"]:
        if all(t < 0 or t in have or (T[t]["data"] and t not in have) for t in o["inputs"]) and \
                any(t in have for t in o["inputs"]):
            ops.append(o)
            have |= set(o["outputs"])
    return dict(tf, ops=ops, inputs=[start])


def real_add(tf, o, a, b):
    """ADD in real arithmetic with one rounding: z_out + round half away ((s_a (a - z_a) + s_b (b - z_b)) / s_out),
    the float32 scales exact: the ideal both TFLite's kernel and any NPU lowering approximate"""
    T = tf["tensors"]
    (za, zb, zo), (sa, sb, so), dt = _add_quant(T, o)
    u = 128 if dt == "int8" else 0
    fa, fb, fo = (Fraction(_f32(float(v))) for v in (sa, sb, so))
    lo, hi = _act_range(o["opt"].get("act", "none"), zo - u, so, dt)
    return [min(hi, max(lo, npunet.rnd((fa * (p + u - za) + fb * (q + u - zb)) / fo, "half_away") + zo - u))
            for p, q in zip(a, b)]


def pow2_add(tf, o, a, b):
    """the power-of-two alternative at its best (docs/format/add.md 6): each input requantised by a conv to zero point
    0 at the common scale 2^-n, the smallest power of two that holds both ranges in int8 (round half away), NPUC's
    Eltwise sum at 2^-m holding the sum's range (the harness's G arithmetic: saturate(round(...))), then a conv
    requantising to the ADD's output: three roundings, the multipliers exact (NPUC's 8-bit slope not counted)"""
    T = tf["tensors"]
    (za, zb, zo), (sa, sb, so), dt = _add_quant(T, o)
    u = 128 if dt == "int8" else 0
    fa, fb, fo = (Fraction(_f32(float(v))) for v in (sa, sb, so))
    ra_, rb_ = max(za, 255 - za), max(zb, 255 - zb)
    p2 = lambda x: Fraction(2) ** math.ceil(math.log2(float(x)))
    si = p2(max(fa * ra_, fb * rb_) / 127)
    se = p2((fa * ra_ + fb * rb_) / 127)
    lo, hi = _act_range(o["opt"].get("act", "none"), zo - u, so, dt)
    sat = lambda v: min(127, max(-128, v))
    y = []
    for p, q in zip(a, b):
        a1 = sat(npunet.rnd(fa * (p + u - za) / si, "half_away"))
        b1 = sat(npunet.rnd(fb * (q + u - zb) / si, "half_away"))
        e = sat(npunet.rnd((a1 + b1) * si / se, "half_away"))
        y.append(min(hi, max(lo, npunet.rnd(e * se / fo, "half_away") + zo - u)))
    return y


def tflref(tf, x, stop=None, add=None, seen=None, npuc_convs=False):
    """the model's int values for its input values (NHWC lists of ints); stop: a tensor index to return instead of
    the output; add: the ADD's arithmetic (tfl_add, TFLite's; conv_add, npunet's --add-as-conv); seen(o, a, b, y): a
    hook called with every ADD's inputs and output; npuc_convs: every conv with NPUC's 8-bit multiplier"""
    T = tf["tensors"]
    vals = {tf["inputs"][0]: list(x)}
    for o in tf["ops"]:
        op = o["op"]
        src = vals.get(o["inputs"][0])
        if src is None:
            raise ValueError("%s reads tensor %d before it is computed" % (op, o["inputs"][0]))
        if op == "ADD":
            y = (add or tfl_add)(tf, o, src, vals[o["inputs"][1]])
            if seen:
                seen(o, src, vals[o["inputs"][1]], y)
        elif op == "CONV_2D":
            y = _conv(tf, o, src, False, npuc_convs)
        elif op == "DEPTHWISE_CONV_2D":
            y = _conv(tf, o, src, True, npuc_convs)
        elif op in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            y = _pool(tf, o, src, op == "AVERAGE_POOL_2D")
        elif op == "RESHAPE":
            y = list(src)
        elif op == "PAD":
            y = _pad(tf, o, src)
        elif op == "FULLY_CONNECTED":
            y = _fc(tf, o, src)
        else:
            raise ValueError("op %s is not in tflref (stop before it)" % op)
        vals[o["outputs"][0]] = y
        if stop is not None and o["outputs"][0] == stop:
            return y
    return vals[tf["outputs"][0] if stop is None else stop]


def _tensor_bytes(vals, dtype):
    return struct.pack("<%d%s" % (len(vals), "b" if dtype == "int8" else "B"), *vals)


def cmd_tflref(a):
    tf = npunet.read_tflite(open(a.model, "rb").read())
    T = tf["tensors"]
    ti = T[tf["inputs"][0]]
    data = open(a.input, "rb").read()
    x = list(struct.unpack("<%d%s" % (len(data), "b" if ti["dtype"] == "int8" else "B"), data))
    stop = None
    if a.stop is not None:
        stop = int(a.stop) if a.stop.isdigit() else next(i for i, t in enumerate(T) if t["name"] == a.stop)
    y = tflref(tf, x, stop)
    t = T[stop if stop is not None else tf["outputs"][0]]
    open(a.out, "wb").write(_tensor_bytes(y, t["dtype"]))
    print("tflref: %s -> tensor %r %s %s, %d values" % (os.path.basename(a.model), t["name"], t["shape"], t["dtype"], len(y)))


# ---- person_detect: npunet's reference against TFLite's ---------------------------------------------------------------
def bmp_gray(path):
    """an 8-bit palette BMP (TFLM's test images) -> its gray values, top row first"""
    d = open(path, "rb").read()
    off, = struct.unpack_from("<I", d, 10)
    w, h = struct.unpack_from("<ii", d, 18)
    bpp, = struct.unpack_from("<H", d, 28)
    if bpp != 8:
        raise ValueError("%s: %d bits per pixel" % (path, bpp))
    pal = d[54:off]
    stride = (w + 3) & ~3
    rows = [d[off + r * stride:off + r * stride + w] for r in range(abs(h))]
    if h > 0:
        rows.reverse()                       # bottom-up rows
    return [pal[4 * p] for row in rows for p in row]  # palette B, G, R, 0: gray -> any channel


def cmd_person(a):
    p = model_path("person_detect")
    tfd = open(p, "rb").read()
    tf = npunet.read_tflite(tfd)
    T = tf["tensors"]
    ti = T[tf["inputs"][0]]
    logits = next(o["inputs"][0] for o in tf["ops"] if o["op"] == "RESHAPE")
    inputs = []
    for n in ("person_detect.person.bmp", "person_detect.no_person.bmp"):
        g = bmp_gray(model_path(n))
        # as TFLM's person_detection test feeds them: the pixel bytes as int8 (generate_cc_arrays.py), rows top first
        inputs.append((n.split(".")[1], [v - 256 if v > 127 else v for v in g]))
    rng = random.Random(0x50455253)
    for k in range(a.inputs):
        inputs.append(("random%d" % k, [rng.randrange(-128, 128) for _ in range(96 * 96)]))
    lows = {"fold": npunet.load_model(p, 1, True, True), "shifts": npunet.load_model(p, 1, "shifts", True)}
    print("person_detect %s (sha256 %s), input %s %s scale %g zp %d; NPU part output = tensor %d %r (the logits)"
          % (p, sha256(p)[:12], ti["shape"], ti["dtype"], ti["scale"][0], ti["zero_point"][0], logits, T[logits]["name"]))
    tot = {k: [0, 0, 0] for k in lows}
    for name, x in inputs:
        want = tflref(tf, x, logits)
        line = "%-10s TFLite logits %s" % (name, want)
        for k, low in lows.items():
            y = npunet.ref_npu(low, npunet.model_in_to_chw(low, _tensor_bytes(x, "int8")))
            got = [v - 128 for v in y]
            d = [abs(g - w) for g, w in zip(got, want)]
            same = (got.index(max(got)) == want.index(max(want)))
            tot[k][0] += max(d)
            tot[k][1] += sum(d)
            tot[k][2] += same
            line += " | %s %s (diff %s, top-1 %s)" % (k, got, d, "same" if same else "DIFFERS")
        print(line)
    n = len(inputs)
    for k, (mx, sm, same) in tot.items():
        print("%s: mean |diff| %.2f LSB of the logits, top-1 = TFLite's on %d of %d inputs" % (k, sm / (2 * n), same, n))


# ---- ADD with any scales: npunet's --add-as-conv against TFLite's ADD ------------------------------------------------
def model_input(tf, kind, seed=0):
    """an input in the model's dtype: "person" (TFLM's person.bmp, nearest-resized, gray in every channel) or seeded
    uniform noise"""
    ti = tf["tensors"][tf["inputs"][0]]
    _, H, W, C = ti["shape"]
    signed = ti["dtype"] == "int8"
    if kind == "person":
        g = bmp_gray(model_path("person_detect.person.bmp"))
        px = [g[(y * 96 // H) * 96 + x * 96 // W] for y in range(H) for x in range(W)]
        return [v - 128 if signed else v for v in px for _ in range(C)]
    r = random.Random(seed)
    return [r.randrange(-128, 128) if signed else r.randrange(256) for _ in range(H * W * C)]


def cmd_addcheck(a):
    """per ADD and at the cut, each ADD arithmetic against TFLite's (tfl_add), the rest of the model TFLite's: npunet's
    --add-as-conv (conv_add), the exact real sum (real_add) and the power-of-two alternative (pow2_add)"""
    tf = npunet.read_tflite(open(a.model, "rb").read())
    T = tf["tensors"]
    if a.input is not None:
        tf = from_tensor(tf, int(a.input) if a.input.isdigit() else next(i for i, t in enumerate(T) if t["name"] == a.input))
    stop = None
    if a.output is not None:
        stop = int(a.output) if a.output.isdigit() else next(i for i, t in enumerate(T) if t["name"] == a.output)
        tf = npunet.tflite_cut(tf, stop)
        stop = tf["outputs"][0]
    adds = [o for o in tf["ops"] if o["op"] == "ADD"]
    impls = [("conv", conv_add), ("real", real_add), ("pow2", pow2_add)][:3 if a.all else 1]
    per = {(k, id(o)): [0, 0, 0] for k, _ in impls for o in adds}        # values, differing, max |d|

    def seen(o, x, y_, want):
        for k, f in impls:
            d = [abs(g - w) for g, w in zip(f(tf, o, x, y_), want)]
            st = per[k, id(o)]
            st[0] += len(d)
            st[1] += sum(1 for v in d if v)
            st[2] = max(st[2], max(d))
    inputs = [("person", model_input(tf, "person"))] + [("random%d" % k, model_input(tf, "random", k))
                                                         for k in range(a.inputs)]
    tot = {k: [0, 0, 0, 0] for k, _ in impls}
    print("addcheck %s: %d ADDs up to tensor %d %r %s" % (os.path.basename(a.model), len(adds), tf["outputs"][0],
                                                       T[tf["outputs"][0]]["name"][:60], T[tf["outputs"][0]]["shape"]))
    per_tensor = all(len(set(T[o["inputs"][1]]["scale"])) == 1 for o in tf["ops"]
                     if o["op"] in ("CONV_2D", "DEPTHWISE_CONV_2D"))
    runs = impls + ([("npuc", None)] if a.all and per_tensor else [])
    tot.update({k: [0, 0, 0, 0] for k, _ in runs})
    for name, x in inputs:
        want = tflref(tf, x, seen=seen)
        for k, f in runs:
            y = tflref(tf, x, npuc_convs=True) if k == "npuc" else tflref(tf, x, add=f)
            d = [abs(g - w) for g, w in zip(y, want)]
            nd = sum(1 for v in d if v)
            t_ = tot[k]
            t_[0] += len(d)
            t_[1] += nd
            t_[2] = max(t_[2], max(d))
            t_[3] += sum(d)
            print("  %-8s %-4s at the cut: %d of %d values differ (%.3f %%), max |d| %d LSB, mean |d| %.4f LSB" % (
                name, k, nd, len(d), 100.0 * nd / len(d), max(d), sum(d) / len(d)))
    for j, o in enumerate(adds):
        (za, zb, zo), (sa, sb, so), _ = _add_quant(T, o)
        print("  ADD %2d %-36s s_a/s_o %.4f s_b/s_o %.4f z %d %d %d:" % (j, T[o["outputs"][0]]["name"][-36:], sa / so,
                                                                       sb / so, za, zb, zo) +
              "".join(" %s %.3f %% (max %d)" % (k, 100.0 * per[k, id(o)][1] / per[k, id(o)][0], per[k, id(o)][2])
                      for k, _ in impls))
    if "npuc" in tot:
        t_ = tot["npuc"]
        print("addcheck npuc: TFLite's ADDs, every conv with NPUC's 8-bit multiplier (for scale): at the cut: %d of %d "
              "(%.3f %%), max |d| %d, mean |d| %.4f LSB" % (t_[1], t_[0], 100.0 * t_[1] / t_[0], t_[2], t_[3] / t_[0]))
    for k, _ in impls:
        n = sum(per[k, id(o)][0] for o in adds)
        nd = sum(per[k, id(o)][1] for o in adds)
        t_ = tot[k]
        print("addcheck %s: every ADD alone: %d of %d values differ (%.3f %%), max |d| %d; at the cut: %d of %d "
              "(%.3f %%), max |d| %d, mean |d| %.4f LSB" % (
                  k, nd, n, 100.0 * nd / max(n, 1), max([per[k, id(o)][2] for o in adds] or [0]), t_[1], t_[0],
                  100.0 * t_[1] / t_[0], t_[2], t_[3] / t_[0]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="camera.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("fetch")
    p.add_argument("models", nargs="*")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("tflref")
    p.add_argument("model")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--stop")
    p.set_defaults(fn=cmd_tflref)
    p = sub.add_parser("addcheck")
    p.add_argument("model")
    p.add_argument("--output", help="cut the model at this tensor (index or name)")
    p.add_argument("--input", help="start the model at this tensor (index or name): a CPU head left out")
    p.add_argument("--inputs", type=int, default=2, help="seeded random inputs besides the person image")
    p.add_argument("--all", action="store_true", help="also the exact real sum and the power-of-two alternative")
    p.set_defaults(fn=cmd_addcheck)
    p = sub.add_parser("person")
    p.add_argument("--inputs", type=int, default=4)
    p.set_defaults(fn=cmd_person)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
