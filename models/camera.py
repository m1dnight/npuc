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

tflref covers what the camera models' NPU parts use: CONV_2D, DEPTHWISE_CONV_2D (any multiplier), AVERAGE_POOL_2D,
MAX_POOL_2D, RESHAPE, FULLY_CONNECTED; int8 (per-channel or per-tensor weights) and uint8 (per-tensor). Anything else
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


def _conv(tf, o, x, depthwise):
    """CONV_2D / DEPTHWISE_CONV_2D, reference_integer_ops (per-channel) or reference_ops (uint8 per-tensor): NHWC"""
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
                v = mbqm(acc, m, s) + zo
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


def tflref(tf, x, stop=None):
    """the model's int values for its input values (NHWC lists of ints); stop: a tensor index to return instead of
    the output"""
    T = tf["tensors"]
    vals = {tf["inputs"][0]: list(x)}
    for o in tf["ops"]:
        op = o["op"]
        src = vals.get(o["inputs"][0])
        if src is None:
            raise ValueError("%s reads tensor %d before it is computed" % (op, o["inputs"][0]))
        if op == "CONV_2D":
            y = _conv(tf, o, src, False)
        elif op == "DEPTHWISE_CONV_2D":
            y = _conv(tf, o, src, True)
        elif op in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            y = _pool(tf, o, src, op == "AVERAGE_POOL_2D")
        elif op == "RESHAPE":
            y = list(src)
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
    p = sub.add_parser("person")
    p.add_argument("--inputs", type=int, default=4)
    p.set_defaults(fn=cmd_person)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
