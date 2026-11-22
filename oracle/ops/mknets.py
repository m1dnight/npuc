#!/usr/bin/env python3
"""PORT-32: the "npuc-net 2" test networks for the layer kinds npuc-net added (docs/harness-ops.md).

  python3 -I oracle/ops/mknets.py <outdir>

Writes <outdir>/<name>.npuc (our own seeded random weights, the same bytes on every run) and <outdir>/expect.tsv: one line
per network, "<name>\t<PASS|REFUSE>\t<what to find>". For PASS, every "|"-separated marker must appear in NPUC's
compile log (log.txt) and the NCP must exist; for REFUSE, npuc-net must exit 1 before calling NPUC, with that text in
its "npuc-net: FAIL" line. check-ops.sh runs them through the local oracle (npuc-local.sh).
"""
import os
import random
import struct
import sys


def fh(x):
    """a scale as npuc-net wants it: the float32 bit pattern"""
    return '0x%08x' % struct.unpack('<I', struct.pack('<f', x))[0]


R = random.Random(7)
S7 = fh(1 / 128)  # fractional length 7
S6 = fh(1 / 64)


def conv(name, inp, cin, cout, k=3, zp=0, s_out=1 / 128, s_in=1 / 128):
    w = bytes(R.randrange(256) for _ in range(cout * cin * k * k)).hex()
    b = ' '.join(str(R.randrange(-200, 200)) for _ in range(cout))
    return (f"conv {name} {inp} {cout} {k} {k} 1 1 SAME 0 {fh(s_in)} {zp} {fh(1 / 256)} 128 {fh(s_out)} {zp} "
            f"{w} {cout} {b}")


def hdr(q=3, c=8, v=2):
    return [f"npuc-net {v}", "# PORT-32 ops test (oracle/ops/mknets.py)", "soc 2", f"quant {q}",
            f"input data0 {c} 16 16"]


ELT = 'ncp_eltwise_op|multiple-input data set'
NETS = [
    # residual add: conv output + the network input, then a conv (the ResNet block shape)
    ('add-res-q3', hdr() + [conv('c1', 'data0', 8, 8), f"eltwise add1 c1 data0 Add {S7} 0 {S7} 0 {S7} 0",
                            conv('c2', 'add1', 8, 8)], 'PASS', ELT),
    # quant 1 compiles too, but NPUC's Eltwise has no zero-point term (G: wrong by the zero point on the NPU)
    ('add-res-q1', hdr(1) + [conv('c1', 'data0', 8, 8, zp=128), f"eltwise add1 c1 data0 Add {S7} 128 {S7} 128 {S7} 128",
                             conv('c2', 'add1', 8, 8, zp=128)], 'PASS', ELT),
    ('add-2conv-q3', hdr() + [conv('c1', 'data0', 8, 8), conv('c2', 'data0', 8, 8),
                              f"eltwise add1 c1 c2 Add {S7} 0 {S7} 0 {S7} 0"], 'PASS', ELT),
    # out scale 2^-6: half the input resolution, no overflow (the meta data carries 7, 7, 6)
    ('add16-q3', hdr(c=16) + [conv('c1', 'data0', 16, 16), f"eltwise a1 c1 data0 Add {S7} 0 {S7} 0 {S6} 0",
                              conv('c2', 'a1', 16, 16, s_in=1 / 64)], 'PASS', ELT + '|out 6'),
    ('concat-q3', hdr() + [conv('c1', 'data0', 8, 8), conv('c2', 'data0', 8, 8), f"concat cat1 1 2 c1 c2 {S7} 0",
                           conv('c3', 'cat1', 16, 8)], 'PASS', 'addConcatGraphNode Node Name cat1|output DRAM is concatenated'),
    ('concat3-q3', hdr() + [conv('c1', 'data0', 8, 8), conv('c2', 'data0', 8, 8), conv('c4', 'data0', 8, 8),
                            f"concat cat1 1 3 c1 c2 c4 {S7} 0", conv('c3', 'cat1', 24, 8)], 'PASS',
     'addConcatGraphNode Node Name cat1'),
    ('concat16-q3', hdr(c=16) + [conv('c1', 'data0', 16, 16), conv('c2', 'data0', 16, 16),
                                 f"concat cat1 1 2 c1 c2 {S7} 0", conv('c3', 'cat1', 32, 16)], 'PASS',
     'addConcatGraphNode Node Name cat1'),
    ('relu-q3', hdr() + [conv('c1', 'data0', 8, 8), "relu r1 c1 ReLU", conv('c2', 'r1', 8, 8)], 'PASS',
     'Type is      ReLU|setOpDataNode Node Name: r1'),
    ('relu6-q3', hdr() + [conv('c1', 'data0', 8, 8), "relu r1 c1 ReLU6", conv('c2', 'r1', 8, 8)], 'PASS',
     'Type is      ReLU6|setOpDataNode Node Name: r1'),
    ('add-relu-q3', hdr() + [conv('c1', 'data0', 8, 8), f"eltwise add1 c1 data0 Add {S7} 0 {S7} 0 {S7} 0",
                             "relu r1 add1 ReLU", conv('c2', 'r1', 8, 8)], 'PASS', ELT + '|setOpDataNode Node Name: r1'),
    # what NPUC 1.36.1.a cannot compile, refused by npuc-net before any NPUC call
    ('mul-q3', hdr() + [conv('c1', 'data0', 8, 8), f"eltwise m1 c1 data0 Multiply {S7} 0 {S7} 0 {S7} 0"], 'REFUSE',
     'NPUC compiles Add only'),
    ('add-same-q3', hdr() + [conv('c1', 'data0', 8, 8), f"eltwise a1 c1 c1 Add {S7} 0 {S7} 0 {S7} 0"], 'REFUSE',
     'both inputs are c1'),
    ('add-pow2-q3', hdr() + [conv('c1', 'data0', 8, 8), f"eltwise a1 c1 data0 Add {S7} 0 {S7} 0 {fh(0.01)} 0"],
     'REFUSE', 'is not 2^-n'),
    ('add-zp-q3', hdr() + [conv('c1', 'data0', 8, 8), f"eltwise a1 c1 data0 Add {S7} 0 {S7} 0 {S7} 3"], 'REFUSE',
     'zero points 0, 0, 3 differ'),
    ('concat-in-q3', hdr() + [conv('c1', 'data0', 8, 8), f"concat cat1 1 2 data0 c1 {S7} 0"], 'REFUSE',
     'is the network input'),
    ('concat-scale-q3', hdr() + [conv('c1', 'data0', 8, 8), conv('c2', 'data0', 8, 8, s_out=1 / 64),
                                 f"concat cat1 1 2 c1 c2 {S7} 0"], 'REFUSE', 'NPUC does not rescale between layers'),
    ('reader-scale-q3', hdr() + [conv('c1', 'data0', 8, 8), conv('c2', 'c1', 8, 8, s_in=1 / 64)], 'REFUSE',
     'reads c1 as scale 0.015625'),
    ('v1-eltwise', hdr(v=1) + [conv('c1', 'data0', 8, 8), f"eltwise a1 c1 data0 Add {S7} 0 {S7} 0 {S7} 0"], 'REFUSE',
     'needs an "npuc-net 2" file'),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'expect.tsv'), 'w') as ex:
        for name, lines, outcome, marker in NETS:
            with open(os.path.join(out, name + '.npuc'), 'w') as f:
                f.write('\n'.join(lines + ['end']) + '\n')
            ex.write(f"{name}\t{outcome}\t{marker}\n")
    print(f"{len(NETS)} networks in {out}")


if __name__ == '__main__':
    main()
