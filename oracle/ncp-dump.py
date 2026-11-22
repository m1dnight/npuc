#!/usr/bin/env python3
"""Every byte of an NCP as a diffable listing (dossier/npu-compiler.md section 10): the header's named words, the
address / memory / group vectors, each io descriptor's 31 words, each group's intrinsic (hex) and CMDQ stream (one
instruction per line), the weight blocks and any other byte (hex), the zero tail; then a coverage line (every byte
belongs to exactly one region, or the gaps are named). Reads only; prints facts.

  python3 -I ncp-dump.py <ncp> [--log <NPUC compile log>] [--no-hex] [--section header|av|mv|gv|io|isa|intr|weights|other]...

--log: the instruction text NPUC printed for each machine code it emitted ("code line: set_reg(r9, 0x1)" ... "machine
code: 804900000001") is shown next to the instruction, so register names come from the compile itself. --no-hex
replaces hex dumps by their sha256 (for structure-only diffs).

Structures: GPL ncp_header_v21.h (v22 uses reserved[0..1] as ncp_end and the zero tail's length); instructions are 6
bytes, u32 value then u16 opcode (ncp-isa.py).
"""
import hashlib
import re
import struct
import sys

HDR = ["magic_number1", "hdr_version", "hdr_size", "intrinsic_version", "net_id", "unique_id", "priority", "flags",
       "period", "workload", "address_vector_offset", "address_vector_cnt", "memory_vector_offset",
       "memory_vector_cnt", "group_vector_offset", "group_vector_cnt", "body_version", "body_offset", "body_size",
       "io_vector_offset", "io_vector_cnt", "ncp_end", "zero_tail"] + ["reserved[%d]" % i for i in range(2, 10)] + \
      ["magic_number2"]
TYPES = {0: "IN_FMAP", 1: "OT_FMAP", 2: "IM_FMAP", 3: "OT_PIX0", 4: "OT_PIX1", 5: "OT_PIX2", 6: "OT_PIX3",
         7: "WEIGHT", 8: "WMASK", 9: "LUT", 10: "NCP", 11: "GOLDEN", 12: "CUCODE"}
IO = ["uid", "bid", "sgid", "wait_flag", "trig_flag", "mode", "pad", "scale", "bias"]
BLOB = ["vector", "offset", "format", "bpp", "n", "c", "h", "w", "w_stride", "c_stride", "n_stride"]
BLOCKS = {0x8: ("CMDQ", 0x401E0000), 0x9: ("MTC", 0x40120000), 0xA: ("NFUBR", 0x40160000), 0xB: ("INTCOM", 0x40100000)}


def log_names(path):
    """machine code (int, op << 32 | value) -> NPUC's instruction text, from its compile log"""
    names = {}
    last = None
    rx_line = re.compile(r"code line: (.*?),?\s*$")
    rx_mc = re.compile(r"machine code: ([0-9a-fA-F]+)\s*$")
    for raw in open(path, errors="replace"):
        m = rx_line.search(raw)
        if m:
            last = m.group(1).strip()
            continue
        m = rx_mc.search(raw)
        if m and last is not None:
            names.setdefault(int(m.group(1), 16), last)
            last = None
    return names


class Dump:
    def __init__(self, d, names, hexdump, sections):
        self.d, self.names, self.hexdump, self.sections = d, names, hexdump, sections
        self.owner = [None] * len(d)
        self.lines = []

    def claim(self, start, end, what):
        for i in range(max(0, start), min(end, len(self.d))):
            if self.owner[i] is None:
                self.owner[i] = what

    def out(self, sec, s):
        if not self.sections or sec in self.sections:
            self.lines.append(s)

    def hexlines(self, sec, start, end, label):
        b = self.d[start:end]
        if not self.hexdump:
            self.out(sec, "  %s %#06x..%#06x %d B sha256 %s" % (label, start, end, len(b), hashlib.sha256(b).hexdigest()[:16]))
            return
        for o in range(0, len(b), 16):
            self.out(sec, "  %s +%#05x %s" % (label, o, b[o:o + 16].hex(" ")))


def dump(path, names, hexdump=True, sections=()):
    d = open(path, "rb").read()
    D = Dump(d, names, hexdump, sections)
    h = struct.unpack_from("<32I", d, 0)
    D.out("header", "# ncp %d B sha256 %s" % (len(d), hashlib.sha256(d).hexdigest()))
    D.claim(0, 128, "header")
    for i, n in enumerate(HDR):
        D.out("header", "header w%-2d %-22s %#010x %d" % (i, n, h[i], h[i]))
    avo, avc, mvo, mvc, gvo, gvc = h[10:16]
    body, bsize, iov, ioc, end, tail = h[17], h[18], h[19], h[20], h[21], h[22]
    av = [struct.unpack_from("<4I", d, avo + 16 * i) for i in range(avc)]
    D.claim(avo, avo + 16 * avc, "av")
    for i, a in enumerate(av):
        D.out("av", "av %2d index %d m_addr %#x s_addr %#x size %d" % ((i,) + a))
    mv = [struct.unpack_from("<8I", d, mvo + 32 * i) for i in range(mvc)]
    D.claim(mvo, mvo + 32 * mvc, "mv")
    for i, m in enumerate(mv):
        D.out("mv", "mv %2d %-7s pixel_format %#x w %d h %d c %d wstride %d cstride %d av %d"
              % ((i, TYPES.get(m[0], str(m[0]))) + m[1:]))
    gv = [struct.unpack_from("<10I", d, gvo + 40 * i) for i in range(gvc)]
    D.claim(gvo, gvo + 40 * gvc, "gv")
    for i, g in enumerate(gv):
        D.out("gv", "gv %2d index %d type %d size %d status %d flags %#x batch %d intrinsic_offset %#x intrinsic_size %d "
              "isa_offset %#x isa_size %d" % ((i,) + g))
    D.claim(iov, iov + 124 * ioc, "io")
    for i in range(ioc):
        w = struct.unpack_from("<31I", d, iov + 124 * i)
        D.out("io", "io %2d " % i + " ".join("%s %#x" % (n, v) for n, v in zip(IO, w[:9])))
        D.out("io", "io %2d   src " % i + " ".join("%s %d" % (n, v) for n, v in zip(BLOB, w[9:20])))
        D.out("io", "io %2d   dst " % i + " ".join("%s %d" % (n, v) for n, v in zip(BLOB, w[20:31])))
    hdr_end = max(avo + 16 * avc, mvo + 32 * mvc, gvo + 40 * gvc, iov + 124 * ioc)
    if any(d[hdr_end:h[2]]):
        D.hexlines("other", hdr_end, h[2], "hdrpad")
    D.claim(hdr_end, h[2], "hdrpad")
    for gi, g in enumerate(gv):
        io_, isz, so, ssz = body + g[6], g[7], body + g[8], g[9]
        D.claim(io_, io_ + isz, "intr%d" % gi)
        D.out("intr", "intr g%d %#x+%d" % (gi, g[6], isz))
        D.hexlines("intr", io_, io_ + isz, "intr g%d" % gi)
        D.claim(so, so + ssz, "isa%d" % gi)
        D.out("isa", "isa g%d %#x+%d (%d instructions)" % (gi, g[8], ssz, ssz // 6))
        for k in range(ssz // 6):
            val, op = struct.unpack_from("<IH", d, so + 6 * k)
            txt = names.get(op << 32 | val, "")
            blk = op >> 12
            if blk in BLOCKS and op >> 8 not in (0x3A, 0x2B):
                bn, base = BLOCKS[blk]
                D.out("isa", "isa g%d %4d %-6s %#010x = %#010x  %s" % (gi, k, bn, base + (op & 0xFFF) * 4, val, txt))
            else:
                D.out("isa", "isa g%d %4d op%04x %10s   %#010x  %s" % (gi, k, op, "", val, txt))
    for i, m in enumerate(mv):
        if TYPES.get(m[0]) == "WEIGHT" and m[7] < len(av):
            a = av[m[7]]
            D.claim(a[1], a[1] + a[3], "weights%d" % i)
            D.out("weights", "weights mv %d av %d %#x+%d" % (i, m[7], a[1], a[3]))
            D.hexlines("weights", a[1], a[1] + a[3], "w%d" % i)
    # the rest up to ncp_end, then the zero tail
    i = 0
    while i < min(end, len(d)):
        if D.owner[i] is None:
            j = i
            while j < min(end, len(d)) and D.owner[j] is None:
                j += 1
            nz = any(d[i:j])
            D.out("other", "gap %#06x..%#06x %d B %s" % (i, j, j - i, "nonzero" if nz else "zero"))
            if nz:
                D.hexlines("other", i, j, "gap")
            D.claim(i, j, "gap")
            i = j
        else:
            i += 1
    D.claim(end, len(d), "tail")
    D.out("other", "tail %#06x..%#06x %d B (header zero_tail %d) %s" % (end, len(d), len(d) - end, tail,
                                                                       "all zero" if not any(d[end:]) else "NONZERO"))
    gaps = sum(1 for i, o in enumerate(D.owner) if o == "gap" and d[i])
    D.out("other", "coverage: %d B, end %#x + tail %d %s, nonzero bytes outside named regions: %d"
          % (len(d), end, tail, "=" if end + tail == len(d) else "!= file length", gaps))
    return D.lines


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    names = {}
    if "--log" in a:
        i = a.index("--log")
        names = log_names(a[i + 1])
        del a[i:i + 2]
    hexdump = "--no-hex" not in a
    a = [x for x in a if x != "--no-hex"]
    secs = []
    while "--section" in a:
        i = a.index("--section")
        secs.append(a[i + 1])
        del a[i:i + 2]
    if len(a) != 1:
        sys.exit(__doc__)
    for line in dump(a[0], names, hexdump, secs):
        print(line)


if __name__ == "__main__":
    main()
