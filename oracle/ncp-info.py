#!/usr/bin/env python3
"""Print an NCP's header, vectors and io descriptors (PORT-32 N3b), optionally next to a second NCP's header.

Structures from the GPL ncp_header_v21.h (the v22 header uses two of v21's reserved words: w21 the NCP's end, w22 its
zero tail; journal/plans/npu-n2-plan.md section 2). io_desc: uid, bid, sgid, wait_flag, trig_flag, mode, pad, scale,
bias, then either a simple DMA option (src_vector, src_offset, dst_vector, dst_offset, size) or a blob option (two
ncp_blob: vector, offset, format, bpp, n, c, h, w, w_stride, c_stride, n_stride). Reads only; prints facts.

  python3 -I ncp-info.py <ncp> [<other ncp to compare the header with>]
"""
import hashlib
import struct
import sys

HDR = ["magic_number1", "hdr_version", "hdr_size", "intrinsic_version", "net_id", "unique_id", "priority", "flags",
       "period", "workload", "address_vector_offset", "address_vector_cnt", "memory_vector_offset",
       "memory_vector_cnt", "group_vector_offset", "group_vector_cnt", "body_version", "body_offset", "body_size",
       "io_vector_offset", "io_vector_cnt", "w21 ncp_end", "w22 zero_tail"] + ["reserved[%d]" % i for i in range(2, 10)] + \
      ["magic_number2"]
TYPES = {0: "IN_FMAP", 1: "OT_FMAP", 2: "IM_FMAP", 3: "OT_PIX0", 4: "OT_PIX1", 5: "OT_PIX2", 6: "OT_PIX3",
         7: "WEIGHT", 8: "WMASK", 9: "LUT", 10: "NCP", 11: "GOLDEN", 12: "CUCODE"}


def load(path):
    d = open(path, "rb").read()
    h = struct.unpack_from("<32I", d, 0)
    return d, h


def info(path):
    d, h = load(path)
    print("ncp %s: %d B sha256 %s" % (path, len(d), hashlib.sha256(d).hexdigest()))
    print("  file length %s w21 + w22 = %#x" % ("=" if len(d) == h[21] + h[22] else "!=", h[21] + h[22]))
    avo, avc, mvo, mvc, gvo, gvc = h[10:16]
    iov, ioc = h[19], h[20]
    av = [struct.unpack_from("<4I", d, avo + 16 * i) for i in range(avc)]
    print("address vectors (%d): index, m_addr, s_addr, size" % avc)
    for a in av:
        print("  av %3d m_addr %#9x s_addr %#x size %d" % a)
    print("memory vectors (%d): type, pixel_format, w x h x c, wstride, cstride, av" % mvc)
    roles = {}
    for i in range(mvc):
        t, pf, w, hh, c, ws, cs, a = struct.unpack_from("<8I", d, mvo + 32 * i)
        roles.setdefault(a, []).append(TYPES.get(t, str(t)))
        print("  mv %3d %-8s fmt %#x %dx%dx%d wstride %d cstride %d av %d (%d B at %#x)"
              % (i, TYPES.get(t, t), pf, w, hh, c, ws, cs, a, av[a][3] if a < avc else -1, av[a][1] if a < avc else -1))
    print("group vectors (%d): index, type, size, status, flags, batch, intrinsic off/size, isa off/size" % gvc)
    for i in range(gvc):
        g = struct.unpack_from("<10I", d, gvo + 40 * i)
        print("  gv %3d type %d size %d status %d flags %#x batch %d intrinsic %#x+%d isa %#x+%d" % g)
    print("io descriptors (%d)" % ioc)
    for i in range(ioc):
        w = struct.unpack_from("<31I", d, iov + 124 * i)
        head = "  io %3d uid %d bid %d sgid %d wait %#x trig %#x mode %d pad %d scale %d bias %d" % ((i,) + w[:9])
        s, t = w[9:20], w[20:31]
        def blob(b):
            role = "/".join(roles.get(b[0], ["?"]))
            return "av %d(%s) +%d fmt %d bpp %d n%d c%d h%d w%d ws %d cs %d ns %d" % ((b[0], role) + b[1:])
        print(head)
        print("        simple: src av %d +%d -> dst av %d +%d, %d B" % w[9:14])
        print("        blob:   src %s" % blob(s))
        print("                dst %s" % blob(t))
    return d, h


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    d, h = info(sys.argv[1])
    if len(sys.argv) == 3:
        d2, h2 = load(sys.argv[2])
        print("header, %s against %s" % (sys.argv[1], sys.argv[2]))
        for i, n in enumerate(HDR):
            print("  w%-2d %-22s %#10x %#10x%s" % (i, n, h[i], h2[i], "" if h[i] == h2[i] else "   differs"))


if __name__ == "__main__":
    main()
