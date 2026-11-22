#!/usr/bin/env python3
"""Decode the CMDQ instruction streams of an NCP's groups into register writes (PORT-32 N3c).

Encoding (V: NPUC's own compile log prints every instruction it emits as "code line: set_reg(<reg>, <value>)" with its
"machine code" and "register addr"; the NCP packs them as 6 bytes each, little endian: the 32-bit value, then the
16-bit opcode): opcode bits 15..12 = the block, 11..0 = the register's word index in it, for
set_reg; other opcodes (waits, nops) are printed raw.
    block 0x8 CMDQ   0x401e0000      block 0x9 MTC    0x40120000
    block 0xa NFUBR  0x40160000      block 0xb INTCOM 0x40100000
A "3a"/"2b" high byte is a wait (waiti3/wait3), 0 a nop.

  ncp-isa.py <ncp> [--block MTC|NFUBR|...] [--group N]      one line per instruction: group, index, block, reg, value
"""
import struct
import sys

BLOCKS = {0x8: ("CMDQ", 0x401E0000), 0x9: ("MTC", 0x40120000), 0xA: ("NFUBR", 0x40160000), 0xB: ("INTCOM", 0x40100000)}


def decode(path):
    d = open(path, "rb").read()
    h = struct.unpack_from("<32I", d, 0)
    gvo, gvc, body = h[14], h[15], h[17]
    out = []
    for g in range(gvc):
        gv = struct.unpack_from("<10I", d, gvo + 40 * g)
        off, size = body + gv[8], gv[9]
        for i in range(size // 6):
            val, op = struct.unpack_from("<IH", d, off + 6 * i)
            blk = op >> 12
            if blk in BLOCKS and op >> 8 not in (0x3A, 0x2B):
                name, base = BLOCKS[blk]
                out.append((g, i, name, base + (op & 0xFFF) * 4, val, op))
            else:
                out.append((g, i, "op%04x" % op, None, val, op))
    return out


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    blk = a[a.index("--block") + 1] if "--block" in a else None
    grp = int(a[a.index("--group") + 1]) if "--group" in a else None
    for g, i, name, reg, val, op in decode(a[0]):
        if (blk and name != blk) or (grp is not None and g != grp):
            continue
        if reg is None:
            if op or val:
                print("g%-3d %4d %-7s %10s %#010x" % (g, i, name, "", val))
        else:
            print("g%-3d %4d %-7s %#010x %#010x" % (g, i, name, reg, val))


if __name__ == "__main__":
    main()
