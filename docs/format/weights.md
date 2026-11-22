# NCP format: the WEIGHT blocks (PORT-32, open NCP generator, part `weights`)

How NPUC 1.36.1.a (the host oracle, `oracle/npuc-local.sh`) packs a conv layer's weights into the NCP's
WEIGHT address-vector blocks, measured by differential compiles and read from `libnpuc_template.so` (NPU2020 LIR kernel
pack). Code: `ncpgen/weights.py` (`weight_blocks(low)`, `weight_block_sizes(low)`, `conv_block(L, quant_mode)`,
`weight_geometry(low)`, `check <variant dir>...`). Tags: V = seen in bytes, logs or disassembly; G = guess.

## Status (2026-11-22)

- `check` over this part's sweep (`$NCPGEN_WORK/sweep/weights/v/`, `v2/`, `res/`, plus `sweep/c3` and
  `run/net-current`): **543 of 543 byte-exact**, nothing refused for being unmodelled (the stride-2 7x7/8x8 quant 1
  corner is decoded: layout section 6). 39 further variants NPUC itself aborts on are refused by the generator for
  the same reason (strides 3/4, non-square strides, too many sub-batches in either pass, a chunk over 511 non-zero
  values, the planner's abort of section 7): 39 of 39 as NPUC.
- Every other compile under `output/npu` with a lowered.json (stream, frame incl. `frame/fs`, tile, corpus, random,
  random2, realish, gap, gap2, poollim, widen, grelu, graph, graph-random, graph-random2, ops-check; not made for this
  part): **2159 of 2159 exact**, 2 refused as NPUC refuses; whole files (`ncpgen.py check`) of the 38 corner compiles
  38/38 exact. Final unseen sets: `final-chains` 69/69, `final-graph` 57/57 (after the two rules above
  from g513 and g555); ops-check (`output/npu/n3/ops-check/root-b53daf10`, from net.npuc) 13/13,
  `output/npu/n3/npunet-graph` 10/10, npunet's example-residual (NPU_ROOT=root-b53daf10, --quant 3 --allow-none,
  JSON and TFLite) 2/2.
- Family covered (all exact): convs of 1 to 3 layers with pools, Cin 1..4096, Cout 1..1024, kernels 1x1..13x13 SAME
  and up to 11x7 VALID, rectangular kernels, stride 1 and 2, quant 1 and 3, weight zero point 0/100/128/255, weights
  zero, ones, random in [-24, 24], full range, sparse (50..99.9 %), mixed per output channel, coordinate-coded; inputs
  1x1..128x128; blocks from 128 B to 2.6 MB; 160 random networks (`b9.txt`) all exact.
- The concat-reader sweep `$NCPGEN_WORK/sweep/weights/cat/`: 118/118 blocks exact (section 7).
- Planning sizes (`weight_planning_sizes`, = the log's compWeight): **5160 of 5160** conv/ReLU blocks of every
  compile incl. the final sets (section 7).
- Everything at once (every lowered.json under output/npu, 3106 dirs, incl. this part's sweeps and the final sets):
  blocks 2944/2944 exact, 41 refused as NPUC refuses, 121 skipped (NPUC failed outside the weights, or no NCP).
- Decompiles (Ghidra 12.2, project $NCPGEN_WORK/n3/ghidra/proj): `oracle/ghidra/targets-weights.txt` and
  `targets-weights2.txt` (output in $NCPGEN_WORK/n3/ghidra/decomp/, git-ignored). Ghidra addresses = objdump + 0x100000;
  the objdump addresses are used below unless marked (Ghidra).
- Sweep specs: `$NCPGEN_WORK/sweep/weights/b1.txt` .. `b17.txt` (one line per variant: name | weight post-op | quant |
  npuc-variant arguments); `tools/sweep.sh <spec>` compiles them (6 in parallel), `tools/mk.py` writes the JSON (our
  own weights: full-range, sparse, coordinate codes). The `.runs` directories were pruned (logs kept).

## The layout (V)

A chain (convs and pools) has one WEIGHT block per conv layer, in layer order (= the NCP's WEIGHT memory vectors'
order; their mv has pixel_format 1 and zero w/h/c, av size = block size). Graphs: section 6. Nothing in a conv's
block depends on the spatial size, the padding, the activation, the bias or the other layers, except a concat in
SRAM before it (section 6). NPUC's log prints its size as `compWeight=` (the planner's pack, section 7; equal to the
block except in section 7's cases).

1. **Stored bytes.** Quant 1: the lowered OIHW uint8 as is. Quant 3: value - 128 (KernelONFiller), i.e. XOR 0x80.
   The compression's "zero" is the stored weight zero point (q1: w_zp, V for 0, 100, 128, 255; q3: 0).
2. **Input channels over cores.** cpc = 2 * ceil(Cin / 8) channels per core, ncores = ceil(Cin / cpc) (1..4); core c
   holds channels c*cpc .. c*cpc + cpc - 1, those >= Cin absent. (Cin 1 -> 1 core of 2, Cin 9 -> 3 cores of 4.)
3. **Output channels over groups and loads.** 4 groups of gs = 2 * ceil(Cout / 8) channels (group g = channels
   g*gs .. g*gs + gs - 1); nloads = ceil(gs / 4); load l holds lanes q = 0..3 of every group: o = g*gs + 4l + q when
   4l + q < gs and o < Cout, else an empty slot. Cout 8: gs 2, so each group has 2 empty lanes (NPUC's "Convolution Ofm
   will be rounded to 16 from 8 due to HW bug"). Cout 20: gs 6, loads {0-3, 6-9, 12-15, 18-19} and {4-5, 10-11, 16-17}.
4. **Stride 2** (the only stride > 1 NPUC accepts; square only: `"_strideWidth == _strideHeight"`, `"Stride > 2 is not
   supported here!"`): a space-to-depth. Virtual kernel ceil(KH/2) x ceil(KW/2); each core channel pair becomes 8
   virtual channels j = (ci // 2) * 8 + phase * 2 + ci % 2, phase = (y % 2) * 2 + x % 2, tap (y // 2, x // 2); taps
   the 3x3 (or 5x5, 7x7) kernel does not cover hold the zero point. NPUC logs it as e.g. `kernel 2 x 2
   numInputChannels 32` for 3x3 s2 on 8 channels. Stride 1 = one phase, j = ci.
5. **Positions** in a chunk: p = j * VKH * VKW + ky * VKW + kx (channel-major, then kernel raster).
6. **Sub-batches.** A core's virtual channels split into nb sub-batches of cpb channels (last one shorter) when one
   output channel's positions exceed NPUC's counter (`NPULIRKernelMemoryData::getMinNumberOfBatches` @0x2c4ef8 with the
   callback `LIRKernelData::checkMaxValuesInBatch` @0x2c5110, `updateBatchesForFile` @0x2c7be0):
   - the counter walks one (core, output channel) in position order and cuts when a sub-batch would hold more than
     1016 values or 512 values it calls non-zero; **non-zero = stored byte != 0**, not != the zero point (NPUC passes
     0): in quant 1 a zero weight (128) counts, an int8 -128 (byte 0) does not; empty stride taps count as the zero
     point, absent channels as 0. maxrec = the shortest cut (the value at a cut is dropped, a shorter later cut
     restarts the walk);
   - cpb0 = floor(maxrec / K / 2) * 2, nb = ceil(C / cpb0); kept if C % nb != 0 and round_up_even(ceil(C / nb)) * K
     < maxrec, else raised to the smallest divisor of C >= nb (decompiled, Ghidra 0x3c4ef8). **C = the IFM
     ordering's getChannelCount / its IDPs**, not the layout's channels per core: stride 1 the layout's cpc (all
     exact); stride 2 the code's ordering sees s^2 * Cin channels (C = 4 Cin / ncores: Cin 250 -> 250, not 256);
   - over the conv (updateBatchesForFile, Ghidra 0x3c7be0): nb = the largest over every IFM IDP and real output
     channel; then tryBuildData (Ghidra 0x3c5794) sets every batch's cpb = round_up_even(ceil(vcpc / nb)) with vcpc =
     getNumChannelsPerIDP (the layout's virtual channels per core). One split for the whole conv (V: `b15.txt`).
   - **Two passes, either can abort** (each prints `tryBuildData:235 ... kernel VKH x VKW numInputChannels N
     maxPossibleBatches = M` in the log): the planner's (section 7) runs first with a StridedChannelOrdering over
     s^2 * round_up_even(Cin) channels and assignForChannelCountCompact IDPs (Ghidra 0x207410: g = ceil(C/2), ceil(g /
     ceil(g/4))), so for stride 2 and odd Cin its C differs (Cin 253: planner 254 = 2 x 127 -> 127 sub-batches > 16,
     abort; the code's 253 would give 11). Stride 2, 7x7/8x8, quant 1 (the former corner): 64 of 64 variants right
     (35 compiled byte-exact, 23 aborts in the planner, 4 in the code, i.e. the two log lines before the assertion).
     The non-uniform splits once noted (q7i255) were a misreading: all batches share nb and cpb.
   - So the split is **data dependent** (V: Cin 256 3x3 q3 sparse -> 1 sub-batch of 64 channels, dense -> 2 of 32;
     Cin 512 q1 random -> 44 + 44 + 40).
   - Bound: tryBuildData's maxPossibleBatches = the smallest divisor of vcpc >= max(ceil(K vcpc / 511), ceil(K vcpc /
     1016)) (decompiled; vcpc = numInputChannels / IDPs from the log line); more sub-batches abort
     the compile (`"_numBatches[file] <= maxPossibleBatches"`). A chunk with more than 511 non-zero bytes aborts in
     GenBitmasks2020 (`"(nzvSize & 0xFE00) == 0"`): it happens in quant 1 when byte-0 weights (int8 -128) make the counter
     keep a sub-batch that has more than 511 values != 128 (V: Cin 256, all weights -128).
7. **Chunk** = one output channel (slot) over one sub-batch's positions: big-endian u16 header `bitmap_bytes << 9 |
   nonzero_count` (log: `out.size(): 23 bmSize: 3 nzvSize: 18`, out = 2 + bm + nz), the bitmap (ceil(positions / 8)
   bytes, bit p LSB-first set when the stored byte != zero), then the non-zero bytes in p order. An empty slot is a
   header and an all-zero bitmap.
8. **Sub-batch** = lead byte + 8 chunks, slot t = (g % 2) * 4 + q for groups g = 2b, 2b + 1 (batch b) and lanes q.
   Lead byte (`tryBuildData` 0x2c62f8 and the rewrite at 0x2c6b80..0x2c6c2c, getByteAlignment = 64): first
   (64 + body) / 64, then rewritten to the 64-byte rows the sub-batch reaches from its start, floor((size + s - 1) / 64)
   with size = body + 1 and s = its offset % 64 (ceil(size / 64) when s = 0), **except** when the first value is a
   multiple of 64: kept (a quirk: the rewrite tests the lead byte modulo 64). V on all 5992 leads of multi-sub-batch
   convs in the sweep (and every single one); the quirk shows
   on bodies of 4032..4095 bytes after an unaligned start (lead 64 where the rows would be 63).
9. **Batch** = its sub-batches back to back (not row-aligned between them), zero-padded to R * 64 bytes, R = the
   largest batch of the conv in rows (round_up(max batch bytes, 64), all cores and loads).
10. **Block** = for core c: for load l: batch 0, batch 1. Core-major: core stride = nloads * 2 * R * 64, size =
    ncores * nloads * 2 * R * 64.

## 6. Graphs: eltwise, concat, standalone ReLU (V: graph 98/98, graph-random 47/47, graph-random2 66/66)

- **Order.** One block per conv, eltwise and ReLU layer, in NPUC's execution order (the stream part's rule, cmdq.md
  7c: from the single output back, a layer entered once every reader is, inputs in order, depth first; reversed).
- **Convs** pack as in section 1 to 10, with one exception: a conv that reads a **concat kept in SRAM** (the stream
  part's rule: every input a pool or a conv with a multiple of 16 channels, not the output, no eltwise reading it, no
  fork that an eltwise joins) sees per core [input 1's core share, input 2's, ...]: each input split over the cores
  on its own (cpc_t = 2 ceil(C_t / 8)), shares back to back, absent channels at the zero point (cat16, cat16-3,
  cat16-chain, cat-32-16, g141, g161, g169), also through a pool with one reader (c16_16/c16_48/c32_16-catpool-k3).
  A concat in DRAM (g103, g167, g161 l8) and a pool of an SRAM concat with several readers (stored and loaded back:
  g141, g169) give the normal order.
- **Eltwise add** = its CU_BIN (the "META_DATA" of npuc-harness-ops.md): per core of the input's layout (ceil(C/2)
  planes, ceil(planes/4) per core; for exactly 4 channels 4 regions, the last two empty, when an operand is loaded
  from DRAM, i.e. the network input or a concat (rc4-*, g120, g134), else 2 (both operands in SRAM: g513; the stream
  part's 4 IR elements follow the same condition)), three arrays (lowered `fl`: input 1, input 2,
  output), one byte per channel of the core, C split evenly over those cores (ceil(C / cores) each, the rest on the
  last: C = 36 gives 9 per core though the planes give 10, 10, 10, 6), each array padded to 4 bytes, the core's
  region to 32 (C = 2: 32 B; 3: 64; 4: 128; 12: 96; 36..64: 256).
- **Standalone ReLU/ReLU6** = a 1x1 dummy conv over its C channels, normal batches (Cin = Cout = C):
  - reading a conv or a concat: every weight at the zero point, every chunk empty (cat-relu, cat16-relu, g008, g029,
    g147, g161, grelu r*);
  - a ReLU reading a ReLU takes the inner one's form (V: relu<concat zero, g555; relu<eltwise and relu<pool<conv
    identity, g002, g016);
  - reading a pool or an eltwise: NPUC's depthwise identity, through IterateOverAllValues' slotted path
    (@0x2c48b8, decoded): for output o (o = 0xFFFFFFFF for a lane at or past the group size gs; a lane under gs whose
    o >= C keeps its o) and core c the walk visits w = c * floor(C/4) + 2 pos + ci + 4 ((o mod floor(C/4)) div 4)
    for the core's channel pairs pos, ci = 0, 1, and emits a position when (w & ~3) mod floor(C/4) = (o & ~3) mod
    floor(C/4) (ARM udiv: x mod 0 = x): the stored byte 1 where w = o (a real output; 1 in quant 1 too: u8 1, V
    g-pool-relu-q1), else the zero point. floor(C/4) is not the layout's cpc: for C not a multiple of 8 the windows
    overlap or skip channels (C = 18: cores see 0-5, 4-9, 8-13; channels 14-17 never pass), so NPUC's ReLU layer
    drops or doubles channels there (G for what the NPU then computes). 196 of 196 such blocks exact.
  - reading the network input: not compiled; taken as the identity form (G).

## 7. NPUC's two packs: the planner's (compWeight) and the code's

NPUC packs each kernel twice (two `tryBuildData:235` log lines): first in its policy maker (makeCompressedKernel
@0x29c618 and its worker @0x29df7c), whose size is the log's `compWeight=` and the WEIGHT SRAM chunk the allocator
asks for, then for the code (getConvKerLoader, the IFM buffer's AllocationChannelOrdering). The blocks come from the
second. Decoded from the worker: the planner's IFM ordering is a StridedChannelOrdering (type field 1; channel(pos,
ci, idp) = 2S (idp nper/2S + pos div S) + pos mod S + S ci, nper = 2S ceil(ceil(ceil(C/S)/2)/IDPs), S = 1 or the
stride's phases; @0x1c1aa0, @0x13d7b8, @0x11f434) over the IFM's own channel count and IDP set
(assignForChannelCountCompact): the normal order. For a conv whose input is a Concat (or a Pool of one), it also
collects one (vfunc 0x28, 0x30) pair per concat input and packs a second "MF" kernel over a buffer with
setChannelMFAccIndex (allocateMFWeight @0x118518), kept as the MF compressed kernel (setIdToMFCompKernel).

**The MF kernel** (DynamicBuffer::setChannelMFAccIndex, Ghidra 0x2237d0, decompiled): the buffer's channel table
gets one level per IDP (n = assignForChannelCountCompact of the whole concat: P = ceil(C/2) planes, n = ceil(P /
ceil(P/4))); for each concat input t (a_t = planes before it, b_t = ceil(C_t/2) planes) every IDP gets the table rows
ceil(a_t/n) .. ceil((a_t+b_t)/n) - 1, 2 channel slots a row, numbered on from 2 a_t across IDP 0, then 1, ... The
number is used as the kernel channel as is, so when an input does not fill its rows the numbers run into the next
input's channels (and past C: absent). `_mf_conv(L, chans)` builds that kernel; packed like any conv.

`weight_planning_sizes(low)` (one per WEIGHT block, NCP order; `weight_layer_info(low)` adds the geometry) = compWeight
on **5160 of 5160** conv/ReLU blocks of every compile under output/npu (the rest refused like NPUC):
- a conv whose first input is a concat (or a pool of one): max(the normal order's pack, the MF pack), the MF one
  only when every concat input has an even channel count (V: 224 of 224 concat readers in graph*/ and weights/cat,
  incl. g103, g161 l13, g167, c12_20-k5-conv, c2_30-k5-conv; odd splits 1+7, 3+5, 9+7, 3+3+2... never show it: G
  for why, presumably a pair-granule check before allocateMFWeight). The normal order is the unpermuted one even for
  an SRAM concat (c16_8_8-k5-pool 14848 vs block 14336);
- stride 2: the planner's own sub-batch split (layout section 6; differs for odd Cin: q7i255 135168 vs block 134144);
- a standalone ReLU's identity dummy conv: its block counted on 4 cores (C <= 16: 512 B; C = 18: 1024 for 768);
- everything else (chains, zero-form ReLUs): the block's size.
- For the stream part (V: g103, g167, whole file): the WEIGHT ring's second slot sits at the chunk base + 2 x
  planning rows, the load within a slot at + rows_per_batch (MTC 0x40120300.. = R << 16 | wbase + 2 wslot prows +
  j R: NPUC 0x70/0x93 where 2 R would give 0x6e/0x91).
- **Aborts in the planner** (V: Cin 81..83 k5, 226 k3, frame/s18/r3-080; 84..122 k5, 234 k3 compile): its counter
  sees floor(Cin/4) channels per core; when none reaches the counter's limit it keeps one sub-batch and the chunk
  overflows GenBitmasks. `_planner_check` refuses exactly these (G for the channels it walks).
- Concat-reader sweep for this (`$NCPGEN_WORK/sweep/weights/cat/`, `tools/catsweep.py`, `tools/catcompile.sh`, NPU_ROOT
  root-b53daf10, quant 3): splits 4+32, 32+4, 3+5, 5+3, 12+20, 20+12, 8+8+16, 16+8+8, 4+12, 16+16, 32+16, 2+30, 6+10,
  1+7, 24+40, 4+4+28, 16+48, and (DRAM, `d*`) 4+4, 2+6, 6+2, 1+3, 1+15, 3+13, 5+11, 7+9, 9+7, 11+21, 13+19, 1+31,
  3+29, 2+2+4, 1+1+6, 3+3+2; producers convs or pools of convs; readers K 1/3/5, and concat -> pool -> conv: 118 of
  151 compiled (NPUC crashes on SRAM concats of pools with odd channel counts, createModuleFwdBuffer), blocks 118/118.

## What the other parts need (V unless marked)

- **io descriptors** (container/frame): one per load and conv, in load order: src = the WEIGHT av, offset l * 2R * 64,
  `fmt 288 bpp 128 n 1 c <ncores> h <2R> w 64`, src strides 0; dst = vector 0 (SRAM) at the weight chunk's offset,
  same shape, dst c_stride = 0x40000 - 2R * 64 (0 when ncores = 1): 1354 of 1354 weight io descriptors.
  `weight_geometry(low)` returns ncores, nloads, rows_per_batch (R), load_h (2R), load_src_offsets, core_stride, size.
  Note (G): with nloads > 1 the core stride in DRAM (nloads * 2R * 64) is not the load size, yet src c_stride is 0;
  the firmware must take it from elsewhere (av size / c fits). Never run on the NPU (all phone runs had nloads 1).
- **CMDQ** (stream): the conv's MTC/weight-fetch registers presumably carry ncores, the sub-batch count and channels,
  R and the load count (not decoded here, G). NPUC's `weightsSizeInBytesForOneWBuf` = cpc * K (V) and the OFM group
  size gs (the "rounded to 16 from 8" message for Cout 8) are the same geometry.
- **Memory planning**: NPUC aborts when an SRAM chunk exceeds getMaxChunkSize = 0x40000
  (`"size <= ...getMaxChunkSize()"`, V on Cin 900 5x5 Cout 37); the WEIGHT chunk it allocates is logged as
  `WEIGHT chunk, ... size=` (equal to the per-core load bytes for small convs, larger for big ones: not modelled here).
- NPUC limits seen on the way: SAME padding > 6 fails (`ncp_nfu_pooling: Invalid pad size`, 15x15 SAME).

## What remains

- Not seen and not handled: a batch with no chunk (tryBuildData writes lead 1, size 1 when a load has <= 4 DRUs:
  0x2c639c; never reached by a conv on this path), depthwise/grouped convs (refused upstream by npunet).
- On the NPU: sub-batches, nloads > 1, stride 2, graphs and quant 3 multi-layer blocks have never run; the bytes equal
  NPUC's, so they stand or fall with NPUC (proposal: run `fo20` (Cout 20, 2 loads) and `fi256` (2 sub-batches) through
  npu-run with the CPU reference; and a ReLU layer with C = 18 after a pool, to see NPUC's channel drop on the NPU).
