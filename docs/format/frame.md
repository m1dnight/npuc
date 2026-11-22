# The NCP container, the ARM intrinsics and the NPU's data layout (PORT-32, NCP format, part "frame")

Exynos 990 NPU (NPU2020 ISA, SoC model Neus), NCPs compiled by Samsung's NPUC 1.36.1.a run on the host as a byte-exact
oracle (`oracle/npuc-local.sh`). This part covers everything in an NCP that is not a CMDQ stream, a GReLU
table or a WEIGHT block: the header, the address/memory/group vectors, the io (DMA) descriptors, the body and weight
placement, the zero tail, the ARM intrinsics, and the byte layout of the input and output feature maps. Code:
`ncpgen/frame.py`; sweep definitions: `ncpgen/frame_sweep.py` (compiled
variants under `$NCPGEN_WORK/sweep/frame/`, git-ignored).

Tags: **V** verified (oracle bytes or log, NPUC disassembly, or an NPU run), **G** guess. Library addresses are the
libraries' own. No vendor bytes here: field layouts, formulas, register/table addresses and observed constants only.

## 1. Summary

- `frame(low, streams, intrinsics, weights)` and `intrinsics(low)` rebuild **every one of the 1167 frame-sweep
  compiles byte for byte** (whole file, with the oracle's own streams and weight blocks cut out of it; 26 families,
  `frame_sweep.py`): channel/shape sweeps (C 1..300, Cout 1..512, H and W 1..1000, odd sizes, 1x1), 430 random chains
  of up to 5 conv (k 1..7, stride 1/2, SAME/VALID, with/without activation, quant 1/3, zero and sparse weights) and
  max/avg pools, tiled plans (by height, width, both and SRAM; strata of up to 3 convs, conv + pool, several strata,
  tiled and untiled blocks, split output read-back), weights streamed through rings (untiled and tiled, partial
  sums, odd Cout, pass-by-pass stores) and NPUC's copy group (section 6.7). Also the stream part's single-tile and
  tile sweeps (376/376), every chain of the regression sets (corpus 40, random 77, random2 95, realish 5, gap 5, gap2
  3, poollim 10) and 202 of the 211 graph compiles (section 6.8; the others: 7 with uninitialised descriptor words in
  NPUC's output, 2 the stream part's SRAM offsets). With the other parts (`ncpgen.py check`, whole files generated
  entirely by our code): frame's sweeps 289/289, tile 73/73, stream 302/303, regression 220/220, graphs 201/211.
- The whole frame follows from NPUC's **plan**, which `plan()` reproduces: one group per layer plus an output group;
  the ops each layer lowers to (PAD, REFORMAT, CONV, POOL); NPUC's internal shapes; per-IDP SRAM chunks placed by its
  "bigger first" allocator; the DMAs. `plan()` also gives the stream generator the SRAM offsets it needs.
- **The data layout** (the open question for other shapes): the IFM is NPUC's format 0x120 (section 8):
  `byte = (c / 2) * Hp * 2Wp + (y / 4) * 4 * 2Wp + (x / 8) * 64 + (c % 2) * 32 + (y % 4) * 8 + x % 8`, with
  Hp = H rounded up to 4 and Wp = W rounded up to 8, in a buffer of n * pp * Hp * 2Wp bytes (channels dealt to n of
  the 4 IDPs in blocks of cpi = 2 * ceil(C/8)). It is the N3b-measured 8x16x16 rule, now derived for every shape from
  the descriptors (V), the per-plane SRAM addresses in the stream (V) and the weights' channel-to-core split (V);
  the byte order of other shapes is G until probed (8.5 lists 13 NPU runs). The OFM is dense CHW for every shape
  (the output intrinsic's formatter; V on the NPU for 8x16x16 and 8x8x8).
- **Two NPUC defects in DMA strides** (G until run, section 9): outputs with 3, 5 or 7 channels are read back from
  before their DRAM buffer, and a conv after a pool that kept an extra column stores overlapping planes.
  `frame(..., fix_npuc=True)` writes the consistent strides.

## 2. The file, region by region (V)

| Region | Offset | Content |
|---|---|---|
| header | 0 | 32 words (below) |
| address vectors | 0x80 | 16 B each: index, m_addr, s_addr (0), size |
| memory vectors | after the av | 32 B each: type, pixel_format (1), w, h, c, w_stride (0), c_stride (0), av index |
| group vectors | after the mv | 40 B each: index, type 0, size = intrinsic + isa size, status 0, flags 0, batch 1, intrinsic_offset, intrinsic_size, isa_offset, isa_size (offsets relative to the body) |
| io descriptors | after the gv | 124 B each (section 5) |
| header pad | | zero up to hdr_size = io end rounded up to 16 |
| body | hdr_size | the groups' intrinsics back to back (each at the running offset, sizes multiple of 4), zero pad to 16, then the CMDQ streams back to back (each a multiple of 16 B) |
| weights | | each WEIGHT block at a 64-byte aligned file offset (its av m_addr; "Byte align size added", getByteAlignment() = 64), in layer order, zero gaps |
| zero tail | ncp_end | 360 B per io descriptor ("rq vector size"; the firmware's request queue space) |

Header words: w0 0x0C0FFEE0, w1 22, w2 hdr_size, w3 21 (intrinsic version), w4 100 (net id), w5-w9 0, w10/11 av
offset/count, w12/13 mv offset/count, w14/15 gv offset/count, w16 0, w17 body offset (= hdr_size), w18 body size
(intrinsics + pad + streams, not the weights), w19/20 io offset/count, w21 ncp_end (end of the last weight block, or
of the body), w22 zero tail length, w23-w30 0, w31 0xC0DEC0DE. The file length is ncp_end + tail.

## 3. Address and memory vectors (V)

In this order, one av per mv (av index = mv index):

| # | mv type | shape | av size |
|---|---|---|---|
| 0-3 | OT_PIX0..3 (3..6) | 0 | 1 MiB each: the four IDP SRAMs (DMAs address IDP k as vector 0 + k * 0x40000, or as vector k) |
| 4 | IN_FMAP (0) | (W, H, C) | n * pp * Hp * 2Wp (section 8) |
| 5 | OT_FMAP (1) | (W, H, C) of the output | C * H * W (dense CHW) |
| 6 | IM_FMAP (2) | below | the DRAM blob the last layer is stored to, read back by the output group |
| 7.. | WEIGHT (7) | 0 | the block size; m_addr = its file offset |
| last | IM_FMAP (2) | (8, 8, 2) | 138 (always present) |

The IM vector (NCPObjectGenerator2020::FillVectors @0x9c230, disassembled; NPUDevice getCellWidth() = 8,
getCellHeight() = 4): the DRAM blob holds roundup(C, 16) channels in format 0x120 on NPUC's internal output shape
(section 6.2), size S. If S does not exceed C * H * W of the output's true shape, the vector is (W, H, C) (rounded to
8, 4 and even); else ("Shape is not matching with allocation size") with q = S / 4: q <= 8192 gives (q / 2, 4, 2);
otherwise w = the largest power of two <= 8192 dividing q (at least 8) and c = q / w rounded up to even, h = 4. The av
size is then raw + 10 * (raw >> 8) + 10 with raw = w * h * c (so for big blobs the av is larger than the blob: 34816
for a 33792-byte blob of 44x44).

## 4. Groups and the body (V)

One group per layer (in order), then the output group. A group's intrinsic is the 32-byte stub when the group owns
at least one io descriptor (sgid = its index), nothing (size 0, offset = the running offset) when it owns none (a
layer whose input and output stay in SRAM), and the output function (section 7) for the output group.

## 5. io descriptors (V)

Words: uid (1..n in order), bid (batch, below), sgid (the group),
wait_flag 0, trig_flag = 1 << ((uid - 1) mod 16) (the wrap seen at the 17th DMA), mode 2, pad 0, scale 1, bias 0,
then two blobs (source, destination) of 11 words: vector, offset, format 288 (0x120), bpp 128, n 1, c, h, w,
w_stride 0, c_stride, n_stride 0. A blob is c blocks of h x w bytes, block i at offset i * (h * w + c_stride).

Order: the IFM load(s), every conv's weight loads (in the conv's group), the last layer's store(s) (in its group),
the output group's read-back.

Batches are declared by the CMDQ streams (V on all 840): each group's stream writes `ev2set` (CMDQ 0x401e2024,
opcode 0x8809) once per batch with the batch's trigger bits << 16, in uid order (a batch of more than 16 DMAs gets a
second write for the wrapped bits). The write after a full mask (0xffff) continues its batch (bc-241: 1 + 15 then 1;
ws-167-512, r_c256h40: 4 + 12 then 3) unless one of its loads targets an SRAM offset a load of the batch already
fills: the next round of a weight ring is a new batch (ws-64-512k5, r3-056: 1 + 15, then 15), or, in a tile group
with a TRIM or a weight ring, once the batch holds 15 weight loads (the stream part's cap, cmdq.md 7b; tiling3 tr-c8:
1 + 15, then the 16th pass alone). `frame()` reads the bids off the streams it is given (`batches_from_streams`), so
the io descriptors always agree with the stream generator; `plan()`'s own rule (runs of one direction in one group,
at most 15 DMAs) is the fallback and disagrees only with some batches of exactly 16 loads.

| DMA | Form |
|---|---|
| IFM load | spread over the n IDPs: c = n, h = pp * Hp, w = 2Wp, SRAM c_stride = 0x40000 - h * w (DRAM 0). One IDP (C <= 2): c 1, h = size / 64, w 64. **Split** when the first op is a stride-1 conv without padding (it reads the IFM directly): the IDPs holding pp planes in one blob, the last IDP (fewer planes) on its own from DRAM offset (n - 1) * pp * plane to vector OT_PIX<n-1>, 64 B wide. With a PAD or POOL first the LIR loads plane by plane; when n * pp >= 64 (CODEGEN_MINIMAL_LOADS_NUM_FOR_BATCH_CLUSTERING) NPUC's DMACluster makes 4 (CODEGEN_CLUSTER_NUM_BATCHES) DMAs of ceil(pp/4) planes per IDP (the last the rest), DRAM c_stride per_idp - size. A REFORMAT first: one blob |
| weight loads | per conv: nloads = ceil(Cout / 16) loads of part = (block / n_in) / nloads bytes per IDP (n_in = the IDPs of its input): c = n_in, h = part / 64, w 64; DRAM offset k * part, DRAM c_stride block/n_in - part (0 when one load or one IDP); SRAM offset = the WEIGHT chunk + (k mod R) * part (R = the ring's slots, section 6.6), SRAM c_stride 0x40000 - part |
| store | from the last layer's output chunk to IM offset 0: spread over the IDPs holding pp planes; the last IDP, when it holds fewer, on its own from vector OT_PIX<i>, 64 B wide, to IM offset (planes before it) * plane. Its plane count: its real planes after a conv; after a pool pp rounded down to even (V on 10 variants) |
| read-back | into SRAM offset 0 of the output group: spread (c = n, IM c_stride 0, SRAM c_stride 0x40000 - h*w) when C = n * cpi and (n = 4 or cpi = 2); else gathered into IDP 0: one blob 64 B wide of ceil(C/2) planes, except when cpi = 2 (one plane per IDP, C = 3, 5, 7): c = n, h = Hp, w = 2Wp, IM c_stride **-2 * plane** (NPUC's 'STRIDE' pattern; section 9) |

## 6. NPUC's plan (V unless marked)

### 6.1 Ops

A conv lowers to PAD (if any pad), REFORMAT (if stride > 1) and CONV; a pool to POOL (its padding inside); then one
UNFLATTEN for the output. Ops are numbered 1.. in order (NPUC's 'order'), the unflatten last.

### 6.2 Internal shapes

NPUC's backend shapes equal the lowered (TFLite) ones except after a VALID pool whose window does not cover the
input: `NNPoolLayer::computeOutputShape` (frontend 0x90f98) computes out = (in - k) / s + 1, warns "Padding in Pooling
has negative value!" when (out - 1) * s + k - in < 0, and **when only the horizontal value is negative keeps one more
column** (25 variants; G for the exact condition in the code). The wider tensor flows on: a conv's output is
(in + its lowered pads - k) / s + 1 on it, a pool applies the same rule again; the output intrinsic drops the column.

### 6.3 SRAM chunks (per IDP)

INPUT n_in's pp * Hp * 2Wp; PAD the same on the padded shape; REFORMAT (stride s) cpi * s * s / 2 planes of
ceil(Hpad / s) x ceil(Wpad / s) (a space-to-depth inside each IDP; G for s > 2); CONV the output with roundup(Cout,
16) channels ("Convolution Ofm will be rounded to 16 ... due to HW bug"); POOL the output with its real channels;
WEIGHT block / n_in, or R of its passes when the weights stream (section 6.6); PARTIAL_SUM (a conv whose weights
come in sub-batches, weights.py's count) 2 x 7 planes over the MTC input (padded, or the space-to-depth input), 896 B
per cell. Live ranges [order of the defining op, order of its consumer]; WEIGHT from the layer's first op to its
CONV; PARTIAL_SUM its CONV only; INPUT [1, 1].

### 6.4 The allocator ('bigger first')

Allocation groups: the input, then each layer's chunks; within a group by size, largest first, ties to the chunk
created last (WEIGHT, then CONV/REFORMAT, then PAD: every tie seen), PARTIAL_SUM last; each at the lowest offset clear
of every placed chunk whose live range overlaps (inclusive). The output group's input is at SRAM 0.

When a group ends past 256 KiB ("Cannot allocate with current strategy, bail out!") NPUC tries **backward**: the
chunks by the start of their live range, latest first, equal starts smaller first (the reverse of bigger first; V on
the 45 logged compiles that try it: a CONV and its WEIGHT starting together go smaller first, final-chains f014,
tiling2 b10), PARTIAL_SUM still last; then **"all other permutations"** of that order, PARTIAL_SUM included, in lexicographic order
(std::next_permutation), keeping the first that fits (V: r_c256co128 untiled, 4 tries; r3-064; t-32x32-256-512,
ts24-160, ts40-128 tiled). Chunk offsets are 64-aligned (V: r3-095, after a 183504-byte ring). When no order fits a
layer of a single-tile block, NPUC copies the layer's input (section 6.7).

### 6.5 Tiled plans (`plan_tiled`, V on 150 tiled and mixed compiles unless marked)

**When, which chains, how many tiles: NPUC's policy (6.10, `npuc_policy`).** A layer is tiled when its input
reaches its bound (a padded conv: 256 - pt - pb rows or 256 s - pl - pr columns; NFU ops are cut into command pieces
instead, ceil(rows / 248) by ceil(columns / 112), rows outer: the stream part's rule) or when NPUC's memory estimate
of the whole layer, even with a weight ring, passes 256 KiB. Chains of layers share one stratum (tiled together,
intermediate maps in SRAM) as 6.10 grows them from the tail; the other layers stay untiled ("single" blocks, exactly
the single-tile plan of section 6.1-6.4, reading the previous block's output from DRAM). Each stratum and each single
block writes its output to its own IM vector (in order); the next block reads it.

**Tiles.** The stratum's last OFM is cut into nh x nw tiles of rup(ceil(D / n), cell) per axis (cell 4 rows, 8
columns; the last tile the rest), (nh, nw) from the policy. The tile's input is loaded from the cell boundary at or
before the first needed row/column (a stride-2 first conv may load wider than 255: its reformat reads 128-column
pieces, V: tc_s2 8x600 in 2 tiles with 305-column loads). Each layer's ranges and pads come from back-propagating
the tile through the stratum (clipped; a tile whose input reaches the tensor's edge gets the layer's whole pad on
that side, even when its outputs need none of it: tiling2 b01, b02, a 5x5 conv's next-to-last tile ending on the last
input row is padded 2 rows below). A tile whose output reaches the stratum output's last row (column) computes every
layer's output to that layer's last row (column), its input to the tensor's end with the trailing pad: NPUC's layouts
add each layer's rounding left-over to the last tile (6.10), so a layer whose output the next does not wholly read (a
VALID 2x2 pool on an odd size) is still computed whole there (V: tiling2 e03, e04; the stream part's tile_to_end). Tiles are row-major, one NCP group each; tile 0's group also loads all the stratum's
weights, unless they stream (then every tile reloads every pass into its ring slot, section 6.6).

**Ops and SRAM.** Per tile: SPLIT (the loaded input), TRIM when some tile's input starts inside a cell, then per
layer PAD (if the layer pads), REFORMAT (stride 2), CONV or POOL; chunk sizes = the largest over the tiles. Orders:
SPLIT [2, 2], tile 0's ops from 2 ([o, o + 1], the last [o, o]). When a tile has more than one op the WEIGHT chunks
are first citizens, live over all of tile 0, allocated first, each at the end of the SRAM holding less so far, the
bottom on a tie: the first at 0, the second at the top (0x40000 - size), a third under the second or after the first
(V: 19 three-conv strata of tiling2; G for the rule); with one op they are ordinary chunks of its group. Second citizens by group: SPLIT,
then tile 0's op chunks, a conv's PARTIAL_SUM last. When 'bigger first' overflows 256 KiB NPUC falls back as in 6.4
(160x160x24 fits two tiles this way). The output read-back chunk is at 0.

**DMAs.** Between a DRAM tensor (IDP-major, real planes per IDP) and SRAM: a tile spanning the whole width moves
h = rows (cell-rounded), w = 2Wp, DRAM c_stride = block - h * w; a narrower tile moves cell rows: h = rows / 4,
w = cells * 64, DRAM w_stride = the skipped cells, c_stride = block - h * (w + w_stride). One plane per IDP: one DMA
over the IDPs (SRAM c_stride 0x40000 - h * w). More planes: one DMA per IDP through its own SRAM vector, c = planes
(DRAM c_stride = plane - span, SRAM planes contiguous, SRAM n_stride 0x40000 - planes * h * w; 0 for the read-back,
for one plane, and for the input of a one-layer stratum whose conv reads it directly (k1, no pad: ts128c32k1, nsA,
nsD, nsE; a longer stratum starting so keeps it: nsC, m3-mnet-stem224)). Loads move every IDP's pp planes; stores
from an IDP holding fewer planes (the last one) move only its real planes (a pool's: pp rounded down to even, as
untiled; V: tw300c9, tw300c20, tw320pc20).

**Output.** The read-back is cut in rows by NPUC's policy on the output (unflatten) node (6.10; V: every output
node's tile count and memory print in the 1577 logged chain compiles): whole when its input (16-rounded channels
behind a conv) plus its output, per IDP in cells, fit 256 KiB; else the fewest tileable row counts n >= 2 whose first
tile of rup(ceil(H / n), 4) rows fits twice (input and output, both of the real channels: ts208 one; t2d-300x300,
ts216 two; rb120w72, final-chains f034 three; tiling3 tr-2nd four, tr-320 six). This replaces the L = 274432 bracket
of the rb sweep. Each tile is its own output group whose intrinsic writes its rows at dst offset r0 * W with dst
c_stride (H - rows) * W. An untiled network whose read-back
needs it is planned by `plan_tiled` (one single block; V: final-chains f001, f061).

### 6.6 Weight rings and partial sums (V unless marked)

A conv's weights come as P = ceil(Cout/16) passes of one slot each (slot = block / n_in / P). When they do not fit
beside its feature maps NPUC **streams** them: the WEIGHT chunk holds R slots and pass k loads into slot k mod R, the
next round's loads waiting for the conv that last used the slot (the stream part's emission; `batches_from_streams`
then starts a new io batch, section 5).

- **Untiled** (the stream part's rule, `stream.weight_ring`; V: c256 6 of 16, r_c256h24 8, r_c320h24 7 of 20,
  r_c512h16 7 of 32, r_c512h24 2, the wstream, wlive and ringc sweeps, r3-*, mhead512): R = P when the conv node (its
  MTC input = REFORMAT, PAD or the input chunk, its CONV chunk, its PARTIAL_SUM, plus every slot) fits 256 KiB; else
  R_max = (256 KiB - feature maps) // slot and R = ceil(P / ceil(P / R_max)) (balanced rounds). R_max < 1: tiling.
  NPUC's own rule (6.10, getBestProcessingParams; `plan()` takes R from `npuc_policy`) gives the same R on all 2245
  untiled rings of the regression sets, and one more case: when every pass fits only with a short lifetime and a ring
  of 5 or more slots fits with an extended one, NPUC takes the ring (tiling2 a02: 6 of 12, extended).
- **Tiled** (V: NPUC's code, 6.10, and its log on c512 2 of 32, r_c384 6 of 24, r_c448 4 of 28, t-32x32-256-512 11
  of 32, th-256-h64 4 of 16): a tiled conv alone (chains always hold all their weights) with P < 33 passes first
  looks, column count by column count, for the fewest row tiles holding every pass; when none fits it takes rings of
  ceil(P / k) buffers, k = P .. 1, growing while the row count does not rise (P >= 33: rings only). Every tile
  reloads every pass.
- **The ring's chunk** (V: ws-272-248k5, r3-046, r3-095, r3-103, rc-*; 20 compiles): NPUC sizes a slot from the bytes
  per output channel, e = 16 x ceil(per / Cout) (per = the conv's block per IDP), which is the load size when Cout
  is a multiple of 16 and slightly more otherwise (it counts the last pass as full). The WEIGHT chunk is R x e; slot
  k starts at k x round_up(e, 64) (r3-095: the third slot passes the chunk's end by 48 B). R itself uses the load size.
- **The ring's live range** (V: the wlive sweep and r3-*: R 1..4 in 9 compiles, R 5..16 in 30): a ring of at most 4
  slots lives on its CONV only; a larger ring, like resident weights, from the layer's first op (its PAD): NPUC's
  short and extended weight lifetimes (`weights_have_extended_lifetime`, 6.10: an extended ring needs 5 slots or more).
- **Pass-by-pass stores** (V: rc-167-488/500/504/510, rc-200-500; G for the condition): when the last conv's weights
  ring, Cout is not a multiple of 16 and its weights have no sub-batches, its output is stored pass by pass: the io
  descriptors alternate weight load p, store of pass p (planes 2p and 2p + 1 of every IDP: the IDPs holding as many as
  IDP 0 in one blob of h = planes x Hp, w = 2Wp, SRAM c_stride 0x40000 - h x w, DRAM c_stride = the IDP's block - h x
  w; an IDP holding fewer on its own through its SRAM vector, 64 B wide). With sub-batches (rc-272-*, rc-512-*) or a
  multiple of 16 (rc-167-496, ws-167-512) the store stays one DMA after the last pass.
- **PARTIAL_SUM** (a conv whose weights come in sub-batches): section 6.3; it counts in the node and is allocated last
  in its group. weights.py's `conv_block(...)[0]["sub_batches"]` gives the count; when weights.py cannot pack a conv,
  `plan()` refuses single-tile plans over 240 KiB rather than guess.

### 6.7 The copy group: "trim and spill" (V: NPUC's log; ws-191-256x2, r3-076, r3-077, r3-096, r3-111 and the spill sweep, 18 compiles)

When 'bigger first', 'backward' and every permutation fail for a layer of a single-tile block ("Failed all allocation
strategies, rely on trim and spill!"), NPUC "reloads the IFM into the start of the available address space": the
layer's input (the previous layer's output chunk, "active IFM") is copied by a **TRIM** node into a new chunk of the
same size at the start of the SRAM (0: 18 compiles), or at its end (0x40000 - size) when a copy at 0 would overlap the
input itself (final-chains f014: input at 23040, 27648 B, copy at 234496, "Demanded address"; G for the exact
test), live at the layer's first atom only (its order: the TRIM
takes no order of its own); the input's live range is cut to end before it; then the layer's chunks are placed again
bigger first, equal sizes in creation order (PAD before CONV: ws-191-256x2, r3-096). If even that fails NPUC aborts
(r3-103: GlobalMemoryAllocator.cpp:839). The copy is its own NCP group, right before the layer's group: no DMA, no
io descriptor, **no intrinsic** (gv intrinsic size 0, `intrinsics()` returns b"" for it), only a stream. Vectors are
unchanged. `plan()` marks it: the TRIM chunk carries `copy: True`, `src` (the input chunk's offset) and `layer` (the
reader); `groups[g]["copy"] = {layer, src, dst, size}`.

**What the stream must emit for the copy group** (V: decoded from the 9 compiles' streams, e.g. ws-191-256x2 g1): one
NFU copy per plane slot s = 0 .. ppc - 1 of the input tensor (ppc of its real channel count: 32 for 256, 29 for 232,
25 for 200, 16 for 128; V: sp-c232, sp-c240, sp-c248q3), each exactly `emit_trim` with no skip and one piece
(input = output = the tensor's H x W rows and columns; NFU pieces beyond 248 rows or 112 columns: G):

```
r9  (0x401e0124) = s                     ; emit_group's op index
r1  (0x401e0104) = 4
INTC 0x40100028 = cm ; INTC 0x40100024 = cm
NFU 0x40160800 = 5 ; 0x40160228 = 0x10000 ; 0x40160800 = 0xa ; 0x40160228 = 0x10000 ; 0x40160800 = cm
NFU 0x40160208 = 0
NFU 0x40160214 = u8 | (2 x W) << 8 | H   ; u8 = 1 << 26 in quant 1, 0 in quant 3
NFU 0x40160218 = same (written twice)
NFU 0x4016022c = 0x10000 (quant 1) / 0 (quant 3)
NFU 0x4016020c = ceil(W / 8) << 16 | (src / 64 + s x cells(H, W))
NFU 0x40160210 = ceil(W / 8) << 16 | (dst / 64 + s x cells(H, W))
NFU 0x40160200 = 0x20100                 ; copy
NFU 0x401602fc = cm
nop x 4 (not after the last copy)
op3a01 4                                 ; waiti3(0, 1, 4): NFU done
INTC 0x40100024 = 0 ; INTC 0x40100028 = cm
```

cm = 0xf in every case seen, including a conv output whose last core holds fewer planes (r3-077, 218 channels: 28
copies, all 0xf). No ev2set and no wait for events (the group has no DMA). The group ends as any other: r1..r15 = 0,
r10 = the group index, nop x 4, padded to 8 instructions. ws-191-256x2: 32 copies, src 134144 (0x830 cells) step 6
cells, dst 0, 0x214 = 0x04000215 (21 rows, width 1), 816 instructions. The next group (the reader) is unchanged except
that its first atom reads the copy (dst) instead of the previous output chunk, and the allocation of its chunks
follows the rule above.

### 6.8 Graph networks: eltwise add, concat, standalone ReLU (`plan_graph`; V on the graph, graph-random and graph-random2 compiles unless marked)

The groups and DMAs are the stream part's (`stream.Plan`: NPUC's layer order `exec_order`, one group per layer
except concats, the unflatten last; every DMA op carries its tensor and SRAM offset; cmdq.md 7c); `plan_graph()`
builds the descriptors and vectors from them.

- **Groups.** A group with DMAs has the 32-byte intrinsic (the eltwise's too), one without none (a pool between two
  layers in SRAM), the unflatten its output intrinsic.
- **Vectors.** OT_PIX0..3, IN, OT; an IM per tensor stored to DRAM, in the order of its first store (an eltwise
  operand, a concat's buffer, the output), by the IM rule of section 3 on the 16-channel-rounded size; then one IM per
  concat by the same rule on its real channel count (not rounded; unused by any DMA: cat2, g005, g007, g019); then a
  WEIGHT per conv and ReLU layer and a META_DATA per eltwise, in execution order (weights.py's `weight_blocks` gives
  the blocks in that order); IM 8x8x2.
- **META_DATA** (`weights.eltwise_block`): per core used, three byte arrays (lowered.json "fl": input 1, input 2, output), each
  one entry per channel slot of a core (cpi) zero-padded to a multiple of 4, in a 32-byte block (64 B when the
  arrays pass 32). Its load moves ceil(size / 64) // cores rows of 64 B per core (0 for 32-byte blocks on several
  cores, 1 on one).
- **DMAs.** A tensor load or store moves the whole tensor spread over its cores (one DMA, or two when the last core
  holds fewer planes, as section 5); weights as section 5 at the stream's slot offsets. A producer of a concat in DRAM
  stores its own tensor (IDP-major) at its offset in the buffer: the earlier inputs' channel counts x half a plane
  (V: cat2, g030, and the weights part's cat sweep with uneven splits 1+7, 3+5, 5+3, 20+12: 86/86). A tensor holding an SRAM
  concat's segments (a pool of it) stores each segment as a tensor of its own after the earlier ones: per core and
  segment, 64 B wide, or per core with one plane per segment when they are alike (g141, g169).
- **Not reproducible:** a 4-channel eltwise (rc4-*, g120, g134) loads each operand in two parts whose second
  descriptor holds uninitialised words in NPUC's output (format fields that are heap addresses, an offset of -128):
  no generator can match those bytes. g103, g167: the stream part's SRAM offsets (a conv reading a concat, cmdq.md
  7c misses).

### 6.9 NPUC's "CMDQ seems too big" cut (`_cut_plan`; V on the stream part's cut sweep, $NCPGEN_WORK/sweep/cut, 43 compiles)

When a layer's estimated CMDQ (16350 + 1020 per pass + 60 per padded plane slot + 102 per reformatted one + 150 per
pooled one; cmdq.md 7d) passes 61440, NPUC cuts its group in chunks, each its own NCP group with the 32-byte stub.
`plan()` then takes the groups, the DMA order, the event flags and the batch ids from `stream.plan_dmas` (a group
with 64 loads or more has them in one batch while its triggers come in sub-batches, which the masks cannot tell, so
`frame()` keeps those bids) and builds the descriptors:
- the input of a 1x1 conv: its usual loads, in the first chunk; of a layer an NFU op reads first: per chunk the
  plane slots it pads, as the clustered loads (DRAM offset s0 x plane, h = slots x Hp, DRAM c_stride = block - size);
- each pass's weight load in its pass's chunk at the stream's slot; a kernel over 4 MiB in all is loaded core by
  core, 4 descriptors per pass: source c x (block / 4) + p x part, 1 x part/64 rows, to SRAM vector c at the slot
  (k3c64-ci512, s2c64; the same in uncut single-tile groups and tiled strata, every ring round included: tiling3's
  28 ra networks of 4.7-4.9 MB with short rings, e.g. ra002, ra067; `_weight_load`);
- per chunk with passes p0 .. p1 one store of their planes: SRAM OFM + 2 p0 x plane, DRAM 2 p0 x plane, h = 2 (p1 -
  p0) x Hp, the IDPs holding fewer planes on their own (`_range_store`); none when the output stays in SRAM.
`frame.plan` calls the stream part, which calls `frame.plan` back for tiled plans: a module flag stops the recursion.

### 6.10 NPUC's tiling policy, read from its code (`npuc_policy`; V: decompiled, `oracle/ghidra/targets-tiling.txt`)

V on 1577 logged compiles (every chain set of section 11's regression list and final-chains): `npuc_policy` gives
every node's tile count NPUC prints ("Node id#k ... numTiles=[H: W:]"), every stratum ("On nodes {..} trying to use
stratum style tiling with (H: W:) tiles") and, in the same order, every "Memory requirements calculated on tiled
nodes" print (its IFM and Total). Addresses are Ghidra's (libnpuc_template file offset + 0x100000); Ghidra keeps
these names mangled, so the targets match `NewPolicyMaker18computeTilesNumber` and the like.

**Two passes.** `NewPolicyMaker::computeTilesNumber` (0x3a3b48) walks the nodes from the output back and asks
`getBestProcessingParams(node, 256 KiB)` (0x3adc8c) for each one's processing: whole, or tiles (nh, nw), with all
its weights or `nbuf` weight buffers. Then `recalculateTilingForStratum` (0x3ae948) grows chains ("strata") from the
tail and tiles each chain of two or more nodes with one layout; the other tiled nodes keep their own.

**A node's input bound** (`PTraits::getMaxNodeInputShape` 0x34f7c0, with PTraits2020's values; the logs fix both
configuration flags): a conv without padding (256 s_h, 256 s_w); a padded conv min(256, 256 s_h) - pt - pb rows
(`getMaxPadInputShape`: 256 rows, no width bound) and 256 s_w - pl - pr columns; a pool without padding none (V:
300-row pools stay whole), a padded pool 256 - pt - pb rows. Every tile must also stay below it.

**getBestProcessingParams.** An input at or past its bound goes to tiles. Else, with the estimate below:
1. all weights, extended lifetime (`weights_have_extended_lifetime`: from the node's first op) fit: whole;
2. else `tryAllocateWeightsPartially` (0x3b0e2c): the first ceil(P / k), k = 2, 3, .. (P = ceil(Cout16 / 16)) whose
   estimate fits, extended; if all weights with a short lifetime do not fit, that ring is kept when it has 5 buffers
   or more, else the same search with a short lifetime; if they do fit, a ring of more than 4 buffers (extended) is
   still preferred, else all weights, short lifetime;
3. else tiles: nw = 1 .. ceil(Wo / 8) outer, nh = 1 .. ceil(Ho / 4) inner (FUN_003b0fe8 0x3b0fe8), 1 x 1 skipped,
   counts only where `isTileable` (`calcBasicTileLengths` 0x3b309c: n tiles of rup(ceil(D / n), unit), the last the
   rest, exactly n non-empty): the loaded input tile below the bound and `estimateStratumMemoryRequirements` within
   256 KiB. A conv with P < 33 tries all weights first, then rings of ceil(P / k) buffers, k = P .. 1, growing while
   nh does not rise (`findNumTilesAlongHwithWeightsPartiallyAllocated`); P >= 33 rings only. Nothing: assert.

**Chains** (recalculateTilingForStratum, nodes from the output back). A candidate is a tiled node with all its
weights, or a node whose single predecessor is one and that either feeds one or shrinks the data (its input's memory
above its output's, a conv or pool, no eltwise successor: the "pool joins" of 6.5). A chain starts at a candidate and
takes its predecessor while `getStratumTilesLayout` finds a layout for the longer chain: nw from the tail's own column
count up, nh from 1 (isTileable, not 1 x 1), each node's largest input tile below its bound, the head's largest input
tile longer than 3x the layout's overlap along both axes (the overlap stays k - s even along an untiled axis: this is
the old "untiled side under 16" rule), the chain's estimate within 256 KiB; at most 10 nodes
(NEW_POLICY_MAKER_STRATUM_CHAIN_MAX_LENGTH). A chain of one node is dropped (the node keeps its own tiling).

**Layouts** (`KernelNodeTileBackPropagator::calcInputTilesLayout` 0x3b3868, per axis: count, first, middle, last,
overlap): input lengths k + (n - 1) s for each output length, the first less the leading pad, the last less the
trailing one (one tile: less both), overlap k - s behind an untiled layout and k + (ov - 1) s behind a tiled one,
the rounding left-over (0 <= d < s; a pool: |d| < s) added to the last tile. Tile i starts at (first - ov) + (i - 1)
(middle - ov). `getLargestTileShape` (0x3503d0) is the tile with the most memory, the first such in W-outer, H-inner
order (so a shorter first tile can stand for a longer middle one); "aligned", from the cell boundary at or before
its start.

**The estimate.** Blocks per IDP: ceil(C / 2) planes rounded up to the IDPs used, each rup(h, 4) x 2 rup(w, 8)
(`getRequiredMemory` 0x342e64, `calcRequiredMemory` 0x343e04). A conv: IFM (C rounded to 16 behind a conv: "Convolution
successor's ifm will be rounded to 16"), OFM (Cout rounded to 16), working blocks by its mode (`getMode` 0x345f80;
p = min(max(pt, pb), max(pl, pr))): stride 1 and p in 1..3: PAD of h + 2 max(pt, pb) x w + 2 max(pl, pr) (both
sides doubled); stride 2 and p = 0: REFORMAT of ceil(h / 2) x ceil(w / 2) x 4 rup(C, 2); stride 2 and p in 1..3:
PAD, then the REFORMAT of the padded block (4 C); none otherwise (a k1 or VALID stride-1 conv reads its input).
Weights: ceil(block / IDPs), or ceil(that / Cout16) x 16 per buffer (`calcRequiredWeights`). Partial sums
(`PNodeConvolution::getRequiredPartialSummationBlock` 0x346070, `requiredPartialSummationBlock` 0x336220) when one
weight buffer, s^2 p1 x 2 B x ceil(kh / s) ceil(kw / s) (p1 = the IFM's planes per IDP, rounded to even for s > 1),
passes 511 B (`getMaxNzvInBatch`): 896 B per cell over ceil((w + pt + pb) / s) columns x ceil((h + pl + pr) / s) rows
(NPUC adds the vertical pads to the width). A pool: its OFM; a padded average pool one PAD block and no sharing (G for
which pool type that is).
- whole node (`estimateFullMemoryRequirements` 0x34481c): max(IFM + w0, w0 + w1, w1 + w2, OFM + w_last + PS [+ W
  for a short lifetime]) [+ W for an extended one]; without working blocks IFM + OFM + W + PS;
- tiles (`estimateStratumMemoryRequirements` 0x3b1364): the weights and partial sums of every node + the largest
  pair of consecutive blocks along [IFM, a TRIM block when some tile start is not cell-aligned (the IFM then being
  the aligned shape), per node: its working blocks, its OFM], each node's blocks from its largest input and output
  tiles; a tile past a node's bound makes it infinite.

These blocks are not the allocator's chunks (6.4-6.6): the pads are doubled, the partial sums follow the 511-byte
rule (weights.py's sub-batches set the allocated PARTIAL_SUM chunk), one tile stands for all. So NPUC tiles final-chains
f076's 7x7 stride-2 conv (3 row tiles, with its pool) whose whole node would allocate with a one-slot ring: its
estimate counts a 94080-byte partial-sum block the allocation never makes.

**What the final-chains misses were** (all exact now, `frame.py check` 69/69): f076 the partial-sum block above
(layers 0-1 one chain, 3 row tiles); f002 layers 1-2 one chain in 6 x 2 tiles (no row count fits with one column),
layer 0 alone in 3 row tiles; f012 all four layers one chain, 2 x 2; f059, f062 their first three layers (an average
pool included) one chain, 8 and 6 row tiles; f065 its 7x7 conv alone in 3 x 2 (no row count fits with one column);
f014 untiled, its trim-and-spill copy at the top of the SRAM (6.7); f059 also loads its one-IDP input tiles as one
blob 64 B wide (h = rows x 2Wp / 64), as the untiled loads do.

**Sweep around them** (`frame_sweep.py run $NCPGEN_WORK/sweep tiling2`, 84 networks, 7 families of 12 around the 7
cases; b04 makes NPUC itself assert, "Tile end row", after the policy): `frame_sweep.py policy` 84/84 as NPUC,
`frame.py check` 83/83 after four allocation and geometry rules it exposed (the third stratum weight, the backward
order, the edge pads, a ring of 5+ slots preferred: 6.4-6.6). Still G: which pool type has the pad working block, the
2 DRUs per IDP of the partial-sum block for inputs on fewer IDPs, getTile's starts, the leading-edge pad rule, the
copy's start/end test. `frame_sweep.py policy <dirs>` checks the policy against any compile's log.

## 7. The intrinsics (V: disassembled from every variant; our own A32 encoder writes them)

Both start with the same prologue and end with the same epilogue (our `_prologue`/`_epilogue`):

    push {r4-r11, lr}; sub sp, sp, #140; mov r4, #0; mov r5, #1; mov r7, pc
    ...
    mov r0, r4; add sp, sp, #140; pop {r4-r11, pc}

The stub (32 B) has nothing in between. The output function (328 B, 332 when a word needs a movt) does:

1. `mov r3, #0x1d000; ldr r3, [r3, #0x14]; movw/movt r0, #(0x10000 | g); blx r3`: firmware table entry 5 (g = the
   group index).
2. Fills 31 words at sp+12, the io-descriptor layout: uid 0, bid 0, sgid g, wait 0, trig 0, mode 2 (each
   `movw rX, #v; str rX, [sp, #off]`, X alternating r0/r1), pad/scale/bias as `str r4/r5/r4`, then the source and
   destination blobs (X cycling r0, r1, r2, restarting at r0 for the destination; movt added for values > 0xffff,
   e.g. -1):
   - source: vector 0 (the SRAM), offset 0, format 0x120, bpp 64, n, c, h = H, w = W', w_stride = Wp - W',
     c_stride = (Hp - H) * Wp, n_stride 0, where (n, c) = (IDPs, cpi) when the read-back is spread, else (1, C), and
     W' is the internal width (section 6.2);
   - destination: vector 5 (OT_FMAP), offset 0, format 0x110, bpp 64, n 1, c = C, h = H, w = W',
     w_stride = W - W' (0, or -1 after a widened pool), c_stride 0, n_stride 0.
3. `mov r3, #0x1d000; ldr r3, [r3, #0x20]; add r0, sp, #12; mov r1, #0; blx r3`: entry 8, the formatter (0x120 to
   0x110 = dense CHW).
4. `mov r3, #0x1d000; ldr r3, [r3, #0xc]; movw r0, #g; movt r0, #0xffff; blx r3`: entry 3; then returns 0.

## 8. The data layout

### 8.1 The rule (frame.py `cell`, `ifm_addr`, `ifm_size`)

- Cells of 8 columns x 4 rows x 2 channels = 64 B (getCellWidth 8, getCellHeight 4, getInterleavedChannelCount 2;
  inside a cell (c % 2) * 32 + (y % 4) * 8 + x % 8: V by the N3b impulse probe).
- A plane holds two channels: Hp x 2Wp bytes, cell rows of 4 * 2Wp bytes, cells along x.
- Channels go to IDPs in blocks: cpi = 2 * ceil(C / 8) per IDP, n = ceil(C / cpi) IDPs, pp = cpi / 2 planes each,
  plane-major inside an IDP; in DRAM the IDPs follow each other, pp planes each. So channel pair c / 2 is simply
  plane c / 2: the formula of section 1, in a buffer of n * pp * plane bytes (the last IDP may be partly unused;
  padding rows, columns and the odd channel's twin are don't-care: fill them with the zero point).

Evidence: (V) the IN_FMAP size and IFM DMA shapes of every variant (c = n, h = pp * Hp, w = 2Wp, block per IDP);
(V) the PAD op's registers address plane k of an IDP at chunk + k * plane (NFUBR 0x4016020c/0x40160210, in 64 B
units: cin16 plane 1 at 512); (V) the weights part sees input channel j in core j / cpc with the same cpc = cpi
(`weights.py`); (V) a conv's output channel o sits in IDP o / cpi (the split stores of Cout 9/20/25/33) and the
output intrinsic reads n IDPs x cpi channels, Hp x Wp elements per channel; (V on the NPU) 8x16x16. The block (rather
than round-robin) assignment is also what makes NPUC use 3 IDPs for 9..12 channels.

### 8.2 The output

Dense CHW, C x H x W bytes (OT_FMAP), for every shape: the output intrinsic's formatter writes format 0x110 with
n 1, c C, h H, w W (V for the descriptors; V on the NPU for 8x16x16 and 8x8x8). With quant mode 3 the bytes are int8
(as before). After a widened pool the formatter writes W + 1 columns with a -1 row stride (G: correct values, and
one byte written past the OFM's end).

### 8.3 Between layers

In single-tile plans intermediate tensors never leave SRAM (V: no IM vector but the output's in any of the 840
variants); a layer reads its predecessor's chunk directly, in the same cell format. Only the last layer is stored
(IM) and read back for the output intrinsic.

### 8.4 Integration

`frame.ncp_layout(low, ncp)` is a drop-in for `ncpgen/npunet.py`'s `ncp_layout` that accepts every single-tile shape
(it checks the NCP's IN_FMAP/OT_FMAP vectors and IFM DMAs against `plan()`: all modelled variants accepted, the refused ones raise);
npunet's `ifm_addr` with plane_h = Hp, plane_w = 2Wp is already the general rule; its `to_npu` must size the buffer
with `ifm_size` (it already does from the descriptor). npunet's current check (`want = (288, 1, ceil(C/2), Hp, 2Wp)`)
refuses C > 8, C <= 2 and the split loads: replace it by `frame.ncp_layout`.

### 8.5 Probe plan (main machine, the phone)

Each probe is one `npu/npu-run.sh check/compile/run` with two random inputs (`head -c $((H*W*C)) /dev/urandom`),
after npunet uses `frame.ncp_layout`. Bit-exact verifies the IFM rule and the OFM for that shape; a mismatch on the
input side is then mapped with impulses (conv-ref.py's probe generalised to non-square shapes: one impulse per
address bit of the buffer, decoded by output window).

| # | Network (npuc-variant.py args) | What it settles |
|---|---|---|
| P1 | `--in 16,16,16 --conv 8,3` | two planes per IDP: plane-major order and block assignment |
| P2 | `--in 16,16,3 --conv 8,3` | an odd C over 2 IDPs (the odd channel's twin ignored) |
| P3 | `--in 16,16,1 --conv 8,3` | one IDP, the 64-byte-wide single DMA |
| P4 | `--in 7,9,8 --conv 8,3` | rows/columns of cell padding ignored (Hp 8, Wp 16) |
| P5 | `--in 16,16,9 --conv 8,3` | 3 IDPs, the last one half used |
| P6 | `--in 16,16,21 --conv 8,1` | the split IFM DMA (IDP 3 through its own vector) |
| P7 | `--in 16,16,8 --conv 16,3` | OFM read back spread, n4 c4 |
| P8 | `--in 16,16,8 --conv 12,3` | OFM gathered into IDP 0, n1 c12 |
| P9 | `--in 16,16,8 --conv 3,3`, NPUC's NCP then `fix_npuc` | defect 9.1: expect channel 2 wrong, right with the fix |
| P10 | `--in 16,17,8 --conv 8,3 --pool max,2,2,valid` | the widened pool and the -1 row stride of the formatter |
| P11 | `--in 16,17,8 --conv 8,3 --pool max,2,2,valid --conv 8,3`, NPUC's then `fix_npuc` | defect 9.2 |
| P12 | `--in 32,32,256 --conv 256,3` (c256) | a weight ring (6 slots for 16 passes) with partial sums: the reloaded slots and the sums |
| P13 | `--in 32,32,512 --conv 512,3` (c512) | a tiled ring (2 tiles, 2 slots, every pass reloaded per tile) |

P1, P2, P5, P7, P8 matter most for real models (C 3 inputs, C > 8 everywhere).

## 9. NPUC defects found (G until the NPU runs P9/P11)

1. **Outputs with 3, 5 or 7 channels**: the read-back gathers one plane per IDP into IDP 0, and NPUC's LIR loads plane
   k from IM offset -k * plane (its log: "STRIDE : infos.dramOffset 4294966784" for plane 1 of a 16x16 output; "DRAM C
   stride exceeding designated size (0xffff0) from tiled fffffe00"). The descriptor gets IM c_stride -2 * plane, so
   planes 1.. (channels 2..) are read from before the buffer. The consistent value is c_stride 0.
2. **A conv after a pool that kept an extra column**, when that column pushes Wp to the next multiple of 8: its store
   steps the DRAM planes by the true width's plane (IM c_stride = pp * (true plane - internal plane), negative), so
   planes overlap, while the read-back reads them contiguously. Consistent: c_stride 0. (A pool that widened its own
   output stores with c_stride 0.)
3. (G, harmless) After a widened pool the formatter writes one byte past the OFM (section 8.2).

## 10. Device constants (V, NPUDeviceNeus getters in libnpuc_common)

Cell 8 x 4, interleaved channels 2, byte alignment 64, HW address shift 6, SRAM 0x40000 per IDP, 4 IDPs, 4 NFUs,
8 DRUs, 16 MAAs, micro tile 16 x 16, max PU width/height 128/256, max RU file 128, max values per batch 0x3f8, max
non-zero values per batch 0x1ff, command queue 0xf000, instruction length 6, intrinsic parameter alignment 4.

## 11. For the other parts

- **stream**: `plan(low, weight_sizes)["chunks"]` gives every op's SRAM offset/size (the oracle's on every exact
  variant, PARTIAL_SUM chunks and rings included: a WEIGHT chunk with `ring` R < passes streams); in tiled plans
  `["strata"]` holds the blocks (tiled: layers, nh/nw, rows/cols, per-tile ranges and pads, ops, chunks, and `ring`
  {conv layer: R}; single: layers) and `["outputs"]` the output groups (section 6.5). `Plan.frame_stratum` loads the
  weights in tile 0 only: a stratum whose `ring[li]` is below the conv's passes needs every pass reloaded per tile into
  slot p mod R (today such strata all have sub-batches, which `frame_plan` refuses, so stream's own planner runs:
  c512, r_c384, r_c448, t-32x32-*); a ring's slots are `step` bytes apart (round_up(e, 64), 6.6), the copy group
  of 6.7 (`groups[g]["copy"]`), pass-by-pass stores (6.6);
  `["shapes"]` NPUC's internal shapes (a widened pool changes the following ops' widths); groups are one per layer +
  the output group; trigger bits are `(uid - 1) mod 16`; the stream's `ev2set` writes define the io batches (frame
  follows them), and NPUC splits a group's loads 15 + rest except for some batches of 16 (open: the rule behind the
  16-load batches of Cout 225-230 and 233-236 at 8x8x8, section 5). Since 6.10 the strata are NPUC's own (V on 1577
  logged compiles): a stratum may hold a halo pool (k != stride: final-chains f012's max 2/1) and convs with weight
  sub-batches (f002, f059: their PARTIAL_SUM chunks in `chunks`, equal to NPUC's allocation), so `frame_plan`'s two
  refusals (halo pools, sub-batches) now send exactly these networks to the stream's own planner, which fails on them.
- **weights**: blocks are placed 64-aligned; the io descriptors assume each conv's block splits into n_in equal
  per-IDP parts of nloads = ceil(Cout/16) equal loads (true for every block weights.py writes).
- **npunet / npu-run**: use `frame.ncp_layout` (8.4); the input conversion is unchanged.
- **ncpgen gen**: pass `fix_npuc=True` to `frame()` once P9/P11 confirm section 9.
