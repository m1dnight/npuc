# The CMDQ instruction streams (PORT-32, NCP format, part "stream")

Exynos 990 NPU (NPU2020 ISA, SoC model Neus), NCPs compiled by Samsung's NPUC 1.36.1.a run as a byte-exact oracle on
the host (`oracle/npuc-local.sh`). What every instruction of every group's isa area means, and the rules
that produce it from the lowered network. Code: `ncpgen/stream.py` (`group_streams(low)`, `check`,
`show --diff`); sweep list: `ncpgen/stream_variants.txt` run by `ncpgen/stream_sweep.sh` (compiled
variants under `$NCPGEN_WORK/sweep/stream/<name>/`, git-ignored).

Tags: **V** verified (bytes or NPUC's log over the named variants), **G** guess (a rule fitted on few cases, or a
meaning not proven on the NPU). No vendor bytes: addresses, fields, formulas and observed constants only.

## 1. Summary

- `stream.py check` over the whole sweep (303 compiled networks): **293 byte-exact, 0 differing, 10 refused** (every
  group's isa bytes, trailing nops and padding included): 9 tiled networks, and ci64k7s2, whose weight packing
  weights.py refuses (stride 2 with 3 sub-batches). Unseen test: the orchestrator's 80 random networks
  (`$NCPGEN_WORK/sweep/random/`, seed 2026, 77 compiled): **77/77 exact** (60/77 before section 9's rules). Also the
  shared corpus `$NCPGEN_WORK/sweep/corpus/` 40/40, the c3 reference, and net-current (the oracle NCP with fixed=False;
  the GReLU-patched NCP with fixed=True).
- Tiled networks (section 7b): every stratum of NPUC's policy (frame.md 6.10) is now emitted from frame.plan: halo
  pools, convs with weight sub-batches, tiled weight rings, stride-2 tiled convs. The frame part's 153 tiled compiles
  153 exact; the tile sweep 73/73.
- Everything compiled so far (2026-11-22; every set under $NCPGEN_WORK/sweep): all developed sets exact; the final unseen
  sets: final-graph 57/57, **final-chains 69/69** (whole files 69/69; 61/69 and 64/69 before), **tiling2 83/83**
  (whole files 83/83; 40/83 before); **tiling3 302/302** (our tiling3 sweep `stream_variants_tiling3.txt`, 31, and
  the short-ring sweep `stream_variants_ring3.txt`, 271; whole files 302/302; 210/302 before): the trigger batches now
  come from NPUC's event schedule run on a generated memory profile (7b), which replaced the fitted batch rules.
- Graphs (section 7c): eltwise add, concat (in DRAM and in SRAM), standalone ReLU/ReLU6, any DAG of them with convs and
  pools: graph sweep 98/98, random graphs 47/47 and 66/66 unseen, final-graph 57/57, ops-check 12/12.
- Covered: chains of convolutions and pools in any order (a pool may come first), odd sizes anywhere, conv
  kernels 1..13 square and rectangular, stride 1 and 2, SAME and VALID, Cin 1..256, Cout 1..255, H/W 1..250
  (non-multiples of 4/8 included), quant modes 1 and 3, any zero points and scales, ReLU or none, max/avg pools
  (k 2..3, stride 1..2, SAME/VALID), several output passes, weight sub-batches with partial sums, 16+ input planes
  per core (chunked input load), more than 16 DMAs (event wrap), NFU width splitting.
- Not covered: conv strides other than 1/2 (NPUC supports only 1 and 2); pads above 6 (NPUC refuses: "Pad size
  ... can be up to: 6"); a group whose memory profile has no fit (an op kind not in 7b's list) is refused.
- Dependencies: `grelu.py` (the activation table; the slope is NPUC's round-half-away one) and `weights.py` (rows per
  weight batch, sub-batch split). Without them stream.py uses its own derivations and refuses what needs them.

## 2. Encoding (V)

6 bytes per instruction, little endian: u32 value, u16 opcode. set_reg: opcode = block << 12 | (addr - base) >> 2:
CMDQ 8 @0x401e0000, MTC 9 @0x40120000, NFUBR a @0x40160000, INTCOM b @0x40100000. `nop` = 0/0.
`waiti3(irN, rM, v)` = opcode 0x3a00 | N << 4 | M, value v; `wait3(rN, rM, v)` = 0x2b00 | N << 4 | M.
CMDQ names from NPUC's log: r1..r15 = 0x401e0104 + 4 (i - 1); ev2set 0x401e2024; ev2 0x401e2028; swi 0x401e0180.

Group padding (V, all variants): after the last instruction at least 4 nops, then nops up to a multiple of 8
instructions (48 bytes). The last group is further zero-padded (bytes, even a partial instruction) so that the isa
areas of all groups together are a multiple of 64 bytes (ci17: g1 = 414 + 2 bytes).

## 3. Groups and IR elements (V: NPUC's log, "IRIndex", "create group", every variant)

One group per layer, then the unflatten group. The first group loads the input; the last layer's group stores the
output; feature maps between layers stay in SRAM. A group is an ordered list of IR elements:

| Layer | IR elements in order |
|---|---|
| conv | [load IFM (first group)], pads (per plane slot, per width piece), [reformats (stride 2: per slot, per 128-column piece)], then per output pass: load weights, conv; [stores] |
| pool | [load IFM], pools (per plane slot, per width piece); [stores] |
| unflatten (last group) | load (from the output DRAM), nop op, unflatten (ARM intrinsic) |

With 16+ input plane slots per core and a pad or pool first, the input load is split into 4 chunks of ceil(ppc / 4)
slots, each placed just before the NFU ops of its slots (V: ci124, ci128, ci136 (5,5,5,2), ci160, ci192, ci256,
ci128s2, pfirst128; not when a reformat or the conv reads the input: ci128s2h20, ci128v, ci128k1).

### Events (V)

DMA n of the network (loads and stores, in group then IR order; stores of a group after its loads) signals event
1 << (n mod 16) (= the io descriptor's trig_flag; co224..co255 wrap). Stores: 1 or 2 (section 7), all triggered
and waited together.

### Emission of a group

For each compute element i (pad, pool, reformat, conv, nop op):
1. `r9 = i` (the IR index; profiling) — not for the unflatten element.
2. If NPUC's event schedule gives it a wait list and no ev2set line is open: `ev2set = mask << 16`, the mask being
   the events of every wait up to the next close (a trigger to a DMA, a pool reset; 7b "The ev2set lines").
3. Its wait, when it has one: `r1 = m; nop x4; waiti3(ir6, r1, m); nop x4; ev2 = m; nop x4` (m = its wait list,
   the DMAs whose representative successor it is, 7b).
4. Its register writes, the engine kick, then `nop x4` (none when it is the group's last element, V: pmax, two, cpc),
   `waiti3(ir0, r1, done)` and the interrupt registers reset.

A store: `ev2set = m << 16; nop x4; r1 = m; nop x4; waiti3(ir6, r1, m); nop x4; ev2 = m` (no trailing nops).
End of group: r1..r15 = 0, then `r10 = group index`; the last group adds `swi = 0x8000`.
The nop op: `r9 = i; nop x5` then its load trigger/wait. The unflatten: `swi = 0x10000 | g; nop; r1 = 0xffff0000 | g;
nop x4; wait3(r15, r1, 0xffff0000 | g); nop x4; r15 = 0; r1 = 0; r3 = 0` (g = its group index).

`r1` before the engine kick is NPUC's "must be more than 4 commands before 'wait' command": its position is fixed per
op type (pad/pool: first write; reformat: after 0x4016030c; conv: before 0x40120300).

## 4. Channel distribution (V: ci1..ci256, co1..co255 masks, per-core registers, bias/shift order)

Channels go in planes of 2 (interleaved). planes = ceil(C / 2), ppc = ceil(planes / 4) planes per core, cores =
ceil(planes / ppc), core mask = (1 << cores) - 1; core k holds planes [k ppc, (k + 1) ppc). Same rule for inputs and
outputs (= grelu.md's chunk = 2 ceil(C / 8)). A conv runs in passes = ceil(ppc_out / 2): pass p computes plane slots
2p, 2p + 1 of every output core; slot (core k, slot j, member c) holds channel 2 (k ppc_out + 2p + j) + c when it
exists. The OFM chunk holds ceil(ceil(Cout / 16) x 16 / 8) planes per core ("rounded to 16 ... due to HW bug").
A feature-map plane is cells(h, w) = ceil(h / 4) x ceil(w / 8) cells of 64 bytes; SRAM addresses are in 64-byte units.

## 5. SRAM allocation (G: fitted on every allocation logged by the sweep, 0 misses in 293 sweep + 77 random networks)

Node order: input, then per layer [pad] [reformat] conv | pool, then unflatten. Chunks (bytes per core):

| Chunk | Size | Live (node order, inclusive) |
|---|---|---|
| INPUT | ppc_in x cells(H, W) x 64 | at its first consumer only |
| PAD | ppc_in x cells(Hp, Wp) x 64 | pad .. conv |
| REFORMAT | ppc_in x 4 x cells(ceil(Hp/2), ceil(Wp/2)) x 64 | reformat .. conv |
| CONV (OFM) | OFM planes x cells(Ho, Wo) x 64 | conv .. next consumer |
| WEIGHT | passes x 2 x R x 64 (R rows per batch, weights.py) | first op of the layer .. conv |
| PARTIAL_SUM (weight sub-batches > 1) | 2 x 7 x cells(MTC input) x 64 | conv only; placed after the layer's other chunks |
| POOL | ppc_out x cells(Ho, Wo) x 64 | pool .. next consumer |

Placement: batch by batch (the input at 0, then each layer), biggest first within a batch (equal sizes: the later
chunk first), each at the lowest 64-aligned offset clear of every placed chunk whose live range overlaps.

## 6. Registers

### Pad (NFU pooling unit; done interrupt 4; per plane slot and width piece)

| Register | Value | Evidence |
|---|---|---|
| CMDQ r1 | 4 | all |
| INTCOM 0x40100028, 0x40100024 | core mask (after done: 0x24 = 0, 0x28 = mask) | ci* |
| NFUBR 0x40160800 / 0x40160228 | 5 / 0x10000, 0xa / 0x10000, then 0x800 = core mask (G: unit select + init) | constant |
| NFUBR 0x40160208 | pads, nibbles: bottom, top << 4, right << 8, left << 12 (left on the first piece only, right on the last) | k*, s2same (0,1,0,1), w* |
| NFUBR 0x40160214 | bit 26 = u8 (quant 1); (2 x skip) << 20; (2 x input width from the piece's first cell) << 8; H | hw*, w116..w250 |
| NFUBR 0x40160218 (written twice) | bit 26 = u8; (2 x piece output width) << 8; Hp | hw*, k* |
| NFUBR 0x4016022c | quant 1: 0x10000 \| input zero point (the pad value); quant 3: 0 | inzp, q3 |
| NFUBR 0x4016020c | ceil(W / 8) << 16 \| (input addr + plane offset + first cell) | hw*, ci17 |
| NFUBR 0x40160210 | ceil(Wp / 8) << 16 \| (PAD addr + plane offset + piece start / 8) | hw*, ci17 |
| NFUBR 0x40160200 | 0x30000 (pad mode; 0 for a pool) | all |
| NFUBR 0x401602fc | core mask (kick) | all |

### Pool (same unit; per plane slot and width piece; V: pmax*, pavg*, w*pool, w240max3, w200avg3)

As the pad with: 0x208 = the pool's pads (same nibbles and piece rule); 0x214 input rows hu = min(H, (Ho - 1) sh +
kh - pt - pb) and width: first piece = its input end, last = end - first cell x 8, middle = end - start (G, 3
networks); 0x218 = output piece; **0x40160204** = type << 16 (max 0, avg 2) | kw << 12 | kh << 8 | sw << 4 | sh
(G: only square pools were compiled, so the kh/kw and sh/sw order is unproven); 0x20c/0x210 from the input and POOL
chunks; 0x200 = 0.

### NFU width pieces (V: w104..w250, w113k5, w*pool, w*s2, w120ci16, w130ci16s2)

A pad or pool command covers at most 116 columns (G for the exact bound: 116 out of a pad does not split, 118
does): n = ceil(max(W_in, W_out) / 116) pieces of round_up(ceil(W_out / n), 8) output columns, each reading from the
cell of its first input column (skip = the columns before it in that cell). A reformat command reads at most 128
input columns: pieces of 128. Pieces are separate IR elements, slot-major (all pieces of slot 0, then slot 1...).

### Reformat (stride 2: space to depth; NFU; done interrupt 2; V: s2*, w*s2, ci128s2)

Two commands per IR element (halves 1, 2), each: INTCOM 0x40100018, 0x40100014 = mask; 0x800/0x310 init as the
pad (0x310 instead of 0x228); **0x4016030c** = half << 30 | ceil(w / 8) << 24 | ceil(Hp / 4) << 16 | w << 8 | Hp
(w = the piece's input columns); `r1 = 2`; 0x40160304 = ceil(Wp / 8) << 16 | (src + x0 / 8); 0x40160308 =
ceil(Wr / 8) << 16 | (REFORMAT + 4 slot cells + x0 / 16); 0x40160300 = cells(Hr, Wr) << 16; 0x40160314 = quant 1:
0x10000 | zp, else 0; 0x40160318 = 0; 0x401603fc = mask (kick); after done 0x14 = 0, 0x18 = mask. Hr, Wr =
ceil(Hp / 2), ceil(Wp / 2); each input slot gives 4 phase planes. The conv then sees 4x the channels and a
ceil(k / 2) kernel.

### Conv (MTC; done interrupt 0x100; one per output pass)

| Register | Value | Evidence |
|---|---|---|
| MTC 0x40120004 | 0x420 | constant |
| MTC 0x40120040 | quant 1: 0x07000000; quant 3: 0x300 | q3 |
| MTC 0x401201b0 / 0x401201b4 (quant 1 only) | in zp << 16 \| w zp (128) / out zp | inzp, ozp |
| MTC 0x40120008, 0x4012019c | 0 | constant |
| MTC 0x40120780 | output cells (10 bits: NPUC masks, t192 0x480 -> 0x080) | hw*, t192, t200 |
| MTC 0x401200a0 | ceil(Win / 8) << 16 \| input addr (PAD, REFORMAT or the input chunk) | hw*, k1 |
| MTC 0x40120180 | input cells | hw*, k* |
| MTC 0x40120240 + 4k (k < input cores) | input channels per core (2 ppc_in; 8 ppc_in for stride 2) | ci*, s2* |
| MTC 0x40120044 | input core mask | ci* |
| MTC 0x401201a0 / 0x40120788 | Hin << 16 \| Win (padded or phase plane) / Ho << 16 \| Wo | hw*, valid |
| MTC 0x40120480 | kh << 16 \| kw (stride 2: ceil(k / 2)) | k3x1, k1x3, s2* |
| MTC 0x40120488 | (ceil((kw - 1) / 8) + 1) << 8 \| (ceil((kh - 1) / 4) + 1): cell rows/cols the kernel spans | k1..k13 |
| MTC 0x40120784 | number of output passes | co17..co255 |
| MTC 0x40120940 | quant 1 bit 28; GReLU mode 3 << 8; low byte: weight sub-batch count if the output is more than one cell, else 1 (G) | q3, ci64k7, ci256, ci256h6 |
| MTC 0x40120400..0x4012040c, 0x40120440..0x4012045c | 0 | constant |
| CMDQ r1 | 0x100 | all |
| MTC 0x40120300 + 8k + 4j (k < input cores, j = 0, 1) | R << 16 \| (WEIGHT + (2p + j) R): weight batch j of pass p (batch j = output cores 2j, 2j + 1) | k5, wzero, co17, ci17 |
| MTC 0x40120580 / 0x40120048 / 0x4012004c | 0xffffff / 0xffff / 0xffff | constant |
| CMDQ 0x401e0050 = 1, GReLU block, = 0 | grelu.py (22 writes per pass) | grelu.md |
| MTC 0x40120200 | sub-batches << 16 \| channels per sub-batch (weights.py) | ci128s2, ci256, ci64k7 |
| MTC 0x401200b0 | ceil(Wo / 8) << 16 | hw* |
| MTC 0x40120700 + 8k + 4j (all 4 cores) | OFM addr + (2p + j) x output cells | hw8, k7, co17 |
| MTC 0x40120900 + 8k + 4j | per channel slot (lo: member 0, hi: member 1) shift << 11 \| shift << 6 if the slot holds a channel | sh8..sh12, co1..co7 |
| MTC 0x40120800 + 32k + 8(2j + c) | the slot's int32 bias as int64 (lo, hi = sign) | all |
| MTC 0x40120590..0x4012059c | 0; with sub-batches: 0xf << 4k | ci64k7... |
| MTC 0x40120540 (sub-batches only, before 0x590) | 7 ceil(Win / 8) << 16 \| 7 x input cells (G: the 7) | ci64k7, ci36k9, ci256 |
| MTC 0x40120500 + 4k (sub-batches only, after 0x59c) | (PS + 7 x input cells) << 16 \| PS (PS = PARTIAL_SUM addr) | same |
| INTCOM 0x40100008, 0x40100004 = 0x11; MTC 0x40120000 = 0x11 (kick) | after done: 0x04 = 0, 0x08 = 0x11 | all |

The shift is NPUC's (doublings + 8, ScaleONFiller::setScale), the slope in the GReLU block is round-half-away
(grelu.npuc_slope; old lowered.json slopes may be 1 low).

## 7. Odd sizes and uneven channel counts (V: the random set r004..r068 and sweep s12..s15)

NPUC's own feature-map sizes. stream.py tracks them layer by layer instead of trusting lowered.json's: a conv's
output is (in + pads - k) // stride + 1 on NPUC's input; a pool's the same, except **when only the width's padding
is negative** (VALID on an odd width with an even-covered height: (Ho - 1) s + k - pads - H >= 0 > the same for W):
NPUC logs "Padding in Pooling has negative value!" and computes one more output column with a right pad
(the smaller of the negated padding W - (Wo - 1) s - k and the extra window's overhang Wo s + k - W, Wo before the
widening: 1 for k2 s2, 1 for k3 s3 on 8, 10 and 52 columns: rnd057, rnd067, r2-139): 0x40160208 gets it, 0x40160214 keeps the full width, 0x40160218 the
wider output, the POOL chunk is sized for it, and every later layer computes on the wider map (the store crops; V:
r004, r017, r029, r035, r045, r055, r063, r065 (two widened pools in a row), po_e_o1..3, po_k3_h15w16, po_chain,
po_conv, po_convv, po_s2). When the height's padding is negative too, or only the height's, NPUC crops instead
(r006, r013, r025, r007, r068, po_o_o1/2, po_o_e, po_k3_h16w15, pmax3).

Written plane slots. A tensor has "all slots written" when it was loaded from DRAM or is a conv output whose planes
fill every core (planes = cores x ppc). Otherwise (a conv output with a short last core) the last core has only the
slots of the conv's 2-slot passes: 2 floor(ppc / 2), whatever planes it really holds. Pads, pools and reformats run
each plane slot s on the cores that have it written (the core mask in INTCOM 0x24/0x28 or 0x14/0x18, NFUBR 0x800
and the kick) and keep their input's written slots (V, last core's slots: c10p, c14p, r015 (ppc 2: 2), c20p, c22p,
r013 (ppc 3: 2), c26p, c28p (ppc 4: 4), c34p, c38p (ppc 5: 4), c50p (ppc 7: 6); full after a loaded input:
ci9, pfirst20s, pp20, u20s2first; pads c20pad, c34pad, c42pad; reformats r038, r061, c20s2, c38s2).

Stores: one per distinct per-core plane count (NPUC merges equal per-core DMAs): a conv output with a short last
core takes 2 (co9); a pool output 2 only when its input lacks slots and ppc is odd (r013, r068: 20 channels, ppc 3),
else 1 (r015, r067: 9 channels, ppc 2; c34p...).

## 7b. Tiling, weight rings, allocation fallbacks (V: tile sweep `stream_variants_tile.txt`, 73 networks; the
frame part's tiled compiles `$NCPGEN_WORK/sweep/frame/fs/{tiled,t2..t5,dim}`, 153)

**Who decides.** stream.py takes every tiling decision from `frame.plan(low)`
(NPUC's own policy since frame.md 6.10: strata, tiles, back-propagated per-layer ranges and pads, SRAM chunks, a tiled
conv's ring and slot step) so both halves of the NCP agree; the two old refusals (a stratum with a halo pool, a conv
with weight sub-batches) are gone (V: final-chains f002, f012, f059; tiling2 a*, c*, d*, g*: 50 networks; the tile
sweep's w300pool, tch5, r_c256h40, c512, r_c384, r_c448, t-32x32-* stay exact through `frame_stratum`). The old
planner of its own (`Plan.tiled_conv`, `Plan.segment`, which ran only without frame.py) is gone: a layer frame.plan does
not tile but the untiled layout cannot hold raises NeedsTiling. Checks and overrides of ours (V, below): a tile that
reaches the end of the stratum's output computes every layer to its end (`tile_to_end`; frame.plan's tiles already
are, and a tile it would extend is refused), and the read-back tiling.

**A tile's own geometry** (V: tiling2 e03, e04: 3x3 conv, 2x2 pools on 54 x 62 -> 27 x 31 -> 13 x 15): a tile whose
output reaches the stratum output's last row (column) extends every layer's output to that layer's last row (column)
and its input to the tensor's end with the layer's own trailing pad; an untiled axis is one such tile. NPUC's layout
adds each layer's rounding left-over (0 <= d < s, frame.md 6.10) to the last tile, so a layer whose output the next one
does not read entirely (a VALID pool on an odd size) is still computed whole: e04's second pool reads 30 of 31 columns
but the first pool computes 31, the stride-2 conv pads 128 columns to 129. The larger tiles can change the stratum's
chunks (e04's PAD chunk: 17 cells wide, not 16): stream.py re-runs frame.py's `_stratum_ops` / `_stratum_alloc` on
them. Every tile of a padding layer pads, also a tile with no pad of its own (0x208 = 0; V: f002, c02..c11: a stride-2
conv padded (0, 1, 0, 1) in 6 x 2 tiles).

**When NPUC tiles** (V): a padded input height > 255, a conv whose MTC input width (padded width / stride) > 255,
or a layer needing more than 256 KiB alone (`ts208` 264896 B, `ts128c32k1` 262656 B; `t200`, `ts160c16` whole);
pools are not tiled for their size (`w300pool`). Tiles: columns round_up(ceil(W / n), 8), rows round_up(ceil(H / n),
4), the fewest that fit (frame.py's search, which also merges chains from the tail).

**A tile** = one NCP group: load the tile's input region (start aligned down to a cell; one DMA for all cores, or one
per core with several planes per core), a **TRIM** when some tile starts inside a cell (NFU copy, 0x40160200 =
0x20100: 0x214 = u8 | (2 skip_w) << 20 | skip_h << 18 | (2 x width from the piece's cell) << 8 | rows from the
piece's cell; 0x218 = the piece's output; 0x208 = 0; 0x22c = 0x10000 in quant 1), then per layer its pad (the tile's
own edges only), reformat, conv (weights loaded in tile 0 and kept; a tiled ring, R < passes, reloads every pass
in every tile into slot p mod R, frame.plan's `wstep` apart, the slot's load waiting for the conv that last read it;
partial sums at frame.plan's PARTIAL_SUM chunk) or pool, then the store(s) (one, or one per core with several output
planes per core). Weights of a stratum: the first chunk at 0, the second at the top of the SRAM (frame.py's
allocation). The read-back (unflatten) follows NPUC's policy on the output node (getBestProcessingParams; frame.md
6.10, frame.py `_unflatten_tiles`, = `npuc_policy(low)["readback"]`; V on every output node's log, 2080 compiles):
whole when its input (the output's planes, 16-rounded channels when a conv makes the output) and its output, per core
in cells, fit 256 KiB together; else the fewest tileable row counts n >= 2 (n tiles of round_up(ceil(H / n), 4) rows,
the last the rest, all non-empty) whose first tile fits twice in 256 KiB (V: ts208 one tile; ts216, t2d-300x300 two;
rb120w72, rb136, f034 three; tiling3 tr-2nd four, tr-320 six). Each row tile is its own group (loads per core with
several planes per core). stream's `unflatten_tiles` is the same rule (the earlier 270 KiB fit gave the same counts
on every set).

**NFU command pieces** (all NFU ops, tiled or not): columns n = ceil(input width / 112), rows n = ceil(input rows /
248), pieces of round_up(ceil(outputs / n), 8 columns / 4 rows), rows outer; each piece reads from the cell holding
its first input; leading pads on the first piece, trailing on the last (V: w104..w250, the 225-wide stems 2 x 120,
dw440 3 x 80, dh500 2 row pieces; G for 248 and the row order). A reformat reads at most 128 rows and 128 columns
per command (the stems).

**Weight ring** (untiled, V: c256 6 of 16 passes, r_c256h24 8, r_c320h24 7 of 20, r_c512h16 7 of 32, r_c512h24 2;
all passes: r_c256h16, r_c192h32, r_c256co128, co512): all passes when the conv node fits (its input, output,
partial sums and every pass's weights within 256 KiB), else R slots, R_max = (256 KiB - those feature maps) // slot,
R = ceil(passes / ceil(passes / R_max)). Pass p uses slot p mod R; its load is triggered only once the conv that last
used the slot has been emitted. Tiled (our planner; G, 3 cases: r_c384 6 of 24, r_c448 4 of 28, c512 2 of 32): R =
the largest divisor of passes fitting beside the tile; when every pass's weights could fit, more row tiles instead
(r_c256h40). A ring reloads every pass in every tile. `plan_dmas()` gives each weight load's SRAM slot offset (996 of
996 weight loads match the oracle's io dst offsets).

**Short rings**: a weight ring of 4 slots or fewer (and fewer than the passes) lives on its conv only, so it may sit
over the layer's pad and reformat chunks (frame.md 6.7; V: f043, wl-512-h21); the per-layer tiling probe uses it too
(f043 fits whole, NPUC does not tile it).

**The first round of a short ring** (a ring of 4 slots or fewer that lives on its conv only and sits over chunks the
layer's NFU ops read; the sweep `stream_variants_ring3.txt`, 271 networks in $NCPGEN_WORK/sweep/tiling3: ra (stride 1,
5x5 over a 1x1 conv's output), rc (stride 1, 3x3, 2..3 sub-batches, the same), rb (stride 2 VALID over a 3x3 conv's
output, the reformats read it), rd (stride 2 padded, over its PAD or over the conv output its pads read); R 2..4, 2 to
6 sub-batches). NPUC triggers the loads of the first round at the first conv up to a cut c, the rest with the c-th
load's own conv, together with the next round's loads whose slot is free (W0..W_{c-1} at conv 0, W_c.. at conv c).
What we emit: the full round at conv 0 (c = R), except for a ring over the layer's PAD (stride 2), where c = 1: its
loads wait for the last reformat (an input loaded with them goes alone: sr-1l 0xf), W1.. for the first conv. The
outcome per family (c = R / c < R): rc 38 / 0, ra 87 / 5 (and 12 rings of 2 whose kernel, over 4 MiB, loads core by
core: c = 1, then pairs W1 + W2, W3 + W4 ...), rb 40 / 29, rd 40 / 32 (c = 1: 19, 2: 12, 3: 1); f043, sr-* and
sx-p100/w28 (over the PAD) all c = 1. So the ring3 streams: rc 38/38, ra 75/92, rb 40/69, rd 28/72 (40/72 with c = R
for the PAD rings too, which loses f043 and 8 of the sr/sx cases: neither rule is NPUC's). **Solved (V)**: c is no
rule of its own; it falls out of NPUC's event schedule run on the memory profile (below: graph, representatives,
event pool, ev2set lines), which stream.py now generates: ring3 271/271, tiling3 302/302 (sx-c2, sx-c2v included).

What decides c, read from NPUC's code (libnpuc_backend; decompiles local in $NCPGEN_WORK/n3/ghidra/decomp/targets-batch
and targets-trigger, target lists in the stream part's scratch):
- **The ev2set lines are retroactive (V, code: IsaFunc2020::addWaitFromCPU).** A wait for DMA events emits
  `ev2set = mask << 16` only when the running trigger mask is empty; otherwise it ORs its mask into the running mask
  and rewrites the ev2set line it emitted earlier. The running mask is cleared by PostProcessing of an element whose
  IRElement byte 0x77 is set. So one ev2set stands for a run of waits, placed at the first of them: our "batch
  triggered at its first need, the rest at the next need".
- **The io batch ids (V, code: BlobTransfer::SetBatchTransferInGroup).** Runs of consecutive ncp_blob_load (or
  ncp_blob_store) IR elements each get a batch id; a run whose head has no incoming event (its wait list, IRElement
  0x38..0x40, empty or -1) takes the previous run's id (`transfer` 0), one with an incoming event a new id; after an
  element flagged 0x73 (a DMA with no consumer, a store followed by a load) the next two runs get new ids.
- **The incoming events (V, code: EventSchedule; the first reading, made exact in the bullets below).** BuildGraphByDataDeps builds the dependency graph from the memory
  profile (CODEGEN_FULL_DEP_SCAN is 0 in these compiles: a load records one dependency, the last element that
  touched its region); UpdateRepresentative gives each element its latest predecessor and earliest successor of the
  other kind (repPrecede, repSucceed; a DMA's predecessor of any kind); ReorganizeRepresent moves each compute -> DMA
  link to the last compute element before the DMA (the computes in between point to it; links of the DMAs they fed
  are dropped; at most 16 DMAs apart); UpdateEventWaitFieldDataDeps gives every link an event: DMA -> compute from the
  16-event pool (round robin from a network-wide cursor, released where the consumer's representative chain reaches
  the next DMA, the group's end when none; when all 16 are taken the mask resets and the consumer of the DMA that
  filled it gets 0x77), compute -> DMA from a one-event pool (two live at once would abort NPUC: "event 2 cpu must be
  one"). The compute -> DMA links are the loads' incoming events, so the loads that start a batch.
- **The dependency graph (V, code: BuildGraphByDataDeps, MemoryManager).** A per-group list of MemObjs (start, end
  inclusive, reference count, owner element). An element's reads: `ref` finds every MemObj whose range contains the
  read's start or its end (end - 1 tolerated: a MemObj inside a longer read is not found), counts a reference on each,
  and returns the last one in the list; its owner -> element is an edge, the element becomes its owner; with none the
  read is pushed as a MemObj. A load's writes are `ref`-ed the same way first. Every write is then `push`-ed: each
  MemObj it overlaps (same test) must have no reference left, gives the edges owner -> writer (or, when the owner has
  successors, each successor -> writer: a write after reads) and is removed; the write is appended, owned by the
  writer. The log prints exactly these refs ("Refers ... of <owner>:<index>"), not the push edges.
- **The representatives, exactly (V, code + disassembly of ReorganizeRepresent; `stream.npuc_reps`; the decompile's
  two `joined_r0x001b0dd4` labels hide which branches reset the DMA count: macOS `objdump -d --start-address=0xb0a60
  --stop-address=0xb1184` on the local libnpuc_backend.so shows it, Ghidra addresses being file addresses + 0x100000;
  w27 is the count).** A DMA's
  representative successor is its earliest successor of another type (a CMDQ or ARM element; a store): no DMA -> load
  link (per-core loads feeding one NFU op each wait on their own: f002 g1..g15). The DMA count of the walk resets after
  every CMDQ element that got a DMA link (also after the walk; the old replay carried it). The branch the old replay
  lacked: a CMDQ element with no successor scans forward for the next CMDQ element that has one, but when the 16th DMA
  of the count comes first, the latest CMDQ element before that DMA with no successor of its own (or the element
  itself) links to it (a forced compute -> DMA link) and the element points to it. So a run of DMAs never holds more
  than 15 events: the first conv with 16+ passes (ra/rc g0: W14 starts a batch at conv 13, events 0..14 released there,
  no pool exhaustion), f038 g1, the spill groups, rc-120-500, ws-128-512.
- **The event pool (V, code: UpdateEventWaitFieldDataDeps, AllocEventId, ReleaseResource, InsertResourceRelease-
  Sequence, FindNextReleasePosition; `stream.npuc_schedule`).** Both masks reset at each group, the 16-pool cursor is
  network-wide. In IR order: release the events whose position is this element; skip unflatten/permute/flatten; an
  element whose representative successor has another type takes an event: to a CMDQ element from the 16-pool
  (cursor .. 15, then 0 .. cursor - 1), released where the consumer's representative chain (followed while CMDQ)
  reaches a DMA, the group's end when none; to a DMA or an ARM element from the one-event pool ("event 2 cpu must be
  one": always event 0, released at its target). When the 16-pool is full ("AllocEventId ir(n):... cannot allocate
  event" in the log: f002 ir 36) the mask resets, the DMA gets flag 75, and the element recorded when the pool last
  filled (the consumer of that DMA) gets flag 77. The event goes into the target's wait list, except a store followed
  by a load (flag 73). A DMA with no successor takes a pool event released at the next CMDQ element (flag 73). The
  cursor never skips a busy event on any set, so a DMA's event is still n mod 16 (`Plan.event`).
- **The ev2set lines (V, code: IsaFunc2020::addWaitFromCPU, addSwiTrigger, PostProcessing; `stream.npuc_lines`).** A
  CMDQ element with a wait list waits for its events (r1, waiti3 ir6, ev2); it writes a new `ev2set` line when the
  running trigger mask is empty, else ORs its events into the open line (rewritten). The running mask is cleared after
  an element that signals a DMA (addSwiTrigger: a compute -> DMA link) and after an element flagged 77.
- **The io batch ids (V, code: SetBatchTransferInGroup; `stream.npuc_batch_ids`).** Per run of DMAs of one name: a
  new id when the head has a wait, the previous run's otherwise; a run after one ending in flag 73 and the run after
  it get new ids.
- **Replayed (V)**: `stream.py events <dir>...` runs graph, representatives, pool and lines on NPUC's logged memory
  profile and compares every ev2set line (position and mask) and every wait with NPUC's stream: **every CMDQ group
  with a DMA on every set, 7891 of 7891** (tiling3, final-chains, frame/fs/{wlive, wstream, spill, ringc}: 1345 of
  1345; the old batch-start replay 1187 of 1396). ARM groups (eltwise) keep the firmware's events (stream.elt_group's
  flags); after one the cursor continues at stream.py's next flag (graph sets: 20 + 18 groups otherwise off by those).
- **The memory profile, generated (V; `stream.npuc_profile`)**: every element of every non-ARM group of every set
  (chains, tiles, graphs, the stream sweep: 137198 elements) matches NPUC's logged profile element by element (the
  read-back sized by NPUC's own output: wq-w's extra pool column; a ReLU's dummy conv reads its pass's slots: g502),
  and the ev2set lines and waits of the schedule run on it equal NPUC's on all 7891 groups with a DMA. Byte
  ranges per IDP (core k at k x 0x40000), as NPUC logs them (some are not the real accesses: they are what its graph
  sees):
  - a load writes one range per IDP of its cores, its SRAM offset + k x 0x40000, the bytes per core (an input: planes
    per core x cells x 64; weights: the slot, 2 x rows x 64, on the conv's input cores, a core-by-core weight load on
    its own core); a **per-core load writes plane j at IDP core + j** (f002 g1, ts128-80: NPUC steps planes by the
    IDP span); an input loaded in parts (one DMA per distinct per-core plane count) covers the cores of its count; the
    read-back's load: per IDP (planes per core x cells) when one plane per core or 4 cores of full planes (C = 8 ppc),
    else all planes in one range at 0;
  - a store reads per IDP its planes of each core, one element per distinct per-core count (pass-by-pass and cut
    stores: the pass's or chunk's planes); a tile's per-core stores: one element per core, one range per plane;
  - pad, pool, trim (one NFU piece): one read per core of 2 x rows x columns bytes from the piece's first cell (rows,
    columns: from the cell to the piece's input end; a pool's last piece to the input's end), and its output rows
    (the piece's width in cells, 4-line rows) written back to back from the piece's first output cell (not at the
    plane's row stride);
  - reformat: its input rows (piece width, at the plane's row stride) and, per phase plane, its output rows (half
    the piece width, at the phase plane's row stride);
  - conv: every row of every MTC input plane (kin / 2 planes, ceil(hin / 4) rows) on each input core, then 8 weight
    halves: k x 0x40000 + the slot + (k odd) x rows x 64, rows x 64 bytes, for k < 2 x the input cores, empty ranges
    at k x 0x40000 beyond; then its pass's two OFM plane slots, row by row, on the cores of the 16-rounded output;
  - the nop reads and writes each IDP whole; the unflatten reads 64 bytes a plane per IDP when its channels fill
    the cores evenly (C = 2 x cores x ppc, one plane per core or 4 cores), else C x 32 bytes at 0.
- **Generated batches (V)**: `emit_group` runs `npuc_schedule` and `npuc_lines` on the generated profile
  (`group_schedule`; its DMA events must equal the flags, else the group is refused) and writes its ev2set lines and
  waits; `npuc_batch_ids` gives `plan_dmas` its bids (no special case for 64 loads or more). The
  fitted batch rules below (63 units, 15 weights, `after`, the short-ring cut) are gone from stream.py: on every
  compiled group (2802 compiles, about 10000 groups) the schedule stands, and a group with loads the profile does not
  model raises Unsupported (ARM groups need no CMDQ trigger: the firmware triggers their DMAs).

**Allocation fallbacks** (V: r_c256co128): within a batch NPUC tries "bigger first", then "backward" (reversed,
partial sums still last), then each next permutation of the backward order, keeping the first order where no chunk
ends past 256 KiB.

**Trigger batches** (V: the trigger sweep `stream_variants_trig.txt`, 81 networks around Cout 210..512, k1 / padded
k3 / valid, 1 or 4 input loads, rings or not; ws-*, r_c256h40, t-32x32-256-512, co240/co255, r3-074). The DMA batches
(the io bid) come first: from the first untriggered load, in IR order, a batch takes loads up to a load whose ring slot
a conv not yet emitted still reads, or up to a store (a batch is a run of loads), and at most
- 15 weight loads plus every input load before them, in a group with a weight ring (R < passes) whose input an NFU op
  reads first, or in a tile of a **one-layer** stratum with a TRIM (ws-64-512k5, ws-167/198/256-512, r_c256h40: 19
  DMAs; f002, c02, c04..c08, c11, th-256-*, tiling3 tr-s1, tr-s2, tr-c8, tr-200, tr-240); a tile of a chain (conv +
  pool, conv + conv) counts its input loads like the next rule (V: f076, tiling2 a04, a05, a07, a10, c00, c01, c03,
  c09, c10; tiling3 tr-s1p, tr-s1a, tr-s1c, tr-s2p, tr-c8p, tr-200p, tr-240p, tr-320p: 0x7fff = the input loads and
  11 weights, then the rest; the same first conv alone, 16 passes and 4 input loads, takes 0xffff = 4 + 12, then 3);
- else 63 per-core units, each DMA 4, a weight load fewer when its pass leaves cores without planes: so 15 loads, 16
  when the last pass is short (Cout 225..236 except 232: its last core has fewer passes; k1-238: not; 240: not), and
  for a conv whose output stays in SRAM the planes count rounded to 16 channels (k1-228c, k1-228p: 15).

A batch is triggered (one ev2set) up to its first load whose event is still in flight; the rest of it goes at the next
need, before any new batch (ws-167-512: 0xffff, then 0x7 at conv 12). The bids follow: `plan_dmas` gives each DMA its
batch (= the oracle's io bid on 989 of 991 compiles of the tiled, trigger, stream, cut, rb, final-chains, tiling2 and
tiling3 sets; the 2: sx-c2, sx-c2v); frame.py's `batches_from_streams`, which reads them back from the masks, cannot
tell a full mask that ends a batch of 15 weights from one a batch continues (tiling3 tr-c8: 1 input + 15 weights =
0xffff, then W15 alone is a new batch).
Loads into SRAM a still-live chunk overlaps wait for its last reader: a weight chunk placed over the input waits for
the last pad (wl-512-h21). Pass-by-pass stores (below) end every batch at one load.

**From frame.plan (frame.md 6.6-6.7)**: stream.py takes an untiled network's chunk offsets, its rings (R, slot step:
16 x ceil(per-IDP size / Cout) rounded to 64 B when Cout is not a multiple of 16; a ring of 4 slots or fewer lives on
its conv only) and NPUC's trim-and-spill copies from `frame.plan(low)` (`Plan.frame_single`). **Copy group** (V:
ws-191-256x2, r3-076/077/096/111, frame/fs/spill, s8 sp-*, tiling2 b03, b05): when no order fits a layer's chunks
NPUC copies its input to a new chunk; its own group before the layer's: per plane slot of the input one NFU TRIM-mode
copy per NFU command piece (the usual pieces: b03's 138 columns in 72 + 66, b05's 128 in 64 + 64; each piece its own
IR element), H x W in and out, all cores (cm 0xf), no DMA, no event; the layer then reads the copy. **Pass-by-pass stores** (V: frame/fs/ringc rc-167-488..510, rc-200-500): the last conv with a ring,
Cout not a multiple of 16 and no sub-batches stores each pass's two plane slots right after its conv, one DMA per
distinct per-core count of them, each its own IR element, followed by 4 nops (the wait's trailing nops) when more
ops follow.

**Remaining**: none on the compiled sets: the trigger batches come from the generated event schedule (above), the
rules of this paragraph and the previous one stay only as the record of what was fitted before.

## 7c. Graph ops: eltwise add, concat, standalone ReLU (V: `$NCPGEN_WORK/sweep/graph*`, compiled with the harness run
directory `$NCPGEN_WORK/n3/root-current`, npuc-net 2; `ncpgen/stream_graphs.py` writes the npu-net/1 variants)

Rates: graph sweep 98/98 exact (`stream_graphs.py compile`), random graphs (`stream_graphs.py random <dir> <seed> <n>`)
seeds 1..50: 47/47; unseen seeds 100..179: 62/66 (14 more crash NPUC itself, one refused by npunet); the ops-check nets
of `oracle/ops/mknets.py` 12/12 (after setting quant-3 zero points to 128: see 9). `plan_dmas` = the
oracle's io (flag, bid, group, direction, SRAM offset) on 208 of 211.

**Order (V: NODE ORDER of every graph compile of the sweeps that logs one, 278; the zoo's resnet8 and its variants).**
NPUC numbers its graph's nodes breadth first from the network input, each tensor's readers in layer-list order
(`run/ir_graph_init.txt`), then runs, of the layers whose inputs are all done, the one with the highest number
(`exec_order`). So of two branches from one tensor the one listed last runs first, and an eltwise's input order does
not matter (resnet8: a stage's 1x1 shortcut conv, listed after the block's two 3x3 convs, runs before them; listed
before them, after them; the add's inputs swapped, the same NCP). One NCP group per layer, none for a concat. Atoms
(NPUC nodes, the order numbers of live ranges): pad, reformat, conv; pool; ReLU: 1; eltwise: 2 (a "pre" node holding
CU_BIN and the shift chunks, then the eltwise); concat: 1; then the unflatten.

**Where tensors live.** Every tensor stays in SRAM until its last reader (live range to that reader's first atom),
except:
- the network output (stored; the unflatten loads it);
- the operands of an eltwise that reads a tensor in DRAM (the network input, a concat's buffer): the other operand's
  producer stores it, and the eltwise loads both back (to their own chunks: the INPUT chunk, the producer's chunk);
- a concat in DRAM: each producer stores into its buffer (channel offset in the descriptor), its reader loads it;
- the root of a concat's branches ("Input sharing"): when all branches start at one computed tensor made by a
  single-node layer (pool, ReLU, conv without pad or stride), it is stored, its first reader loads it back and the
  others share it (V: g005, g008, g009, g030); after a padded conv or an eltwise it is shared in SRAM directly (V:
  g003, g012, g025, g036). Rooted at the network input, the input is shared (one INPUT chunk from 0 to its last
  reader).

**Reloads.** A reader of a DRAM tensor (input, concat buffer) loads it unless the input is shared and an earlier
branch loaded it, or the layer run just before loaded the same tensor the same way: per core (an NFU op reads it first
and a core holds several planes, or an odd channel count) vs whole (V: add2, in-two: no reload; add-branches, g020,
g025, g175: reload). A loaded intermediate comes in one DMA per distinct per-core plane count when loaded whole, in one
when per core (V: g008, g030, g047); the network input always in one.

**Concat in SRAM (V: cat16, cat16-3, cat-32-16, g021, g118).** When every input is a pool or a conv with a multiple of
16 channels, the concat is not the output, no eltwise reads it and it does not fork into branches an eltwise joins (V:
g029, g144 in DRAM): one CONCAT_STORAGE chunk (planes per core = the sum of the inputs'), each producer writing its
planes at a slot offset (sum of the earlier inputs' ppc); no group, no DMA. Each core then holds [input 1's planes,
input 2's planes, ...]: a permuted channel order for every reader. A pool of it keeps the segments, and storing it
takes one DMA per segment per core, or one per core when the segments are alike (V: g169 2 x 4, g141 3 x 4 and 1 x 4).

**SRAM chunks.** First citizens, in order of live start, each at 0 when free there, else at the top of SRAM below any
top one live with it (V: cat16, g021, g141, g169): the shared INPUT, stored branch roots, a concat buffer read by one
layer (from the concat atom), CONCAT_STORAGE. Then the per-reader chunks of DRAM tensors read by several layers (one
each, live at its reader only: NPUC's "Hack on LA"; INPUT before the layers, a concat's at the concat's place), then
per layer. Eltwise: SHIFT_OUTPUT1/2 32 B, CU_BIN 128 B, ELTWISE ppc x cells x 64, META_DATA 32 B (64 B past 4 planes per
core). ReLU: CONV as a conv's (16-channel rounded) and WEIGHT.

**Eltwise group (an ARM intrinsic).** IR: the ARM's loads (operands first if any: those read from DRAM first, two
loads per operand when the cores hold different plane counts or for exactly 4 channels; then META_DATA), one element
per core (4 for 4 channels loaded back; in SRAM one per core: g513), a nop, [store (each of its DMAs an IR element:
g540), nop]. Stream: `r9 = nop; nop x5; swi = 0x10000 | g; nop; r1 = 0xffff0000 |
g; nop x4; wait3(r15, r1); nop x4; r15 = 0`, then with a store `nop x4; r9 = last nop; nop x5; ev2set(store) + the
usual wait; swi = 0x10000 | g; nop; swi = 0xffff0000 | g; nop x4; r1 = 0xffff0000 | g; nop x4; wait3; r15 = r1 = r3 =
0`; then the group end. One core: no first nop; the second form follows the elements directly with an event of its
own (a network flag, no DMA; V: rc2-*). Events: the ARM's loads count 1, 2, 4... within the group; META_DATA takes the
network's next flag when 4 or fewer planes per core and several cores, even one the group's loads use (g537), else the
group's next (V: res, rc4, res40..res64, rc2). The ARM's loads are triggered by the firmware: no ev2set in the CMDQ.

**ReLU layer (V: res, relu-*, cat16-relu, g008, ops-check relu-q3).** NPUC's 1x1 dummy conv: per pass 0x940 = 0x101
(GReLU mode 1, no table; a ReLU6: mode 3 with grelu.py's npuc_relu6_blocks), shift codes and biases 0, weights from
weights.py's relu conv (all at the zero point). Reading a tensor a conv wrote (a conv, a concat in SRAM) or a concat's
buffer, or a ReLU that read so (g555): every pass reads all planes (kin = 2 ppc, src fixed); else (after a pool, an
eltwise, a per-pass ReLU) pass p reads its
two plane slots (kin = 4, src + 2p cells). The per-pass WEIGHT chunk is ceil(4 x passes x 2 x rows x 64 / cores) and
pass slots are its rows over the passes (V: relu-c1..c24; 3 cores: 171 B a pass, 3 rows apart).

**Weights of a graph conv** (V: `$NCPGEN_WORK/sweep/weights/cat`, 86/86): the WEIGHT chunk is R x 2 x the planning
rows x 64 (NPUC's planner packs the kernel itself, in the unpermuted channel order for a conv reading a concat in
SRAM), while the loads, the 0x300 registers and the ring step use the emitted block's rows, sub-batches and
sub-channels (`weights.weight_layer_info`); a ring's slots sit 2 x planning rows apart, each slot's second half at the
emitted rows (V: g103, g167). **Misses (unseen set).** none left (g103, g161, g167 fixed with weights.py's planning
rows; g134: a 4-channel eltwise's
second operand load skips a flag: fixed since (that load, the second plane slot of a 4-channel operand, takes the
network's next flag the group does not use, like META_DATA; V: g134, rc4-*, g120).

## 7d. NPUC's "CMDQ seems too big" cut (V: the cut sweep `stream_variants_cut.txt` in `$NCPGEN_WORK/sweep/cut`, 49
networks; big4; MobileNet's 1024 x 1024 1x1 at 7x7 and 14x14)

**The estimate** (V: NPUC's "Heuristically predicted size of CMDQ" on 1619 layer blocks, 0 misfits after the rules
below): per layer block (one layer's NCP group) 16350 + 1020 per conv pass + 60 per padded plane slot + 102 per
reformatted one + 150 per pooled one + 1500 per eltwise; NFU ops count once per plane slot whatever their width
pieces. Its "Number of instructions in a group" is n = 61440 // ((estimate - 16350) // instructions), the
instructions being the passes plus those plane-slot ops (integer divisions: ci100k5 1800 // 14 = 128 -> 480).

**The cut**: when the estimate passes 61440 (60 KiB: k1c44 61230 whole, k3c44 61710 and k1c46 63270 cut), n is halved
and the layer's instructions, in IR order (its NFU ops first, then the passes), go in chunks of n, each its own NCP
group: big4 (64 passes, n 60 -> 30) 30 + 30 + 4; k3c44-ci64 (8 pads + 44 passes, n 70 -> 35) 8 pads + 27 passes, then
17; k3c64-ci512 (64 pads + 64 passes, n 56) 56 pads, then 8 pads + 48 passes, then 16. In each chunk: its loads go
with their consumers (a weight load with its pass; a 1x1 conv's input load stays in the first chunk; an NFU-first
layer's input is loaded chunk by chunk for the slots it pads, 4 parts of ceil(slots / 4) when 16 or more); a layer
that stores stores per chunk, the planes of the chunk's passes (2 p0 .. 2 p1 per core, from OFM + 2 p0 cells x 64,
one DMA, two when the cores' counts differ), never pass by pass (k1c1000-h1); a layer whose output stays in SRAM
stores nothing (k1c64x2). The events follow the new order (one counter); each chunk group's triggers follow 7b.

**Seen with it**: a kernel over 4 MiB in all is loaded core by core, 4 DMAs per pass to the same SRAM slot, kept in one
batch (k3c64-ci512 5.3 MB, s2c64 5.8 MB; one DMA at 2.6 MB: u512-ci512); k3c64-ci512's loads share one batch id while
its triggers come in sub-batches (once read as NPUC's CODEGEN_MINIMAL_LOADS_NUM_FOR_BATCH_CLUSTERING = 64; it is plain
SetBatchTransferInGroup: no load of that group has a wait, 7b, and the tiling3 ra rings loaded core by core, 112..128
loads, get one id per trigger); the weight-only batch count (7b, gone) needed the ring's input read first by an NFU op
(a 1x1 conv's ring: 15 in all, k1c64-ci2048, mb1024-14b).

Not modelled: a cut tile group (none seen: c512's tiles estimate 58590), a cut layer with pools or eltwise (NPUC would
need more than 300 pooled plane slots).

## 8. Unknowns and proposals

- Tiling: the policy is NPUC's own (frame.md 6.10, read from its code); the tile geometry rule (`tile_to_end`) and
  the read-back's 270 KiB per-tile bound are fits (7b), V on every compile so far.
- Meanings that never vary in the sweep (0x40120004 = 0x420, 0x40120040, 0x40120580/48/4c, the 0x800 select
  sequence, the 7 of the partial-sum registers): constants here, meaning G. The pool's kh/kw and sh/sw nibble order.
- Pad width bound: 116 (Wp 116 one command, 117 untested by itself, 118 split).
- The trigger batches (7b) are NPUC's own EventSchedule now (graph, representatives, event pool, ev2set lines, io
  batch ids), run on a generated memory profile; the profile formats are V on every compiled element, but they are
  NPUC's logged view (per-core loads at IDP core + j, NFU writes back to back), so a new op kind needs its own fit
  against the log before its batches can be trusted (stream.py refuses a group it cannot profile).
- NPU runs (no phone here) that would settle meanings rather than bytes: the compare-on-device of a chain with a
  multi-pass conv (co24) and one with sub-batches (ci64k7) against the CPU reference, both untested on the NPU.

## 9. What the other parts must know

- Events: DMA n -> flag 1 << (n mod 16), io descriptors in group order, loads then stores; 1 or 2 stores (section 7:
  a conv output with a short last core, or a pool on one with odd ppc); chunked input loads (4 io descriptors) with
  16+ planes per core and a pad or pool first.
- `stream.plan_dmas(low)`: every DMA in io-descriptor order with its trig_flag, bid (trigger batch), sgid (= NCP
  group), kind, SRAM offset, and for tiles the tile, core and tensor region (rows y0..y1, columns x0..x1: the loaded
  input region aligned to cells, or the stored output region); weight loads carry layer, pass and ring slot offset.
  Count, flag, bid and sgid match the oracle's io descriptors on every compile (2635; bids from `npuc_batch_ids`,
  7b; the tiling3 ra rings loaded core by core were 40 misses under the fitted rule).
- **frame.py** took the tile geometry (`tile_to_end`), the read-back count (now NPUC's policy, which stream uses
  too) and a 15-weight cap in `batches_from_streams` (c634f9d).
- NPUC's feature-map sizes can exceed lowered.json's (a VALID pool on an odd width with an even-covered height adds a
  column, which every later layer keeps: section 7). Anything sized from them (SRAM chunks, store descriptors, the
  intermediate DRAM) must use NPUC's sizes; `stream.Plan(low).info[i]` holds them ("h", "w", "ho", "wo").
- SRAM offsets (section 5) are what the io descriptors' dst/src offsets use (weights at WEIGHT, input at 0).
- grelu.py gets the whole layer and returns every pass's block; stream.py splits them at the 0x401e0050 brackets.
- Graphs (7c): `plan_dmas` adds the ARM's DMAs of an eltwise group (`arm: True`, kinds eltwise_in / eltwise_meta,
  their own flags), `tensor` on input loads and stores (the DRAM tensor: the network input, a stored intermediate, a
  concat's buffer), `srams` per part for segmented stores. Event flags are no longer 1 << (n mod 16) in graphs: the
  eltwise's flags are its own. A concat in SRAM permutes its readers' input channels (weights.py and the reference
  must follow it). grelu.py: NPUC writes zero point 0 in quant 3 whatever out_zp (mknets' zp-0 nets), not (zp - 128).
