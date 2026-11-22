# The GReLU activation table (PORT-32, NCP format, part "grelu")

Exynos 990 NPU (NPU2020 ISA, SoC model Neus), NCPs compiled by Samsung's NPUC 1.36.1.a (the phone's own compiler, run
as a byte-exact oracle on the host). How NPUC fills the activation ("GReLU") table of every convolution, why its table
is wrong for small convolutions, and what the corrected table is. Code: `ncpgen/grelu.py`; sweep:
`ncpgen/grelu_sweep.sh` (variants under `$NCPGEN_WORK/sweep/grelu/v/`, git-ignored).

Tags: **V** verified (bytes, log, disassembly or an NPU run), **G** guess. Addresses are the libraries' own (Ghidra
shows them +0x100000). No vendor bytes here: register addresses, field layouts, formulas and observed constants only.

## 1. Summary

- NPUC's on-device quantisation path stores **each DRU's requant slope and output zero point** in the 16 slots that
  `ncp_mtc_set_file` writes as the GReLU table (index = core x 4 + lane). The activation code then edits those slots
  as if they were a piece table (ReLU zeroes slots 0 and 1). The hardware reads them as **one piece table for every
  channel**: entry 0 for acc < T1, entry 2 for acc >= T2. So the table is right only when slot 2 (core 0, lane 2)
  holds a channel's slope: **Cout in 9..16, 25..32, 41..48, ... (ceil(Cout/8) even; every multiple of 16)**; for
  Cout <= 8 (all our NPU tests) entry 2 is empty, so every acc >= 0 lands on (0, 0): the measured defect (V).
- What looked like "ReLU's identity one group too far" is the slope/zero point of DRUs 4.. (cores 1-3), not a piece.
- `grelu.py` reproduces NPUC's GReLU writes **byte for byte on all 92 compiled variants** (activation none/ReLU,
  quant 1/3, 9 output zero points, 10 scale sets with non-power-of-two slopes, Cout 1..255, 23
  shape/kernel/stride/padding variants up to 128x128 and Cin 512, 6 chains of up to 3 convs with pools, soc 1 and 2,
  plus the c3 and net-current references) and the corrected table identical to `ncp-grelu-patch.py auto` on 90/92
  (the 2 others: Cout <= 2 ReLU, which auto cannot repair; see 8).
- Side finding (V): NPUC's slope is `round_half_away(m x 256)` (fcvtas), not `trunc` (seen in 5 variants: 161 vs
  NPUC 162, 194/195, 90/91, 9/10, 6/7); every module now computes it the same way (`common.requant`).

## 2. The block (V)

Written by libnpuc_backend `NPU2020::NcpMtcConvolution::ncp_mtc_set_file` @0x1422d8 when the conv's GReLU mode
(ncp_mtc_conv +0x1a4) is 2 or 3, between the 0x4012004c and 0x40120200 writes of each MTC convolution:

| # | Block | Register | Value |
|---|---|---|---|
| 1 | CMDQ | 0x401e0050 | 1 |
| 2 | NFUBR | 0x40163800 | 0xff |
| 3-18 | NFUBR | 0x40163044 + 4k, k = 0..15 | `(int)(int16)A[k] \| (uint16)B[k] << 16` (A sign-extended: a negative A would clobber B; NPUC's A is always 0..255) |
| 19-20 | NFUBR | 0x40163084, 0x40163088 | T1, T2 (uint32), **mode 3 only** |
| 21 | NFUBR | 0x40163800 | 0 |
| 22 | CMDQ | 0x401e0050 | 0 |

0x401e0050 is written only around GReLU blocks (every variant, and 1390 = 2 x 695 times in the stock NFD.ncp).
The mode itself goes to **MTC 0x40120940 bits 8..15** (V: `ncp_mtc_set_folder` @0x141c38 writes
`bit0 | (conv+0xc == 1) << 24 | flag << 28 | mode << 8`): 0x10000301 for our quant-1 convs, 0x301 in quant 3,
0x10000101 for a standalone-ReLU dummy conv (mode 1, no table); NFD: 695 x 0x301 (its GReLU convs) and 145 x 0x001
(mode 0: FL convs without activation and without a block).
One block per **output-channel pass** (one MTC convolution each); none for pooling layers; spatial tiling does not
repeat it (128x128 and 112x112 inputs: one block per pass).

## 3. Where the values come from (V, code path)

| Step | Where | What |
|---|---|---|
| ConvAttr.fusedActivation -> node act | frontend `NNGraphNodeFusedActivationFiller::setFusedActivation` @0x9ee1c | the TFLite enum: 0 -> 0x42 (none), 1 -> 0x18 (ReLU), 3 -> 0x19 (ReLU6), 2/4/5 -> 0x43 ("Unsupported FusedActivation type") |
| node act -> NNActivationType | frontend `NNGraphCgObject::addConvolutionGraphNode` @0x66dd0 | soc 1..3: 0x42 -> 11 `GRELU_BYPASS_ONDEVICE`, 0x18 -> 10 `GRELU_RELU_ONDEVICE`, anything else (0x43, ReLU6 0x19) -> 3; soc 0: always 3 (`ACTIVATION_TYPE_LUT`) |
| per-DRU fill | template, lambda @0x1bf3d0 (vtable 0x374768) run by `NPUDRUSet::iter` inside `LIRGenExecutor2020::convCreateOperation` @0x1bbe0c | for each populated DRU d = core x 4 + lane: bias -> conv +0x84/+0xc4 [d]; if CompileOptions +0x138 == 1 (**setCompileOptions always writes 1**): shift word conv +0x160 [d] = (s << 11) \| (s << 6), **B[d] = slope (conv +0x1a6 + 2d), A[d] = out zero point byte (conv +0x109) (conv +0x1c6 + 2d)**; else (the offline FL path): shift word = the FL shift (bits 0..5) [\| s2 << 6 for type 5], table untouched |
| activation edit | same function, after the fill: `switch (NNActivationType)` | 1: mode 1. **2: B = (0, 1, 0), A = (0, 0, 0x60), T1 = 0, T2 = 0x60, mode 3** (exactly the stock NFD.ncp's table: a ReLU6 in the FL format). 3: mode 2. 5: B[0] = B[1] = slope(DRU 0) >> 16, B[2] = slope(DRU 0) & 0xffff, A[0..2] = 0, T = 0, mode 3 (a PReLU in FL, G). **10: B[0] = B[1] = 0, T1 = T2 = 0, mode 3. 11: T1 = T2 = 0, mode 3.** default: mode 0 |
| to the IR | backend `MakeIRList::Visit(ConvInstrDef2020)` @0x622d8 | the instruction def embeds ncp_mtc_conv at +0x28: attributes `activation` (+0x1cc), `druLutScale` (16 x i16 at +0x1ce), `druLutBias` (+0x1ee), `druLutBoundaries` (+0x210, +0x214) |
| to the NCP | backend `NcpMtcConvolution::Execute` @0x13e868 -> `ncp_mtc_set_file` | B = druLutScale, A = druLutBias, T1/T2 = druLutBoundaries |

ncp_mtc_conv GReLU fields: +0x1a0 u32 (1), +0x1a4 u8 mode, +0x1a6 u16 B[16], +0x1c6 i16 A[16], +0x1e8 u32 T1,
+0x1ec u32 T2; +0x109 u8 output zero point (0 in quant mode 3); +0x160 u32 shift word [16] (written to
0x40120900 + 4j as word[2j] \| word[2j+1] << 16); +0x84/+0xc4 bias low/high [16] (0x40120800 + 8d, + 4).

**Why it fails** (V): nothing is uninitialised and the firmware is not involved. The on-device ScaleON path keeps the
per-channel requantisation in the same 16 slots the GReLU unit reads as its piece table, and the activation edit
assumes slot 2 holds "the" slope. With all 16 DRUs populated (full passes, one slope per layer) both views agree;
with <= 2 lanes per core slot 2 is empty: BYPASS leaves (0, 0) above T2 = 0, ReLU's zeroed slots 0/1 give the zero
point below 0 and slot 2 gives 0 above. Samsung's usual layers (Cout a multiple of 16) never hit it.

## 4. Channel -> DRU and passes (V, sweep Cout 1..255)

```
chunk  = 2 * ceil(Cout / 8)          channels per core (core k: channels [k*chunk, (k+1)*chunk))
passes = ceil(chunk / 4)             one MTC convolution and one GReLU block each, in channel order
pass p, core k, lane l (0..3):       channel k*chunk + 4p + l (if < Cout and 4p + l < chunk) -> DRU k*4 + l
```

Examples (V): Cout 3 -> DRUs {0, 1, 4}; 8 -> {0,1,4,5,8,9,12,13}; 9 -> 0..8; 17 -> 12 + {0,1,4,5,8}; 33 -> 15 + 12 +
6; 100 -> 5 x 16 + 14 + 6; 255 -> 15 x 16 + 15. The same DRU index places the bias (0x40120800 + 8d) and the shift
word (0x40120900): checked on c3 and c17. Independent of Cin (3..512), kernel (1..7, 1x3), stride, padding, H x W
(4..128), quant mode and soc.

## 5. The slope (V)

frontend `NNGraphNodeScaleONFiller::setScale` @0x9da64: m = f32(f32(in x w) / out); while m < 0.5 (at most 7 times)
m = 2m; **slope = fcvtas(m x 256)** (round to nearest, ties away; @0x9dd40); assert slope < 0x10000; shift =
doublings + 8. NPUC logs `slope_data(162), slope_shift(10)` for s6 where npunet.py's lowered.json says 161:
npunet.py's `npuc_multiplier` (and so its CPU reference) must use `int(m * 256 + 0.5)`. grelu.py computes its own.
The table's B is this slope; the MTC shift word carries the shift in both 5-bit fields (bits 6..10 and 11..15).

## 6. The table on the NPU: proven vs guess

| Fact | Tag | Evidence |
|---|---|---|
| out = A + round_half_away(acc x B / 2^shift), A = zero point (0 in mode 3), B = slope | V | 20480 + bytes bit-exact (9.11-9.16) |
| one table for every channel: group 0 (entries 0..3) serves lanes 0 and 1 of all 4 cores | V | unpatched quant-3 ReLU, Cout 8 (set 169b0667): groups 1-3 held (0, 16) identity pieces, yet every byte was 0 |
| acc < T1 -> entry 0; acc >= T2 -> entry 2 (T1 = T2 = 0) | V | measured defect; N3d `piece2` (only entry 2 set) bit-exact; the 4 samples with acc = 0 in run 121349 came back 0 (entries 0/1 held the zero point): x = T2 is not entry 0 or 1 |
| entry 1 is never selected when T1 = T2 | V | as above |
| entry 1 = T1 <= acc < T2 (a middle piece) | G | NFD: (0,1) between 0 and 0x60, "ReLU6-like" output (N2, not checked bit for bit) |
| entry 3 and groups 1-3 unused | G | never needed in a bit-exact run; zero in NFD |
| lanes 2 and 3 (Cout >= 9) also read group 0 | G | never run (E1, E2 below) |
| thresholds compare the raw int32 accumulator (before slope/shift), signed | G | NFD's T2 = 0x60 with slope 1 and FL shift in bits 0..5 suggests "after the FL pre-shift" (E3, E5) |
| shift word bits 0..5 = pre-shift of acc (FL path), bits 6..10 = shift of pieces 0-1, bits 11..15 = shift of piece 2 | G | NPUC writes FL shifts in 0..5 (NFD: 7, 8, ...), s in both 6..10 and 11..15 on device, and for type 5 (PReLU) only 6..10 next to B[2] = 1 (E4) |
| 0x40163800 = 0xff: a broadcast mask over 8 NFU units; 0x401e0050: CMDQ access to them | G | names only |

So a generator can already emit, bit-exactly: no activation and ReLU for any Cout (the robust table below). A
bounded ReLU (ReLU6, ReLU-N1-to-1 clamps) needs only T2 > 0 and a constant piece 2 (`relu6_pieces`), G until E3;
leaky ReLU needs a fractional slope on piece 0: with one shared shift, B0 = round(alpha x slope) (lossy), exact if E4
confirms per-piece shifts.

## 7. Does NPUC ever emit a correct table itself?

- **Yes, for Cout with ceil(Cout/8) even** (9..16, 25..32, ...): entry 2 then holds (zp, slope): ReLU = (zp,0) (zp,0)
  (zp,s) (zp,s), none = (zp,s) x 4 (V bytes; NPU-correct if lanes 2-3 read group 0: E1). Padding a conv's outputs to a
  multiple of 16 channels makes NPUC's own NCP right for "none" whatever the group mapping (groups 1-3 are then
  identity too).
- No other input helps (V, code): soc 1 gives byte-identical NCPs to soc 2 (c8, n8, c16 compared whole); soc 3 aborts
  (no Kitt intrinsic generator, 9.13); soc 0 takes the LUT path (Makalu NCP v9, refused by the firmware);
  CompileOptions +0x138 (the switch to the per-DRU fill) is forced to 1 by setCompileOptions; +0xf4 does not reach the
  conv (9.13). ConvAttr act 2..5 (ReLU-N1-to-1, ReLU6, tanh, sign) reach addConvolutionGraphNode as 0x43/0x19 and become
  type 3 = mode 2 (16 per-DRU slots, no T1/T2 written: the previous conv's thresholds stay; G for the hardware) —
  npuc-net accepts only act 0/1, so this was read in the code (V for those two functions; a graph filter in between
  was not ruled out), not compiled.
- **Standalone activation layers** (V, oracle bytes; "npuc-net 2" `relu` lines of the harness-ops dev build,
  snapshot `$NCPGEN_WORK/sweep/grelu/root-dev`, variants `$NCPGEN_WORK/sweep/grelu/r/`): a ReLU6 layer after a conv
  becomes a dummy conv with **type 2: NPUC's own NFD table** (per-DRU fill slope 1, zero point 0, then B = 0,1,0,
  A = 0,0,0x60, T1 = 0, T2 = 0x60), shift words 0, bias 0: a clamp at the fixed accumulator value 96 with offset 0,
  whatever the scales and zero points (G: right at most for quant 3 with out scale 1/16). A ReLU layer becomes a dummy
  conv in **mode 1 with no block at all** (the NFU keeps the previous block's table, G). `grelu.py` models both
  (layer kind "relu"); 5/5 exact. Type 2 is NPUC's own statement of the piece semantics (entry 1 = identity between
  T1 and T2, entry 2 = constant above), but not a usable on-device ReLU6.

## 8. grelu.py

`grelu_regs(layer, quant_mode, fixed=False)` -> [(addr, value)] for a lowered conv (lowered.json's layer dict), all
blocks of all passes in order, brackets included (`grelu_blocks()` gives them per pass for interleaving with the
pass's MTC writes).

- `fixed=False`: NPUC's writes (slope from the scales, section 5; A = out_zp, 0 in quant 3).
- `fixed=True`: `ncp-grelu-patch.py auto` (relu -> zp/identity/identity in every group, none -> entry 2 := entry 0
  per group, else untouched) plus one repair: a ReLU pass with no slope left in NPUC's table (Cout <= 2) gets group
  0 = (zp,0) (zp,s) (zp,s) (0,0) copied to all groups (auto leaves it broken).
- `fixed="robust"` (**recommended for the generator** until E1/E2): every block = group 0 above (ReLU) or (zp,s) (zp,s)
  (zp,s) (0,0) (none), copied to groups 1-3, T1 = T2 = 0. Byte for byte the table of every bit-exact NPU run so far
  (N3d piece2, npu-run v2/v4/v5, net-current: all Cout 8), and safe whichever group lanes 2-3 read. It differs from
  auto only where auto keeps NPUC's groups 1-3 (Cout >= 9, or empty groups for Cout <= 6 without activation).
- `pwl_regs(pieces, t1, t2)`, `relu6_pieces(layer, q)`: an arbitrary 3-piece table (G).
- `python3 -I grelu.py check <variant dir>...`: per variant, NPUC writes vs the oracle NCP (exact/DIFFERS), fixed vs
  auto (run in memory with ncp-grelu-patch.py's own functions), robust vs auto; `table <ncp>`, `regs <lowered.json>`,
  `selftest` (15 checks, no compile).

Result (2026-11-21): `check $NCPGEN_WORK/sweep/grelu/v/*/ $NCPGEN_WORK/sweep/c3 $NCPGEN_WORK/sweep/grelu/ref/net-current-oracle`
-> 92 variants, NPUC writes exact 92/92; fixed = auto 90, intended deviation 2 (c1, c2), differs 0; robust = auto 47.
`check $NCPGEN_WORK/sweep/grelu/r/*/` (standalone ReLU/ReLU6, dev harness) -> 5/5 exact. The patched
`output/npu/run/net-current/net.ncp` (bit-exact on the NPU) equals `fixed=True` and `"robust"`.
Reproduce: `ncpgen/grelu_sweep.sh` (about 2 minutes, 6 compiles at a time), then the check.

## 9. Proposed NPU runs (main machine; `npu/npu-run.sh compile <net> --ncp-patch <spec>`, then `run`)

Nets from `npuc-variant.py` (16x16x8 int8 input, the measured IFM layout). `reg=val` specs apply to every block.

| Run | Net / spec | Prediction | Settles |
|---|---|---|---|
| E1 | `--conv 16,3` (ReLU) and `--conv 16,3,1,same,none` (`--allow-none`), `--ncp-patch off` | bit-exact both | NPUC's own table right for Cout 16; lanes 2-3 read group 0 (ReLU: groups 1-3 are identity there, so a lane on group 1 would pass negatives) |
| E2 | E1's ReLU net, `--ncp-patch 0x40163050=0x00400011,0x40163054=0x00400011,0x40163058=0x00400011,0x4016305c=0x00400011,0x40163060=0x00400011,0x40163064=0x00400011,0x40163068=0x00400011,0x4016306c=0x00400011,0x40163070=0x00400011,0x40163074=0x00400011,0x40163078=0x00400011,0x4016307c=0x00400011,0x40163080=0x00400011` | bit-exact | entry 3 and groups 1-3 unused for every lane |
| E3 | the example's first conv alone (`--conv 8,3`, slope 128/2^15, zp 128), `--ncp-patch 0x40163044=0x00000080,0x40163048=0x00800080,0x4016304c=0x00000094,0x40163050=0,0x40163084=0,0x40163088=4992` (+ the same 4 entries for groups 1-3 if E2 fails) | output = min(CPU ReLU reference, 148) byte for byte | T2 compares the raw accumulator; a constant piece 2 (bounded ReLU) works; if outputs > 148 remain, T2 is compared after scaling |
| E4 | needs MTC patching (not in ncp-grelu-patch.py): `--conv 8,3,1,same,none` robust table, shift words 0x40120900.. = (s << 11) \| ((s + 1) << 6) | negative outputs halved, positive exact | per-piece shift fields (exact leaky ReLU) |
| E5 | `--conv 8,3,1,same,none` (`--allow-none`), `--ncp-patch 0x40163044=0x00000076,0x40163048=0x00800080,0x4016304c=0x00800080,0x40163050=0,0x40163084=0xfffff680,0x40163088=0` (entry 0 = (118, 0) below T1 = -2432, identity from there) | output = max(CPU reference, 118) byte for byte | signed thresholds; piece 1 = [T1, T2) |

## 10. For the other parts

- CMDQ streams: one GReLU block per pass, inside the pass's MTC sequence after 0x4012004c and before 0x40120200
  (`grelu_blocks()`); passes and DRU placement per section 4 (bias 0x40120800 + 8d, shift words 0x40120900); the
  mode (3) in 0x40120940 bits 8..15.
- Weights/requant: the slope rounds half away (section 5); npunet.py's lowered `slope` is not NPUC's in about half of
  non-power-of-two multipliers.
- Harness: npuc-net.cpp accepts act 0/1 only; allowing the TFLite enum 0..5 (ConvAttr.fusedActivation) would let the
  oracle show the mode-2 block of ReLU6/tanh.
