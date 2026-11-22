# NPUC as an oracle

ncpgen is checked against Samsung's NPUC 1.36.1.a, the compiler in the stock vendor partition of the Exynos 990 phones
(G980F firmware G980FXXSNHYB1 here), run outside Android. It is deterministic and host-independent: the same network
gives the same NCP on the phone, in an arm64 container on a Mac and on aarch64 Linux (with one exception: a 4-channel
elementwise add's operand descriptor holds uninitialised words). This is how to set it up; everything it uses or makes
stays in the work folder (`NCPGEN_WORK`, default `work/`; `oracle/env.sh` describes the layout).

## 1. The vendor and stock files (from your own firmware)

| Work folder | From | Files |
|---|---|---|
| `vendor-libs/` | the stock `vendor` partition, `/lib64` | `libnpuc_{api,controller,frontend,graph,template,cmdq,backend,common}.so` (and, for study, `nn-eden-drv`, `libeden_rt.so`) |
| `bionic/` + `SHA256SUMS` | the stock `system` partition: `/system/apex/com.android.runtime.apex` (its `apex_payload.img`) and `/system/lib64` | `linker64`, `libc.so`, `libm.so`, `libdl.so` (the APEX), `liblog.so`, `ld-android.so`, `libc++.so` |

The porting kit's dossier (`ports/samsung-x1s/dossier/npu-compiler.md` §9.1) has the extraction path from the AP tar
(`super.img.lz4`, the LP partitions, the APEX) and the checksums of this firmware's files.

## 2. Build the harness and the run directory

Our programs call NPUC in-process, as Android's NNAPI HAL does, through objects reconstructed in `harness/npuc.h`:
`npuc-probe` (load test), `npuc-conv` (one conv) and `npuc-net` (a network from `npunet.py lower`'s `net.npuc`: conv, pool,
eltwise add, concat, relu). They need the Android NDK r27d, Linux x86-64 build
(`android-ndk-r27d-linux.zip`, sha256 `601246087a682d1944e1e16dd85bc6e49560fe8b6d61255be2829178c8ed15d9`), in
`work/tools/android-ndk-r27d` or `NDK=`.

```bash
oracle/harness/build.sh                    # on x86-64 Linux; on a Mac, in an amd64 container:
docker run --rm --platform linux/amd64 -v "$PWD/..":"$PWD/.." -w "$PWD" python:3.12-slim-bookworm oracle/harness/build.sh
```

It writes `work/n3/root-<sha8>/` (Bionic + NPUC + the programs, SHA256SUMS) and links `work/n3/root-current` to it.
The build is reproducible from any path (the repository and the NDK are mapped to `.` and `/ndk` in the debug info):
with this firmware's files and NDK r27d it gives **`root-314e61a6`** (`npuc-net-6f90837b`, `npuc-probe-85fe33b0`,
`npuc-conv-5500f56c`). `check-symbols.py` checks every import binds in Bionic's lookup scopes before anything runs.

## 3. Run it

```bash
oracle/run.sh local work/n3/root-current                          # npuc-probe: expect "npuc-probe: PASS"
oracle/npuc-local.sh net.json out/ [--quant 1|3] [--allow-none]   # a model -> out/net.ncp, out/log.txt, out/run/
```

`run.sh local` runs natively on aarch64 Linux, else in an arm64 Linux container (`debian:bookworm-slim`, pinned by digest;
Apple silicon: the docker engine runs it natively); `run.sh qemu` uses `qemu-aarch64`. A compile takes about 1.3 s.
NPUC's log (`log.txt`) prints every instruction it emits (`code line: set_reg(r9, 0x1)` with its machine code), its
tiling, memory and DMA decisions; it also writes its IR at six stages (`run/ir_graph_*.txt`).

## 4. Map something new

1. Write variants: `oracle/npuc-variant.py out.json --in H,W,C --conv COUT,K[,S[,same|valid[,relu|none]]] --pool max|avg,K[,S] --seed N`
   (our own seeded weights), compile each with `npuc-local.sh`.
2. Diff the NCPs: `oracle/ncp-dump.py net.ncp --log log.txt` lists every byte with NPUC's own instruction text.
3. Write the rule into the part (`ncpgen/stream.py`, `frame.py`, `weights.py`, `grelu.py`) and its note
   (`docs/format/`), then check: `ncpgen/ncpgen.py check <dirs>` (whole files), or the part's own `check`.
4. Where diffs are not enough, read NPUC: `oracle/ghidra/run-ghidra.sh import`, then `decompile <targets.txt>`
   (Ghidra 12.2 and a JDK in `work/tools/`; the targets files name the functions read so far).
5. Record new golden hashes with `tests/make-golden.sh` only for networks NPUC compiled.

A network can also use ops NPUC's frontend knows but the HAL never gives it (`docs/harness-ops.md`): adding one starts in
`harness/npuc-net.cpp`.
