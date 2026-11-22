#!/usr/bin/env bash
# Run one of the harness programs (npuc-probe, npuc-conv, npuc-net) from an NPUC run directory made by
# harness/build.sh, on this host: natively on aarch64 Linux, else in an arm64 Linux container (docker; e.g. an
# Apple-silicon Mac; image $N3_IMAGE), or under qemu-aarch64. No device.
#   oracle/run.sh local <root-dir> [--prog npuc-conv|npuc-net] [--env NAME=VALUE]... [--push f]... [program args]
#   oracle/run.sh qemu  <root-dir> [--prog npuc-conv|npuc-net] [--env NAME=VALUE]... [--push f]... [program args]
# Output: $NCPGEN_WORK/n3/runs/<UTC>-<mode>-<prog>-<sha8>/ (or $N3_RUNS): log.txt, the files the program wrote, and
# fetch.txt (each file the program announced as "FILE <name>", with its sha256). Exit = the program's (npuc-probe:
# 0 PASS, 2 load, 3 symbols, 4 options; npuc-conv: 0 PASS, 2 compile, 3 bad NCP; npuc-net: 0 PASS, 1 bad input,
# 2 compile, 3 bad NCP), or 5 when an announced file was not written.
# --env passes NPUC's knobs (NPUC_LOG_LEVEL...); --push copies an input next to the program (npuc-net's net.npuc).
set -uo pipefail
. "$(dirname "$0")/env.sh"
MODE=${1:-}; R=${2:-}; shift 2 || { sed -n '2,10p' "$0"; exit 1; }
[ -f "$R/SHA256SUMS" ] && [ -x "$R/linker64" ] || { echo "$R is not a run directory"; exit 1; }
(cd "$R" && sha256sum -c --quiet SHA256SUMS) || { echo "$R does not match its SHA256SUMS"; exit 1; }
PROG=npuc-probe; [ "${1:-}" = --prog ] && { PROG=$2; shift 2; }
# --env NAME=VALUE (repeatable): extra environment for the program (NPUC reads NPUC_LOG_LEVEL, NPUC_GOLDEN, ...)
ENVS=()
while [ "${1:-}" = --env ]; do
  case "${2:-}" in [A-Z_][A-Z0-9_]*=*) ;; *) echo "--env wants NAME=VALUE, got '${2:-}'"; exit 1 ;; esac
  case "${2#*=}" in *[!A-Za-z0-9._/:,+-]*) echo "--env value '${2#*=}': letters, digits and ._/:,+- only"; exit 1 ;; esac
  ENVS+=("$2"); shift 2
done
# --push <file> (repeatable): a file the program reads from its directory (npuc-net's --net file)
PUSH=()
while [ "${1:-}" = --push ]; do
  [ -f "${2:-}" ] || { echo "--push: no file '${2:-}'"; exit 1; }
  case "$(basename "$2")" in *[!A-Za-z0-9._-]*|SHA256SUMS|linker64|npuc-*|lib*.so) echo "--push: name '$(basename "$2")' not allowed"; exit 1 ;; esac
  PUSH+=("$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2
done
case "$PROG" in npuc-probe|npuc-conv|npuc-net) ;; *) echo "unknown program $PROG"; exit 1 ;; esac
R=$(cd "$R" && pwd); h=$(sha256sum < "$R/$PROG" | cut -c1-8)
# runs go to $NCPGEN_WORK/n3/runs, or $N3_RUNS (a sweep's own folder); a unique name even when several callers start in
# the same second (mkdir is the atomic claim)
RUNS=${N3_RUNS:-$NCPGEN_WORK/n3/runs}; mkdir -p "$RUNS"
RUN0=$RUNS/$(date -u +%Y%m%dT%H%M%SZ)-$MODE-$PROG-$h; RUN=$RUN0; n=1
until mkdir "$RUN" 2> /dev/null; do RUN=$RUN0-$n; n=$((n + 1)); done
LOG=$RUN/log.txt
# qemu and local: the program wrote its files in the run directory; record each one it announced ("FILE <name>") as
# a fetch from a device would, so a caller checks every mode the same way
written() {
  : > "$RUN/fetch.txt"
  for f in $(tr -d '\r' < "$LOG" | sed -n 's/^FILE \([A-Za-z0-9._-]*\)$/\1/p'); do
    if [ -s "$RUN/$f" ]; then
      echo "fetched $f: $(wc -c < "$RUN/$f" | tr -d ' ') B sha256 $(sha256sum < "$RUN/$f" | cut -d' ' -f1)" >> "$RUN/fetch.txt"
    else
      echo "fetch FAILED $f: announced but not written" >> "$RUN/fetch.txt"; rc=5
    fi
  done
}
case "$MODE" in
  qemu)
    Q=$(command -v qemu-aarch64 || command -v qemu-aarch64-static) || { echo "no qemu-aarch64 on this host"; exit 1; }
    mkdir -p "$RUN"
    for f in "${PUSH[@]}"; do cp "$f" "$RUN/"; done
    # linker64 run as a program; no -L: the linker and every library are opened by absolute path or LD_LIBRARY_PATH
    QE=(-E LD_LIBRARY_PATH="$R"); for e in "${ENVS[@]}"; do QE+=(-E "$e"); done
    (cd "$RUN" && timeout 300 "$Q" "${QE[@]}" "$R/linker64" "$R/$PROG" "$@") > "$LOG" 2>&1
    rc=$?; written ;;
  local)
    # the program runs with the run directory as its cwd, so it reads the pushed inputs and writes its files there
    mkdir -p "$RUN"
    for f in "${PUSH[@]}"; do
      cp "$f" "$RUN/"; echo "push $(basename "$f") sha256 $(sha256sum < "$f" | cut -d' ' -f1)" >> "$RUN/push.txt"
    done
    if [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ]; then
      echo "local: native $(uname -sm)" > "$RUN/how.txt"
      (cd "$RUN" && env LD_LIBRARY_PATH="$R" "${ENVS[@]}" timeout 300 "$R/linker64" "$R/$PROG" "$@") > "$LOG" 2>&1
      rc=$?
    else
      command -v docker > /dev/null || { echo "local mode: not an aarch64 Linux host, and no docker for an arm64 container"; exit 1; }
      # pinned by digest (debian:bookworm-slim, 2026-11-21) so every host runs the same userland around Bionic
      IMG=${N3_IMAGE:-debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc}
      echo "local: container $IMG on $(uname -sm)" > "$RUN/how.txt"
      DE=(-e LD_LIBRARY_PATH=/r); for e in "${ENVS[@]}"; do DE+=(-e "$e"); done
      docker run --rm --platform linux/arm64 -v "$R":/r:ro -v "$RUN":/w -w /w "${DE[@]}" "$IMG" \
        timeout 300 /r/linker64 "/r/$PROG" "$@" > "$LOG" 2>&1
      rc=$?
    fi
    written ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
cat "$LOG"; [ -s "$RUN/fetch.txt" ] && cat "$RUN/fetch.txt"; echo "log: $LOG (exit $rc)"
exit "$rc"
