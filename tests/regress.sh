#!/usr/bin/env bash
# The regression check against NPUC's own compiles: ncpgen.py check (whole files, every part ours) on every compiled set
# in the work folder ($NCPGEN_WORK/sweep/*: lowered.json + NPUC's net.ncp in each variant directory). Needs the work
# folder (vendor-generated NCPs); tests/run.sh is the check that needs none.
#   tests/regress.sh [set...]       (default: every set under $NCPGEN_WORK/sweep)
# Prints "<set>: <exact> of <n>" per set and a total. Known misses: graph 93/98 and graph-random2 64/66 (NPUC's own
# uninitialised bytes, docs/oracle.md); grelu/r 1/5 (hand-edited net.npuc files whose lowered.json does not describe the
# compiled network: ncpgen refuses them); graph-large2 2 (the fresh large random set, stream_graphs.py random <dir> 2000
# 40 --set large: t2006 a spill NPUC survives that our model aborts on, and its weight blocks; t2022 a concat NPUC's
# forwarding keeps out of SRAM: frame.md 6.11); camera 3 (SSD MobileNet v1's heads at 300 px: NPUC tiles its 150x150
# and 75x75 layers, ncpgen's plan does not; models/camera.md); outputs 2 (o-tiled-head, o-big-readback: refused tiled read-backs of several outputs, frame.md 12).
# Every other set must be exact. Exit 0 when only those known misses
# remain.
# The sets are checked in parallel: REGRESS_JOBS (default: the CPU count).
set -uo pipefail
. "$(dirname "$0")/../oracle/env.sh"
S=$NCPGEN_WORK/sweep
cd "$S" || { echo "regress: no $S"; exit 1; }
if [ $# -gt 0 ]; then sets=("$@"); else
  mapfile -t sets < <(find . -mindepth 3 -name lowered.json -path '*/*/lowered.json' | sed 's|^\./||; s|/[^/]*/lowered.json$||' | sort -u)
fi
tot=0; ok=0; bad=0
declare -A KNOWN=([graph]=5 [graph-random2]=2 [grelu/r]=4 [graph-large2]=2 [camera]=3 [outputs]=2)
J=${REGRESS_JOBS:-$(sysctl -n hw.ncpu 2> /dev/null || nproc 2> /dev/null || echo 4)}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# one check per set, J at a time; the results are printed in the sets' order
n_sets=0
for s in "${sets[@]}"; do
  ls -d "$S/$s"/*/ > /dev/null 2>&1 || continue
  printf '%s\n' "$s" > "$T/$n_sets.set"
  ( python3 -I "$NCPGEN/ncpgen/ncpgen.py" check "$S/$s"/*/ 2>&1 | tail -n 1 > "$T/$n_sets.out" ) &
  n_sets=$((n_sets + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
done
wait
for ((k = 0; k < n_sets; k++)); do
  s=$(cat "$T/$k.set"); line=$(cat "$T/$k.out")
  n=$(sed -nE 's/^ncpgen check: ([0-9]+) variants.*/\1/p' <<< "$line"); e=$(sed -nE 's/.*exact: ([0-9]+)$/\1/p' <<< "$line")
  [ -n "$n" ] || { echo "$s: (no result) $line"; bad=1; continue; }
  tot=$((tot + n)); ok=$((ok + e)); miss=$((n - e))
  if [ "$miss" -gt "${KNOWN[$s]:-0}" ]; then echo "$s: $e of $n   REGRESSION"; bad=1; else echo "$s: $e of $n"; fi
done
echo "regress: $ok of $tot whole files generated entirely by ncpgen = NPUC's ($([ $bad = 0 ] && echo PASS || echo FAIL))"
exit $bad
