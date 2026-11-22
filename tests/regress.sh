#!/usr/bin/env bash
# The regression check against NPUC's own compiles: ncpgen.py check (whole files, every part ours) on every compiled set
# in the work folder ($NCPGEN_WORK/sweep/*: lowered.json + NPUC's net.ncp in each variant directory). Needs the work
# folder (vendor-generated NCPs); tests/run.sh is the check that needs none.
#   tests/regress.sh [set...]       (default: every set under $NCPGEN_WORK/sweep)
# Prints "<set>: <exact> of <n>" per set and a total. Known misses: graph 93/98 and graph-random2 64/66 (NPUC's own
# uninitialised bytes, docs/oracle.md); grelu/r 1/5 (hand-edited net.npuc files whose lowered.json does not describe the
# compiled network: ncpgen refuses them); graph-split 3: cat96-64, cat128-128, cat160-64 (concat blocks whose feature
# maps NPUC tiles: ncpgen refuses tiling in a graph). Every other set must be exact. Exit 0 when only those known
# misses remain.
set -uo pipefail
. "$(dirname "$0")/../oracle/env.sh"
S=$NCPGEN_WORK/sweep
cd "$S" || { echo "regress: no $S"; exit 1; }
if [ $# -gt 0 ]; then sets=("$@"); else
  mapfile -t sets < <(find . -mindepth 3 -name lowered.json -path '*/*/lowered.json' | sed 's|^\./||; s|/[^/]*/lowered.json$||' | sort -u)
fi
tot=0; ok=0; bad=0
declare -A KNOWN=([graph]=5 [graph-random2]=2 [grelu/r]=4 [graph-split]=3)
for s in "${sets[@]}"; do
  ls -d "$S/$s"/*/ > /dev/null 2>&1 || continue
  line=$(python3 -I "$NCPGEN/ncpgen/ncpgen.py" check "$S/$s"/*/ 2>&1 | tail -n 1)
  n=$(sed -nE 's/^ncpgen check: ([0-9]+) variants.*/\1/p' <<< "$line"); e=$(sed -nE 's/.*exact: ([0-9]+)$/\1/p' <<< "$line")
  [ -n "$n" ] || { echo "$s: (no result) $line"; bad=1; continue; }
  tot=$((tot + n)); ok=$((ok + e)); miss=$((n - e))
  if [ "$miss" -gt "${KNOWN[$s]:-0}" ]; then echo "$s: $e of $n   REGRESSION"; bad=1; else echo "$s: $e of $n"; fi
done
echo "regress: $ok of $tot whole files generated entirely by ncpgen = NPUC's ($([ $bad = 0 ] && echo PASS || echo FAIL))"
exit $bad
