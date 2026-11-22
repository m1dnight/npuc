#!/usr/bin/env bash
# The regression check against NPUC's own compiles: ncpgen.py check (whole files, every part ours) on every compiled set
# in the work folder ($NCPGEN_WORK/sweep/*: lowered.json + NPUC's net.ncp in each variant directory). Needs the work
# folder (vendor-generated NCPs); tests/run.sh is the check that needs none.
#   tests/regress.sh [set...]       (default: every set under $NCPGEN_WORK/sweep)
# Prints "<set>: <exact> of <n>" per set and a total. Known misses: graph 93/98 and graph-random2 64/66 (NPUC's own
# uninitialised bytes, docs/oracle.md); grelu/r 1/5 (hand-edited net.npuc files whose lowered.json does not describe the
# compiled network: ncpgen refuses them); zoo 1 (mobilenet-v1-100-224: a CMDQ cut and a spill in a tiled plan's single
# blocks: not modelled, models/README.md); graph-large 4 (t1039 a tiled read-back in a graph, t1018 a tiled layer whose
# input heads a concat's branches, t1038 a layer-block only a weight ring fits, t1016's io batch id: frame.md 6.11).
# Every other set must be exact. Exit 0 when only those known misses remain.
# The sets are checked in parallel: REGRESS_JOBS (default: the CPU count).
set -uo pipefail
. "$(dirname "$0")/../oracle/env.sh"
S=$NCPGEN_WORK/sweep
cd "$S" || { echo "regress: no $S"; exit 1; }
if [ $# -gt 0 ]; then sets=("$@"); else
  mapfile -t sets < <(find . -mindepth 3 -name lowered.json -path '*/*/lowered.json' | sed 's|^\./||; s|/[^/]*/lowered.json$||' | sort -u)
fi
tot=0; ok=0; bad=0
declare -A KNOWN=([graph]=5 [graph-random2]=2 [grelu/r]=4 [zoo]=1 [graph-large]=4)
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
