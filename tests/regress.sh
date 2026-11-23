#!/usr/bin/env bash
# The regression check against NPUC's own compiles: ncpgen.py check (whole files, every part ours) on every compiled set
# in the work folder ($NCPGEN_WORK/sweep/*: lowered.json + NPUC's net.ncp in each variant directory). Needs the work
# folder (vendor-generated NCPs); tests/run.sh is the check that needs none.
#   tests/regress.sh [set...]       (default: every set under $NCPGEN_WORK/sweep)
# Prints "<set>: <exact> of <n>" per set and a total. Every variant is EXACT, REFUSED (a part or the frame raises:
# `ncpgen gen` writes nothing) or DIFFERS (gen would write a file that is not NPUC's). Rule 3 (AGENTS.md): a DIFFERS
# is a bug, whatever the set, except the variants of NPUC_UNINIT below, whose NCPs NPUC itself fills with
# uninitialised bytes. A REFUSED is allowed only for the variants of KNOWN_REFUSED, each with the NPUC behaviour ncpgen
# does not model. So a change that turns a refusal into wrong bytes, or an exact network into a refusal, fails here.
# Exit 0 when every variant is EXACT or in its list; an entry of a list that came out EXACT is reported (drop it).
# The sets are checked in parallel: REGRESS_JOBS (default: the CPU count).
set -uo pipefail
. "$(dirname "$0")/../oracle/env.sh"
S=$NCPGEN_WORK/sweep
cd "$S" || { echo "regress: no $S"; exit 1; }
if [ $# -gt 0 ]; then sets=("$@"); else
  mapfile -t sets < <(find . -mindepth 3 -name lowered.json -path '*/*/lowered.json' | sed 's|^\./||; s|/[^/]*/lowered.json$||' | sort -u)
fi
# NPUC's own uninitialised bytes (docs/format/frame.md 6.8, docs/oracle.md): a 4-channel eltwise loads each operand in
# two parts whose second descriptor holds uninitialised words in NPUC's output; no generator can match them
NPUC_UNINIT=(graph/rc4-16x16 graph/rc4-16x8 graph/rc4-24x12 graph/rc4-8x24 graph/rc4-8x8
             graph-random2/g120 graph-random2/g134)
# Refused, each naming what of NPUC's ncpgen does not model:
KNOWN_REFUSED=(
  # hand-edited net.npuc files whose lowered.json does not describe the compiled network (its outputs)
  grelu/r/r6c16 grelu/r/r6c8 grelu/r/r6q3 grelu/r/rlc8
  # SSD MobileNet v2 cut with --add-as-conv: frame refuses a load of a tensor no layer stores (after the 19x19x576
  # depthwise layer's weight ring; models/camera.md, docs/format/add.md 7)
  camera-add/ssd-mobilenet-v2-heads camera-add/ssd-mobilenet-v2-t166 camera-add/ssd-mobilenet-v2-t253
  # a dense depthwise conv in a one-layer tiled stratum that NPUC gives a PARTIAL_SUM chunk though its weights pack in
  # one sub-batch (policy weight buffer 960 B; frame.stratum_psum_unmodelled, models/README.md "Camera resolutions")
  zoo/efficientdet-lite0-320-p4
  # a tiled stratum's output read by several untiled layers and by another tiled stratum (frame.md 6.11)
  graph-camera/v3003
  # strata joined: a tiled stratum's output read by another tiled stratum and a DRAM concat (frame refuses a load of
  # a tensor no layer stores; frame.md 6.11 "Not yet")
  add-camera3/m5003
)
J=${REGRESS_JOBS:-$(sysctl -n hw.ncpu 2> /dev/null || nproc 2> /dev/null || echo 4)}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
in_list() { local x=$1; shift; local y; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }
# the checks, J at a time: each large variant (lowered.json over 30 MB: minutes each) alone and first, the others in
# chunks of 25 per set; the results are printed per set, in the sets' order
n_sets=0; jobs_=()
for s in "${sets[@]}"; do
  ls -d "$S/$s"/*/ > /dev/null 2>&1 || continue
  printf '%s\n' "$s" > "$T/$n_sets.set"
  mapfile -t big < <(find "$S/$s" -mindepth 2 -maxdepth 2 -name lowered.json -size +30M | sed 's|/lowered.json$|/|' | sort)
  small=(); for d in "$S/$s"/*/; do in_list "$d" "${big[@]}" || small+=("$d"); done
  j=0
  for d in "${big[@]}"; do jobs_=("$n_sets.$j $d" "${jobs_[@]}"); j=$((j + 1)); done
  for ((i = 0; i < ${#small[@]}; i += 25)); do jobs_+=("$n_sets.$j ${small[*]:i:25}"); j=$((j + 1)); done
  n_sets=$((n_sets + 1))
done
for jb in "${jobs_[@]}"; do
  read -r id dirs <<< "$jb"
  # shellcheck disable=SC2086
  ( python3 -I "$NCPGEN/ncpgen/ncpgen.py" check $dirs > "$T/$id.out" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
done
wait
tot=0; ok=0; bad=0; nref=0; nuni=0
for ((k = 0; k < n_sets; k++)); do
  s=$(cat "$T/$k.set"); n=0; e=0; nores=""
  for f in "$T/$k".*.out; do
    line=$(tail -n 1 "$f")
    n_=$(sed -nE 's/^ncpgen check: ([0-9]+) variants.*/\1/p' <<< "$line")
    e_=$(sed -nE 's/.*exact: ([0-9]+), refused.*/\1/p' <<< "$line")
    if [ -n "$n_" ] && [ -n "$e_" ]; then n=$((n + n_)); e=$((e + e_)); else nores+=" $line"; fi
  done
  [ -z "$nores" ] || { echo "$s: (no result)$nores"; bad=1; continue; }
  tot=$((tot + n)); ok=$((ok + e))
  msg=""; kr=""; ku=""; st=""
  # the variant lines: "<name> <parts...>  <EXACT|REFUSED|DIFFERS>" (notes are indented, skipped lines end in ")")
  while read -r v cls; do
    case $cls in
      EXACT)   if in_list "$s/$v" "${KNOWN_REFUSED[@]}" "${NPUC_UNINIT[@]}"; then st+=" $v"; fi ;;
      REFUSED) if in_list "$s/$v" "${KNOWN_REFUSED[@]}"; then nref=$((nref + 1)); kr+=" $v"
               else msg+="   REFUSED, not a known refusal: $v"; bad=1; fi ;;
      DIFFERS) if in_list "$s/$v" "${NPUC_UNINIT[@]}"; then nuni=$((nuni + 1)); ku+=" $v"
               else msg+="   DIFFERS, wrong bytes (rule 3): $v"; bad=1; fi ;;
    esac
  done < <(cat "$T/$k".*.out | awk '/^[^ ]/ && ($NF == "EXACT" || $NF == "REFUSED" || $NF == "DIFFERS") {print $1, $NF}')
  echo "$s: $e of $n${kr:+   known refusals:$kr}${ku:+   NPUC uninitialised bytes:$ku}${st:+   now exact, drop from the lists:$st}${msg:+  $msg}"
done
echo "regress: $ok of $tot whole files generated entirely by ncpgen = NPUC's; $nref known refusals, $nuni of NPUC's" \
     "uninitialised-byte files ($([ $bad = 0 ] && echo PASS || echo FAIL))"
exit $bad
