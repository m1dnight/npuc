#!/usr/bin/env bash
# PORT-32: check npuc-net's added layer kinds (eltwise Add, concat, relu) against Samsung's NPUC on this host, no phone
# (docs/harness-ops.md). Generates the test networks (mknets.py), compiles each with the local oracle
# (npuc-local.sh, an arm64 Linux container or a native aarch64 Linux host) and checks the expected outcome: an NCP and
# NPUC's log showing the op, or npuc-net's refusal before any NPUC call. Then the old-input regression: the pinned
# example net ($NCPGEN_WORK/run/net-current/net.npuc, when present) must still give NCP 5c080ec1.
#   oracle/ops/check-ops.sh [run directory]      (default: $NPU_ROOT, oracle/env.sh)
# Writes under $NCPGEN_WORK/n3/ops-check/ (the work folder: compiled NCPs and logs are vendor-generated). Exit 0 all PASS.
set -uo pipefail
. "$(dirname "$0")/../env.sh"
S=$ORACLE
ROOT=${1:-$NPU_ROOT}
[ -x "$ROOT/npuc-net" ] || { echo "check-ops: no npuc-net in $ROOT"; exit 1; }
O=$NCPGEN_WORK/n3/ops-check/$(basename "$(cd "$ROOT" && pwd -P)"); mkdir -p "$O/nets"
python3 -I "$S/ops/mknets.py" "$O/nets" || exit 1
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
no() { fail=$((fail + 1)); echo "FAIL $1"; }
while IFS=$'\t' read -r name want markers; do
  d=$O/$name
  out=$(NPU_ROOT=$ROOT bash "$S/npuc-local.sh" "$O/nets/$name.npuc" "$d" 2>&1); rc=$?
  log=$d/log.txt; [ -f "$log" ] || log=/dev/null
  if [ "$want" = PASS ]; then
    [ $rc = 0 ] && [ -s "$d/net.ncp" ] || { no "$name: compile failed (exit $rc): $(printf '%s\n' "$out" | tail -n 2 | tr '\n' ' ')"; continue; }
    miss=""
    IFS='|' read -ra ms <<< "$markers"
    for m in "${ms[@]}"; do grep -qF -- "$m" "$log" || miss="$miss '$m'"; done
    [ -z "$miss" ] && ok "$name: NCP $(sha256sum < "$d/net.ncp" | cut -c1-8), log shows: $markers" || no "$name: log lacks$miss"
  else
    line=$(grep -m1 '^npuc-net: FAIL' "$log" | tr -d '\r')
    if [ $rc = 1 ] && [[ "$line" == *"$markers"* ]] && ! grep -q 'NPUCompiler::compile (' "$log"; then
      ok "$name: refused before NPUC ($markers)"
    else
      no "$name: want a refusal with '$markers', got exit $rc: $line"
    fi
  fi
done < "$O/nets/expect.tsv"
REF=$NCPGEN_WORK/run/net-current/net.npuc
if [ -f "$REF" ]; then
  out=$(NPU_ROOT=$ROOT bash "$S/npuc-local.sh" "$REF" "$O/regress-net-current" 2>&1 | tail -n 1)
  case "$out" in
    *" 5c080ec1e2b1a778ea68eda59faebec2db934cbc42457e5355226870bea4e167") ok "regression: net-current (npuc-net 1) still gives NCP 5c080ec1" ;;
    *) no "regression: net-current: $out" ;;
  esac
  sed '1s/^npuc-net 1$/npuc-net 2/' "$REF" > "$O/nets/net-current-v2.npuc"
  out=$(NPU_ROOT=$ROOT bash "$S/npuc-local.sh" "$O/nets/net-current-v2.npuc" "$O/regress-net-current-v2" 2>&1 | tail -n 1)
  case "$out" in
    *" 5c080ec1e2b1a778ea68eda59faebec2db934cbc42457e5355226870bea4e167") ok "regression: the same net under an \"npuc-net 2\" header gives NCP 5c080ec1" ;;
    *) no "regression: net-current as npuc-net 2: $out" ;;
  esac
fi
echo "check-ops: $pass PASS, $fail FAIL ($(basename "$ROOT"), outputs in $O)"
[ $fail = 0 ]
