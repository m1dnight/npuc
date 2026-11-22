#!/usr/bin/env bash
# The GReLU sweep (PORT-32, docs/format/grelu.md): compiles variants with the NPUC oracle (npuc-local.sh) to map
# how NPUC 1.36.1.a fills the GReLU table (activation, quant mode, zero points, scales, output channels, shapes,
# chains, soc). Each variant lands in <out>/<name>/ (lowered.json, net.ncp, log.txt, ...); then
#   python3 -I ncpgen/grelu.py check <out>/*/
# compares grelu.py's model with the oracle's bytes.
#   ncpgen/grelu_sweep.sh [out dir (default $NCPGEN_WORK/sweep/grelu/v)] [name-regex] [-j N]
# Compiled NCPs and logs are vendor-generated: they stay under output/ (git-ignored).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../oracle/env.sh"
T=$ORACLE
OUT=${1:-$NCPGEN_WORK/sweep/grelu/v}
RE=${2:-.}
J=6
[ "${3:-}" = "-j" ] && J=${4:-6}
mkdir -p "$OUT"

# name | quant | allow-none (0/1) | npuc-variant arguments (after <out.json>)
VARIANTS=$(cat <<'EOF'
c1|1|0|--in 16,16,8 --conv 1,3
c2|1|0|--in 16,16,8 --conv 2,3
c3|1|0|--in 16,16,8 --conv 3,3
c4|1|0|--in 16,16,8 --conv 4,3
c5|1|0|--in 16,16,8 --conv 5,3
c6|1|0|--in 16,16,8 --conv 6,3
c7|1|0|--in 16,16,8 --conv 7,3
c8|1|0|--in 16,16,8 --conv 8,3
c9|1|0|--in 16,16,8 --conv 9,3
c10|1|0|--in 16,16,8 --conv 10,3
c12|1|0|--in 16,16,8 --conv 12,3
c13|1|0|--in 16,16,8 --conv 13,3
c15|1|0|--in 16,16,8 --conv 15,3
c16|1|0|--in 16,16,8 --conv 16,3
c17|1|0|--in 16,16,8 --conv 17,3
c20|1|0|--in 16,16,8 --conv 20,3
c24|1|0|--in 16,16,8 --conv 24,3
c31|1|0|--in 16,16,8 --conv 31,3
c32|1|0|--in 16,16,8 --conv 32,3
c33|1|0|--in 16,16,8 --conv 33,3
c48|1|0|--in 16,16,8 --conv 48,3
c64|1|0|--in 16,16,8 --conv 64,3
n1|1|1|--in 16,16,8 --conv 1,3,1,same,none
n2|1|1|--in 16,16,8 --conv 2,3,1,same,none
n3|1|1|--in 16,16,8 --conv 3,3,1,same,none
n4|1|1|--in 16,16,8 --conv 4,3,1,same,none
n8|1|1|--in 16,16,8 --conv 8,3,1,same,none
n12|1|1|--in 16,16,8 --conv 12,3,1,same,none
n16|1|1|--in 16,16,8 --conv 16,3,1,same,none
n17|1|1|--in 16,16,8 --conv 17,3,1,same,none
n32|1|1|--in 16,16,8 --conv 32,3,1,same,none
q3c3|3|0|--in 16,16,8 --conv 3,3
q3c8|3|0|--in 16,16,8 --conv 8,3
q3c16|3|0|--in 16,16,8 --conv 16,3
q3c32|3|0|--in 16,16,8 --conv 32,3
q3n8|3|1|--in 16,16,8 --conv 8,3,1,same,none
q3n16|3|1|--in 16,16,8 --conv 16,3,1,same,none
q3n5|3|1|--in 16,16,8 --conv 5,3,1,same,none
z-128|1|0|--in 16,16,8 --conv 8,3 --ozp -128
z-100|1|0|--in 16,16,8 --conv 8,3 --ozp -100
z-1|1|0|--in 16,16,8 --conv 8,3 --ozp -1
z5|1|0|--in 16,16,8 --conv 8,3 --ozp 5
z127|1|0|--in 16,16,8 --conv 8,3 --ozp 127
zn-77|1|1|--in 16,16,8 --conv 8,3,1,same,none --ozp -77
zn99|1|1|--in 16,16,8 --conv 16,3,1,same,none --ozp 99
zi-50|1|0|--in 16,16,8 --in-zp -50 --conv 8,3 --ozp 20
s1|1|0|--in 16,16,8 --conv 8,3 --oscale 0.01
s2|1|0|--in 16,16,8 --conv 8,3 --oscale 0.0123
s3|1|0|--in 16,16,8 --conv 8,3 --wscale 0.003
s4|1|0|--in 16,16,8 --conv 8,3 --oscale 0.5
s5|1|0|--in 16,16,8 --conv 8,3 --oscale 0.00001
s6|1|0|--in 16,16,8 --in-scale 0.02 --conv 8,3 --wscale 0.0071 --oscale 0.0009
s7|1|0|--in 16,16,8 --conv 16,3 --wscale 0.0377 --oscale 0.0031
s8|1|1|--in 16,16,8 --conv 8,3,1,same,none --oscale 0.0123
s9|3|0|--in 16,16,8 --conv 8,3 --oscale 0.0123
s10|3|1|--in 16,16,8 --conv 16,3,1,same,none --wscale 0.0071 --oscale 0.0009
h8|1|0|--in 8,8,8 --conv 8,3
h4|1|0|--in 4,4,8 --conv 8,3
h32|1|0|--in 32,32,8 --conv 8,3
h64|1|0|--in 64,64,8 --conv 8,3
w64|1|0|--in 16,64,8 --conv 8,3
i3|1|0|--in 16,16,3 --conv 8,3
i16|1|0|--in 16,16,16 --conv 8,3
i32|1|0|--in 16,16,32 --conv 8,3
i64|1|0|--in 16,16,64 --conv 16,3
k1|1|0|--in 16,16,8 --conv 8,1
k5|1|0|--in 16,16,8 --conv 8,5
k7|1|0|--in 16,16,8 --conv 8,7
st2|1|0|--in 16,16,8 --conv 8,3,2
va|1|0|--in 16,16,8 --conv 8,3,1,valid
k13|1|0|--in 16,16,8 --conv 8,1 --kw 3
cc|1|0|--in 16,16,8 --conv 8,3 --conv 8,3
cpc|1|0|--in 16,16,8 --conv 8,3 --pool max,2 --conv 8,3
cpc16|1|0|--in 16,16,8 --conv 16,3 --oscale 0.01 --pool max,2 --conv 8,3 --oscale 0.02
cnr|1|1|--in 16,16,8 --conv 8,3,1,same,none --conv 16,3 --oscale 0.03 --ozp -5
c3l|1|1|--in 16,16,8 --conv 4,3 --conv 12,3,1,same,none --conv 8,1 --oscale 0.011
pac|1|0|--in 16,16,8 --pool avg,2 --conv 8,3
t128|1|0|--in 128,128,8 --conv 8,3
t96|1|0|--in 96,96,16 --conv 20,3
t112|1|1|--in 112,112,3 --conv 16,3,2,same,none
t64c|1|0|--in 64,64,32 --conv 40,3 --oscale 0.02
b1|1|0|--in 8,8,256 --conv 64,3 --wscale 0.0005
b2|1|0|--in 16,16,128 --conv 128,3 --wscale 0.0005
b3|1|0|--in 32,32,64 --conv 96,5 --wscale 0.0003
b4|1|0|--in 8,8,512 --conv 36,3 --wscale 0.0002
b5|1|0|--in 16,16,8 --conv 100,3
b6|1|0|--in 16,16,8 --conv 255,1
EOF
)

run_one() {
  local line=$1 name q an args d
  IFS='|' read -r name q an args <<<"$line"
  d=$OUT/$name
  mkdir -p "$d"
  # shellcheck disable=SC2086
  python3 -I "$T/npuc-variant.py" "$d/net.json" $args > "$d/variant.txt" 2>&1 || { echo "$name: variant FAIL"; return; }
  local extra=(--quant "$q"); [ "$an" = 1 ] && extra+=(--allow-none)
  if bash "$T/npuc-local.sh" "$d/net.json" "$d" "${extra[@]}" > "$d/compile.txt" 2>&1; then
    echo "$name: PASS"
  else
    echo "$name: FAIL ($(tail -n 1 "$d/compile.txt"))"
  fi
}
export -f run_one
export OUT T

printf '%s\n' "$VARIANTS" | awk -F'|' -v re="$RE" '$1 ~ re' | xargs -P "$J" -I{} bash -c 'run_one "$@"' _ {}

# soc 1: c8's and n8's net.npuc with "soc 1" (the NPU2020 device model instead of Neus)
for base in c8 n8 c16; do
  [[ "soc1-$base" =~ $RE ]] || continue
  [ -f "$OUT/$base/net.npuc" ] || continue
  d=$OUT/soc1-$base; mkdir -p "$d"
  sed 's/^soc 2 /soc 1 /; s/^soc 2$/soc 1/' "$OUT/$base/net.npuc" > "$d/src.npuc"
  cp "$OUT/$base/lowered.json" "$d/lowered.json"
  grep -q '^soc 1' "$d/src.npuc" || { echo "soc1-$base: no soc line rewritten"; continue; }
  if bash "$T/npuc-local.sh" "$d/src.npuc" "$d" > "$d/compile.txt" 2>&1; then echo "soc1-$base: PASS"; else echo "soc1-$base: FAIL"; fi
done

# standalone activation layers ("npuc-net 2" relu lines; needs a compiler root whose npuc-net reads them, e.g. the
# harness-ops build): GRELU_ROOT2=<root> ncpgen/grelu_sweep.sh <out> '^r'
# Variants land in <out>/../r/<name>/ with a lowered.json that adds {"kind": "relu", "type": ..., "cout": ...}.
if [ -n "${GRELU_ROOT2:-}" ]; then
  R=$(dirname "$OUT")/r
  for spec in base8:c8:1: rlc8:c8:0:ReLU r6c8:c8:0:ReLU6 r6c16:c16:0:ReLU6 r6q3:q3c8:0:ReLU6; do
    IFS=: read -r name src act typ <<<"$spec"
    [[ "$name" =~ $RE ]] || continue
    [ -f "$OUT/$src/net.npuc" ] || { echo "$name: compile $src first"; continue; }
    d=$R/$name; mkdir -p "$d"
    { sed -e 's/^npuc-net 1$/npuc-net 2/' -e "s/^\(conv l1_conv1 data0 [0-9]* 3 3 1 1 SAME\) [01] /\1 $act /" \
        "$OUT/$src/net.npuc" | grep -v '^end$'
      [ -n "$typ" ] && echo "relu r1 l1_conv1 $typ"
      echo end; } > "$d/src.npuc"
    python3 -I - "$OUT/$src/lowered.json" "$d/lowered.json" "$act" "$typ" <<'PY'
import json, sys
low = json.load(open(sys.argv[1]))
c = low["layers"][0]
c["act"] = int(sys.argv[3])
if sys.argv[4]:
    low["layers"].append({"kind": "relu", "name": "r1", "input": c["name"], "type": sys.argv[4], "cout": c["cout"]})
json.dump(low, open(sys.argv[2], "w"), indent=1)
PY
    if NPU_ROOT=$GRELU_ROOT2 bash "$T/npuc-local.sh" "$d/src.npuc" "$d" > "$d/compile.txt" 2>&1; then
      echo "$name: PASS"; else echo "$name: FAIL ($(tail -n 1 "$d/compile.txt"))"; fi
  done
fi
