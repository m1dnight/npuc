#!/usr/bin/env bash
# Ghidra headless on the NPUC libraries (and nn-eden-drv, libeden_rt). Host only.
#   run-ghidra.sh import                 import + auto-analyse the binaries below into $NCPGEN_WORK/n3/ghidra/proj
#   run-ghidra.sh decompile <targets>    DecompileTargets.java over every imported binary
#                                        -> $NCPGEN_WORK/n3/ghidra/decomp/<targets>/<binary>/ (decompiled vendor code:
#                                        stays in the work folder)
# GHIDRA_HOME (default $NCPGEN_WORK/tools/ghidra_12.2_PUBLIC), JAVA_HOME (default: a JDK in $NCPGEN_WORK/tools).
# One Ghidra process at a time (the project locks). BINS_OVERRIDE="a.so b.so" limits the binaries.
set -euo pipefail
. "$(dirname "$0")/../env.sh"
V=$NCPGEN_WORK/vendor-libs; G=$NCPGEN_WORK/n3/ghidra; TL=$NCPGEN_WORK/tools
# a JDK in the work folder's tools (a macOS one has Contents/Home) unless JAVA_HOME names a
# working one; first on PATH too, since Ghidra's launcher also runs the java it finds there
if [ -z "${JAVA_HOME:-}" ] || ! "$JAVA_HOME/bin/java" -version > /dev/null 2>&1; then
  JAVA_HOME=$(ls -d "$TL"/jdk-*/Contents/Home 2> /dev/null | head -n 1)
  [ -n "$JAVA_HOME" ] || JAVA_HOME=$(ls -d "$TL"/jdk-* 2> /dev/null | head -n 1)
fi
export JAVA_HOME PATH=$JAVA_HOME/bin:$PATH
AH=${GHIDRA_HOME:-$TL/ghidra_12.2_PUBLIC}/support/analyzeHeadless
HERE=$(cd "$(dirname "$0")" && pwd)
BINS=(${BINS_OVERRIDE:-nn-eden-drv libnpuc_frontend.so libnpuc_common.so libnpuc_controller.so libnpuc_api.so libnpuc_graph.so libeden_rt.so libnpuc_backend.so libnpuc_template.so})
mkdir -p "$G/proj" "$G/decomp"
case "${1:-}" in
  import)
    for b in "${BINS[@]}"; do
      [ -e "$G/proj/npuc.rep/idata" ] && grep -rqs "$b" "$G/proj/npuc.rep/idata/~index.dat" && { echo "$b: already imported"; continue; }
      "$AH" "$G/proj" npuc -import "$V/$b" -overwrite -analysisTimeoutPerFile 3600 > "$G/import-$b.log" 2>&1
      grep -E 'REPORT|ERROR' "$G/import-$b.log" | tail -3
    done ;;
  decompile)
    T=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
    for b in "${BINS[@]}"; do
      "$AH" "$G/proj" npuc -process "$b" -noanalysis -scriptPath "$HERE" \
        -postScript DecompileTargets.java "$T" "$G/decomp/$(basename "$T" .txt)" > "$G/decompile-$b.log" 2>&1 || true
      grep -E 'DecompileTargets:|ERROR' "$G/decompile-$b.log" | tail -2
    done ;;
  *) sed -n '2,6p' "$0"; exit 1 ;;
esac
