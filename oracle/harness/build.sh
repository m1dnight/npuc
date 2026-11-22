#!/usr/bin/env bash
# Build our NPUC harness programs with the Android NDK (r27d) for arm64 Bionic and assemble an NPUC run directory: the
# stock Bionic runtime ($NCPGEN_WORK/bionic/), the eight NPUC libraries ($NCPGEN_WORK/vendor-libs/libnpuc_*.so) and
# the programs, flat, with SHA256SUMS. Host only. Everything it writes is in the work folder (vendor and stock files
# never go into a repository).
#   npuc-probe  (npuc-probe.c)   dlopen()s NPUC and resolves its entry points
#   npuc-conv   (npuc-conv.cpp)  one convolution through npuc.h's reconstructed objects
#   npuc-net    (npuc-net.cpp)   a network read from a file (ncpgen/npunet.py lower): conv, pool, eltwise, concat, relu
# out: $NCPGEN_WORK/n3/<prog>-<sha8>     each binary, named by its sha256 (never overwritten)
#      $NCPGEN_WORK/n3/root-<sha8>/      the run directory (+ root-<sha8>.symbols.txt); n3/root-current -> it
# Reproducible from any path: the repository is mapped to "." and the NDK to "/ndk" in the debug info, so the same
# NDK gives the same bytes on any host (the NDK's Linux toolchain: run it on x86-64 Linux, or in an amd64 container).
#   NDK=<android-ndk-r27d> oracle/harness/build.sh       (default $NCPGEN_WORK/tools/android-ndk-r27d)
set -euo pipefail
. "$(dirname "$0")/../env.sh"
O=$NCPGEN_WORK; S=$ORACLE/harness
NDK=${NDK:-$NCPGEN_WORK/tools/android-ndk-r27d}
# its real path: clang records its resource directory by the resolved path, which the /ndk map must cover
NDK=$(cd "$NDK" && pwd -P)
API=30
TB=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
CC=$TB/aarch64-linux-android$API-clang; CXX=$TB/aarch64-linux-android$API-clang++; RE=$TB/llvm-readelf
grep -q '^Pkg.ReleaseName = r27d$' "$NDK/source.properties" || { echo "$NDK is not NDK r27d"; exit 1; }
(cd "$O/bionic" && sha256sum -c --quiet SHA256SUMS) || { echo "$O/bionic does not match its SHA256SUMS"; exit 1; }
B=$O/n3/build; mkdir -p "$B"
cd "$S"   # the compile directory goes into the debug info: fixed, so the bytes do not depend on the caller
# Two steps with fixed paths and prefix maps, so the same source gives the same bytes
"$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -fPIE -ffile-prefix-map="$NCPGEN"=. -ffile-prefix-map="$NDK"=/ndk -c -o "$B/npuc-probe.o" "$S/npuc-probe.c"
"$CC" -pie -Wl,--export-dynamic -Wl,--build-id=sha1 -o "$B/npuc-probe" "$B/npuc-probe.o" -ldl
# npuc-conv: npuc.h's static_asserts check the layouts against the NDK's libc++ headers at this step. It links
# against the vendor libraries' symbol tables (no byte of them goes into the binary) and uses NPUC's own exported
# libc++ (-nostdlib++): one C++ runtime in the process, NPUC's. Our code stays hidden except __android_log_print.
"$CXX" -std=c++17 -O2 -g -Wall -Wextra -Werror -fPIE -fvisibility=hidden -fvisibility-inlines-hidden \
  -ffile-prefix-map="$NCPGEN"=. -ffile-prefix-map="$NDK"=/ndk -c -o "$B/npuc-conv.o" "$S/npuc-conv.cpp"
"$CXX" -pie -nostdlib++ -Wl,--build-id=sha1 -o "$B/npuc-conv" "$B/npuc-conv.o" \
  -L"$O/vendor-libs" -lnpuc_frontend -lnpuc_api -lnpuc_common -ldl
"$CXX" -std=c++17 -O2 -g -Wall -Wextra -Werror -fPIE -fvisibility=hidden -fvisibility-inlines-hidden \
  -ffile-prefix-map="$NCPGEN"=. -ffile-prefix-map="$NDK"=/ndk -c -o "$B/npuc-net.o" "$S/npuc-net.cpp"
"$CXX" -pie -nostdlib++ -Wl,--build-id=sha1 -o "$B/npuc-net" "$B/npuc-net.o" \
  -L"$O/vendor-libs" -lnpuc_frontend -lnpuc_api -lnpuc_common -ldl
for p in npuc-probe npuc-conv npuc-net; do
  "$RE" --dyn-syms "$B/$p" | grep -E ' __android_log_print$' | grep -q DEFAULT \
    || { echo "$p does not export __android_log_print"; exit 1; }
done
declare -A OUT
for p in npuc-probe npuc-conv npuc-net; do
  h=$(sha256sum "$B/$p" | cut -c1-8); out=$O/n3/$p-$h
  [ -e "$out" ] || cp "$B/$p" "$out"
  cmp -s "$B/$p" "$out" || { echo "$out exists and differs"; exit 1; }
  (cd "$O/n3" && grep -q " $p-$h\$" SHA256SUMS 2>/dev/null || sha256sum "$p-$h" >> SHA256SUMS)
  OUT[$p]=$out
  echo "$p: $out ($(stat -c %s "$out") bytes, sha256 $(sha256sum < "$out" | cut -d' ' -f1))"
done
T=$(mktemp -d "$O/n3/root.XXXXXX"); trap 'rm -rf "$T"' EXIT
cp "$O"/bionic/{linker64,ld-android.so,libc.so,libm.so,libdl.so,liblog.so,libc++.so} "$T/"
cp "$O"/vendor-libs/libnpuc_{api,controller,frontend,graph,template,cmdq,backend,common}.so "$T/"
cp "${OUT[npuc-probe]}" "$T/npuc-probe"; cp "${OUT[npuc-conv]}" "$T/npuc-conv"; cp "${OUT[npuc-net]}" "$T/npuc-net"
chmod 755 "$T/linker64" "$T/npuc-probe" "$T/npuc-conv" "$T/npuc-net"
(cd "$T" && sha256sum linker64 npuc-probe npuc-conv npuc-net *.so > SHA256SUMS)
R=$O/n3/root-$(sha256sum < "$T/SHA256SUMS" | cut -c1-8)
if [ -e "$R" ]; then cmp -s "$T/SHA256SUMS" "$R/SHA256SUMS" || { echo "$R exists and differs"; exit 1; }
else mv "$T" "$R"; fi
(cd "$R" && sha256sum -c --quiet SHA256SUMS) || { echo "$R does not match its SHA256SUMS"; exit 1; }
ln -sfn "$(basename "$R")" "$O/n3/root-current"
# The RTLD_NOW load, checked statically on the host (no emulator): every import bound in the directory
# Bionic scopes: npuc-probe dlopen()s libnpuc_api.so (only), npuc-conv and npuc-net have the libraries in their DT_NEEDED
python3 -I "$S/check-symbols.py" "$RE" "$R" npuc-probe:libnpuc_api.so npuc-conv npuc-net > "$R.symbols.txt" || { cat "$R.symbols.txt"; exit 1; }
tail -1 "$R.symbols.txt"
# Self-test of the checker: the one-by-one dlopen order that failed on the phone (run 104734Z: libnpuc_graph,
# libnpuc_backend underlinked) must fail here too
if python3 -I "$S/check-symbols.py" "$RE" "$R" npuc-probe:libnpuc_common.so,libnpuc_graph.so,libnpuc_cmdq.so,libnpuc_backend.so,libnpuc_template.so,libnpuc_frontend.so,libnpuc_controller.so,libnpuc_api.so > /dev/null; then
  echo "check-symbols missed the underlinked libraries"; exit 1
fi
echo "check-symbols self-test: the one-by-one order fails, as on the phone"
echo "run directory: $R ($(ls "$R" | wc -l) files, $(du -sb "$R" | cut -f1) bytes)"
