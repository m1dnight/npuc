# Shared paths of the oracle scripts (sourced, not run).
#   NCPGEN       this repository
#   NCPGEN_WORK  the work folder (default <repo>/work, git-ignored): everything vendor-derived or generated lives there,
#                never in the repository. Layout:
#                  vendor-libs/      the NPUC libraries (libnpuc_*.so) and the other stock libraries studied
#                  bionic/           the stock Bionic runtime (linker64, libc, libm, libdl, liblog, ld-android, libc++)
#                  n3/root-<sha8>/   a run directory made by harness/build.sh (Bionic + NPUC + our programs);
#                  n3/root-current   a link to the one the scripts use by default
#                  n3/runs/          one folder per NPUC run (run.sh)
#                  n3/ghidra/        the Ghidra project and decompiles (ghidra/run-ghidra.sh)
#                  sweep/            compiled variants (npuc-local.sh outputs)
#                  tools/            android-ndk-r27d/, ghidra_12.2_PUBLIC/, jdk-*/ (or set NDK, GHIDRA_HOME, JAVA_HOME)
#   NPU_ROOT     the NPUC run directory to use (default $NCPGEN_WORK/n3/root-current)
NCPGEN=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NCPGEN_WORK=${NCPGEN_WORK:-$NCPGEN/work}
ORACLE=$NCPGEN/oracle
NPU_ROOT=${NPU_ROOT:-$NCPGEN_WORK/n3/root-current}
