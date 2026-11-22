#!/usr/bin/env bash
# Record tests/golden.txt from compiled sweeps in the work folder (needs NPUC's compiles: oracle/npuc-local.sh outputs in
# $NCPGEN_WORK/sweep/corpus/<name>/ and $NCPGEN_WORK/sweep/final-chains/<name>/ with specs.txt). Hashes only: no vendor
# byte goes into the repository.
set -euo pipefail
. "$(dirname "$0")/../oracle/env.sh"
cd "$NCPGEN" && python3 -I - "$NCPGEN_WORK" <<'PY'
import hashlib, os, sys
sys.path.insert(0, 'ncpgen'); import ncpgen
W = sys.argv[1]; rows = []
for name, spec, opts in ncpgen.CORPUS:
    d = os.path.join(W, 'sweep/corpus', name, 'net.ncp')
    if os.path.exists(d): rows.append((name, spec, opts, hashlib.sha256(open(d, 'rb').read()).hexdigest()))
for line in open(os.path.join(W, 'sweep/final-chains/specs.txt')):
    name, spec, opts = line.rstrip('\n').split('|')
    d = os.path.join(W, 'sweep/final-chains', name, 'net.ncp')
    if os.path.exists(d): rows.append(('final-' + name, spec, opts, hashlib.sha256(open(d, 'rb').read()).hexdigest()))
with open('tests/golden.txt', 'w') as f:
    f.write("# name | npuc-variant.py arguments | lowering options | sha256 of the NCP Samsung's NPUC 1.36.1.a compiled for it\n")
    f.write('# (our own seeded networks; ncpgen gen --grelu npuc must write the same bytes). Made by tests/make-golden.sh.\n')
    for r in rows: f.write('|'.join(r) + '\n')
print(len(rows), 'golden networks')
PY
