#!/usr/bin/env bash
# Record tests/golden.txt from compiled sweeps in the work folder (needs NPUC's compiles: oracle/npuc-local.sh outputs in
# $NCPGEN_WORK/sweep/corpus/<name>/ and $NCPGEN_WORK/sweep/final-chains/<name>/ with specs.txt, and the zoo's models
# that ncpgen generates exactly: models/zoo.py compile, then check, in $NCPGEN_WORK/sweep/zoo/<model>/), and npunet's stride
# rewrite networks (tests/rewrite_ref.py --write, compiled in $NCPGEN_WORK/sweep/restride/<case>/ with specs.txt). Hashes
# only: no vendor byte goes into the repository.
set -euo pipefail
. "$(dirname "$0")/../oracle/env.sh"
cd "$NCPGEN" && python3 -I - "$NCPGEN_WORK" <<'PY'
import hashlib, os, sys
sys.path.insert(0, '.')
from ncpgen.ncpgen import CORPUS
W = sys.argv[1]; rows = []
for name, spec, opts in CORPUS:
    d = os.path.join(W, 'sweep/corpus', name, 'net.ncp')
    if os.path.exists(d): rows.append((name, spec, opts, hashlib.sha256(open(d, 'rb').read()).hexdigest()))
for line in open(os.path.join(W, 'sweep/final-chains/specs.txt')):
    name, spec, opts = line.rstrip('\n').split('|')
    d = os.path.join(W, 'sweep/final-chains', name, 'net.ncp')
    if os.path.exists(d): rows.append(('final-' + name, spec, opts, hashlib.sha256(open(d, 'rb').read()).hexdigest()))
sys.path.insert(0, 'models'); import zoo
rows += zoo.golden_rows(W)
sp = os.path.join(W, 'sweep/restride/specs.txt')
for line in open(sp) if os.path.exists(sp) else []:
    name, opts = line.rstrip('\n').split('|')
    d = os.path.join(W, 'sweep/restride', name, 'net.ncp')
    if os.path.exists(d): rows.append(('restride-' + name, 'restride:' + name, opts, hashlib.sha256(open(d, 'rb').read()).hexdigest()))
with open('tests/golden.txt', 'w') as f:
    f.write("# name | npuc-variant.py arguments, zoo:<model> (models/zoo.py) or restride:<case> (tests/rewrite_ref.py) | "
            "lowering options | sha256 of the NCP Samsung's "
            "NPUC 1.36.1.a compiled for it\n")
    f.write('# (our own seeded networks; ncpgen gen --grelu npuc must write the same bytes). Made by tests/make-golden.sh.\n')
    for r in rows: f.write('|'.join(r) + '\n')
print(len(rows), 'golden networks')
PY
