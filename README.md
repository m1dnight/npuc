# NEPCGen

An open compiler for the NPU of Samsung's Exynos 990 (Galaxy S20 family). It takes quantized TFLite models and writes
**NCPs**, the binaries the NPU's firmware runs, byte for byte what Samsung's own compiler writes. Pure Python, no
dependencies, no Samsung code. Unofficial; not affiliated with Samsung.

## Requirements

- Python 3.8 or newer (standard library only).
- To run an NCP: an Exynos 990 phone, its NPU firmware and a driver (not included).
- To work on the compiler against Samsung's own compiler: Docker and files from your own phone's firmware
  (`docs/oracle.md`). Not needed to use ncpgen.

## Setup

```bash
git clone https://github.com/m1dnight/npuc ncpgen && cd ncpgen
tests/run.sh                 # the self-test, about 10 s; needs no vendor file
```

Nothing to install: run the commands from the repository's root.

## Example

Compile LeNet-5 from the built-in model zoo, then compute what the NPU must output for an input:

```bash
python3 models/zoo.py write example lenet5                        # example/lenet5.tflite (int8)
python3 -m ncpgen gen example/lenet5.tflite example/lenet5.ncp --allow-none

python3 -m ncpgen lower example/lenet5.tflite example/low --allow-none
head -c 784 /dev/urandom > example/input.bin                      # one 28x28x1 int8 image
python3 -m ncpgen ref example/low/lowered.json example/input.bin example/expected.bin
```

Your own int8/uint8 `.tflite` works the same way. `--allow-none` accepts layers without an activation (LeNet's
logits). By default ncpgen corrects a defect in NPUC's activation table; `--grelu npuc` writes NPUC's exact bytes.
`python3 -m ncpgen -h` lists every command.

## More

- What is supported, the status and the code layout: `docs/support.md`
- How common and camera models fare: `models/README.md`, `models/camera.md`
- The NCP format: `docs/format/`
- Working on the compiler (the oracle, the tests): `AGENTS.md`, `docs/oracle.md`

## Vendor files

Nothing from Samsung is in this repository: no library, firmware, compiled NCP or decompiled code. They stay in the
git-ignored work folder (`work/`). The work was done for interoperability (EU Directive 2009/24/EC, article 6); this
is not legal advice.

## Licence

To be decided before anything is published.
