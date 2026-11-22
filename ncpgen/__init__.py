"""ncpgen: an open compiler for the NPU of Samsung's Exynos 990. A quantized network in (TFLite, or npunet's JSON),
the NCP the NPU's firmware runs out, byte for byte what Samsung's NPUC 1.36.1.a writes (README.md, docs/format/).

Modules:
  ncpgen    the generator: generate(low) and the gen / check / corpus commands
  npunet    models in: TFLite or JSON lowered to NPUC's arithmetic (lowered.json), the exact CPU reference, the
            NPU's input and output formats
  stream    the CMDQ instruction streams (docs/format/cmdq.md)
  frame     the container, NPUC's plan and tiling policy, the intrinsics, the data layout (docs/format/frame.md)
  weights   the WEIGHT blocks (docs/format/weights.md)
  grelu     the activation table (docs/format/grelu.md)
  common    what they share: float32 arithmetic, topology, the instruction encoding, an NCP's sections

python3 -m ncpgen <command> runs them (ncpgen/__main__.py); each module also runs as a script (python3 -I
ncpgen/<module>.py ...).
"""
