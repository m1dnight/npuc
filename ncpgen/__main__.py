"""python3 -m ncpgen <command> [arguments]

  gen | check | corpus                   the generator (ncpgen.py)
  lower | ref | layout | to-npu | from-npu | compare | example | example-residual | variants
                                         models in, the CPU reference, the NPU's formats (npunet.py)
  npunet | stream | frame | weights | grelu <command> ...
                                         a module's own commands (its checks, listings and selftests)
  frame-sweep | stream-graphs <command> ...   the sweeps behind frame.py and the graph ops

Each module also runs as a script: python3 -I ncpgen/<module>.py <command> ... ("-h" for its usage).
"""
import importlib
import sys

if not __package__:     # run as a script (python3 ncpgen/__main__.py)
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = {"npunet": "npunet", "stream": "stream", "frame": "frame", "weights": "weights", "grelu": "grelu",
           "frame-sweep": "frame_sweep", "stream-graphs": "stream_graphs"}
NPUNET = ("lower", "ref", "layout", "to-npu", "from-npu", "compare", "example", "example-residual", "variants")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if argv[0] in ("gen", "check", "corpus"):
        return importlib.import_module("ncpgen.ncpgen").main(argv)
    if argv[0] in NPUNET:
        return importlib.import_module("ncpgen.npunet").main(argv)
    if argv[0] in MODULES:
        return importlib.import_module("ncpgen." + MODULES[argv[0]]).main(argv[1:])
    print("ncpgen: unknown command %r\n%s" % (argv[0], __doc__), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
