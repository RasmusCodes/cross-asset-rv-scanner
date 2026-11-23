"""Build the web demo's data files (demo/data/) from the processed store and the experiment results.

    python scripts/build_demo.py                   # writes demo/data/
    python scripts/build_demo.py --out some/dir    # elsewhere (e.g. a website's apps/ folder, data only)

The licence policy is enforced inside rvscan.demo.export (see its docstring) and re-checked after writing.
Serve demo/ with any static server to try it: python -m http.server -d demo 8811
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvscan.demo import export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    print(json.dumps(export.build(Path(args.out) if args.out else None), indent=1))


if __name__ == "__main__":
    main()
