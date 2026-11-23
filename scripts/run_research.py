"""Run the registered experiments (results/), e.g.

    python scripts/run_research.py                # all
    python scripts/run_research.py --only e1 e2   # a subset
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvscan.research import experiments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = experiments.run(args.only)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
