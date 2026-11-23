"""Download and process every instrument in config/universe.yaml.

    python scripts/fetch_data.py                 # use the cache where fresh, fetch the rest
    python scripts/fetch_data.py --refresh       # re-download everything
    python scripts/fetch_data.py --only yahoo    # one source, asset class or instrument id (repeatable)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvscan.data import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--only", action="append")
    ap.add_argument("--skip-snapshots", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = build.run(refresh=args.refresh, only=args.only, skip_snapshots=args.skip_snapshots)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
