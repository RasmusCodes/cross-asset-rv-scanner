"""Build reports/REPORT.md and reports/figures/ from results/ (run scripts/run_research.py first)."""
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvscan.research import report  # noqa: E402

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    print(report.build())
