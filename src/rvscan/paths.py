"""Project paths. Everything is relative to the repository root so a fresh clone works anywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"                 # untouched downloads + manifest.json (git-ignored)
PROCESSED = DATA / "processed"     # parsed per-instrument parquet files + catalog.json (git-ignored)
RESULTS = ROOT / "results"         # experiment outputs (summary tables are committed, caches are not)
REPORTS = ROOT / "reports"         # research report + figures (committed)
DEMO = ROOT / "demo"               # static web demo (source)
DEMO_DATA = DEMO / "data"          # demo JSON built by scripts/build_demo.py


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
