"""HTTP fetching with an on-disk raw cache and a provenance manifest.

Every download is stored untouched under data/raw/<source>/<name> and recorded in data/raw/manifest.json with
its URL, SHA-256, size, fetch time and licence tag. Parsers only ever read from the cache, so a parse can be
re-run without touching the network and every processed number can be traced to a raw file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .. import paths

USER_AGENT = "Mozilla/5.0 (compatible; cross-asset-rv-scanner/0.1; research; +https://github.com/RasmusCodes/cross-asset-rv-scanner)"

LICENSES = {
    "public-domain": "US federal government work (no copyright).",
    "open-attribution": "Reuse permitted with attribution (central bank / OECD / open-data terms).",
    "vendor": "Third-party data; may not be redistributed row by row. Derived analytics only.",
}


class FetchError(RuntimeError):
    pass


class Fetcher:
    """Cached, throttled GET. One instance per run; thread-safe manifest writes."""

    def __init__(self, raw_dir: Path = paths.RAW, refresh: bool = False, min_interval: float = 0.35,
                 timeout: float = 60.0, retries: int = 3):
        self.raw_dir = Path(raw_dir)
        self.refresh = refresh
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()
        self.manifest_path = self.raw_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text("utf-8")) if self.manifest_path.exists() else {}

    # ------------------------------------------------------------------ helpers
    def path_for(self, source: str, name: str) -> Path:
        return self.raw_dir / source / name

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        with self._lock:
            wait = self._last_hit.get(host, 0) + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            self._last_hit[host] = time.time()

    def _record(self, rel: str, url: str, blob: bytes, license: str, source: str) -> None:
        entry = {
            "url": url,
            "source": source,
            "license": license,
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self.manifest[rel] = entry
            tmp = self.manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.manifest, indent=1, sort_keys=True), "utf-8")
            os.replace(tmp, self.manifest_path)

    # ------------------------------------------------------------------ public
    def get(self, url: str, source: str, name: str, license: str, max_age_days: float | None = None,
            params: dict | None = None, headers: dict | None = None, accept_status: tuple = (200,),
            timeout: float | None = None, retries: int | None = None) -> bytes:
        """Return the raw bytes for `url`, from cache when fresh enough."""
        if license not in LICENSES:
            raise ValueError(f"unknown licence tag {license!r}")
        path = self.path_for(source, name)
        if path.exists() and not self.refresh:
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if max_age_days is None or age_days <= max_age_days:
                return path.read_bytes()
        last_exc: Exception | None = None
        for attempt in range(retries or self.retries):
            self._throttle(url)
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=timeout or self.timeout)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    last_exc = FetchError(f"429 from {url}")
                    continue
                if r.status_code not in accept_status:
                    raise FetchError(f"HTTP {r.status_code} for {r.url}")
                blob = r.content
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(blob)
                os.replace(tmp, path)
                full_url = r.url
                self._record(path.relative_to(self.raw_dir).as_posix(), full_url, blob, license, source)
                return blob
            except (requests.RequestException, FetchError) as exc:
                last_exc = exc
                if isinstance(exc, FetchError) and "HTTP 4" in str(exc) and "429" not in str(exc):
                    break  # client errors other than rate limits do not improve with retries
                time.sleep(1.5 * (attempt + 1))
        if path.exists():  # stale cache beats nothing; caller sees the file's age in the manifest
            return path.read_bytes()
        raise FetchError(f"failed to fetch {url}: {last_exc}")
