"""Source adapters: download via the cached Fetcher, parse into tidy pandas objects.

Each adapter returns a DataFrame indexed by local trading date (DatetimeIndex, no tz) with float columns.
Licence tags (see cache.LICENSES) are attached by the caller from config/universe.yaml.

Known source facts (checked 2026-11-21):
- Yahoo chart API: `range=max` silently downsamples to monthly bars; use period1=0&period2=now for daily history.
  `close` is split-adjusted, `adjclose` split- and dividend-adjusted. Timestamps are session starts in UTC and must
  be converted to the exchange time zone before taking the date. Expired futures contracts return 404.
- EIA stopped publishing NYMEX futures contract 1-4 prices on 2024-04-05; spot series continue.
- NY Fed rates API: SOFR/TGCR/BGCR from 2018-04-02; use FRED for EFFR history.
- iShares fund-data downloads now return an HTML page; SSGA NAV histories still download as xlsx.
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd

from .cache import Fetcher

DAY = 86400


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({".": np.nan, "-": np.nan, "": np.nan, "ND": np.nan}), errors="coerce")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.isna()]
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


# ---------------------------------------------------------------------------------------------------- Yahoo
def yahoo_chart(f: Fetcher, symbol: str, max_age_days: float = 0.5) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Daily OHLCV + adjclose for a Yahoo symbol. Returns (bars, meta, events)."""
    period2 = int(time.time() // DAY * DAY + DAY)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=0&period2={period2}&interval=1d"
           f"&events=div%2Csplit&includeAdjustedClose=true")
    name = symbol.replace("^", "_").replace("=", "_") + ".json"
    blob = f.get(url, "yahoo", name, "vendor", max_age_days=max_age_days, accept_status=(200, 404))
    payload = json.loads(blob)
    result = (payload.get("chart") or {}).get("result")
    if not result:
        return pd.DataFrame(), {}, pd.DataFrame()
    res = result[0]
    meta = res.get("meta", {})
    ts = res.get("timestamp") or []
    if not ts:
        return pd.DataFrame(), meta, pd.DataFrame()
    tz = meta.get("exchangeTimezoneName") or "UTC"
    dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")}, index=dates)
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df["adjclose"] = adj if adj is not None else np.nan
    df = df.astype(float)
    df = df[df["close"].notna() & (df["close"] > 0)]
    # Yahoo leaves the latest session's adjclose empty at times; carry the last adjustment ratio forward.
    ratio = (df["adjclose"] / df["close"]).ffill().fillna(1.0)
    df["adjclose"] = df["adjclose"].fillna(df["close"] * ratio)
    events = []
    for kind in ("dividends", "splits"):
        for item in (res.get("events", {}).get(kind) or {}).values():
            d = pd.to_datetime(item["date"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
            if kind == "dividends":
                events.append({"date": d, "kind": "div", "value": float(item["amount"])})
            else:
                events.append({"date": d, "kind": "split", "value": float(item["numerator"]) / float(item["denominator"])})
    ev = pd.DataFrame(events, columns=["date", "kind", "value"])
    meta_out = {k: meta.get(k) for k in ("currency", "symbol", "exchangeName", "fullExchangeName", "instrumentType",
                                         "exchangeTimezoneName", "gmtoffset", "longName", "shortName")}
    return _finish(df), meta_out, ev


# ---------------------------------------------------------------------------------------------------- FRED
def fred(f: Fetcher, series_id: str, license: str, max_age_days: float = 1.0) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    blob = f.get(url, "fred", f"{series_id}.csv", license, max_age_days=max_age_days)
    df = pd.read_csv(io.BytesIO(blob))
    date_col = df.columns[0]
    out = pd.DataFrame({"value": _to_float(df[series_id].astype(str)).to_numpy()}, index=pd.to_datetime(df[date_col]).to_numpy())
    return _finish(out.dropna())


# ---------------------------------------------------------------------------------------------------- US Treasury
TREASURY_TENORS = {"1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25, "4 Mo": 4 / 12, "6 Mo": 0.5,
                   "1 Yr": 1, "2 Yr": 2, "3 Yr": 3, "5 Yr": 5, "7 Yr": 7, "10 Yr": 10, "20 Yr": 20, "30 Yr": 30,
                   "5 YR": 5, "7 YR": 7, "10 YR": 10, "20 YR": 20, "30 YR": 30}


def treasury_curve(f: Fetcher, kind: str = "nominal", first_year: int = 1990) -> pd.DataFrame:
    """Daily Treasury par yield curve (nominal) or real yield curve (TIPS), columns = maturity in years."""
    typ = "daily_treasury_yield_curve" if kind == "nominal" else "daily_treasury_real_yield_curve"
    if kind != "nominal":
        first_year = max(first_year, 2003)
    this_year = pd.Timestamp.today().year
    frames = []
    for year in range(first_year, this_year + 1):
        url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
               f"{year}/all?type={typ}&field_tdr_date_value={year}&page&_format=csv")
        max_age = 1.0 if year >= this_year - 0 else None  # past years never change
        blob = f.get(url, "treasury", f"{typ}_{year}.csv", "public-domain", max_age_days=max_age, timeout=150, retries=5)
        if not blob.strip():
            continue
        df = pd.read_csv(io.BytesIO(blob))
        df.index = pd.to_datetime(df.pop("Date"), format="%m/%d/%Y")
        df = df.rename(columns={c: TREASURY_TENORS[c] for c in df.columns if c in TREASURY_TENORS})
        df = df[[c for c in df.columns if isinstance(c, float) or isinstance(c, int)]]
        frames.append(df.apply(_to_float))
    out = pd.concat(frames).sort_index()
    out.columns = [float(c) for c in out.columns]
    out = out.T.groupby(level=0).first().T  # merge duplicate tenor labels across years
    return _finish(out)


# ---------------------------------------------------------------------------------------------------- EIA
def eia_series(f: Fetcher, code: str, area: str = "pet") -> pd.DataFrame:
    url = f"https://www.eia.gov/dnav/{area}/hist_xls/{code}d.xls"
    blob = f.get(url, "eia", f"{code}d.xls", "public-domain", max_age_days=1.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(io.BytesIO(blob), sheet_name="Data 1", skiprows=2, engine="xlrd")
    out = pd.DataFrame({"value": _to_float(df.iloc[:, 1]).to_numpy()}, index=pd.to_datetime(df.iloc[:, 0]).to_numpy())
    return _finish(out.dropna())


# ---------------------------------------------------------------------------------------------------- ECB
def ecb_series(f: Fetcher, key: str, start: str = "1990-01-01") -> pd.DataFrame:
    flow, rest = key.split("/", 1)
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{rest}?format=csvdata&startPeriod={start}"
    blob = f.get(url, "ecb", key.replace("/", "_") + ".csv", "open-attribution", max_age_days=1.0)
    if not blob.strip():
        return pd.DataFrame(columns=["value"])
    df = pd.read_csv(io.BytesIO(blob))
    idx = pd.PeriodIndex(df["TIME_PERIOD"], freq="M").to_timestamp(how="end").normalize() if len(str(df["TIME_PERIOD"].iloc[0])) == 7 \
        else pd.to_datetime(df["TIME_PERIOD"])
    out = pd.DataFrame({"value": _to_float(df["OBS_VALUE"].astype(str)).to_numpy()}, index=pd.DatetimeIndex(idx))
    return _finish(out.dropna())


# ---------------------------------------------------------------------------------------------------- Bundesbank
def bundesbank_series(f: Fetcher, key: str) -> pd.DataFrame:
    url = f"https://api.statistiken.bundesbank.de/rest/download/{key.replace('.', '/', 1)}?format=csv&lang=en"
    blob = f.get(url, "bundesbank", key + ".csv", "open-attribution", max_age_days=1.0)
    lines = blob.decode("utf-8-sig", "replace").splitlines()
    rows = [ln.split(",") for ln in lines if len(ln) >= 10 and ln[:4].isdigit() and ln[4] == "-"]
    out = pd.DataFrame({"value": _to_float(pd.Series([r[1] for r in rows])).to_numpy()},
                       index=pd.to_datetime([r[0] for r in rows]))
    return _finish(out.dropna())


# ---------------------------------------------------------------------------------------------------- Japan MoF
def mof_jgb(f: Fetcher) -> pd.DataFrame:
    base = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
    frames = []
    for url, name, age in ((base + "historical/jgbcme_all.csv", "jgbcme_all.csv", 7.0), (base + "jgbcme.csv", "jgbcme.csv", 1.0)):
        blob = f.get(url, "mof", name, "open-attribution", max_age_days=age)
        df = pd.read_csv(io.BytesIO(blob), skiprows=1, encoding="cp932")
        df.index = pd.to_datetime(df.pop("Date"), format="%Y/%m/%d", errors="coerce")
        df.columns = [float(c.rstrip("Y")) for c in df.columns]
        frames.append(df.apply(lambda s: _to_float(s.astype(str))))
    out = pd.concat(frames)
    return _finish(out)


# ---------------------------------------------------------------------------------------------------- Bank of Canada
def boc_valet(f: Fetcher, series: list[str], name: str, start: str = "1990-01-01") -> pd.DataFrame:
    url = f"https://www.bankofcanada.ca/valet/observations/{','.join(series)}/json?start_date={start}"
    blob = f.get(url, "boc", name + ".json", "open-attribution", max_age_days=1.0)
    obs = json.loads(blob).get("observations", [])
    rows = {o["d"]: {s: (o.get(s) or {}).get("v") for s in series} for o in obs}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    return _finish(df.apply(lambda s: _to_float(s.astype(str))))


# ---------------------------------------------------------------------------------------------------- RBA
def rba_f2(f: Fetcher) -> pd.DataFrame:
    url = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"   # daily (f2.1 is monthly)
    blob = f.get(url, "rba", "f2-data.csv", "open-attribution", max_age_days=1.0)
    lines = blob.decode("utf-8-sig", "replace").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("Series ID"))
    ids = lines[head].split(",")[1:]
    rows = [ln.split(",") for ln in lines[head + 1:] if ln[:2].strip().isdigit()]
    df = pd.DataFrame([r[1:len(ids) + 1] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%d-%b-%Y"), columns=ids)
    return _finish(df.apply(_to_float))


# ---------------------------------------------------------------------------------------------------- Bank of England
def boe_series(f: Fetcher, codes: list[str], name: str, start: str = "01/Jan/1990") -> pd.DataFrame:
    url = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           f"&Datefrom={start}&Dateto=now&SeriesCodes={','.join(codes)}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    blob = f.get(url, "boe", name + ".csv", "open-attribution", max_age_days=1.0)
    df = pd.read_csv(io.BytesIO(blob))
    df.index = pd.to_datetime(df.pop("DATE"), format="%d %b %Y")
    return _finish(df.apply(lambda s: _to_float(s.astype(str))))


# ---------------------------------------------------------------------------------------------------- NY Fed
def nyfed_rate(f: Fetcher, path: str, start: str = "2018-04-02") -> pd.DataFrame:
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    url = f"https://markets.newyorkfed.org/api/rates/{path}/search.json?startDate={start}&endDate={end}"
    blob = f.get(url, "nyfed", path.replace("/", "_") + ".json", "open-attribution", max_age_days=1.0)
    rows = json.loads(blob).get("refRates", [])
    out = pd.DataFrame({"value": [r.get("percentRate") for r in rows]}, index=pd.to_datetime([r["effectiveDate"] for r in rows]))
    return _finish(out.dropna())


# ---------------------------------------------------------------------------------------------------- SSGA NAV
def ssga_nav(f: Fetcher, ticker: str) -> pd.DataFrame:
    url = f"https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/navhist-us-en-{ticker.lower()}.xlsx"
    blob = f.get(url, "ssga", f"navhist-{ticker.lower()}.xlsx", "vendor", max_age_days=1.0)
    raw = pd.read_excel(io.BytesIO(blob), header=None)
    head = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")][0]
    body = raw.iloc[head + 1:, :4]
    body.columns = ["date", "nav", "shares", "tna"]
    body = body[pd.to_datetime(body["date"], format="%d-%b-%Y", errors="coerce").notna()]
    body.index = pd.to_datetime(body.pop("date"), format="%d-%b-%Y")
    return _finish(body.apply(pd.to_numeric, errors="coerce"))
