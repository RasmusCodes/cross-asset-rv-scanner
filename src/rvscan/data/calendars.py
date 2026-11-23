"""Futures expiry rules (NYMEX energy, CME equity index) and roll dates for continuous contract series.

The exchange's own trading calendar is taken from the dates on which the price series has observations, so
holidays need no separate table.

NYMEX rules (as published by CME):
- CL (WTI): trading terminates 3 business days before the 25th calendar day of the month prior to the contract
  month; if the 25th is not a business day, 3 business days before the last business day preceding the 25th.
- NG (Henry Hub): 3rd-last business day of the month prior to the contract month.
- RB (RBOB), HO (heating oil / ULSD): last business day of the month prior to the contract month.
- ES/NQ/RTY/YM: third Friday of March, June, September and December (09:30 ET special opening quotation).
EIA's "contract 1" switches to the next contract on the first trading day after expiry.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def _month_days(tdays: pd.DatetimeIndex, year: int, month: int) -> pd.DatetimeIndex:
    return tdays[(tdays.year == year) & (tdays.month == month)]


def expiry(root: str, contract_year: int, contract_month: int, tdays: pd.DatetimeIndex) -> pd.Timestamp | None:
    py, pm = (contract_year, contract_month - 1) if contract_month > 1 else (contract_year - 1, 12)
    prior = _month_days(tdays, py, pm)
    if len(prior) == 0:
        return None
    if root == "CL":
        ref = pd.Timestamp(py, pm, 25)
        before = tdays[tdays < ref]
        if ref in tdays:
            i = tdays.get_loc(ref)
            return tdays[i - 3] if i >= 3 else None
        if len(before) < 4:
            return None
        return before[-4]
    if root == "NG":
        return prior[-3] if len(prior) >= 3 else None
    if root in ("RB", "HO"):
        return prior[-1]
    raise ValueError(root)


def roll_days(root: str, tdays: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day after each expiry (the day the continuous 'contract 1' series switches)."""
    tdays = pd.DatetimeIndex(sorted(set(tdays)))
    out = []
    for y in range(tdays[0].year, tdays[-1].year + 2):
        for m in range(1, 13):
            e = expiry(root, y, m, tdays)
            if e is None:
                continue
            nxt = tdays[tdays > e]
            if len(nxt) and nxt[0] <= tdays[-1]:
                out.append(nxt[0])
    return pd.DatetimeIndex(sorted(set(out)))


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = dt.date(year, month, 15)
    offset = (4 - d.weekday()) % 7
    return pd.Timestamp(d + dt.timedelta(days=offset))


def quarterly_expiries(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    out = []
    for y in range(start.year, end.year + 2):
        for m in (3, 6, 9, 12):
            out.append(third_friday(y, m))
    return pd.DatetimeIndex(out)


def roll_adjusted_returns(front: pd.Series, nxt: pd.Series, rolls: pd.DatetimeIndex) -> pd.Series:
    """Daily returns of holding the front contract, using yesterday's NEXT contract as the base on roll days
    (yesterday's contract 2 is today's contract 1)."""
    df = pd.concat({"f": front, "n": nxt}, axis=1).dropna()
    r = df["f"] / df["f"].shift(1) - 1
    on_roll = df.index.isin(rolls)
    r[on_roll] = (df["f"] / df["n"].shift(1) - 1)[on_roll]
    # Non-positive prices (WTI contract 1 settled at -37.63 on 2020-04-20) make returns undefined: leave them out.
    base = df["f"].shift(1).where(~on_roll, df["n"].shift(1))
    return r.where((df["f"] > 0) & (base > 0))


def roll_check(front: pd.Series, nxt: pd.Series, rolls: pd.DatetimeIndex) -> dict:
    """Continuity diagnostic: on true roll days, today's contract 1 is close to yesterday's contract 2."""
    df = pd.concat({"f": front, "n": nxt}, axis=1).dropna()
    lf = np.log(df["f"])
    jump_same = (lf - lf.shift(1)).abs()
    jump_next = (lf - np.log(df["n"]).shift(1)).abs()
    on = df.index.isin(rolls)
    better = (jump_next[on] < jump_same[on]).mean()
    return {"roll_days": int(on.sum()), "share_next_closer_on_rolls": float(better),
            "median_jump_same_on_rolls": float(jump_same[on].median()), "median_jump_next_on_rolls": float(jump_next[on].median()),
            "share_next_closer_off_rolls": float((jump_next[~on] < jump_same[~on]).mean())}
