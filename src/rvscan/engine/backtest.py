"""Signals and backtests for spread trades.

Timing (conservative by default): the signal is computed from data up to the close of day t; the trade is done at
the close of day t + lag (lag = 1); the position earns P&L from the following day. Costs are charged on every change
of position at the stress-scaled round-trip rate / 2 per unit traded, plus holding costs (funding, borrow), hedge
rebalancing, and a full re-trade on roll days. Positions are forced flat on `no_trade` dates (futures contract
switches) and around them.

Rules: z = deviation / trailing sd(deviation). Enter when |z| >= entry_z AND (for relationships with a no-arbitrage
band) the deviation lies outside the band; exit when |z| <= exit_z; stop out when |z| >= stop_z (then wait for |z|
to fall below entry_z before re-entering); time stop after max_hold periods. Optional gates: regime probability
(Markov-switching filter) or Bayesian probability of mean reversion, both computed walk-forward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..scanners.base import SpreadData


@dataclass
class TradeParams:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    lookback: int = 126            # periods for the z-score standard deviation
    max_hold: int = 126            # periods (time stop)
    lag: int = 1                   # periods between signal and execution
    cost_mult: float = 1.0
    use_band: bool = True
    gate: str | None = None        # None | "regime" | "bayes"
    gate_threshold: float = 0.5
    direction_override: str | None = None
    rule: str = "auto"             # "zscore" (mean reversion), "carry" (hold the positive-carry side), "auto"
    min_carry_to_vol: float = 0.25

    def to_dict(self):
        return asdict(self)


def zscore(sd: SpreadData, lookback: int) -> pd.Series:
    dev = sd.deviation
    sig = dev.rolling(lookback, min_periods=max(10, lookback // 2)).std()
    return dev / sig


def _band_at(x, idx):
    if isinstance(x, pd.Series):
        return x.reindex(idx).ffill().values
    return np.full(len(idx), float(x))


def carry_positions(sd: SpreadData, p: TradeParams) -> pd.Series:
    """Carry rule: hold the side that earns the rate differential when carry / volatility is large enough."""
    ctv = sd.meta.get("carry_to_vol")
    spread = sd.spread
    if ctv is None:
        return np.sign(spread).fillna(0.0)
    ctv = ctv.reindex(spread.index)
    pos = np.where(ctv.abs() >= p.min_carry_to_vol, np.sign(spread), 0.0)
    return pd.Series(pos, index=spread.index).fillna(0.0)


def rule_for(sd: SpreadData, p: TradeParams) -> str:
    if p.rule != "auto":
        return p.rule
    return "carry" if sd.rel.params.get("builder") == "fx_carry" else "zscore"


def positions(sd: SpreadData, p: TradeParams, gate: pd.Series | None = None) -> pd.Series:
    """Target position decided at each close (+1 long spread, -1 short, 0 flat), using data up to that close."""
    if rule_for(sd, p) == "carry":
        return carry_positions(sd, p)
    z = zscore(sd, p.lookback)
    dev = sd.deviation
    idx = z.index
    zv, dv = z.values, dev.reindex(idx).values
    bu, bd = _band_at(sd.band_up, idx), _band_at(sd.band_down, idx)
    gv = gate.reindex(idx).values if gate is not None else None
    direction = p.direction_override or sd.direction
    allow_long = direction in ("both", "long_only")
    allow_short = direction in ("both", "short_only")
    no_trade = set(sd.no_trade) if sd.no_trade is not None else set()
    # Flat on the day before and the day of a contract switch.
    nt_mask = np.zeros(len(idx), dtype=bool)
    if no_trade:
        ntv = idx.isin(list(no_trade))
        nt_mask = ntv | np.r_[ntv[1:], False]
    pos = np.zeros(len(idx))
    cur, held, cooldown = 0.0, 0, False
    for t in range(len(idx)):
        zt = zv[t]
        if not np.isfinite(zt) or nt_mask[t]:
            cur, held = 0.0, 0
            pos[t] = 0.0
            continue
        if cur != 0:
            held += 1
            if abs(zt) <= p.exit_z or (cur > 0 and zt >= 0) or (cur < 0 and zt <= 0):
                cur, held = 0.0, 0
            elif abs(zt) >= p.stop_z:
                cur, held, cooldown = 0.0, 0, True
            elif held >= p.max_hold:
                cur, held, cooldown = 0.0, 0, True
        if cur == 0:
            if cooldown and abs(zt) < p.entry_z:
                cooldown = False
            gate_ok = gv is None or (np.isfinite(gv[t]) and gv[t] >= p.gate_threshold)
            if not cooldown and gate_ok and abs(zt) >= p.entry_z and abs(zt) < p.stop_z:
                d = dv[t]
                # A band applies only where an arbitrage mechanism defines one (finite); inf = no bound.
                ok_short = not p.use_band or not np.isfinite(bu[t]) or d > bu[t]
                ok_long = not p.use_band or not np.isfinite(bd[t]) or -d > bd[t]
                if zt > 0 and allow_short and ok_short:
                    cur, held = -1.0, 0
                elif zt < 0 and allow_long and ok_long:
                    cur, held = 1.0, 0
        pos[t] = cur
    return pd.Series(pos, index=idx)


@dataclass
class BacktestResult:
    daily: pd.DataFrame            # gross, carry, hold, costs, net, position
    trades: pd.DataFrame
    metrics: dict
    params: dict = field(default_factory=dict)


def _series(x, idx) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(idx).ffill().fillna(0.0)
    return pd.Series(float(x), index=idx)


def run(sd: SpreadData, p: TradeParams | None = None, gate: pd.Series | None = None,
        start=None, end=None) -> BacktestResult:
    p = p or TradeParams()
    if sd.pnl is None:
        raise ValueError(f"{sd.rel.id} is monitor-only (no tradable P&L)")
    target = positions(sd, p, gate)
    idx = target.index
    held = target.shift(p.lag).fillna(0.0)               # position after trading at the close of t
    pnl = sd.pnl.reindex(idx)
    prev = held.shift(1).fillna(0.0)                       # position carried into day t
    gross = prev * pnl.fillna(0.0)
    carry = prev * _series(sd.carry, idx)
    hold = prev.abs() * _series(sd.hold_cost, idx)
    rebal = prev.abs() * _series(sd.rebal_cost, idx)
    cost_rt = _series(sd.cost_rt, idx) * p.cost_mult
    trade_cost = (held - prev).abs() * cost_rt / 2.0
    roll_cost = pd.Series(0.0, index=idx)
    if sd.roll_days is not None and len(sd.roll_days):
        on = idx.isin(sd.roll_days)
        roll_cost[on] = (prev.abs() * cost_rt)[on]
    costs = trade_cost + roll_cost + rebal
    net = gross + carry - hold - costs
    changed = (held != prev).astype(float)
    daily = pd.DataFrame({"position": held, "gross": gross, "carry": carry, "hold": -hold, "costs": -costs, "net": net,
                          "cost_entry": -(held.abs() * changed * cost_rt / 2.0),
                          "cost_exit": -(prev.abs() * changed * cost_rt / 2.0),
                          "cost_other": -(roll_cost + rebal),
                          "spread": sd.spread.reindex(idx), "z": zscore(sd, p.lookback).reindex(idx)})
    if start is not None:
        daily = daily[daily.index >= pd.Timestamp(start)]
    if end is not None:
        daily = daily[daily.index <= pd.Timestamp(end)]
    trades = trade_list(daily, sd)
    return BacktestResult(daily, trades, metrics(daily, trades, sd.periods_per_year()), p.to_dict())


def trade_list(daily: pd.DataFrame, sd: SpreadData) -> pd.DataFrame:
    pos = daily["position"].values
    rows = []
    t, n = 0, len(daily)
    idx = daily.index
    while t < n:
        if pos[t] != 0 and (t == 0 or pos[t - 1] != pos[t]):
            start = t
            d = pos[t]
            u = t + 1
            while u < n and pos[u] == d:
                u += 1
            # P&L accrues from the day after entry through the day the position is closed (u). Trading costs are
            # split so a flip (long -> short on one day) charges its exit half to the old trade, entry to the new.
            seg = daily.iloc[start + 1:min(u + 1, n)]
            gross_cum = seg["gross"].cumsum()
            entry_cost = float(daily["cost_entry"].iloc[start])
            exit_cost = float(daily["cost_exit"].iloc[u]) if u < n else 0.0
            other = float(seg["cost_other"].sum())
            costs_total = entry_cost + exit_cost + other
            net_total = float(seg["gross"].sum() + seg["carry"].sum() + seg["hold"].sum()) + costs_total
            mae = float(min(0.0, (gross_cum.min() if len(gross_cum) else 0.0)))
            mfe = float(max(0.0, (gross_cum.max() if len(gross_cum) else 0.0)))
            rows.append({"entry": idx[start], "exit": idx[min(u, n - 1)], "direction": int(d),
                         "entry_z": float(daily["z"].iloc[start]), "entry_spread": float(daily["spread"].iloc[start]),
                         "periods": int(u - start), "gross": float(seg["gross"].sum()), "carry": float(seg["carry"].sum()),
                         "costs": costs_total, "hold": float(seg["hold"].sum()),
                         "net": net_total, "mae": mae, "mfe": mfe, "open": bool(u >= n)})
            t = u
        else:
            t += 1
    return pd.DataFrame(rows)


def metrics(daily: pd.DataFrame, trades: pd.DataFrame, ppy: float) -> dict:
    net = daily["net"]
    gross = daily["gross"]
    active = daily["position"].shift(1).fillna(0) != 0
    years = max(len(daily) / ppy, 1e-9)
    sd = net.std()
    out = {
        "periods": int(len(daily)), "years": round(years, 2),
        "start": str(daily.index[0].date()) if len(daily) else None, "end": str(daily.index[-1].date()) if len(daily) else None,
        "net_total": float(net.sum()), "gross_total": float(gross.sum()),
        "carry_total": float(daily["carry"].sum()), "hold_total": float(daily["hold"].sum()),
        "cost_total": float(daily["costs"].sum()),
        "net_per_year": float(net.sum() / years), "vol_per_year": float(sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        "sharpe": float(net.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        "gross_sharpe": float(gross.mean() / gross.std() * np.sqrt(ppy)) if gross.std() > 0 else 0.0,
        "time_in_market": float(active.mean()),
        "max_drawdown": float((net.cumsum() - net.cumsum().cummax()).min()) if len(net) else 0.0,
        "trades": int(len(trades)),
    }
    if len(trades):
        closed = trades[~trades["open"]]
        out.update({
            "hit_rate": float((closed["net"] > 0).mean()) if len(closed) else float("nan"),
            "avg_net_per_trade": float(closed["net"].mean()) if len(closed) else float("nan"),
            "avg_gross_per_trade": float(closed["gross"].mean()) if len(closed) else float("nan"),
            "avg_cost_per_trade": float(-closed["costs"].mean()) if len(closed) else float("nan"),
            "avg_periods": float(closed["periods"].mean()) if len(closed) else float("nan"),
            "mae_median": float(closed["mae"].median()) if len(closed) else float("nan"),
            "mae_p95": float(closed["mae"].quantile(0.05)) if len(closed) else float("nan"),
            "trades_per_year": float(len(trades) / years),
        })
        costs = -out["cost_total"] - out["hold_total"]
        out["breakeven_cost_mult"] = float((out["gross_total"] + out["carry_total"]) / costs) if costs > 0 else float("inf")
    return out


def convergence_episodes(sd: SpreadData, entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 126,
                         horizon: int = 252) -> pd.DataFrame:
    """Every time |z| crossed entry_z: how long until |z| fell below exit_z, and the worst move against."""
    z = zscore(sd, lookback).dropna()
    dev = sd.deviation.reindex(z.index)
    zv = z.values
    rows = []
    t = 1
    while t < len(zv):
        if abs(zv[t]) >= entry_z and abs(zv[t - 1]) < entry_z:
            sign = np.sign(zv[t])
            end = None
            worst = 0.0
            for u in range(t + 1, min(t + horizon, len(zv))):
                adverse = (dev.iloc[u] - dev.iloc[t]) * sign
                worst = max(worst, adverse)
                if abs(zv[u]) <= exit_z:
                    end = u
                    break
            rows.append({"start": z.index[t], "z": float(zv[t]), "deviation": float(dev.iloc[t]),
                         "periods_to_revert": (end - t) if end else np.nan, "reverted": end is not None,
                         "max_adverse": float(worst)})
            t = (end or t) + 1
        else:
            t += 1
    return pd.DataFrame(rows)
