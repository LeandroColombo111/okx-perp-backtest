"""Position sizing, single-position account bookkeeping, and Sharpe/drawdown metrics.

Strategy-agnostic: nothing here knows what generates entries, only how to
size one against risk and score the resulting equity curve.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Risk:
    capital: float = 10000.
    fraction: float = .02
    fee_bps: float = 6.
    slippage_bps: float = 3.
    max_exposure: float = 1.
    max_drawdown: float = .25
    quantity_step: float = .001
    minimum_notional: float = 10.

    def __post_init__(self):
        if not .02 <= self.fraction <= .03:
            raise ValueError("Risk fraction must be 2% to 3%")
        if not (self.capital > 0 and 0 < self.max_exposure <= 1 and 0 < self.max_drawdown < 1):
            raise ValueError("Invalid capital, exposure or drawdown limit")
        if min(self.fee_bps, self.slippage_bps, self.minimum_notional) < 0 or self.quantity_step <= 0:
            raise ValueError("Invalid costs or quantity step")


@dataclass
class Account:
    cash: float
    peak: float
    equity: float
    qty: float = 0.
    entry: float = 0.
    stop: float = 0.
    target: float = 0.
    age: int = 0
    entry_time: str = ""
    trade_cost: float = 0.
    halted: bool = False
    trades: int = 0
    initial_distance: float = 0.
    peak_favorable_r: float = 0.

    @classmethod
    def new(cls, risk: Risk) -> "Account":
        return cls(risk.capital, risk.capital, risk.capital)


def size(equity: float, entry: float, distance: float, risk: Risk) -> float:
    """Quantity risking risk.fraction of equity given a stop distance, capped by max_exposure and rounded to quantity_step."""
    unit_risk = distance + entry * (2 * risk.fee_bps + risk.slippage_bps) / 10000
    q = min(equity * risk.fraction / unit_risk, equity * risk.max_exposure / entry)
    q = math.floor(q / risk.quantity_step) * risk.quantity_step
    return q if q * entry >= risk.minimum_notional else 0.


def daily_returns(equity: pd.Series, capital: float) -> pd.Series:
    """Resamples an intraday equity curve to daily percentage returns; close timestamps on midnight belong to the prior day."""
    e = equity.copy()
    e.index = e.index - pd.Timedelta(nanoseconds=1)
    daily = e.resample("1D").last()
    return daily.pct_change().fillna(daily.iloc[0] / capital - 1)


def metrics(equity: pd.Series, trades: list[dict], capital: float) -> dict:
    """Annualized daily Sharpe, total return, max drawdown and basic trade stats."""
    r = daily_returns(equity, capital)
    sd = float(r.std(ddof=1))
    wealth = np.r_[capital, equity.to_numpy()]
    dd = 1 - wealth / np.maximum.accumulate(wealth)
    return {
        "sharpe": float(np.sqrt(365) * r.mean() / sd) if sd > 0 else 0.,
        "return": float(equity.iloc[-1] / capital - 1),
        "max_drawdown": float(dd.max()),
        "trades": len(trades),
        "long_trades": sum(t["side"] == 1 for t in trades),
        "short_trades": sum(t["side"] == -1 for t in trades),
        "win_rate": sum(t["net_pnl"] > 0 for t in trades) / max(1, len(trades)),
        "days": len(r),
    }
