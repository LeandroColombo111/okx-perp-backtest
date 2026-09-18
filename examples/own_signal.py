"""Plug your OWN data and signal into the engine (synthetic bars keep it runnable).

Replace synthetic_bars() with your candles, e.g.:
    bars = pd.read_csv("btc_1h.csv", index_col="time", parse_dates=True)
    bars.index = pd.to_datetime(bars.index, utc=True)
The signal below is a plain Donchian breakout, only to show the contract; it is not a strategy.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from demo import synthetic_bars
from okx_perp_backtest.account import Risk, metrics
from okx_perp_backtest.data import validate_bars, validate_signal
from okx_perp_backtest.execution import ExecutionSimulator
from okx_perp_backtest.friction.fees import FeeModel
from okx_perp_backtest.friction.funding import FundingModel
from okx_perp_backtest.friction.liquidation import LiquidationModel, PositionTiers
from okx_perp_backtest.friction.slippage import NullLiquidityBook, SlippageModel


@dataclass(frozen=True)
class MyParams:
    """The engine reads exactly these attributes from your params object."""
    lookback: int = 96
    atr_period: int = 24
    stop_atr: float = 3.0
    trail_atr: float = 3.0
    reward: float = 3.0
    trail_start_r: float = 2.0
    max_hours: int = 240


def my_signal(bars: pd.DataFrame, p: MyParams) -> pd.DataFrame:
    """Row i may use only bars up to and including bar i; the engine acts on it at the open of bar i+1."""
    upper = bars.high.shift(1).rolling(p.lookback).max()
    lower = bars.low.shift(1).rolling(p.lookback).min()
    signal = np.where(bars.close > upper, 1, np.where(bars.close < lower, -1, 0))
    prev = bars.close.shift(1)
    true_range = pd.concat([bars.high - bars.low, (bars.high - prev).abs(), (bars.low - prev).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / p.atr_period, adjust=False, min_periods=p.atr_period).mean()
    # No higher-timeframe trend here, so anchor mirrors the signal: a position closes once the signal no longer agrees.
    return pd.DataFrame({"signal": signal, "anchor": signal, "atr": atr}, index=bars.index)


def main():
    bars = validate_bars(synthetic_bars())
    p = MyParams()
    signal = validate_signal(my_signal(bars, p), bars)
    risk = Risk(capital=10000.)
    sim = ExecutionSimulator(FundingModel(), SlippageModel(NullLiquidityBook(), impact_k=1.0),
                             FeeModel.regular_tier(), LiquidationModel(PositionTiers(), "isolated"), leverage=1.0)
    equity, trades, _ = sim.run(bars, signal, p, risk)
    m = metrics(equity, trades, risk.capital)
    print(f"trades={m['trades']} sharpe={m['sharpe']:.2f} return={m['return']:.1%} max_drawdown={m['max_drawdown']:.1%}")


if __name__ == "__main__":
    main()
