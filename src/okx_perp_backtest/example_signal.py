"""Placeholder signal used only to demo the engine end-to-end. NOT a production trading strategy.

Swap this module for your own signal generator. It just needs to return a
DataFrame, indexed like `bars`, with three columns consumed by
execution.ExecutionSimulator.step:
  - "signal": desired position for the NEXT bar (+1 long, -1 short, 0 flat)
  - "anchor": higher-timeframe trend; a flip forces an exit even before the
    stop/target is touched
  - "atr": used to size the initial stop distance and the trailing stop
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExampleParams:
    fast: int = 12
    slow: int = 48
    atr_period: int = 14
    stop_atr: float = 2.5
    trail_atr: float = 3.0
    reward: float = 4.0
    max_hours: int = 240
    trail_start_r: float = 1.0

    def __post_init__(self):
        if not (1 < self.fast < self.slow):
            raise ValueError("fast must be less than slow")
        if min(self.stop_atr, self.trail_atr, self.reward, self.atr_period) <= 0:
            raise ValueError("Invalid indicator parameters")


def example_signal(bars: pd.DataFrame, p: ExampleParams) -> pd.DataFrame:
    """A bare EMA-crossover long/short flip. Deliberately simple: it exists to exercise the engine, not to trade with."""
    fast = bars.close.ewm(span=p.fast, adjust=False, min_periods=p.fast).mean()
    slow = bars.close.ewm(span=p.slow, adjust=False, min_periods=p.slow).mean()
    side = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    prev_close = bars.close.shift(1)
    true_range = pd.concat(
        [bars.high - bars.low, (bars.high - prev_close).abs(), (bars.low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / p.atr_period, adjust=False, min_periods=p.atr_period).mean()
    return pd.DataFrame({"signal": side, "anchor": side, "atr": atr}, index=bars.index)
