"""Includes the regression test for the warmup mistake this project made and fixed.

A slow indicator computed from scratch inside a short test window never
converges, so the strategy does not trade. If one variant is run with warmup
and its baseline without, the comparison silently favours the variant.
"""
import pandas as pd
import pytest

from okx_perp_backtest.example_signal import ExampleParams, example_signal
from okx_perp_backtest.walkforward import WalkForwardEngine

SLOW = ExampleParams(fast=12, slow=1200)   # 50-day slow EMA, longer than the 30-day test window
SPANS = dict(train_span=pd.Timedelta(days=120), test_span=pd.Timedelta(days=30), step=pd.Timedelta(days=30))


def test_windows_are_ordered_and_test_starts_where_train_ends(sim, bars):
    wf = WalkForwardEngine(sim, example_signal)
    windows = wf.windows(bars.index, **SPANS)
    assert len(windows) > 3
    for train, test in windows:
        assert train[1] == test[0] and train[0] < train[1] < test[1]


def test_no_window_fits_is_an_error(sim, bars):
    wf = WalkForwardEngine(sim, example_signal)
    with pytest.raises(ValueError):
        wf.windows(bars.index, pd.Timedelta(days=900), pd.Timedelta(days=90), pd.Timedelta(days=30))


def test_without_warmup_a_slow_indicator_never_trades_out_of_sample(sim, bars, risk):
    wf = WalkForwardEngine(sim, example_signal)
    cold = wf.run(bars, [SLOW], risk, warmup=pd.Timedelta(0), **SPANS)
    assert (cold.out_of_sample_return == 0.).all(), "starved indicator should produce no trades in the test window"


def test_with_warmup_the_same_strategy_does_trade(sim, bars, risk):
    wf = WalkForwardEngine(sim, example_signal)
    warm = wf.run(bars, [SLOW], risk, warmup=pd.Timedelta(days=100), **SPANS)
    assert (warm.out_of_sample_return != 0.).any()


def test_warmup_does_not_shift_the_scored_window(sim, bars, risk):
    wf = WalkForwardEngine(sim, example_signal)
    warm = wf.run(bars, [SLOW], risk, warmup=pd.Timedelta(days=100), **SPANS)
    assert warm.test_start.iloc[0] == wf.windows(bars.index, **SPANS)[0][1][0]
