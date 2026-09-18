import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from okx_perp_backtest.data import validate_bars, validate_signal
from okx_perp_backtest.example_signal import ExampleParams, example_signal


def problems_of(fn, *args, **kw):
    with pytest.raises(ValueError) as e:
        fn(*args, **kw)
    return str(e.value)


def test_valid_bars_pass_and_are_returned_unchanged(bars):
    assert validate_bars(bars) is bars


def test_naive_index_is_explained(bars):
    assert "timezone-aware" in problems_of(validate_bars, bars.tz_localize(None))


def test_non_utc_index_is_explained(bars):
    assert "not in UTC" in problems_of(validate_bars, bars.tz_convert("America/Argentina/Buenos_Aires"))


def test_gap_is_located(bars):
    msg = problems_of(validate_bars, bars.drop(bars.index[100:103]))
    assert "irregular step" in msg and str(bars.index[103]) in msg


def test_duplicates_and_unsorted(bars):
    assert "duplicate" in problems_of(validate_bars, pd.concat([bars, bars.iloc[:2]]))
    assert "not sorted" in problems_of(validate_bars, bars.iloc[::-1])


def test_missing_volume_is_flagged_because_slippage_needs_it(bars):
    assert "volume" in problems_of(validate_bars, bars.drop(columns="volume"))


def test_funding_can_be_skipped_but_not_by_accident(bars):
    no_funding = bars.drop(columns="funding")
    assert "funding" in problems_of(validate_bars, no_funding)
    validate_bars(no_funding, require_funding=False)


def test_percentage_style_funding_is_caught(bars):
    bad = bars.copy()
    bad["funding"] = bad["funding"] * 10000   # 0.0001 written as 1.0 (percent-ish)
    assert "FRACTION" in problems_of(validate_bars, bad)


def test_inconsistent_ohlc_and_nan(bars):
    bad = bars.copy()
    bad.iloc[5, bad.columns.get_loc("high")] = bad["low"].iloc[5] - 1
    assert "inconsistent OHLC" in problems_of(validate_bars, bad)
    bad = bars.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = np.nan
    assert "NaN" in problems_of(validate_bars, bad)


def test_all_problems_are_reported_together(bars):
    msg = problems_of(validate_bars, bars.tz_localize(None).drop(columns=["volume", "funding"]))
    assert "3 problem(s)" in msg


def test_valid_signal_passes(bars):
    sig = example_signal(bars, ExampleParams())
    assert validate_signal(sig, bars) is sig


def test_signal_problems_are_explained(bars):
    sig = example_signal(bars, ExampleParams())
    assert "missing column" in problems_of(validate_signal, sig.drop(columns="atr"), bars)
    assert "index differs" in problems_of(validate_signal, sig.iloc[:-5], bars)
    bad = sig.copy(); bad["signal"] = bad["signal"] * 2
    assert "only -1, 0 or 1" in problems_of(validate_signal, bad, bars)
    bad = sig.copy(); bad.loc[bad.index[-1], ["signal", "atr"]] = [1, np.nan]
    assert "atr" in problems_of(validate_signal, bad, bars)


def test_run_refuses_a_misaligned_signal_instead_of_mispairing_rows(bars, risk, sim):
    p = ExampleParams()
    with pytest.raises(ValueError, match="signal index must equal bars index"):
        sim.run(bars, example_signal(bars, p).iloc[10:], p, risk)


def test_own_signal_example_runs_end_to_end():
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "examples" / "own_signal.py")], cwd=root,
                         capture_output=True, text=True, timeout=120,
                         env={"PYTHONPATH": str(root / "src"), "PATH": ""})
    assert out.returncode == 0, out.stderr[-600:]
    assert "trades=" in out.stdout
