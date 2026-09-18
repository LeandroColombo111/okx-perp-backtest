"""Input validation whose errors say what to fix.

The engine trusts its inputs: a gap, a percentage-style funding column, a
missing volume or a misaligned signal would otherwise give a plausible but
wrong backtest. Call these before running anything on your own data.
"""
import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
SIGNAL_COLUMNS = ["signal", "anchor", "atr"]
MAX_SHOWN = 10


def _raise(problems: list) -> None:
    if problems:
        shown = problems[:MAX_SHOWN]
        extra = "" if len(problems) <= MAX_SHOWN else "\n  ... and %d more" % (len(problems) - MAX_SHOWN)
        raise ValueError("Invalid input, %d problem(s):\n  - " % len(problems) + "\n  - ".join(shown) + extra)


def validate_bars(bars: pd.DataFrame, frequency: str = "1h", require_funding: bool = True) -> pd.DataFrame:
    """Check hourly OHLCV(+funding) bars. Returns bars unchanged, or raises ValueError listing every problem.

    Contract: index = bar OPEN time in UTC, regular spacing, sorted, no duplicates;
    columns open/high/low/close/volume, plus funding (fraction per 8h settlement,
    e.g. 0.0001 = 0.01%, positive means longs pay) which the engine reads only on
    bars opening at 00:00, 08:00 and 16:00 UTC.
    """
    if not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must be a DataFrame indexed by a DatetimeIndex of bar OPEN times (UTC)")
    if bars.empty:
        raise ValueError("bars is empty")
    problems = []
    idx = bars.index
    if idx.tz is None:
        problems.append("index is not timezone-aware; use bars.index = pd.to_datetime(bars.index, utc=True)")
    elif not (idx.tz_convert("UTC").tz_localize(None) == idx.tz_localize(None)).all():
        problems.append("index is not in UTC; convert with bars.index = bars.index.tz_convert('UTC')")
    if idx.has_duplicates:
        problems.append("index has duplicate timestamps (%d); drop them" % idx.duplicated().sum())
    if not idx.is_monotonic_increasing:
        problems.append("index is not sorted ascending; use bars = bars.sort_index()")
    elif not idx.has_duplicates:
        step = pd.Timedelta(frequency)
        gaps = idx.to_series().diff().dropna()
        bad = gaps[gaps != step]
        if len(bad):
            first = bad.index[0]
            problems.append("%d irregular step(s) (expected %s), first ending at %s (gap of %s); fill or remove missing bars"
                            % (len(bad), frequency, first, bad.iloc[0]))
    need = PRICE_COLUMNS + ["volume"] + (["funding"] if require_funding else [])
    missing = [c for c in need if c not in bars.columns]
    for c in missing:
        hint = {"volume": "required: the slippage model sizes impact against it",
                "funding": "required unless require_funding=False; use 0.0 if you want to ignore funding"}.get(c, "required")
        problems.append("missing column '%s' (%s)" % (c, hint))
    present = [c for c in need if c in bars.columns]
    if present:
        values = bars[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        bad_cells = ~np.isfinite(values)
        if bad_cells.any():
            cols = [present[i] for i in np.flatnonzero(bad_cells.any(axis=0))]
            problems.append("NaN/inf/non-numeric values in: %s" % ", ".join(cols))
    if not missing and not any("NaN" in p for p in problems):
        o, h, l, c = (bars[k] for k in PRICE_COLUMNS)
        if (bars[PRICE_COLUMNS] <= 0).any().any():
            problems.append("non-positive prices found")
        n = int(((h < pd.concat([o, c, l], axis=1).max(axis=1)) | (l > pd.concat([o, c], axis=1).min(axis=1))).sum())
        if n:
            problems.append("%d bar(s) with inconsistent OHLC (high below open/close/low, or low above open/close)" % n)
        if (bars["volume"] < 0).any():
            problems.append("negative volume found")
        if require_funding and bars["funding"].abs().max() > 0.01:
            problems.append("funding max |value| is %.4g; the engine expects a FRACTION per 8h (0.0001 = 0.01%%), "
                            "not a percentage" % bars["funding"].abs().max())
    _raise(problems)
    return bars


def validate_signal(signal: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Check a signal frame against the bars it will be run on. Returns signal, or raises ValueError.

    Row i is what your strategy knows at the CLOSE of bar i; the engine acts on it at the
    OPEN of bar i+1. Compute it only from bars up to and including bar i.
    """
    problems = []
    if not isinstance(signal, pd.DataFrame):
        raise ValueError("signal must be a DataFrame with columns: signal, anchor, atr")
    missing = [c for c in SIGNAL_COLUMNS if c not in signal.columns]
    if missing:
        problems.append("missing column(s): %s (signal=+1/-1/0 desired position, anchor=+1/-1/0 trend "
                        "used to force exits, atr=stop-distance unit in price)" % ", ".join(missing))
    if not signal.index.equals(bars.index):
        problems.append("signal index differs from bars index (lengths %d vs %d); build it from the same bars"
                        % (len(signal), len(bars)))
    if not missing:
        for col in ("signal", "anchor"):
            vals = signal[col]
            if vals.isna().any() or not vals.isin([-1, 0, 1]).all():
                problems.append("'%s' must contain only -1, 0 or 1 (no NaN)" % col)
        live = signal["signal"] != 0
        atr = signal["atr"]
        if (live & ~(np.isfinite(atr) & (atr > 0))).any():
            problems.append("'atr' must be finite and > 0 on every row where signal != 0")
    _raise(problems)
    return signal
