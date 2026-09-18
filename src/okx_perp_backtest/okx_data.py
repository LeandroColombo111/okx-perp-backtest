"""Download hourly OKX perpetual candles and funding into the bar format the engine expects.

Uses only OKX's public endpoints: no account, API key or personal data is involved.

    bars = fetch_okx_bars("BTC-USDT-SWAP", start="2026-07-01", end="2026-09-01")
    python -m okx_perp_backtest.okx_data --start 2026-07-01 --out btc_1h.csv

Conventions and limits, stated rather than hidden:
- volume is `volCcy` (base currency, e.g. BTC), not OKX's `vol` (contracts).
- only completed candles are returned.
- OKX's funding-rate-history endpoint serves roughly the last 3 months. Asking for
  funding older than that raises instead of filling zeros; pass include_funding=False
  (funding = 0) if you accept that.
- the result is checked with data.validate_bars, so gaps in OKX's data raise too.
"""
import argparse
import time
import urllib.error

import pandas as pd

from .data import validate_bars
from .okx_http import get_json

HOUR_MS = 3_600_000
FUNDING_PERIOD_MS = 8 * HOUR_MS
RETRYABLE_CODES = {"50011", "50013", "50026"}  # rate limited / system busy


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _call(path: str, params: dict, timeout: float, retries: int, pause: float) -> list:
    """One OKX call with backoff on transport errors and rate limits. Other API errors raise at once."""
    last = None
    for attempt in range(retries + 1):
        try:
            payload = get_json(path, params, timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last = exc
        else:
            code = payload.get("code")
            if code == "0":
                return payload["data"]
            if code not in RETRYABLE_CODES:
                raise ValueError("OKX error %s on %s: %s" % (code, path, payload.get("msg")))
            last = ValueError("OKX rate limited (%s)" % code)
        if attempt < retries:
            time.sleep(pause * 2 ** (attempt + 2))
    raise ValueError("OKX request failed after %d retries on %s: %s" % (retries, path, last))


def _fetch_candles(inst_id, start_ms, end_ms, timeout, retries, pause) -> dict:
    rows, cursor = {}, end_ms
    while True:
        data = _call("/api/v5/market/history-candles",
                     {"instId": inst_id, "bar": "1H", "limit": "100", "after": str(cursor)}, timeout, retries, pause)
        if not data:
            break
        for r in data:
            ts = int(r[0])
            if r[8] == "1" and start_ms <= ts < end_ms:
                rows[ts] = r
        oldest = min(int(r[0]) for r in data)
        if oldest >= cursor:
            raise ValueError("OKX candle pagination did not advance; refusing to loop")
        cursor = oldest
        if cursor <= start_ms:
            break
        time.sleep(pause)
    return rows


def _fetch_funding(inst_id, start_ms, end_ms, timeout, retries, pause) -> dict:
    events, cursor = {}, end_ms
    while True:
        data = _call("/api/v5/public/funding-rate-history",
                     {"instId": inst_id, "limit": "100", "after": str(cursor)}, timeout, retries, pause)
        if not data:
            break
        for r in data:
            ts = int(r["fundingTime"])
            if start_ms <= ts < end_ms:
                events[ts] = float(r.get("realizedRate") or r["fundingRate"])
        oldest = min(int(r["fundingTime"]) for r in data)
        if oldest >= cursor:
            raise ValueError("OKX funding pagination did not advance; refusing to loop")
        cursor = oldest
        if cursor <= start_ms:
            break
        time.sleep(pause)
    return events


def _check_funding_coverage(events: dict, start_ms: int, end_ms: int, inst_id: str) -> None:
    slack = FUNDING_PERIOD_MS + 60_000
    if not events:
        raise ValueError("OKX returned no funding for %s in the requested range (its API serves only about the last "
                         "3 months). Use include_funding=False to run with funding = 0." % inst_id)
    times = sorted(events)
    first_ok = pd.Timestamp(times[0], unit="ms", tz="UTC")
    if times[0] - start_ms > slack:
        raise ValueError("OKX funding history starts at %s but you asked from %s (the API serves about 3 months). "
                         "Start later, or pass include_funding=False to use funding = 0."
                         % (first_ok, pd.Timestamp(start_ms, unit="ms", tz="UTC")))
    if end_ms - times[-1] > slack:
        raise ValueError("Funding ends at %s, well before the requested end; refusing to fill the rest with zeros."
                         % pd.Timestamp(times[-1], unit="ms", tz="UTC"))
    if any(b - a > slack for a, b in zip(times, times[1:])):
        raise ValueError("Funding history has a gap larger than 8h; refusing to fill it with zeros.")


def fetch_okx_bars(inst_id: str = "BTC-USDT-SWAP", start="2026-01-01", end=None, include_funding: bool = True,
                   *, timeout: float = 10., retries: int = 4, pause: float = 0.12) -> pd.DataFrame:
    """Hourly bars for [start, end) in UTC. end defaults to the last completed hour."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start_t = _utc(start).ceil("h")
    end_t = min(_utc(end).floor("h") if end is not None else now, now)
    if start_t >= end_t:
        raise ValueError("start must be before end (and end cannot be in the future)")
    start_ms, end_ms = int(start_t.timestamp() * 1000), int(end_t.timestamp() * 1000)

    candles = _fetch_candles(inst_id, start_ms, end_ms, timeout, retries, pause)
    if not candles:
        raise ValueError("OKX returned no completed candles for %s in the requested range" % inst_id)
    ordered = [candles[ts] for ts in sorted(candles)]
    bars = pd.DataFrame(
        [[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in ordered],
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([pd.Timestamp(int(r[0]), unit="ms", tz="UTC") for r in ordered], name="time"),
    )
    if include_funding:
        events = _fetch_funding(inst_id, start_ms, end_ms, timeout, retries, pause)
        _check_funding_coverage(events, start_ms, end_ms, inst_id)
        series = pd.Series(events)
        series.index = pd.to_datetime(series.index, unit="ms", utc=True).floor("h")
        bars["funding"] = series.groupby(level=0).sum().reindex(bars.index, fill_value=0.0)
    else:
        bars["funding"] = 0.0
    return validate_bars(bars)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Download OKX hourly candles + funding to CSV (public data only).")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--start", required=True, help="UTC date/time, e.g. 2026-07-01")
    ap.add_argument("--end", help="UTC date/time, exclusive; default = last completed hour")
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--no-funding", action="store_true", help="write funding = 0 (needed for ranges older than ~3 months)")
    args = ap.parse_args(argv)
    bars = fetch_okx_bars(args.inst, args.start, args.end, include_funding=not args.no_funding)
    bars.to_csv(args.out)
    print("%s: %d hourly bars, %s -> %s, saved to %s"
          % (args.inst, len(bars), bars.index[0], bars.index[-1], args.out))


if __name__ == "__main__":
    main()
