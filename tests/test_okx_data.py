"""The OKX loader against a fake server that speaks OKX's response format. No network is used."""
import urllib.error

import pandas as pd
import pytest

from okx_perp_backtest import okx_data
from okx_perp_backtest.data import validate_bars

T0 = pd.Timestamp("2026-01-01", tz="UTC")
HOURS = 24 * 10


def ms(t):
    return int(t.timestamp() * 1000)


class FakeOKX:
    """Serves OKX-shaped pages, newest first, strictly older than `after`, 100 rows max."""

    def __init__(self, funding_floor=None, newest_unconfirmed=True, fail_first=0, code="0"):
        self.calls, self.fail_first, self.code = [], fail_first, code
        self.candles = [(ms(T0 + pd.Timedelta(hours=h)), 60000. + h) for h in range(HOURS)]
        self.unconfirmed_ts = self.candles[-1][0] if newest_unconfirmed else None
        floor = ms(funding_floor) if funding_floor is not None else 0
        self.funding = [(ms(T0 + pd.Timedelta(hours=h)), 0.0001 + h * 1e-7)
                        for h in range(0, HOURS, 8) if ms(T0 + pd.Timedelta(hours=h)) >= floor]

    def __call__(self, path, params, timeout):
        self.calls.append(path)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise urllib.error.URLError("temporary")
        if self.code != "0":
            return {"code": self.code, "msg": "nope", "data": []}
        after = int(params["after"])
        if "candles" in path:
            rows = [c for c in reversed(self.candles) if c[0] < after][:int(params["limit"])]
            data = [[str(ts), str(p), str(p + 5), str(p - 5), str(p + 1), "9999", "12.5", "1", "0" if ts == self.unconfirmed_ts else "1"]
                    for ts, p in rows]
        else:
            rows = [f for f in reversed(self.funding) if f[0] < after][:int(params["limit"])]
            data = [{"fundingTime": str(ts), "fundingRate": str(rate + 1.0), "realizedRate": str(rate)} for ts, rate in rows]
        return {"code": "0", "msg": "", "data": data}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(okx_data.time, "sleep", lambda s: None)


def use(monkeypatch, fake):
    monkeypatch.setattr(okx_data, "get_json", fake)
    monkeypatch.setattr(okx_data.pd.Timestamp, "now", classmethod(lambda cls, tz=None: T0 + pd.Timedelta(hours=HOURS + 2)))
    return fake


def test_pages_are_stitched_sorted_and_pass_validation(monkeypatch):
    use(monkeypatch, FakeOKX())
    bars = okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=HOURS))
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "funding"]
    assert bars.index.is_monotonic_increasing and bars.index.tz is not None
    validate_bars(bars)
    assert len(bars) == HOURS - 1, "the newest, unconfirmed candle must be dropped"


def test_range_is_respected_start_inclusive_end_exclusive(monkeypatch):
    use(monkeypatch, FakeOKX())
    bars = okx_data.fetch_okx_bars("X-USDT-SWAP", T0 + pd.Timedelta(hours=24), T0 + pd.Timedelta(hours=72))
    assert bars.index[0] == T0 + pd.Timedelta(hours=24) and bars.index[-1] == T0 + pd.Timedelta(hours=71)
    assert len(bars) == 48


def test_volume_is_base_currency_not_contracts(monkeypatch):
    use(monkeypatch, FakeOKX())
    bars = okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=48))
    assert (bars.volume == 12.5).all(), "volCcy (col 6) is base currency; vol (col 5) is contracts"


def test_funding_lands_on_settlement_bars_and_prefers_realized_rate(monkeypatch):
    use(monkeypatch, FakeOKX())
    bars = okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=HOURS))
    nonzero = bars.funding[bars.funding != 0]
    assert set(nonzero.index.hour) == {0, 8, 16}
    assert nonzero.iloc[0] == pytest.approx(0.0001)          # realizedRate, not the +1.0 fundingRate decoy
    assert (bars.funding.abs() < 0.01).all()


def test_funding_older_than_the_api_serves_raises_instead_of_zero_filling(monkeypatch):
    use(monkeypatch, FakeOKX(funding_floor=T0 + pd.Timedelta(hours=96)))
    with pytest.raises(ValueError, match="funding history starts at"):
        okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=HOURS))


def test_no_funding_at_all_raises_with_the_escape_hatch_named(monkeypatch):
    use(monkeypatch, FakeOKX(funding_floor=T0 + pd.Timedelta(days=400)))
    with pytest.raises(ValueError, match="include_funding=False"):
        okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=HOURS))


def test_include_funding_false_gives_explicit_zeros(monkeypatch):
    fake = use(monkeypatch, FakeOKX(funding_floor=T0 + pd.Timedelta(days=400)))
    bars = okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=HOURS), include_funding=False)
    assert (bars.funding == 0.).all() and not any("funding" in c for c in fake.calls)


def test_transient_errors_are_retried_then_succeed(monkeypatch):
    use(monkeypatch, FakeOKX(fail_first=2))
    assert len(okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=48), include_funding=False)) == 48


def test_persistent_failure_raises_after_retries(monkeypatch):
    use(monkeypatch, FakeOKX(fail_first=99))
    with pytest.raises(ValueError, match="failed after"):
        okx_data.fetch_okx_bars("X-USDT-SWAP", T0, T0 + pd.Timedelta(hours=48), include_funding=False, retries=2)


def test_api_error_code_is_surfaced_not_retried_forever(monkeypatch):
    fake = use(monkeypatch, FakeOKX(code="51001"))
    with pytest.raises(ValueError, match="51001"):
        okx_data.fetch_okx_bars("BAD-INSTRUMENT", T0, T0 + pd.Timedelta(hours=48), include_funding=False)
    assert len(fake.calls) == 1


def test_bad_ranges_are_refused(monkeypatch):
    use(monkeypatch, FakeOKX())
    with pytest.raises(ValueError):
        okx_data.fetch_okx_bars("X-USDT-SWAP", T0 + pd.Timedelta(hours=10), T0 + pd.Timedelta(hours=5))


def test_cli_writes_a_csv_the_engine_can_read_back(monkeypatch, tmp_path):
    use(monkeypatch, FakeOKX())
    out = tmp_path / "bars.csv"
    okx_data.main(["--inst", "X-USDT-SWAP", "--start", str(T0), "--end", str(T0 + pd.Timedelta(hours=72)), "--out", str(out)])
    back = pd.read_csv(out, index_col="time", parse_dates=True)
    back.index = pd.to_datetime(back.index, utc=True)
    validate_bars(back)
    assert len(back) == 72
