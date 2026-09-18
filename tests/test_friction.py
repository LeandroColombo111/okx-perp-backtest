import pandas as pd
import pytest

from okx_perp_backtest.friction.fees import FeeModel
from okx_perp_backtest.friction.funding import FundingModel
from okx_perp_backtest.friction.liquidation import LiquidationModel, PositionTiers
from okx_perp_backtest.friction.slippage import (
    InsufficientLiquidity, LiquidityBook, NullLiquidityBook, SlippageModel)

UTC = "UTC"


def test_funding_settles_only_at_okx_instants():
    m = FundingModel()
    assert m.is_funding_instant(pd.Timestamp("2026-01-01 08:00", tz=UTC))
    assert not m.is_funding_instant(pd.Timestamp("2026-01-01 09:00", tz=UTC))
    assert not m.is_funding_instant(pd.Timestamp("2026-01-01 08:30", tz=UTC))
    with pytest.raises(ValueError):
        m.is_funding_instant(pd.Timestamp("2026-01-01 08:00"))


def test_funding_sign_follows_side():
    m = FundingModel()
    assert m.settle(1, 1.0, 0.0001, 60000) == pytest.approx(6.0)     # long pays
    assert m.settle(-1, 1.0, 0.0001, 60000) == pytest.approx(-6.0)   # short receives


def test_fees_use_taker_rate_for_liquidation_on_okx():
    m = FeeModel.regular_tier()
    assert m.fee(10000, "maker") == pytest.approx(2.0)
    assert m.fee(10000, "taker") == pytest.approx(5.0)
    assert m.liquidation_fee(10000) == pytest.approx(m.fee(10000, "taker"))
    with pytest.raises(ValueError):
        FeeModel(-1, 5, 5)


def test_liquidation_price_side_and_leverage():
    m = LiquidationModel(PositionTiers(), "isolated")
    entry, qty = 60000., 0.5
    margin = qty * entry / 10.
    assert m.liquidation_price(entry, 1, qty, margin) < entry
    assert m.liquidation_price(entry, -1, qty, margin) > entry
    # fully collateralised long cannot be liquidated at a positive price
    assert m.liquidation_price(entry, 1, qty, qty * entry) == 0.


def test_liquidation_check_is_pessimistic_intrabar():
    m = LiquidationModel(PositionTiers(), "isolated")
    assert m.check(bar_low=99., bar_high=110., side=1, liq_price=100.)
    assert not m.check(bar_low=101., bar_high=110., side=1, liq_price=100.)
    assert m.check(bar_low=90., bar_high=101., side=-1, liq_price=100.)


def test_cross_margin_is_refused_not_faked():
    with pytest.raises(NotImplementedError):
        LiquidationModel(PositionTiers(), "cross")


def test_proxy_slippage_moves_price_against_the_order():
    m = SlippageModel(NullLiquidityBook(), impact_k=1.0)
    ts = pd.Timestamp("2026-01-01", tz=UTC)
    buy = m.fill_price(ts, 60000., 1, 10000., 5e6, 500.)
    sell = m.fill_price(ts, 60000., -1, 10000., 5e6, 500.)
    assert buy.price > 60000. > sell.price and buy.source == "proxy"


def test_walk_book_averages_levels_and_refuses_to_invent_liquidity():
    m = SlippageModel(NullLiquidityBook(), impact_k=1.0)
    levels = [(60010., 0.2), (60020., 0.3), (60030., 1.0)]
    assert m._walk_book(levels, 0.4) == pytest.approx((60010 * 0.2 + 60020 * 0.2) / 0.4)
    with pytest.raises(InsufficientLiquidity):
        m._walk_book(levels, 10.)


def test_real_book_is_used_when_it_covers_the_timestamp():
    class Book(LiquidityBook):
        def has_real_book(self, timestamp):
            return True

        def levels_at(self, timestamp, side):
            return [(60010., 5.)]

    fill = SlippageModel(Book(), 1.0).fill_price(pd.Timestamp("2026-01-01", tz=UTC), 60000., 1, 6000., 5e6, 500.)
    assert fill.source == "real_book" and fill.price == pytest.approx(60010.)


# --- OKX position tiers: sized in contracts, not dollars -------------------------------------------------

REAL_ROWS = [  # first three BTC-USDT isolated tiers as returned by OKX on 2026-09-18 (maxSz is in contracts)
    {"tier": "2", "minSz": "1000.01", "maxSz": "5000", "mmr": "0.005", "maxLever": "66.66"},
    {"tier": "1", "minSz": "0", "maxSz": "1000", "mmr": "0.004", "maxLever": "100"},
    {"tier": "3", "minSz": "5000.01", "maxSz": "20000", "mmr": "0.0075", "maxLever": "50"},
]


def test_okx_rows_are_converted_from_contracts_to_base_currency_and_sorted():
    from okx_perp_backtest.friction.liquidation import PositionTiers
    t = PositionTiers.from_okx_rows(REAL_ROWS, ct_val=0.01)
    assert t.rows == ((10.0, 100.0, 0.004), (50.0, 66.66, 0.005), (200.0, 50.0, 0.0075))


def test_tier_lookup_boundaries_and_oversize_is_refused():
    from okx_perp_backtest.friction.liquidation import PositionTiers
    t = PositionTiers.from_okx_rows(REAL_ROWS, ct_val=0.01)
    assert t.tier_for_size(10.0)[1] == 0.004        # boundary belongs to the lower tier
    assert t.tier_for_size(10.01)[1] == 0.005
    with pytest.raises(ValueError, match="largest tier"):
        t.tier_for_size(201.)


def test_tier_depends_on_size_not_on_price():
    from okx_perp_backtest.friction.liquidation import LiquidationModel, PositionTiers
    m = LiquidationModel(PositionTiers.from_okx_rows(REAL_ROWS, 0.01), "isolated")
    # same 20 BTC position at two very different prices: same tier, so the liquidation gap scales with price
    low = m.liquidation_price(1000., 1, 20., 20. * 1000. / 10.)
    high = m.liquidation_price(100000., 1, 20., 20. * 100000. / 10.)
    assert high / 100000. == pytest.approx(low / 1000.)


def test_offline_snapshot_is_valid_and_dated():
    from okx_perp_backtest.friction.liquidation import PositionTiers
    t = PositionTiers()
    assert len(t.rows) > 50 and "fetched" in t.source
    mmrs = [r[2] for r in t.rows]
    assert mmrs == sorted(mmrs), "maintenance margin ratio should not fall as size grows"
    assert t.tier_for_size(1.0) == (100.0, 0.004)


def test_fetch_okx_parses_both_endpoints_without_touching_the_network(monkeypatch):
    from okx_perp_backtest.friction import liquidation as liq

    def fake(path, params, timeout):
        if "position-tiers" in path:
            return {"code": "0", "data": REAL_ROWS}
        return {"code": "0", "data": [{"ctVal": "0.01"}]}

    monkeypatch.setattr(liq, "_get_json", fake)
    t = liq.PositionTiers.fetch_okx()
    assert t.rows[0] == (10.0, 100.0, 0.004) and t.source == "OKX live BTC-USDT-SWAP"


def test_fetch_okx_refuses_an_error_response(monkeypatch):
    from okx_perp_backtest.friction import liquidation as liq
    monkeypatch.setattr(liq, "_get_json", lambda *a: {"code": "51001", "data": []})
    with pytest.raises(ValueError):
        liq.PositionTiers.fetch_okx()
