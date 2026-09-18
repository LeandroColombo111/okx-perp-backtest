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
