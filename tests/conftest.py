import numpy as np
import pandas as pd
import pytest

from okx_perp_backtest.account import Risk
from okx_perp_backtest.execution import ExecutionSimulator
from okx_perp_backtest.friction.fees import FeeModel
from okx_perp_backtest.friction.funding import FundingModel
from okx_perp_backtest.friction.liquidation import LiquidationModel, PositionTiers
from okx_perp_backtest.friction.slippage import NullLiquidityBook, SlippageModel


def make_bars(days=300, seed=3):
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    trend = np.sin(np.linspace(0, 12, n)) * 3000 + np.linspace(0, 4000, n)
    close = 60000 + trend + np.cumsum(rng.normal(0, 35, n))
    funding = np.zeros(n)
    instants = np.isin(idx.hour, [0, 8, 16]) & (idx.minute == 0)
    funding[instants] = rng.normal(0.0001, 0.00005, instants.sum())
    return pd.DataFrame({
        "open": close + rng.normal(0, 15, n),
        "high": close + rng.uniform(10, 60, n),
        "low": close - rng.uniform(10, 60, n),
        "close": close,
        "volume": rng.uniform(200, 800, n),
        "funding": funding,
    }, index=idx)


@pytest.fixture(scope="session")
def bars():
    return make_bars()


@pytest.fixture()
def risk():
    return Risk(capital=10000., fraction=.02, fee_bps=6., slippage_bps=3., max_exposure=1., max_drawdown=.25)


def build_sim(fee_model=None):
    return ExecutionSimulator(
        funding_model=FundingModel(),
        slippage_model=SlippageModel(NullLiquidityBook(), impact_k=1.0),
        fee_model=fee_model or FeeModel.regular_tier(),
        liquidation_model=LiquidationModel(PositionTiers(), "isolated"),
        leverage=1.0,
    )


@pytest.fixture()
def sim():
    return build_sim()
