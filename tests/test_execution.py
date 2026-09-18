import pytest

from okx_perp_backtest.example_signal import ExampleParams, example_signal
from okx_perp_backtest.friction.fees import FeeModel
from conftest import build_sim


def test_run_returns_curve_trades_and_records_equity_before(bars, risk, sim):
    p = ExampleParams()
    equity, trades, state = sim.run(bars, example_signal(bars, p), p, risk)
    assert len(equity) == len(bars) and trades
    assert all(t["equity_before"] > 0 for t in trades)
    assert {t["side"] for t in trades} <= {1, -1}


def test_higher_fees_reduce_final_equity(bars, risk):
    p = ExampleParams()
    signal = example_signal(bars, p)
    cheap, _, _ = build_sim(FeeModel(0., 0., 0.)).run(bars, signal, p, risk)
    dear, _, _ = build_sim(FeeModel(20., 50., 50.)).run(bars, signal, p, risk)
    assert cheap.iloc[-1] > dear.iloc[-1]


def test_run_is_deterministic(bars, risk, sim):
    p = ExampleParams()
    a, _, _ = sim.run(bars, example_signal(bars, p), p, risk)
    b, _, _ = sim.run(bars, example_signal(bars, p), p, risk)
    assert a.equals(b)


def test_leverage_must_be_positive():
    from okx_perp_backtest.execution import ExecutionSimulator
    good = build_sim()
    with pytest.raises(ValueError):
        ExecutionSimulator(good.funding_model, good.slippage_model, good.fee_model, good.liquidation_model, leverage=0)


def test_empty_range_is_refused(bars, risk, sim):
    p = ExampleParams()
    with pytest.raises(ValueError):
        sim.run(bars, example_signal(bars, p), p, risk, start="2030-01-01")
