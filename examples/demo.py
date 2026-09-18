"""End-to-end demo on synthetic data: execution -> Monte Carlo -> walk-forward -> report.

Uses example_signal.py, a deliberately simple EMA crossover -- not a real
strategy. Swap it for your own signal_fn to backtest your own bot.
"""
import numpy as np
import pandas as pd

from okx_perp_backtest.account import Risk
from okx_perp_backtest.example_signal import ExampleParams, example_signal
from okx_perp_backtest.execution import ExecutionSimulator
from okx_perp_backtest.friction.funding import FundingModel
from okx_perp_backtest.friction.slippage import SlippageModel, NullLiquidityBook
from okx_perp_backtest.friction.fees import FeeModel
from okx_perp_backtest.friction.liquidation import LiquidationModel, PositionTiers
from okx_perp_backtest.montecarlo import MonteCarloEngine
from okx_perp_backtest.walkforward import WalkForwardEngine
from okx_perp_backtest.report import ReportBuilder


def synthetic_bars(n_hours=24 * 220, seed=3) -> pd.DataFrame:
    """Random-walk-with-drift OHLCV + OKX-style funding, only to exercise the engine without needing real market data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="1h", tz="UTC")
    trend = np.sin(np.linspace(0, 10, n_hours)) * 3000 + np.linspace(0, 4000, n_hours)
    noise = np.cumsum(rng.normal(0, 35, n_hours))
    close = 60000 + trend + noise
    openp = close + rng.normal(0, 15, n_hours)
    high = np.maximum(openp, close) + rng.uniform(10, 60, n_hours)
    low = np.minimum(openp, close) - rng.uniform(10, 60, n_hours)
    volume = rng.uniform(200, 800, n_hours)
    funding = np.zeros(n_hours)
    funding_instants = np.isin(idx.hour, [0, 8, 16]) & (idx.minute == 0)
    funding[funding_instants] = rng.normal(0.0001, 0.00005, funding_instants.sum())
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume, "funding": funding},
        index=idx,
    )


def main():
    print("Synthetic random-walk data and a toy signal: negative or noisy results are EXPECTED here.")
    print("This demo shows the plumbing (frictions, Monte Carlo, walk-forward, report), not a strategy.\n")
    bars = synthetic_bars()
    risk = Risk(capital=10000., fraction=.02, fee_bps=6., slippage_bps=3., max_exposure=1., max_drawdown=.25)
    params = ExampleParams()

    sim = ExecutionSimulator(
        funding_model=FundingModel(),
        slippage_model=SlippageModel(NullLiquidityBook(), impact_k=1.0),
        fee_model=FeeModel.regular_tier(),
        liquidation_model=LiquidationModel(PositionTiers(), "isolated"),
        leverage=1.0,
    )

    signal = example_signal(bars, params)
    equity, trades, _ = sim.run(bars, signal, params, risk)
    print(f"1) execution      -> {len(trades)} trades")

    mc = MonteCarloEngine(trades, risk.capital)
    mc_summary = mc.summarize(mc.bootstrap(1000, block_size=5))
    print(f"2) monte carlo    -> p5={mc_summary['p5']:.2f} p50={mc_summary['p50']:.2f} "
          f"p95={mc_summary['p95']:.2f} prob_below_1.5={mc_summary['prob_below_threshold']:.2%}")

    wf = WalkForwardEngine(sim, signal_fn=example_signal)
    grid = [ExampleParams(fast=12, slow=48), ExampleParams(fast=8, slow=32)]
    # warmup gives each window's indicators history before it starts; use the same value for every variant you compare.
    wf_results = wf.run(bars, grid, risk, train_span=pd.Timedelta(days=90),
                         test_span=pd.Timedelta(days=30), step=pd.Timedelta(days=30),
                         warmup=pd.Timedelta(days=30))
    print(f"3) walk-forward   -> {len(wf_results)} windows")
    print(wf_results[["train_start", "test_start", "in_sample_sharpe", "out_of_sample_sharpe"]].to_string(index=False))

    report = ReportBuilder().build(
        run_id="demo-run-001", equity=equity, trades=trades, mc_summary=mc_summary,
        wf_results=wf_results, capital=risk.capital, params={"fast": params.fast, "slow": params.slow},
    )
    print(f"4) report         -> sharpe={report['sharpe']:.2f} max_drawdown={report['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
