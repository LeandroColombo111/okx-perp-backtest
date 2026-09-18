import json

import pandas as pd

from okx_perp_backtest.example_signal import ExampleParams, example_signal
from okx_perp_backtest.montecarlo import MonteCarloEngine
from okx_perp_backtest.report import ReportBuilder
from okx_perp_backtest.walkforward import WalkForwardEngine


def test_report_is_json_safe_even_with_dataclass_params_and_timestamps(bars, risk, sim):
    p = ExampleParams()
    equity, trades, _ = sim.run(bars, example_signal(bars, p), p, risk)
    mc = MonteCarloEngine(trades, risk.capital)
    summary = mc.summarize(mc.bootstrap(50, block_size=3))
    wf = WalkForwardEngine(sim, example_signal).run(
        bars, [p], risk, pd.Timedelta(days=90), pd.Timedelta(days=30), pd.Timedelta(days=30),
        warmup=pd.Timedelta(days=30))
    rb = ReportBuilder()
    report = rb.build("test-run", equity, trades, summary, wf, risk.capital, {"params": p})
    rows = rb.to_bigquery_rows(report)
    assert len(rows) == 1 and rows[0]["run_id"] == "test-run"
    json.dumps(rows[0])
    assert 0. <= report["max_drawdown"] <= 1.
