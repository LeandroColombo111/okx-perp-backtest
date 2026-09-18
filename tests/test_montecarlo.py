import numpy as np
import pytest

from okx_perp_backtest.example_signal import ExampleParams, example_signal
from okx_perp_backtest.montecarlo import MonteCarloEngine


@pytest.fixture()
def mc(bars, risk, sim):
    p = ExampleParams()
    _, trades, _ = sim.run(bars, example_signal(bars, p), p, risk)
    return MonteCarloEngine(trades, risk.capital)


def test_bootstrap_and_shuffle_have_one_sharpe_per_simulation(mc):
    assert len(mc.bootstrap(50, block_size=3)) == 50
    assert len(mc.shuffle_order(50)) == 50


def test_summary_percentiles_are_ordered_and_probability_valid(mc):
    s = mc.summarize(mc.bootstrap(200, block_size=3), threshold=1.5)
    assert s["p5"] <= s["p50"] <= s["p95"]
    assert 0. <= s["prob_below_threshold"] <= 1.


def test_shuffle_is_tighter_than_bootstrap(mc):
    np.random.seed(0)
    assert np.std(mc.shuffle_order(300)) < np.std(mc.bootstrap(300))


def test_invalid_inputs_are_refused(mc):
    with pytest.raises(ValueError):
        mc.bootstrap(0)
    with pytest.raises(ValueError):
        MonteCarloEngine([], 10000.).bootstrap(10)
