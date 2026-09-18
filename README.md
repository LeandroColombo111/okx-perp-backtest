# okx-perp-backtest

[![tests](https://github.com/LeandroColombo111/okx-perp-backtest/actions/workflows/tests.yml/badge.svg)](https://github.com/LeandroColombo111/okx-perp-backtest/actions/workflows/tests.yml)

A backtesting engine for OKX crypto perpetual futures that models real trading
friction instead of assuming frictionless fills at the close price.

Most backtests answer "what would this signal have earned on historical
candles?" This one also answers: does the edge survive real funding
payments, realistic slippage, real exchange fees, forced liquidation, and a
reshuffled trade sequence? A backtest that only clears the first question is
not enough evidence to risk capital on.

## What it models

- **Funding** settled at OKX's exact UTC instants (00:00 / 08:00 / 16:00),
  not smeared across every candle.
- **Slippage** as market impact, not a flat percentage: it walks a real
  order book when historical L2 depth is available, and falls back to a
  square-root impact model (calibratable against your own live fill data)
  when it isn't.
- **Fees** from OKX's real maker/taker schedule, keyed by the simulated
  order type of each fill (a breakout entry is a taker fill; a limit target
  is a maker fill).
- **Liquidation** as a distinct event from your strategy's own stop-loss:
  the exchange force-closes you when margin falls below the tiered
  maintenance requirement, regardless of what your stop says.

## What it answers beyond a single Sharpe number

- **Monte Carlo** over the closed trade sequence: bootstrap resampling,
  order shuffling, and entry/exit timing perturbation, producing a
  *distribution* of Sharpe ratios (p5/p50/p95, and P(Sharpe < your
  threshold)) instead of one point estimate that could be a lucky sequence.
- **Walk-forward analysis**: rolling train/test windows comparing in-sample
  vs out-of-sample performance, to catch overfitting to the past before it
  costs real money.
- **A flat report schema** (equity curve, drawdown, per-trade return
  distribution, Monte Carlo summary) ready to insert into BigQuery to
  compare runs against each other over time.

## Case study: what it found in a real bot

The engine was built to validate a long/short BTC perpetuals bot that runs in
demo/paper trading. Running it changed what I believed about that bot. All
figures are backtests on BTC, 14 rolling 90-day windows, parameters fixed (never
re-optimized per window), identical warmup for every variant, compared against
buy-and-hold on the same windows. They are not live results.

- **The bot was in the market only ~15% of the time.** In the quarters where BTC
  rose, buy-and-hold made about +34% on average and the bot about +3%. In the
  quarters where BTC fell, buy-and-hold lost about -17% and the bot made about
  +4%. It protects capital but captures roughly a tenth of the upside.
- **It does not beat buy-and-hold on return** (about +57% vs +192% over the
  same windows), though its worst window was far smaller (-11% vs -27%). On
  ETH and SOL it showed no risk-adjusted edge.
- **A hybrid is the realistic use.** Half buy-and-hold, half bot cut the worst
  drawdown from about 54% to 24% for a return of +143% instead of +192%. The
  confidence interval on the Sharpe improvement includes zero, so this is
  suggestive, not proof.
- **Ideas that looked good did not survive checking.** About 44 variants were
  tried on the same history, which inflates the odds that something looks good
  by luck. A faster-entry variant that was the best on BTC did worse on ETH and
  SOL and on neighbouring parameter values, so it was rejected.

## A mistake this tool caught in its own author

An early result said the bot "fails out of sample" and that a macro filter fixed
it. Both claims came from an unfair comparison: the baseline was run with no
warmup and the variant with warmup. With no history before each 90-day test
window, the baseline's slow indicators never converged and it barely traded.
Compared like-for-like, the baseline was slightly *better* than the "improved"
version. The conclusion was retracted before anything was changed.

That is why `WalkForwardEngine.run()` takes a `warmup` argument, why
`tests/test_walkforward.py` has a regression test for exactly this failure, and
why the docstring says every variant you compare must use the same warmup.

## What's intentionally NOT here

The `example_signal.py` module is a bare EMA crossover — just enough to
exercise the engine end-to-end. The actual trading strategy this engine was
built to validate is not included; the engine is strategy-agnostic by
design, so plug in your own signal generator (see the docstring in
`execution.ExecutionSimulator.step` for the exact column contract it
expects: `signal`, `anchor`, `atr`).

## Status

Built to validate a bot that trades OKX BTC-USDT perpetuals long/short on a
1h signal anchored to a 4h trend filter, currently running in demo/paper
trading only. This is a backtesting and research tool, not a live trading
system, and nothing here places real orders.

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest -q      # 24 tests
python examples/demo.py
```

The demo generates synthetic OHLCV data (no API keys or network access
needed) and runs the full pipeline: execution with friction models → Monte
Carlo → walk-forward → report.

## Layout (tests in `tests/`)

```
src/okx_perp_backtest/
├── account.py          # Risk/Account dataclasses, position sizing, Sharpe/drawdown metrics
├── example_signal.py   # placeholder signal (NOT a production strategy)
├── execution.py         # ExecutionSimulator: orchestrates funding/liquidation/fees/slippage per bar
├── montecarlo.py        # bootstrap, shuffle, and timing-perturbation Sharpe distributions
├── walkforward.py       # rolling train/test window analysis
├── report.py            # flat, BigQuery-ready run report
└── friction/
    ├── funding.py        # 8h OKX funding settlement
    ├── slippage.py        # order-book walk / square-root impact proxy
    ├── fees.py            # OKX maker/taker/liquidation fee schedule
    └── liquidation.py     # tiered maintenance-margin liquidation price
```

## Disclaimer

This is a research and engineering project, not financial advice. Backtests,
including the friction-adjusted ones here, cannot guarantee future
performance. Nothing in this repository executes real trades.
