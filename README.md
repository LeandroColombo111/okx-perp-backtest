# okx-perp-backtest

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
pip install -e .
python examples/demo.py
```

The demo generates synthetic OHLCV data (no API keys or network access
needed) and runs the full pipeline: execution with friction models → Monte
Carlo → walk-forward → report.

## Layout

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
