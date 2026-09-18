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
- **Slippage** as market impact, not a flat percentage: the model walks an
  order book through the `LiquidityBook` interface when one is plugged in,
  and otherwise uses a square-root impact model whose constant you calibrate
  against your own fills. No book data loader ships with this repo, so out of
  the box it always uses the impact model (see Known limitations).
- **Fees** from OKX's public regular-tier maker/taker schedule for USDT
  perpetuals (2 / 5 bps), keyed by the simulated order type of each fill (a
  breakout entry is a taker fill; a limit target is a maker fill). Liquidations
  are billed at the taker rate, as OKX does.
- **Liquidation** as a distinct event from your strategy's own stop-loss:
  the exchange force-closes you when margin falls below a tiered maintenance
  requirement, regardless of what your stop says. Tiers follow OKX's real
  size-based table (a dated offline snapshot for BTC-USDT, or fetched live with
  `PositionTiers.fetch_okx()`). Isolated margin only.

## What it answers beyond a single Sharpe number

- **Monte Carlo** over the closed trade sequence: bootstrap resampling,
  order shuffling, and a latency test that delays the whole signal by a
  few bars and re-simulates, producing a
  *distribution* of Sharpe ratios (p5/p50/p95, and P(Sharpe < your
  threshold)) instead of one point estimate that could be a lucky sequence.
- **Walk-forward analysis**: rolling train/test windows comparing in-sample
  vs out-of-sample performance, to catch overfitting to the past before it
  costs real money.
- **A flat report schema** (equity curve, drawdown, per-trade return
  distribution, Monte Carlo summary) shaped as rows you can load into
  BigQuery to compare runs over time. The upload itself is not included.

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

## Known limitations

Read these before trusting a number from this engine.

- **No order book data ships with it.** The book-walk logic and the
  `LiquidityBook` interface exist and are unit tested, but nothing loads
  historical L2 data, so the default `NullLiquidityBook` makes every fill use
  the impact model. OKX's own historical book coverage also has gaps.
- **`impact_k` is an uncalibrated placeholder (1.0).** Calibrate it from your
  own fills. In the author's BTC runs at 1x exposure the result barely moved
  between `impact_k=0` and `10`, but that will not hold for larger sizes or
  thinner markets.
- **The default liquidation tier table is a dated snapshot** of OKX's BTC-USDT
  tiers (see `friction/okx_tiers_snapshot.py`). OKX revises tiers, so call
  `PositionTiers.fetch_okx()` for a fresh table before trusting a liquidation
  distance, and pass `inst_id`/`inst_family` for other instruments. The
  liquidation price also ignores the entry fee already paid, so it looks
  marginally farther away than it is.
- **Isolated margin only.** Cross margin raises `NotImplementedError` instead of
  being simulated badly.
- **Regular fee tier only.** No VIP levels, rebates or token discounts.
- **Funding uses the bar's open as the mark price**, an approximation.
- **One instrument, one position at a time.** No portfolios or hedged legs.
- **Monte Carlo resamples percentage returns on the original time skeleton.**
  Compounding-order effects are ignored, and the latency test only delays
  signals (advancing them would look ahead).
- **Walk-forward with fixed parameters measures stability, not prediction,** if
  those parameters were chosen on the same history. Trying many variants on one
  history inflates the chance that one looks good by luck.
- **The tests use synthetic data.** The real-market numbers in the case study
  came from private runs on real OKX data and are not reproducible from this
  repo alone. The buy-and-hold comparisons there exclude fees and funding.

## Roadmap

In the order I would do them:

1. ~~Load OKX's position tiers from its public endpoint~~ (done: `PositionTiers.fetch_okx()`).
2. A loader for OKX historical L2 book data, with explicit handling of the coverage gaps.
3. Calibrate `impact_k` from real fills once a bot has enough of them.
4. Cross margin and multi-position accounts.
5. A BigQuery uploader for the report rows.
6. A loader for OKX candles and funding history into the bar format above.

## Status

Built to validate a bot that trades OKX BTC-USDT perpetuals long/short on a
1h signal anchored to a 4h trend filter, currently running in demo/paper
trading only. This is a backtesting and research tool, not a live trading
system, and nothing here places real orders.

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest -q      # 44 tests
python examples/demo.py
```

The demo generates synthetic OHLCV data (no API keys or network access
needed) and runs the full pipeline: execution with friction models → Monte
Carlo → walk-forward → report.

## Use it with your own data and signal

**1. Bars.** A `DataFrame` of 1h candles, indexed by the bar's **open** time in UTC:

| column | meaning |
|---|---|
| `open` `high` `low` `close` | prices |
| `volume` | traded volume in base units (the slippage model sizes impact against it) |
| `funding` | OKX funding rate as a **fraction** per 8h (`0.0001` = 0.01%; positive means longs pay). Read only on bars opening at 00:00, 08:00 and 16:00 UTC, so it may be `0` elsewhere |

```python
bars = pd.read_csv("btc_1h.csv", index_col="time", parse_dates=True)
bars.index = pd.to_datetime(bars.index, utc=True)
validate_bars(bars)   # raises one error listing every problem: gaps, timezone, NaN, bad OHLC, percent-style funding...
```

**2. Signal.** A `DataFrame` on the same index with three columns:

| column | meaning |
|---|---|
| `signal` | desired entry: `+1` long, `-1` short, `0` none |
| `anchor` | higher-timeframe trend (`+1`/`-1`/`0`); an open position is closed when it stops agreeing. With no such trend, pass the same series as `signal` |
| `atr` | volatility in price units; sets the stop distance (`stop_atr * atr`). Must be `> 0` wherever `signal != 0` |

Row *i* may use only information up to the **close** of bar *i*; the engine acts on it at the **open** of bar *i+1*. Calling
`validate_signal(signal, bars)` checks the contract, and `run()` refuses a signal whose index differs from the bars.

**3. Params.** Any object exposing `stop_atr`, `trail_atr`, `reward`, `trail_start_r` and `max_hours`
(a dataclass works). See [`examples/own_signal.py`](examples/own_signal.py) for a complete, runnable
example that wires a custom signal through validation, the friction models and the metrics:

```bash
python examples/own_signal.py
```

When you compare variants in walk-forward, give every one the same `warmup` (see the case study above).

## Layout (tests in `tests/`)

```
src/okx_perp_backtest/
├── data.py             # validate_bars / validate_signal: input checks with actionable errors
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
