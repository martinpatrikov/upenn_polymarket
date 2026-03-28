# Autonomous Market Maker for AI Prediction Markets

**An Avellaneda-Stoikov market-making agent that continuously quotes Polymarket's "Best AI Model" markets by synthesizing Chatbot Arena ELO scores, real-time news signals, and orderbook microstructure — all through a Rust-powered data pipeline.**

> The AI model race is the most actively traded prediction market category on Polymarket.
> New model releases cause instant repricing. Arena leaderboard updates shift probabilities overnight.
> This agent detects those shifts before the market does, quotes both sides of every outcome, and profits from the convergence.

---

### How It Works — In 30 Seconds

1. **Scrapes the [Chatbot Arena](https://lmarena.ai/leaderboard/text) leaderboard** → converts ELO scores to win probabilities via temperature-scaled softmax
2. **Ingests real-time AI news** from Twitter/RSS → adjusts probabilities using Bayesian likelihood ratios
3. **Blends model output with market prices** → weighted by time-to-expiry (trust ELO more as resolution approaches)
4. **Generates optimal bid/ask quotes** using the Avellaneda-Stoikov framework → inventory-adjusted reservation price + spread
5. **Executes against Polymarket's CLOB** via a Rust sidecar that streams orderbooks over WebSocket

The result: **5-cent spreads, 70-80% win rate, Sharpe 2-5 across backtested scenarios.**

---

### Backtest Results

Simulated across 4 scenarios representing distinct market regimes:

| Scenario | Narrative | P&L | Sharpe | Win Rate | Trades |
|----------|-----------|-----|--------|----------|--------|
| GPT-5 Hype Cycle | Market overreacts to launch; agent sells the hype | +$63 | 7.50 | 78% | 74 |
| Gemini 2.5 Takeover | Google tops Arena; agent buys early on ELO signal | +$7 | 0.50 | 70% | 57 |
| Quiet Market | No launches; pure spread capture | +$45 | 5.88 | 80% | 51 |
| DeepSeek Pump & Dump | Formatting tricks inflate ELO; agent sells the spike | +$67 | 4.95 | 78% | 69 |
| **Aggregate** | | **+$182** | **4.71** | **77%** | **251** |

<p align="center">
  <img src="output/dashboard.png" width="800" alt="Performance Dashboard">
</p>

<details>
<summary><b>Equity Curves & Fair Value Charts</b> (click to expand)</summary>

#### GPT-5 Hype Cycle
*Agent detects that Arena ELO barely moves despite market pricing OpenAI at 52¢. Sells into the hype, profits as market corrects to 29¢.*

<img src="output/equity_gpt_5_hype_cycle.png" width="700">
<img src="output/fv_vs_market_gpt_5_hype_cycle.png" width="700">

#### Gemini 2.5 Arena Takeover
*Google genuinely tops Arena (ELO 1470→1530). Agent's ELO model detects immediately, buys at 30¢ before market catches up to 58¢.*

<img src="output/equity_gemini_2_5_arena_takeover.png" width="700">
<img src="output/fv_vs_market_gemini_2_5_arena_takeover.png" width="700">

#### DeepSeek Pump & Dump
*DeepSeek-R2 briefly tops Arena via formatting tricks. Market panics to 42¢. Arena applies style control, ELO crashes. Agent sells the spike.*

<img src="output/equity_deepseek_pump_dump.png" width="700">
<img src="output/fv_vs_market_deepseek_pump_dump.png" width="700">

#### Cumulative Edge Capture
<img src="output/cumulative_edge.png" width="700">

</details>

The full backtest terminal output is saved in [`output/backtest_terminal_log.txt`](output/backtest_terminal_log.txt).

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                 Python Agent                      │
│  ┌────────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Arena      │ │ AI News  │ │ Avellaneda-    │ │
│  │ Scraper    │ │ RSS      │ │ Stoikov Engine │ │
│  └─────┬──────┘ └────┬─────┘ └───────┬────────┘ │
│        │             │               │           │
│  ┌─────▼─────────────▼───────────────▼────────┐  │
│  │     ELO Fair Value → Signal Adjustment      │  │
│  │     → Blending → Inventory Management       │  │
│  └─────────────────────┬──────────────────────┘  │
│                        │ HTTP                     │
└────────────────────────┼─────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Rust Sidecar      │
              │   (polyfill-rs)     │
              │ • WebSocket stream  │
              │ • BTreeMap books    │
              │ • REST API @ :8080  │
              └──────────┬──────────┘
                         │ WSS
              ┌──────────▼──────────┐
              │   Polymarket CLOB   │
              └─────────────────────┘
```

**Why a Rust sidecar?** Polymarket's CLOB streams orderbook deltas over WebSocket. Processing these in Python introduces GIL contention and latency. The Rust sidecar (built on [polyfill-rs](https://github.com/nicksenger/polyfill-rs)) maintains in-memory BTreeMap orderbooks with zero-copy updates and serves snapshots to the Python agent via local HTTP — giving us microsecond book updates with millisecond strategy execution.

---

## Mathematical Framework

### Fair Value: ELO → Probability

Each company's best model ELO score is converted to a win probability via **temperature-scaled softmax with time adjustment**:

```
τ_eff(t) = τ_base + σ_elo × √(T − t)

p_c = exp(E_c / τ_eff) / Σ_k exp(E_k / τ_eff)
```

The effective temperature `τ_eff` increases with time-to-expiry: far from resolution, probabilities are pulled toward uniform (high uncertainty); close to resolution, ELO rankings dominate. This is then **blended with market prices** using a time-dependent weight — trusting market wisdom more when expiry is distant, and trusting ELO more as the Arena snapshot approaches.

### Signal Adjustment: Bayesian Likelihood Ratios

Real-time news signals (model launches, benchmark results, Arena updates) shift probabilities via likelihood ratios in odds space:

```
LR = exp(α × relevance × sentiment × credibility × decay(t))
odds_new = odds_old × LR  →  renormalize to sum to 1
```

Each signal is scored for relevance (keyword matching), sentiment (VADER), and source credibility (Twitter official accounts > news > generic). Time decay ensures stale signals fade.

### Avellaneda-Stoikov Quoting

For each outcome, the reservation price adjusts fair value for inventory risk, and the optimal spread balances adverse selection against order flow:

```
Reservation:  r = p̂ − q × γ × σ² × T_norm
Spread:       δ = γ × σ² × T_norm + (2/γ) × ln(1 + γ/κ)
Quotes:       bid = r − δ/2,  ask = r + δ/2
```

Time is normalized to `[0,1]` to prevent spread explosion on long-dated markets. With `γ=0.5, κ=40`, this produces **~5-cent spreads** — competitive for Polymarket's AI model markets.

| Parameter | Value | Effect |
|-----------|-------|--------|
| γ (risk aversion) | 0.5 | Moderate — balances spread capture vs inventory control |
| κ (order arrival) | 40 | High — reflects decent prediction market flow |
| τ_base | 150 | Base ELO temperature — controls probability sharpness |
| σ_elo | 10 | Daily ELO std dev — drives volatility estimates |

---

## Data Sources

| Source | What | Latency | Role |
|--------|------|---------|------|
| [Chatbot Arena](https://lmarena.ai/leaderboard/text) | ELO scores for 300+ models | ~hours | **Primary fair value signal** — resolution oracle |
| AI News RSS (6 feeds) | Model launches, benchmarks | ~minutes | **Signal adjustment** — Bayesian prior updates |
| Polymarket CLOB | Live orderbooks | real-time | **Market prices + execution** |
| Polymarket Gamma API | Event metadata, token IDs | ~seconds | **Market discovery** |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (all defaults work for paper trading)
cp .env.example .env

# Run backtest simulation
python3 scripts/run_mm_backtest.py

# Run live paper trading (optional: start Rust sidecar first)
./scripts/start_sidecar.sh          # Terminal 1
python3 scripts/run_mm_live.py      # Terminal 2

# Run tests
pytest tests/ -v                     # 34 tests
```

---

## Project Structure

```
├── agent/
│   ├── core.py                     # Main orchestrator — runs the trading loop
│   ├── data_ingestion/
│   │   ├── arena_scraper.py        # Chatbot Arena ELO scraper
│   │   ├── twitter_fetcher.py      # AI news RSS aggregator
│   │   ├── sidecar_client.py       # HTTP client for Rust sidecar
│   │   └── market_data.py          # Polymarket Gamma API client
│   ├── fair_value/
│   │   ├── elo_model.py            # ELO → probability (softmax + market blend)
│   │   ├── signal_adjuster.py      # Bayesian likelihood ratio adjustment
│   │   └── volatility_estimator.py # Per-outcome σ from ELO uncertainty
│   ├── market_making/
│   │   ├── avellaneda_stoikov.py   # Core A-S reservation price + spread
│   │   ├── inventory_manager.py    # Multi-outcome position tracking
│   │   └── quote_manager.py        # Quote lifecycle + simulated fills
│   └── backtest/
│       ├── scenarios.py            # 4 AI model market scenarios
│       ├── simulator.py            # Replay engine with adverse selection
│       └── report_generator.py     # Rich terminal + matplotlib + markdown
├── sidecar/                        # Rust orderbook service (polyfill-rs)
├── config/settings.py              # All parameters + data models
├── output/                         # Backtest charts + reports
│   ├── backtest_report.md
│   ├── dashboard.png
│   ├── backtest_terminal_log.txt
│   └── *.png                       # Equity curves, fair value charts
├── scripts/
│   ├── run_mm_live.py              # Live paper trading
│   ├── run_mm_backtest.py          # Backtest simulation
│   └── start_sidecar.sh            # Rust sidecar launcher
└── tests/                          # 34 unit tests
```

---

## Target Markets

Polymarket's **"Best AI Model"** prediction markets, resolved monthly by the Chatbot Arena leaderboard:

- [End of April](https://polymarket.com/event/which-company-has-the-best-ai-model-end-of-april)
- [End of May](https://polymarket.com/event/which-company-has-the-best-ai-model-end-of-may)
- [End of June](https://polymarket.com/event/which-company-has-best-ai-model-end-of-june)

Outcomes include: Anthropic, OpenAI, Google, xAI, DeepSeek, Meta, Mistral, and others.

---

## Configuration

All parameters are configurable via environment variables. See [`.env.example`](.env.example) for the full list with defaults.
