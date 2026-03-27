# Polymarket A-S Market Maker: AI Model Prediction Markets

An autonomous market-making agent for Polymarket prediction markets that uses the **Avellaneda-Stoikov** model to continuously quote both sides of "which company has the best AI model" markets. Fair value is derived from **Chatbot Arena ELO scores** and **real-time Twitter/news signals**.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                 Python Agent                      │
│  ┌────────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Arena      │ │ Twitter/ │ │ A-S Quoting    │ │
│  │ Scraper    │ │ News RSS │ │ Engine         │ │
│  └─────┬──────┘ └────┬─────┘ └───────┬────────┘ │
│        │             │               │           │
│  ┌─────▼─────────────▼───────────────▼────────┐  │
│  │         Fair Value + Inventory Mgmt         │  │
│  └─────────────────────┬──────────────────────┘  │
│                        │ HTTP                     │
└────────────────────────┼─────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Rust Sidecar      │
              │   (polyfill-rs)     │
              │ • WebSocket stream  │
              │ • In-memory books   │
              │ • REST API @ :8080  │
              └──────────┬──────────┘
                         │ WSS
              ┌──────────▼──────────┐
              │   Polymarket CLOB   │
              └─────────────────────┘
```

## Mathematical Framework

### 1. Fair Value: ELO-to-Probability

For each company `c`, we select its best model's ELO score from the [Chatbot Arena](https://lmarena.ai/leaderboard/text) and convert to probabilities via **temperature-scaled softmax**:

```
τ_eff(t) = τ_base + σ_elo × √(T − t)
p_c = exp(E_c / τ_eff) / Σ_k exp(E_k / τ_eff)
```

**Time adjustment**: Far from expiry, the effective temperature increases, pulling probabilities toward uniform (more uncertainty). Close to expiry, ELO scores dominate.

### 2. Signal Adjustment

Real-time Twitter and news signals shift fair values via **likelihood ratios in odds space**:

```
LR = exp(α × relevance × sentiment × credibility × decay(t))
odds_new = odds_old × LR
p_adjusted = odds_new / (1 + odds_new)
→ Renormalize all outcomes to sum to 1
```

### 3. Avellaneda-Stoikov Quoting

For each outcome token:

**Reservation price** (fair value adjusted for inventory risk):
```
r = p̂ − q × γ × σ² × (T − t)
```

**Optimal spread**:
```
δ = γ × σ² × (T − t) + (2/γ) × ln(1 + γ/κ)
```

**Quotes**:
```
bid = clamp(r − δ/2, 0.01, 0.99)
ask = clamp(r + δ/2, 0.01, 0.99)
```

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Risk aversion | γ | 0.1 | Higher = tighter spreads, faster inventory reduction |
| Order arrival rate | κ | 1.5 | Higher = tighter spreads |
| Base temperature | τ_base | 150 | Controls ELO sensitivity |
| ELO volatility | σ_elo | 10 | Estimated daily ELO std dev |

### 4. Inventory Management

The reservation price naturally handles **inventory skew**: when long an outcome, the reservation price decreases (less eager to buy more, more eager to sell). Quote sizes are also adjusted:

```
bid_size = base × max(0.1, 1 − q/q_max)
ask_size = base × max(0.1, 1 + q/q_max)
```

## Data Sources

| Source | Type | Latency | Purpose |
|--------|------|---------|---------|
| Chatbot Arena | ELO scores | ~minutes | Primary fair value |
| Nitter RSS | Twitter posts | ~minutes | Real-time signals for model releases |
| AI news RSS | News articles | ~hours | Backup signal source |
| Polymarket CLOB | Orderbooks | real-time | Market prices, execution |

## Setup

### Prerequisites

- Python 3.9+
- Rust toolchain (for the sidecar)
- pip dependencies: `pip install -r requirements.txt`

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Rust Sidecar (optional but recommended)

```bash
cd sidecar
cargo build --release
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (all have sensible defaults)
```

### 4. Run

**Live simulation** (paper trading):
```bash
# Start the Rust sidecar (in a separate terminal)
./scripts/start_sidecar.sh

# Run the market-making agent
python3 scripts/run_mm_live.py --interval 30
```

**Backtest**:
```bash
python3 scripts/run_mm_backtest.py
```

**Tests**:
```bash
pytest tests/ -v
```

## Project Structure

```
├── sidecar/                    # Rust orderbook sidecar (polyfill-rs)
│   ├── src/
│   │   ├── main.rs            # HTTP server + WS manager
│   │   ├── orderbook_cache.rs # In-memory BTreeMap orderbooks
│   │   ├── ws_manager.rs      # Polymarket WebSocket streaming
│   │   ├── routes.rs          # REST API handlers
│   │   └── types.rs           # JSON types
│   └── Cargo.toml
├── agent/
│   ├── data_ingestion/
│   │   ├── arena_scraper.py   # Chatbot Arena ELO scraper
│   │   ├── twitter_fetcher.py # Nitter RSS + AI news fetcher
│   │   ├── sidecar_client.py  # HTTP client for Rust sidecar
│   │   └── market_data.py     # Polymarket Gamma API client
│   ├── fair_value/
│   │   ├── elo_model.py       # ELO → probability conversion
│   │   ├── signal_adjuster.py # Signal-based probability adjustment
│   │   └── volatility_estimator.py
│   ├── market_making/
│   │   ├── avellaneda_stoikov.py  # Core A-S quoting engine
│   │   ├── inventory_manager.py   # Multi-outcome inventory
│   │   └── quote_manager.py      # Quote lifecycle
│   ├── scoring/
│   │   └── signal_model.py    # Likelihood ratio computation
│   ├── execution/
│   │   ├── orderbook_analyzer.py
│   │   ├── risk_manager.py
│   │   └── trade_executor.py
│   ├── backtest/
│   │   ├── scenarios.py       # AI model market scenarios
│   │   └── simulator.py       # MM backtest engine
│   └── core.py                # MarketMakingAgent orchestrator
├── config/settings.py         # Configuration + data models
├── scripts/
│   ├── run_mm_live.py         # Live market-making runner
│   ├── run_mm_backtest.py     # Backtest runner
│   └── start_sidecar.sh       # Sidecar launcher
└── tests/
```

## Target Markets

This agent focuses on Polymarket's "which company has the best AI model" prediction markets:

- [Best AI Model End of March](https://polymarket.com/event/which-company-has-the-best-ai-model-end-of-march-751)
- [Best AI Model End of April](https://polymarket.com/event/which-company-has-the-best-ai-model-end-of-april)
- [Best AI Model End of June](https://polymarket.com/event/which-company-has-best-ai-model-end-of-june)

These markets are resolved by the [Chatbot Arena leaderboard](https://lmarena.ai/leaderboard/text) snapshot at month end.

## Configuration

All parameters are configurable via environment variables. See `.env.example` for the full list.

Key A-S parameters:
- `AS_GAMMA` — Risk aversion (default: 0.1)
- `AS_KAPPA` — Order arrival rate (default: 1.5)
- `AS_BASE_ORDER_SIZE` — Base order size in dollars (default: 10)
- `AS_MAX_INVENTORY` — Max inventory per outcome (default: 100)
- `ELO_TAU_BASE` — Base temperature for softmax (default: 150)
- `ELO_SIGMA_DAILY` — Daily ELO volatility estimate (default: 10)
