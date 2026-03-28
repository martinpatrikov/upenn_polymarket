"""Market-making backtest simulator — replay engine.

Walks through scenario events, calls the real A-S + ELO models,
records every decision, and tracks data for charting.

Includes realistic market microstructure effects:
  - Adverse selection: ~30% of fills experience slippage (0.5-2.5¢)
  - Inventory mark-to-market creates natural drawdowns between events
  - Not all trades are profitable — some get filled at adverse prices
"""

from __future__ import annotations

import logging
import math
import hashlib
from datetime import datetime, timezone, timedelta

from config.settings import (
    Settings,
    BacktestResult,
    NewsSignal,
    Side,
)
from agent.fair_value.elo_model import EloFairValueModel
from agent.fair_value.signal_adjuster import SignalAdjuster
from agent.fair_value.volatility_estimator import VolatilityEstimator
from agent.market_making.avellaneda_stoikov import AvellanedaStoikov
from agent.market_making.inventory_manager import InventoryManager
from agent.backtest.scenarios import MMScenario, MMScenarioEvent

logger = logging.getLogger(__name__)

# --- Realistic market microstructure noise ---
# Fraction of fills that experience adverse slippage
ADVERSE_FILL_RATE = 0.38
# Max adverse slippage in price units (5 cents — can exceed half-spread,
# creating genuinely losing trades)
MAX_ADVERSE_SLIPPAGE = 0.050
# Seed for deterministic "randomness" (reproducible results)
NOISE_SEED = 42


class MMBacktestSimulator:
    """Replays A-S market-making scenarios through the full model stack."""

    def __init__(self, config: Settings | None = None):
        self.config = config or Settings()

        self.elo_model = EloFairValueModel(
            tau_base=self.config.elo_tau_base,
            sigma_elo_daily=self.config.elo_sigma_daily,
        )
        self.signal_adjuster = SignalAdjuster(
            alpha=self.config.signal_alpha,
            decay_half_life_hours=self.config.decay_half_life_hours,
        )
        self.volatility = VolatilityEstimator(
            sigma_elo_daily=self.config.elo_sigma_daily,
        )
        # Use larger order sizes for backtest to produce meaningful P&L
        # (live trading uses config values; backtest amplifies for visibility)
        backtest_order_size = self.config.as_base_order_size * 10  # 100 shares
        backtest_max_inventory = self.config.as_max_inventory * 10  # 1000 max
        self.as_engine = AvellanedaStoikov(
            gamma=self.config.as_gamma,
            kappa=self.config.as_kappa,
            base_order_size=backtest_order_size,
            max_inventory=backtest_max_inventory,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scenario(self, scenario: MMScenario) -> BacktestResult:
        """Run a single scenario through the full A-S pipeline."""
        inventory = InventoryManager(
            max_inventory_per_outcome=self.config.as_max_inventory,
        )
        trades: list[dict] = []
        portfolio_values: list[tuple[datetime, float]] = []
        event_snapshots: list[dict] = []
        start_time = datetime.now(timezone.utc)
        total_expiry_hours = scenario.duration_hours

        for event in scenario.events:
            timestamp = start_time + timedelta(hours=event.hours_offset)

            # ---- 1. Days remaining ----
            days_remaining = max(
                0.01, (total_expiry_hours - event.hours_offset) / 24.0
            )

            # ---- 2. ELO base probabilities ----
            elo_probs = self.elo_model.compute_probabilities(
                event.elo_scores, days_remaining
            )

            # ---- 3. Blend with market prices ----
            blended = self.elo_model.blend_with_market(
                elo_probs, event.market_prices, days_remaining
            )

            # ---- 4. News signals ----
            signals: list[NewsSignal] = []
            for item in event.news_items:
                signals.append(
                    NewsSignal(
                        source=item.source,
                        title=item.title,
                        summary=item.summary,
                        url="",
                        published_at=start_time + timedelta(hours=item.hours_offset),
                        relevance_score=0.8,
                        sentiment_score=0.6,
                        source_credibility=0.7,
                    )
                )

            fair_values = self.signal_adjuster.adjust_probabilities(blended, signals)

            # ---- 5. Volatility ----
            tau_eff = self.elo_model._effective_temperature(days_remaining)
            sigmas = self.volatility.estimate(fair_values, tau_eff)

            # ---- 6. A-S quotes ----
            token_ids = {o: f"sim_{o}" for o in scenario.outcomes}
            inventories = inventory.get_all_inventories()
            quotes = self.as_engine.generate_quotes(
                fair_values=fair_values,
                inventories=inventories,
                sigmas=sigmas,
                time_to_expiry=days_remaining,
                token_ids=token_ids,
            )

            # ---- 7. Simulate fills (with realistic adverse selection) ----
            event_fills: list[dict] = []
            trade_counter = len(trades)
            event_idx = scenario.events.index(event)

            for quote in quotes:
                mkt_price = event.market_prices.get(quote.outcome)
                if mkt_price is None:
                    continue

                spread = max(quote.spread, 0.001)

                # --- BUY fill: our bid vs market best ask (approx mkt_price) ---
                if quote.bid_size > 0:
                    fill_prob = _fill_probability(quote.bid_price, mkt_price, spread)
                    if (
                        fill_prob > 0.15
                        and not inventory.should_reduce(quote.outcome, Side.BUY)
                    ):
                        fill_size = quote.bid_size * fill_prob
                        # Apply adverse selection slippage
                        fill_price, is_adverse = _apply_slippage(
                            quote.bid_price, "BUY", event_idx,
                            quote.outcome, trade_counter,
                        )
                        edge = quote.fair_value - fill_price  # Can be negative!
                        inventory.update(
                            quote.outcome, Side.BUY, fill_size, fill_price
                        )
                        fill_rec = {
                            "time": timestamp.isoformat(),
                            "outcome": quote.outcome,
                            "side": "BUY",
                            "price": fill_price,
                            "size": fill_size,
                            "fair_value": quote.fair_value,
                            "market_price": mkt_price,
                            "edge": edge,
                            "adverse": is_adverse,
                        }
                        trades.append(fill_rec)
                        event_fills.append(fill_rec)
                        trade_counter += 1

                # --- SELL fill: our ask vs market best bid (approx mkt_price) ---
                if quote.ask_size > 0:
                    fill_prob = _fill_probability(mkt_price, quote.ask_price, spread)
                    if (
                        fill_prob > 0.15
                        and not inventory.should_reduce(quote.outcome, Side.SELL)
                    ):
                        fill_size = quote.ask_size * fill_prob
                        # Apply adverse selection slippage
                        fill_price, is_adverse = _apply_slippage(
                            quote.ask_price, "SELL", event_idx,
                            quote.outcome, trade_counter,
                        )
                        edge = fill_price - quote.fair_value  # Can be negative!
                        inventory.update(
                            quote.outcome, Side.SELL, fill_size, fill_price
                        )
                        fill_rec = {
                            "time": timestamp.isoformat(),
                            "outcome": quote.outcome,
                            "side": "SELL",
                            "price": fill_price,
                            "size": fill_size,
                            "fair_value": quote.fair_value,
                            "market_price": mkt_price,
                            "edge": edge,
                            "adverse": is_adverse,
                        }
                        trades.append(fill_rec)
                        event_fills.append(fill_rec)
                        trade_counter += 1

            # ---- 8. Portfolio value ----
            inv_state = inventory.get_state(event.market_prices)
            pv = (
                self.config.starting_bankroll
                + inv_state.realized_pnl
                + inv_state.unrealized_pnl
            )
            portfolio_values.append((timestamp, pv))

            # ---- 9. Event snapshot for charting ----
            quotes_dict: dict[str, dict] = {}
            for q in quotes:
                quotes_dict[q.outcome] = {
                    "bid": q.bid_price,
                    "ask": q.ask_price,
                    "reservation": q.reservation_price,
                    "spread": q.spread,
                }

            event_snapshots.append(
                {
                    "hours": event.hours_offset,
                    "elo_probs": dict(elo_probs),
                    "fair_values": dict(fair_values),
                    "market_prices": dict(event.market_prices),
                    "quotes": quotes_dict,
                    "inventories": dict(inventory.get_all_inventories()),
                    "portfolio_value": pv,
                    "news": [item.title for item in event.news_items],
                    "fills": event_fills,
                }
            )

        # ==================================================================
        # Post-processing: compute metrics
        # ==================================================================
        final_prices = scenario.events[-1].market_prices if scenario.events else {}
        final_state = inventory.get_state(final_prices)
        final_pnl = final_state.realized_pnl + final_state.unrealized_pnl
        return_pct = (final_pnl / self.config.starting_bankroll) * 100

        # -- Win rate (based on edge sign: positive edge = win) --
        wins = sum(1 for t in trades if t.get("edge", 0) > 0)
        win_rate = wins / len(trades) if trades else 0.0

        # -- Max drawdown --
        max_dd = 0.0
        peak = self.config.starting_bankroll
        for _, pv in portfolio_values:
            peak = max(peak, pv)
            dd = (peak - pv) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # -- Per-event returns (for Sharpe / Sortino) --
        returns: list[float] = []
        prev_pv = self.config.starting_bankroll
        for _, pv in portfolio_values:
            ret = (pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0
            returns.append(ret)
            prev_pv = pv

        scenario_days = max(total_expiry_hours / 24.0, 1.0)

        # -- Sharpe Ratio --
        sharpe = _sharpe(returns, scenario_days)

        # -- Sortino Ratio --
        sortino = _sortino(returns, scenario_days)

        # -- Profit Factor (from trade edges, not returns) --
        gross_profits = sum(t["edge"] * t["size"] for t in trades if t.get("edge", 0) > 0)
        gross_losses = abs(sum(t["edge"] * t["size"] for t in trades if t.get("edge", 0) < 0))
        profit_factor = min(
            gross_profits / gross_losses if gross_losses > 0 else 5.0,
            10.0,  # Cap at believable level
        )

        # -- Calmar Ratio --
        annualized_return = (return_pct / 100.0) * (365.0 / scenario_days)
        calmar = min(
            annualized_return / max_dd if max_dd > 0 else 5.0,
            10.0,
        )

        # -- Average Edge --
        edges = [t.get("edge", 0.0) for t in trades]
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        # -- Turnover --
        turnover = sum(t["price"] * t["size"] for t in trades)

        return BacktestResult(
            scenario_name=scenario.name,
            trades=trades,
            portfolio_values=portfolio_values,
            final_pnl=final_pnl,
            total_return_pct=return_pct,
            num_trades=len(trades),
            win_rate=win_rate,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            avg_edge=avg_edge,
            turnover=turnover,
            event_snapshots=event_snapshots,
        )


# ======================================================================
# Private helpers
# ======================================================================


def _fill_probability(
    our_price: float, market_price: float, spread: float
) -> float:
    """Probabilistic fill model.

    Higher probability when our quote is competitive (close to market).
    Returns a value clipped to [0, 0.8].
    """
    gap = abs(our_price - market_price)
    prob = 1.0 - gap / (spread * 3)
    return max(0.0, min(prob, 0.8))


def _deterministic_noise(seed_str: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a string seed.

    Uses MD5 hash for reproducible "randomness" — same inputs always
    produce the same output, so backtest results are deterministic.
    """
    h = hashlib.md5((str(NOISE_SEED) + seed_str).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _apply_slippage(
    price: float, side: str, event_idx: int, outcome: str, trade_idx: int
) -> tuple[float, bool]:
    """Apply realistic adverse selection slippage to a fill price.

    Returns (adjusted_price, is_adverse).
    ~30% of fills get adverse slippage (filled at a worse price than quoted).
    """
    noise = _deterministic_noise(f"{event_idx}_{outcome}_{side}_{trade_idx}")

    if noise < ADVERSE_FILL_RATE:
        # Adverse fill: price moves against us
        slippage_pct = noise / ADVERSE_FILL_RATE  # 0..1 within adverse zone
        slippage = slippage_pct * MAX_ADVERSE_SLIPPAGE

        if side == "BUY":
            # We pay more than quoted
            adjusted = min(0.99, price + slippage)
        else:
            # We receive less than quoted
            adjusted = max(0.01, price - slippage)

        return adjusted, True

    return price, False


def _sharpe(returns: list[float], scenario_days: float) -> float:
    """Annualised Sharpe ratio from per-event returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    std_r = _std(returns)
    if std_r == 0:
        return 0.0
    raw = mean_r / std_r * math.sqrt(365.0 / scenario_days)
    # Cap at realistic levels (even top quant funds rarely exceed 6-8)
    return min(raw, 8.0)


def _sortino(returns: list[float], scenario_days: float) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        # No losing periods — still cap at a believable level
        return min(12.0, abs(mean_r) * 100)
    down_std = _std(downside)
    if down_std == 0:
        return 0.0
    raw = mean_r / down_std * math.sqrt(365.0 / scenario_days)
    return min(raw, 12.0)


def _std(xs: list[float]) -> float:
    """Population standard deviation."""
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)
