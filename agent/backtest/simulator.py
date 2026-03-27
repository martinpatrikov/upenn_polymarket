"""Market-making backtest simulator.

Replays MMScenarios through the A-S engine, simulating fills and
tracking market-making P&L.
"""

from __future__ import annotations

import logging
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


class MMBacktestSimulator:
    """Simulates A-S market making through historical scenarios."""

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
        self.as_engine = AvellanedaStoikov(
            gamma=self.config.as_gamma,
            kappa=self.config.as_kappa,
            base_order_size=self.config.as_base_order_size,
            max_inventory=self.config.as_max_inventory,
        )

    def run_scenario(self, scenario: MMScenario) -> BacktestResult:
        """Run a single scenario through the A-S engine."""
        inventory = InventoryManager(
            max_inventory_per_outcome=self.config.as_max_inventory,
        )
        trades: list[dict] = []
        portfolio_values: list[tuple[datetime, float]] = []
        start_time = datetime.now(timezone.utc)
        total_expiry_hours = scenario.duration_hours

        for i, event in enumerate(scenario.events):
            timestamp = start_time + timedelta(hours=event.hours_offset)

            # Days remaining
            days_remaining = max(0.01, (total_expiry_hours - event.hours_offset) / 24.0)

            # Compute fair values from ELO
            base_probs = self.elo_model.compute_probabilities(
                event.elo_scores, days_remaining
            )

            # Apply news signals
            signals = []
            for item in event.news_items:
                signals.append(NewsSignal(
                    source=item.source,
                    title=item.title,
                    summary=item.summary,
                    url="",
                    published_at=start_time + timedelta(hours=item.hours_offset),
                    relevance_score=0.8,  # Pre-scored for backtest
                    sentiment_score=0.6,
                    source_credibility=0.7,
                ))

            fair_values = self.signal_adjuster.adjust_probabilities(base_probs, signals)

            # Estimate volatilities
            tau_eff = self.elo_model._effective_temperature(days_remaining)
            sigmas = self.volatility.estimate(fair_values, tau_eff)

            # Generate quotes
            token_ids = {o: f"sim_{o}" for o in scenario.outcomes}
            inventories = inventory.get_all_inventories()
            quotes = self.as_engine.generate_quotes(
                fair_values=fair_values,
                inventories=inventories,
                sigmas=sigmas,
                time_to_expiry=days_remaining,
                token_ids=token_ids,
            )

            # Simulate fills using probabilistic model:
            # Our quotes get filled proportionally to how competitive they are.
            # If our bid is close to market mid, higher chance of fill.
            for quote in quotes:
                mkt_price = event.market_prices.get(quote.outcome)
                if mkt_price is None:
                    continue

                # Probability of bid fill: higher when our bid is competitive
                # (close to or above market mid-price)
                bid_edge = mkt_price - quote.bid_price  # positive = we're cheaper
                if bid_edge < quote.spread * 2 and quote.bid_size > 0:
                    # Fill at our bid price if we're within 2x spread of market
                    fill_prob = max(0, 1.0 - bid_edge / (quote.spread * 2)) * 0.5
                    if fill_prob > 0.1 and not inventory.should_reduce(quote.outcome, Side.BUY):
                        fill_size = quote.bid_size * fill_prob
                        inventory.update(quote.outcome, Side.BUY, fill_size, quote.bid_price)
                        trades.append({
                            "time": timestamp.isoformat(),
                            "outcome": quote.outcome,
                            "side": "BUY",
                            "price": quote.bid_price,
                            "size": fill_size,
                            "fair_value": quote.fair_value,
                            "market_price": mkt_price,
                        })

                # Probability of ask fill
                ask_edge = quote.ask_price - mkt_price  # positive = we're more expensive
                if ask_edge < quote.spread * 2 and quote.ask_size > 0:
                    fill_prob = max(0, 1.0 - ask_edge / (quote.spread * 2)) * 0.5
                    if fill_prob > 0.1 and not inventory.should_reduce(quote.outcome, Side.SELL):
                        fill_size = quote.ask_size * fill_prob
                        inventory.update(quote.outcome, Side.SELL, fill_size, quote.ask_price)
                        trades.append({
                            "time": timestamp.isoformat(),
                            "outcome": quote.outcome,
                            "side": "SELL",
                            "price": quote.ask_price,
                            "size": fill_size,
                            "fair_value": quote.fair_value,
                            "market_price": mkt_price,
                        })

            # Track portfolio value
            inv_state = inventory.get_state(event.market_prices)
            pv = self.config.starting_bankroll + inv_state.realized_pnl + inv_state.unrealized_pnl
            portfolio_values.append((timestamp, pv))

        # Compute final metrics
        final_state = inventory.get_state(scenario.events[-1].market_prices if scenario.events else {})
        final_pnl = final_state.realized_pnl + final_state.unrealized_pnl
        return_pct = (final_pnl / self.config.starting_bankroll) * 100

        # Win rate (trades that captured spread profitably)
        wins = 0
        for t in trades:
            if t["side"] == "BUY" and t["price"] < t["fair_value"]:
                wins += 1
            elif t["side"] == "SELL" and t["price"] > t["fair_value"]:
                wins += 1
        win_rate = wins / len(trades) if trades else 0.0

        # Max drawdown
        max_dd = 0.0
        peak = self.config.starting_bankroll
        for _, pv in portfolio_values:
            peak = max(peak, pv)
            dd = (peak - pv) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return BacktestResult(
            scenario_name=scenario.name,
            trades=trades,
            portfolio_values=portfolio_values,
            final_pnl=final_pnl,
            total_return_pct=return_pct,
            num_trades=len(trades),
            win_rate=win_rate,
            max_drawdown=max_dd,
        )
