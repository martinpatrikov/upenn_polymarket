"""Main market-making agent orchestrator.

The MarketMakingAgent runs a continuous loop:
1. Fetch AI model events from Polymarket
2. Subscribe to outcome tokens via Rust sidecar
3. Fetch Arena leaderboard ELO scores
4. Fetch Twitter/news signals
5. Compute fair values (ELO + signals)
6. Estimate volatilities
7. Get real-time orderbooks from sidecar
8. Generate Avellaneda-Stoikov quotes for all outcomes
9. Risk check
10. Execute (paper trade: log quotes + simulate fills)
11. Update inventory, print dashboard
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import (
    Settings,
    MultiOutcomeMarket,
    ASQuote,
    PortfolioState,
    Side,
)
from agent.data_ingestion.arena_scraper import ArenaScraper
from agent.data_ingestion.twitter_fetcher import TwitterFetcher
from agent.data_ingestion.sidecar_client import SidecarClient
from agent.data_ingestion.market_data import MarketDataClient
from agent.data_ingestion.nlp_processor import NLPProcessor
from agent.fair_value.elo_model import EloFairValueModel
from agent.fair_value.signal_adjuster import SignalAdjuster
from agent.fair_value.volatility_estimator import VolatilityEstimator
from agent.market_making.avellaneda_stoikov import AvellanedaStoikov
from agent.market_making.inventory_manager import InventoryManager
from agent.market_making.quote_manager import QuoteManager
from agent.scoring.signal_model import SignalModel

logger = logging.getLogger(__name__)
console = Console()


class MarketMakingAgent:
    """Avellaneda-Stoikov market making agent for AI model prediction markets."""

    def __init__(self, config: Settings):
        self.config = config

        # Data ingestion
        self.arena = ArenaScraper(
            company_prefixes=config.company_model_prefixes,
        )
        self.twitter = TwitterFetcher(
            nitter_instances=config.nitter_instances,
            accounts=config.ai_twitter_accounts,
            ai_rss_feeds=config.ai_news_rss_feeds,
            max_age_hours=config.news_max_age_hours,
        )
        self.sidecar = SidecarClient(host=config.sidecar_host)
        self.market_client = MarketDataClient(
            clob_host=config.clob_host,
            gamma_host=config.gamma_host,
        )
        self.nlp = NLPProcessor()

        # Fair value
        self.elo_model = EloFairValueModel(
            tau_base=config.elo_tau_base,
            sigma_elo_daily=config.elo_sigma_daily,
        )
        self.signal_adjuster = SignalAdjuster(
            alpha=config.signal_alpha,
            decay_half_life_hours=config.decay_half_life_hours,
            max_shift=config.signal_max_shift,
        )
        self.volatility = VolatilityEstimator(
            sigma_elo_daily=config.elo_sigma_daily,
            min_sigma=config.min_sigma,
            max_sigma=config.max_sigma,
        )

        # Market making
        self.as_engine = AvellanedaStoikov(
            gamma=config.as_gamma,
            kappa=config.as_kappa,
            base_order_size=config.as_base_order_size,
            max_inventory=config.as_max_inventory,
        )
        self.inventory_mgr = InventoryManager(
            max_inventory_per_outcome=config.as_max_inventory,
        )
        self.quote_mgr = QuoteManager(
            inventory_manager=self.inventory_mgr,
            refresh_interval_seconds=config.quote_refresh_seconds,
        )

        # State
        self.portfolio = PortfolioState(cash=config.starting_bankroll)
        self.cycle_count = 0
        self._current_event: Optional[MultiOutcomeMarket] = None

    def run_cycle(self) -> list[ASQuote]:
        """Run one full market-making cycle."""
        self.cycle_count += 1
        console.print(
            f"\n[bold cyan]{'='*70}[/]"
            f"\n[bold cyan]  A-S Market Making | Cycle #{self.cycle_count} | "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/]"
            f"\n[bold cyan]{'='*70}[/]"
        )

        # Step 1: Find AI model markets
        events = self._find_ai_model_events()
        if not events:
            console.print("[yellow]No AI model markets found[/]")
            return []

        event = events[0]  # Focus on the most relevant event
        self._current_event = event
        console.print(f"[green]Target market:[/] {event.question}")
        console.print(f"  Outcomes: {', '.join(o['name'] for o in event.outcomes)}")

        # Step 2: Subscribe to orderbooks via sidecar
        token_ids = [o["token_id"] for o in event.outcomes if o.get("token_id")]
        sidecar_ok = self.sidecar.health_check()
        if sidecar_ok and token_ids:
            self.sidecar.subscribe(token_ids)

        # Step 3: Fetch Arena ELO scores
        arena_models = self.arena.fetch_leaderboard()
        target_companies = [o["name"] for o in event.outcomes]
        company_elos = self.elo_model.extract_company_elos(arena_models, target_companies)

        if not company_elos:
            console.print("[yellow]No Arena ELO data available — using market prices as fallback[/]")
            company_elos = {o["name"]: 1200.0 for o in event.outcomes}

        console.print("\n[bold]Arena ELO Scores:[/]")
        for company, elo in sorted(company_elos.items(), key=lambda x: -x[1]):
            console.print(f"  {company}: {elo:.0f}")

        # Step 4: Compute days to expiry
        days_to_expiry = self._days_to_expiry(event)
        console.print(f"\n  Days to expiry: {days_to_expiry:.1f}")

        # Step 5: Compute base fair values from ELO
        base_probs = self.elo_model.compute_probabilities(company_elos, days_to_expiry)

        # Step 6: Fetch and process signals
        raw_signals = self.twitter.fetch_all_signals()
        processed_signals = []
        if raw_signals:
            news_dicts = [
                {
                    "title": s.title,
                    "summary": s.summary,
                    "published_at": s.published_at,
                    "source_name": s.source,
                    "url": s.url,
                }
                for s in raw_signals
            ]
            processed_signals = self.nlp.process_news_for_market(
                news_dicts, event.question, min_relevance=self.config.min_relevance_threshold
            )

        # Step 7: Adjust fair values with signals
        adjusted_probs = self.signal_adjuster.adjust_probabilities(base_probs, processed_signals)

        console.print(f"\n[bold]Fair Values[/] ({len(processed_signals)} signals processed):")
        for company in sorted(adjusted_probs, key=lambda x: -adjusted_probs[x]):
            base = base_probs.get(company, 0)
            adj = adjusted_probs[company]
            shift = adj - base
            shift_str = f" ({shift:+.3f})" if abs(shift) > 0.001 else ""
            console.print(f"  {company}: {adj:.4f}{shift_str}")

        # Step 8: Estimate volatilities
        tau_eff = self.elo_model._effective_temperature(days_to_expiry)
        sigmas = self.volatility.estimate(adjusted_probs, tau_eff)

        # Step 9: Get orderbooks — try sidecar, fall back to CLOB REST, then Gamma prices
        token_id_map = {o["name"]: o["token_id"] for o in event.outcomes if o.get("token_id")}
        gamma_prices = {o["name"]: o.get("price", 0.0) for o in event.outcomes}
        orderbook_summaries = {}
        for outcome, tid in token_id_map.items():
            summary = None
            # Try sidecar first
            if sidecar_ok:
                summary = self.sidecar.get_summary(tid)
            # Fall back to CLOB REST API
            if not summary:
                book = self.market_client.get_orderbook(tid)
                if book and (book.bids or book.asks):
                    summary = {
                        "best_bid": book.best_bid,
                        "best_ask": book.best_ask,
                        "midpoint": book.midpoint,
                    }
            # Final fallback: use Gamma API prices (YES token price = midpoint proxy)
            if not summary and gamma_prices.get(outcome):
                price = gamma_prices[outcome]
                summary = {
                    "best_bid": max(0.01, price - 0.01),
                    "best_ask": min(0.99, price + 0.01),
                    "midpoint": price,
                }
            if summary:
                orderbook_summaries[outcome] = summary

        # Step 10: Generate A-S quotes
        inventories = self.inventory_mgr.get_all_inventories()
        quotes = self.as_engine.generate_quotes(
            fair_values=adjusted_probs,
            inventories=inventories,
            sigmas=sigmas,
            time_to_expiry=days_to_expiry,
            token_ids=token_id_map,
        )

        # Step 11: Update quote manager and simulate fills
        self.quote_mgr.update_quotes(quotes)
        fills = self.quote_mgr.simulate_fills(orderbook_summaries)
        if fills:
            for fill in fills:
                console.print(
                    f"  [bold green]FILL:[/] {fill['side']} {fill['outcome']} "
                    f"@ {fill['price']:.4f} x {fill['size']:.1f}"
                )

        # Step 12: Print dashboard
        self._print_dashboard(quotes, orderbook_summaries, adjusted_probs)

        return quotes

    def start_live(self, interval_seconds: Optional[float] = None):
        """Start the live market-making loop."""
        interval = interval_seconds or self.config.polling_interval_seconds
        console.print(
            Panel(
                "[bold green]Polymarket A-S Market Maker[/]\n"
                f"Mode: PAPER TRADING | Interval: {interval}s\n"
                f"Starting bankroll: ${self.config.starting_bankroll:,.2f}\n"
                f"Parameters: gamma={self.config.as_gamma} kappa={self.config.as_kappa}",
                title="Agent Started",
                border_style="green",
            )
        )

        try:
            while True:
                self.run_cycle()
                console.print(
                    f"\n[dim]Next cycle in {interval}s... (Ctrl+C to stop)[/]"
                )
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Agent stopped by user[/]")
            self._print_final_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Known event slugs for AI model markets
    AI_MODEL_EVENT_SLUGS = [
        "which-company-has-the-best-ai-model-end-of-april",
        "which-company-has-best-ai-model-end-of-june",
        "which-company-has-the-best-ai-model-end-of-may",
        "which-company-has-the-best-ai-model-end-of-march-751",
    ]

    def _find_ai_model_events(self) -> list[MultiOutcomeMarket]:
        """Search Polymarket for AI model prediction markets.

        These events have multiple binary YES/NO markets (one per company).
        We aggregate them into a single MultiOutcomeMarket view.
        """
        events = []
        import json
        import httpx

        for slug in self.AI_MODEL_EVENT_SLUGS:
            try:
                resp = httpx.get(
                    f"{self.config.gamma_host}/events",
                    params={"slug": slug},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    continue

                event_data = data[0]
                title = event_data.get("title", "")
                markets = event_data.get("markets", [])

                # Each market is a binary YES/NO for one company
                outcomes = []
                total_volume = 0.0
                end_date = None

                for m in markets:
                    if not m.get("active", False) or m.get("closed", True):
                        continue

                    question = m.get("question", "")
                    # Extract company name: "Will X have the best AI model..."
                    company = self._extract_company_from_question(question)
                    if not company:
                        continue

                    try:
                        clob_ids = json.loads(m.get("clobTokenIds", "[]"))
                        prices = json.loads(m.get("outcomePrices", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # YES token is first
                    if clob_ids and prices:
                        yes_price = float(prices[0]) if prices[0] else 0.0
                        if yes_price < 0.001:
                            continue  # Skip near-zero markets
                        outcomes.append({
                            "name": company,
                            "token_id": clob_ids[0],
                            "price": yes_price,
                            "condition_id": m.get("conditionId", ""),
                        })

                    total_volume += float(m.get("volume", 0) or 0)

                    if not end_date and m.get("endDate"):
                        try:
                            end_date = datetime.fromisoformat(
                                m["endDate"].replace("Z", "+00:00")
                            )
                        except (ValueError, AttributeError):
                            pass

                if len(outcomes) >= 2:
                    events.append(MultiOutcomeMarket(
                        event_id=slug,
                        question=title,
                        outcomes=sorted(outcomes, key=lambda o: -o["price"]),
                        end_date=end_date,
                        volume=total_volume,
                    ))

            except Exception as e:
                logger.warning(f"Failed to fetch event {slug}: {e}")
                continue

        # Sort: prefer events with end dates >7 days out (enough time to trade),
        # then by volume. Skip nearly-expired markets.
        now = datetime.now(timezone.utc)
        min_days_remaining = 5.0

        def sort_key(e):
            if e.end_date:
                days_left = (e.end_date - now).total_seconds() / 86400.0
                has_time = days_left > min_days_remaining
            else:
                has_time = True
                days_left = 30.0
            return (has_time, e.volume)

        events.sort(key=sort_key, reverse=True)
        return events

    @staticmethod
    def _extract_company_from_question(question: str) -> str | None:
        """Extract company name from 'Will X have the best AI model...' format."""
        import re
        # Match "Will <Company> have the best/2nd/..."
        match = re.match(r"Will\s+(.+?)\s+have\s+the\s+", question, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            # Skip generic "Company A", "Company B" placeholders
            if re.match(r"^Company\s+[A-Z]$", company):
                return None
            return company
        return None

    def _days_to_expiry(self, event: MultiOutcomeMarket) -> float:
        """Calculate days to market expiration."""
        if event.end_date:
            now = datetime.now(timezone.utc)
            if event.end_date.tzinfo is None:
                end = event.end_date.replace(tzinfo=timezone.utc)
            else:
                end = event.end_date
            delta = (end - now).total_seconds() / 86400.0
            return max(0.01, delta)
        return 30.0  # Default: 30 days

    def _print_dashboard(
        self,
        quotes: list[ASQuote],
        orderbooks: dict[str, dict],
        fair_values: dict[str, float],
    ):
        """Print the market-making dashboard."""
        table = Table(title="A-S Market Making Dashboard", show_header=True)
        table.add_column("Outcome", style="cyan")
        table.add_column("Fair Value", justify="right")
        table.add_column("Mkt Mid", justify="right")
        table.add_column("Reserv.", justify="right")
        table.add_column("Bid", justify="right", style="green")
        table.add_column("Ask", justify="right", style="red")
        table.add_column("Spread", justify="right")
        table.add_column("Inv", justify="right")

        for q in sorted(quotes, key=lambda x: -x.fair_value):
            book = orderbooks.get(q.outcome, {})
            mkt_mid = book.get("midpoint")
            mkt_mid_str = f"{mkt_mid:.4f}" if mkt_mid else "N/A"

            table.add_row(
                q.outcome,
                f"{q.fair_value:.4f}",
                mkt_mid_str,
                f"{q.reservation_price:.4f}",
                f"{q.bid_price:.4f}",
                f"{q.ask_price:.4f}",
                f"{q.spread:.4f}",
                f"{q.inventory:.1f}",
            )

        console.print(table)

        # Inventory summary
        inv_state = self.inventory_mgr.get_state(fair_values)
        console.print(
            f"  Realized P&L: ${inv_state.realized_pnl:+.2f} | "
            f"Unrealized P&L: ${inv_state.unrealized_pnl:+.2f} | "
            f"Fills: {len(self.quote_mgr.get_fill_history())}"
        )

    def _print_final_summary(self):
        """Print summary when agent stops."""
        fills = self.quote_mgr.get_fill_history()
        fair_values = {}
        if self._current_event:
            for o in self._current_event.outcomes:
                fair_values[o["name"]] = o.get("price", 0.5)

        inv_state = self.inventory_mgr.get_state(fair_values)
        total_pnl = inv_state.realized_pnl + inv_state.unrealized_pnl

        console.print(
            Panel(
                f"[bold]Total Fills:[/] {len(fills)}\n"
                f"[bold]Realized P&L:[/] ${inv_state.realized_pnl:+,.2f}\n"
                f"[bold]Unrealized P&L:[/] ${inv_state.unrealized_pnl:+,.2f}\n"
                f"[bold]Total P&L:[/] ${total_pnl:+,.2f}\n"
                f"[bold]Cycles Run:[/] {self.cycle_count}",
                title="Session Summary",
                border_style="cyan",
            )
        )
