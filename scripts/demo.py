#!/usr/bin/env python3
"""Quick demo: fetch live markets, analyze news signals, and score confidence.

This script demonstrates the full pipeline in a single pass:
1. Fetch top markets from Polymarket
2. For each, fetch relevant news
3. Process NLP signals
4. Run Bayesian scoring
5. Print formatted decision summary

Usage:
    python scripts/demo.py
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import Settings, Side
from agent.data_ingestion.news_fetcher import NewsFetcher
from agent.data_ingestion.market_data import MarketDataClient
from agent.data_ingestion.nlp_processor import NLPProcessor
from agent.scoring.signal_model import SignalModel
from agent.scoring.bayesian_engine import BayesianEngine
from agent.scoring.kelly_criterion import KellyCriterion
from agent.execution.orderbook_analyzer import OrderBookAnalyzer

console = Console()


def main():
    logging.basicConfig(level=logging.WARNING)

    config = Settings()

    console.print(
        Panel(
            "[bold green]Polymarket Trading Agent - Demo[/]\n"
            "Fetching live markets, analyzing news, scoring confidence...",
            border_style="green",
        )
    )

    # Initialize components
    market_client = MarketDataClient(
        clob_host=config.clob_host,
        gamma_host=config.gamma_host,
    )
    news_fetcher = NewsFetcher(feeds=config.rss_feeds)
    nlp = NLPProcessor()
    signal_model = SignalModel(
        alpha=config.signal_alpha,
        decay_half_life_hours=config.decay_half_life_hours,
    )
    bayesian = BayesianEngine(
        signal_model=signal_model,
        min_edge=config.min_edge_to_trade,
        min_confidence=config.min_confidence,
        confidence_beta=config.confidence_beta,
    )
    kelly = KellyCriterion(
        kelly_fraction=config.kelly_fraction,
        max_position_pct=config.max_position_pct,
    )
    book_analyzer = OrderBookAnalyzer()

    # Step 1: Fetch top markets
    console.print("\n[bold]Step 1:[/] Fetching active markets from Polymarket...")
    markets = market_client.get_active_markets(limit=5)

    if not markets:
        console.print("[red]No markets found. Check your internet connection.[/]")
        return

    console.print(f"[green]Found {len(markets)} markets[/]\n")

    # Summary table
    summary_table = Table(title="Analysis Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Market", min_width=40)
    summary_table.add_column("Price", justify="center")
    summary_table.add_column("Signals", justify="center")
    summary_table.add_column("Posterior", justify="center")
    summary_table.add_column("Edge", justify="center")
    summary_table.add_column("Confidence", justify="center")
    summary_table.add_column("Action", justify="center")

    for market in markets:
        if not market.tokens:
            continue

        yes_token = market.tokens[0]
        token_id = yes_token["token_id"]
        market_price = float(yes_token.get("price", 0.5))

        console.print(f"[bold]Analyzing:[/] {market.question[:80]}")

        # Step 2: Get orderbook
        book = market_client.get_orderbook(token_id)
        if book:
            analysis = book_analyzer.analyze(book)
            market_price = book.midpoint if book.midpoint > 0 else market_price
            console.print(
                f"  Orderbook: bid={analysis.best_bid:.3f} ask={analysis.best_ask:.3f} "
                f"spread={analysis.spread:.3f} depth=${analysis.bid_depth_total + analysis.ask_depth_total:.0f}"
            )
        else:
            console.print("  [yellow]No orderbook data[/]")

        # Step 3: Fetch news
        news_items = news_fetcher.fetch_news_for_market(
            market.question, max_age_hours=24
        )
        console.print(f"  News articles found: {len(news_items)}")

        # Step 4: Process NLP
        news_dicts = [
            {
                "title": item.title,
                "summary": item.summary,
                "published_at": item.published_at,
                "source_name": item.source_name,
                "url": item.url,
            }
            for item in news_items
        ]
        signals = nlp.process_news_for_market(
            news_dicts, market.question, min_relevance=config.min_relevance_threshold
        )
        console.print(f"  Relevant signals: {len(signals)}")

        # Step 5: Bayesian scoring
        estimate = bayesian.compute_posterior(market_price, signals)

        for i, reason in enumerate(estimate.reasoning[:3]):
            console.print(f"    Signal {i+1}: {reason}")

        console.print(
            f"  Posterior: {estimate.posterior_probability:.4f} | "
            f"Edge: {estimate.edge:+.4f} | "
            f"Confidence: {estimate.confidence:.3f} | "
            f"Eff. Edge: {estimate.effective_edge:+.4f}"
        )

        # Step 6: Trade decision
        should_trade = bayesian.should_trade(estimate)
        if should_trade:
            side = Side.BUY if estimate.edge > 0 else Side.SELL
            position = kelly.compute_position_size(
                portfolio_value=10000.0,
                estimated_prob=estimate.posterior_probability,
                market_price=market_price,
                side=side,
            )
            action = (
                f"[green]{side.value} ${position.dollar_amount:.0f}[/]"
                if side == Side.BUY
                else f"[red]{side.value} ${position.dollar_amount:.0f}[/]"
            )
            console.print(
                f"  [bold]Decision:[/] {side.value} | Kelly: {position.kelly_scaled:.3f} | "
                f"Size: ${position.dollar_amount:.2f}"
            )
        else:
            action = "[yellow]HOLD[/]"
            console.print("  [bold]Decision:[/] HOLD (insufficient edge or confidence)")

        summary_table.add_row(
            market.question[:45] + "..." if len(market.question) > 45 else market.question,
            f"{market_price:.3f}",
            str(len(signals)),
            f"{estimate.posterior_probability:.3f}",
            f"{estimate.edge:+.3f}",
            f"{estimate.confidence:.3f}",
            action,
        )
        console.print()

    # Print summary
    console.print(summary_table)
    console.print("\n[bold green]Demo complete![/]")


if __name__ == "__main__":
    main()
