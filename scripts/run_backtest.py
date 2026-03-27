#!/usr/bin/env python3
"""Run backtest scenarios to evaluate the trading agent's strategy.

Usage:
    python scripts/run_backtest.py
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from config.settings import Settings
from agent.backtest.scenarios import SCENARIOS
from agent.backtest.simulator import BacktestSimulator
from agent.backtest.report_generator import ReportGenerator

console = Console()


def main():
    logging.basicConfig(level=logging.WARNING)

    config = Settings()

    console.print(
        "\n[bold cyan]Polymarket Trading Agent - Backtest Mode[/]"
        f"\nStarting bankroll: ${config.starting_bankroll:,.2f}"
        f"\nKelly fraction: {config.kelly_fraction}"
        f"\nMin edge to trade: {config.min_edge_to_trade}"
        f"\nMin confidence: {config.min_confidence}"
        f"\nScenarios: {len(SCENARIOS)}\n"
    )

    simulator = BacktestSimulator(
        starting_bankroll=config.starting_bankroll,
        signal_alpha=config.signal_alpha,
        decay_half_life=config.decay_half_life_hours,
        kelly_fraction=config.kelly_fraction,
        max_position_pct=config.max_position_pct,
        min_edge=config.min_edge_to_trade,
        min_confidence=config.min_confidence,
        confidence_beta=config.confidence_beta,
    )

    results = simulator.run_all_scenarios(SCENARIOS)

    reporter = ReportGenerator()
    report = reporter.generate_report(results)

    # Save report
    os.makedirs("backtest_results", exist_ok=True)
    reporter.save_report(report, "backtest_results/report.txt")

    console.print("\n[bold green]Backtest complete![/]")


if __name__ == "__main__":
    main()
