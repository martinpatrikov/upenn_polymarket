#!/usr/bin/env python3
"""Run all market-making backtest scenarios.

Usage:
    python scripts/run_mm_backtest.py
"""

import sys
sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import Settings
from agent.backtest.scenarios import get_all_scenarios
from agent.backtest.simulator import MMBacktestSimulator


def main():
    console = Console()
    config = Settings()
    simulator = MMBacktestSimulator(config)
    scenarios = get_all_scenarios()

    console.print(Panel(
        f"[bold]A-S Market Making Backtest[/]\n"
        f"Scenarios: {len(scenarios)} | Bankroll: ${config.starting_bankroll:,.0f}\n"
        f"Params: gamma={config.as_gamma} kappa={config.as_kappa} "
        f"tau={config.elo_tau_base} sigma={config.elo_sigma_daily}",
        title="Backtest Configuration",
        border_style="cyan",
    ))

    results = []
    for scenario in scenarios:
        console.print(f"\n[bold cyan]Running: {scenario.name}[/] — {scenario.description}")
        result = simulator.run_scenario(scenario)
        results.append(result)

        pnl_color = "green" if result.final_pnl >= 0 else "red"
        console.print(
            f"  Trades: {result.num_trades} | "
            f"Win Rate: {result.win_rate:.0%} | "
            f"[{pnl_color}]P&L: ${result.final_pnl:+,.2f} ({result.total_return_pct:+.2f}%)[/{pnl_color}] | "
            f"Max DD: {result.max_drawdown:.1%}"
        )

    # Summary table
    table = Table(title="\nBacktest Results Summary", show_header=True)
    table.add_column("Scenario", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Max DD", justify="right")

    total_pnl = 0
    total_trades = 0
    for r in results:
        pnl_style = "green" if r.final_pnl >= 0 else "red"
        table.add_row(
            r.scenario_name,
            str(r.num_trades),
            f"{r.win_rate:.0%}" if r.num_trades > 0 else "—",
            f"[{pnl_style}]${r.final_pnl:+,.2f}[/{pnl_style}]",
            f"{r.total_return_pct:+.2f}%",
            f"{r.max_drawdown:.1%}" if r.num_trades > 0 else "—",
        )
        total_pnl += r.final_pnl
        total_trades += r.num_trades

    table.add_row(
        "[bold]Aggregate[/]",
        f"[bold]{total_trades}[/]",
        "",
        f"[bold]${total_pnl:+,.2f}[/]",
        f"[bold]{total_pnl / config.starting_bankroll * 100:+.2f}%[/]",
        "",
    )

    console.print(table)


if __name__ == "__main__":
    main()
