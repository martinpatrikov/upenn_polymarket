#!/usr/bin/env python3
"""Run the A-S market making simulation across all scenarios.

Produces:
  - Rich terminal output: step-by-step replay of agent decisions
  - Matplotlib charts: equity curves, fair value vs market, dashboard
  - Markdown report: full backtest report with embedded charts

Usage:
    python scripts/run_mm_backtest.py [--output-dir OUTPUT_DIR]
"""

import sys
import os
import argparse

sys.path.insert(0, ".")

from rich.console import Console
from rich.panel import Panel

from config.settings import Settings
from agent.backtest.scenarios import get_all_scenarios
from agent.backtest.simulator import MMBacktestSimulator
from agent.backtest.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="A-S Market Making Simulation")
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for charts and markdown report (default: output/)",
    )
    args = parser.parse_args()

    console = Console()
    config = Settings()
    simulator = MMBacktestSimulator(config)
    scenarios = get_all_scenarios()

    console.print(Panel(
        "[bold cyan]Avellaneda-Stoikov Market Making Agent[/]\n"
        "[bold]Simulated Backtest Across AI Model Prediction Markets[/]\n\n"
        f"Scenarios: [bold]{len(scenarios)}[/] | "
        f"Bankroll: [bold]${config.starting_bankroll:,.0f}[/]\n"
        f"Parameters: γ={config.as_gamma}  κ={config.as_kappa}  "
        f"τ_base={config.elo_tau_base}  σ_elo={config.elo_sigma_daily}\n"
        f"Data Sources: Chatbot Arena ELO + AI News RSS + Bayesian Signal Processing",
        title="[bold white]Backtest Configuration[/]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Run all scenarios
    results = []
    for scenario in scenarios:
        console.print(f"\n[bold cyan]{'═' * 70}[/]")
        console.print(f"[bold]  SCENARIO: {scenario.name}[/]")
        console.print(f"  {scenario.description}")
        console.print(f"  Duration: {scenario.duration_hours:.0f}h | "
                       f"Events: {len(scenario.events)} | "
                       f"Outcomes: {', '.join(scenario.outcomes)}")
        console.print(f"[bold cyan]{'═' * 70}[/]")

        result = simulator.run_scenario(scenario)
        results.append(result)

        pnl_color = "green" if result.final_pnl >= 0 else "red"
        console.print(Panel(
            f"[bold]Final P&L:[/] [{pnl_color}]${result.final_pnl:+,.2f} "
            f"({result.total_return_pct:+.2f}%)[/{pnl_color}]\n"
            f"Trades: {result.num_trades} | Win Rate: {result.win_rate:.0%} | "
            f"Sharpe: {result.sharpe_ratio:.2f}\n"
            f"Max DD: {result.max_drawdown:.1%} | "
            f"Profit Factor: {result.profit_factor:.2f} | "
            f"Avg Edge: {result.avg_edge * 100:.1f}¢",
            title=f"[bold]{scenario.name} — Summary[/]",
            border_style=pnl_color,
        ))

    # Generate full report (terminal replay + charts + markdown)
    console.print(f"\n[bold cyan]{'═' * 70}[/]")
    console.print("[bold]  Generating charts and markdown report...[/]")
    console.print(f"[bold cyan]{'═' * 70}[/]")

    reporter = ReportGenerator(output_dir=args.output_dir)
    report_path = reporter.generate_full_report(results, scenarios)

    console.print(f"\n[bold green]✓ Report saved to: {report_path}[/]")
    console.print(f"[bold green]✓ Charts saved to: {args.output_dir}/[/]")

    # Print aggregate summary
    total_pnl = sum(r.final_pnl for r in results)
    total_trades = sum(r.num_trades for r in results)
    avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results) if results else 0
    worst_dd = max(r.max_drawdown for r in results) if results else 0

    console.print(Panel(
        f"[bold]Aggregate P&L:[/] [{'green' if total_pnl >= 0 else 'red'}]"
        f"${total_pnl:+,.2f}[/]\n"
        f"Total Trades: {total_trades} | Avg Sharpe: {avg_sharpe:.2f} | "
        f"Worst Drawdown: {worst_dd:.1%}\n"
        f"All {len(results)} scenarios profitable: "
        f"{'[green]✓ YES[/]' if all(r.final_pnl > 0 for r in results) else '[red]✗ NO[/]'}",
        title="[bold white]Aggregate Performance[/]",
        border_style="green" if total_pnl >= 0 else "red",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    main()
