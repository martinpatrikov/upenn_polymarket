"""Formatted backtest report generator using rich tables."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import BacktestResult

console = Console()


class ReportGenerator:
    """Produces formatted backtest reports."""

    def generate_report(self, results: list[BacktestResult]) -> str:
        """Generate and print a comprehensive backtest report."""
        lines: list[str] = []

        # Header
        console.print(
            Panel(
                "[bold]Polymarket Trading Agent - Backtest Report[/]",
                border_style="cyan",
            )
        )

        # Per-scenario results table
        table = Table(
            title="Scenario Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Scenario", style="white", min_width=30)
        table.add_column("Trades", justify="center")
        table.add_column("Win Rate", justify="center")
        table.add_column("P&L", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Max DD", justify="right")

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0

        for r in results:
            pnl_color = "green" if r.final_pnl >= 0 else "red"
            table.add_row(
                r.scenario_name,
                str(r.num_trades),
                f"{r.win_rate:.0%}",
                f"[{pnl_color}]${r.final_pnl:+,.2f}[/]",
                f"[{pnl_color}]{r.total_return_pct:+.2f}%[/]",
                f"{r.max_drawdown:.1%}",
            )
            total_pnl += r.final_pnl
            total_trades += r.num_trades
            total_wins += int(r.win_rate * r.num_trades)

            lines.append(
                f"{r.scenario_name}: P&L=${r.final_pnl:+,.2f} "
                f"({r.total_return_pct:+.2f}%) | "
                f"{r.num_trades} trades | Win rate: {r.win_rate:.0%}"
            )

        console.print(table)

        # Aggregate stats
        avg_return = sum(r.total_return_pct for r in results) / len(results) if results else 0
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0

        agg_table = Table(title="Aggregate Statistics", show_header=True)
        agg_table.add_column("Metric", style="cyan")
        agg_table.add_column("Value", justify="right")

        pnl_color = "green" if total_pnl >= 0 else "red"
        agg_table.add_row("Total P&L", f"[{pnl_color}]${total_pnl:+,.2f}[/]")
        agg_table.add_row("Average Return", f"{avg_return:+.2f}%")
        agg_table.add_row("Total Trades", str(total_trades))
        agg_table.add_row("Overall Win Rate", f"{overall_win_rate:.0%}")
        agg_table.add_row(
            "Max Drawdown (worst)",
            f"{max(r.max_drawdown for r in results):.1%}" if results else "N/A",
        )

        console.print(agg_table)

        # Trade detail for each scenario
        for r in results:
            if not r.trades:
                continue

            detail_table = Table(
                title=f"Trade Log: {r.scenario_name}",
                show_header=True,
                header_style="bold",
            )
            detail_table.add_column("Time (h)", justify="center")
            detail_table.add_column("Side", justify="center")
            detail_table.add_column("Size", justify="right")
            detail_table.add_column("Price", justify="right")
            detail_table.add_column("Edge", justify="right")
            detail_table.add_column("Confidence", justify="right")
            detail_table.add_column("Eff. Edge", justify="right")
            detail_table.add_column("Kelly", justify="right")
            detail_table.add_column("Signals", justify="center")

            for t in r.trades:
                side_color = "green" if t["side"] == "BUY" else "red"
                detail_table.add_row(
                    f"{t['hours_offset']:.1f}",
                    f"[{side_color}]{t['side']}[/]",
                    f"${t['size']:.2f}",
                    f"{t['price']:.4f}",
                    f"{t['edge']:+.3f}",
                    f"{t['confidence']:.3f}",
                    f"{t['effective_edge']:+.3f}",
                    f"{t['kelly_scaled']:.3f}",
                    str(t["signals"]),
                )

            console.print(detail_table)

            lines.append(f"\nTrades for {r.scenario_name}:")
            for t in r.trades:
                lines.append(
                    f"  t={t['hours_offset']:.1f}h | {t['side']} ${t['size']:.2f} "
                    f"@ {t['price']:.4f} | edge={t['edge']:+.3f} "
                    f"conf={t['confidence']:.3f}"
                )

        return "\n".join(lines)

    def save_report(self, report: str, path: str):
        """Save report text to a file."""
        with open(path, "w") as f:
            f.write(report)
        console.print(f"\n[green]Report saved to {path}[/]")
