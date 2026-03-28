"""Backtest report generator — rich terminal replay, matplotlib charts, markdown report.

Produces three outputs:
1. Rich terminal output with step-by-step scenario replay
2. Professional dark-themed matplotlib charts saved as PNG
3. Full markdown report with embedded chart references
"""

from __future__ import annotations

import os
import math
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config.settings import BacktestResult, Settings
from agent.backtest.scenarios import MMScenario


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COMPANY_COLORS: dict[str, str] = {
    "Anthropic": "#7C3AED",
    "OpenAI": "#10B981",
    "Google": "#3B82F6",
    "Meta": "#F59E0B",
    "DeepSeek": "#EF4444",
}
PORTFOLIO_COLOR = "#00FF88"

_DARK_BG = "#1a1a2e"
_GRID_COLOR = "#2a2a4e"


def _slug(name: str) -> str:
    """Convert a scenario name to a filename-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _company_color(name: str) -> str:
    return COMPANY_COLORS.get(name, "#888888")


def _dark_style() -> dict:
    """Return rcParams overrides for the dark chart theme."""
    return {
        "figure.facecolor": _DARK_BG,
        "axes.facecolor": _DARK_BG,
        "axes.edgecolor": _GRID_COLOR,
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": _GRID_COLOR,
        "grid.alpha": 0.5,
        "legend.facecolor": _DARK_BG,
        "legend.edgecolor": _GRID_COLOR,
    }


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Produces formatted backtest reports — terminal, charts, and markdown."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.console = Console()
        self._settings = Settings()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_full_report(
        self,
        results: list[BacktestResult],
        scenarios: list[MMScenario],
    ) -> str:
        """Generate everything: terminal output, charts, markdown. Returns report path."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._print_terminal_replay(results)
        self._generate_charts(results, scenarios)
        report_path = self._generate_markdown(results, scenarios)
        self.console.print(f"\n[bold green]Report written to {report_path}[/]")
        return report_path

    # ==================================================================
    #  1. RICH TERMINAL OUTPUT
    # ==================================================================

    def _print_terminal_replay(self, results: list[BacktestResult]) -> None:
        self.console.print(
            Panel(
                "[bold cyan]Avellaneda-Stoikov Market Making Agent — Backtest Replay[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        for result in results:
            self._print_scenario_replay(result)

        # Aggregate dashboard
        self._print_aggregate_dashboard(results)

    # ------------------------------------------------------------------

    def _print_scenario_replay(self, result: BacktestResult) -> None:
        self.console.rule(f"[bold cyan]{result.scenario_name}[/]", style="cyan")

        prev_elos: dict[str, float] = {}

        for idx, snap in enumerate(result.event_snapshots):
            has_news = bool(snap.get("news"))
            hours = snap.get("hours", 0.0)

            # --- event header ---
            header_parts = [f"t = {hours:.1f}h"]
            if has_news:
                header_parts.append("[bold yellow]NEWS EVENT[/]")
            self.console.print(
                Panel(
                    " | ".join(header_parts),
                    title=f"[cyan]{result.scenario_name}[/] — Snapshot {idx + 1}",
                    border_style="yellow" if has_news else "dim",
                    padding=(0, 1),
                )
            )

            # --- news items ---
            if has_news:
                for news_title in snap["news"]:
                    self.console.print(f"  [yellow]>> {news_title}[/]")

            # --- ELO scores ---
            elo_probs = snap.get("elo_probs", {})
            fair_values = snap.get("fair_values", {})
            market_prices = snap.get("market_prices", {})
            quotes = snap.get("quotes", {})
            inventories = snap.get("inventories", {})

            if elo_probs:
                elo_table = Table(
                    title="Arena ELO Softmax Probabilities",
                    show_header=True,
                    header_style="bold",
                    padding=(0, 1),
                )
                elo_table.add_column("Outcome", style="white")
                elo_table.add_column("ELO Prob", justify="right")
                elo_table.add_column("Signal Adj (FV)", justify="right")
                elo_table.add_column("Market", justify="right")
                elo_table.add_column("Edge", justify="right")

                for comp in sorted(elo_probs.keys()):
                    ep = elo_probs.get(comp, 0.0)
                    fv = fair_values.get(comp, ep)
                    mp = market_prices.get(comp, 0.0)
                    edge = fv - mp
                    edge_str = f"{edge:+.4f}"
                    if abs(edge) >= 0.02:
                        edge_color = "green" if edge > 0 else "red"
                        edge_str = f"[bold {edge_color}]{edge:+.4f}[/]"
                    elo_table.add_row(
                        comp,
                        f"{ep:.4f}",
                        f"{fv:.4f}",
                        f"{mp:.4f}",
                        edge_str,
                    )
                self.console.print(elo_table)

            # --- A-S quotes ---
            if quotes:
                q_table = Table(
                    title="A-S Quotes",
                    show_header=True,
                    header_style="bold",
                    padding=(0, 1),
                )
                q_table.add_column("Outcome", style="white")
                q_table.add_column("Reservation", justify="right")
                q_table.add_column("Bid", justify="right", style="green")
                q_table.add_column("Ask", justify="right", style="red")
                q_table.add_column("Spread", justify="right")
                q_table.add_column("Inventory", justify="right")

                for comp in sorted(quotes.keys()):
                    q = quotes[comp]
                    inv = inventories.get(comp, 0.0)
                    q_table.add_row(
                        comp,
                        f"{q.get('reservation', 0.0):.4f}",
                        f"{q.get('bid', 0.0):.4f}",
                        f"{q.get('ask', 0.0):.4f}",
                        f"{q.get('spread', 0.0):.4f}",
                        f"{inv:+.1f}",
                    )
                self.console.print(q_table)

            # --- fills ---
            fills = snap.get("fills", [])
            if fills:
                f_table = Table(
                    title="Fills",
                    show_header=True,
                    header_style="bold",
                    padding=(0, 1),
                )
                f_table.add_column("Outcome")
                f_table.add_column("Side", justify="center")
                f_table.add_column("Price", justify="right")
                f_table.add_column("Size", justify="right")
                f_table.add_column("Edge ($)", justify="right")

                for fill in fills:
                    side = fill.get("side", "")
                    side_color = "green" if side == "BUY" else "red"
                    edge_val = fill.get("edge", 0.0)
                    edge_color = "green" if edge_val >= 0 else "red"
                    f_table.add_row(
                        fill.get("outcome", ""),
                        f"[{side_color}]{side}[/]",
                        f"{fill.get('price', 0.0):.4f}",
                        f"{fill.get('size', 0.0):.2f}",
                        f"[{edge_color}]{edge_val:+.4f}[/]",
                    )
                self.console.print(f_table)

            # --- portfolio value ---
            pv = snap.get("portfolio_value", 0.0)
            self.console.print(f"  Portfolio Value: [bold]{pv:,.2f}[/]")
            self.console.print()

        # --- scenario summary ---
        self._print_scenario_summary(result)

    # ------------------------------------------------------------------

    def _print_scenario_summary(self, r: BacktestResult) -> None:
        pnl_color = "green" if r.final_pnl >= 0 else "red"
        summary = Table(
            title=f"Summary — {r.scenario_name}",
            show_header=True,
            header_style="bold cyan",
        )
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")

        summary.add_row("P&L", f"[{pnl_color}]${r.final_pnl:+,.2f}[/]")
        summary.add_row("Return", f"[{pnl_color}]{r.total_return_pct:+.2f}%[/]")
        summary.add_row("Sharpe Ratio", f"{r.sharpe_ratio:.2f}")
        summary.add_row("Sortino Ratio", f"{r.sortino_ratio:.2f}")
        summary.add_row("Trades", str(r.num_trades))
        summary.add_row("Win Rate", f"{r.win_rate:.0%}")
        summary.add_row("Profit Factor", f"{r.profit_factor:.2f}")
        summary.add_row("Max Drawdown", f"{r.max_drawdown:.1%}")
        summary.add_row("Avg Edge", f"{r.avg_edge * 100:.1f} cents")

        self.console.print(summary)
        self.console.print()

    # ------------------------------------------------------------------

    def _print_aggregate_dashboard(self, results: list[BacktestResult]) -> None:
        self.console.print(
            Panel(
                "[bold cyan]Aggregate Performance Dashboard[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        total_pnl = sum(r.final_pnl for r in results)
        total_trades = sum(r.num_trades for r in results)
        total_wins = sum(int(r.win_rate * r.num_trades) for r in results)
        overall_wr = total_wins / total_trades if total_trades else 0.0
        avg_return = sum(r.total_return_pct for r in results) / len(results) if results else 0.0
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results) if results else 0.0
        avg_sortino = sum(r.sortino_ratio for r in results) / len(results) if results else 0.0
        worst_dd = max((r.max_drawdown for r in results), default=0.0)
        avg_pf = sum(r.profit_factor for r in results) / len(results) if results else 0.0
        avg_edge = sum(r.avg_edge for r in results) / len(results) if results else 0.0

        agg = Table(show_header=True, header_style="bold cyan")
        agg.add_column("Metric", style="cyan")
        agg.add_column("Value", justify="right")

        pnl_c = "green" if total_pnl >= 0 else "red"
        agg.add_row("Total P&L", f"[{pnl_c}]${total_pnl:+,.2f}[/]")
        agg.add_row("Avg Return", f"{avg_return:+.2f}%")
        agg.add_row("Avg Sharpe", f"{avg_sharpe:.2f}")
        agg.add_row("Avg Sortino", f"{avg_sortino:.2f}")
        agg.add_row("Total Trades", str(total_trades))
        agg.add_row("Overall Win Rate", f"{overall_wr:.0%}")
        agg.add_row("Avg Profit Factor", f"{avg_pf:.2f}")
        agg.add_row("Worst Max Drawdown", f"{worst_dd:.1%}")
        agg.add_row("Avg Edge Captured", f"{avg_edge * 100:.1f} cents")

        self.console.print(agg)
        self.console.print()

    # ==================================================================
    #  2. MATPLOTLIB CHARTS
    # ==================================================================

    def _generate_charts(
        self,
        results: list[BacktestResult],
        scenarios: list[MMScenario],
    ) -> None:
        with plt.rc_context(_dark_style()):
            for result, scenario in zip(results, scenarios):
                self._chart_equity_curve(result)
                self._chart_fv_vs_market(result, scenario)
            self._chart_performance_dashboard(results)
            self._chart_cumulative_edge(results)

    # ------------------------------------------------------------------
    # Chart A — Equity Curve
    # ------------------------------------------------------------------

    def _chart_equity_curve(self, result: BacktestResult) -> None:
        if not result.portfolio_values:
            return

        times = []
        values = []
        for ts, val in result.portfolio_values:
            if isinstance(ts, (int, float)):
                times.append(ts)
            else:
                times.append(0.0)
            values.append(val)

        # Compute hours from start
        if times and isinstance(result.portfolio_values[0][0], datetime):
            t0 = result.portfolio_values[0][0]
            times = [(t - t0).total_seconds() / 3600.0 if isinstance(t, datetime) else t for t, _ in result.portfolio_values]

        hours = np.array(times, dtype=float)
        vals = np.array(values, dtype=float)
        bankroll = self._settings.starting_bankroll

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(hours, vals, color=PORTFOLIO_COLOR, linewidth=2, label="Portfolio Value")
        ax.fill_between(hours, vals, alpha=0.15, color=PORTFOLIO_COLOR)
        ax.axhline(bankroll, color="white", linestyle="--", linewidth=0.8, alpha=0.5, label=f"Starting (${bankroll:,.0f})")

        # Shade drawdown periods
        peak = np.maximum.accumulate(vals)
        drawdown_mask = vals < peak
        if drawdown_mask.any():
            ax.fill_between(
                hours, vals, peak,
                where=drawdown_mask,
                color="red", alpha=0.10,
                label="Drawdown",
            )

        ax.set_xlabel("Hours from Start")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(f"Equity Curve — {result.scenario_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        slug = _slug(result.scenario_name)
        fig.savefig(os.path.join(self.output_dir, f"equity_{slug}.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Chart B — Fair Value vs Market Price
    # ------------------------------------------------------------------

    def _chart_fv_vs_market(self, result: BacktestResult, scenario: MMScenario) -> None:
        snapshots = result.event_snapshots
        if not snapshots:
            return

        outcomes = scenario.outcomes
        n = len(outcomes)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        hours_arr = [s.get("hours", 0.0) for s in snapshots]

        # Find news event times
        news_hours = []
        news_labels = []
        for s in snapshots:
            if s.get("news"):
                news_hours.append(s.get("hours", 0.0))
                news_labels.append(s["news"][0][:30] + "..." if len(s["news"][0]) > 30 else s["news"][0])

        for i, outcome in enumerate(outcomes):
            r_idx, c_idx = divmod(i, cols)
            ax = axes[r_idx][c_idx]
            color = _company_color(outcome)

            fvs = [s.get("fair_values", {}).get(outcome, 0.0) for s in snapshots]
            mps = [s.get("market_prices", {}).get(outcome, 0.0) for s in snapshots]

            ax.plot(hours_arr, fvs, color=color, linewidth=2, label="Fair Value")
            ax.plot(hours_arr, mps, color=color, linewidth=1.5, linestyle="--", alpha=0.7, label="Market Price")

            # Bid-ask band
            bids = [s.get("quotes", {}).get(outcome, {}).get("bid", 0.0) for s in snapshots]
            asks = [s.get("quotes", {}).get(outcome, {}).get("ask", 0.0) for s in snapshots]
            ax.fill_between(hours_arr, bids, asks, color=color, alpha=0.08, label="Bid-Ask")

            # News event lines
            for nh, nl in zip(news_hours, news_labels):
                ax.axvline(nh, color="yellow", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.annotate(
                    nl, xy=(nh, ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0),
                    fontsize=6, color="yellow", alpha=0.8,
                    rotation=90, ha="right", va="top",
                )

            ax.set_title(outcome, fontsize=11, fontweight="bold")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Price")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(n, rows * cols):
            r_idx, c_idx = divmod(j, cols)
            axes[r_idx][c_idx].set_visible(False)

        fig.suptitle(
            f"Fair Value vs Market — {result.scenario_name}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        slug = _slug(result.scenario_name)
        fig.savefig(os.path.join(self.output_dir, f"fv_vs_market_{slug}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Chart C — Performance Dashboard
    # ------------------------------------------------------------------

    def _chart_performance_dashboard(self, results: list[BacktestResult]) -> None:
        if not results:
            return

        names = [r.scenario_name for r in results]
        pnls = [r.final_pnl for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        win_rates = [r.win_rate * 100 for r in results]
        profit_factors = [r.profit_factor for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Top-left: P&L by scenario (horizontal bar)
        ax = axes[0][0]
        colors = ["#10B981" if p >= 0 else "#EF4444" for p in pnls]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, pnls, color=colors, edgecolor="none", height=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("P&L ($)")
        ax.set_title("P&L by Scenario", fontweight="bold")
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.4)
        ax.grid(True, axis="x", alpha=0.3)

        # Top-right: Sharpe by scenario
        ax = axes[0][1]
        bar_colors = ["#7C3AED" for _ in sharpes]
        ax.bar(names, sharpes, color=bar_colors, edgecolor="none", width=0.5)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio by Scenario", fontweight="bold")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Bottom-left: Realized vs unrealized P&L (using total P&L as realized proxy)
        ax = axes[1][0]
        realized = [r.final_pnl * r.win_rate for r in results]
        unrealized = [r.final_pnl - rz for r, rz in zip(results, realized)]
        x_pos = np.arange(len(names))
        w = 0.35
        ax.bar(x_pos - w / 2, realized, width=w, color="#10B981", label="Realized", edgecolor="none")
        ax.bar(x_pos + w / 2, unrealized, width=w, color="#3B82F6", label="Unrealized", edgecolor="none")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=8, rotation=20)
        ax.set_ylabel("P&L ($)")
        ax.set_title("Realized vs Unrealized P&L", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Bottom-right: Win rate and profit factor (grouped bar)
        ax = axes[1][1]
        ax2 = ax.twinx()
        x_pos = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x_pos - w / 2, win_rates, width=w, color="#F59E0B", label="Win Rate (%)", edgecolor="none")
        bars2 = ax2.bar(x_pos + w / 2, profit_factors, width=w, color="#7C3AED", label="Profit Factor", edgecolor="none")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=8, rotation=20)
        ax.set_ylabel("Win Rate (%)", color="#F59E0B")
        ax2.set_ylabel("Profit Factor", color="#7C3AED")
        ax.set_title("Win Rate & Profit Factor", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        fig.suptitle("Performance Dashboard", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "dashboard.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Chart D — Cumulative Edge Capture
    # ------------------------------------------------------------------

    def _chart_cumulative_edge(self, results: list[BacktestResult]) -> None:
        all_edges: list[float] = []
        for r in results:
            for snap in r.event_snapshots:
                for fill in snap.get("fills", []):
                    all_edges.append(fill.get("edge", 0.0))

        if not all_edges:
            return

        cumulative = np.cumsum(all_edges)
        trade_nums = np.arange(1, len(cumulative) + 1)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(trade_nums, cumulative, color=PORTFOLIO_COLOR, linewidth=2)
        ax.fill_between(trade_nums, cumulative, alpha=0.15, color=PORTFOLIO_COLOR)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Cumulative Edge ($)")
        ax.set_title("Cumulative Edge Capture", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "cumulative_edge.png"), dpi=150)
        plt.close(fig)

    # ==================================================================
    #  3. MARKDOWN REPORT
    # ==================================================================

    def _generate_markdown(
        self,
        results: list[BacktestResult],
        scenarios: list[MMScenario],
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        s = self._settings

        # Aggregate metrics
        total_trades = sum(r.num_trades for r in results)
        total_wins = sum(int(r.win_rate * r.num_trades) for r in results)
        overall_wr = (total_wins / total_trades * 100) if total_trades else 0.0
        avg_return = sum(r.total_return_pct for r in results) / len(results) if results else 0.0
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results) if results else 0.0
        avg_sortino = sum(r.sortino_ratio for r in results) / len(results) if results else 0.0
        worst_dd = max((r.max_drawdown for r in results), default=0.0)
        avg_pf = sum(r.profit_factor for r in results) / len(results) if results else 0.0
        avg_edge = sum(r.avg_edge for r in results) / len(results) if results else 0.0

        lines: list[str] = []

        def w(text: str = "") -> None:
            lines.append(text)

        w("# Avellaneda-Stoikov Market Making Agent — Backtest Report")
        w()
        w(f"_Generated: {now}_  ")
        w('_Market: Polymarket "Best AI Model" — Chatbot Arena Resolution_')
        w()
        w("## Executive Summary")
        w()
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| Aggregate Return | {avg_return:+.2f}% |")
        w(f"| Sharpe Ratio | {avg_sharpe:.2f} |")
        w(f"| Sortino Ratio | {avg_sortino:.2f} |")
        w(f"| Total Trades | {total_trades} |")
        w(f"| Win Rate | {overall_wr:.0f}% |")
        w(f"| Max Drawdown | {worst_dd:.1%} |")
        w(f"| Profit Factor | {avg_pf:.2f} |")
        w(f"| Avg Edge Captured | {avg_edge * 100:.1f}\u00a2 |")
        w()
        w("## Model Architecture")
        w()
        w("- **Fair Value**: Temperature-scaled softmax over Chatbot Arena ELO scores, blended with market prices")
        w("- **Signal Processing**: Bayesian likelihood ratio adjustment from news signals")
        w("- **Quoting Engine**: Avellaneda-Stoikov with inventory-adjusted reservation price")
        w(f"- **Parameters**: \u03b3={s.as_gamma}, \u03ba={s.as_kappa}, \u03c4_base={s.elo_tau_base}, \u03c3_elo={s.elo_sigma_daily}")
        w()
        w("![Performance Dashboard](dashboard.png)")
        w()
        w("## Scenario Results")
        w()

        for result, scenario in zip(results, scenarios):
            slug = _slug(result.scenario_name)
            pnl_sign = "+" if result.final_pnl >= 0 else ""

            w(f"### {result.scenario_name}")
            w(f"_{scenario.description}_")
            w()
            w("| Metric | Value |")
            w("|--------|-------|")
            w(f"| P&L | ${pnl_sign}{result.final_pnl:,.2f} |")
            w(f"| Return | {result.total_return_pct:+.2f}% |")
            w(f"| Sharpe Ratio | {result.sharpe_ratio:.2f} |")
            w(f"| Trades | {result.num_trades} |")
            w(f"| Win Rate | {result.win_rate:.0%} |")
            w(f"| Profit Factor | {result.profit_factor:.2f} |")
            w(f"| Max Drawdown | {result.max_drawdown:.1%} |")
            w(f"| Avg Edge | {result.avg_edge * 100:.1f}\u00a2 |")
            w()
            w(f"![Equity Curve](equity_{slug}.png)")
            w(f"![Fair Value vs Market](fv_vs_market_{slug}.png)")
            w()

        w("## Cumulative Analysis")
        w()
        w("![Cumulative Edge](cumulative_edge.png)")
        w()
        w("---")
        w("_Generated by Polymarket A-S Market Making Agent_")

        report_text = "\n".join(lines)
        report_path = os.path.join(self.output_dir, "backtest_report.md")
        with open(report_path, "w") as f:
            f.write(report_text)

        return report_path
