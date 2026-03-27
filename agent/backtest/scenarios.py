"""Backtest scenarios for AI model market making.

Each scenario defines a timeline of events with ELO scores, market prices,
and news signals. The backtest simulator replays these through the A-S engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


@dataclass
class ScenarioNewsItem:
    title: str
    summary: str
    source: str
    hours_offset: float  # Hours from scenario start


@dataclass
class MMScenarioEvent:
    """A point in time for market-making backtest."""
    hours_offset: float
    elo_scores: dict[str, float]        # company -> ELO
    market_prices: dict[str, float]     # outcome -> market mid-price
    news_items: list[ScenarioNewsItem] = field(default_factory=list)


@dataclass
class MMScenario:
    """Market-making backtest scenario for AI model markets."""
    name: str
    description: str
    outcomes: list[str]                 # ["Anthropic", "OpenAI", "Google", ...]
    duration_hours: float
    events: list[MMScenarioEvent]
    final_winner: str                   # Which company wins at resolution


def get_all_scenarios() -> list[MMScenario]:
    """Return all backtest scenarios."""
    return [
        _stable_leader(),
        _close_race(),
        _new_model_shock(),
        _gradual_shift(),
    ]


def _stable_leader() -> MMScenario:
    """Scenario 1: Anthropic leads throughout with small ELO fluctuations."""
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]
    events = [
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.40, "OpenAI": 0.30, "Google": 0.20, "Meta": 0.10},
        ),
        MMScenarioEvent(
            hours_offset=24,
            elo_scores={"Anthropic": 1506, "OpenAI": 1478, "Google": 1462, "Meta": 1398},
            market_prices={"Anthropic": 0.42, "OpenAI": 0.28, "Google": 0.20, "Meta": 0.10},
        ),
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={"Anthropic": 1502, "OpenAI": 1482, "Google": 1458, "Meta": 1402},
            market_prices={"Anthropic": 0.41, "OpenAI": 0.29, "Google": 0.19, "Meta": 0.11},
            news_items=[
                ScenarioNewsItem(
                    title="Claude continues to lead Arena leaderboard",
                    summary="Anthropic's Claude maintains top position with steady ELO score.",
                    source="twitter",
                    hours_offset=48,
                ),
            ],
        ),
        MMScenarioEvent(
            hours_offset=72,
            elo_scores={"Anthropic": 1508, "OpenAI": 1476, "Google": 1464, "Meta": 1396},
            market_prices={"Anthropic": 0.43, "OpenAI": 0.27, "Google": 0.21, "Meta": 0.09},
        ),
        MMScenarioEvent(
            hours_offset=120,
            elo_scores={"Anthropic": 1510, "OpenAI": 1478, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.45, "OpenAI": 0.28, "Google": 0.19, "Meta": 0.08},
        ),
    ]
    return MMScenario(
        name="Stable Leader",
        description="Anthropic leads throughout. Small ELO fluctuations. Tests steady spread capture.",
        outcomes=outcomes,
        duration_hours=120,
        events=events,
        final_winner="Anthropic",
    )


def _close_race() -> MMScenario:
    """Scenario 2: OpenAI and Anthropic neck-and-neck."""
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]
    events = [
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={"Anthropic": 1500, "OpenAI": 1498, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.35, "OpenAI": 0.33, "Google": 0.20, "Meta": 0.12},
        ),
        MMScenarioEvent(
            hours_offset=24,
            elo_scores={"Anthropic": 1497, "OpenAI": 1502, "Google": 1462, "Meta": 1398},
            market_prices={"Anthropic": 0.32, "OpenAI": 0.36, "Google": 0.21, "Meta": 0.11},
            news_items=[
                ScenarioNewsItem(
                    title="GPT-5 turbo tops Arena with narrow margin",
                    summary="OpenAI's latest model edges past Claude in Arena rankings.",
                    source="twitter",
                    hours_offset=24,
                ),
            ],
        ),
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={"Anthropic": 1503, "OpenAI": 1499, "Google": 1458, "Meta": 1402},
            market_prices={"Anthropic": 0.34, "OpenAI": 0.34, "Google": 0.20, "Meta": 0.12},
        ),
        MMScenarioEvent(
            hours_offset=72,
            elo_scores={"Anthropic": 1505, "OpenAI": 1496, "Google": 1464, "Meta": 1396},
            market_prices={"Anthropic": 0.37, "OpenAI": 0.31, "Google": 0.22, "Meta": 0.10},
            news_items=[
                ScenarioNewsItem(
                    title="Claude regains top spot after Arena update",
                    summary="New batch of Arena votes pushes Claude back to #1.",
                    source="the_verge",
                    hours_offset=72,
                ),
            ],
        ),
        MMScenarioEvent(
            hours_offset=120,
            elo_scores={"Anthropic": 1504, "OpenAI": 1498, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.36, "OpenAI": 0.33, "Google": 0.20, "Meta": 0.11},
        ),
    ]
    return MMScenario(
        name="Close Race",
        description="OpenAI and Anthropic within 5 ELO. Higher volatility, tests inventory management.",
        outcomes=outcomes,
        duration_hours=120,
        events=events,
        final_winner="Anthropic",
    )


def _new_model_shock() -> MMScenario:
    """Scenario 3: Google releases a breakthrough model mid-period."""
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]
    events = [
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.40, "OpenAI": 0.30, "Google": 0.20, "Meta": 0.10},
        ),
        MMScenarioEvent(
            hours_offset=24,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.40, "OpenAI": 0.30, "Google": 0.20, "Meta": 0.10},
        ),
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1520, "Meta": 1400},
            market_prices={"Anthropic": 0.25, "OpenAI": 0.20, "Google": 0.45, "Meta": 0.10},
            news_items=[
                ScenarioNewsItem(
                    title="Google releases Gemini Ultra 2 - tops all benchmarks",
                    summary="Google DeepMind releases Gemini Ultra 2 which immediately takes #1 on Arena.",
                    source="twitter",
                    hours_offset=48,
                ),
                ScenarioNewsItem(
                    title="Gemini Ultra 2 achieves state-of-the-art on Arena leaderboard",
                    summary="Google's new model achieves 1520 ELO, surpassing Claude.",
                    source="techcrunch",
                    hours_offset=48,
                ),
            ],
        ),
        MMScenarioEvent(
            hours_offset=72,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1525, "Meta": 1400},
            market_prices={"Anthropic": 0.22, "OpenAI": 0.18, "Google": 0.50, "Meta": 0.10},
        ),
        MMScenarioEvent(
            hours_offset=120,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1530, "Meta": 1400},
            market_prices={"Anthropic": 0.18, "OpenAI": 0.15, "Google": 0.58, "Meta": 0.09},
        ),
    ]
    return MMScenario(
        name="New Model Shock",
        description="Google releases breakthrough model. Tests signal reaction + inventory recovery.",
        outcomes=outcomes,
        duration_hours=120,
        events=events,
        final_winner="Google",
    )


def _gradual_shift() -> MMScenario:
    """Scenario 4: Meta slowly climbs rankings over 2 weeks."""
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]
    events = [
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1400},
            market_prices={"Anthropic": 0.40, "OpenAI": 0.30, "Google": 0.20, "Meta": 0.10},
        ),
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1430},
            market_prices={"Anthropic": 0.38, "OpenAI": 0.28, "Google": 0.19, "Meta": 0.15},
            news_items=[
                ScenarioNewsItem(
                    title="Llama 4 shows strong Arena performance",
                    summary="Meta's Llama 4 climbing Arena rankings steadily.",
                    source="twitter",
                    hours_offset=48,
                ),
            ],
        ),
        MMScenarioEvent(
            hours_offset=96,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1460},
            market_prices={"Anthropic": 0.35, "OpenAI": 0.26, "Google": 0.19, "Meta": 0.20},
        ),
        MMScenarioEvent(
            hours_offset=168,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1490},
            market_prices={"Anthropic": 0.32, "OpenAI": 0.24, "Google": 0.17, "Meta": 0.27},
            news_items=[
                ScenarioNewsItem(
                    title="Llama 4 now competitive with top models",
                    summary="Meta's Llama 4 approaches top 3 in Arena with rapid ELO gains.",
                    source="the_verge",
                    hours_offset=168,
                ),
            ],
        ),
        MMScenarioEvent(
            hours_offset=336,
            elo_scores={"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1500},
            market_prices={"Anthropic": 0.30, "OpenAI": 0.22, "Google": 0.16, "Meta": 0.32},
        ),
    ]
    return MMScenario(
        name="Gradual Shift",
        description="Meta slowly climbs rankings. Tests A-S adaptation to trending fair value.",
        outcomes=outcomes,
        duration_hours=336,
        events=events,
        final_winner="Anthropic",  # Still Anthropic barely wins at close
    )
