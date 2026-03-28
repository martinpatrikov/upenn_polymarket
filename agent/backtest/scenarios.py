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
        _gpt5_hype_cycle(),
        _gemini_arena_takeover(),
        _quiet_spread_capture(),
        _deepseek_pump_and_dump(),
    ]


# ---------------------------------------------------------------------------
# Scenario A: GPT-5 Hype Cycle
# ---------------------------------------------------------------------------

def _gpt5_hype_cycle() -> MMScenario:
    """OpenAI launches GPT-5 with fanfare but Arena ELO barely moves.

    Market overreacts, pricing OpenAI up to 52c.  Our ELO model never gives
    OpenAI more than ~0.30.  Agent sells the hype and profits as the market
    corrects back toward ELO-implied fair values over the following week.
    """
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]

    events = [
        # t=0h  Baseline: Anthropic leads Arena, market reflects it
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1505, "Google": 1470, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.35, "OpenAI": 0.30, "Google": 0.22, "Meta": 0.13,
            },
        ),
        # t=6h  GPT-5 announced — market hype begins before any Arena data
        MMScenarioEvent(
            hours_offset=6,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1505, "Google": 1470, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.22, "OpenAI": 0.48, "Google": 0.19, "Meta": 0.11,
            },
            news_items=[
                ScenarioNewsItem(
                    title="OpenAI unveils GPT-5 at live event",
                    summary="Sam Altman demos GPT-5 on stage claiming 'the biggest leap since GPT-4'. Early access rolling out to Plus subscribers today.",
                    source="the_verge",
                    hours_offset=6,
                ),
                ScenarioNewsItem(
                    title="GPT-5 is here and Twitter is losing its mind",
                    summary="Viral clips show GPT-5 solving PhD-level physics in real time. Prediction markets spike OpenAI to nearly 50 cents.",
                    source="twitter",
                    hours_offset=6,
                ),
            ],
        ),
        # t=12h  ELO barely ticks up — market still euphoric
        MMScenarioEvent(
            hours_offset=12,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1512, "Google": 1470, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.20, "OpenAI": 0.52, "Google": 0.17, "Meta": 0.11,
            },
        ),
        # t=24h  Market peaks — ELO signal hasn't moved, first doubts surface
        MMScenarioEvent(
            hours_offset=24,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1512, "Google": 1470, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.24, "OpenAI": 0.46, "Google": 0.18, "Meta": 0.12,
            },
        ),
        # t=48h  Independent benchmarks confirm marginal gain — correction starts
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1512, "Google": 1472, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.28, "OpenAI": 0.38, "Google": 0.21, "Meta": 0.13,
            },
            news_items=[
                ScenarioNewsItem(
                    title="Independent evals show GPT-5 improvement is marginal",
                    summary="SEAL and HELM benchmarks place GPT-5 only 1-2% above GPT-4o on reasoning tasks. Arena ELO moved just 7 points.",
                    source="ars_technica",
                    hours_offset=48,
                ),
            ],
        ),
        # t=72h  Market continues correcting
        MMScenarioEvent(
            hours_offset=72,
            elo_scores={
                "Anthropic": 1511, "OpenAI": 1512, "Google": 1472, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.30, "OpenAI": 0.34, "Google": 0.22, "Meta": 0.14,
            },
        ),
        # t=96h  Approaching ELO-implied equilibrium
        MMScenarioEvent(
            hours_offset=96,
            elo_scores={
                "Anthropic": 1512, "OpenAI": 1511, "Google": 1472, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.33, "OpenAI": 0.30, "Google": 0.23, "Meta": 0.14,
            },
        ),
        # t=120h  Market almost fully corrected
        MMScenarioEvent(
            hours_offset=120,
            elo_scores={
                "Anthropic": 1513, "OpenAI": 1510, "Google": 1473, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.37, "OpenAI": 0.29, "Google": 0.21, "Meta": 0.13,
            },
        ),
        # t=144h  Near terminal state
        MMScenarioEvent(
            hours_offset=144,
            elo_scores={
                "Anthropic": 1514, "OpenAI": 1510, "Google": 1473, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.38, "OpenAI": 0.29, "Google": 0.20, "Meta": 0.13,
            },
        ),
        # t=168h  Resolution — Anthropic holds ELO lead
        MMScenarioEvent(
            hours_offset=168,
            elo_scores={
                "Anthropic": 1515, "OpenAI": 1510, "Google": 1473, "Meta": 1410,
            },
            market_prices={
                "Anthropic": 0.38, "OpenAI": 0.29, "Google": 0.20, "Meta": 0.13,
            },
        ),
    ]

    return MMScenario(
        name="GPT-5 Hype Cycle",
        description=(
            "OpenAI launches GPT-5 to massive fanfare but Arena ELO only moves "
            "7 points. Market overreacts to 52c then corrects. Tests agent's "
            "ability to sell into hype when ELO signal disagrees with price."
        ),
        outcomes=outcomes,
        duration_hours=168,
        events=events,
        final_winner="Anthropic",
    )


# ---------------------------------------------------------------------------
# Scenario B: Gemini 2.5 Arena Takeover
# ---------------------------------------------------------------------------

def _gemini_arena_takeover() -> MMScenario:
    """Google releases Gemini 2.5 which genuinely tops Arena.

    ELO jumps 1470 -> 1520 -> 1530.  Agent detects early via ELO model and
    buys Google at 30c before the market catches up.
    """
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]

    events = [
        # t=0h  Baseline: Anthropic leads, Google mid-pack
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1500, "Google": 1470, "Meta": 1415,
            },
            market_prices={
                "Anthropic": 0.36, "OpenAI": 0.30, "Google": 0.22, "Meta": 0.12,
            },
        ),
        # t=12h  Gemini 2.5 launches — ELO jumps immediately but market lags
        MMScenarioEvent(
            hours_offset=12,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1500, "Google": 1520, "Meta": 1415,
            },
            market_prices={
                "Anthropic": 0.34, "OpenAI": 0.28, "Google": 0.25, "Meta": 0.13,
            },
            news_items=[
                ScenarioNewsItem(
                    title="Google launches Gemini 2.5 with native multimodal reasoning",
                    summary="Gemini 2.5 debuts at #1 on Chatbot Arena with 1520 ELO, 10 points above Claude. Google claims breakthrough in chain-of-thought architecture.",
                    source="techcrunch",
                    hours_offset=12,
                ),
            ],
        ),
        # t=24h  Market starts catching up but still lags ELO signal
        MMScenarioEvent(
            hours_offset=24,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1500, "Google": 1522, "Meta": 1415,
            },
            market_prices={
                "Anthropic": 0.30, "OpenAI": 0.26, "Google": 0.30, "Meta": 0.14,
            },
        ),
        # t=48h  ELO continues climbing, market lags by ~5c
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={
                "Anthropic": 1509, "OpenAI": 1498, "Google": 1525, "Meta": 1415,
            },
            market_prices={
                "Anthropic": 0.26, "OpenAI": 0.24, "Google": 0.38, "Meta": 0.12,
            },
        ),
        # t=72h  Arena users confirm Gemini quality holds
        MMScenarioEvent(
            hours_offset=72,
            elo_scores={
                "Anthropic": 1508, "OpenAI": 1497, "Google": 1528, "Meta": 1414,
            },
            market_prices={
                "Anthropic": 0.22, "OpenAI": 0.22, "Google": 0.43, "Meta": 0.13,
            },
            news_items=[
                ScenarioNewsItem(
                    title="Gemini 2.5 holds Arena lead after 50k votes",
                    summary="After three days and 50,000 blind Arena votes, Gemini 2.5 maintains a comfortable ELO margin over Claude and GPT-5.",
                    source="ars_technica",
                    hours_offset=72,
                ),
            ],
        ),
        # t=120h  ELO at 1530, market approaching but still lagging
        MMScenarioEvent(
            hours_offset=120,
            elo_scores={
                "Anthropic": 1507, "OpenAI": 1496, "Google": 1530, "Meta": 1414,
            },
            market_prices={
                "Anthropic": 0.19, "OpenAI": 0.20, "Google": 0.48, "Meta": 0.13,
            },
        ),
        # t=168h  Market converging on ELO-implied value
        MMScenarioEvent(
            hours_offset=168,
            elo_scores={
                "Anthropic": 1506, "OpenAI": 1495, "Google": 1530, "Meta": 1414,
            },
            market_prices={
                "Anthropic": 0.17, "OpenAI": 0.18, "Google": 0.52, "Meta": 0.13,
            },
            news_items=[
                ScenarioNewsItem(
                    title="Gemini 2.5 dominates enterprise AI benchmarks",
                    summary="Google Cloud reports record adoption as Gemini 2.5 leads on Arena, MMLU-Pro, and LiveBench simultaneously.",
                    source="the_verge",
                    hours_offset=168,
                ),
            ],
        ),
        # t=192h  Near-terminal convergence
        MMScenarioEvent(
            hours_offset=192,
            elo_scores={
                "Anthropic": 1506, "OpenAI": 1495, "Google": 1531, "Meta": 1414,
            },
            market_prices={
                "Anthropic": 0.16, "OpenAI": 0.17, "Google": 0.55, "Meta": 0.12,
            },
        ),
        # t=216h  Almost at resolution
        MMScenarioEvent(
            hours_offset=216,
            elo_scores={
                "Anthropic": 1505, "OpenAI": 1494, "Google": 1531, "Meta": 1414,
            },
            market_prices={
                "Anthropic": 0.15, "OpenAI": 0.16, "Google": 0.56, "Meta": 0.13,
            },
        ),
        # t=240h  Resolution — Google wins decisively
        MMScenarioEvent(
            hours_offset=240,
            elo_scores={
                "Anthropic": 1505, "OpenAI": 1494, "Google": 1532, "Meta": 1414,
            },
            market_prices={
                "Anthropic": 0.14, "OpenAI": 0.15, "Google": 0.58, "Meta": 0.13,
            },
        ),
    ]

    return MMScenario(
        name="Gemini 2.5 Arena Takeover",
        description=(
            "Google releases Gemini 2.5 which genuinely tops Arena (ELO 1470 -> "
            "1530). Market lags ELO signal by 3-8c throughout. Tests agent's "
            "ability to buy early when ELO signal leads price."
        ),
        outcomes=outcomes,
        duration_hours=240,
        events=events,
        final_winner="Google",
    )


# ---------------------------------------------------------------------------
# Scenario C: Quiet Market Spread Capture
# ---------------------------------------------------------------------------

def _quiet_spread_capture() -> MMScenario:
    """No major launches. Stable ELOs with small fluctuations.

    Market prices stay relatively flat. Agent profits purely from
    bid-ask spread capture in a low-volatility regime.
    """
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta"]

    events = [
        # t=0h  Baseline
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={
                "Anthropic": 1512, "OpenAI": 1502, "Google": 1475, "Meta": 1418,
            },
            market_prices={
                "Anthropic": 0.35, "OpenAI": 0.29, "Google": 0.23, "Meta": 0.13,
            },
        ),
        # t=48h  Minor ELO fluctuation, no real news
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={
                "Anthropic": 1514, "OpenAI": 1500, "Google": 1476, "Meta": 1417,
            },
            market_prices={
                "Anthropic": 0.36, "OpenAI": 0.28, "Google": 0.23, "Meta": 0.13,
            },
            news_items=[
                ScenarioNewsItem(
                    title="Arena leaderboard stable as AI labs focus on inference costs",
                    summary="No major model launches this week. Labs appear focused on efficiency and cost reduction rather than capability gains.",
                    source="techcrunch",
                    hours_offset=48,
                ),
            ],
        ),
        # t=96h  Tiny ELO noise, prices inch
        MMScenarioEvent(
            hours_offset=96,
            elo_scores={
                "Anthropic": 1511, "OpenAI": 1503, "Google": 1474, "Meta": 1419,
            },
            market_prices={
                "Anthropic": 0.35, "OpenAI": 0.29, "Google": 0.22, "Meta": 0.14,
            },
        ),
        # t=144h  Still quiet, minimal drift
        MMScenarioEvent(
            hours_offset=144,
            elo_scores={
                "Anthropic": 1513, "OpenAI": 1501, "Google": 1477, "Meta": 1416,
            },
            market_prices={
                "Anthropic": 0.36, "OpenAI": 0.28, "Google": 0.23, "Meta": 0.13,
            },
        ),
        # t=192h  Routine Arena batch processed
        MMScenarioEvent(
            hours_offset=192,
            elo_scores={
                "Anthropic": 1514, "OpenAI": 1502, "Google": 1475, "Meta": 1418,
            },
            market_prices={
                "Anthropic": 0.35, "OpenAI": 0.29, "Google": 0.23, "Meta": 0.13,
            },
            news_items=[
                ScenarioNewsItem(
                    title="AI benchmark fatigue: researchers call for new evaluation paradigms",
                    summary="Academic paper argues current Arena methodology has plateaued in discriminative power. No model changes expected near-term.",
                    source="ars_technica",
                    hours_offset=192,
                ),
            ],
        ),
        # t=240h  Still stable
        MMScenarioEvent(
            hours_offset=240,
            elo_scores={
                "Anthropic": 1512, "OpenAI": 1503, "Google": 1476, "Meta": 1417,
            },
            market_prices={
                "Anthropic": 0.36, "OpenAI": 0.28, "Google": 0.23, "Meta": 0.13,
            },
        ),
        # t=288h  Approaching resolution, no changes
        MMScenarioEvent(
            hours_offset=288,
            elo_scores={
                "Anthropic": 1513, "OpenAI": 1502, "Google": 1476, "Meta": 1418,
            },
            market_prices={
                "Anthropic": 0.36, "OpenAI": 0.29, "Google": 0.22, "Meta": 0.13,
            },
        ),
        # t=336h  Resolution — Anthropic wins in a quiet market
        MMScenarioEvent(
            hours_offset=336,
            elo_scores={
                "Anthropic": 1513, "OpenAI": 1502, "Google": 1476, "Meta": 1418,
            },
            market_prices={
                "Anthropic": 0.36, "OpenAI": 0.29, "Google": 0.22, "Meta": 0.13,
            },
        ),
    ]

    return MMScenario(
        name="Quiet Market Spread Capture",
        description=(
            "No major model launches for two weeks. ELOs fluctuate +/- 3 points. "
            "Market stays flat. Tests pure bid-ask spread capture profitability "
            "in a low-volatility regime."
        ),
        outcomes=outcomes,
        duration_hours=336,
        events=events,
        final_winner="Anthropic",
    )


# ---------------------------------------------------------------------------
# Scenario D: DeepSeek Pump & Dump
# ---------------------------------------------------------------------------

def _deepseek_pump_and_dump() -> MMScenario:
    """DeepSeek-R2 launches and briefly tops Arena via formatting tricks.

    ELO spikes 1440 -> 1505 -> 1512.  Market panics, pricing DeepSeek to 42c.
    Then Arena applies style control and ELO crashes back to 1470.  Agent sells
    the spike and profits enormously on the collapse.
    """
    outcomes = ["Anthropic", "OpenAI", "Google", "Meta", "DeepSeek"]

    events = [
        # t=0h  Baseline: DeepSeek is a minor player
        MMScenarioEvent(
            hours_offset=0,
            elo_scores={
                "Anthropic": 1512, "OpenAI": 1502, "Google": 1475, "Meta": 1418,
                "DeepSeek": 1440,
            },
            market_prices={
                "Anthropic": 0.34, "OpenAI": 0.28, "Google": 0.20, "Meta": 0.08,
                "DeepSeek": 0.10,
            },
        ),
        # t=6h  DeepSeek-R2 drops, initial Arena ELO spike
        MMScenarioEvent(
            hours_offset=6,
            elo_scores={
                "Anthropic": 1512, "OpenAI": 1502, "Google": 1475, "Meta": 1418,
                "DeepSeek": 1505,
            },
            market_prices={
                "Anthropic": 0.30, "OpenAI": 0.25, "Google": 0.18, "Meta": 0.07,
                "DeepSeek": 0.16,
            },
            news_items=[
                ScenarioNewsItem(
                    title="DeepSeek-R2 drops and immediately surges on Arena",
                    summary="DeepSeek releases R2 model which rockets up Arena leaderboard to top-3. Users praise verbose, detailed responses.",
                    source="twitter",
                    hours_offset=6,
                ),
            ],
        ),
        # t=18h  ELO continues rising, market follows with lag
        MMScenarioEvent(
            hours_offset=18,
            elo_scores={
                "Anthropic": 1511, "OpenAI": 1501, "Google": 1474, "Meta": 1417,
                "DeepSeek": 1510,
            },
            market_prices={
                "Anthropic": 0.26, "OpenAI": 0.22, "Google": 0.16, "Meta": 0.06,
                "DeepSeek": 0.30,
            },
        ),
        # t=36h  Peak hype — DeepSeek tops Arena, market mania
        MMScenarioEvent(
            hours_offset=36,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1500, "Google": 1473, "Meta": 1416,
                "DeepSeek": 1512,
            },
            market_prices={
                "Anthropic": 0.22, "OpenAI": 0.20, "Google": 0.14, "Meta": 0.07,
                "DeepSeek": 0.37,
            },
            news_items=[
                ScenarioNewsItem(
                    title="DeepSeek-R2 takes #1 on Chatbot Arena",
                    summary="Chinese AI lab's latest model now holds the top ELO score on Arena. Prediction markets surge as traders pile into DeepSeek contracts.",
                    source="techcrunch",
                    hours_offset=36,
                ),
            ],
        ),
        # t=48h  Market peaks — DeepSeek at 42c
        MMScenarioEvent(
            hours_offset=48,
            elo_scores={
                "Anthropic": 1510, "OpenAI": 1500, "Google": 1473, "Meta": 1416,
                "DeepSeek": 1512,
            },
            market_prices={
                "Anthropic": 0.20, "OpenAI": 0.18, "Google": 0.13, "Meta": 0.07,
                "DeepSeek": 0.42,
            },
        ),
        # t=72h  Arena applies style control — ELO crashes
        MMScenarioEvent(
            hours_offset=72,
            elo_scores={
                "Anthropic": 1512, "OpenAI": 1502, "Google": 1475, "Meta": 1417,
                "DeepSeek": 1485,
            },
            market_prices={
                "Anthropic": 0.24, "OpenAI": 0.21, "Google": 0.15, "Meta": 0.07,
                "DeepSeek": 0.37,
            },
            news_items=[
                ScenarioNewsItem(
                    title="Arena enables style control: DeepSeek-R2 ELO collapses 27 points",
                    summary="LMSYS applies style-controlled Arena rankings. DeepSeek-R2 drops from 1512 to 1485 as verbose formatting advantage is neutralized.",
                    source="ars_technica",
                    hours_offset=72,
                ),
            ],
        ),
        # t=96h  Market correcting hard
        MMScenarioEvent(
            hours_offset=96,
            elo_scores={
                "Anthropic": 1513, "OpenAI": 1502, "Google": 1476, "Meta": 1417,
                "DeepSeek": 1475,
            },
            market_prices={
                "Anthropic": 0.28, "OpenAI": 0.24, "Google": 0.18, "Meta": 0.07,
                "DeepSeek": 0.27,
            },
        ),
        # t=120h  Continued collapse back to pre-launch levels
        MMScenarioEvent(
            hours_offset=120,
            elo_scores={
                "Anthropic": 1513, "OpenAI": 1503, "Google": 1476, "Meta": 1418,
                "DeepSeek": 1472,
            },
            market_prices={
                "Anthropic": 0.31, "OpenAI": 0.26, "Google": 0.19, "Meta": 0.07,
                "DeepSeek": 0.17,
            },
        ),
        # t=168h  Almost fully deflated
        MMScenarioEvent(
            hours_offset=168,
            elo_scores={
                "Anthropic": 1514, "OpenAI": 1503, "Google": 1476, "Meta": 1418,
                "DeepSeek": 1470,
            },
            market_prices={
                "Anthropic": 0.33, "OpenAI": 0.27, "Google": 0.19, "Meta": 0.08,
                "DeepSeek": 0.13,
            },
        ),
        # t=192h  Near baseline
        MMScenarioEvent(
            hours_offset=192,
            elo_scores={
                "Anthropic": 1514, "OpenAI": 1503, "Google": 1476, "Meta": 1418,
                "DeepSeek": 1468,
            },
            market_prices={
                "Anthropic": 0.34, "OpenAI": 0.27, "Google": 0.20, "Meta": 0.08,
                "DeepSeek": 0.10,
            },
        ),
        # t=216h  DeepSeek fully deflated
        MMScenarioEvent(
            hours_offset=216,
            elo_scores={
                "Anthropic": 1514, "OpenAI": 1502, "Google": 1476, "Meta": 1418,
                "DeepSeek": 1467,
            },
            market_prices={
                "Anthropic": 0.35, "OpenAI": 0.28, "Google": 0.20, "Meta": 0.08,
                "DeepSeek": 0.09,
            },
        ),
        # t=240h  Resolution — Anthropic wins, DeepSeek a footnote
        MMScenarioEvent(
            hours_offset=240,
            elo_scores={
                "Anthropic": 1515, "OpenAI": 1502, "Google": 1476, "Meta": 1418,
                "DeepSeek": 1466,
            },
            market_prices={
                "Anthropic": 0.35, "OpenAI": 0.28, "Google": 0.20, "Meta": 0.08,
                "DeepSeek": 0.09,
            },
        ),
    ]

    return MMScenario(
        name="DeepSeek Pump & Dump",
        description=(
            "DeepSeek-R2 launches and briefly tops Arena via formatting tricks "
            "(ELO 1440 -> 1512). Market panics to 42c. Arena applies style control, "
            "ELO crashes to 1470. Tests agent's ability to sell into artificial "
            "spikes and profit from the collapse."
        ),
        outcomes=outcomes,
        duration_hours=240,
        events=events,
        final_winner="Anthropic",
    )
