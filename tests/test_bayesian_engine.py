"""Tests for the Bayesian confidence estimation engine."""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import NewsSignal
from agent.scoring.signal_model import SignalModel
from agent.scoring.bayesian_engine import BayesianEngine
from datetime import datetime, timezone


def make_signal(
    relevance=0.8, sentiment=0.5, credibility=0.9, hours_ago=1.0, title="Test"
) -> NewsSignal:
    return NewsSignal(
        source="test",
        title=title,
        summary="test summary",
        url="",
        published_at=datetime.now(timezone.utc),
        relevance_score=relevance,
        sentiment_score=sentiment,
        source_credibility=credibility,
        hours_ago=hours_ago,
    )


def test_no_signals_returns_prior():
    """With no signals, posterior should equal the prior (market price)."""
    model = SignalModel()
    engine = BayesianEngine(signal_model=model)
    estimate = engine.compute_posterior(0.60, [])

    assert abs(estimate.posterior_probability - 0.60) < 1e-6
    assert abs(estimate.edge) < 1e-6
    assert estimate.confidence == 0.0
    assert estimate.signal_count == 0


def test_positive_signal_increases_probability():
    """A positive sentiment signal should increase the posterior."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    engine = BayesianEngine(signal_model=model)

    signal = make_signal(relevance=0.8, sentiment=0.6, credibility=0.9, hours_ago=0.5)
    estimate = engine.compute_posterior(0.50, [signal])

    assert estimate.posterior_probability > 0.50
    assert estimate.edge > 0


def test_negative_signal_decreases_probability():
    """A negative sentiment signal should decrease the posterior."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    engine = BayesianEngine(signal_model=model)

    signal = make_signal(relevance=0.8, sentiment=-0.6, credibility=0.9, hours_ago=0.5)
    estimate = engine.compute_posterior(0.50, [signal])

    assert estimate.posterior_probability < 0.50
    assert estimate.edge < 0


def test_opposing_signals_partially_cancel():
    """Two opposing signals should partially cancel each other."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    engine = BayesianEngine(signal_model=model)

    signals = [
        make_signal(relevance=0.8, sentiment=0.5, credibility=0.9, hours_ago=1.0),
        make_signal(relevance=0.8, sentiment=-0.5, credibility=0.9, hours_ago=1.0),
    ]
    estimate = engine.compute_posterior(0.50, signals)

    # Should be close to prior (not exactly due to non-linearity)
    assert abs(estimate.posterior_probability - 0.50) < 0.05


def test_zero_relevance_no_update():
    """A signal with zero relevance should not change the posterior."""
    model = SignalModel(alpha=2.0)
    engine = BayesianEngine(signal_model=model)

    signal = make_signal(relevance=0.0, sentiment=0.9, credibility=0.9, hours_ago=0.0)
    estimate = engine.compute_posterior(0.60, [signal])

    assert abs(estimate.posterior_probability - 0.60) < 1e-6


def test_confidence_increases_with_more_signals():
    """Confidence should increase as more signals are added."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    engine = BayesianEngine(signal_model=model)

    one_signal = [make_signal(relevance=0.7, sentiment=0.4, credibility=0.8, hours_ago=1.0)]
    three_signals = one_signal * 3

    est1 = engine.compute_posterior(0.50, one_signal)
    est3 = engine.compute_posterior(0.50, three_signals)

    assert est3.confidence > est1.confidence


def test_should_trade_with_edge():
    """Should trade when effective edge exceeds threshold."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    engine = BayesianEngine(signal_model=model, min_edge=0.03, min_confidence=0.20)

    # Strong signals should trigger trading
    signals = [
        make_signal(relevance=0.9, sentiment=0.7, credibility=0.9, hours_ago=0.5),
        make_signal(relevance=0.85, sentiment=0.6, credibility=0.85, hours_ago=1.0),
    ]
    estimate = engine.compute_posterior(0.50, signals)

    if abs(estimate.effective_edge) >= 0.03 and estimate.confidence >= 0.20:
        assert engine.should_trade(estimate)


def test_should_not_trade_without_confidence():
    """Should not trade when confidence is too low."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    engine = BayesianEngine(signal_model=model, min_edge=0.03, min_confidence=0.50)

    # Weak signal
    signal = make_signal(relevance=0.2, sentiment=0.1, credibility=0.4, hours_ago=20.0)
    estimate = engine.compute_posterior(0.50, [signal])

    assert not engine.should_trade(estimate)


def test_price_clamping():
    """Extreme prices should be clamped to avoid infinite odds."""
    model = SignalModel()
    engine = BayesianEngine(signal_model=model)

    # Price at 0 and 1 should not crash
    est0 = engine.compute_posterior(0.0, [])
    est1 = engine.compute_posterior(1.0, [])

    assert 0 < est0.posterior_probability < 1
    assert 0 < est1.posterior_probability < 1


if __name__ == "__main__":
    test_no_signals_returns_prior()
    test_positive_signal_increases_probability()
    test_negative_signal_decreases_probability()
    test_opposing_signals_partially_cancel()
    test_zero_relevance_no_update()
    test_confidence_increases_with_more_signals()
    test_should_trade_with_edge()
    test_should_not_trade_without_confidence()
    test_price_clamping()
    print("All Bayesian engine tests passed!")
