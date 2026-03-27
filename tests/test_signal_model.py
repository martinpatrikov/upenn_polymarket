"""Tests for the signal model (likelihood ratio computation)."""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from config.settings import NewsSignal
from agent.scoring.signal_model import SignalModel


def make_signal(**kwargs) -> NewsSignal:
    defaults = dict(
        source="test",
        title="Test headline",
        summary="Test summary",
        url="",
        published_at=datetime.now(timezone.utc),
        relevance_score=0.8,
        sentiment_score=0.5,
        source_credibility=0.9,
        hours_ago=1.0,
    )
    defaults.update(kwargs)
    return NewsSignal(**defaults)


def test_decay_at_zero():
    """Decay at t=0 should be 1.0 (full weight)."""
    model = SignalModel(decay_half_life_hours=4.0)
    assert abs(model.compute_decay(0.0) - 1.0) < 1e-6


def test_decay_at_half_life():
    """Decay at t=half_life should be ~0.5."""
    model = SignalModel(decay_half_life_hours=4.0)
    assert abs(model.compute_decay(4.0) - 0.5) < 0.01


def test_decay_at_two_half_lives():
    """Decay at t=2*half_life should be ~0.25."""
    model = SignalModel(decay_half_life_hours=4.0)
    assert abs(model.compute_decay(8.0) - 0.25) < 0.01


def test_zero_relevance_lr_is_one():
    """Zero relevance should give LR=1 (no information)."""
    model = SignalModel(alpha=2.0)
    signal = make_signal(relevance_score=0.0, sentiment_score=0.9)
    result = model.compute_likelihood_ratio(signal)
    assert abs(result.likelihood_ratio - 1.0) < 1e-6


def test_zero_sentiment_lr_is_one():
    """Zero sentiment should give LR=1 (no information)."""
    model = SignalModel(alpha=2.0)
    signal = make_signal(relevance_score=0.9, sentiment_score=0.0)
    result = model.compute_likelihood_ratio(signal)
    assert abs(result.likelihood_ratio - 1.0) < 1e-6


def test_positive_sentiment_lr_greater_than_one():
    """Positive sentiment with high relevance should give LR > 1."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    signal = make_signal(relevance_score=0.8, sentiment_score=0.6, hours_ago=1.0)
    result = model.compute_likelihood_ratio(signal)
    assert result.likelihood_ratio > 1.0


def test_negative_sentiment_lr_less_than_one():
    """Negative sentiment should give LR < 1."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)
    signal = make_signal(relevance_score=0.8, sentiment_score=-0.6, hours_ago=1.0)
    result = model.compute_likelihood_ratio(signal)
    assert result.likelihood_ratio < 1.0


def test_lr_clamping():
    """LR should be clamped between 0.1 and 10.0."""
    model = SignalModel(alpha=100.0)  # Extreme alpha
    signal = make_signal(relevance_score=1.0, sentiment_score=1.0, source_credibility=1.0, hours_ago=0.0)
    result = model.compute_likelihood_ratio(signal)
    assert result.likelihood_ratio <= 10.0

    signal_neg = make_signal(relevance_score=1.0, sentiment_score=-1.0, source_credibility=1.0, hours_ago=0.0)
    result_neg = model.compute_likelihood_ratio(signal_neg)
    assert result_neg.likelihood_ratio >= 0.1


def test_higher_credibility_stronger_signal():
    """Higher source credibility should produce a stronger likelihood ratio."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)

    high_cred = make_signal(source_credibility=0.9, hours_ago=0.0)
    low_cred = make_signal(source_credibility=0.3, hours_ago=0.0)

    lr_high = model.compute_likelihood_ratio(high_cred).likelihood_ratio
    lr_low = model.compute_likelihood_ratio(low_cred).likelihood_ratio

    # Both should be > 1 (positive sentiment), but high cred should be more extreme
    assert lr_high > lr_low


def test_older_news_weaker_signal():
    """Older news should produce weaker signals."""
    model = SignalModel(alpha=2.0, decay_half_life_hours=4.0)

    fresh = make_signal(hours_ago=0.0)
    old = make_signal(hours_ago=12.0)

    strength_fresh = model.compute_signal_strength(fresh)
    strength_old = model.compute_signal_strength(old)

    assert strength_fresh > strength_old


if __name__ == "__main__":
    test_decay_at_zero()
    test_decay_at_half_life()
    test_decay_at_two_half_lives()
    test_zero_relevance_lr_is_one()
    test_zero_sentiment_lr_is_one()
    test_positive_sentiment_lr_greater_than_one()
    test_negative_sentiment_lr_less_than_one()
    test_lr_clamping()
    test_higher_credibility_stronger_signal()
    test_older_news_weaker_signal()
    print("All signal model tests passed!")
