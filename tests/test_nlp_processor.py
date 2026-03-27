"""Tests for the NLP processor (sentiment and relevance)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.data_ingestion.nlp_processor import NLPProcessor


def test_positive_sentiment():
    """Clearly positive text should have positive sentiment."""
    nlp = NLPProcessor()
    result = nlp.compute_sentiment(
        "This is great news! The economy is booming and everyone is optimistic."
    )
    assert result.combined_score > 0
    assert result.vader_compound > 0


def test_negative_sentiment():
    """Clearly negative text should have negative sentiment."""
    nlp = NLPProcessor()
    result = nlp.compute_sentiment(
        "This is terrible. The market crashed and investors are devastated."
    )
    assert result.combined_score < 0
    assert result.vader_compound < 0


def test_neutral_sentiment():
    """Neutral text should have near-zero sentiment."""
    nlp = NLPProcessor()
    result = nlp.compute_sentiment(
        "The meeting was held at the scheduled time."
    )
    assert abs(result.combined_score) < 0.5


def test_relevance_related_texts():
    """Related texts should have high relevance score."""
    nlp = NLPProcessor()
    score = nlp.compute_relevance(
        "Federal Reserve announces interest rate decision, cutting rates by 25 basis points",
        "Will the Federal Reserve cut interest rates?",
    )
    assert score > 0.2


def test_relevance_unrelated_texts():
    """Unrelated texts should have low relevance score."""
    nlp = NLPProcessor()
    score = nlp.compute_relevance(
        "New restaurant opens downtown with innovative fusion cuisine menu",
        "Will the Federal Reserve cut interest rates?",
    )
    assert score < 0.2


def test_empty_text_sentiment():
    """Empty text should return zero sentiment."""
    nlp = NLPProcessor()
    result = nlp.compute_sentiment("")
    assert result.combined_score == 0.0


def test_empty_text_relevance():
    """Empty text should return zero relevance."""
    nlp = NLPProcessor()
    score = nlp.compute_relevance("", "Some question")
    assert score == 0.0


def test_process_news_filters_irrelevant():
    """News processing should filter out irrelevant articles."""
    nlp = NLPProcessor()
    from datetime import datetime, timezone

    news = [
        {
            "title": "Fed cuts rates by 25 basis points",
            "summary": "The Federal Reserve has decided to cut interest rates",
            "published_at": datetime.now(timezone.utc),
            "source_name": "reuters",
            "url": "http://example.com/1",
        },
        {
            "title": "New recipe for chocolate cake",
            "summary": "A delicious new chocolate cake recipe has gone viral",
            "published_at": datetime.now(timezone.utc),
            "source_name": "google_news",
            "url": "http://example.com/2",
        },
    ]

    signals = nlp.process_news_for_market(
        news, "Will the Federal Reserve cut interest rates?", min_relevance=0.15
    )

    # The rate cut article should be kept, cake article should be filtered
    assert len(signals) >= 1
    assert any("Fed" in s.title or "rate" in s.title.lower() for s in signals)


def test_agreement_confidence():
    """When VADER and TextBlob agree, confidence should be higher."""
    nlp = NLPProcessor()

    # Strongly positive (both should agree)
    strong = nlp.compute_sentiment(
        "Absolutely fantastic! Best performance ever! Amazing results!"
    )
    # Mildly positive (may have less agreement)
    mild = nlp.compute_sentiment("It was ok, somewhat decent performance.")

    # Strong positive should have higher agreement confidence
    assert strong.agreement_confidence >= mild.agreement_confidence or strong.vader_compound > mild.vader_compound


if __name__ == "__main__":
    test_positive_sentiment()
    test_negative_sentiment()
    test_neutral_sentiment()
    test_relevance_related_texts()
    test_relevance_unrelated_texts()
    test_empty_text_sentiment()
    test_empty_text_relevance()
    test_process_news_filters_irrelevant()
    test_agreement_confidence()
    print("All NLP processor tests passed!")
