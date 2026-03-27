"""NLP processor for sentiment analysis and relevance scoring.

Uses an ensemble of VADER (social media optimized) and TextBlob (general purpose)
for sentiment, and TF-IDF cosine similarity for relevance scoring.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from config.settings import NewsSignal

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    vader_compound: float
    textblob_polarity: float
    combined_score: float
    agreement_confidence: float


class NLPProcessor:
    """Extracts sentiment and relevance from news text using lightweight NLP."""

    VADER_WEIGHT = 0.65
    TEXTBLOB_WEIGHT = 0.35

    def __init__(self):
        # Initialize VADER
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()

        # Initialize TextBlob (import check)
        try:
            from textblob import TextBlob
            self._textblob_available = True
        except ImportError:
            logger.warning("TextBlob not available, using VADER only")
            self._textblob_available = False

        # Initialize TF-IDF vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._tfidf_fitted = False

    def compute_sentiment(self, text: str) -> SentimentResult:
        """Compute sentiment using VADER + TextBlob ensemble.

        VADER: Optimized for social media and news headlines. Returns compound
        score in [-1, 1]. Handles capitalization, punctuation emphasis, slang.

        TextBlob: Pattern-based sentiment. Returns polarity in [-1, 1].
        More conservative, better for formal text.

        Combined: Weighted average (0.65 VADER, 0.35 TextBlob).
        Agreement confidence: Higher when both models agree on direction.
        """
        # Clean text
        text = self._clean_text(text)
        if not text:
            return SentimentResult(0.0, 0.0, 0.0, 0.0)

        # VADER
        vader_scores = self._vader.polarity_scores(text)
        vader_compound = vader_scores["compound"]

        # TextBlob
        textblob_polarity = 0.0
        if self._textblob_available:
            from textblob import TextBlob
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity

        # Combined score
        combined = (
            self.VADER_WEIGHT * vader_compound
            + self.TEXTBLOB_WEIGHT * textblob_polarity
        )

        # Agreement confidence: both positive or both negative = high confidence
        if vader_compound == 0 and textblob_polarity == 0:
            agreement = 0.5
        elif (vader_compound >= 0 and textblob_polarity >= 0) or (
            vader_compound <= 0 and textblob_polarity <= 0
        ):
            # Same direction — confidence based on magnitude agreement
            agreement = 0.7 + 0.3 * (1 - abs(vader_compound - textblob_polarity))
        else:
            # Disagreement — lower confidence
            agreement = 0.3 * (1 - abs(vader_compound - textblob_polarity))

        return SentimentResult(
            vader_compound=vader_compound,
            textblob_polarity=textblob_polarity,
            combined_score=combined,
            agreement_confidence=agreement,
        )

    def compute_relevance(self, news_text: str, market_question: str) -> float:
        """Compute relevance using an ensemble of TF-IDF cosine similarity and keyword overlap.

        Combines two approaches for robustness:
        1. TF-IDF cosine similarity (captures semantic similarity via term frequency)
        2. Keyword overlap (robust for short texts where TF-IDF is sparse)

        Returns a score in [0, 1] where 1 means perfectly relevant.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        news_text = self._clean_text(news_text)
        market_question = self._clean_text(market_question)

        if not news_text or not market_question:
            return 0.0

        # Method 1: TF-IDF cosine similarity
        tfidf_score = 0.0
        try:
            tfidf_matrix = self._tfidf.fit_transform([market_question, news_text])
            tfidf_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception as e:
            logger.debug(f"TF-IDF relevance failed: {e}")

        # Method 2: Keyword overlap (more robust for short texts)
        keyword_score = self._keyword_fallback_relevance(news_text, market_question)

        # Ensemble: take the max of both approaches, with a small boost for agreement
        combined = max(tfidf_score, keyword_score)
        if tfidf_score > 0.1 and keyword_score > 0.1:
            # Both methods agree there's relevance — boost slightly
            combined = min(1.0, combined + 0.1)

        return float(max(0.0, min(1.0, combined)))

    def _keyword_fallback_relevance(
        self, news_text: str, market_question: str
    ) -> float:
        """Keyword overlap relevance with prefix matching for better recall.

        Uses both exact and prefix matching (first 5 chars) to handle
        morphological variants (e.g., 'candidate' matches 'candidates').
        Filters stop words for cleaner signal.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "can", "could", "of", "in", "to", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "this", "that",
            "these", "those", "it", "its", "and", "but", "or", "if", "not",
        }

        news_words = set(w for w in news_text.lower().split() if w not in stop_words and len(w) > 2)
        question_words = set(w for w in market_question.lower().split() if w not in stop_words and len(w) > 2)

        if not question_words:
            return 0.0

        # Exact match count
        exact_overlap = len(news_words & question_words)

        # Prefix match (first 5 chars) for morphological variants
        news_prefixes = set(w[:5] for w in news_words if len(w) >= 5)
        question_prefixes = set(w[:5] for w in question_words if len(w) >= 5)
        prefix_overlap = len(news_prefixes & question_prefixes)

        # Combined: weight exact matches more heavily
        total_overlap = exact_overlap + 0.5 * max(0, prefix_overlap - exact_overlap)
        score = total_overlap / len(question_words)

        return min(1.0, score)

    def process_news_for_market(
        self,
        news_items: list[dict],
        market_question: str,
        min_relevance: float = 0.15,
    ) -> list[NewsSignal]:
        """Full NLP pipeline: compute relevance and sentiment for each news item.

        Args:
            news_items: List of dicts with keys: title, summary, published_at, source_name, url
            market_question: The market question to score relevance against
            min_relevance: Minimum relevance threshold to keep

        Returns:
            List of NewsSignal objects, filtered and sorted by relevance descending.
        """
        signals: list[NewsSignal] = []

        for item in news_items:
            title = item.get("title", "")
            summary = item.get("summary", "")
            text = f"{title}. {summary}"

            # Compute relevance
            relevance = self.compute_relevance(text, market_question)
            if relevance < min_relevance:
                continue

            # Compute sentiment
            sentiment = self.compute_sentiment(text)

            # Compute hours since publication
            published_at = item.get("published_at", datetime.now(timezone.utc))
            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(published_at)
                except ValueError:
                    published_at = datetime.now(timezone.utc)

            hours_ago = (
                datetime.now(timezone.utc) - published_at.replace(tzinfo=timezone.utc)
                if published_at.tzinfo is None
                else datetime.now(timezone.utc) - published_at
            ).total_seconds() / 3600

            # Get source credibility
            from config.settings import SOURCE_CREDIBILITY
            source_name = item.get("source_name", "generic")
            credibility = SOURCE_CREDIBILITY.get("generic", 0.5)
            for key, score in SOURCE_CREDIBILITY.items():
                if key in source_name.lower():
                    credibility = score
                    break

            signals.append(
                NewsSignal(
                    source=source_name,
                    title=title,
                    summary=summary,
                    url=item.get("url", ""),
                    published_at=published_at,
                    relevance_score=relevance,
                    sentiment_score=sentiment.combined_score,
                    source_credibility=credibility,
                    hours_ago=max(0.0, hours_ago),
                )
            )

        # Sort by relevance descending
        signals.sort(key=lambda s: s.relevance_score, reverse=True)
        return signals

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for NLP processing."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
