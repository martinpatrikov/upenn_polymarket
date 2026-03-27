"""Twitter/News fetcher for AI model signals via Nitter RSS and AI news feeds."""

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import feedparser
import httpx

from config.settings import (
    Tweet,
    NewsSignal,
    NITTER_INSTANCES,
    AI_TWITTER_ACCOUNTS,
    AI_NEWS_RSS_FEEDS,
    SOURCE_CREDIBILITY,
)

logger = logging.getLogger(__name__)


class TwitterFetcher:
    """Fetches AI model news from Nitter RSS feeds and AI news RSS feeds."""

    def __init__(
        self,
        nitter_instances: Optional[list[str]] = None,
        accounts: Optional[list[str]] = None,
        ai_rss_feeds: Optional[dict[str, str]] = None,
        max_age_hours: int = 24,
    ):
        self.nitter_instances = nitter_instances or NITTER_INSTANCES
        self.accounts = accounts or AI_TWITTER_ACCOUNTS
        self.ai_rss_feeds = ai_rss_feeds or AI_NEWS_RSS_FEEDS
        self.max_age_hours = max_age_hours
        self._http = httpx.Client(timeout=15.0, follow_redirects=True)

    def fetch_all_signals(self) -> list[NewsSignal]:
        """Fetch from all sources: Nitter + AI news RSS."""
        signals: list[NewsSignal] = []

        # Source 1: Nitter RSS for official AI company accounts
        tweets = self._fetch_nitter_tweets()
        for tweet in tweets:
            signals.append(NewsSignal(
                source="twitter" if tweet.is_official_account else "nitter",
                title=tweet.text[:100],
                summary=tweet.text,
                url=tweet.url,
                published_at=tweet.created_at,
                source_credibility=SOURCE_CREDIBILITY.get(
                    "twitter" if tweet.is_official_account else "nitter", 0.5
                ),
            ))

        # Source 2: AI news RSS feeds
        news = self._fetch_ai_news_rss()
        signals.extend(news)

        # Deduplicate by title similarity
        signals = self._deduplicate(signals)

        logger.info(f"Fetched {len(signals)} AI signals ({len(tweets)} tweets, {len(news)} news)")
        return signals

    def _fetch_nitter_tweets(self) -> list[Tweet]:
        """Fetch tweets from Nitter RSS feeds for tracked accounts.
        Returns empty list if no Nitter instances are configured.
        """
        if not self.nitter_instances:
            return []

        tweets: list[Tweet] = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)

        for account in self.accounts:
            for instance in self.nitter_instances:
                try:
                    url = f"{instance}/{account}/rss"
                    resp = self._http.get(url)
                    if resp.status_code != 200:
                        continue

                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries:
                        pub_date = self._parse_date(entry.get("published", ""))
                        if pub_date and pub_date > cutoff:
                            text = self._clean_html(entry.get("title", "") or entry.get("summary", ""))
                            tweets.append(Tweet(
                                author=account,
                                text=text,
                                created_at=pub_date,
                                engagement=0,
                                is_official_account=True,
                                url=entry.get("link", ""),
                            ))
                    # Success - don't try other instances for this account
                    break
                except Exception as e:
                    logger.debug(f"Nitter {instance}/{account} failed: {e}")
                    continue

        return tweets

    def _fetch_ai_news_rss(self) -> list[NewsSignal]:
        """Fetch from AI-focused news RSS feeds."""
        signals: list[NewsSignal] = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)

        # AI keywords for filtering
        ai_keywords = {
            "ai model", "language model", "llm", "chatbot", "benchmark",
            "arena", "leaderboard", "anthropic", "openai", "claude",
            "gpt", "gemini", "llama", "grok", "mistral", "deepseek",
            "sota", "state of the art", "elo", "ranking",
        }

        for source_name, feed_url in self.ai_rss_feeds.items():
            try:
                resp = self._http.get(feed_url)
                if resp.status_code != 200:
                    continue

                feed = feedparser.parse(resp.text)
                for entry in feed.entries:
                    pub_date = self._parse_date(entry.get("published", ""))
                    if not pub_date or pub_date < cutoff:
                        continue

                    title = entry.get("title", "")
                    summary = self._clean_html(entry.get("summary", ""))
                    combined = (title + " " + summary).lower()

                    # Only include AI-relevant articles
                    if not any(kw in combined for kw in ai_keywords):
                        continue

                    signals.append(NewsSignal(
                        source=source_name,
                        title=title,
                        summary=summary[:500],
                        url=entry.get("link", ""),
                        published_at=pub_date,
                        source_credibility=SOURCE_CREDIBILITY.get(source_name, 0.5),
                    ))
            except Exception as e:
                logger.debug(f"AI news RSS {source_name} failed: {e}")
                continue

        return signals

    def _deduplicate(self, signals: list[NewsSignal]) -> list[NewsSignal]:
        """Remove duplicate signals based on title word overlap."""
        if not signals:
            return signals

        unique: list[NewsSignal] = []
        seen_words: list[set[str]] = []

        for signal in signals:
            words = set(signal.title.lower().split())
            if len(words) < 3:
                unique.append(signal)
                seen_words.append(words)
                continue

            is_dup = False
            for prev_words in seen_words:
                if not prev_words:
                    continue
                overlap = len(words & prev_words) / max(len(words | prev_words), 1)
                if overlap > 0.7:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(signal)
                seen_words.append(words)

        return unique

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds."""
        if not date_str:
            return None
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str).replace(tzinfo=timezone.utc)
        except Exception:
            pass
        # Try ISO format
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text).strip()
