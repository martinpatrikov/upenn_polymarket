"""RSS-based news fetcher for real-time news ingestion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import feedparser

from config.settings import RSS_FEEDS, SOURCE_CREDIBILITY

logger = logging.getLogger(__name__)

# Common English stop words for keyword extraction
STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very and but or "
    "if this that these those it its what which who whom".split()
)


@dataclass
class RawNewsItem:
    title: str
    summary: str
    published_at: datetime
    source_name: str
    url: str


class NewsFetcher:
    """Fetches and parses news from RSS feeds including Google News search."""

    GOOGLE_NEWS_SEARCH_URL = (
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    def __init__(self, feeds: Optional[dict[str, str]] = None):
        self.feeds = feeds or RSS_FEEDS
        self._seen_urls: set[str] = set()

    def fetch_news_for_market(
        self, market_question: str, max_age_hours: int = 24
    ) -> list[RawNewsItem]:
        """Fetch news relevant to a specific market question."""
        items: list[RawNewsItem] = []

        # Build a Google News search from the market question
        search_terms = self._extract_search_terms(market_question)
        if search_terms:
            google_url = self.GOOGLE_NEWS_SEARCH_URL.format(
                query=quote_plus(search_terms)
            )
            items.extend(self._parse_feed(google_url, "google_news"))

        # Also pull from standing feeds
        items.extend(self.fetch_all_feeds())

        # Filter by age
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        items = [
            item
            for item in items
            if item.published_at.timestamp() > cutoff
        ]

        return self._deduplicate(items)

    def fetch_all_feeds(self) -> list[RawNewsItem]:
        """Pull articles from all configured RSS feeds."""
        items: list[RawNewsItem] = []
        for name, url in self.feeds.items():
            items.extend(self._parse_feed(url, name))
        return self._deduplicate(items)

    def _parse_feed(self, url: str, source_name: str) -> list[RawNewsItem]:
        """Parse a single RSS feed URL."""
        try:
            feed = feedparser.parse(url)
            items = []
            for entry in feed.entries[:30]:  # cap per feed
                published = self._parse_date(entry)
                item = RawNewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    published_at=published,
                    source_name=source_name,
                    url=entry.get("link", ""),
                )
                items.append(item)
            return items
        except Exception as e:
            logger.warning(f"Failed to parse feed {source_name}: {e}")
            return []

    def _parse_date(self, entry) -> datetime:
        """Extract publication date from a feed entry."""
        for attr in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    from time import mktime
                    return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
                except Exception:
                    pass
        return datetime.now(timezone.utc)

    def _extract_search_terms(self, question: str) -> str:
        """Extract meaningful keywords from a market question for search."""
        # Remove punctuation and lowercase
        text = re.sub(r"[^\w\s]", "", question.lower())
        words = text.split()
        # Filter stop words and short words
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        # Take top 5 keywords
        return " ".join(keywords[:5])

    def _deduplicate(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        """Remove duplicate articles by URL and title similarity."""
        unique: list[RawNewsItem] = []
        seen_titles: list[set[str]] = []

        for item in items:
            # Skip exact URL duplicates
            if item.url in self._seen_urls:
                continue

            # Check title similarity (Jaccard on word sets)
            title_words = set(item.title.lower().split())
            is_dup = False
            for seen in seen_titles:
                if not title_words or not seen:
                    continue
                jaccard = len(title_words & seen) / len(title_words | seen)
                if jaccard > 0.7:
                    is_dup = True
                    break

            if not is_dup:
                self._seen_urls.add(item.url)
                seen_titles.append(title_words)
                unique.append(item)

        return unique

    def get_source_credibility(self, source_name: str) -> float:
        """Get credibility score for a news source."""
        # Check for partial matches
        source_lower = source_name.lower()
        for key, score in SOURCE_CREDIBILITY.items():
            if key in source_lower:
                return score
        return SOURCE_CREDIBILITY.get("generic", 0.5)
