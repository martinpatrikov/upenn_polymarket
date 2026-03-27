"""Reddit-based sentiment fetcher. Optional -- degrades gracefully if no credentials."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    title: str
    body: str
    score: int
    num_comments: int
    created_utc: datetime
    top_comments: list[str] = field(default_factory=list)
    subreddit: str = ""


class RedditFetcher:
    """Fetches Reddit posts and comments for market-related sentiment."""

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "polymarket-agent/1.0",
    ):
        self._reddit = None
        self._available = False

        if client_id and client_secret:
            try:
                import praw

                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                # Verify connection
                self._reddit.read_only = True
                self._available = True
                logger.info("Reddit API connected successfully")
            except Exception as e:
                logger.warning(f"Reddit API unavailable: {e}. Continuing without Reddit data.")
        else:
            logger.info("Reddit credentials not configured. Skipping Reddit data source.")

    @property
    def available(self) -> bool:
        return self._available

    def fetch_relevant_posts(
        self,
        query: str,
        subreddits: list[str] | None = None,
        max_posts: int = 25,
        max_age_hours: int = 48,
    ) -> list[RedditPost]:
        """Search subreddits for posts matching query terms."""
        if not self._available:
            return []

        subreddits = subreddits or ["polymarket", "politics", "worldnews"]
        posts: list[RedditPost] = []
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)

        try:
            for sub_name in subreddits:
                try:
                    subreddit = self._reddit.subreddit(sub_name)
                    for submission in subreddit.search(query, sort="new", limit=max_posts // len(subreddits)):
                        if submission.created_utc < cutoff:
                            continue

                        # Get top comments
                        submission.comment_sort = "best"
                        submission.comments.replace_more(limit=0)
                        top_comments = [
                            c.body
                            for c in submission.comments[:5]
                            if hasattr(c, "body")
                        ]

                        posts.append(
                            RedditPost(
                                title=submission.title,
                                body=submission.selftext or "",
                                score=submission.score,
                                num_comments=submission.num_comments,
                                created_utc=datetime.fromtimestamp(
                                    submission.created_utc, tz=timezone.utc
                                ),
                                top_comments=top_comments,
                                subreddit=sub_name,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to search r/{sub_name}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")

        return posts

    def fetch_subreddit_sentiment(
        self, subreddit: str, limit: int = 50
    ) -> list[RedditPost]:
        """Get hot posts from a subreddit for general sentiment."""
        if not self._available:
            return []

        posts: list[RedditPost] = []
        try:
            sub = self._reddit.subreddit(subreddit)
            for submission in sub.hot(limit=limit):
                posts.append(
                    RedditPost(
                        title=submission.title,
                        body=submission.selftext or "",
                        score=submission.score,
                        num_comments=submission.num_comments,
                        created_utc=datetime.fromtimestamp(
                            submission.created_utc, tz=timezone.utc
                        ),
                        subreddit=subreddit,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to fetch r/{subreddit}: {e}")

        return posts
