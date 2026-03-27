"""Chatbot Arena leaderboard scraper for AI model ELO scores.

The Arena page (lmarena.ai) is a Next.js app that embeds leaderboard data
in React Server Components (RSC) payloads inside <script> tags. The entries
are in an "entries" array with fields:
  - modelDisplayName, modelOrganization, rating, ratingUpper, ratingLower, votes
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from config.settings import ArenaModel, COMPANY_MODEL_PREFIXES

logger = logging.getLogger(__name__)

LEADERBOARD_URL = "https://lmarena.ai/leaderboard/text"


class ArenaScraper:
    """Fetches and parses Chatbot Arena leaderboard data."""

    def __init__(self, cache_ttl_seconds: int = 300, company_prefixes: Optional[dict] = None):
        self.cache_ttl = cache_ttl_seconds
        self.company_prefixes = company_prefixes or COMPANY_MODEL_PREFIXES
        self._cache: list[ArenaModel] = []
        self._cache_time: float = 0.0
        self._http = httpx.Client(timeout=30.0, follow_redirects=True)

    def fetch_leaderboard(self) -> list[ArenaModel]:
        """Fetch all models from the Arena leaderboard. Uses cache if fresh."""
        if self._cache and (time.time() - self._cache_time) < self.cache_ttl:
            return self._cache

        try:
            models = self._scrape_leaderboard()
            if models:
                self._cache = models
                self._cache_time = time.time()
                logger.info(f"Fetched {len(models)} models from Arena leaderboard")
            return models
        except Exception as e:
            logger.error(f"Failed to scrape Arena leaderboard: {e}")
            return self._cache

    def get_company_best_models(self) -> dict[str, ArenaModel]:
        """Get the best model per company based on ELO scores."""
        models = self.fetch_leaderboard()
        company_best: dict[str, ArenaModel] = {}

        for model in models:
            company = self._classify_company(model.model_name)
            if company and (company not in company_best or model.elo_score > company_best[company].elo_score):
                company_best[company] = ArenaModel(
                    model_name=model.model_name,
                    company=company,
                    elo_score=model.elo_score,
                    elo_ci=model.elo_ci,
                    vote_count=model.vote_count,
                    last_updated=model.last_updated,
                )

        return company_best

    def _classify_company(self, model_name: str) -> Optional[str]:
        """Map a model name to a company using prefix matching."""
        name_lower = model_name.lower()
        for company, prefixes in self.company_prefixes.items():
            for prefix in prefixes:
                if name_lower.startswith(prefix):
                    return company
        return None

    def _scrape_leaderboard(self) -> list[ArenaModel]:
        """Scrape the leaderboard page and extract model data."""
        resp = self._http.get(LEADERBOARD_URL)
        resp.raise_for_status()
        html = resp.text

        # Primary: extract from RSC payload (Next.js embedded data)
        models = self._extract_from_rsc_payload(html)
        if models:
            return models

        # Fallback: try generic JSON patterns
        models = self._extract_from_json_patterns(html)
        if models:
            return models

        logger.warning("Could not extract leaderboard data from page")
        return []

    def _extract_from_rsc_payload(self, html: str) -> list[ArenaModel]:
        """Extract leaderboard entries from Next.js RSC payload.

        The Arena page embeds data in self.__next_f.push() calls inside
        <script> tags. The leaderboard entries are in an "entries" array
        within a large script containing the arena data.
        """
        now = datetime.now(timezone.utc)
        models = []

        scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)

        for script in scripts:
            # Look for scripts with entries and rating — the leaderboard data
            # Raw HTML has escaped quotes: \"entries\" or \\"entries\\"
            if 'entries' not in script or 'rating' not in script:
                continue
            if len(script) < 10000:
                continue

            try:
                # RSC format: self.__next_f.push([1,"...escaped content..."])
                # Content is double-escaped: \\\" in the raw HTML
                content = script
                # Unescape: replace \\\" and \" with plain "
                content = content.replace('\\\\\\"', '"').replace('\\"', '"')

                # Find the "entries" array
                entries_idx = content.find('"entries":[')
                if entries_idx < 0:
                    continue

                arr_start = content.find('[', entries_idx)
                if arr_start < 0:
                    continue

                # Parse array by tracking bracket depth
                depth = 0
                arr_end = arr_start
                for j in range(arr_start, min(arr_start + 500000, len(content))):
                    if content[j] == '[':
                        depth += 1
                    elif content[j] == ']':
                        depth -= 1
                        if depth == 0:
                            arr_end = j + 1
                            break

                entries = json.loads(content[arr_start:arr_end])
                logger.debug(f"Found {len(entries)} entries in RSC payload")

                for entry in entries:
                    name = entry.get("modelDisplayName", "")
                    org = entry.get("modelOrganization", "")
                    rating = entry.get("rating", 0)
                    rating_upper = entry.get("ratingUpper", 0)
                    rating_lower = entry.get("ratingLower", 0)
                    votes = entry.get("votes", 0)

                    if not name or not rating:
                        continue

                    # CI is the half-width of the confidence interval
                    ci = (rating_upper - rating_lower) / 2.0 if rating_upper and rating_lower else 0.0

                    models.append(ArenaModel(
                        model_name=name,
                        company=org,
                        elo_score=float(rating),
                        elo_ci=ci,
                        vote_count=int(votes) if votes else 0,
                        last_updated=now,
                    ))

                if models:
                    return models

            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
                logger.debug(f"Failed to parse RSC payload: {e}")
                continue

        return models

    def _extract_from_json_patterns(self, html: str) -> list[ArenaModel]:
        """Fallback: try to find leaderboard data via generic JSON patterns."""
        now = datetime.now(timezone.utc)
        models = []

        # Look for any JSON array with rating/elo fields
        patterns = [
            r'"entries"\s*:\s*(\[.*?\])',
            r'"leaderboard"\s*:\s*(\[.*?\])',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, html, re.DOTALL):
                try:
                    data = json.loads(match.group(1))
                    if not isinstance(data, list) or len(data) < 5:
                        continue

                    for item in data:
                        name = (
                            item.get("modelDisplayName")
                            or item.get("model_name")
                            or item.get("name")
                            or item.get("model")
                            or ""
                        )
                        rating = (
                            item.get("rating")
                            or item.get("elo")
                            or item.get("score")
                            or 0
                        )
                        votes = item.get("votes", item.get("num_battles", 0))

                        if name and rating:
                            models.append(ArenaModel(
                                model_name=str(name),
                                company=item.get("modelOrganization", ""),
                                elo_score=float(rating),
                                elo_ci=0.0,
                                vote_count=int(votes) if votes else 0,
                                last_updated=now,
                            ))

                    if models:
                        return models
                except (json.JSONDecodeError, ValueError):
                    continue

        return models

    def _parse_table_data(self, headers: list, rows: list) -> list[ArenaModel]:
        """Parse tabular data (headers + rows) into ArenaModel list."""
        models = []
        now = datetime.now(timezone.utc)

        name_idx = None
        elo_idx = None
        ci_idx = None
        votes_idx = None

        for i, h in enumerate(headers):
            h_lower = str(h).lower()
            if "model" in h_lower or "name" in h_lower:
                name_idx = i
            elif "elo" in h_lower or "rating" in h_lower or "score" in h_lower:
                elo_idx = i
            elif "ci" in h_lower or "interval" in h_lower:
                ci_idx = i
            elif "vote" in h_lower or "battle" in h_lower:
                votes_idx = i

        if name_idx is None or elo_idx is None:
            return models

        for row in rows:
            if len(row) <= max(name_idx, elo_idx):
                continue
            try:
                name = re.sub(r'<[^>]+>', '', str(row[name_idx])).strip()
                elo = float(re.sub(r'[^\d.]', '', str(row[elo_idx])))
                ci = 0.0
                if ci_idx is not None and ci_idx < len(row):
                    ci_str = re.sub(r'[^\d.]', '', str(row[ci_idx]))
                    ci = float(ci_str) if ci_str else 0.0
                votes = 0
                if votes_idx is not None and votes_idx < len(row):
                    votes_str = re.sub(r'[^\d]', '', str(row[votes_idx]))
                    votes = int(votes_str) if votes_str else 0

                if name and elo > 0:
                    models.append(ArenaModel(
                        model_name=name, company="", elo_score=elo,
                        elo_ci=ci, vote_count=votes, last_updated=now,
                    ))
            except (ValueError, IndexError):
                continue

        return models

    def set_manual_elos(self, elo_scores: dict[str, float]) -> None:
        """Manually set ELO scores (fallback if scraping fails)."""
        now = datetime.now(timezone.utc)
        self._cache = [
            ArenaModel(
                model_name=name, company="", elo_score=elo,
                elo_ci=0.0, vote_count=0, last_updated=now,
            )
            for name, elo in elo_scores.items()
        ]
        self._cache_time = time.time()
        logger.info(f"Set {len(self._cache)} manual ELO scores")
