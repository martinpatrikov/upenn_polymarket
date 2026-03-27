"""Tests for the Arena leaderboard scraper."""

import pytest
import sys
sys.path.insert(0, ".")

from agent.data_ingestion.arena_scraper import ArenaScraper
from config.settings import ArenaModel


class TestArenaScraper:
    def setup_method(self):
        self.scraper = ArenaScraper()

    def test_classify_company_anthropic(self):
        assert self.scraper._classify_company("claude-opus-4") == "Anthropic"
        assert self.scraper._classify_company("claude-sonnet-4") == "Anthropic"

    def test_classify_company_openai(self):
        assert self.scraper._classify_company("gpt-4o") == "OpenAI"
        assert self.scraper._classify_company("o1-preview") == "OpenAI"

    def test_classify_company_google(self):
        assert self.scraper._classify_company("gemini-2.0-flash") == "Google"

    def test_classify_company_meta(self):
        assert self.scraper._classify_company("llama-3.1-405b") == "Meta"

    def test_classify_unknown(self):
        assert self.scraper._classify_company("unknown-model") is None

    def test_manual_elos(self):
        self.scraper.set_manual_elos({"claude-opus-4": 1504, "gpt-4o": 1490})
        models = self.scraper.fetch_leaderboard()
        assert len(models) == 2
        assert models[0].elo_score in [1504, 1490]

    def test_get_company_best_with_manual(self):
        self.scraper.set_manual_elos({
            "claude-opus-4": 1504,
            "claude-sonnet-4": 1480,
            "gpt-4o": 1490,
        })
        company_best = self.scraper.get_company_best_models()
        assert "Anthropic" in company_best
        assert company_best["Anthropic"].elo_score == 1504
        assert "OpenAI" in company_best
        assert company_best["OpenAI"].elo_score == 1490

    def test_parse_table_data(self):
        headers = ["Model", "ELO", "CI", "Votes"]
        rows = [
            ["claude-opus-4", "1504", "6", "10000"],
            ["gpt-4o", "1490", "5", "8000"],
        ]
        models = self.scraper._parse_table_data(headers, rows)
        assert len(models) == 2
        assert models[0].elo_score == 1504
