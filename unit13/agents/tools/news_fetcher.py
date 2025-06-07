from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import feedparser


@dataclass
class NewsFetcher:
    """Fetch latest Bitcoin news from a public RSS feed."""

    name: str = "news_fetcher"
    description: str = "Fetch the latest Bitcoin-related news headlines."

    FEED_URL: str = "https://news.bitcoin.com/feed/"

    def prompt_description(self) -> str:
        return f'''
{self.name}:
    Example usage: {self.name} 30
    Return the latest 30 news about bitcoin in plain text
'''.strip()

    def __call__(self, max_entries: int = 30) -> str:
        url = self.FEED_URL

        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:max_entries]
            headlines = [str(entry.title) for entry in entries]
            return "\n".join(headlines)
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching news: {exc}"


if __name__ == "__main__":
    tool = NewsFetcher()
    print(tool())
