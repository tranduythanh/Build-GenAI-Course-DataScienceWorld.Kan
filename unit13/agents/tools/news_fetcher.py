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

    def __call__(self, args: Dict[str, Any] | None = None) -> str:
        url = self.FEED_URL
        if args:
            url = args.get("url", url)
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:3]
            headlines = [entry.title for entry in entries]
            return " | ".join(headlines)
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching news: {exc}"


if __name__ == "__main__":
    tool = NewsFetcher()
    print(tool())
