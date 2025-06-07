from __future__ import annotations

from typing import Any

import feedparser
from langchain_core.tools import BaseTool


class NewsFetcher(BaseTool):
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

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        max_entries = args[0] if args else kwargs.get('max_entries', 30)
        url = self.FEED_URL

        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:max_entries]
            headlines = [str(entry.title) for entry in entries]
            return "\n".join(headlines)
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching news: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        # For RSS feed parsing without specific async requirements,
        # we can delegate to the sync implementation
        return self._run(*args, **kwargs)


if __name__ == "__main__":
    tool = NewsFetcher()
    print(tool._run())
