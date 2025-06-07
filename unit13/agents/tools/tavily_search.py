from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

from tavily import TavilyClient


@dataclass
class TavilySearcher:
    """Perform web searches using Tavily's search API."""

    name: str = "tavily_search"
    description: str = (
        "Search the web for current information using Tavily's powerful search API."
    )

    def prompt_description(self) -> str:
        return f'''
{self.name}:
    Example usage: {self.name} "latest Bitcoin news"
    Search the web for real-time information about any topic and return results in JSON format
'''.strip()

    def __call__(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "advanced",
        topic: Literal["general", "news", "finance"] = "finance",
        max_results: int = 5,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> str:
        """
        Search the web using Tavily API.
        
        Args:
            query: The search query
            search_depth: "basic" or "advanced" search depth
            topic: "general" or "news" search topic
            max_results: Maximum number of results (0-20)
            include_answer: Include AI-generated answer
            include_raw_content: Include cleaned HTML content
            include_images: Include related images
            include_domains: List of domains to specifically include
            exclude_domains: List of domains to exclude
        """
        try:
            # Get API key from environment
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Error: TAVILY_API_KEY environment variable not set"

            # Initialize Tavily client
            client = TavilyClient(api_key=api_key)

            # Perform search
            response = client.search(
                query=query,
                search_depth=search_depth,
                topic=topic,
                max_results=max_results,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            # Return formatted JSON response
            return json.dumps(response, indent=2)

        except Exception as exc:  # pragma: no cover - network access
            return f"Error performing search: {exc}"


if __name__ == "__main__":
    tool = TavilySearcher()
    print(tool("latest developments in AI related to finance and Bitcoin")) 