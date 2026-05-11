"""
Research tools — pluggable, async, with graceful degradation.
Each tool returns List[Source] with credibility scoring.
"""
from __future__ import annotations
import asyncio
import hashlib
from datetime import datetime
from typing import Any
import httpx
import structlog

from app.core.config import settings
from app.schemas.research import Source, SourceType

log = structlog.get_logger(__name__)

HEADERS = {
    "User-Agent": "MultiAgentResearch/1.0 (research assistant; contact@example.com)",
    "Accept": "application/json",
}


def _credibility_score(source: dict[str, Any]) -> float:
    """Heuristic credibility scoring from source metadata."""
    score = 0.5
    url = source.get("url", "").lower()
    # Academic/gov/org domains get a boost
    if any(d in url for d in [".edu", ".gov", ".org", "arxiv", "pubmed", "nature.com"]):
        score += 0.25
    if source.get("citation_count", 0) > 10:
        score += 0.1
    if source.get("published_date"):
        try:
            pub_date = datetime.fromisoformat(str(source["published_date"]))
            days_old = (datetime.utcnow() - pub_date).days
            if days_old < 365:
                score += 0.1
        except Exception:
            pass
    return min(round(score, 3), 1.0)


class WebSearchTool:
    """
    Web search via Brave Search API (primary) or SerpAPI (fallback).
    Falls back to DuckDuckGo scrape if neither API key is configured.
    """

    async def search(self, query: str, max_results: int = 8) -> list[Source]:
        if settings.brave_api_key:
            return await self._brave_search(query, max_results)
        elif settings.serpapi_key:
            return await self._serp_search(query, max_results)
        else:
            return await self._fallback_search(query, max_results)

    async def _brave_search(self, query: str, max_results: int) -> list[Source]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={**HEADERS, "X-Subscription-Token": settings.brave_api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("web", {}).get("results", [])
                return [
                    Source(
                        url=r.get("url", ""),
                        title=r.get("title", "Untitled"),
                        snippet=r.get("description", ""),
                        source_type=SourceType.WEB,
                        credibility_score=_credibility_score({"url": r.get("url", "")}),
                    )
                    for r in results
                    if r.get("url")
                ]
        except Exception as exc:
            log.warning("web_search.brave_failed", error=str(exc), query=query[:50])
            return []

    async def _serp_search(self, query: str, max_results: int) -> list[Source]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "num": max_results,
                        "api_key": settings.serpapi_key,
                        "engine": "google",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("organic_results", [])
                return [
                    Source(
                        url=r.get("link", ""),
                        title=r.get("title", "Untitled"),
                        snippet=r.get("snippet", ""),
                        source_type=SourceType.WEB,
                        credibility_score=_credibility_score({"url": r.get("link", "")}),
                    )
                    for r in results
                    if r.get("link")
                ]
        except Exception as exc:
            log.warning("web_search.serp_failed", error=str(exc), query=query[:50])
            return []

    async def _fallback_search(self, query: str, max_results: int) -> list[Source]:
        """DuckDuckGo instant answer API (no key required, limited results)."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": 1},
                )
                data = resp.json()
                results = data.get("RelatedTopics", [])[:max_results]
                return [
                    Source(
                        url=r.get("FirstURL", "https://duckduckgo.com"),
                        title=r.get("Text", query)[:100],
                        snippet=r.get("Text", ""),
                        source_type=SourceType.WEB,
                        credibility_score=0.4,
                    )
                    for r in results
                    if isinstance(r, dict) and r.get("FirstURL")
                ]
        except Exception as exc:
            log.warning("web_search.fallback_failed", error=str(exc))
            return []


class ArxivTool:
    """arXiv API client — returns academic papers with citation metadata."""

    async def search(self, query: str, max_results: int = 5) -> list[Source]:
        try:
            import urllib.parse
            encoded = urllib.parse.quote(query)
            url = (
                f"https://export.arxiv.org/api/query"
                f"?search_query=all:{encoded}&start=0&max_results={max_results}"
                f"&sortBy=relevance&sortOrder=descending"
            )
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, headers=HEADERS)
                resp.raise_for_status()

            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            sources = []

            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                published_el = entry.find("atom:published", ns)
                authors = [
                    a.find("atom:name", ns).text  # type: ignore
                    for a in entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None
                ]

                if id_el is None:
                    continue

                arxiv_id = id_el.text or ""
                pub_date = None
                if published_el is not None and published_el.text:
                    try:
                        pub_date = datetime.fromisoformat(
                            published_el.text.replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                sources.append(Source(
                    url=arxiv_id,
                    title=(title_el.text or "").strip() if title_el is not None else query,
                    snippet=(summary_el.text or "")[:400].strip() if summary_el is not None else "",
                    source_type=SourceType.ARXIV,
                    published_date=pub_date,
                    authors=authors[:5],
                    credibility_score=0.85,  # arXiv papers are higher credibility by default
                ))

            return sources

        except Exception as exc:
            log.warning("arxiv.search_failed", error=str(exc), query=query[:50])
            return []


class PDFTool:
    """Fetches and extracts text from PDF URLs."""

    async def extract(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, follow_redirects=True)
                if "pdf" not in resp.headers.get("content-type", "").lower():
                    return ""

            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(resp.content))
            text = ""
            for page in reader.pages[:10]:  # First 10 pages max
                text += page.extract_text() or ""
            return text[:5000]
        except Exception as exc:
            log.warning("pdf.extract_failed", url=url[:80], error=str(exc))
            return ""
