"""Web search and safe direct-PDF download helpers for Industrial AI RAG."""
from __future__ import annotations

import os
import re
from io import BytesIO
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class NamedBytesIO(BytesIO):
    """BytesIO wrapper compatible with the existing PDF upload processor."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def search_web_documents(query: str, max_results: int = 8) -> list[dict]:
    """Search the web and return normalized results for human review."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing. Add it to your .env file and restart Streamlit.")
    if not query or not query.strip():
        return []

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=f"{query.strip()} industrial manufacturing engineering PDF technical manual",
        max_results=max(1, min(int(max_results), 10)),
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )

    results = []
    for item in response.get("results", []):
        url = str(item.get("url", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        results.append({
            "title": item.get("title") or url,
            "url": url,
            "snippet": item.get("content") or "",
            "score": item.get("score"),
            "is_pdf_url": parsed.path.lower().endswith(".pdf"),
        })
    return results


def download_pdf_from_url(url: str, max_bytes: int = 50 * 1024 * 1024) -> NamedBytesIO:
    """Download a direct PDF URL with size and content checks; never execute content."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only valid HTTP/HTTPS URLs can be downloaded.")

    request = Request(url, headers={"User-Agent": "IndustrialAIResearchAssistant/1.0"})
    with urlopen(request, timeout=25) as response:
        final_url = response.geturl()
        final_parsed = urlparse(final_url)
        if final_parsed.scheme not in {"http", "https"}:
            raise ValueError("The download redirected to an unsupported URL scheme.")
        content_type = response.headers.get("Content-Type", "").lower()
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError("PDF exceeds the 50 MB download limit.")
        data = response.read(max_bytes + 1)

    if len(data) > max_bytes:
        raise ValueError("PDF exceeds the 50 MB download limit.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("This URL did not return a valid PDF file. It may be a webpage rather than a direct PDF.")

    filename = unquote(os.path.basename(urlparse(final_url).path)) or "web_document.pdf"
    filename = re.sub(r"[^A-Za-z0-9._ -]", "_", filename).strip(" .")
    if not filename.lower().endswith(".pdf"):
        filename = "web_document.pdf"
    return NamedBytesIO(data, filename)
