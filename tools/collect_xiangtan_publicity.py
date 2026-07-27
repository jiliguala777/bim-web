from __future__ import annotations

import argparse
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

try:
    from .thermal_candidate_extractor import (
        ThermalCandidate,
        detect_html_encoding,
        extract_candidates_from_html,
        write_candidates_csv,
    )
except ImportError:
    from thermal_candidate_extractor import (
        ThermalCandidate,
        detect_html_encoding,
        extract_candidates_from_html,
        write_candidates_csv,
    )


DEFAULT_INDEX_URL = "https://xtjs.xiangtan.gov.cn/6550/28176/index.htm"


@dataclass
class PublicityPage:
    title: str
    url: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            self.links.append((clean("".join(self._text)), self._href))
            self._href = None
            self._text = []


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def is_material_publicity_title(title: str) -> bool:
    compact = clean(title).replace(" ", "")
    return (
        "建筑节能材料" in compact
        and "公示名单" in compact
        and "公布" in compact
    )


def discover_publicity_pages(html: str, *, base_url: str) -> list[PublicityPage]:
    parser = LinkParser()
    parser.feed(html)
    pages: list[PublicityPage] = []
    seen: set[str] = set()
    for title, href in parser.links:
        if not is_material_publicity_title(title):
            continue
        url = urllib.parse.urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        pages.append(PublicityPage(title=title, url=url))
    return pages


def fetch_text(url: str, *, verify_ssl: bool = True) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 bim-web-xiangtan-collector/0.1"})
    context = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=30) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")
    return raw.decode(detect_html_encoding(raw, content_type), errors="ignore")


def index_page_urls(index_url: str, max_pages: int) -> list[str]:
    urls = [index_url]
    if max_pages <= 1:
        return urls
    base = index_url.rsplit("/", 1)[0]
    for page_number in range(1, max_pages):
        urls.append(f"{base}/index_{page_number}.htm")
    return urls


def collect_xiangtan_candidates(
    *,
    index_url: str = DEFAULT_INDEX_URL,
    max_pages: int = 4,
    verify_ssl: bool = False,
) -> tuple[list[PublicityPage], list[ThermalCandidate]]:
    pages: list[PublicityPage] = []
    seen_page_urls: set[str] = set()
    for page_url in index_page_urls(index_url, max_pages):
        try:
            html = fetch_text(page_url, verify_ssl=verify_ssl)
        except Exception:
            continue
        for page in discover_publicity_pages(html, base_url=page_url):
            if page.url in seen_page_urls:
                continue
            seen_page_urls.add(page.url)
            pages.append(page)

    candidates: list[ThermalCandidate] = []
    for page in pages:
        try:
            html = fetch_text(page.url, verify_ssl=verify_ssl)
        except Exception:
            continue
        candidates.extend(
            extract_candidates_from_html(
                html,
                source_url=page.url,
                source_title=page.title,
            )
        )
    return pages, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Xiangtan building energy material publicity history.")
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument(
        "--output",
        default=str(Path("data") / "external_sources" / "review" / "xiangtan_thermal_parameter_candidates.csv"),
    )
    parser.add_argument("--verify-ssl", action="store_true")
    args = parser.parse_args()

    pages, candidates = collect_xiangtan_candidates(
        index_url=args.index_url,
        max_pages=args.max_pages,
        verify_ssl=args.verify_ssl,
    )
    write_candidates_csv(candidates, args.output)
    print(f"pages={len(pages)}")
    print(f"candidates={len(candidates)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
