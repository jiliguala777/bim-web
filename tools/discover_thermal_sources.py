from __future__ import annotations

import argparse
import base64
import csv
import html
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path


TRUSTED_DOMAIN_SUFFIXES = (
    ".gov.cn",
    "gov.cn",
)

DEFAULT_QUERIES = [
    "建筑门窗幕墙热工参数目录 filetype:pdf",
    "典型门窗幕墙热工参数目录 filetype:pdf",
    "建筑材料热物理性能指标计算参数目录 filetype:pdf",
    "建筑节能材料 公示 热传递性能",
    "屋面 传热系数 热阻 导热系数 filetype:pdf",
    "楼板 传热系数 热阻 导热系数 filetype:pdf",
    "外门 传热系数 filetype:pdf",
]

MANIFEST_FIELDS = [
    "title",
    "url",
    "local_path",
    "source_type",
    "component_hint",
    "parameter_hint",
    "confidence",
    "notes",
    "review_status",
    "verify_ssl",
]


@dataclass
class DiscoveryResult:
    title: str
    url: str
    local_path: str
    source_type: str
    component_hint: str
    parameter_hint: str
    confidence: int
    notes: str
    review_status: str = "pending"
    verify_ssl: str = "true"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
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
    return " ".join(str(value or "").strip().split())


def is_trusted_url(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    host = host.lower()
    return any(host == suffix or host.endswith(suffix) for suffix in TRUSTED_DOMAIN_SUFFIXES)


def infer_source_type(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith((".doc", ".docx")):
        return "docx"
    if path.endswith((".xls", ".xlsx", ".csv")):
        return "spreadsheet"
    return "html"


def score_source(title: str, url: str, snippet: str = "") -> DiscoveryResult:
    title = clean(html.unescape(title)) or clean(url)
    url = normalize_result_url(url)
    snippet = clean(html.unescape(snippet))
    text = f"{title} {url} {snippet}"

    if not is_trusted_url(url):
        return DiscoveryResult(
            title=title,
            url=url,
            local_path="",
            source_type=infer_source_type(url),
            component_hint="",
            parameter_hint="",
            confidence=0,
            notes="untrusted domain",
            review_status="reject",
        )

    confidence = 20
    notes: list[str] = ["trusted domain"]
    parameter_hints: list[str] = []
    component_hints: list[str] = []

    term_scores = {
        "传热系数": ("thermal_transmittance_u", 20),
        "热传递性能": ("thermal_transmittance_u", 20),
        "导热系数": ("thermal_conductivity_lambda", 20),
        "热阻": ("thermal_resistance_r", 20),
        "SHGC": ("solar_heat_gain_shgc", 20),
        "太阳得热系数": ("solar_heat_gain_shgc", 20),
        "Ug": ("thermal_transmittance_u", 12),
        "U值": ("thermal_transmittance_u", 12),
    }
    for term, (hint, score) in term_scores.items():
        if term.lower() in text.lower():
            confidence += score
            if hint not in parameter_hints:
                parameter_hints.append(hint)
            notes.append(f"matched {term}")

    component_terms = {
        "门窗": "window",
        "幕墙": "window",
        "玻璃": "window",
        "屋面": "roof",
        "楼板": "floor",
        "外门": "door",
        "外墙": "exterior_wall",
        "保温": "wall_insulation",
    }
    for term, hint in component_terms.items():
        if term in text and hint not in component_hints:
            component_hints.append(hint)
            confidence += 8

    if "热工参数目录" in text or "参数目录" in text:
        confidence += 18
        notes.append("matched catalog")
    if infer_source_type(url) == "pdf":
        confidence += 10
        notes.append("pdf")

    confidence = min(confidence, 100)
    review_status = "pending" if confidence >= 40 and parameter_hints else "needs_check"
    return DiscoveryResult(
        title=title,
        url=url,
        local_path="",
        source_type=infer_source_type(url),
        component_hint=";".join(component_hints),
        parameter_hint=";".join(parameter_hints),
        confidence=confidence,
        notes="; ".join(notes),
        review_status=review_status,
    )


def normalize_result_url(url: str) -> str:
    url = html.unescape(clean(url))
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("bing.com") and parsed.path == "/ck/a":
        params = urllib.parse.parse_qs(parsed.query)
        for key in ("u", "url"):
            if key in params and params[key]:
                value = urllib.parse.unquote(params[key][0])
                decoded = decode_bing_u_param(value)
                return decoded or value
    return url


def decode_bing_u_param(value: str) -> str:
    if not value.startswith("a1"):
        return ""
    payload = value[2:]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8")
    except Exception:
        return ""
    if decoded.startswith(("http://", "https://")):
        return decoded
    return ""


def discover_from_search_html(search_html: str) -> list[DiscoveryResult]:
    parser = LinkParser()
    parser.feed(search_html)
    results: list[DiscoveryResult] = []
    seen: set[str] = set()
    plain_text = clean(re.sub(r"<[^>]+>", " ", search_html))
    for title, url in parser.links:
        normalized = normalize_result_url(url)
        if not normalized.startswith(("http://", "https://")):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        scored = score_source(title, normalized, plain_text[:1000])
        if scored.review_status != "reject":
            results.append(scored)
    return sorted(results, key=lambda item: item.confidence, reverse=True)


def discover_from_seed_html(seed_html: str, *, seed_url: str) -> list[DiscoveryResult]:
    parser = LinkParser()
    parser.feed(seed_html)
    results: list[DiscoveryResult] = []
    seen: set[str] = set()
    for title, url in parser.links:
        absolute_url = urllib.parse.urljoin(seed_url, normalize_result_url(url))
        if not absolute_url.startswith(("http://", "https://")):
            continue
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        scored = score_source(title, absolute_url, "")
        if scored.review_status != "reject" and _is_relevant_seed_link(scored):
            results.append(scored)
    return sorted(results, key=lambda item: item.confidence, reverse=True)


def _is_relevant_seed_link(result: DiscoveryResult) -> bool:
    if result.source_type in {"pdf", "docx", "spreadsheet"} and result.confidence >= 40:
        return True
    if result.parameter_hint and result.confidence >= 60:
        return True
    return False


def fetch_search_html(query: str) -> str:
    search_url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
    request = urllib.request.Request(
        search_url,
        headers={"User-Agent": "Mozilla/5.0 thermal-source-discovery/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="ignore")


def fetch_url_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 thermal-source-discovery/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")
    encoding = "utf-8"
    charset_match = re.search(r"charset=([\w-]+)", content_type, re.I)
    if charset_match:
        encoding = charset_match.group(1)
    elif re.search(rb"<meta[^>]+charset=[\"']?\s*gb", raw[:4096], re.I):
        encoding = "gb18030"
    return raw.decode(encoding, errors="ignore")


def discover_sources(queries: list[str], seed_urls: list[str] | None = None) -> list[DiscoveryResult]:
    results: list[DiscoveryResult] = []
    seen: set[str] = set()
    for seed_url in seed_urls or []:
        try:
            seed_html = fetch_url_text(seed_url)
        except Exception as exc:
            results.append(
                DiscoveryResult(
                    title=f"SEED_FAILED: {seed_url}",
                    url=seed_url,
                    local_path="",
                    source_type="html",
                    component_hint="",
                    parameter_hint="",
                    confidence=0,
                    notes=str(exc),
                    review_status="needs_check",
                )
            )
            continue
        for result in discover_from_seed_html(seed_html, seed_url=seed_url):
            if result.url in seen:
                continue
            seen.add(result.url)
            results.append(result)
    for query in queries:
        try:
            search_html = fetch_search_html(query)
        except Exception as exc:
            results.append(
                DiscoveryResult(
                    title=f"SEARCH_FAILED: {query}",
                    url="",
                    local_path="",
                    source_type="",
                    component_hint="",
                    parameter_hint="",
                    confidence=0,
                    notes=str(exc),
                    review_status="needs_check",
                )
            )
            continue
        for result in discover_from_search_html(search_html):
            if result.url in seen:
                continue
            seen.add(result.url)
            results.append(result)
    return sorted(results, key=lambda item: item.confidence, reverse=True)


def write_source_manifest(results: list[DiscoveryResult], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover trusted thermal parameter source candidates.")
    parser.add_argument("--query", action="append", help="Search query. Can be passed multiple times.")
    parser.add_argument("--seed-url", action="append", help="Trusted page URL whose links should be scored.")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "external_sources" / "manifests" / "discovered_thermal_sources.csv"),
    )
    args = parser.parse_args()

    queries = args.query or ([] if args.seed_url else DEFAULT_QUERIES)
    results = discover_sources(queries, seed_urls=args.seed_url or [])
    write_source_manifest(results, args.output)
    print(f"queries={len(queries)}")
    print(f"sources={len(results)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
