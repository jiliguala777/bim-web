from __future__ import annotations

import argparse
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    from .thermal_candidate_extractor import (
        ThermalCandidate,
        collect_candidates_from_file,
        detect_html_encoding,
        write_candidates_csv,
    )
except ImportError:
    from thermal_candidate_extractor import (
        ThermalCandidate,
        collect_candidates_from_file,
        detect_html_encoding,
        write_candidates_csv,
    )


DEFAULT_PAGE_URLS = [
    "https://zfcxjw.cq.gov.cn/zwxx_166/gsgg/202307/t20230703_12117419.html",
]
RAW_DIR = Path("data") / "external_sources" / "raw"


@dataclass
class ThermalAttachment:
    title: str
    url: str


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def fetch_bytes(url: str, *, verify_ssl: bool = True) -> tuple[bytes, str]:
    context = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 bim-web-chongqing-collector/0.1"})
    with urllib.request.urlopen(request, context=context, timeout=30) as response:
        return response.read(), response.headers.get("content-type", "")


def fetch_text(url: str, *, verify_ssl: bool = True) -> str:
    raw, content_type = fetch_bytes(url, verify_ssl=verify_ssl)
    return raw.decode(detect_html_encoding(raw, content_type), errors="ignore")


def discover_thermal_attachments(html: str, *, page_url: str) -> list[ThermalAttachment]:
    attachments: list[ThermalAttachment] = []
    seen: set[str] = set()
    for match in re.finditer(r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html, re.I | re.S):
        href = match.group(1)
        title = clean(re.sub(r"<[^>]+>", "", match.group(2)))
        absolute_url = urllib.parse.urljoin(page_url, href)
        if absolute_url in seen:
            continue
        if not is_relevant_attachment(title, absolute_url):
            continue
        seen.add(absolute_url)
        attachments.append(ThermalAttachment(title=title, url=absolute_url))
    return attachments


def is_relevant_attachment(title: str, url: str) -> bool:
    lower_url = url.lower()
    text = f"{title} {url}"
    if not lower_url.endswith(".pdf"):
        return False
    return (
        "热工参数目录" in text
        or "门窗幕墙" in text
        or "传热系数" in text
        or "太阳得热系数" in text
    )


def safe_filename_from_url(url: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    return name or "attachment.pdf"


def download_attachment(attachment: ThermalAttachment, *, verify_ssl: bool = True) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / safe_filename_from_url(attachment.url)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    raw, _ = fetch_bytes(attachment.url, verify_ssl=verify_ssl)
    output_path.write_bytes(raw)
    return output_path


def collect_chongqing_candidates(
    *,
    page_urls: list[str] | None = None,
    verify_ssl: bool = True,
) -> tuple[list[ThermalAttachment], list[ThermalCandidate]]:
    attachments: list[ThermalAttachment] = []
    candidates: list[ThermalCandidate] = []
    seen: set[str] = set()
    for page_url in page_urls or DEFAULT_PAGE_URLS:
        html = fetch_text(page_url, verify_ssl=verify_ssl)
        for attachment in discover_thermal_attachments(html, page_url=page_url):
            if attachment.url in seen:
                continue
            seen.add(attachment.url)
            attachments.append(attachment)
            local_path = download_attachment(attachment, verify_ssl=verify_ssl)
            candidates.extend(
                collect_candidates_from_file(
                    local_path,
                    source_url=attachment.url,
                    source_title=attachment.title,
                )
            )
    return attachments, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Chongqing thermal PDF attachments into review candidates.")
    parser.add_argument("--page-url", action="append", help="Chongqing publication page URL. Can be repeated.")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "external_sources" / "review" / "chongqing_thermal_parameter_candidates.csv"),
    )
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    attachments, candidates = collect_chongqing_candidates(
        page_urls=args.page_url,
        verify_ssl=not args.insecure,
    )
    write_candidates_csv(candidates, args.output)
    print(f"attachments={len(attachments)}")
    print(f"candidates={len(candidates)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
