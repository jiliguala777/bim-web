from __future__ import annotations

import csv
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external_sources" / "raw"
MANIFEST = ROOT / "data" / "external_sources" / "manifests" / "material_source_search_2026-07-07_round2.csv"


SEED_URLS = [
    "https://zfjsw.changde.gov.cn/zhdt/tzgg/content_78152907",
    "https://zfjsw.changde.gov.cn/zhdt/tzgg/content_78166005",
    "https://zfjsw.changde.gov.cn/zhdt/tzgg/content_78177942",
    "https://zfjsw.changde.gov.cn/zhdt/tzgg/content_78189601",
    "https://zfjsw.changde.gov.cn/zhdt/tzgg/content_78200083",
    "https://zfjsw.changde.gov.cn/upload/zfjsw/contentmanage/article/file/2024/04/17/2024%E5%B9%B4%E5%B8%B8%E5%BE%B7%E5%B8%82%E5%BB%BA%E7%AD%91%E8%8A%82%E8%83%BD%E6%9D%90%E6%96%99%EF%BC%88%E4%BA%A7%E5%93%81%EF%BC%89%E5%85%AC%E7%A4%BA%E5%90%8D%E5%8D%95%281%29.xls",
    "https://zfjsw.changde.gov.cn/upload/zfjsw/contentmanage/article/file/2023/01/16/%E9%99%84%E4%BB%B6%EF%BC%9A2022%E5%B9%B4%E7%AC%AC%E5%9B%9B%E6%89%B9%E5%B8%B8%E5%BE%B7%E5%B8%82%E5%BB%BA%E7%AD%91%E8%8A%82%E8%83%BD%E6%9D%90%E6%96%99%EF%BC%88%E4%BA%A7%E5%93%81%EF%BC%89%E5%85%AC%E7%A4%BA%E5%90%8D%E5%8D%95.xls",
    "https://jsj.yueyang.gov.cn/54185/54187/54225/content_2162411.html",
    "https://jsj.yueyang.gov.cn/54027/54029/54058/54073/54183/content_2140156.html",
    "https://jsj.yueyang.gov.cn/54185/54186/54204/content_2255655.html",
    "https://jsj.yueyang.gov.cn/54185/54187/54225/content_2282851.html",
    "https://yueyang.gov.cn/web/2570/2585/2931/2932/content_2297407.html",
    "https://jsj.yueyang.gov.cn/54185/54186/54204/content_2309462.html",
    "https://jsj.yueyang.gov.cn/54185/54186/54204/content_2332458.html",
    "https://jsj.yueyang.gov.cn/uploadfiles/202601/2026012815312335808.pdf",
    "https://jsj.yueyang.gov.cn/54167/index_1.htm",
    "https://www.hengyang.gov.cn/xxgk/dtxx/tzgg/gsgg/20260424/i3893209.html",
    "https://www.hengyang.gov.cn/zjw/xxgk/gzdt/tzgg/20220113/i2588097.html",
    "https://zj.shaoyang.gov.cn/syzj/tzgg/202306/c312f0823e4f480db559cf04728e8d18.shtml",
]


@dataclass
class SourceRow:
    title: str
    url: str
    local_path: str
    source_type: str
    material_granularity: str
    thermal_value_status: str
    target_databases: str
    priority: str
    notes: str
    review_status: str
    verify_ssl: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        attr_dict = dict(attrs)
        if tag_name == "a":
            href = attr_dict.get("href")
        elif tag_name in {"img", "iframe"}:
            href = attr_dict.get("src")
            if href:
                self.links.append((tag_name, href))
                return
        else:
            href = None
        if href:
            self._href = href
            self._text = [tag_name]

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


def detect_encoding(raw: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    if match:
        return match.group(1)
    if re.search(rb"charset=[\"']?\s*gb", raw[:4096], re.I):
        return "gb18030"
    for enc in ("utf-8", "gb18030"):
        try:
            raw.decode(enc)
            return enc
        except Exception:
            pass
    return "utf-8"


def infer_ext(url: str, content_type: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for ext in (".xls", ".xlsx", ".pdf", ".doc", ".docx", ".csv", ".txt", ".jpg", ".jpeg", ".png"):
        if path.endswith(ext):
            return ext
    if "pdf" in content_type:
        return ".pdf"
    if "jpeg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "excel" in content_type or "spreadsheet" in content_type:
        return ".xls"
    return ".html"


def slug_from_url(url: str, ext: str) -> str:
    parsed = urllib.parse.urlparse(url)
    stem = Path(urllib.parse.unquote(parsed.path)).stem
    stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", stem).strip("-")
    if not stem:
        stem = re.sub(r"[^\w-]+", "-", parsed.netloc + parsed.path).strip("-")
    return stem[:90] + ext


def fetch(url: str) -> tuple[bytes, str]:
    parsed = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(parsed.path, safe="/%")
    safe_query = urllib.parse.quote(parsed.query, safe="=&?/%")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, safe_query, parsed.fragment))
    request = urllib.request.Request(safe_url, headers={"User-Agent": "Mozilla/5.0 bim-web-material-source-collector/0.1"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=45) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")
    return raw, content_type


def save_url(url: str, *, prefix: str = "") -> tuple[Path, str, bytes]:
    raw, content_type = fetch(url)
    ext = infer_ext(url, content_type)
    name = slug_from_url(url, ext)
    if prefix:
        name = f"{prefix}-{name}"
    out = RAW / name
    counter = 1
    while out.exists() and out.read_bytes() != raw:
        out = RAW / f"{Path(name).stem}-{counter}{ext}"
        counter += 1
    out.write_bytes(raw)
    return out, content_type, raw


def page_title(text: str) -> str:
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", text, re.I | re.S)
    if h1:
        return clean(re.sub(r"<[^>]+>", " ", h1.group(1)))
    title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    if title:
        return clean(re.sub(r"<[^>]+>", " ", title.group(1)))
    for line in text.splitlines():
        if "建筑节能" in line and ("材料" in line or "产品" in line):
            return clean(re.sub(r"<[^>]+>", " ", line))
    return ""


def is_attachment_link(url: str, text: str) -> bool:
    lower = urllib.parse.urlparse(url).path.lower()
    if lower.endswith((".jpg", ".jpeg", ".png")):
        return "uploadfiles" in lower
    if lower.endswith((".xls", ".xlsx", ".pdf", ".doc", ".docx")):
        return True
    return "附件" in text and ("公示名单" in text or "建筑节能" in text)


def source_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".xls", ".xlsx", ".csv"}:
        return "official_product_list_spreadsheet"
    if ext == ".pdf":
        return "official_product_list_pdf"
    if ext in {".jpg", ".jpeg", ".png"}:
        return "official_product_list_image"
    if ext in {".doc", ".docx"}:
        return "official_product_list_doc"
    return "official_product_notice_html"


def collect() -> list[SourceRow]:
    RAW.mkdir(parents=True, exist_ok=True)
    rows: list[SourceRow] = []
    seen: set[str] = set()
    for seed in SEED_URLS:
        if seed in seen:
            continue
        seen.add(seed)
        try:
            local, content_type, raw = save_url(seed, prefix="material-list")
        except Exception as exc:
            rows.append(
                SourceRow(
                    title="FETCH_FAILED",
                    url=seed,
                    local_path="",
                    source_type="unknown",
                    material_granularity="",
                    thermal_value_status="",
                    target_databases="",
                    priority="low",
                    notes=str(exc),
                    review_status="fetch_failed",
                    verify_ssl="false",
                )
            )
            continue
        encoding = detect_encoding(raw, content_type)
        text = raw.decode(encoding, errors="ignore") if local.suffix.lower() == ".html" else ""
        title = page_title(text) or clean(Path(local).stem)
        rows.append(
            SourceRow(
                title=title,
                url=seed,
                local_path=str(local.relative_to(ROOT)).replace("\\", "/"),
                source_type=source_type_for_path(local),
                material_granularity="具体材料/产品",
                thermal_value_status="待解析；优先查是否含导热系数/传热系数，否则作为产品索引",
                target_databases="windows;wall_insulation;exterior_wall;roof;floor",
                priority="high" if any(k in title for k in ("湘潭", "岳阳", "上海")) else "medium",
                notes="round2 material-list search seed",
                review_status="downloaded",
                verify_ssl="false",
            )
        )
        if local.suffix.lower() != ".html":
            continue
        parser = LinkParser()
        parser.feed(text)
        for link_text, href in parser.links:
            absolute = urllib.parse.urljoin(seed, href)
            if absolute in seen or not is_attachment_link(absolute, link_text):
                continue
            seen.add(absolute)
            try:
                attach_local, _, _ = save_url(absolute, prefix="material-list")
            except Exception as exc:
                rows.append(
                    SourceRow(
                        title=f"ATTACHMENT_FAILED: {clean(link_text)}",
                        url=absolute,
                        local_path="",
                        source_type="unknown",
                        material_granularity="具体材料/产品",
                        thermal_value_status="",
                        target_databases="",
                        priority="low",
                        notes=str(exc),
                        review_status="fetch_failed",
                        verify_ssl="false",
                    )
                )
                continue
            rows.append(
                SourceRow(
                    title=clean(link_text) or attach_local.stem,
                    url=absolute,
                    local_path=str(attach_local.relative_to(ROOT)).replace("\\", "/"),
                    source_type=source_type_for_path(attach_local),
                    material_granularity="具体材料/产品",
                    thermal_value_status="待解析；通常为产品清单，需反查型式检验报告补参数",
                    target_databases="windows;wall_insulation;exterior_wall;roof;floor",
                    priority="medium",
                    notes=f"attachment from {title}",
                    review_status="downloaded",
                    verify_ssl="false",
                )
            )
    return rows


def write_manifest(rows: list[SourceRow]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SourceRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    rows = collect()
    write_manifest(rows)
    print(f"sources={len(rows)}")
    print(f"downloaded={sum(1 for row in rows if row.review_status == 'downloaded')}")
    print(f"failed={sum(1 for row in rows if row.review_status == 'fetch_failed')}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
