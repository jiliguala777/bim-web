from __future__ import annotations

import argparse
import csv
import io
import re
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path


CSV_FIELDS = [
    "component_category",
    "parameter_type",
    "name",
    "value",
    "unit",
    "source_title",
    "source_url",
    "source_doc",
    "source_report_id",
    "source_text",
    "notes",
    "review_status",
]


@dataclass
class ThermalCandidate:
    component_category: str
    parameter_type: str
    name: str
    value: float
    unit: str
    source_title: str
    source_url: str
    source_doc: str = ""
    source_report_id: str = ""
    source_text: str = ""
    notes: str = ""
    review_status: str = "pending"


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(clean("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None and self._current_table is not None:
            if any(self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def normalize_unit(unit: str) -> str:
    text = clean(unit)
    text = text.replace("㎡", "m2").replace("m²", "m2")
    text = text.replace("•", "·").replace(".", "·")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    if "W/(m2" in text or "W/m2" in text:
        return "W/(m2·K)"
    if "W/(m·K)" in text or "W/m·K" in text:
        return "W/(m·K)"
    if re.search(r"W/\([^)]*m[^)]*K\)", text, re.I):
        return "W/(m·K)"
    if re.search(r"W/\([^)]*K\)", text, re.I):
        return "W/(m2·K)"
    if "m2·K/W" in text or "(m2·K)/W" in text:
        return "m2·K/W"
    return text


def classify_parameter(unit: str, context: str) -> str:
    normalized = normalize_unit(unit)
    lowered = context.lower()
    if "shgc" in lowered or "太阳得热系数" in context:
        return "solar_heat_gain_shgc"
    if normalized == "W/(m·K)" or "导热系数" in context or "λ" in context:
        return "thermal_conductivity_lambda"
    if normalized == "m2·K/W" or "热阻" in context or re.search(r"\bR\s*=", context, re.I):
        return "thermal_resistance_r"
    if normalized == "W/(m2·K)" or "传热系数" in context or "热传递性能" in context:
        return "thermal_transmittance_u"
    return ""


def classify_component(context: str, parameter_type: str) -> str:
    text = clean(context)
    if "太阳得热系数" in text or "SHGC" in text.upper():
        return "window_shgc"
    if "气密" in text:
        return "window_air_tightness"
    if "窗" in text or "幕墙" in text or "玻璃" in text:
        return "window_u"
    if "门" in text:
        return "door_u"
    if "屋面" in text or "屋顶" in text:
        return "roof_u"
    if "地面" in text or "土壤" in text or "架空" in text:
        return "floor_contact_type"
    if "楼板" in text:
        return "floor_u"
    if "外保温" in text or "墙体保温" in text or "保温系统" in text:
        return "wall_insulation"
    if "外墙" in text or "墙" in text:
        return "exterior_wall"
    if parameter_type == "thermal_conductivity_lambda":
        return "wall_insulation"
    return "exterior_wall"


def parse_value_unit(text: str) -> tuple[float | None, str]:
    value_text = clean(text)
    unit_patterns = [
        r"W\s*/\s*\(?\s*(?:㎡|m2|m²)\s*[·.・]?\s*K\s*\)?",
        r"W\s*/\s*\(?\s*m\s*[·.・]\s*K\s*\)?",
        r"W\s*/\s*\([^)]*K\)",
        r"\(?\s*(?:㎡|m2|m²)\s*[·.・]\s*K\s*\)?\s*/\s*W",
    ]
    for pattern in unit_patterns:
        match = re.search(pattern, value_text, re.I)
        if not match:
            continue
        prefix = value_text[: match.start()]
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", prefix)
        if numbers:
            return float(numbers[-1]), normalize_unit(match.group(0))
    return None, ""


def _candidate_from_parts(
    *,
    name: str,
    context: str,
    source_url: str,
    source_title: str,
    source_doc: str = "",
    source_report_id: str = "",
) -> ThermalCandidate | None:
    value, unit = parse_value_unit(context)
    if value is None:
        shgc_match = re.search(r"(?:SHGC|Shgc|太阳得热系数)\s*[=:：]?\s*(\d+(?:\.\d+)?)", context)
        if shgc_match:
            value = float(shgc_match.group(1))
            unit = ""
        else:
            return None
    parameter_type = classify_parameter(unit, context)
    if not parameter_type:
        return None
    return ThermalCandidate(
        component_category=classify_component(context, parameter_type),
        parameter_type=parameter_type,
        name=clean(name) or _fallback_name(context, parameter_type),
        value=value,
        unit=unit,
        source_title=source_title,
        source_url=source_url,
        source_doc=source_doc,
        source_report_id=clean(source_report_id),
        source_text=clean(context),
    )


def _fallback_name(context: str, parameter_type: str) -> str:
    labels = {
        "thermal_transmittance_u": "传热系数",
        "thermal_conductivity_lambda": "导热系数",
        "thermal_resistance_r": "热阻",
        "solar_heat_gain_shgc": "太阳得热系数",
    }
    component = classify_component(context, parameter_type)
    return f"{component} {labels.get(parameter_type, '热工参数')}"


def extract_candidates_from_html(
    html: str,
    *,
    source_url: str,
    source_title: str,
    source_doc: str = "",
) -> list[ThermalCandidate]:
    parser = TableParser()
    parser.feed(html)
    candidates: list[ThermalCandidate] = []
    for table in parser.tables:
        if not table:
            continue
        headers = table[0]
        for row in table[1:]:
            row_text = " ".join(row)
            candidate = _candidate_from_parts(
                name=_cell_by_header(headers, row, ("材料", "产品", "名称")) or _fallback_table_name(row),
                context=row_text,
                source_url=source_url,
                source_title=source_title,
                source_doc=source_doc,
                source_report_id=_cell_by_header(headers, row, ("报告", "编号")) or _fallback_table_report_id(row),
            )
            if candidate is not None:
                candidates.append(candidate)
    return dedupe_candidates(candidates)


def _fallback_table_name(row: list[str]) -> str:
    if len(row) > 3:
        return row[3]
    return ""


def _fallback_table_report_id(row: list[str]) -> str:
    if len(row) > 6:
        return row[6]
    for cell in row:
        if re.search(r"\b[A-Z]{2,}[A-Z0-9-]{4,}\b", cell):
            return cell
    return ""


def _cell_by_header(headers: list[str], row: list[str], keywords: tuple[str, ...]) -> str:
    for index, header in enumerate(headers):
        if index < len(row) and all(keyword in header for keyword in keywords):
            return row[index]
    return ""


def extract_candidates_from_text(
    text: str,
    *,
    source_url: str,
    source_title: str,
    source_doc: str = "",
) -> list[ThermalCandidate]:
    specialized = extract_chongqing_window_catalog_candidates(
        text,
        source_url=source_url,
        source_title=source_title,
        source_doc=source_doc,
    )
    if specialized:
        return specialized

    candidates: list[ThermalCandidate] = []
    for line in re.split(r"[\r\n。；;]+", text):
        context = clean(line)
        if not context:
            continue
        candidate = _candidate_from_parts(
            name="",
            context=context,
            source_url=source_url,
            source_title=source_title,
            source_doc=source_doc,
        )
        if candidate is not None:
            candidates.append(candidate)
    return dedupe_candidates(candidates)


def extract_chongqing_window_catalog_candidates(
    text: str,
    *,
    source_url: str,
    source_title: str,
    source_doc: str = "",
) -> list[ThermalCandidate]:
    if "典型门窗幕墙热工参数目录" not in text and "典型玻璃的光学" not in text:
        return []

    rows: list[ThermalCandidate] = []
    lines = [clean(line) for line in text.splitlines() if clean(line)]
    for line in lines:
        rows.extend(
            _extract_chongqing_window_u_row(
                line,
                source_url=source_url,
                source_title=source_title,
                source_doc=source_doc,
            )
        )
        rows.extend(
            _extract_chongqing_glass_optical_row(
                line,
                source_url=source_url,
                source_title=source_title,
                source_doc=source_doc,
            )
        )
    return dedupe_candidates(rows)


def _extract_chongqing_window_u_row(
    line: str,
    *,
    source_url: str,
    source_title: str,
    source_doc: str,
) -> list[ThermalCandidate]:
    match = re.match(
        r"^(?:\d+\s+)?(?P<name>.+?)\s+(?P<ug>\d+\.\d+)\s+"
        r"(?P<l1>\d+\.\d+)\s+(?P<l2>\d+\.\d+)\s+(?P<l3>\d+\.\d+)\s+"
        r"(?P<l4>\d+\.\d+)\s+(?P<l5>\d+\.\d+)\s*$",
        line,
    )
    if not match:
        return []
    name = _clean_chongqing_glass_name(match.group("name"))
    values = [
        ("玻璃Ug", float(match.group("ug"))),
        ("典型门窗幕墙1.0级", float(match.group("l1"))),
        ("典型门窗幕墙2.0级", float(match.group("l2"))),
        ("典型门窗幕墙3.0级", float(match.group("l3"))),
        ("典型门窗幕墙4.0级", float(match.group("l4"))),
        ("典型门窗幕墙5.0级", float(match.group("l5"))),
    ]
    return [
        ThermalCandidate(
            component_category="window_u",
            parameter_type="thermal_transmittance_u",
            name=f"{name} - {label}",
            value=value,
            unit="W/(m2·K)",
            source_title=source_title,
            source_url=source_url,
            source_doc=source_doc,
            source_text=line,
            notes="重庆门窗幕墙热工参数目录表格抽取；典型门窗幕墙等级值需审核适用范围。",
        )
        for label, value in values
    ]


def _extract_chongqing_glass_optical_row(
    line: str,
    *,
    source_url: str,
    source_title: str,
    source_doc: str,
) -> list[ThermalCandidate]:
    match = re.match(
        r"^\d+\s+(?P<name>.+?)\s+"
        r"(?P<visible>\d+\.\d+)\s+(?P<shgc>\d+\.\d+)\s+[—-]\s+"
        r"(?P<ug>\d+\.\d+)(?:\s|$)",
        line,
    )
    if not match:
        return []
    name = _clean_chongqing_glass_name(match.group("name"))
    return [
        ThermalCandidate(
            component_category="window_shgc",
            parameter_type="solar_heat_gain_shgc",
            name=f"{name} - 玻璃SHGC",
            value=float(match.group("shgc")),
            unit="",
            source_title=source_title,
            source_url=source_url,
            source_doc=source_doc,
            source_text=line,
            notes="重庆典型玻璃光学热工性能参数表抽取。",
        ),
        ThermalCandidate(
            component_category="window_u",
            parameter_type="thermal_transmittance_u",
            name=f"{name} - 玻璃Ug",
            value=float(match.group("ug")),
            unit="W/(m2·K)",
            source_title=source_title,
            source_url=source_url,
            source_doc=source_doc,
            source_text=line,
            notes="重庆典型玻璃光学热工性能参数表抽取。",
        ),
    ]


def _clean_chongqing_glass_name(name: str) -> str:
    text = clean(name)
    prefixes = [
        "两玻一腔 中空玻璃 普通 ",
        "三玻两腔 中空玻璃 普通 ",
        "热致调光 单腔中空 玻璃 普通 ",
        "热致调光 两腔中空 玻璃 双单银 ",
        "单银 ",
        "双银 ",
        "三银 ",
        "透明 ",
        "普通 ",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text


def dedupe_candidates(candidates: list[ThermalCandidate]) -> list[ThermalCandidate]:
    seen: set[tuple[str, str, str, float, str, str]] = set()
    result: list[ThermalCandidate] = []
    for candidate in candidates:
        key = (
            candidate.component_category,
            candidate.parameter_type,
            candidate.name,
            candidate.value,
            candidate.unit,
            candidate.source_url,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def write_candidates_csv(candidates: list[ThermalCandidate], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def collect_candidates_from_file(
    input_path: Path | str,
    *,
    source_url: str = "",
    source_title: str = "",
) -> list[ThermalCandidate]:
    path = Path(input_path)
    title = source_title or path.stem
    url = source_url or str(path)
    if path.suffix.lower() == ".pdf":
        content = extract_text_from_pdf_bytes(path.read_bytes())
        return extract_candidates_from_text(
            content,
            source_url=url,
            source_title=title,
            source_doc=path.name,
        )
    content = path.read_text(encoding="utf-8-sig", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        return extract_candidates_from_html(
            content,
            source_url=url,
            source_title=title,
            source_doc=path.name,
        )
    return extract_candidates_from_text(
        content,
        source_url=url,
        source_title=title,
        source_doc=path.name,
    )


def collect_candidates_from_url(
    url: str,
    *,
    source_title: str = "",
    verify_ssl: bool = True,
) -> list[ThermalCandidate]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bim-web-thermal-candidate-collector/0.1"},
    )
    with urllib.request.urlopen(
        request,
        timeout=30,
        context=ssl_context_for_verify(verify_ssl),
    ) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")
    if "pdf" in content_type.lower() or url.lower().split("?", 1)[0].endswith(".pdf"):
        text = extract_text_from_pdf_bytes(raw)
        return extract_candidates_from_text(
            text,
            source_url=url,
            source_title=source_title or url,
            source_doc=Path(url.split("?", 1)[0]).name,
        )
    encoding = detect_html_encoding(raw, content_type)
    text = raw.decode(encoding, errors="ignore")
    if "html" in content_type.lower() or "<table" in text.lower():
        return extract_candidates_from_html(text, source_url=url, source_title=source_title or url)
    return extract_candidates_from_text(text, source_url=url, source_title=source_title or url)


def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires pypdf. Run with the bundled Codex Python or install pypdf."
        ) from exc
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def collect_candidates_from_manifest(manifest_path: Path | str) -> list[ThermalCandidate]:
    path = Path(manifest_path)
    rows: list[ThermalCandidate] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for source in csv.DictReader(f):
            title = clean(source.get("title"))
            url = clean(source.get("url"))
            local_path = clean(source.get("local_path"))
            verify_ssl = clean(source.get("verify_ssl")).lower() not in {"false", "0", "no"}
            if local_path:
                rows.extend(
                    collect_candidates_from_file(
                        local_path,
                        source_url=url or local_path,
                        source_title=title,
                    )
                )
            elif url:
                rows.extend(
                    collect_candidates_from_url(
                        url,
                        source_title=title,
                        verify_ssl=verify_ssl,
                    )
                )
    return dedupe_candidates(rows)


def detect_html_encoding(raw: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([\w-]+)", content_type, re.I)
    if charset_match:
        return normalize_encoding_name(charset_match.group(1))
    head = raw[:4096].decode("ascii", errors="ignore")
    meta_match = re.search(r"<meta[^>]+charset=[\"']?\s*([\w-]+)", head, re.I)
    if meta_match:
        return normalize_encoding_name(meta_match.group(1))
    return "utf-8"


def normalize_encoding_name(encoding: str) -> str:
    normalized = encoding.strip().lower()
    if normalized in {"gb2312", "gbk", "gb18030"}:
        return "gb18030"
    return normalized or "utf-8"


def ssl_context_for_verify(verify_ssl: bool) -> ssl.SSLContext:
    if verify_ssl:
        return ssl.create_default_context()
    return ssl._create_unverified_context()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect thermal parameter candidates into a review CSV without modifying SQLite databases."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Public HTML/text URL to collect.")
    source.add_argument("--input", help="Local HTML/text file to collect.")
    source.add_argument("--manifest", help="CSV source manifest for batch collection.")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "external_sources" / "review" / "thermal_parameter_candidates.csv"),
    )
    parser.add_argument("--title", default="", help="Source title stored with each candidate.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for public sites with broken certificate chains.",
    )
    args = parser.parse_args()

    if args.manifest:
        candidates = collect_candidates_from_manifest(args.manifest)
    elif args.url:
        candidates = collect_candidates_from_url(
            args.url,
            source_title=args.title,
            verify_ssl=not args.insecure,
        )
    else:
        candidates = collect_candidates_from_file(args.input, source_title=args.title)
    write_candidates_csv(candidates, args.output)
    print(f"candidates={len(candidates)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
