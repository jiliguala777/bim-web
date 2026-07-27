from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURED_ROOT = BASE_DIR / "data" / "windows_refined_structured_full"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "windows_parameter_extraction"

CARD_FIELDS = [
    "report_id",
    "status",
    "is_window_candidate",
    "reject_reason",
    "sample_name",
    "manufacturer",
    "brand",
    "spec_model",
    "sample_no",
    "test_basis",
    "test_items",
    "test_conclusion",
]

PARAMETER_FIELDS = [
    "report_id",
    "sample_name",
    "parameter_type",
    "parameter_label",
    "value",
    "unit",
    "source_field",
    "source_text",
]

EXCLUDED_CARD_FIELDS = CARD_FIELDS + ["source_doc"]

META_ALIASES = {
    "sample_name": ("样品名称", "产品名称", "sample name"),
    "manufacturer": ("生产单位", "生产厂家", "厂家", "manufacturer"),
    "brand": ("商标", "商 标", "brand"),
    "spec_model": ("规格型号", "型号规格", "规格", "型号", "type/model", "model"),
    "sample_no": ("样品编号", "样品号", "sample no"),
    "test_basis": ("检验依据", "检测依据", "test basis", "test standard", "standard"),
    "test_items": ("检验项目", "检测项目", "test items"),
    "test_conclusion": ("检验结论", "检测结论", "test conclusion"),
}

PARAMETER_KEYWORDS = [
    ("window_u", "传热系数", ("传热系数", "u值", "u 值", "u-value", "thermal transmittance"), "W/m²·K"),
    ("window_shgc", "太阳得热系数", ("太阳得热系数", "太阳能总透射比", "遮阳系数", "shgc", "solar heat gain"), ""),
    ("visible_transmittance", "可见光透射比", ("可见光透射比", "visible light transmittance"), ""),
]

WINDOW_EVIDENCE_KEYWORDS = (
    "门窗",
    "外窗",
    "窗",
    "幕墙",
    "中空玻璃",
    "真空玻璃",
    "Low-E",
    "玻璃钢型材",
    "可见光透射比",
    "遮阳系数",
    "太阳得热系数",
    "SHGC",
    "AAMA 1503",
)

NON_WINDOW_KEYWORDS = (
    "外墙外保温",
    "保温系统",
    "保温装饰",
    "薄抹灰",
    "岩棉",
    "聚苯",
    "挤塑",
    "聚氨酯",
    "玻璃纤维",
    "玻纤",
    "网布",
    "玻璃棉",
    "风管",
    "甲醛",
    "TVOC",
    "集热器",
    "热水器",
    "集热管",
    "PVT组件",
    "真空太阳集热管",
    "真空管型太阳能",
    "导光管采光系统",
    "平板采光系统",
    "司机室门板",
    "悬挑节点",
    "动车车门",
    "车门",
)


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_unique(existing: str, value: str) -> str:
    existing = clean(existing)
    value = clean(value)
    if not value:
        return existing
    if not existing:
        return value
    parts = [part.strip() for part in existing.split(";") if part.strip()]
    if value not in parts:
        parts.append(value)
    return "; ".join(parts)


def meta_key(field: str) -> str | None:
    text = clean(field).lower().replace(" ", "")
    for key, aliases in META_ALIASES.items():
        if any(alias.lower().replace(" ", "") in text for alias in aliases):
            return key
    return None


def source_report_id(report_dir: Path) -> str:
    return re.sub(r"__[0-9a-f]{10,12}$", "", report_dir.name, flags=re.I)


def build_card(report_dir: Path, summary: dict[str, str]) -> dict[str, str]:
    card = {field: "" for field in CARD_FIELDS}
    card["report_id"] = report_dir.name
    card["status"] = clean(summary.get("status"))
    for row in read_csv(report_dir / "meta.csv"):
        key = meta_key(row.get("field", ""))
        if not key:
            continue
        value = clean(row.get("value"))
        if key in {"test_basis", "test_items", "test_conclusion"}:
            card[key] = append_unique(card[key], value)
        elif not card[key]:
            card[key] = value
    keep, reason = is_window_candidate(card)
    card["is_window_candidate"] = "yes" if keep else "no"
    card["reject_reason"] = reason
    return card


def card_evidence(card: dict[str, str]) -> str:
    return clean(
        " ".join(
            clean(card.get(field))
            for field in (
                "sample_name",
                "manufacturer",
                "brand",
                "spec_model",
                "test_basis",
                "test_items",
                "test_conclusion",
            )
        )
    )


def is_window_candidate(card: dict[str, str]) -> tuple[bool, str]:
    text = card_evidence(card)
    lowered = text.lower()
    for keyword in NON_WINDOW_KEYWORDS:
        if keyword.lower() in lowered:
            return False, f"non_window_keyword:{keyword}"
    if "型材" in text and "传热系数" in text:
        return True, "window_profile_u_value"
    if any(keyword.lower() in lowered for keyword in WINDOW_EVIDENCE_KEYWORDS):
        return True, "window_evidence"
    return False, "no_structured_window_evidence"


def number_after_keyword(text: str, keyword: str) -> float | None:
    start = text.lower().find(keyword.lower())
    fragment = text[start:] if start >= 0 else text
    match = re.search(r"(?<![\d.])(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?![\d.])", fragment)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parameter_value_is_plausible(parameter_type: str, value: float) -> bool:
    if parameter_type == "window_u":
        return 0.1 <= value <= 10.0
    if parameter_type in {"window_shgc", "visible_transmittance"}:
        return 0.0 <= value <= 1.5
    return True


def classify_parameter(text: str) -> tuple[str, str, float, str] | None:
    lowered = clean(text).lower()
    for parameter_type, label, keywords, default_unit in PARAMETER_KEYWORDS:
        for keyword in keywords:
            if keyword.lower() not in lowered:
                continue
            value = number_after_keyword(text, keyword)
            if value is None:
                continue
            if not parameter_value_is_plausible(parameter_type, value):
                continue
            return parameter_type, label, value, default_unit
    return None


def parameter_rows_from_meta(report_dir: Path, card: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(report_dir / "meta.csv"):
        key = meta_key(row.get("field", "")) or clean(row.get("field"))
        if key not in {"test_conclusion", "test_items"}:
            continue
        value = clean(row.get("value"))
        # Test item rows usually only say what was tested. Keep them only when
        # they contain an actual number near a target parameter.
        parameter = classify_parameter(value)
        if not parameter:
            continue
        parameter_type, label, number, unit = parameter
        rows.append(
            {
                "report_id": report_dir.name,
                "sample_name": card.get("sample_name") or source_report_id(report_dir),
                "parameter_type": parameter_type,
                "parameter_label": label,
                "value": number,
                "unit": unit,
                "source_field": key,
                "source_text": value,
            }
        )
    return rows


def parameter_rows_from_items(report_dir: Path, card: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(report_dir / "items.csv"):
        context = " ".join(clean(row.get(field)) for field in ("item_name", "result", "raw_row"))
        parameter = classify_parameter(context)
        if not parameter:
            continue
        parameter_type, label, number, unit = parameter
        rows.append(
            {
                "report_id": report_dir.name,
                "sample_name": card.get("sample_name") or source_report_id(report_dir),
                "parameter_type": parameter_type,
                "parameter_label": label,
                "value": number,
                "unit": unit,
                "source_field": "items",
                "source_text": context,
            }
        )
    return rows


def summary_by_report(root: Path) -> dict[str, dict[str, str]]:
    return {
        clean(row.get("report_id")): row
        for row in read_csv(root / "_summary.csv")
        if clean(row.get("report_id"))
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.structured_root)
    output_dir = Path(args.output_dir)
    summary = summary_by_report(root)
    cards: list[dict[str, object]] = []
    excluded_cards: list[dict[str, object]] = []
    parameters: list[dict[str, object]] = []
    seen: set[tuple[str, str, float, str]] = set()

    for report_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        card = build_card(report_dir, summary.get(report_dir.name, {}))
        cards.append(card)
        if card["is_window_candidate"] != "yes":
            excluded = dict(card)
            excluded["source_doc"] = clean(summary.get(report_dir.name, {}).get("source_doc"))
            excluded_cards.append(excluded)
            continue
        for row in parameter_rows_from_items(report_dir, card) + parameter_rows_from_meta(report_dir, card):
            key = (
                clean(row.get("report_id")),
                clean(row.get("parameter_type")),
                float(row.get("value")),
                clean(row.get("source_text")),
            )
            if key in seen:
                continue
            seen.add(key)
            parameters.append(row)

    write_csv(output_dir / "window_report_cards.csv", cards, CARD_FIELDS)
    write_csv(output_dir / "window_excluded_report_cards.csv", excluded_cards, EXCLUDED_CARD_FIELDS)
    write_csv(output_dir / "window_parameters.csv", parameters, PARAMETER_FIELDS)
    print(f"report_cards={len(cards)}")
    print(f"excluded_report_cards={len(excluded_cards)}")
    print(f"parameters={len(parameters)}")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract key window parameters from structured report folders.")
    parser.add_argument("--structured-root", default=str(DEFAULT_STRUCTURED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
