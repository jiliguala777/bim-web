from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURED_ROOT = BASE_DIR / "data" / "wall_insulation_structured_full"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "wall_insulation_excel_exports"

CARD_FIELDS = [
    "report_id",
    "status",
    "work_id",
    "source_doc",
    "converted_docx",
    "sample_name",
    "manufacturer",
    "brand",
    "spec_model",
    "production_date",
    "sample_no",
    "test_type",
    "test_basis",
    "test_items",
    "test_date",
    "test_conclusion",
    "structure_type",
    "table_count",
    "item_rows",
]

ITEM_EXPORT_FIELDS = [
    "report_id",
    "status",
    "work_id",
    "source_doc",
    "table_index",
    "row_index",
    "section",
    "seq",
    "item_name",
    "requirement",
    "result",
    "conclusion",
    "raw_row",
]

TRACE_EXPORT_FIELDS = [
    "report_id",
    "status",
    "work_id",
    "source_doc",
    "sequence",
    "block_type",
    "table_index",
    "row_index",
    "col_index",
    "text",
    "row_text",
]

META_ALIASES = {
    "sample_name": ("样品名称", "sample name", "产品名称"),
    "manufacturer": ("生产单位", "manufacturer", "生产厂家", "厂家"),
    "brand": ("商标", "商 标", "brand"),
    "spec_model": ("规格型号", "型号规格", "规格", "型号", "type/model", "model"),
    "production_date": ("生产日期", "出厂日期"),
    "sample_no": ("样品编号", "sample no", "样品号"),
    "test_type": ("检验类别", "检测类别", "test type"),
    "test_basis": ("检验依据", "检测依据", "test basis", "standard"),
    "test_items": ("检验项目", "检测项目", "test items"),
    "test_date": ("检验日期", "检测日期", "test date"),
    "test_conclusion": ("检验结论", "检测结论", "test conclusion"),
}


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def meta_key(field: str) -> str | None:
    text = clean(field).lower().replace(" ", "")
    for key, aliases in META_ALIASES.items():
        for alias in aliases:
            if alias.lower().replace(" ", "") in text:
                return key
    return None


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


def summary_by_report(root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(root / "_summary.csv")
    return {clean(row.get("report_id")): row for row in rows if clean(row.get("report_id"))}


def build_report_cards(root: Path, summary_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for report_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        report_id = report_dir.name
        summary = summary_rows.get(report_id, {})
        card = {field: "" for field in CARD_FIELDS}
        card.update(
            {
                "report_id": report_id,
                "status": clean(summary.get("status")),
                "work_id": clean(summary.get("work_id")),
                "source_doc": clean(summary.get("source_doc")),
                "converted_docx": clean(summary.get("converted_docx")),
                "structure_type": clean(summary.get("structure_type")),
                "table_count": clean(summary.get("table_count")),
                "item_rows": clean(summary.get("item_rows")),
            }
        )
        for row in read_csv(report_dir / "meta.csv"):
            key = meta_key(row.get("field", ""))
            if not key:
                continue
            value = clean(row.get("value"))
            if key in {"test_items", "test_basis", "test_conclusion"}:
                card[key] = append_unique(card[key], value)
            elif not card[key]:
                card[key] = value
        cards.append(card)
    return cards


def build_items(root: Path, summary_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for report_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = summary_rows.get(report_dir.name, {})
        prefix = {
            "report_id": report_dir.name,
            "status": clean(summary.get("status")),
            "work_id": clean(summary.get("work_id")),
            "source_doc": clean(summary.get("source_doc")),
        }
        for row in read_csv(report_dir / "items.csv"):
            rows.append({**prefix, **row, "report_id": report_dir.name})
    return rows


def build_trace(root: Path, summary_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for report_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = summary_rows.get(report_dir.name, {})
        prefix = {
            "report_id": report_dir.name,
            "status": clean(summary.get("status")),
            "work_id": clean(summary.get("work_id")),
            "source_doc": clean(summary.get("source_doc")),
        }
        for row in read_csv(report_dir / "trace.csv"):
            rows.append({**prefix, **row, "report_id": report_dir.name})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export structured wall-insulation report folders into three flat CSV tables for Excel conversion.")
    parser.add_argument("--structured-root", default=str(DEFAULT_STRUCTURED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    root = Path(args.structured_root)
    output_dir = Path(args.output_dir)
    summary = summary_by_report(root)

    cards = build_report_cards(root, summary)
    items = build_items(root, summary)
    trace = build_trace(root, summary)

    write_csv(output_dir / "wall_insulation_report_cards.csv", cards, CARD_FIELDS)
    write_csv(output_dir / "wall_insulation_items.csv", items, ITEM_EXPORT_FIELDS)
    write_csv(output_dir / "wall_insulation_trace.csv", trace, TRACE_EXPORT_FIELDS)

    print(f"report_cards={len(cards)}")
    print(f"items={len(items)}")
    print(f"trace={len(trace)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
