from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


CARD_FIELDS = [
    "fine_category",
    "report_id",
    "work_id",
    "source_path",
    "destination",
    "structured_report_id",
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
    "confidence",
    "review_reason",
]

PARAMETER_FIELDS = [
    "fine_category",
    "report_id",
    "structured_report_id",
    "parameter_type",
    "parameter_name",
    "numeric_value",
    "unit",
    "raw_result",
    "requirement",
    "conclusion",
    "item_name",
    "section",
    "source_item_path",
    "source_row",
    "recommended_action",
    "notes",
]

SUMMARY_FIELDS = ["section", "key", "count"]

META_ALIASES = {
    "sample_name": ("样品名称", "sample name", "产品名称", "设备名称"),
    "manufacturer": ("生产单位", "制造商", "manufacturer", "生产厂家", "厂家"),
    "brand": ("商标", "商 标", "brand"),
    "spec_model": ("规格型号", "型号规格", "规格", "型号", "type/model", "model"),
    "production_date": ("生产日期", "出厂日期", "manufacture date"),
    "sample_no": ("样品编号", "sample no", "样品号"),
    "test_type": ("检验类别", "检测类别", "test type"),
    "test_basis": ("检验依据", "检测依据", "test basis", "standard"),
    "test_items": ("检验项目", "检测项目", "test items"),
    "test_date": ("检验日期", "检测日期", "test date"),
    "test_conclusion": ("检验结论", "检测结论", "test conclusion"),
}

PARAMETER_PATTERNS = [
    (
        "thermal_transmittance_u",
        ("传热系数", "u值", "u 值", "u-value", "heat transfer coefficient"),
        "W/(m2·K)",
    ),
    (
        "thermal_resistance_r",
        ("热阻", "传热阻", "thermal resistance"),
        "m2·K/W",
    ),
    (
        "thermal_conductivity_lambda",
        ("导热系数", "thermal conductivity"),
        "W/(m·K)",
    ),
    (
        "shgc_or_shading",
        ("遮阳系数", "太阳得热", "shgc", "得热系数"),
        "",
    ),
    (
        "thickness",
        ("厚度",),
        "mm",
    ),
    (
        "dimension",
        ("尺寸", "长度", "宽度", "高度", "长宽高"),
        "mm",
    ),
    (
        "density",
        ("密度", "表观密度"),
        "kg/m3",
    ),
    (
        "cooling_capacity",
        ("制冷量", "名义制冷量"),
        "W",
    ),
    (
        "heating_capacity",
        ("制热量", "名义制热量"),
        "W",
    ),
    (
        "input_power",
        ("输入功率", "功率"),
        "W",
    ),
    (
        "energy_efficiency",
        ("cop", "eer", "能效", "性能系数"),
        "",
    ),
    (
        "air_volume",
        ("风量", "新风量"),
        "m3/h",
    ),
    (
        "luminous_flux",
        ("光通量",),
        "lm",
    ),
    (
        "lighting_power",
        ("灯功率", "额定功率", "功率"),
        "W",
    ),
]

CATEGORY_PARAMETER_TYPES = {
    "exterior_wall": {
        "thermal_transmittance_u",
        "thermal_resistance_r",
        "thermal_conductivity_lambda",
        "thickness",
        "dimension",
        "density",
    },
    "wall_insulation": {
        "thermal_transmittance_u",
        "thermal_resistance_r",
        "thermal_conductivity_lambda",
    },
    "windows": {
        "thermal_transmittance_u",
        "shgc_or_shading",
        "thickness",
        "dimension",
    },
    "roof": {
        "thermal_transmittance_u",
        "thermal_resistance_r",
        "thermal_conductivity_lambda",
        "thickness",
        "dimension",
        "density",
    },
    "floor": {
        "thermal_transmittance_u",
        "thermal_resistance_r",
        "thermal_conductivity_lambda",
        "thickness",
        "dimension",
        "density",
    },
    "curtain_shading": {
        "shgc_or_shading",
        "thermal_transmittance_u",
        "thickness",
        "dimension",
    },
    "hvac_system": {
        "cooling_capacity",
        "heating_capacity",
        "input_power",
        "energy_efficiency",
        "air_volume",
    },
    "lighting": {
        "lighting_power",
        "luminous_flux",
        "input_power",
        "energy_efficiency",
    },
}

REJECT_DIMENSION_CONTEXT = (
    "经纬密度",
    "断裂伸长率",
    "尺寸稳定性",
    "变化率",
    "断裂",
    "压缩",
    "拉伸",
    "燃烧",
    "吸水",
)

REJECT_DENSITY_CONTEXT = (
    "烟密度",
    "湿流密度",
    "线密度",
    "经纬密度",
    "光密度",
)


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def normalize_path_key(value: str) -> str:
    text = clean(value).replace("/", "\\").lower()
    text = re.sub(r"^[a-z]:\\", "", text)
    return text


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


def field_key(field: str) -> str | None:
    text = clean(field).lower().replace(" ", "")
    for key, aliases in META_ALIASES.items():
        for alias in aliases:
            if alias.lower().replace(" ", "") in text:
                return key
    return None


def first_non_empty(existing: str, value: str) -> str:
    existing = clean(existing)
    value = clean(value)
    return existing or value


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


def parse_number_unit(text: str) -> tuple[str, str]:
    value = clean(text)
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    number = number_match.group(0) if number_match else ""
    unit = ""
    unit_patterns = [
        r"W/\(?m[²2]?[·.]?K\)?",
        r"W/[（(]?m[²2]?[·.]?K[）)]?",
        r"[（(]?m[²2][·.]?K[）)]?/W",
        r"kg/m[³3]",
        r"kg/m[²2]",
        r"g/m[²2]",
        r"m[³3]/h",
        r"m[²2]",
        r"mm",
        r"cm",
        r"lm",
        r"kW",
        r"W",
        r"%",
    ]
    for pattern in unit_patterns:
        match = re.search(pattern, value, re.I)
        if match:
            unit = match.group(0)
            break
    return number, unit


def parameter_match(text: str, fine_category: str) -> tuple[str, str, str]:
    lowered = clean(text).lower()
    allowed = CATEGORY_PARAMETER_TYPES.get(fine_category, set())
    for parameter_type, keywords, default_unit in PARAMETER_PATTERNS:
        if allowed and parameter_type not in allowed:
            continue
        if any(keyword.lower() in lowered for keyword in keywords):
            return parameter_type, next(keyword for keyword in keywords if keyword.lower() in lowered), default_unit
    return "", "", ""


def should_reject_parameter(parameter_type: str, text: str) -> str:
    if "经纬密度" in text:
        return "reject"
    if "断裂伸长率" in text:
        return "reject"
    if parameter_type == "density" and any(word in text for word in REJECT_DENSITY_CONTEXT):
        return "reject"
    if parameter_type in {"dimension", "thickness"} and any(word in text for word in REJECT_DIMENSION_CONTEXT):
        return "needs_check"
    if parameter_type in {"dimension", "thickness"} and ("尺寸偏差" in text or "公差" in text):
        return "needs_check"
    if not parameter_type:
        return "reject"
    return "keep"


def load_fine_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    lookup = {}
    for row in rows:
        for value in (
            row.get("destination", ""),
            row.get("source_path", ""),
            row.get("report_id", ""),
            Path(clean(row.get("destination", ""))).stem,
        ):
            if value:
                lookup[normalize_path_key(value)] = row
                lookup[clean(value)] = row
    return lookup


def load_structured_summary(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def match_fine_row(summary_row: dict[str, str], fine_lookup: dict[str, dict[str, str]]) -> dict[str, str]:
    candidates = [
        summary_row.get("source_path", ""),
        summary_row.get("converted_docx", ""),
        summary_row.get("source_docx", ""),
        summary_row.get("report_id", ""),
        Path(clean(summary_row.get("source_path", ""))).stem,
        Path(clean(summary_row.get("converted_docx", ""))).stem,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        row = fine_lookup.get(normalize_path_key(candidate)) or fine_lookup.get(clean(candidate))
        if row:
            return row
    return {}


def report_dir_for_summary(structured_root: Path, summary_row: dict[str, str]) -> Path:
    category = clean(summary_row.get("organized_category") or summary_row.get("category"))
    report_id = clean(summary_row.get("report_id"))
    return structured_root / category / report_id


def build_report_card(
    summary_row: dict[str, str],
    fine_row: dict[str, str],
    meta_rows: list[dict[str, str]],
) -> dict[str, str]:
    card = {field: "" for field in CARD_FIELDS}
    card.update(
        {
            "fine_category": clean(fine_row.get("fine_category")),
            "report_id": clean(fine_row.get("report_id") or summary_row.get("report_id")),
            "work_id": clean(fine_row.get("work_id")),
            "source_path": clean(fine_row.get("source_path")),
            "destination": clean(fine_row.get("destination") or summary_row.get("source_path")),
            "structured_report_id": clean(summary_row.get("report_id")),
            "confidence": clean(fine_row.get("confidence")),
            "review_reason": clean(fine_row.get("review_reason")),
        }
    )

    for row in meta_rows:
        key = field_key(row.get("field", ""))
        if not key:
            continue
        value = clean(row.get("value"))
        if key in {"test_items", "test_basis", "test_conclusion"}:
            card[key] = append_unique(card[key], value)
        else:
            card[key] = first_non_empty(card[key], value)

    if not card["sample_name"]:
        card["sample_name"] = clean(fine_row.get("sample_name"))
    if not card["spec_model"]:
        card["spec_model"] = clean(fine_row.get("spec_model"))
    if not card["test_items"]:
        card["test_items"] = clean(fine_row.get("test_items"))

    return card


def build_parameter_rows(
    summary_row: dict[str, str],
    fine_row: dict[str, str],
    item_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    fine_category = clean(fine_row.get("fine_category"))
    rows = []
    for item in item_rows:
        item_name = clean(item.get("item_name"))
        requirement = clean(item.get("requirement"))
        result = clean(item.get("result"))
        section = clean(item.get("section"))
        context = " / ".join(part for part in [section, item_name, requirement, result] if part)
        parameter_type, parameter_name, default_unit = parameter_match(context, fine_category)
        action = should_reject_parameter(parameter_type, context)
        if action == "reject":
            continue
        numeric_value, unit = parse_number_unit(result or requirement)
        if not numeric_value and parameter_type not in {"shgc_or_shading", "energy_efficiency"}:
            action = "needs_check"
        rows.append(
            {
                "fine_category": fine_category,
                "report_id": clean(fine_row.get("report_id") or summary_row.get("report_id")),
                "structured_report_id": clean(summary_row.get("report_id")),
                "parameter_type": parameter_type,
                "parameter_name": parameter_name,
                "numeric_value": numeric_value,
                "unit": unit or default_unit,
                "raw_result": result,
                "requirement": requirement,
                "conclusion": clean(item.get("conclusion")),
                "item_name": item_name,
                "section": section,
                "source_item_path": context,
                "source_row": clean(item.get("raw_row")),
                "recommended_action": action,
                "notes": "Focused extraction: keep only identification, dimensions, thermal, HVAC, and lighting performance parameters.",
            }
        )
    return rows


def extract_focused(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    structured_root = Path(args.structured_root)
    summary_rows = load_structured_summary(structured_root / "_summary.csv")
    fine_lookup = load_fine_manifest(Path(args.fine_manifest))

    cards = []
    parameters = []
    for summary_row in summary_rows:
        fine_row = match_fine_row(summary_row, fine_lookup)
        if not fine_row:
            continue
        fine_category = clean(fine_row.get("fine_category"))
        if args.categories and fine_category not in args.categories:
            continue
        report_dir = report_dir_for_summary(structured_root, summary_row)
        meta_rows = read_csv(report_dir / "meta.csv")
        item_rows = read_csv(report_dir / "items.csv")
        cards.append(build_report_card(summary_row, fine_row, meta_rows))
        parameters.extend(build_parameter_rows(summary_row, fine_row, item_rows))
    return cards, parameters


def write_summary(path: Path, cards: list[dict[str, str]], parameters: list[dict[str, str]]) -> None:
    rows = []
    rows.append({"section": "total", "key": "report_cards", "count": str(len(cards))})
    rows.append({"section": "total", "key": "parameters", "count": str(len(parameters))})
    for key, count in Counter(row["fine_category"] for row in cards).most_common():
        rows.append({"section": "cards_by_category", "key": key, "count": str(count)})
    for key, count in Counter(row["fine_category"] for row in parameters).most_common():
        rows.append({"section": "parameters_by_category", "key": key, "count": str(count)})
    for key, count in Counter(row["parameter_type"] for row in parameters).most_common():
        rows.append({"section": "parameter_type", "key": key, "count": str(count)})
    for key, count in Counter(row["recommended_action"] for row in parameters).most_common():
        rows.append({"section": "recommended_action", "key": key, "count": str(count)})
    write_csv(path, rows, SUMMARY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract focused report cards and key parameter candidates from structured reports."
    )
    parser.add_argument(
        "--structured-root",
        default=r"G:\shujuku\output_experiment\useful_structured_full",
    )
    parser.add_argument(
        "--fine-manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "fine_classification"
            / "fine_classification_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "focused_extraction"),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Optional fine categories to include, for example wall_insulation windows.",
    )
    args = parser.parse_args()

    cards, parameters = extract_focused(args)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "focused_report_cards.csv", cards, CARD_FIELDS)
    write_csv(output_dir / "focused_parameter_candidates.csv", parameters, PARAMETER_FIELDS)
    write_summary(output_dir / "focused_extraction_summary.csv", cards, parameters)

    print(f"Report cards: {len(cards)}")
    print(f"Parameter candidates: {len(parameters)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
