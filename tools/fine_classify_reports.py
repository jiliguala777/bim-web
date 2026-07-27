from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


FINE_FIELDS = [
    "fine_category",
    "confidence",
    "review_reason",
    "source_group",
    "source_path",
    "destination",
    "work_id",
    "report_id",
    "content_category",
    "load_relevant",
    "matched_keywords",
    "structured_status",
    "structure_type",
    "sample_name",
    "spec_model",
    "test_items",
    "evidence_text",
]

SUMMARY_FIELDS = ["section", "key", "count"]

CATEGORIES = [
    "exterior_wall",
    "wall_insulation",
    "windows",
    "roof",
    "floor",
    "curtain_shading",
    "hvac_system",
    "lighting",
    "needs_review",
    "not_relevant",
]

CATEGORY_LABELS = {
    "exterior_wall": "外墙",
    "wall_insulation": "外墙外保温",
    "windows": "外窗",
    "roof": "屋面",
    "floor": "地面/楼板",
    "curtain_shading": "窗帘/遮阳",
    "hvac_system": "空调系统",
    "lighting": "照明",
    "needs_review": "待复核",
    "not_relevant": "非目标",
}

SCREENING_GLOBS = [
    "content_screening*.csv",
    "rescreen_unknown.csv",
]

STRUCTURED_FILES = {
    "meta": "meta.csv",
    "items": "items.csv",
}


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def clean_lower(value: object) -> str:
    return clean(value).lower()


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


def normalize_path_key(value: str) -> str:
    text = clean(value).replace("/", "\\").lower()
    text = re.sub(r"^[a-z]:\\", "", text)
    return text


def report_short_id(report_id: str) -> str:
    text = clean(report_id)
    parts = [part for part in text.split("__") if part]
    if len(parts) >= 2:
        return parts[1]
    return text


def load_organized_rows(root: Path) -> list[dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(root.glob("_organized_manifest*.csv")):
        for row in read_csv(path):
            group = clean(row.get("group"))
            if not group.startswith("useful/"):
                continue
            destination = clean(row.get("destination"))
            report_id = clean(row.get("report_id"))
            key = (normalize_path_key(destination), report_id)
            rows[key] = row
    return list(rows.values())


def load_screening_rows(output_root: Path) -> dict[str, dict[str, str]]:
    by_source: dict[str, dict[str, str]] = {}
    by_report: dict[str, dict[str, str]] = {}
    for pattern in SCREENING_GLOBS:
        for path in sorted(output_root.glob(pattern)):
            for row in read_csv(path):
                if row.get("source_path"):
                    by_source[normalize_path_key(row["source_path"])] = row
                if row.get("report_id"):
                    by_report[clean(row["report_id"])] = row
    merged = dict(by_report)
    merged.update(by_source)
    return merged


def load_structured_summary(structured_root: Path) -> dict[str, dict[str, str]]:
    rows = {}
    for row in read_csv(structured_root / "_summary.csv"):
        report_id = clean(row.get("report_id"))
        if report_id:
            rows[report_id] = row
        source_path = clean(row.get("source_path"))
        if source_path:
            rows[normalize_path_key(source_path)] = row
            rows[Path(source_path).stem] = row
        converted_docx = clean(row.get("converted_docx") or row.get("source_docx"))
        if converted_docx:
            rows[normalize_path_key(converted_docx)] = row
            rows[Path(converted_docx).stem] = row
    return rows


def field_matches(field: str, names: tuple[str, ...]) -> bool:
    return any(name in field for name in names)


def load_structured_evidence(structured_root: Path) -> dict[str, dict[str, str]]:
    evidence: dict[str, dict[str, str]] = {}
    for meta_path in structured_root.rglob("meta.csv"):
        report_dir = meta_path.parent
        report_id = report_dir.name
        meta = {
            "sample_name": "",
            "spec_model": "",
            "test_items": "",
            "evidence_text": "",
        }
        chunks: list[str] = []
        for row in read_csv(meta_path):
            field = clean(row.get("field"))
            value = clean(row.get("value"))
            if not value:
                continue
            chunks.append(f"{field} {value}")
            if field_matches(field, ("样品名称", "sample name")) and not meta["sample_name"]:
                meta["sample_name"] = value
            elif field_matches(field, ("规格型号", "type/model", "model")) and not meta["spec_model"]:
                meta["spec_model"] = value
            elif field_matches(field, ("检验项目", "检测项目", "test items")):
                existing = meta["test_items"]
                meta["test_items"] = f"{existing}; {value}".strip("; ")

        items_path = report_dir / STRUCTURED_FILES["items"]
        for row in read_csv(items_path)[:20]:
            chunks.append(
                " ".join(
                    clean(row.get(field))
                    for field in ("item_name", "requirement", "result")
                    if clean(row.get(field))
                )
            )
        meta["evidence_text"] = clean(" ".join(chunks))[:2000]
        evidence[report_id] = meta
    return evidence


KEYWORDS = {
    "curtain_shading": [
        "遮阳系数",
        "太阳得热",
        "shgc",
        "遮阳",
        "得热系数",
        "太阳光直接透射比",
    ],
    "windows": [
        "门窗",
        "外窗",
        "窗",
        "玻璃",
        "中空玻璃",
        "真空绝热",
        "真空玻璃",
        "幕墙",
        "传热系数",
        "可见光透射比",
        "露点",
    ],
    "roof": [
        "屋面",
        "屋顶",
        "种植屋面",
        "倒置式屋面",
        "防水卷材",
        "防水涂料",
    ],
    "floor": [
        "楼板",
        "地面",
        "架空楼板",
        "接触室外空气楼板",
        "地下室顶板",
    ],
    "wall_insulation": [
        "保温",
        "外墙外保温",
        "保温系统",
        "保温装饰",
        "薄抹灰",
        "保温板",
        "岩棉",
        "聚苯",
        "挤塑",
        "聚氨酯",
        "导热系数",
        "热工",
        "热阻",
        "燃烧性能",
    ],
    "exterior_wall": [
        "外墙",
        "墙体",
        "砌块",
        "加气混凝土",
        "烧结",
        "多孔砖",
        "混凝土墙",
        "基层墙",
        "围护",
        "围护结构",
    ],
    "hvac_system": [
        "空调",
        "制冷量",
        "制热量",
        "热泵",
        "cop",
        "eer",
        "风机盘管",
        "新风",
        "冷水机组",
        "多联机",
        "风量",
        "输入功率",
        "能效",
    ],
    "lighting": [
        "照明",
        "灯具",
        "led",
        "光通量",
        "功率密度",
        "灯",
    ],
}

EXCLUDE_FROM_ENVELOPE_FINE = [
    "太阳能集热器",
    "集热器",
    "热水器",
    "空气源热泵",
]

WINDOW_PRIORITY_KEYWORDS = [
    "玻璃",
    "中空玻璃",
    "真空玻璃",
    "真空绝热",
    "门窗",
    "外窗",
    "幕墙",
]


def keyword_score(text: str, keywords: list[str]) -> int:
    lowered = clean_lower(text)
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def classify_row(row: dict[str, str], screening: dict[str, str], structured: dict[str, str], evidence: dict[str, str]) -> dict[str, str]:
    source_group = clean(row.get("group"))
    content_category = clean(row.get("content_category") or screening.get("content_category"))
    matched_keywords = clean(screening.get("matched_keywords"))
    report_id = clean(row.get("report_id"))
    structured_report_id = clean(structured.get("report_id"))
    evidence_row = evidence.get(structured_report_id) or evidence.get(report_id) or {}
    report_hint = " ".join(
        [
            report_id,
            report_short_id(report_id),
            clean(row.get("destination")),
            matched_keywords,
            clean(evidence_row.get("sample_name")),
            clean(evidence_row.get("spec_model")),
            clean(evidence_row.get("test_items")),
            clean(evidence_row.get("evidence_text")),
        ]
    )

    if any(word in report_hint for word in EXCLUDE_FROM_ENVELOPE_FINE):
        fine_category = "not_relevant"
        reason = "excluded_equipment_keyword"
    elif content_category == "hvac" or source_group == "useful/hvac":
        fine_category = "hvac_system"
        reason = "coarse_hvac"
    elif content_category == "lighting" or source_group == "useful/lighting":
        fine_category = "lighting"
        reason = "coarse_lighting"
    elif content_category and content_category not in {"envelope", "unknown"}:
        fine_category = "not_relevant"
        reason = f"coarse_{content_category}"
    else:
        scores = {category: keyword_score(report_hint, words) for category, words in KEYWORDS.items()}
        for lower_priority in ("hvac_system", "lighting"):
            scores.pop(lower_priority, None)

        if scores["curtain_shading"] > 0:
            fine_category = "curtain_shading"
            reason = "shading_keyword"
        elif any(word in report_hint for word in WINDOW_PRIORITY_KEYWORDS):
            fine_category = "windows"
            reason = "window_priority_keyword"
        elif scores["windows"] >= 2 or (
            scores["windows"] >= 1 and scores["wall_insulation"] == 0 and scores["roof"] == 0
        ):
            fine_category = "windows"
            reason = "window_keyword"
        elif scores["roof"] > 0:
            fine_category = "roof"
            reason = "roof_keyword"
        elif scores["floor"] > 0:
            fine_category = "floor"
            reason = "floor_keyword"
        elif scores["wall_insulation"] > 0:
            fine_category = "wall_insulation"
            reason = "insulation_keyword"
        elif scores["exterior_wall"] > 0:
            fine_category = "exterior_wall"
            reason = "wall_keyword"
        else:
            fine_category = "needs_review"
            reason = "no_fine_keyword"

    confidence = "high"
    if fine_category in {"needs_review", "not_relevant"}:
        confidence = "low"
    elif not matched_keywords and not evidence_row:
        confidence = "medium"
        reason += ";filename_or_group_only"
    elif clean(structured.get("structure_type")) == "TABLE_WEAK":
        confidence = "medium"
        reason += ";table_weak"

    return {
        "fine_category": fine_category,
        "confidence": confidence,
        "review_reason": reason,
        "source_group": source_group,
        "source_path": clean(row.get("source_path")),
        "destination": clean(row.get("destination")),
        "work_id": clean(row.get("work_id")),
        "report_id": report_id,
        "content_category": content_category,
        "load_relevant": clean(row.get("load_relevant") or screening.get("load_relevant")),
        "matched_keywords": matched_keywords,
        "structured_status": clean(structured.get("status")),
        "structure_type": clean(structured.get("structure_type")),
        "sample_name": clean(evidence_row.get("sample_name")),
        "spec_model": clean(evidence_row.get("spec_model")),
        "test_items": clean(evidence_row.get("test_items")),
        "evidence_text": clean(evidence_row.get("evidence_text")),
    }


def merge_lookup(
    row: dict[str, str],
    screening_by_key: dict[str, dict[str, str]],
    structured_summary: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    source_key = normalize_path_key(row.get("source_path", ""))
    dest_key = normalize_path_key(row.get("destination", ""))
    report_id = clean(row.get("report_id"))
    screening = (
        screening_by_key.get(source_key)
        or screening_by_key.get(dest_key)
        or screening_by_key.get(report_id)
        or {}
    )
    structured = {}
    for candidate in (
        report_id,
        normalize_path_key(row.get("source_path", "")),
        normalize_path_key(row.get("destination", "")),
        Path(clean(row.get("destination"))).stem,
        normalize_path_key(screening.get("converted_docx", "")),
        Path(clean(screening.get("converted_docx"))).stem,
    ):
        if candidate in structured_summary:
            structured = structured_summary[candidate]
            break
    return screening, structured


def safe_name(value: str) -> str:
    text = re.sub(r'[<>:"/\\|*?]+', "_", clean(value))
    return text or "unknown"


def materialize_links(rows: list[dict[str, str]], output_root: Path, mode: str) -> None:
    for row in rows:
        source = Path(row["destination"] or row["source_path"])
        if not source.exists():
            continue
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
        name = f"{safe_name(row['work_id'])}__{safe_name(row['report_id'])}__{digest}{source.suffix}"
        target = output_root / row["fine_category"] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        if mode == "hardlink":
            os.link(source, target)
        elif mode == "copy":
            import shutil

            shutil.copy2(source, target)
        else:
            raise ValueError(f"Unsupported materialize mode: {mode}")


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> None:
    write_csv(output_dir / "fine_classification_manifest.csv", rows, FINE_FIELDS)
    for category in CATEGORIES:
        category_rows = [row for row in rows if row["fine_category"] == category]
        write_csv(output_dir / f"{category}.csv", category_rows, FINE_FIELDS)

    summary_rows: list[dict[str, str]] = []
    for key, count in Counter(row["fine_category"] for row in rows).most_common():
        summary_rows.append({"section": "fine_category", "key": key, "count": str(count)})
    for key, count in Counter(row["confidence"] for row in rows).most_common():
        summary_rows.append({"section": "confidence", "key": key, "count": str(count)})
    for key, count in Counter(row["source_group"] for row in rows).most_common():
        summary_rows.append({"section": "source_group", "key": key, "count": str(count)})
    for category in CATEGORIES:
        reasons = Counter(
            row["review_reason"]
            for row in rows
            if row["fine_category"] == category
        )
        for key, count in reasons.most_common(12):
            summary_rows.append(
                {"section": f"{category}_reason", "key": key, "count": str(count)}
            )
    write_csv(output_dir / "fine_classification_summary.csv", summary_rows, SUMMARY_FIELDS)

    label_rows = [
        {"section": "label", "key": category, "count": CATEGORY_LABELS[category]}
        for category in CATEGORIES
    ]
    write_csv(output_dir / "fine_category_labels.csv", label_rows, SUMMARY_FIELDS)


def classify_reports(args: argparse.Namespace) -> list[dict[str, str]]:
    organized_root = Path(args.organized_root)
    output_experiment_root = Path(args.output_experiment_root)
    structured_root = Path(args.structured_root)

    organized_rows = load_organized_rows(organized_root)
    screening_by_key = load_screening_rows(output_experiment_root)
    structured_summary = load_structured_summary(structured_root)
    evidence = load_structured_evidence(structured_root)

    classified = []
    for row in organized_rows:
        screening, structured = merge_lookup(row, screening_by_key, structured_summary)
        classified.append(classify_row(row, screening, structured, evidence))

    classified.sort(
        key=lambda row: (
            CATEGORIES.index(row["fine_category"]) if row["fine_category"] in CATEGORIES else 99,
            row["confidence"],
            row["report_id"],
            row["destination"],
        )
    )
    return classified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-classify useful reports for website-aligned structural processing."
    )
    parser.add_argument("--organized-root", default=r"G:\shujuku\organized_reports")
    parser.add_argument("--output-experiment-root", default=r"G:\shujuku\output_experiment")
    parser.add_argument(
        "--structured-root",
        default=r"G:\shujuku\output_experiment\useful_structured_full",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "fine_classification"),
    )
    parser.add_argument("--materialize-root", default="")
    parser.add_argument("--materialize-mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()

    rows = classify_reports(args)
    output_dir = Path(args.output_dir)
    write_outputs(rows, output_dir)
    if args.materialize_root:
        materialize_links(rows, Path(args.materialize_root), args.materialize_mode)

    counts = Counter(row["fine_category"] for row in rows)
    print(f"Classified reports: {len(rows)}")
    for category in CATEGORIES:
        print(f"{category}: {counts.get(category, 0)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
