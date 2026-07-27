from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "fine_classification" / "windows.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "fine_classification" / "windows_refined.csv"
DEFAULT_GLASS_OUTPUT = BASE_DIR / "data" / "fine_classification" / "windows_glass_priority.csv"
DEFAULT_PARAMETER_OUTPUT = BASE_DIR / "data" / "fine_classification" / "windows_parameter_priority.csv"

KEEP_KEYWORDS = (
    "门窗",
    "外窗",
    "窗",
    "玻璃",
    "中空玻璃",
    "真空玻璃",
    "真空绝热",
    "幕墙",
    "可见光透射比",
    "露点",
    "遮阳",
    "shgc",
    "传热系数",
)

DROP_KEYWORDS = (
    "外墙外保温",
    "保温系统",
    "保温装饰",
    "薄抹灰",
    "岩棉",
    "聚苯",
    "挤塑",
    "聚氨酯",
    "拉伸粘结",
    "粘结强度",
    "锚固",
    "抗冲击",
    "燃烧性能",
    "玻璃纤维",
    "网布",
    "玻璃棉",
    "风管",
    "甲醛",
    "TVOC",
    "耐碱",
    "集热器",
    "热水器",
    "集热管",
    "PVT",
    "导光管采光系统",
    "平板采光系统",
    "司机室门板",
    "悬挑节点",
    "动车车门",
    "车门",
)

PARAMETER_PRIORITY_KEYWORDS = (
    ("传热系数", 100),
    ("U值", 100),
    ("U 值", 100),
    ("u-value", 100),
    ("Low-E", 80),
    ("中空玻璃", 80),
    ("真空玻璃", 80),
    ("遮阳系数", 70),
    ("太阳得热系数", 70),
    ("SHGC", 70),
    ("可见光透射比", 60),
    ("太阳能总透射比", 60),
    ("幕墙", 40),
    ("门窗", 40),
    ("外窗", 40),
)


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evidence(row: dict[str, str]) -> str:
    return clean(
        " ".join(
            clean(row.get(field))
            for field in (
                "matched_keywords",
                "sample_name",
                "spec_model",
                "test_items",
                "evidence_text",
                "source_path",
                "destination",
            )
        )
    )


def is_refined_window(row: dict[str, str]) -> tuple[bool, str]:
    text = evidence(row)
    keep_hits = [keyword for keyword in KEEP_KEYWORDS if keyword and keyword in text]
    drop_hits = [keyword for keyword in DROP_KEYWORDS if keyword and keyword in text]
    if not keep_hits:
        return False, "no_window_keyword"
    if drop_hits and not any(keyword in text for keyword in ("中空玻璃", "真空玻璃", "幕墙", "门窗", "外窗")):
        return False, "insulation_only_keyword"
    if drop_hits and any(
        keyword in text
        for keyword in (
            "保温系统",
            "外墙外保温",
            "薄抹灰",
            "玻璃纤维",
            "玻璃棉",
            "网布",
            "风管",
            "集热器",
            "热水器",
            "集热管",
            "PVT",
            "导光管采光系统",
            "平板采光系统",
            "司机室门板",
            "悬挑节点",
            "动车车门",
            "车门",
        )
    ):
        return False, "wall_insulation_evidence"
    return True, "refined_window_keyword"


def priority_score(row: dict[str, str]) -> tuple[int, str]:
    text = evidence(row)
    hits = [keyword for keyword, _ in PARAMETER_PRIORITY_KEYWORDS if keyword.lower() in text.lower()]
    score = sum(weight for keyword, weight in PARAMETER_PRIORITY_KEYWORDS if keyword.lower() in text.lower())
    return score, ";".join(hits)


def run(args: argparse.Namespace) -> None:
    rows = read_csv(Path(args.input))
    fields = list(rows[0].keys()) if rows else []
    if "refine_reason" not in fields:
        fields.append("refine_reason")
    for field in ("priority_score", "priority_reason"):
        if field not in fields:
            fields.append(field)

    kept: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        keep, reason = is_refined_window(row)
        row = dict(row)
        row["refine_reason"] = reason
        score, score_reason = priority_score(row)
        row["priority_score"] = str(score)
        row["priority_reason"] = score_reason
        if keep:
            kept.append(row)
        else:
            rejected.append(row)

    write_csv(Path(args.output), kept, fields)
    glass_priority = [
        row
        for row in kept
        if any(
            keyword in evidence(row)
            for keyword in (
                "玻璃",
                "中空玻璃",
                "真空玻璃",
                "可见光透射比",
                "太阳能总透射比",
                "遮阳系数",
                "太阳得热系数",
            )
        )
    ]
    write_csv(Path(args.glass_output), glass_priority, fields)
    parameter_priority = sorted(
        [row for row in kept if int(row.get("priority_score") or 0) > 0],
        key=lambda row: (-int(row.get("priority_score") or 0), clean(row.get("report_id"))),
    )
    write_csv(Path(args.parameter_output), parameter_priority, fields)
    if args.rejected:
        write_csv(Path(args.rejected), rejected, fields)
    print(
        f"input={len(rows)} kept={len(kept)} glass_priority={len(glass_priority)} "
        f"parameter_priority={len(parameter_priority)} rejected={len(rejected)} output={args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine the window manifest by removing obvious wall-insulation false positives.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--glass-output", default=str(DEFAULT_GLASS_OUTPUT))
    parser.add_argument("--parameter-output", default=str(DEFAULT_PARAMETER_OUTPUT))
    parser.add_argument("--rejected", default=str(BASE_DIR / "data" / "fine_classification" / "windows_rejected.csv"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
