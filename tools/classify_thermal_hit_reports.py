from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CARDS = BASE_DIR / "data" / "envelope_thermal_hit_excel_exports_full" / "wall_insulation_report_cards.csv"
DEFAULT_ITEMS = BASE_DIR / "data" / "envelope_thermal_hit_excel_exports_full" / "wall_insulation_items.csv"
DEFAULT_PARAMS = BASE_DIR / "data" / "envelope_thermal_full" / "thermal_parameters.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "report_processing_audit" / "envelope_thermal_hit_fine_classification.csv"
DEFAULT_SUMMARY = BASE_DIR / "data" / "report_processing_audit" / "envelope_thermal_hit_fine_classification_summary.csv"


FIELDS = [
    "report_id",
    "fine_category",
    "target_database",
    "target_table",
    "confidence",
    "review_reason",
    "matched_keywords",
    "sample_name",
    "spec_model",
    "test_items",
    "source_doc",
    "source_path",
]


CATEGORY_DATABASES = {
    "wall_insulation": ("wall_insulation.db", "wall_insulation_systems"),
    "exterior_wall": ("exterior_wall.db", "parameter_options"),
    "window_u": ("windows.db", "parameter_options"),
    "door_u": ("door.db", "parameter_options"),
    "roof_u": ("roof.db", "parameter_options"),
    "floor_u": ("floor.db", "parameter_options"),
    "floor_contact_type": ("floor_contact.db", "parameter_options"),
    "window_shgc": ("windows.db", "parameter_options"),
    "window_air_tightness": ("window_air_tightness.db", "parameter_options"),
    "curtain_shading": ("shading.db", "parameter_options"),
    "needs_review": ("", ""),
}


KEYWORDS = {
    "curtain_shading": ["遮阳系数", "遮阳性能", "遮阳", "shading"],
    "window_shgc": ["太阳得热系数", "太阳能总透射比", "太阳光直接透射比", "shgc", "g值", "g 值"],
    "window_air_tightness": ["气密性能", "气密性", "空气渗透", "单位缝长"],
    "window_u": ["外窗", "门窗", "窗", "玻璃", "中空玻璃", "真空玻璃", "幕墙", "型材", "铝合金窗", "塑料窗", "暖边", "边框"],
    "door_u": ["外门", "入户门", "户门", "保温门", "防火门", "钢质门", "推拉门", "铝合金门", "木门", "折叠门"],
    "roof_u": ["屋面", "屋顶", "顶板", "顶框", "种植屋面", "防水卷材"],
    "floor_u": ["楼板", "地面", "底板", "底框", "地暖板", "架空楼板", "地下室顶板", "楼地面", "架空采暖", "垫层砂浆", "速装板"],
    "wall_insulation": [
        "外墙外保温",
        "墙体保温",
        "保温系统",
        "保温装饰板",
        "保温板",
        "岩棉",
        "xps",
        "eps",
        "聚苯",
        "石墨",
        "真金板",
        "泡沫玻璃",
        "无机保温",
        "复合保温",
        "硬泡聚氨酯",
        "真空绝热",
        "绝热板",
        "玻璃棉",
        "保温砂浆",
        "保温浆料",
        "保温材料",
        "保温隔热",
        "挤塑",
        "挤塑板",
        "发泡水泥",
        "发泡水泥板",
        "聚异氰脲酸酯",
        "pir",
        "酚醛",
        "苯板",
        "夹芯板",
        "夹心板",
        "复合板",
        "内保温",
        "聚氨酯泡沫",
        "聚氨酯",
        "聚氨酯绝热",
        "喷涂聚氨酯",
        "泡沫塑料",
        "碳纳芯板",
        "agpf",
        "玻化微珠",
        "硅酸钙",
        "水泥板",
        "纤维水泥",
        "纤维增强",
        "粉刷石膏",
        "轻集料混凝土",
        "保温强力板",
        "保温装饰",
        "装饰一体化",
        "装饰板",
        "防火板",
        "防火分隔条",
        "外模板",
        "外贴板",
        "隔热",
        "气凝胶",
        "橡塑",
        "保温管",
        "保温材",
        "纳米凝胶",
        "泡棉",
        "聚醚海绵",
        "垫块",
        "锚固件",
        "托架",
        "门芯",
        "相变材料",
        "铝覆板",
        "微晶板",
        "晶板",
        "漂珠",
        "板材",
    ],
    "exterior_wall": ["外墙", "墙体", "墙板", "砌块", "加气混凝土", "混凝土墙", "预制墙", "蒸压"],
}


PRIORITY = [
    "curtain_shading",
    "window_shgc",
    "window_air_tightness",
    "window_u",
    "door_u",
    "roof_u",
    "floor_u",
    "wall_insulation",
    "exterior_wall",
]


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


def normalize_for_match(text: str) -> str:
    return clean(text).lower()


def add_weighted_hits(scores: Counter[str], matches: dict[str, list[str]], text: str, weight: int) -> None:
    normalized = normalize_for_match(text)
    if not normalized:
        return
    for category, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in normalized:
                scores[category] += weight
                if keyword not in matches[category]:
                    matches[category].append(keyword)


def classify(card: dict[str, str], item_text: str, param_text: str) -> tuple[str, str, str, str]:
    scores: Counter[str] = Counter()
    matches: dict[str, list[str]] = defaultdict(list)

    high_value_text = " ".join(
        clean(card.get(key))
        for key in ["sample_name", "spec_model", "test_items", "test_conclusion"]
    )
    add_weighted_hits(scores, matches, high_value_text, 5)
    add_weighted_hits(scores, matches, item_text, 2)
    add_weighted_hits(scores, matches, param_text, 3)

    # "门窗" reports are usually window system reports, not door reports.
    combined = normalize_for_match(f"{high_value_text} {item_text} {param_text}")
    if any(keyword in combined for keyword in ["系统储能密度", "系统发热量", "储能密度", "可再生能源"]):
        return "needs_review", "low", "not_envelope_parameter", ""
    if any(keyword in combined for keyword in ["动车", "车门", "司机室", "塞拉门", "车组", "地板布"]):
        return "needs_review", "low", "non_building_product", ""

    if any(keyword in combined for keyword in ["导光管采光系统", "平板采光系统", "采光系统"]):
        matches["window_u"].append("采光系统")
        return "window_u", "high", "daylighting_system_u_value", "window_u:采光系统"

    if "门窗" in combined and scores["door_u"] <= scores["window_u"]:
        scores["door_u"] = 0
        matches.pop("door_u", None)

    thermal_window = any(keyword in combined for keyword in ["保温性能", "传热系数", "热阻", "u-value", "u值"])
    if thermal_window and scores["window_u"] > 0:
        scores["window_u"] += 5
        if scores["window_air_tightness"] > 0:
            scores["window_air_tightness"] = 0
            matches.pop("window_air_tightness", None)
        if scores["curtain_shading"] > 0 and scores["window_shgc"] == 0:
            scores["curtain_shading"] = 0
            matches.pop("curtain_shading", None)

    if thermal_window and scores["door_u"] > 0:
        scores["door_u"] += 5
        if scores["window_air_tightness"] > 0:
            scores["window_air_tightness"] = 0
            matches.pop("window_air_tightness", None)
        if scores["curtain_shading"] > 0:
            scores["curtain_shading"] = 0
            matches.pop("curtain_shading", None)

    shading_product = any(keyword in combined for keyword in ["蜂巢", "百叶", "遮阳帘", "窗帘", "遮阳板"])
    if scores["curtain_shading"] > 0 and not shading_product:
        scores["curtain_shading"] = 0
        matches.pop("curtain_shading", None)

    insulation_system = any(
        keyword in combined
        for keyword in ["外墙外保温", "保温系统", "保温装饰板", "保温板", "真空绝热", "绝热板", "硬泡聚氨酯"]
    )
    if insulation_system and scores["wall_insulation"] > 0:
        scores["wall_insulation"] += 8
        if scores["exterior_wall"] > 0:
            scores["exterior_wall"] = min(scores["exterior_wall"], scores["wall_insulation"] - 3)

    if "传热系数" in combined or "u-value" in combined or "u值" in combined:
        if scores["window_u"] > 0:
            scores["window_u"] += 2
        if scores["roof_u"] > 0:
            scores["roof_u"] += 2
        if scores["floor_u"] > 0:
            scores["floor_u"] += 2
        if scores["exterior_wall"] > 0:
            scores["exterior_wall"] += 2

    best = ""
    best_score = 0
    for category in PRIORITY:
        if scores[category] > best_score:
            best = category
            best_score = scores[category]

    if not best:
        return "needs_review", "low", "no_category_keyword", ""

    sorted_scores = scores.most_common()
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    if best_score >= 8 and best_score >= second_score + 2:
        confidence = "high"
        reason = "keyword_score_high"
    elif best_score >= 5:
        confidence = "medium"
        reason = "keyword_score_medium"
    else:
        confidence = "low"
        reason = "weak_keyword_match"

    if second_score and best_score - second_score <= 1:
        confidence = "low"
        reason = "ambiguous_close_scores"

    matched = "; ".join(f"{cat}:{'|'.join(words)}" for cat, words in matches.items() if words)
    return best, confidence, reason, matched


def build_rows(cards_path: Path, items_path: Path, params_path: Path) -> list[dict[str, str]]:
    cards = read_csv(cards_path)

    item_text_by_report: defaultdict[str, list[str]] = defaultdict(list)
    for row in read_csv(items_path):
        report_id = clean(row.get("report_id"))
        if not report_id:
            continue
        item_text_by_report[report_id].append(
            " ".join(clean(row.get(key)) for key in ["section", "item_name", "requirement", "result", "conclusion"])
        )

    param_text_by_report: defaultdict[str, list[str]] = defaultdict(list)
    for row in read_csv(params_path):
        report_id = clean(row.get("report_id"))
        if not report_id:
            continue
        param_text_by_report[report_id].append(
            " ".join(clean(row.get(key)) for key in ["parameter_type", "evidence", "raw_value", "unit"])
        )

    rows: list[dict[str, str]] = []
    for card in cards:
        report_id = clean(card.get("report_id"))
        fine_category, confidence, reason, matched = classify(
            card,
            " ".join(item_text_by_report.get(report_id, [])),
            " ".join(param_text_by_report.get(report_id, [])),
        )
        db_name, table_name = CATEGORY_DATABASES[fine_category]
        rows.append(
            {
                "report_id": report_id,
                "fine_category": fine_category,
                "target_database": db_name,
                "target_table": table_name,
                "confidence": confidence,
                "review_reason": reason,
                "matched_keywords": matched,
                "sample_name": clean(card.get("sample_name")),
                "spec_model": clean(card.get("spec_model")),
                "test_items": clean(card.get("test_items")),
                "source_doc": clean(card.get("source_doc")),
                "source_path": clean(card.get("source_path")),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary: list[dict[str, str]] = []
    for category, count in Counter(row["fine_category"] for row in rows).most_common():
        summary.append({"section": "fine_category", "key": category, "count": str(count)})
    for confidence, count in Counter(row["confidence"] for row in rows).most_common():
        summary.append({"section": "confidence", "key": confidence, "count": str(count)})
    write_csv(path, summary, ["section", "key", "count"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify thermal-hit envelope reports into web envelope categories.")
    parser.add_argument("--cards", default=str(DEFAULT_CARDS))
    parser.add_argument("--items", default=str(DEFAULT_ITEMS))
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    rows = build_rows(Path(args.cards), Path(args.items), Path(args.params))
    write_csv(Path(args.output), rows, FIELDS)
    write_summary(Path(args.summary), rows)
    print(f"classified={len(rows)}")
    print(f"output={args.output}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
