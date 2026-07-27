from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data" / "external_sources" / "review"
RAW = ROOT / "data" / "external_sources" / "raw"

OUT_PARAMS = REVIEW / "screened_material_parameters_2026-07-07.csv"
OUT_FOLLOWUP = REVIEW / "material_product_followup_index_2026-07-07.csv"


PARAM_COLUMNS = [
    "screen_status",
    "target_database",
    "component_category",
    "material_name",
    "parameter_type",
    "value",
    "unit",
    "density_kg_m3",
    "specific_heat_j_kgk",
    "heat_storage_w_m2k",
    "wall_correction_factor",
    "roof_correction_factor",
    "floor_correction_factor",
    "source_title",
    "source_url",
    "local_path",
    "source_note",
]


FOLLOWUP_COLUMNS = [
    "screen_status",
    "component_category",
    "product_name",
    "standard",
    "public_id",
    "company",
    "valid_until",
    "source_title",
    "source_url",
    "local_path",
    "followup_needed",
]


DB_BY_CATEGORY = {
    "wall_insulation": "wall_insulation.db",
    "exterior_wall": "exterior_wall.db",
    "roof_u": "roof.db",
    "floor_contact_type": "floor.db",
    "window_u": "windows.db",
    "shading": "shading.db",
}


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_unit(unit: str) -> str:
    unit = clean_text(unit)
    return (
        unit.replace("m2", "m2")
        .replace("㎡", "m2")
        .replace("·", "·")
        .replace(" ", "")
    )


def add_existing_candidates(rows: list[dict[str, str]], csv_name: str) -> None:
    path = REVIEW / csv_name
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            category = clean_text(item.get("component_category"))
            parameter_type = clean_text(item.get("parameter_type"))
            name = clean_text(item.get("name"))
            value = clean_text(item.get("value"))
            unit = normalize_unit(item.get("unit", ""))
            if not name or not value:
                continue
            # Keep concrete material/system records. Skip generic window/glass enumerations here;
            # those are already handled by the dedicated windows extraction workflow.
            if category == "window_u":
                continue
            rows.append(
                {
                    "screen_status": "usable_parameter",
                    "target_database": DB_BY_CATEGORY.get(category, ""),
                    "component_category": category,
                    "material_name": name,
                    "parameter_type": parameter_type,
                    "value": value,
                    "unit": unit,
                    "density_kg_m3": "",
                    "specific_heat_j_kgk": "",
                    "heat_storage_w_m2k": "",
                    "wall_correction_factor": "",
                    "roof_correction_factor": "",
                    "floor_correction_factor": "",
                    "source_title": clean_text(item.get("source_title")),
                    "source_url": clean_text(item.get("source_url")),
                    "local_path": f"data/external_sources/review/{csv_name}",
                    "source_note": "existing_candidate_csv",
                }
            )


def classify_shanghai_material(name: str) -> tuple[str, str]:
    if "预制夹心剪力墙" in name:
        return "exterior_wall", "exterior_wall.db"
    if "屋面" in name:
        return "roof_u", "roof.db"
    if any(k in name for k in ["岩棉", "保温", "泡沫玻璃", "聚苯", "XPS", "无机改性"]):
        return "wall_insulation", "wall_insulation.db"
    return "wall_insulation", "wall_insulation.db"


def parse_shanghai_update(rows: list[dict[str, str]]) -> None:
    path = RAW / "shanghai-building-energy-software-material-update-20240104.txt"
    if not path.exists():
        return
    text = path.read_text(encoding="gbk", errors="ignore")
    source_title = "上海市民用建筑节能设计软件更新内容20240104"
    source_url = "https://ciac.zjw.sh.gov.cn/Shjnscinter/DownloadSoftware/%E4%B8%8A%E6%B5%B7%E5%B8%82%E6%B0%91%E7%94%A8%E5%BB%BA%E7%AD%91%E8%8A%82%E8%83%BD%E8%AE%BE%E8%AE%A1%E8%BD%AF%E4%BB%B6%E6%9B%B4%E6%96%B0%E5%86%85%E5%AE%B920240104.txt"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "材料名称" in line:
            continue
        parts = [clean_text(p) for p in re.split(r"\t+", line) if clean_text(p)]
        if len(parts) < 8:
            parts = [clean_text(p) for p in re.split(r"\s{2,}", line) if clean_text(p)]
        if len(parts) < 8:
            continue
        material_name = parts[0]
        if re.match(r"^\d+[、.]", material_name):
            continue
        if material_name in {"变更前", "变更后"}:
            continue
        numeric = parts[-7:]
        if not all(re.fullmatch(r"-?\d+(?:\.\d+)?", x) for x in numeric):
            continue
        density, conductivity, specific_heat, heat_storage, wall_factor, roof_factor, floor_factor = numeric
        category, database = classify_shanghai_material(material_name)
        rows.append(
            {
                "screen_status": "usable_parameter",
                "target_database": database,
                "component_category": category,
                "material_name": material_name,
                "parameter_type": "thermal_conductivity_lambda",
                "value": conductivity,
                "unit": "W/(m·K)",
                "density_kg_m3": density,
                "specific_heat_j_kgk": specific_heat,
                "heat_storage_w_m2k": heat_storage,
                "wall_correction_factor": wall_factor,
                "roof_correction_factor": roof_factor,
                "floor_correction_factor": floor_factor,
                "source_title": source_title,
                "source_url": source_url,
                "local_path": "data/external_sources/raw/shanghai-building-energy-software-material-update-20240104.txt",
                "source_note": "software_material_update_table",
            }
        )


def write_followup_index() -> int:
    src = REVIEW / "material_product_index_expanded_2026-07-07.csv"
    if not src.exists():
        src = REVIEW / "changde_product_index_2026.csv"
    if not src.exists():
        return 0
    rows: list[dict[str, str]] = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            rows.append(
                {
                    "screen_status": "product_index_no_thermal_value",
                    "component_category": clean_text(item.get("category")),
                    "product_name": clean_text(item.get("product_name")),
                    "standard": clean_text(item.get("standard")),
                    "public_id": clean_text(item.get("public_id")),
                    "company": clean_text(item.get("company")),
                    "valid_until": clean_text(item.get("valid_until")),
                    "source_title": clean_text(item.get("source_title")),
                    "source_url": clean_text(item.get("source_url")),
                    "local_path": str(src.relative_to(ROOT)).replace("\\", "/"),
                    "followup_needed": "反查型式检验报告或企业产品资料，补导热系数/传热系数/热阻",
                }
            )
    with OUT_FOLLOWUP.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FOLLOWUP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["component_category"],
            row["material_name"],
            row["parameter_type"],
            row["value"],
            row["unit"],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def main() -> None:
    rows: list[dict[str, str]] = []
    add_existing_candidates(rows, "xiangtan_thermal_parameter_candidates.csv")
    add_existing_candidates(rows, "thermal_parameter_candidates.csv")
    parse_shanghai_update(rows)
    rows = dedupe(rows)
    rows.sort(key=lambda r: (r["target_database"], r["component_category"], r["material_name"], r["parameter_type"]))
    with OUT_PARAMS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PARAM_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    followup_count = write_followup_index()
    print(f"screened_parameters={len(rows)}")
    print(f"followup_products={followup_count}")
    print(f"out_parameters={OUT_PARAMS}")
    print(f"out_followup={OUT_FOLLOWUP}")


if __name__ == "__main__":
    main()
