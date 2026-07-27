from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURED_ROOT = BASE_DIR / "data" / "wall_insulation_structured_full"
DEFAULT_DB_PATH = BASE_DIR / "data" / "envelope_databases" / "wall_insulation.db"

META_ALIASES = {
    "sample_name": ("样品名称", "sample name", "产品名称"),
    "manufacturer": ("生产单位", "manufacturer", "生产厂家", "厂家"),
    "brand": ("商标", "商 标", "brand"),
    "spec_model": ("规格型号", "型号规格", "规格", "型号", "type/model", "model"),
    "production_date": ("生产日期", "出厂日期"),
    "sample_no": ("样品编号", "sample no", "样品号"),
    "test_basis": ("检验依据", "检测依据", "test basis", "standard"),
    "test_items": ("检验项目", "检测项目", "test items"),
    "test_conclusion": ("检验结论", "检测结论", "test conclusion"),
}

THERMAL_TYPES = [
    ("thermal_transmittance_u", ("传热系数", "u值", "u 值", "u-value")),
    ("thermal_resistance_r", ("热阻", "传热阻")),
    ("thermal_conductivity_lambda", ("导热系数",)),
]


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def source_report_id(report_dir: Path) -> str:
    return re.sub(r"__[0-9a-f]{10,12}$", "", report_dir.name, flags=re.I)


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


def extract_meta(report_dir: Path) -> dict[str, str]:
    meta = {
        "sample_name": "",
        "manufacturer": "",
        "brand": "",
        "spec_model": "",
        "production_date": "",
        "sample_no": "",
        "test_basis": "",
        "test_items": "",
        "test_conclusion": "",
    }
    for row in read_csv(report_dir / "meta.csv"):
        key = meta_key(row.get("field", ""))
        if not key:
            continue
        value = clean(row.get("value"))
        if key in {"test_basis", "test_items", "test_conclusion"}:
            meta[key] = append_unique(meta[key], value)
        elif not meta[key]:
            meta[key] = value
    return meta


def item_parameter_type(item_text: str) -> str:
    lowered = clean(item_text).lower()
    for parameter_type, keywords in THERMAL_TYPES:
        if any(keyword.lower() in lowered for keyword in keywords):
            return parameter_type
    return ""


def parse_number_unit(text: str) -> tuple[float | None, str]:
    value = clean(text)
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    number = as_float(number_match.group(0)) if number_match else None
    unit = ""
    unit_patterns = [
        r"W/[（(]?m[²2]?[·.]?K[）)]?",
        r"W/\(?m[²2]?[·.]?K\)?",
        r"[（(]?m[²2][·.]?K[）)]?/W",
        r"\(?m[²2][·.]?K\)?/W",
    ]
    for pattern in unit_patterns:
        match = re.search(pattern, value, re.I)
        if match:
            unit = clean(match.group(0))
            break
    return number, unit


def extract_item_parameters(report_dir: Path) -> list[dict[str, object]]:
    parameters: list[dict[str, object]] = []
    for row in read_csv(report_dir / "items.csv"):
        context = " / ".join(
            part
            for part in [
                clean(row.get("section")),
                clean(row.get("item_name")),
                clean(row.get("requirement")),
                clean(row.get("result")),
            ]
            if part
        )
        parameter_type = item_parameter_type(context)
        if not parameter_type:
            continue

        # Use measured result only. Do not fall back to requirement; requirements
        # are threshold values, not the measured parameter for the library.
        result = clean(row.get("result"))
        number, unit = parse_number_unit(result)
        if number is None:
            continue
        parameters.append(
            {
                "parameter_type": parameter_type,
                "numeric_value": number,
                "unit": unit,
                "source_item_path": context,
                "source_result": result,
            }
        )
    return parameters


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wall_insulation_systems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            r_value_m2k_w REAL,
            u_wall_w_m2k REAL,
            lambda_w_mk REAL,
            manufacturer TEXT,
            brand TEXT,
            spec_model TEXT,
            production_date TEXT,
            sample_no TEXT,
            test_basis TEXT,
            test_items TEXT,
            test_conclusion TEXT,
            source_report_id TEXT,
            source_item_path TEXT,
            source_result TEXT,
            notes TEXT,
            source TEXT DEFAULT 'structured_items',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_report_id, source_item_path)
        )
        """
    )
    cols = column_names(conn, "wall_insulation_systems")
    additions = {
        "lambda_w_mk": "REAL",
        "manufacturer": "TEXT",
        "brand": "TEXT",
        "spec_model": "TEXT",
        "production_date": "TEXT",
        "sample_no": "TEXT",
        "test_basis": "TEXT",
        "test_items": "TEXT",
        "test_conclusion": "TEXT",
    }
    for name, col_type in additions.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE wall_insulation_systems ADD COLUMN {name} {col_type}")

    info = conn.execute("PRAGMA table_info(wall_insulation_systems)").fetchall()
    needs_rebuild = any(row[1] in {"r_value_m2k_w", "u_wall_w_m2k"} and row[3] for row in info)
    indexes = conn.execute("PRAGMA index_list(wall_insulation_systems)").fetchall()
    has_unique = any(row[2] for row in indexes)
    if needs_rebuild or not has_unique:
        rebuild_table(conn)


def rebuild_table(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE wall_insulation_systems RENAME TO wall_insulation_systems_old")
    conn.execute(
        """
        CREATE TABLE wall_insulation_systems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            r_value_m2k_w REAL,
            u_wall_w_m2k REAL,
            lambda_w_mk REAL,
            manufacturer TEXT,
            brand TEXT,
            spec_model TEXT,
            production_date TEXT,
            sample_no TEXT,
            test_basis TEXT,
            test_items TEXT,
            test_conclusion TEXT,
            source_report_id TEXT,
            source_item_path TEXT,
            source_result TEXT,
            notes TEXT,
            source TEXT DEFAULT 'structured_items',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_report_id, source_item_path)
        )
        """
    )
    old_cols = column_names(conn, "wall_insulation_systems_old")
    target_cols = [
        "id",
        "name",
        "r_value_m2k_w",
        "u_wall_w_m2k",
        "lambda_w_mk",
        "manufacturer",
        "brand",
        "spec_model",
        "production_date",
        "sample_no",
        "test_basis",
        "test_items",
        "test_conclusion",
        "source_report_id",
        "source_item_path",
        "source_result",
        "notes",
        "source",
        "created_at",
    ]
    common = [col for col in target_cols if col in old_cols]
    conn.execute(
        f"""
        INSERT OR IGNORE INTO wall_insulation_systems ({", ".join(common)})
        SELECT {", ".join(common)}
        FROM wall_insulation_systems_old
        """
    )
    conn.execute("DROP TABLE wall_insulation_systems_old")


def choose_name(meta: dict[str, str], report_id: str) -> str:
    name = clean(meta.get("sample_name"))
    if name and name != "------":
        return name
    spec = clean(meta.get("spec_model"))
    if spec and spec != "------":
        return spec
    return report_id


def grouped_parameters(parameters: list[dict[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {"evidence": [], "raw": []}
    for parameter in parameters:
        parameter_type = parameter["parameter_type"]
        value = parameter["numeric_value"]
        if parameter_type == "thermal_resistance_r" and values.get("r_value_m2k_w") is None:
            values["r_value_m2k_w"] = value
        elif parameter_type == "thermal_transmittance_u" and values.get("u_wall_w_m2k") is None:
            values["u_wall_w_m2k"] = value
        elif parameter_type == "thermal_conductivity_lambda" and values.get("lambda_w_mk") is None:
            values["lambda_w_mk"] = value
        values["evidence"].append(clean(parameter["source_item_path"]))
        values["raw"].append(f"{parameter_type}: {clean(parameter['source_result'])}")
    return values


def import_structured(
    root: Path,
    db_path: Path,
    remove_word_extract: bool = False,
    replace_source: bool = False,
) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        if replace_source:
            conn.execute("DELETE FROM wall_insulation_systems WHERE source = ?", ("structured_items",))
        if remove_word_extract:
            conn.execute("DELETE FROM wall_insulation_systems WHERE source = ?", ("word_extract",))

        imported = 0
        for report_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            report_id = source_report_id(report_dir)
            meta = extract_meta(report_dir)
            params = extract_item_parameters(report_dir)
            if not params:
                continue
            values = grouped_parameters(params)
            if not any(values.get(key) is not None for key in ("r_value_m2k_w", "u_wall_w_m2k", "lambda_w_mk")):
                continue
            conn.execute(
                """
                INSERT INTO wall_insulation_systems (
                    name,
                    r_value_m2k_w,
                    u_wall_w_m2k,
                    lambda_w_mk,
                    manufacturer,
                    brand,
                    spec_model,
                    production_date,
                    sample_no,
                    test_basis,
                    test_items,
                    test_conclusion,
                    source_report_id,
                    source_item_path,
                    source_result,
                    notes,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_report_id, source_item_path) DO UPDATE SET
                    name = excluded.name,
                    r_value_m2k_w = excluded.r_value_m2k_w,
                    u_wall_w_m2k = excluded.u_wall_w_m2k,
                    lambda_w_mk = excluded.lambda_w_mk,
                    manufacturer = excluded.manufacturer,
                    brand = excluded.brand,
                    spec_model = excluded.spec_model,
                    production_date = excluded.production_date,
                    sample_no = excluded.sample_no,
                    test_basis = excluded.test_basis,
                    test_items = excluded.test_items,
                    test_conclusion = excluded.test_conclusion,
                    source_result = excluded.source_result,
                    notes = excluded.notes,
                    source = excluded.source
                """,
                (
                    choose_name(meta, report_id),
                    values.get("r_value_m2k_w"),
                    values.get("u_wall_w_m2k"),
                    values.get("lambda_w_mk"),
                    clean(meta.get("manufacturer")),
                    clean(meta.get("brand")),
                    clean(meta.get("spec_model")),
                    clean(meta.get("production_date")),
                    clean(meta.get("sample_no")),
                    clean(meta.get("test_basis")),
                    clean(meta.get("test_items")),
                    clean(meta.get("test_conclusion")),
                    report_id,
                    " | ".join(dict.fromkeys(v for v in values["evidence"] if v)),
                    " | ".join(dict.fromkeys(v for v in values["raw"] if v)),
                    "Imported from structured meta/items CSV only. trace.csv was not used.",
                    "structured_items",
                ),
            )
            imported += 1
        conn.commit()
        return imported
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import wall-insulation thermal parameters from structured meta/items CSV folders.")
    parser.add_argument("--structured-root", default=str(DEFAULT_STRUCTURED_ROOT))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--remove-word-extract", action="store_true")
    parser.add_argument("--append", action="store_true", help="Deprecated: append is now the default.")
    parser.add_argument("--replace-source", action="store_true", help="Delete prior structured_items rows before importing.")
    args = parser.parse_args()

    count = import_structured(
        Path(args.structured_root),
        Path(args.db_path),
        args.remove_word_extract,
        replace_source=args.replace_source,
    )
    print(f"Imported structured wall insulation rows: {count}")
    print(f"Database: {Path(args.db_path)}")


if __name__ == "__main__":
    main()
