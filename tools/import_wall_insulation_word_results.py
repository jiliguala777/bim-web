from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "wall_insulation_word_extraction_first50"
DEFAULT_DB_PATH = BASE_DIR / "data" / "envelope_databases" / "wall_insulation.db"


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def as_float(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


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
            source TEXT DEFAULT 'word_extract',
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

    # Legacy sample schema had NOT NULL r/u columns and no unique key. Rebuild once so
    # lambda-only extracted material records can be stored safely.
    info = conn.execute("PRAGMA table_info(wall_insulation_systems)").fetchall()
    needs_rebuild = any(row[1] in {"r_value_m2k_w", "u_wall_w_m2k"} and row[3] for row in info)
    indexes = conn.execute("PRAGMA index_list(wall_insulation_systems)").fetchall()
    has_unique = any(row[2] for row in indexes)
    if needs_rebuild or not has_unique:
        rebuild_wall_table(conn)


def rebuild_wall_table(conn: sqlite3.Connection) -> None:
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
            source TEXT DEFAULT 'word_extract',
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


def choose_name(card: dict[str, str]) -> str:
    return clean(card.get("sample_name")) or clean(card.get("spec_model")) or clean(card.get("report_id")) or "未命名外保温系统"


def group_parameters(parameters: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"evidence": [], "raw": []})
    for row in parameters:
        report_id = clean(row.get("report_id"))
        parameter_type = clean(row.get("parameter_type"))
        value = as_float(row.get("numeric_value"))
        if not report_id or value is None:
            continue
        item = grouped[report_id]
        if parameter_type == "thermal_resistance_r" and item.get("r_value_m2k_w") is None:
            item["r_value_m2k_w"] = value
        elif parameter_type == "thermal_transmittance_u" and item.get("u_wall_w_m2k") is None:
            item["u_wall_w_m2k"] = value
        elif parameter_type == "thermal_conductivity_lambda" and item.get("lambda_w_mk") is None:
            item["lambda_w_mk"] = value
        item["evidence"].append(clean(row.get("evidence")))
        item["raw"].append(f"{parameter_type}: {clean(row.get('raw_value'))}")
    return grouped


def import_results(input_dir: Path, db_path: Path, replace_source: bool = False) -> int:
    cards = read_csv(input_dir / "report_cards.csv")
    params = read_csv(input_dir / "thermal_parameters.csv")
    grouped = group_parameters(params)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        if replace_source:
            conn.execute("DELETE FROM wall_insulation_systems WHERE source = ?", ("word_extract",))

        imported = 0
        for card in cards:
            report_id = clean(card.get("report_id"))
            values = grouped.get(report_id)
            if not values:
                continue
            if not any(values.get(key) is not None for key in ("r_value_m2k_w", "u_wall_w_m2k", "lambda_w_mk")):
                continue
            source_item_path = " | ".join(dict.fromkeys(v for v in values["evidence"] if v))
            source_result = " | ".join(dict.fromkeys(v for v in values["raw"] if v))
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
                    choose_name(card),
                    values.get("r_value_m2k_w"),
                    values.get("u_wall_w_m2k"),
                    values.get("lambda_w_mk"),
                    clean(card.get("manufacturer")),
                    clean(card.get("brand")),
                    clean(card.get("spec_model")),
                    clean(card.get("production_date")),
                    clean(card.get("sample_no")),
                    clean(card.get("test_basis")),
                    clean(card.get("test_items")),
                    clean(card.get("test_conclusion")),
                    report_id,
                    source_item_path,
                    source_result,
                    "Imported from direct Word extraction. Thermal resistance, U-value, and material lambda are stored separately.",
                    "word_extract",
                ),
            )
            imported += 1
        conn.commit()
        return imported
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import direct Word wall-insulation extraction results into wall_insulation.db.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--append", action="store_true", help="Deprecated: append is now the default.")
    parser.add_argument("--replace-source", action="store_true", help="Delete prior word_extract rows before importing.")
    args = parser.parse_args()

    count = import_results(Path(args.input_dir), Path(args.db_path), replace_source=args.replace_source)
    print(f"Imported wall insulation rows: {count}")
    print(f"Database: {Path(args.db_path)}")


if __name__ == "__main__":
    main()
