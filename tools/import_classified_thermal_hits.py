from __future__ import annotations

import argparse
import csv
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "data" / "envelope_databases"

DEFAULT_CLASSIFICATION = BASE_DIR / "data" / "report_processing_audit" / "envelope_thermal_hit_fine_classification.csv"
DEFAULT_CARDS = BASE_DIR / "data" / "envelope_thermal_hit_excel_exports_full" / "wall_insulation_report_cards.csv"
DEFAULT_PARAMS = BASE_DIR / "data" / "envelope_thermal_full" / "thermal_parameters.csv"
DEFAULT_AUDIT = BASE_DIR / "data" / "report_processing_audit" / "classified_thermal_hit_import_audit.csv"

SOURCE_NAME = "classified_thermal_hits"

CATEGORY_DATABASES = {
    "wall_insulation": ("wall_insulation.db", "wall_insulation_systems"),
    "exterior_wall": ("exterior_wall.db", "parameter_options"),
    "window_u": ("windows.db", "parameter_options"),
    "door_u": ("door.db", "parameter_options"),
    "roof_u": ("roof.db", "parameter_options"),
    "floor_u": ("floor.db", "parameter_options"),
    "window_shgc": ("windows.db", "parameter_options"),
    "curtain_shading": ("shading.db", "parameter_options"),
}

U_CATEGORIES = {"exterior_wall", "window_u", "door_u", "roof_u", "floor_u"}

AUDIT_FIELDS = [
    "status",
    "reason",
    "report_id",
    "fine_category",
    "target_database",
    "target_table",
    "name",
    "value",
    "unit",
    "parameter_type",
    "source_result",
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


def as_float(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def report_key(report_id: str) -> str:
    text = clean(report_id)
    text = re.sub(r"__[0-9a-f]{10,12}$", "", text, flags=re.I)
    text = re.sub(r"_[0-9a-f]{10,12}$", "", text, flags=re.I)
    return text


def short_report_id(report_id: str) -> str:
    return report_key(report_id).split("\\")[-1]


def choose_name(row: dict[str, str], fallback: str) -> str:
    for key in ("sample_name", "spec_model", "report_id"):
        value = clean(row.get(key))
        if value and value != "------" and value != "——":
            return value
    return fallback


def base_display_name(row: dict[str, str]) -> str:
    return choose_name(row, short_report_id(row.get("report_id", "")))


def load_cards(path: Path) -> dict[str, dict[str, str]]:
    cards = {}
    for row in read_csv(path):
        rid = clean(row.get("report_id"))
        if rid:
            cards[rid] = row
            cards[report_key(rid)] = row
    return cards


def load_parameters(path: Path) -> dict[str, list[dict[str, str]]]:
    by_report: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        rid = clean(row.get("report_id"))
        if not rid:
            continue
        by_report[rid].append(row)
        key = report_key(rid)
        if key != rid:
            by_report[key].append(row)
    return by_report


def group_wall_values(params: list[dict[str, str]]) -> dict[str, object]:
    values: dict[str, object] = {"evidence": [], "raw": []}
    for row in params:
        parameter_type = clean(row.get("parameter_type"))
        value = as_float(row.get("numeric_value"))
        if value is None:
            continue
        if parameter_type == "thermal_resistance_r" and values.get("r_value_m2k_w") is None:
            values["r_value_m2k_w"] = value
        elif parameter_type == "thermal_transmittance_u" and values.get("u_wall_w_m2k") is None:
            values["u_wall_w_m2k"] = value
        elif parameter_type == "thermal_conductivity_lambda" and values.get("lambda_w_mk") is None:
            values["lambda_w_mk"] = value
        values["evidence"].append(clean(row.get("evidence")))
        values["raw"].append(f"{parameter_type}: {clean(row.get('raw_value'))}")
    return values


def select_u_parameter(params: list[dict[str, str]]) -> dict[str, str] | None:
    for row in params:
        if clean(row.get("parameter_type")) == "thermal_transmittance_u" and as_float(row.get("numeric_value")) is not None:
            return row
    return None


def ensure_wall_table(conn: sqlite3.Connection) -> None:
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
            source TEXT DEFAULT 'classified_thermal_hits',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_report_id, source_item_path)
        )
        """
    )


def ensure_parameter_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parameter_options (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            notes TEXT,
            source TEXT DEFAULT 'classified_thermal_hits',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, name)
        )
        """
    )


def backup_databases(categories: set[str]) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = []
    seen_db_names = set()
    for category in sorted(categories):
        db_name, _ = CATEGORY_DATABASES[category]
        if db_name in seen_db_names:
            continue
        seen_db_names.add(db_name)
        src = DB_DIR / db_name
        if not src.exists():
            continue
        dst = DB_DIR / f"{src.stem}_before_classified_thermal_hits_{timestamp}{src.suffix}"
        shutil.copy2(src, dst)
        backups.append(dst)
    return backups


def existing_parameter_names(db_name: str, category: str) -> set[str]:
    db_path = DB_DIR / db_name
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    try:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'parameter_options'"
        ).fetchone()
        if not table:
            return set()
        return {
            clean(row[0])
            for row in conn.execute(
                "SELECT name FROM parameter_options WHERE category = ? AND source <> ?",
                (category, SOURCE_NAME),
            ).fetchall()
            if clean(row[0])
        }
    finally:
        conn.close()


def allocate_numbered_name(base_name: str, used_names: set[str]) -> str:
    base = clean(base_name) or "未命名"
    if base not in used_names:
        used_names.add(base)
        return base
    index = 1
    while True:
        candidate = f"{base}{index}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def delete_classified_source(conn: sqlite3.Connection, table_name: str) -> None:
    if table_name == "wall_insulation_systems":
        conn.execute("DELETE FROM wall_insulation_systems WHERE source = ?", (SOURCE_NAME,))
    else:
        conn.execute("DELETE FROM parameter_options WHERE source = ?", (SOURCE_NAME,))


def import_rows(args: argparse.Namespace) -> tuple[Counter[str], list[dict[str, str]]]:
    classifications = read_csv(Path(args.classification))
    cards = load_cards(Path(args.cards))
    params_by_report = load_parameters(Path(args.params))
    audit_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    import_categories = {
        clean(row.get("fine_category"))
        for row in classifications
        if clean(row.get("fine_category")) in CATEGORY_DATABASES
    }
    if not args.dry_run and not args.no_backup:
        for backup in backup_databases(import_categories):
            print(f"backup={backup}")

    connections: dict[str, sqlite3.Connection] = {}
    used_parameter_names: dict[tuple[str, str], set[str]] = {}
    try:
        for row in classifications:
            category = clean(row.get("fine_category"))
            report_id = clean(row.get("report_id"))
            if category == "needs_review":
                counts["skipped_needs_review"] += 1
                continue
            if category not in CATEGORY_DATABASES:
                counts["skipped_unknown_category"] += 1
                continue

            db_name, table_name = CATEGORY_DATABASES[category]
            card = cards.get(report_id) or cards.get(report_key(report_id)) or row
            params = params_by_report.get(report_id) or params_by_report.get(report_key(report_id)) or []
            if not params:
                counts["skipped_no_parameters"] += 1
                audit_rows.append(
                    {
                        "status": "skipped",
                        "reason": "no_parameters",
                        "report_id": report_id,
                        "fine_category": category,
                        "target_database": db_name,
                        "target_table": table_name,
                    }
                )
                continue

            db_path = DB_DIR / db_name
            if not args.dry_run and db_name not in connections:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(db_path)
                if table_name == "wall_insulation_systems":
                    ensure_wall_table(conn)
                else:
                    ensure_parameter_table(conn)
                if args.replace_classified_source:
                    delete_classified_source(conn, table_name)
                connections[db_name] = conn

            if category == "wall_insulation":
                values = group_wall_values(params)
                if not any(values.get(key) is not None for key in ("r_value_m2k_w", "u_wall_w_m2k", "lambda_w_mk")):
                    counts["skipped_no_compatible_parameter"] += 1
                    continue
                source_item_path = " | ".join(dict.fromkeys(v for v in values["evidence"] if v))
                source_result = " | ".join(dict.fromkeys(v for v in values["raw"] if v))
                name = choose_name(card, report_id)
                if not args.dry_run:
                    connections[db_name].execute(
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
                            name,
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
                            "Imported from thermal-hit fine classification.",
                            SOURCE_NAME,
                        ),
                    )
                counts[f"imported_{category}"] += 1
                audit_rows.append(
                    {
                        "status": "imported",
                        "reason": "",
                        "report_id": report_id,
                        "fine_category": category,
                        "target_database": db_name,
                        "target_table": table_name,
                        "name": name,
                        "value": clean(values.get("u_wall_w_m2k") or values.get("lambda_w_mk") or values.get("r_value_m2k_w")),
                        "unit": "",
                        "parameter_type": "mixed_wall_parameters",
                        "source_result": source_result,
                    }
                )
                continue

            if category not in U_CATEGORIES:
                counts["skipped_no_importer_for_category"] += 1
                audit_rows.append(
                    {
                        "status": "skipped",
                        "reason": "no_compatible_importer_for_category",
                        "report_id": report_id,
                        "fine_category": category,
                        "target_database": db_name,
                        "target_table": table_name,
                    }
                )
                continue

            param = select_u_parameter(params)
            if not param:
                counts["skipped_no_u_value"] += 1
                audit_rows.append(
                    {
                        "status": "skipped",
                        "reason": "no_thermal_transmittance_u",
                        "report_id": report_id,
                        "fine_category": category,
                        "target_database": db_name,
                        "target_table": table_name,
                    }
                )
                continue

            value = as_float(param.get("numeric_value"))
            name_key = (db_name, category)
            if name_key not in used_parameter_names:
                used_parameter_names[name_key] = existing_parameter_names(db_name, category)
            name = allocate_numbered_name(base_display_name({**card, **row}), used_parameter_names[name_key])
            unit = clean(param.get("unit")) or "W/(m2·K)"
            notes = (
                f"报告 {report_id}; 分类 {category}; "
                f"{clean(param.get('evidence'))}; 原值 {clean(param.get('raw_value'))}"
            )
            if not args.dry_run:
                existing = connections[db_name].execute(
                    "SELECT id FROM parameter_options WHERE category = ? AND name = ? LIMIT 1",
                    (category, name),
                ).fetchone()
                if existing:
                    connections[db_name].execute(
                        """
                        UPDATE parameter_options
                        SET value = ?, unit = ?, notes = ?, source = ?
                        WHERE id = ?
                        """,
                        (value, unit, notes, SOURCE_NAME, existing[0]),
                    )
                else:
                    connections[db_name].execute(
                        """
                        INSERT INTO parameter_options (category, name, value, unit, notes, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (category, name, value, unit, notes, SOURCE_NAME),
                    )
            counts[f"imported_{category}"] += 1
            audit_rows.append(
                {
                    "status": "imported",
                    "reason": "",
                    "report_id": report_id,
                    "fine_category": category,
                    "target_database": db_name,
                    "target_table": table_name,
                    "name": name,
                    "value": clean(value),
                    "unit": unit,
                    "parameter_type": clean(param.get("parameter_type")),
                    "source_result": clean(param.get("raw_value")),
                }
            )

        if not args.dry_run:
            for conn in connections.values():
                conn.commit()
    finally:
        for conn in connections.values():
            conn.close()

    write_csv(Path(args.audit), audit_rows, AUDIT_FIELDS)
    return counts, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import classified thermal-hit reports into envelope category databases.")
    parser.add_argument("--classification", default=str(DEFAULT_CLASSIFICATION))
    parser.add_argument("--cards", default=str(DEFAULT_CARDS))
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--replace-classified-source",
        action="store_true",
        help="Delete only prior classified_thermal_hits rows before importing.",
    )
    args = parser.parse_args()

    counts, _ = import_rows(args)
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")
    print(f"audit={args.audit}")


if __name__ == "__main__":
    main()
