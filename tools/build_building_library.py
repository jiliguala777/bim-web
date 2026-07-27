import csv
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "candidate_material_parameters.csv"
DB_PATH = DATA_DIR / "building_library.db"
PREVIEW_PATH = DATA_DIR / "building_library_preview.csv"


def clean(value):
    return " ".join(str(value or "").strip().split())


def as_float(value):
    value = clean(value)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fallback_name(row, prefix):
    sample_name = clean(row.get("sample_name"))
    if sample_name:
        return sample_name
    report_id = clean(row.get("source_report_id"))
    return f"{prefix} {report_id}".strip()


def infer_material_category(row):
    text = " ".join(
        [
            clean(row.get("sample_name")),
            clean(row.get("item_path")),
            clean(row.get("test_items")),
        ]
    )
    if "玻璃" in text or "窗" in text:
        return "window_material"
    if "屋面" in text:
        return "roof_material"
    if "保温" in text or "聚苯" in text or "聚氨酯" in text or "岩棉" in text:
        return "insulation"
    return "material"


def reset_schema(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS source_reports;
        DROP TABLE IF EXISTS materials;
        DROP TABLE IF EXISTS wall_assemblies;
        DROP TABLE IF EXISTS candidate_parameter_reviews;

        CREATE TABLE source_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            sample_name TEXT,
            manufacturer TEXT,
            brand TEXT,
            spec_model TEXT,
            test_basis TEXT,
            test_items TEXT
        );

        CREATE TABLE materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            lambda_w_mk REAL,
            density_kg_m3 REAL,
            thickness_mm REAL,
            source_report_id TEXT,
            source_item_path TEXT,
            source_result TEXT,
            notes TEXT,
            UNIQUE(name, category, source_report_id, source_item_path)
        );

        CREATE TABLE wall_assemblies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            r_value_m2k_w REAL,
            u_wall_w_m2k REAL,
            source_report_id TEXT,
            source_item_path TEXT,
            source_result TEXT,
            notes TEXT,
            UNIQUE(name, source_report_id, source_item_path)
        );

        CREATE TABLE candidate_parameter_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_status TEXT,
            recommended_use TEXT,
            parameter_type TEXT,
            numeric_value REAL,
            unit TEXT,
            source_report_id TEXT,
            sample_name TEXT,
            manufacturer TEXT,
            spec_model TEXT,
            item_path TEXT,
            requirement TEXT,
            result TEXT,
            conclusion TEXT,
            notes TEXT
        );
        """
    )


def upsert_report(conn, row):
    conn.execute(
        """
        INSERT INTO source_reports (
            report_id, sample_name, manufacturer, brand, spec_model, test_basis, test_items
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(report_id) DO UPDATE SET
            sample_name = COALESCE(NULLIF(excluded.sample_name, ''), source_reports.sample_name),
            manufacturer = COALESCE(NULLIF(excluded.manufacturer, ''), source_reports.manufacturer),
            brand = COALESCE(NULLIF(excluded.brand, ''), source_reports.brand),
            spec_model = COALESCE(NULLIF(excluded.spec_model, ''), source_reports.spec_model),
            test_basis = COALESCE(NULLIF(excluded.test_basis, ''), source_reports.test_basis),
            test_items = COALESCE(NULLIF(excluded.test_items, ''), source_reports.test_items)
        """,
        (
            clean(row.get("source_report_id")),
            clean(row.get("sample_name")),
            clean(row.get("manufacturer")),
            clean(row.get("brand")),
            clean(row.get("spec_model")),
            clean(row.get("test_basis")),
            clean(row.get("test_items")),
        ),
    )


def insert_review_row(conn, row):
    conn.execute(
        """
        INSERT INTO candidate_parameter_reviews (
            review_status, recommended_use, parameter_type, numeric_value, unit,
            source_report_id, sample_name, manufacturer, spec_model, item_path,
            requirement, result, conclusion, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clean(row.get("review_status")),
            clean(row.get("recommended_use")),
            clean(row.get("parameter_type")),
            as_float(row.get("numeric_value")),
            clean(row.get("unit")),
            clean(row.get("source_report_id")),
            clean(row.get("sample_name")),
            clean(row.get("manufacturer")),
            clean(row.get("spec_model")),
            clean(row.get("item_path")),
            clean(row.get("requirement")),
            clean(row.get("result")),
            clean(row.get("conclusion")),
            clean(row.get("notes")),
        ),
    )


def build_library(rows):
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    reset_schema(conn)

    preview_rows = []
    inserted_materials = 0
    inserted_walls = 0

    for row in rows:
        status = clean(row.get("review_status")).lower()
        if status in {"keep", "needs_check"}:
            upsert_report(conn, row)
            insert_review_row(conn, row)
        if status != "keep":
            continue

        parameter_type = clean(row.get("parameter_type"))
        numeric_value = as_float(row.get("numeric_value"))
        if numeric_value is None:
            continue

        if parameter_type == "thermal_conductivity_lambda":
            name = fallback_name(row, "未命名材料")
            category = infer_material_category(row)
            conn.execute(
                """
                INSERT OR IGNORE INTO materials (
                    name, category, lambda_w_mk, source_report_id,
                    source_item_path, source_result, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    category,
                    numeric_value,
                    clean(row.get("source_report_id")),
                    clean(row.get("item_path")),
                    clean(row.get("result")),
                    "Imported from reviewed report parameter.",
                ),
            )
            inserted_materials += conn.total_changes
            preview_rows.append(
                {
                    "library_table": "materials",
                    "name": name,
                    "category": category,
                    "parameter": "lambda_w_mk",
                    "value": numeric_value,
                    "unit": "W/(m·K)",
                    "source_report_id": clean(row.get("source_report_id")),
                    "source_item_path": clean(row.get("item_path")),
                }
            )

        elif parameter_type == "thermal_resistance_r":
            name = fallback_name(row, "未命名外墙构造")
            u_value = 1.0 / numeric_value if numeric_value else None
            conn.execute(
                """
                INSERT OR IGNORE INTO wall_assemblies (
                    name, r_value_m2k_w, u_wall_w_m2k, source_report_id,
                    source_item_path, source_result, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    numeric_value,
                    u_value,
                    clean(row.get("source_report_id")),
                    clean(row.get("item_path")),
                    clean(row.get("result")),
                    "U value is calculated as 1 / reported R value. Confirm whether report R includes surface resistance before engineering use.",
                ),
            )
            inserted_walls += conn.total_changes
            preview_rows.append(
                {
                    "library_table": "wall_assemblies",
                    "name": name,
                    "category": "wall_assembly",
                    "parameter": "u_wall_w_m2k",
                    "value": round(u_value, 4) if u_value is not None else "",
                    "unit": "W/(m²·K)",
                    "source_report_id": clean(row.get("source_report_id")),
                    "source_item_path": clean(row.get("item_path")),
                }
            )

    conn.commit()

    material_count = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
    wall_count = conn.execute("SELECT COUNT(*) FROM wall_assemblies").fetchone()[0]
    review_count = conn.execute("SELECT COUNT(*) FROM candidate_parameter_reviews").fetchone()[0]
    conn.close()

    with PREVIEW_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "library_table",
            "name",
            "category",
            "parameter",
            "value",
            "unit",
            "source_report_id",
            "source_item_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(preview_rows)

    return material_count, wall_count, review_count


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {CSV_PATH}")
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    material_count, wall_count, review_count = build_library(rows)
    print(f"Wrote SQLite library: {DB_PATH}")
    print(f"Wrote preview CSV: {PREVIEW_PATH}")
    print(f"materials={material_count}, wall_assemblies={wall_count}, review_rows={review_count}")


if __name__ == "__main__":
    main()
