import os
import sqlite3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DB_PATH = os.environ.get(
    "BUILDING_LIBRARY_DB_PATH",
    os.path.join(BASE_DIR, "data", "building_library.db"),
)
ENVELOPE_DB_DIR = os.environ.get(
    "ENVELOPE_DB_DIR",
    os.path.join(BASE_DIR, "data", "envelope_databases"),
)
ENVELOPE_CATEGORY_DATABASES = {
    "exterior_wall": "exterior_wall.db",
    "window_u": "windows.db",
    "window_shgc": "windows.db",
    "roof_u": "roof.db",
    "floor_u": "floor.db",
    "curtain_shading": "shading.db",
    "door_u": "door.db",
    "floor_contact_type": "floor_contact.db",
    "window_air_tightness": "window_air_tightness.db",
}
WALL_INSULATION_DB_PATH = os.path.join(ENVELOPE_DB_DIR, "wall_insulation.db")


def get_library_connection():
    conn = sqlite3.connect(LIBRARY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row):
    return {key: row[key] for key in row.keys()}


def get_envelope_db_connection(db_name):
    db_path = os.path.join(ENVELOPE_DB_DIR, db_name)
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_wall_assemblies():
    if os.path.exists(WALL_INSULATION_DB_PATH):
        conn = sqlite3.connect(WALL_INSULATION_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT
                    id,
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
                FROM wall_insulation_systems
                ORDER BY
                    CASE source
                        WHEN 'structured_items' THEN 0
                        WHEN 'sample' THEN 1
                        ELSE 2
                    END,
                    name,
                    source_report_id
                """
            ).fetchall()
            items = [row_to_dict(row) for row in rows]
            if items:
                return items
        finally:
            conn.close()

    if not os.path.exists(LIBRARY_DB_PATH):
        return []

    conn = get_library_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                name,
                r_value_m2k_w,
                u_wall_w_m2k,
                source_report_id,
                source_item_path,
                source_result,
                notes
            FROM wall_assemblies
            ORDER BY name, source_report_id
            """
        ).fetchall()
        return [row_to_dict(row) for row in rows]
    finally:
        conn.close()


def list_materials():
    if not os.path.exists(LIBRARY_DB_PATH):
        return []

    conn = get_library_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                name,
                category,
                lambda_w_mk,
                density_kg_m3,
                thickness_mm,
                source_report_id,
                source_item_path,
                source_result,
                notes
            FROM materials
            ORDER BY category, name, source_report_id
            """
        ).fetchall()
        return [row_to_dict(row) for row in rows]
    finally:
        conn.close()


def list_envelope_parameters(category):
    db_name = ENVELOPE_CATEGORY_DATABASES.get(category)
    if db_name:
        conn = get_envelope_db_connection(db_name)
        if conn is not None:
            try:
                rows = conn.execute(
                    """
                    SELECT
                    id,
                    category,
                    name,
                    value,
                    unit,
                    notes,
                    source
                    FROM parameter_options
                    WHERE category = ?
                    ORDER BY
                        CASE source
                            WHEN 'structured_windows' THEN 0
                            WHEN 'sample' THEN 1
                            ELSE 2
                        END,
                        name
                    """,
                    (category,),
                ).fetchall()
                items = [row_to_dict(row) for row in rows]
                if items:
                    return items
            finally:
                conn.close()

    if not os.path.exists(LIBRARY_DB_PATH):
        return []

    conn = get_library_connection()
    try:
        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "envelope_parameters" not in table_names:
            return []

        rows = conn.execute(
            """
            SELECT
                id,
                category,
                name,
                value,
                unit,
                notes
            FROM envelope_parameters
            WHERE category = ?
            ORDER BY name
            """,
            (category,),
        ).fetchall()
        return [row_to_dict(row) for row in rows]
    finally:
        conn.close()
