import argparse
import csv
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "data" / "envelope_databases"

CATEGORY_DATABASES = {
    "exterior_wall": "exterior_wall.db",
    "window_u": "windows.db",
    "window_shgc": "windows.db",
    "roof_u": "roof.db",
    "floor_u": "floor.db",
    "curtain_shading": "shading.db",
}


def ensure_parameter_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parameter_options (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            notes TEXT,
            source TEXT DEFAULT 'import',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, name)
        )
        """
    )


def import_parameter_options(category, csv_path):
    db_name = CATEGORY_DATABASES[category]
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_DIR / db_name)
    ensure_parameter_table(conn)

    inserted = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            conn.execute(
                """
                INSERT INTO parameter_options (category, name, value, unit, notes, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, name) DO UPDATE SET
                    value = excluded.value,
                    unit = excluded.unit,
                    notes = excluded.notes,
                    source = excluded.source
                """,
                (
                    category,
                    row["name"].strip(),
                    float(row["value"]),
                    (row.get("unit") or "").strip(),
                    (row.get("notes") or "").strip(),
                    (row.get("source") or "import").strip(),
                ),
            )
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Import envelope parameter options into category SQLite databases.")
    parser.add_argument("category", choices=sorted(CATEGORY_DATABASES))
    parser.add_argument("csv_path")
    args = parser.parse_args()

    count = import_parameter_options(args.category, args.csv_path)
    print(f"Imported {count} rows into {CATEGORY_DATABASES[args.category]}")


if __name__ == "__main__":
    main()
