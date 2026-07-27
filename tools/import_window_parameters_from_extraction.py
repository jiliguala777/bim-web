from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "windows_parameter_priority_100_extraction" / "window_parameters.csv"
DEFAULT_DB = BASE_DIR / "data" / "envelope_databases" / "windows.db"
SUPPORTED_TYPES = {"window_u", "window_shgc"}


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parameter_options (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            notes TEXT,
            source TEXT DEFAULT 'sample',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, name)
        )
        """
    )


def unique_display_names(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    base_counts = Counter(clean(row.get("sample_name")) or clean(row.get("report_id")) for row in rows)
    result: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        base_name = clean(row.get("sample_name")) or clean(row.get("report_id"))
        if base_counts[base_name] > 1:
            report_id = clean(row.get("report_id")).split("__", 1)[0]
            row["display_name"] = f"{base_name}（{report_id}）"
        else:
            row["display_name"] = base_name
        result.append(row)
    return result


def import_rows(args: argparse.Namespace) -> tuple[int, int]:
    rows = [
        row
        for row in read_csv(Path(args.input))
        if clean(row.get("parameter_type")) in SUPPORTED_TYPES
    ]
    rows = unique_display_names(rows)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        ensure_table(conn)
        if args.replace_source:
            conn.execute("DELETE FROM parameter_options WHERE source = ?", (args.source,))

        imported = 0
        for row in rows:
            category = clean(row.get("parameter_type"))
            name = clean(row.get("display_name"))
            value = float(clean(row.get("value")))
            unit = clean(row.get("unit"))
            notes = (
                f"报告 {clean(row.get('report_id'))}; "
                f"{clean(row.get('source_field'))}: {clean(row.get('source_text'))}"
            )
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
                (category, name, value, unit, notes, args.source),
            )
            imported += 1
        conn.commit()
        return imported, len(rows)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import extracted window parameters into windows.db parameter_options.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--source", default="structured_windows")
    parser.add_argument("--replace-source", action="store_true")
    args = parser.parse_args()
    imported, supported = import_rows(args)
    print(f"supported_rows={supported}")
    print(f"imported={imported}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
