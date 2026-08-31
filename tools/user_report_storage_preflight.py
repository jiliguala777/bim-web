"""Read-only deployment preflight for user-scoped energy report storage."""

import argparse
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_report_storage import user_storage_key


class StorageLayoutError(RuntimeError):
    """Raised when persisted report ownership and the storage tree disagree."""


def _persisted_owner_keys(db_path: Path) -> set[str]:
    if not db_path.is_file():
        raise StorageLayoutError(f"report database is missing: {db_path}")

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT DISTINCT username FROM reports").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise StorageLayoutError(f"cannot read persisted report owners: {exc}") from exc

    keys = set()
    for (username,) in rows:
        if not isinstance(username, str) or not username.strip():
            raise StorageLayoutError("reports contains an empty username")
        keys.add(user_storage_key(username))
    return keys


def validate_layout(db_path: str | Path, uploads_root: str | Path, *, runtime_root: str | Path | None = None) -> None:
    """Fail closed unless every top-level energy entry is a persisted owner key.

    The function only reads SQLite metadata and directory entries. It never creates,
    moves, deletes, or changes permissions on either supplied path.
    """

    db_path = Path(db_path)
    uploads_root = Path(uploads_root)
    if runtime_root is not None:
        runtime_root = Path(runtime_root)
        if runtime_root.is_symlink() or not runtime_root.is_dir():
            raise StorageLayoutError(f"runtime root is not a real directory: {runtime_root}")
        runtime_root = runtime_root.resolve(strict=True)
        if db_path.is_symlink() or uploads_root.is_symlink():
            raise StorageLayoutError("database and uploads root must not be symlinks")
        if db_path.parent.resolve(strict=True) != runtime_root or uploads_root.parent.resolve(strict=True) != runtime_root:
            raise StorageLayoutError("database and uploads root must be direct runtime-root children")
        if db_path.name != "users.db" or uploads_root.name != "uploads" or not uploads_root.is_dir():
            raise StorageLayoutError("expected direct users.db file and uploads directory")
    owner_keys = _persisted_owner_keys(db_path)
    energy_root = uploads_root / "energy"
    if not energy_root.exists():
        if owner_keys:
            raise StorageLayoutError(f"energy directory is missing despite persisted reports: {energy_root}")
        return
    if not energy_root.is_dir() or energy_root.is_symlink():
        raise StorageLayoutError(f"energy directory is not a real directory: {energy_root}")

    problems = []
    for entry in energy_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            problems.append(f"invalid energy entry: {entry}")
            continue
        if entry.name not in owner_keys:
            problems.append(f"legacy or unknown owner directory: {entry}")
            continue
        for report_dir in entry.iterdir():
            if report_dir.is_symlink() or not report_dir.is_dir():
                problems.append(f"invalid report entry: {report_dir}")
    if problems:
        raise StorageLayoutError("\n".join(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="path to users.db")
    parser.add_argument("--uploads", required=True, type=Path, help="UPLOAD_FOLDER directory")
    parser.add_argument("--runtime-root", required=True, type=Path, help="runtime directory containing users.db and uploads")
    args = parser.parse_args(argv)
    try:
        validate_layout(args.db, args.uploads, runtime_root=args.runtime_root)
    except StorageLayoutError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    print("Preflight passed: every energy owner directory matches a persisted report username.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
