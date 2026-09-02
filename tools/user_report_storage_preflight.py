"""Read-only deployment preflight for user-scoped energy report storage."""

import argparse
import os
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_report_storage import (
    ensure_unique_user_storage_keys,
    InvalidReportPath,
    StorageIdentityCollision,
    user_storage_key,
    validate_report_number,
)


class StorageLayoutError(RuntimeError):
    """Raised when persisted report ownership and the storage tree disagree."""


def _persisted_layout(db_path: Path) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    if not db_path.is_file():
        raise StorageLayoutError(f"report database is missing: {db_path}")

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            users = connection.execute("SELECT username FROM users").fetchall()
            reports = connection.execute(
                "SELECT username, report_number FROM reports"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise StorageLayoutError(f"cannot read persisted report owners: {exc}") from exc

    principals = []
    for (username,) in users:
        if not isinstance(username, str) or not username.strip():
            raise StorageLayoutError("users contains an empty username")
        principals.append(username)

    if os.environ.get("ADMIN_PASSWORD"):
        admin_username = os.environ.get("ADMIN_USER", "admin")
        if not isinstance(admin_username, str) or not admin_username.strip():
            raise StorageLayoutError("environment administrator identity is invalid")
        principals.append(admin_username)

    try:
        ensure_unique_user_storage_keys(principals)
    except (StorageIdentityCollision, ValueError) as exc:
        raise StorageLayoutError("identity storage-key collision detected") from exc

    principal_by_name = {username: user_storage_key(username) for username in principals}
    expected_by_owner: dict[str, set[str]] = {}
    expected_pairs: set[tuple[str, str]] = set()
    for username, report_number in reports:
        if not isinstance(username, str) or not username.strip():
            raise StorageLayoutError("reports contains an empty username")
        owner_key = principal_by_name.get(username)
        if owner_key is None:
            raise StorageLayoutError("report owner is not an authentication principal")
        try:
            safe_report_number = validate_report_number(report_number)
        except InvalidReportPath as exc:
            raise StorageLayoutError("reports contains an invalid report number") from exc
        expected_by_owner.setdefault(owner_key, set()).add(safe_report_number)
        expected_pairs.add((owner_key, safe_report_number))
    return expected_by_owner, expected_pairs


def validate_layout(db_path: str | Path, uploads_root: str | Path, *, runtime_root: str | Path) -> None:
    """Fail closed unless every top-level energy entry is a persisted owner key.

    The function only reads SQLite metadata and directory entries. It never creates,
    moves, deletes, or changes permissions on either supplied path.
    """

    db_path = Path(db_path)
    uploads_root = Path(uploads_root)
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
    expected_by_owner, expected_pairs = _persisted_layout(db_path)
    energy_root = uploads_root / "energy"
    if not energy_root.exists():
        if expected_pairs:
            raise StorageLayoutError("energy directory is missing despite persisted reports")
        return
    if not energy_root.is_dir() or energy_root.is_symlink():
        raise StorageLayoutError(f"energy directory is not a real directory: {energy_root}")

    problems = []
    actual_pairs = set()
    for entry in energy_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            problems.append(f"invalid energy entry: {entry}")
            continue
        if entry.name not in expected_by_owner:
            problems.append(f"legacy or unknown owner directory: {entry}")
            continue
        for report_dir in entry.iterdir():
            if report_dir.is_symlink() or not report_dir.is_dir():
                problems.append(f"invalid report entry: {report_dir}")
                continue
            actual_pairs.add((entry.name, report_dir.name))
            if report_dir.name not in expected_by_owner[entry.name]:
                problems.append("storage owner/report mapping does not match persisted identities")
    if actual_pairs != expected_pairs:
        problems.append("storage owner/report mapping does not match persisted identities")
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
