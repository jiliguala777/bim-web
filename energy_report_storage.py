"""Safe, user-scoped filesystem locations for energy reports."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import unicodedata


@dataclass(frozen=True)
class EnergyReportContext:
    owner_username: str
    owner_storage_key: str
    report_number: str
    energy_root: Path
    owner_root: Path
    report_dir: Path


class InvalidReportPath(ValueError):
    """Raised when a user or report value cannot safely name a path."""


class ReportAccessDenied(PermissionError):
    """Raised when a non-admin requests another user's report."""


class StorageIdentityCollision(ReportAccessDenied):
    """Raised when distinct authentication principals share one storage key."""


_READABLE_DISALLOWED = re.compile(r"[^\w.-]+", re.UNICODE)
_MAX_REPORT_NUMBER_UTF8_BYTES = 255
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def normalized_username(username: str) -> str:
    """Return the exact normalization used by the report storage layout."""

    if not isinstance(username, str):
        raise ValueError("username must be a string")
    return unicodedata.normalize("NFKC", username).strip()


def validate_canonical_username(username: str) -> str:
    """Require a non-empty registration identity already in canonical form."""

    normalized = normalized_username(username)
    if not normalized or normalized != username:
        raise ValueError("username must already be normalized and trimmed")
    return username


def user_storage_key(username: str) -> str:
    """Return a readable, stable, filesystem-safe key for *username*."""

    normalized = normalized_username(username)
    readable = _READABLE_DISALLOWED.sub("-", normalized).strip(".-")[:48].strip(".-")
    if not readable:
        readable = "user"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def colliding_user_storage_keys(usernames) -> set[str]:
    """Return keys assigned to more than one supplied authentication principal."""

    seen: set[str] = set()
    collisions: set[str] = set()
    for username in usernames:
        key = user_storage_key(username)
        if key in seen:
            collisions.add(key)
        seen.add(key)
    return collisions


def ensure_unique_user_storage_keys(usernames) -> None:
    """Enforce the one-storage-key-per-authentication-principal invariant."""

    if colliding_user_storage_keys(usernames):
        raise StorageIdentityCollision("distinct principals share a report storage key")


def validate_report_number(report_number: str) -> str:
    """Validate and return a single safe report directory name."""

    if not isinstance(report_number, str):
        raise InvalidReportPath("report number must be a string")
    normalized = unicodedata.normalize("NFKC", report_number)
    if not normalized or normalized != normalized.strip() or normalized in {".", ".."}:
        raise InvalidReportPath("invalid report number")
    if Path(normalized).is_absolute() or "/" in normalized or "\\" in normalized:
        raise InvalidReportPath("report number must be a single path component")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise InvalidReportPath("report number contains a control character")
    if any(char in normalized for char in '<>:"|?*'):
        raise InvalidReportPath("report number contains an invalid filename character")
    if normalized.endswith((".", " ")):
        raise InvalidReportPath("report number has a non-portable trailing character")
    basename = normalized.split(".", 1)[0].upper()
    if basename in _WINDOWS_RESERVED_BASENAMES:
        raise InvalidReportPath("report number uses a reserved device name")
    if len(normalized.encode("utf-8")) > _MAX_REPORT_NUMBER_UTF8_BYTES:
        raise InvalidReportPath("report number is too long")
    return normalized


def resolve_report_owner(
    session_username: str,
    is_admin: bool,
    requested_owner: str | None,
) -> str:
    """Resolve the owner requested by a session, enforcing admin access."""

    if not isinstance(session_username, str) or not session_username.strip():
        raise ReportAccessDenied("a signed-in username is required")
    if requested_owner is None:
        return session_username
    if not isinstance(requested_owner, str):
        raise ReportAccessDenied("requested owner must be a string")
    if not requested_owner.strip():
        return session_username
    if not is_admin and requested_owner != session_username:
        raise ReportAccessDenied("only an administrator may select another owner")
    return requested_owner


def _existing_symlink_components(path: Path) -> list[Path]:
    """Find symlink components without resolving through any of them."""

    found: list[Path] = []
    current = Path(path.anchor) if path.anchor else Path()
    for part in path.parts:
        if path.anchor and part == path.anchor:
            continue
        current = current / part
        try:
            if current.is_symlink():
                found.append(current)
        except OSError:
            # A disappearing component is handled by normal path resolution.
            continue
    return found


def _reject_symlinks(path: Path) -> None:
    if _existing_symlink_components(path):
        raise InvalidReportPath("report path cannot contain symlinks")


def _canonical_directory(path: str | os.PathLike[str], label: str) -> Path:
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise InvalidReportPath(f"{label} is invalid")
    try:
        return directory.resolve(strict=True)
    except OSError as exc:
        raise InvalidReportPath(f"{label} is invalid") from exc


def _fixed_child_component(component: str) -> str:
    if (
        not isinstance(component, str)
        or not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or Path(component).is_absolute()
    ):
        raise InvalidReportPath("report child component is invalid")
    return component


def resolve_report_child_destination(
    report_dir: str | os.PathLike[str],
    filename: str,
) -> Path:
    """Return a fixed direct-child destination suitable for atomic replacement."""

    canonical_report = _canonical_directory(report_dir, "report directory")
    return canonical_report / _fixed_child_component(filename)


def resolve_report_child_file(
    report_dir: str | os.PathLike[str],
    filename: str,
    *,
    required: bool = True,
) -> Path:
    """Return a regular, non-symlink fixed file directly below a report."""

    candidate = resolve_report_child_destination(report_dir, filename)
    if candidate.is_symlink():
        raise InvalidReportPath("report child file is invalid")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        if required:
            raise
        return candidate
    except OSError as exc:
        raise InvalidReportPath("report child file is invalid") from exc
    if resolved.parent != candidate.parent or not resolved.is_file():
        raise InvalidReportPath("report child file is invalid")
    return resolved


def resolve_report_subdirectory(
    report_dir: str | os.PathLike[str],
    *components: str,
    create: bool = False,
) -> Path:
    """Resolve fixed nested directories without following an existing alias."""

    current = _canonical_directory(report_dir, "report directory")
    if not components:
        return current
    for component in components:
        candidate = current / _fixed_child_component(component)
        if candidate.is_symlink():
            raise InvalidReportPath("report child directory is invalid")
        if not candidate.exists():
            if not create:
                raise FileNotFoundError(candidate.name)
            try:
                candidate.mkdir()
            except FileExistsError:
                pass
        if candidate.is_symlink() or not candidate.is_dir():
            raise InvalidReportPath("report child directory is invalid")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise InvalidReportPath("report child directory is invalid") from exc
        if resolved.parent != current:
            raise InvalidReportPath("report child directory is invalid")
        current = resolved
    return current


def resolve_energy_report_context(
    upload_root: str | os.PathLike[str],
    session_username: str,
    is_admin: bool,
    report_number: str,
    requested_owner: str | None = None,
    create: bool = False,
) -> EnergyReportContext:
    """Resolve an energy report directory below the selected user's root.

    Reads never create directories.  Creation is limited to the energy,
    owner, and report directories after all existing path components have
    passed symlink and containment checks.
    """

    owner_username = resolve_report_owner(session_username, is_admin, requested_owner)
    safe_report_number = validate_report_number(report_number)
    root = Path(upload_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.absolute()
    _reject_symlinks(root)
    root_resolved = root.resolve(strict=False)

    energy_root = root / "energy"
    _reject_symlinks(energy_root)
    if create:
        energy_root.mkdir(parents=True, exist_ok=True)
    if energy_root.exists() and not energy_root.is_dir():
        raise InvalidReportPath("energy root is not a directory")
    energy_resolved = energy_root.resolve(strict=False)
    if energy_resolved.parent != root_resolved:
        raise InvalidReportPath("energy root escapes upload root")

    owner_key = user_storage_key(owner_username)
    owner_root = energy_root / owner_key
    _reject_symlinks(owner_root)
    if create:
        owner_root.mkdir(parents=True, exist_ok=True)
    if owner_root.exists() and not owner_root.is_dir():
        raise InvalidReportPath("owner root is not a directory")
    owner_resolved = owner_root.resolve(strict=False)
    if owner_resolved.parent != energy_resolved:
        raise InvalidReportPath("owner root escapes energy root")

    report_dir = owner_root / safe_report_number
    _reject_symlinks(report_dir)
    if create:
        report_dir.mkdir(parents=True, exist_ok=True)
    if report_dir.exists() and not report_dir.is_dir():
        raise InvalidReportPath("report path is not a directory")
    report_resolved = report_dir.resolve(strict=False)
    if report_resolved.parent != owner_resolved:
        raise InvalidReportPath("report path escapes owner root")

    return EnergyReportContext(
        owner_username=owner_username,
        owner_storage_key=owner_key,
        report_number=safe_report_number,
        energy_root=energy_root,
        owner_root=owner_root,
        report_dir=report_dir,
    )
