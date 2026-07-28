"""Configuration for the local-only annotation application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil


DEFAULT_DATA_ROOT = Path(r"G:\bim网页\标注数据")


def _contains_poppler(directory: Path) -> bool:
    return any(
        all((directory / executable).is_file() for executable in names)
        for names in (
            ("pdfinfo.exe", "pdftoppm.exe"),
            ("pdfinfo", "pdftoppm"),
        )
    )


def resolve_poppler_path(explicit: str | os.PathLike | None = None) -> str | None:
    configured = os.fspath(explicit).strip() if explicit else ""
    if not configured:
        configured = os.environ.get("POPPLER_PATH", "").strip()

    candidates = []
    if configured:
        candidates.append(Path(configured))

    pdfinfo = shutil.which("pdfinfo")
    pdftoppm = shutil.which("pdftoppm")
    if pdfinfo and pdftoppm:
        pdfinfo_dir = Path(pdfinfo).resolve().parent
        if pdfinfo_dir == Path(pdftoppm).resolve().parent:
            candidates.append(pdfinfo_dir)

    repository = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repository / "tools" / "poppler" / "Library" / "bin",
            repository / "tools" / "poppler" / "bin",
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "native"
            / "poppler"
            / "Library"
            / "bin",
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _contains_poppler(resolved):
            return str(resolved)
    return None


@dataclass(frozen=True)
class AnnotationConfig:
    data_root: Path = DEFAULT_DATA_ROOT

    def __post_init__(self):
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())

    @classmethod
    def from_env(cls) -> "AnnotationConfig":
        configured = os.environ.get("ANNOTATION_DATA_ROOT", "").strip()
        return cls(data_root=Path(configured) if configured else DEFAULT_DATA_ROOT)
