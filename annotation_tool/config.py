"""Configuration for the local-only annotation application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(r"G:\bim网页\标注数据")


@dataclass(frozen=True)
class AnnotationConfig:
    data_root: Path = DEFAULT_DATA_ROOT

    def __post_init__(self):
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())

    @classmethod
    def from_env(cls) -> "AnnotationConfig":
        configured = os.environ.get("ANNOTATION_DATA_ROOT", "").strip()
        return cls(data_root=Path(configured) if configured else DEFAULT_DATA_ROOT)
