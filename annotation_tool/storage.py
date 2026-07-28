"""Safe external storage and version history for annotation projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from .config import AnnotationConfig


PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
REGION_ID_RE = re.compile(r"^[a-f0-9]{12}$")
VALID_STATUSES = {"draft", "preannotated", "confirmed"}


@dataclass(frozen=True)
class SourceRecord:
    sha256: str
    original_name: str
    stored_path: Path


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    name: str
    source_sha256: str
    manifest_path: Path


@dataclass(frozen=True)
class MaskVersion:
    version_id: str
    status: str
    author: str
    created_at: str
    mask_sha256: str
    mask_path: Path
    previous_version_id: str | None


@dataclass(frozen=True)
class PageRecord:
    page_number: int
    status: str
    version_id: str


@dataclass(frozen=True)
class AnnotationTarget:
    page_number: int
    status: str
    version_id: str
    region_id: str | None = None
    region_name: str | None = None
    crop_bbox_px: tuple[int, int, int, int] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:50] or "project"


class AnnotationStore:
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.root = config.data_root
        for child in ("sources", "projects", "annotations", "exports", "models"):
            (self.root / child).mkdir(parents=True, exist_ok=True)

    def _inside_root(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path must stay inside annotation data root") from exc
        return resolved

    def project_directory(self, project_id: str) -> Path:
        if not PROJECT_ID_RE.fullmatch(project_id):
            raise ValueError("project_id is invalid")
        return self._inside_root(self.root / "projects" / project_id)

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _atomic_npy(path: Path, mask: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, mask, allow_pickle=False)
        os.replace(temporary, path)

    def import_source(self, pdf_path: str | Path) -> SourceRecord:
        source = Path(pdf_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = _sha256_file(source)
        stored = self._inside_root(self.root / "sources" / f"{digest}.pdf")
        metadata_path = self._inside_root(self.root / "sources" / f"{digest}.json")
        if not stored.exists():
            temporary = stored.with_name(stored.name + f".{uuid.uuid4().hex}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, stored)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            original_names = list(metadata.get("original_names", []))
        else:
            original_names = []
        if source.name not in original_names:
            original_names.append(source.name)
            self._atomic_json(
                metadata_path,
                {
                    "sha256": digest,
                    "original_name": original_names[0],
                    "original_names": original_names,
                    "stored_path": stored.name,
                },
            )
        return SourceRecord(
            sha256=digest,
            original_name=source.name,
            stored_path=stored,
        )

    def create_project(self, name: str, source_sha256: str) -> ProjectRecord:
        source_path = self.root / "sources" / f"{source_sha256}.pdf"
        if not source_path.is_file():
            raise FileNotFoundError("Imported source PDF does not exist")
        project_id = f"{_slug(name)}-{source_sha256[:8]}"
        project_dir = self.project_directory(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = project_dir / "project.json"
        if not manifest_path.exists():
            source_metadata_path = self.root / "sources" / f"{source_sha256}.json"
            source_metadata = (
                json.loads(source_metadata_path.read_text(encoding="utf-8"))
                if source_metadata_path.is_file()
                else {}
            )
            created_at = _now()
            self._atomic_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "project_id": project_id,
                    "name": name.strip() or "Project",
                    "source_sha256": source_sha256,
                    "source_original_name": source_metadata.get(
                        "original_name",
                        source_path.name,
                    ),
                    "page_count": None,
                    "pages": {},
                    "created_at": created_at,
                    "updated_at": created_at,
                },
            )
        return ProjectRecord(
            project_id=project_id,
            name=name.strip() or "Project",
            source_sha256=source_sha256,
            manifest_path=manifest_path,
        )

    def save_page_manifest(
        self,
        project_id: str,
        page_number: int,
        manifest: dict,
    ) -> dict:
        if page_number < 1:
            raise ValueError("page_number must be positive")
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a dictionary")
        manifest_path, project = self._load_project(project_id)
        updated_at = _now()
        pages = project.setdefault("pages", {})
        page = pages.setdefault(
            str(page_number),
            {
                "page_number": page_number,
                "status": "unprepared",
                "current_version": None,
            },
        )
        page["preparation"] = manifest
        if page.get("status") == "unprepared":
            page["status"] = "prepared"
        page["updated_at"] = updated_at
        page_count = manifest.get("page_count")
        if page_count is not None:
            if not isinstance(page_count, int) or page_count < page_number:
                raise ValueError("page_count must include page_number")
            project["page_count"] = page_count
        project["updated_at"] = updated_at
        self._atomic_json(manifest_path, project)
        return project

    def _load_project(self, project_id: str) -> tuple[Path, dict]:
        directory = self.project_directory(project_id)
        manifest_path = directory / "project.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Project does not exist")
        return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_project(self, project_id: str) -> dict:
        return self._load_project(project_id)[1]

    def list_projects(self) -> list[dict]:
        projects = []
        projects_root = self.root / "projects"
        for manifest_path in projects_root.glob("*/project.json"):
            try:
                project = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.project_directory(project["project_id"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            projects.append(project)
        return sorted(
            projects,
            key=lambda project: project.get("updated_at", ""),
            reverse=True,
        )

    def update_project(self, project_id: str, **changes) -> dict:
        manifest_path, project = self._load_project(project_id)
        protected = {"project_id", "source_sha256", "created_at"}
        if protected.intersection(changes):
            raise ValueError("immutable project fields cannot be changed")
        project.update(changes)
        project["updated_at"] = _now()
        self._atomic_json(manifest_path, project)
        return project

    @staticmethod
    def _validated_region_id(region_id: str) -> str:
        value = str(region_id)
        if not REGION_ID_RE.fullmatch(value):
            raise ValueError("region_id is invalid")
        return value

    def create_region(
        self,
        project_id: str,
        page_number: int,
        name: str,
        crop_bbox_px,
        *,
        region_id: str | None = None,
        preparation: dict | None = None,
    ) -> dict:
        if page_number < 1:
            raise ValueError("page_number must be positive")
        region_name = str(name).strip()
        if not 1 <= len(region_name) <= 80:
            raise ValueError("region name must contain 1..80 characters")
        if (
            not isinstance(crop_bbox_px, (list, tuple))
            or len(crop_bbox_px) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in crop_bbox_px
            )
        ):
            raise ValueError("crop_bbox_px must contain four integers")
        x0, y0, x1, y1 = (int(value) for value in crop_bbox_px)
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("crop_bbox_px must be a positive normalized rectangle")

        manifest_path, project = self._load_project(project_id)
        page = project.get("pages", {}).get(str(page_number))
        if not page or "preparation" not in page:
            raise ValueError("page must be prepared before creating a region")
        regions = page.setdefault("regions", {})
        if any(payload.get("name") == region_name for payload in regions.values()):
            raise ValueError("region name must be unique within the page")
        region_id = (
            self._validated_region_id(region_id)
            if region_id is not None
            else uuid.uuid4().hex[:12]
        )
        while region_id in regions:
            if preparation is not None:
                raise ValueError("region_id already exists")
            region_id = uuid.uuid4().hex[:12]
        created_at = _now()
        region = {
            "region_id": region_id,
            "name": region_name,
            "source_page_number": page_number,
            "crop_bbox_px": [x0, y0, x1, y1],
            "status": "prepared",
            "current_version": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        if preparation is not None:
            region["preparation"] = preparation
        regions[region_id] = region
        page["updated_at"] = created_at
        project["updated_at"] = created_at
        self._atomic_json(manifest_path, project)
        return dict(region)

    def list_regions(self, project_id: str, page_number: int) -> list[dict]:
        if page_number < 1:
            raise ValueError("page_number must be positive")
        _, project = self._load_project(project_id)
        page = project.get("pages", {}).get(str(page_number))
        if not page:
            return []
        return [
            dict(region)
            for region in sorted(
                page.get("regions", {}).values(),
                key=lambda payload: payload.get("created_at", ""),
            )
        ]

    def save_region_mask_version(
        self,
        project_id: str,
        page_number: int,
        region_id: str,
        mask: np.ndarray,
        *,
        status: str,
        author: str,
    ) -> MaskVersion:
        region_id = self._validated_region_id(region_id)
        if page_number < 1:
            raise ValueError("page_number must be positive")
        if status not in VALID_STATUSES:
            raise ValueError("status is invalid")
        array = np.asarray(mask)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError("mask must be a two-dimensional uint8 array")
        values = np.unique(array)
        if np.any((values < 0) | (values > 3)):
            raise ValueError("mask values must be within 0..3")

        manifest_path, project = self._load_project(project_id)
        page = project.get("pages", {}).get(str(page_number))
        region = (page or {}).get("regions", {}).get(region_id)
        if region is None:
            raise FileNotFoundError("Annotation region does not exist")
        previous = region.get("current_version")
        version_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        versions_dir = self._inside_root(
            self.root
            / "annotations"
            / project_id
            / f"page-{page_number}"
            / "regions"
            / region_id
            / "versions"
        )
        mask_path = versions_dir / f"{version_id}.npy"
        self._atomic_npy(mask_path, array)
        created_at = _now()
        version = MaskVersion(
            version_id=version_id,
            status=status,
            author=author.strip() or "unknown",
            created_at=created_at,
            mask_sha256=_sha256_file(mask_path),
            mask_path=mask_path,
            previous_version_id=previous,
        )
        payload = asdict(version)
        payload["mask_path"] = str(mask_path.relative_to(self.root))
        self._atomic_json(versions_dir / f"{version_id}.json", payload)

        region.update(
            {
                "status": status,
                "current_version": version_id,
                "updated_at": created_at,
            }
        )
        page["updated_at"] = created_at
        project["updated_at"] = created_at
        self._atomic_json(manifest_path, project)
        return version

    def load_current_region_mask(
        self,
        project_id: str,
        page_number: int,
        region_id: str,
    ) -> tuple[np.ndarray, MaskVersion] | None:
        region_id = self._validated_region_id(region_id)
        _, project = self._load_project(project_id)
        page = project.get("pages", {}).get(str(page_number))
        region = (page or {}).get("regions", {}).get(region_id)
        if region is None:
            raise FileNotFoundError("Annotation region does not exist")
        version_id = region.get("current_version")
        if not version_id:
            return None
        version_path = (
            self.root
            / "annotations"
            / project_id
            / f"page-{page_number}"
            / "regions"
            / region_id
            / "versions"
            / f"{version_id}.json"
        )
        payload = json.loads(version_path.read_text(encoding="utf-8"))
        mask_path = self._inside_root(self.root / payload["mask_path"])
        version = MaskVersion(
            version_id=payload["version_id"],
            status=payload["status"],
            author=payload["author"],
            created_at=payload["created_at"],
            mask_sha256=payload["mask_sha256"],
            mask_path=mask_path,
            previous_version_id=payload.get("previous_version_id"),
        )
        return np.load(mask_path, allow_pickle=False), version

    def save_mask_version(
        self,
        project_id: str,
        page_number: int,
        mask: np.ndarray,
        *,
        status: str,
        author: str,
    ) -> MaskVersion:
        if page_number < 1:
            raise ValueError("page_number must be positive")
        if status not in VALID_STATUSES:
            raise ValueError("status is invalid")
        array = np.asarray(mask)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError("mask must be a two-dimensional uint8 array")
        values = np.unique(array)
        if np.any((values < 0) | (values > 3)):
            raise ValueError("mask values must be within 0..3")

        manifest_path, project = self._load_project(project_id)
        page_key = str(page_number)
        previous = project.get("pages", {}).get(page_key, {}).get("current_version")
        version_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        versions_dir = self._inside_root(
            self.root / "annotations" / project_id / f"page-{page_number}" / "versions"
        )
        mask_path = versions_dir / f"{version_id}.npy"
        self._atomic_npy(mask_path, array)
        created_at = _now()
        version = MaskVersion(
            version_id=version_id,
            status=status,
            author=author.strip() or "unknown",
            created_at=created_at,
            mask_sha256=_sha256_file(mask_path),
            mask_path=mask_path,
            previous_version_id=previous,
        )
        version_path = versions_dir / f"{version_id}.json"
        payload = asdict(version)
        payload["mask_path"] = str(mask_path.relative_to(self.root))
        self._atomic_json(version_path, payload)

        pages = project.setdefault("pages", {})
        page = pages.setdefault(page_key, {"page_number": page_number})
        page.update(
            {
                "status": status,
                "current_version": version_id,
                "updated_at": created_at,
            }
        )
        project["updated_at"] = created_at
        self._atomic_json(manifest_path, project)
        return version

    def load_current_mask(
        self,
        project_id: str,
        page_number: int,
    ) -> tuple[np.ndarray, MaskVersion] | None:
        _, project = self._load_project(project_id)
        page = project.get("pages", {}).get(str(page_number))
        if not page:
            return None
        version_id = page["current_version"]
        version_path = (
            self.root
            / "annotations"
            / project_id
            / f"page-{page_number}"
            / "versions"
            / f"{version_id}.json"
        )
        payload = json.loads(version_path.read_text(encoding="utf-8"))
        mask_path = self._inside_root(self.root / payload["mask_path"])
        version = MaskVersion(
            version_id=payload["version_id"],
            status=payload["status"],
            author=payload["author"],
            created_at=payload["created_at"],
            mask_sha256=payload["mask_sha256"],
            mask_path=mask_path,
            previous_version_id=payload.get("previous_version_id"),
        )
        return np.load(mask_path, allow_pickle=False), version

    def list_confirmed_pages(self, project_id: str) -> list[PageRecord]:
        _, project = self._load_project(project_id)
        confirmed = []
        for payload in project.get("pages", {}).values():
            if payload.get("status") == "confirmed":
                confirmed.append(
                    PageRecord(
                        page_number=int(payload["page_number"]),
                        status="confirmed",
                        version_id=payload["current_version"],
                    )
                )
        return sorted(confirmed, key=lambda item: item.page_number)

    def list_confirmed_targets(self, project_id: str) -> list[AnnotationTarget]:
        _, project = self._load_project(project_id)
        confirmed = []
        for payload in project.get("pages", {}).values():
            page_number = int(payload["page_number"])
            if payload.get("status") == "confirmed":
                confirmed.append(
                    AnnotationTarget(
                        page_number=page_number,
                        status="confirmed",
                        version_id=payload["current_version"],
                    )
                )
            for region in payload.get("regions", {}).values():
                if region.get("status") != "confirmed":
                    continue
                confirmed.append(
                    AnnotationTarget(
                        page_number=page_number,
                        status="confirmed",
                        version_id=region["current_version"],
                        region_id=region["region_id"],
                        region_name=region["name"],
                        crop_bbox_px=tuple(region["crop_bbox_px"]),
                    )
                )
        return sorted(
            confirmed,
            key=lambda item: (item.page_number, item.region_id or ""),
        )
