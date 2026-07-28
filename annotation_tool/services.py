"""Application services for PDF preparation, annotation, and dataset export."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any
import uuid

import cv2
import numpy as np
import pdfplumber
from PIL import Image

from floorplan_onnx import get_segmenter
from floorplan_page_pipeline import _write_image, prepare_pdf_page

from .config import resolve_poppler_path
from .storage import AnnotationStore, MaskVersion, ProjectRecord


PREPROCESSING_VERSION = 1
ARTIFACT_NAMES = {
    "render": "page_render.png",
    "cleaned": "cleaned_page.png",
    "model_view": "model_view.png",
    "model_input_512": "model_input_512.png",
    "metadata": "preprocessing.json",
}


class PreparationInProgressError(RuntimeError):
    def __init__(self, project_id: str, page_number: int):
        super().__init__("该页面正在准备，请等待当前任务完成")
        self.project_id = project_id
        self.page_number = page_number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def _json_safe_project(project: dict) -> dict:
    return {
        key: value
        for key, value in project.items()
        if key
        in {
            "schema_version",
            "project_id",
            "name",
            "source_sha256",
            "source_original_name",
            "page_count",
            "pages",
            "created_at",
            "updated_at",
        }
    }


class AnnotationService:
    def __init__(
        self,
        store: AnnotationStore,
        *,
        poppler_path: str | None = None,
        segmenter_factory=None,
    ):
        self.store = store
        self.poppler_path = resolve_poppler_path(poppler_path)
        self.segmenter_factory = segmenter_factory
        self._preparing_lock = threading.Lock()
        self._preparing_pages: set[tuple[str, int]] = set()

    def _segmenter(self):
        if self.segmenter_factory is not None:
            return self.segmenter_factory()
        return get_segmenter(os.environ.get("ONNX_MODEL_PATH") or None)

    def list_projects(self) -> list[dict]:
        return [_json_safe_project(project) for project in self.store.list_projects()]

    def create_project(self, uploaded_file, name: str) -> dict:
        if uploaded_file is None or not getattr(uploaded_file, "filename", ""):
            raise ValueError("pdf upload is required")
        original_name = Path(uploaded_file.filename).name
        if not original_name.lower().endswith(".pdf"):
            raise ValueError("uploaded file must be a PDF")

        runtime_dir = self.store.root / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime_dir) as temporary_dir:
            safe_name = original_name.replace("\x00", "").strip() or "source.pdf"
            temporary_pdf = Path(temporary_dir) / safe_name
            uploaded_file.save(temporary_pdf)
            try:
                with pdfplumber.open(str(temporary_pdf)) as document:
                    page_count = len(document.pages)
            except Exception as exc:
                raise ValueError("uploaded file is not a readable PDF") from exc
            if page_count < 1:
                raise ValueError("PDF must contain at least one page")
            source = self.store.import_source(temporary_pdf)

        project_record = self.store.create_project(name, source.sha256)
        project = self.store.update_project(
            project_record.project_id,
            page_count=page_count,
            source_original_name=original_name,
        )
        return _json_safe_project(project)

    def _project(self, project_id: str) -> dict:
        return self.store.load_project(project_id)

    def _source_path(self, project: dict) -> Path:
        path = self.store.root / "sources" / f"{project['source_sha256']}.pdf"
        if not path.is_file():
            raise FileNotFoundError("Source PDF does not exist")
        return path

    def _page_dir(self, project_id: str, page_number: int) -> Path:
        if page_number < 1:
            raise ValueError("page_number must be positive")
        project_dir = self.store.project_directory(project_id)
        page_dir = (project_dir / "pages" / str(page_number)).resolve()
        page_dir.relative_to(project_dir)
        return page_dir

    @staticmethod
    def _validate_page(project: dict, page_number: int) -> None:
        page_count = project.get("page_count")
        if page_number < 1 or (page_count is not None and page_number > page_count):
            raise ValueError(f"page_number must be between 1 and {page_count}")

    def list_pages(self, project_id: str) -> list[dict]:
        project = self._project(project_id)
        page_count = int(project.get("page_count") or 0)
        stored_pages = project.get("pages", {})
        result = []
        for page_number in range(1, page_count + 1):
            page = dict(
                stored_pages.get(
                    str(page_number),
                    {
                        "page_number": page_number,
                        "status": "unprepared",
                        "current_version": None,
                    },
                )
            )
            result.append(page)
        return result

    def prepare_page(self, project_id: str, page_number: int) -> dict:
        key = (project_id, page_number)
        with self._preparing_lock:
            if key in self._preparing_pages:
                raise PreparationInProgressError(project_id, page_number)
            self._preparing_pages.add(key)
        try:
            project = self._project(project_id)
            self._validate_page(project, page_number)
            existing_page = project.get("pages", {}).get(str(page_number), {})
            if existing_page.get("current_version"):
                raise ValueError(
                    "page has an annotation; create a new project before re-preparing it"
                )
            return self._prepare_page_once(project_id, page_number, project)
        finally:
            with self._preparing_lock:
                self._preparing_pages.discard(key)

    def _prepare_page_once(
        self,
        project_id: str,
        page_number: int,
        project: dict,
    ) -> dict:
        if not self.poppler_path:
            raise ValueError(
                "未找到 Poppler。请设置 POPPLER_PATH，"
                "或使用 start_annotation_tool.ps1 启动标注工具。"
            )
        output_dir = self._page_dir(project_id, page_number)
        segmenter = self._segmenter()
        prepared = prepare_pdf_page(
            self._source_path(project),
            page_number,
            poppler_path=self.poppler_path,
            segmenter=segmenter,
            output_dir=output_dir,
        )
        artifacts = {}
        for key, payload in prepared.artifacts.items():
            artifact_path = output_dir / payload["path"]
            artifacts[key] = {
                "path": str(artifact_path.relative_to(self.store.root)),
                "sha256": payload["sha256"],
            }
        preparation = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "page_number": page_number,
            "page_count": prepared.page_count,
            "image_size": [prepared.render_bgr.shape[1], prepared.render_bgr.shape[0]],
            "model_input": prepared.model_input_metadata,
            "scale_calibration": prepared.scale_calibration,
            "vector_analysis": prepared.vector_analysis,
            "vector_cleanup": prepared.vector_cleanup,
            "inference_roi": prepared.inference_roi,
            "artifacts": artifacts,
        }
        project = self.store.save_page_manifest(project_id, page_number, preparation)
        return project["pages"][str(page_number)]

    def artifact_path(self, project_id: str, page_number: int, name: str) -> Path:
        if name not in ARTIFACT_NAMES:
            raise ValueError("artifact name is invalid")
        project = self._project(project_id)
        self._validate_page(project, page_number)
        page = project.get("pages", {}).get(str(page_number), {})
        artifact = page.get("preparation", {}).get("artifacts", {}).get(name)
        if artifact is None:
            raise FileNotFoundError("Artifact does not exist")
        path = (self.store.root / artifact["path"]).resolve()
        page_dir = self._page_dir(project_id, page_number)
        try:
            path.relative_to(page_dir)
        except ValueError as exc:
            raise ValueError("artifact path is invalid") from exc
        if not path.is_file():
            raise FileNotFoundError("Artifact does not exist")
        return path

    def _preparation(self, project_id: str, page_number: int) -> dict:
        project = self._project(project_id)
        self._validate_page(project, page_number)
        preparation = (
            project.get("pages", {}).get(str(page_number), {}).get("preparation")
        )
        if not preparation:
            raise ValueError("page must be prepared before annotation")
        return preparation

    def _region_dir(
        self,
        project_id: str,
        page_number: int,
        region_id: str,
    ) -> Path:
        region_id = self.store._validated_region_id(region_id)
        page_dir = self._page_dir(project_id, page_number)
        path = (page_dir / "regions" / region_id).resolve()
        path.relative_to(page_dir)
        return path

    def _region_preparation(
        self,
        project_id: str,
        page_number: int,
        region_id: str,
    ) -> dict:
        region_id = self.store._validated_region_id(region_id)
        project = self._project(project_id)
        self._validate_page(project, page_number)
        region = (
            project.get("pages", {})
            .get(str(page_number), {})
            .get("regions", {})
            .get(region_id)
        )
        if region is None:
            raise FileNotFoundError("Annotation region does not exist")
        preparation = region.get("preparation")
        if not preparation:
            raise ValueError("region preparation is missing")
        return preparation

    @staticmethod
    def _letterbox_image(image: np.ndarray) -> tuple[np.ndarray, dict]:
        height, width = image.shape[:2]
        resize_scale = 512 / max(width, height)
        resized_width = max(1, round(width * resize_scale))
        resized_height = max(1, round(height * resize_scale))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        top = (512 - resized_height) // 2
        left = (512 - resized_width) // 2
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        metadata = {
            "original_size": [width, height],
            "resized_size": [resized_width, resized_height],
            "padding": [
                top,
                left,
                512 - resized_height - top,
                512 - resized_width - left,
            ],
            "resize_scale": float(resize_scale),
        }
        return canvas, metadata

    def create_region(
        self,
        project_id: str,
        page_number: int,
        name: str,
        crop_bbox_px,
    ) -> dict:
        page_preparation = self._preparation(project_id, page_number)
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
        page_width, page_height = (
            int(value)
            for value in page_preparation["model_input"]["original_size"]
        )
        crop_width = x1 - x0
        crop_height = y1 - y0
        if (
            x0 < 0
            or y0 < 0
            or x1 > page_width
            or y1 > page_height
            or crop_width < 128
            or crop_height < 128
            or crop_width * crop_height < page_width * page_height * 0.01
        ):
            raise ValueError(
                "crop_bbox_px must be inside the page and at least 128x128 pixels"
            )

        region_id = uuid.uuid4().hex[:12]
        page_dir = self._page_dir(project_id, page_number)
        regions_dir = page_dir / "regions"
        regions_dir.mkdir(parents=True, exist_ok=True)
        final_dir = self._region_dir(project_id, page_number, region_id)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{region_id}-", dir=regions_dir)
        )
        try:
            cropped_images = {}
            artifact_filenames = {
                "render": ARTIFACT_NAMES["render"],
                "cleaned": ARTIFACT_NAMES["cleaned"],
                "model_view": ARTIFACT_NAMES["model_view"],
            }
            artifact_hashes = {}
            for artifact_name, filename in artifact_filenames.items():
                source = _read_image(
                    self.artifact_path(project_id, page_number, artifact_name)
                )
                if source is None:
                    raise ValueError(f"{artifact_name} artifact cannot be read")
                cropped = np.ascontiguousarray(source[y0:y1, x0:x1])
                if cropped.shape[:2] != (crop_height, crop_width):
                    raise ValueError("crop_bbox_px produced an invalid image")
                target = temporary_dir / filename
                _write_image(target, cropped)
                cropped_images[artifact_name] = cropped
                artifact_hashes[artifact_name] = _sha256(target)

            model_input, model_input_metadata = self._letterbox_image(
                cropped_images["model_view"]
            )
            model_input_path = temporary_dir / ARTIFACT_NAMES["model_input_512"]
            _write_image(model_input_path, model_input)
            artifact_hashes["model_input_512"] = _sha256(model_input_path)
            metadata = {
                "preprocessing_version": PREPROCESSING_VERSION,
                "source_page_number": page_number,
                "region_id": region_id,
                "region_name": str(name).strip(),
                "crop_bbox_px": [x0, y0, x1, y1],
                "image_size": [crop_width, crop_height],
                "model_input": model_input_metadata,
                "inference_roi": None,
            }
            metadata_path = temporary_dir / ARTIFACT_NAMES["metadata"]
            self.store._atomic_json(metadata_path, metadata)
            artifact_hashes["metadata"] = _sha256(metadata_path)

            os.replace(temporary_dir, final_dir)
            artifacts = {}
            for artifact_name, filename in ARTIFACT_NAMES.items():
                artifact_path = final_dir / filename
                artifacts[artifact_name] = {
                    "path": str(artifact_path.relative_to(self.store.root)),
                    "sha256": artifact_hashes[artifact_name],
                }
            metadata["artifacts"] = artifacts
            try:
                return self.store.create_region(
                    project_id,
                    page_number,
                    name,
                    [x0, y0, x1, y1],
                    region_id=region_id,
                    preparation=metadata,
                )
            except Exception:
                shutil.rmtree(final_dir, ignore_errors=True)
                raise
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

    def list_regions(self, project_id: str, page_number: int) -> list[dict]:
        return self.store.list_regions(project_id, page_number)

    def region_artifact_path(
        self,
        project_id: str,
        page_number: int,
        region_id: str,
        name: str,
    ) -> Path:
        if name not in ARTIFACT_NAMES:
            raise ValueError("artifact name is invalid")
        preparation = self._region_preparation(
            project_id,
            page_number,
            region_id,
        )
        artifact = preparation.get("artifacts", {}).get(name)
        if artifact is None:
            raise FileNotFoundError("Artifact does not exist")
        path = (self.store.root / artifact["path"]).resolve()
        region_dir = self._region_dir(project_id, page_number, region_id)
        try:
            path.relative_to(region_dir)
        except ValueError as exc:
            raise ValueError("artifact path is invalid") from exc
        if not path.is_file():
            raise FileNotFoundError("Artifact does not exist")
        return path

    def save_region_mask(
        self,
        project_id: str,
        page_number: int,
        region_id: str,
        payload: Any,
        *,
        status: str = "draft",
        author: str = "local-user",
    ) -> tuple[MaskVersion, list[str]]:
        preparation = self._region_preparation(
            project_id,
            page_number,
            region_id,
        )
        mask = self.decode_mask(payload)
        width, height = (
            int(value) for value in preparation["model_input"]["original_size"]
        )
        if mask.shape != (height, width):
            raise ValueError(
                f"mask dimensions must be {width}x{height}; "
                f"received {mask.shape[1]}x{mask.shape[0]}"
            )
        model_mask = self._mask_512(mask, preparation["model_input"])
        version = self.store.save_region_mask_version(
            project_id,
            page_number,
            region_id,
            mask,
            status=status,
            author=author,
        )
        model_path = version.mask_path.with_name(
            f"{version.version_id}.mask_512.npy"
        )
        self.store._atomic_npy(model_path, model_mask)
        return version, self._mask_warnings(mask, model_mask)

    @staticmethod
    def decode_mask(payload: Any) -> np.ndarray:
        if isinstance(payload, np.ndarray):
            array = payload
        elif isinstance(payload, (bytes, bytearray)):
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "PNG":
                        raise ValueError("mask file must be PNG")
                    if image.mode in {"P", "L", "I", "I;16"}:
                        array = np.asarray(image).copy()
                    else:
                        colour = np.asarray(image.convert("RGB"))
                        if not (
                            np.array_equal(colour[:, :, 0], colour[:, :, 1])
                            and np.array_equal(colour[:, :, 1], colour[:, :, 2])
                        ):
                            raise ValueError("mask PNG must contain indexed class values")
                        array = colour[:, :, 0]
            except (OSError, SyntaxError) as exc:
                raise ValueError("mask PNG could not be decoded") from exc
        else:
            array = np.asarray(payload)
        if array.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("mask must contain integer class values")
        if array.size and (int(array.min()) < 0 or int(array.max()) > 3):
            raise ValueError("mask values must be within 0..3")
        return array.astype(np.uint8, copy=False)

    @staticmethod
    def _mask_512(mask: np.ndarray, metadata: dict) -> np.ndarray:
        resized_w, resized_h = (int(value) for value in metadata["resized_size"])
        top, left, _, _ = (int(value) for value in metadata["padding"])
        resized = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        model_mask = np.zeros((512, 512), dtype=np.uint8)
        model_mask[top:top + resized_h, left:left + resized_w] = resized
        return model_mask

    @staticmethod
    def _mask_warnings(mask: np.ndarray, model_mask: np.ndarray) -> list[str]:
        warnings = []
        for class_id, class_name in ((1, "wall"), (2, "window"), (3, "door")):
            ratio = float(np.mean(mask == class_id))
            if ratio > 0.5:
                warnings.append(f"{class_name} class covers an unusually large area")
            if np.any(mask == class_id) and not np.any(model_mask == class_id):
                warnings.append(f"{class_name} class disappears at 512 model resolution")
        wall = (model_mask == 1).astype(np.uint8)
        component_count = cv2.connectedComponents(wall)[0] - 1 if np.any(wall) else 0
        if component_count > 100:
            warnings.append("wall annotation has many disconnected fragments or gaps")
        return warnings

    def save_mask(
        self,
        project_id: str,
        page_number: int,
        payload: Any,
        *,
        status: str = "draft",
        author: str = "local-user",
    ) -> tuple[MaskVersion, list[str]]:
        preparation = self._preparation(project_id, page_number)
        mask = self.decode_mask(payload)
        width, height = (int(value) for value in preparation["model_input"]["original_size"])
        if mask.shape != (height, width):
            raise ValueError(
                f"mask dimensions must be {width}x{height}; received {mask.shape[1]}x{mask.shape[0]}"
            )
        model_mask = self._mask_512(mask, preparation["model_input"])
        version = self.store.save_mask_version(
            project_id,
            page_number,
            mask,
            status=status,
            author=author,
        )
        model_path = version.mask_path.with_name(f"{version.version_id}.mask_512.npy")
        self.store._atomic_npy(model_path, model_mask)
        return version, self._mask_warnings(mask, model_mask)

    def current_mask(self, project_id: str, page_number: int):
        return self.store.load_current_mask(project_id, page_number)

    def confirm_page(
        self,
        project_id: str,
        page_number: int,
        *,
        author: str,
    ) -> MaskVersion:
        current = self.store.load_current_mask(project_id, page_number)
        if current is None:
            raise ValueError("page has no mask to confirm")
        mask, _ = current
        version, _ = self.save_mask(
            project_id,
            page_number,
            mask,
            status="confirmed",
            author=author,
        )
        return version

    def preannotate_page(
        self,
        project_id: str,
        page_number: int,
        *,
        author: str,
        allow_blank: bool = False,
    ) -> tuple[MaskVersion, list[str]]:
        preparation = self._preparation(project_id, page_number)
        cleaned_path = self.artifact_path(project_id, page_number, "cleaned")
        cleaned = _read_image(cleaned_path)
        if cleaned is None:
            raise ValueError("prepared page image cannot be read")
        segmenter = self._segmenter()
        inference_failure = None
        try:
            prediction = segmenter.predict(
                cleaned,
                use_preprocessing=True,
                inference_roi=preparation.get("inference_roi"),
                preserve_full_context=True,
            )
            mask = prediction["mask"]
        except Exception as exc:
            inference_failure = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if not allow_blank:
                raise ValueError(
                    f"preannotation failed ({type(exc).__name__}); "
                    "choose a blank mask explicitly if needed"
                ) from exc
            mask = np.zeros(cleaned.shape[:2], dtype=np.uint8)
        version, warnings = self.save_mask(
            project_id,
            page_number,
            mask,
            status="preannotated",
            author=author,
        )
        if inference_failure is not None:
            warnings.insert(
                0,
                "ONNX preannotation failed; an explicit blank mask was created",
            )
        model_path = Path(
            os.environ.get(
                "ONNX_MODEL_PATH",
                Path(__file__).resolve().parents[1] / "models" / "M2_pub_plus_user.onnx",
            )
        )
        evidence = {
            "version_id": version.version_id,
            "inference_status": "failed" if inference_failure else "passed",
            "failure": inference_failure,
            "model_path": model_path.name,
            "model_sha256": _sha256(model_path) if model_path.is_file() else None,
        }
        self.store._atomic_json(
            version.mask_path.with_name(f"{version.version_id}.preannotation.json"),
            evidence,
        )
        return version, warnings

    def export_confirmed(self, project_id: str) -> dict:
        project = self._project(project_id)
        confirmed = self.store.list_confirmed_pages(project_id)
        if not confirmed:
            raise ValueError("at least one confirmed page is required for export")

        validated_samples = []
        for page in confirmed:
            current = self.store.load_current_mask(project_id, page.page_number)
            if current is None:
                raise ValueError("confirmed page mask is missing")
            full_mask, version = current
            if _sha256(version.mask_path) != version.mask_sha256:
                raise ValueError("confirmed mask hash does not match its version record")
            preparation = self._preparation(project_id, page.page_number)
            if preparation.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError("confirmed page preprocessing version is incompatible")
            width, height = (
                int(value) for value in preparation["model_input"]["original_size"]
            )
            if full_mask.shape != (height, width):
                raise ValueError("confirmed full-resolution mask dimensions are invalid")
            if full_mask.size and (
                int(full_mask.min()) < 0 or int(full_mask.max()) > 3
            ):
                raise ValueError("confirmed mask values must be within 0..3")
            image_source = self.artifact_path(
                project_id,
                page.page_number,
                "model_input_512",
            )
            artifact = preparation["artifacts"]["model_input_512"]
            if _sha256(image_source) != artifact.get("sha256"):
                raise ValueError("model input artifact hash does not match its manifest")
            image = _read_image(image_source)
            if image is None or image.shape[:2] != (512, 512):
                raise ValueError("model input artifact must be a readable 512x512 image")
            model_mask_path = version.mask_path.with_name(
                f"{version.version_id}.mask_512.npy"
            )
            if not model_mask_path.is_file():
                raise ValueError("confirmed page is missing its 512 model-space mask")
            model_mask = np.load(model_mask_path, allow_pickle=False)
            if (
                model_mask.shape != (512, 512)
                or model_mask.dtype != np.uint8
                or (
                    model_mask.size
                    and (int(model_mask.min()) < 0 or int(model_mask.max()) > 3)
                )
            ):
                raise ValueError("confirmed 512 mask is invalid")
            expected_model_mask = self._mask_512(
                full_mask,
                preparation["model_input"],
            )
            if not np.array_equal(model_mask, expected_model_mask):
                raise ValueError(
                    "confirmed 512 mask does not match the full-resolution version"
                )
            validated_samples.append(
                (page, version, image_source, model_mask)
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_id = f"{project_id}-{timestamp}"
        export_dir = (self.store.root / "exports" / export_id).resolve()
        export_dir.relative_to(self.store.root / "exports")
        images_dir = export_dir / "images"
        masks_dir = export_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=False)
        masks_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        class_pixels = {str(class_id): 0 for class_id in range(4)}
        for page, version, image_source, mask in validated_samples:
            stem = f"{project_id}-page-{page.page_number}"
            image_target = images_dir / f"{stem}.png"
            mask_target = masks_dir / f"{stem}.npy"
            mask_png_target = masks_dir / f"{stem}.png"
            shutil.copyfile(image_source, image_target)
            self.store._atomic_npy(mask_target, mask)
            _write_image(mask_png_target, mask)
            for class_id in range(4):
                class_pixels[str(class_id)] += int(np.sum(mask == class_id))
            samples.append(
                {
                    "sample_id": stem,
                    "page_number": page.page_number,
                    "status": "confirmed",
                    "version_id": version.version_id,
                    "image": str(image_target.relative_to(export_dir)),
                    "mask": str(mask_target.relative_to(export_dir)),
                    "mask_preview": str(mask_png_target.relative_to(export_dir)),
                    "image_sha256": _sha256(image_target),
                    "mask_sha256": _sha256(mask_target),
                }
            )
        experiment_type = "single_page_overfit" if len(samples) == 1 else "fine_tune"
        manifest = {
            "schema_version": 1,
            "export_id": export_id,
            "project_id": project_id,
            "source_sha256": project["source_sha256"],
            "class_map": {"0": "background", "1": "wall", "2": "window", "3": "door"},
            "experiment_type": experiment_type,
            "samples": samples,
        }
        report = {
            "sample_count": len(samples),
            "confirmed_pages": [sample["page_number"] for sample in samples],
            "class_pixels": class_pixels,
            "experiment_type": experiment_type,
        }
        self.store._atomic_json(export_dir / "manifest.json", manifest)
        self.store._atomic_json(export_dir / "dataset_report.json", report)
        return {
            "export_id": export_id,
            "export_path": str(export_dir),
            "experiment_type": experiment_type,
            "sample_count": len(samples),
        }
