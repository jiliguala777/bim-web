"""Deterministic preparation of one PDF page for floorplan recognition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path

from floorplan_ocr import extract_numeric_text_spans
from vector_pdf_scale import (
    build_dimension_annotation_mask,
    build_nonstructural_vector_mask,
    build_structural_vector_mask,
    calibrate_from_overall_dimensions,
    detect_building_roi,
    detect_dimension_candidates,
    extract_vector_page,
    remove_dimension_annotations,
)


DEFAULT_VECTOR_TIMEOUT_SECONDS = 15.0
VECTOR_WORKER_SCRIPT = Path(__file__).with_name("floorplan_vector_worker.py")
PROCESS_TERMINATE_TIMEOUT_SECONDS = 1.0
PROCESS_KILL_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class VectorAnalysisOutcome:
    page_data: dict[str, Any] | None
    evidence: dict[str, Any]


def _failure_outcome(timeout_seconds: float, reason: str) -> VectorAnalysisOutcome:
    return VectorAnalysisOutcome(
        None,
        {
            "status": "failed",
            "mode": "raster_fallback",
            "timeout_seconds": timeout_seconds,
            "reason": reason,
        },
    )


def _terminate_and_reap(
    process,
    *,
    terminate_timeout_seconds: float = PROCESS_TERMINATE_TIMEOUT_SECONDS,
    kill_timeout_seconds: float = PROCESS_KILL_TIMEOUT_SECONDS,
) -> None:
    """Terminate a worker, escalating to kill without an unbounded wait."""
    process.terminate()
    try:
        process.wait(timeout=terminate_timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        process.kill()
    try:
        process.wait(timeout=kill_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("vector extraction worker could not be reaped") from exc


def _close_process_handle(process) -> None:
    """Explicitly close the Windows process handle after wait/reaping."""
    handle = getattr(process, "_handle", None)
    close = getattr(handle, "Close", None)
    if callable(close):
        close()


def _run_vector_analysis(
    source: Path,
    page_index: int,
    dpi: int,
    timeout_seconds: float,
    *,
    worker_script: str | Path | None = None,
) -> VectorAnalysisOutcome:
    if timeout_seconds <= 0:
        raise ValueError("vector_timeout_seconds must be positive")
    script = Path(worker_script or VECTOR_WORKER_SCRIPT).resolve()
    with tempfile.TemporaryDirectory(prefix="floorplan-vector-") as directory:
        result_path = Path(directory) / "result.json"
        process = None
        cleanup_attempted = False
        try:
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(script),
                        str(result_path),
                        str(Path(source).resolve()),
                        str(page_index),
                        str(dpi),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
            except OSError as exc:
                return _failure_outcome(
                    timeout_seconds,
                    f"vector extraction worker could not start: {exc}",
                )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                cleanup_attempted = True
                _terminate_and_reap(process)
                return VectorAnalysisOutcome(
                    None,
                    {
                        "status": "timed_out",
                        "mode": "raster_fallback",
                        "timeout_seconds": timeout_seconds,
                        "reason": (
                            f"vector extraction exceeded {timeout_seconds:g} seconds"
                        ),
                    },
                )
            if not result_path.is_file():
                return _failure_outcome(
                    timeout_seconds,
                    (
                        "vector extraction worker exited without a result "
                        f"(exit code {process.returncode})"
                    ),
                )
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                return _failure_outcome(
                    timeout_seconds,
                    f"invalid vector worker result: {exc}",
                )
            if not isinstance(payload, dict) or type(payload.get("ok")) is not bool:
                return _failure_outcome(
                    timeout_seconds,
                    "invalid vector worker result: payload schema is invalid",
                )
            if payload["ok"]:
                if not isinstance(payload.get("page_data"), dict):
                    return _failure_outcome(
                        timeout_seconds,
                        "invalid vector worker result: page_data is invalid",
                    )
                return VectorAnalysisOutcome(
                    payload["page_data"],
                    {
                        "status": "completed",
                        "mode": "vector",
                        "timeout_seconds": timeout_seconds,
                        "reason": None,
                    },
                )
            error = payload.get("error")
            if not isinstance(error, str) or not error:
                return _failure_outcome(
                    timeout_seconds,
                    "invalid vector worker result: error is invalid",
                )
            return _failure_outcome(timeout_seconds, error)
        finally:
            if process is not None:
                try:
                    if process.returncode is None and not cleanup_attempted:
                        _terminate_and_reap(process)
                finally:
                    _close_process_handle(process)


@dataclass(frozen=True)
class PreparedFloorplanPage:
    page_number: int
    page_count: int
    render_bgr: np.ndarray
    cleaned_bgr: np.ndarray
    model_view_rgb: np.ndarray
    model_input_512: np.ndarray
    model_input_metadata: dict[str, Any]
    vector_analysis: dict[str, Any]
    scale_calibration: dict[str, Any]
    vector_cleanup: dict[str, Any]
    inference_roi: list[int] | None
    cleanup_mask: np.ndarray
    structural_support_mask: np.ndarray | None
    artifacts: dict[str, dict[str, str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_image(
    path: Path,
    image: np.ndarray,
    parameters: list[int] | None = None,
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    if not extension:
        raise OSError(f"Cannot write image artifact without an extension: {path.name}")
    try:
        if parameters is None:
            encoded, buffer = cv2.imencode(extension, image)
        else:
            encoded, buffer = cv2.imencode(extension, image, parameters)
    except cv2.error as exc:
        raise OSError(f"Cannot write image artifact: {path.name}") from exc
    if not encoded:
        raise OSError(f"Cannot write image artifact: {path.name}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(buffer.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return {"path": path.name, "sha256": _sha256(path)}


def _letterbox_preview(model_view_rgb: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    resized_w, resized_h = metadata["resized_size"]
    top, left, _, _ = metadata["padding"]
    preview = np.zeros((512, 512, 3), dtype=np.uint8)
    resized = cv2.resize(model_view_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    preview[top:top + resized_h, left:left + resized_w] = resized
    return preview


def _default_scale_calibration(method: str) -> dict[str, Any]:
    return {
        "status": "manual_required",
        "method": method,
        "scale_m_per_px": None,
        "confidence": 0.0,
        "horizontal": None,
        "vertical": None,
        "axis_difference_percent": None,
        "evidence": [],
    }


def prepare_pdf_page(
    pdf_path: str | Path,
    page_number: int,
    *,
    poppler_path: str | None,
    segmenter,
    output_dir: str | Path | None = None,
    vector_timeout_seconds: float = DEFAULT_VECTOR_TIMEOUT_SECONDS,
) -> PreparedFloorplanPage:
    """Prepare one 1-based PDF page with the same inputs used by recognition."""
    source = Path(pdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with pdfplumber.open(str(source)) as document:
        page_count = len(document.pages)
    if page_number < 1 or page_number > page_count:
        raise ValueError(f"page_number must be between 1 and {page_count}")

    vector_dpi = 100
    vector_outcome = _run_vector_analysis(
        source,
        page_index=page_number - 1,
        dpi=vector_dpi,
        timeout_seconds=vector_timeout_seconds,
    )
    page_data = vector_outcome.page_data
    using_raster_fallback = page_data is None
    render_dpi = vector_dpi if page_data is None or page_data["is_vector_pdf"] else 200
    rendered = convert_from_path(
        str(source),
        dpi=render_dpi,
        first_page=page_number,
        last_page=page_number,
        poppler_path=poppler_path,
    )
    if len(rendered) != 1:
        raise ValueError("PDF renderer did not return exactly one selected page")
    render_bgr = cv2.cvtColor(np.asarray(rendered[0]), cv2.COLOR_RGB2BGR)
    if page_data is None:
        page_data = {
            "page_size_pt": [
                render_bgr.shape[1] * 72.0 / render_dpi,
                render_bgr.shape[0] * 72.0 / render_dpi,
            ],
            "render_size_px": [render_bgr.shape[1], render_bgr.shape[0]],
            "dpi": render_dpi,
            "text_spans": [],
            "segments": [],
            "styled_edges": [],
            "vector_text_count": 0,
            "vector_segment_count": 0,
            "styled_edge_count": 0,
            "has_vector_text": False,
            "has_vector_geometry": False,
            "is_vector_pdf": False,
        }
    page_data["page_count"] = page_count
    page_data["render_size_px"] = [render_bgr.shape[1], render_bgr.shape[0]]

    scale_calibration = _default_scale_calibration(
        "raster_fallback_manual"
        if using_raster_fallback
        else (
            "vector_pdf_overall_dimensions"
            if page_data["is_vector_pdf"]
            else "scanned_pdf_manual"
        )
    )
    dimension_candidates: list[dict[str, Any]] = []
    cleanup_candidates: list[dict[str, Any]] = []
    ocr_evidence = {
        "status": "not_needed",
        "candidate_count": 0,
        "accepted_count": 0,
    }
    if page_data["is_vector_pdf"] and not page_data.get("has_vector_text"):
        try:
            ocr_spans, ocr_evidence = extract_numeric_text_spans(
                render_bgr,
                page_data["page_size_pt"],
            )
            page_data["text_spans"] = ocr_spans
        except Exception as exc:
            ocr_evidence = {
                "status": "failed",
                "candidate_count": 0,
                "accepted_count": 0,
                "reason": str(exc),
            }
    if page_data["is_vector_pdf"]:
        dimension_candidates = detect_dimension_candidates(page_data)
        scale_calibration = calibrate_from_overall_dimensions(page_data)
        if page_data.get("has_vector_text"):
            cleanup_candidates = dimension_candidates
        scale_calibration["text_source"] = (
            "pdf_text"
            if page_data.get("has_vector_text")
            else "rapidocr"
            if ocr_evidence.get("accepted_count")
            else "none"
        )

    cleanup_mask = np.zeros(render_bgr.shape[:2], dtype=np.uint8)
    if cleanup_candidates:
        cleanup_mask = build_dimension_annotation_mask(page_data, cleanup_candidates)

    structural_support_mask = None
    inference_roi = None
    vector_cleanup = {
        "enabled": False,
        "has_vector_geometry": bool(page_data.get("has_vector_geometry")),
        "has_vector_text": bool(page_data.get("has_vector_text")),
        "ocr": ocr_evidence,
        "nonstructural_mask": {"enabled": False},
        "structural_mask": {"enabled": False},
        "building_roi": {"enabled": False, "bbox_px": None},
    }
    if page_data.get("has_vector_geometry"):
        nonstructural_mask, nonstructural_evidence = build_nonstructural_vector_mask(page_data)
        structural_support_mask, structural_evidence = build_structural_vector_mask(page_data)
        building_roi = detect_building_roi(page_data)
        if nonstructural_evidence.get("enabled"):
            cleanup_mask = cv2.bitwise_or(cleanup_mask, nonstructural_mask)
        if building_roi.get("enabled"):
            inference_roi = [int(value) for value in building_roi["bbox_px"]]
        vector_cleanup = {
            "enabled": bool(
                nonstructural_evidence.get("enabled")
                or structural_evidence.get("enabled")
                or building_roi.get("enabled")
            ),
            "has_vector_geometry": True,
            "has_vector_text": bool(page_data.get("has_vector_text")),
            "ocr": ocr_evidence,
            "nonstructural_mask": nonstructural_evidence,
            "structural_mask": structural_evidence,
            "building_roi": building_roi,
        }

    cleaned_bgr = (
        remove_dimension_annotations(render_bgr, cleanup_mask)
        if np.any(cleanup_mask)
        else render_bgr.copy()
    )
    colour_cleaned_bgr, _ = segmenter.remove_annotations(cleaned_bgr)
    model_view_rgb = segmenter.preprocess_dark_cad(colour_cleaned_bgr)
    _, model_input_512, model_input_metadata = segmenter.prepare_model_input(model_view_rgb)

    artifacts: dict[str, dict[str, str]] = {}
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        artifacts["render"] = _write_image(destination / "page_render.png", render_bgr)
        artifacts["cleaned"] = _write_image(destination / "cleaned_page.png", cleaned_bgr)
        artifacts["model_view"] = _write_image(
            destination / "model_view.png",
            cv2.cvtColor(model_view_rgb, cv2.COLOR_RGB2BGR),
        )
        artifacts["model_input_512"] = _write_image(
            destination / "model_input_512.png",
            cv2.cvtColor(
                _letterbox_preview(model_view_rgb, model_input_metadata),
                cv2.COLOR_RGB2BGR,
            ),
        )
        metadata_path = destination / "preprocessing.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "page_number": page_number,
                    "page_count": page_count,
                    "render_dpi": render_dpi,
                    "model_input": model_input_metadata,
                    "vector_analysis": vector_outcome.evidence,
                    "scale_calibration": scale_calibration,
                    "vector_cleanup": vector_cleanup,
                    "inference_roi": inference_roi,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts["metadata"] = {"path": metadata_path.name, "sha256": _sha256(metadata_path)}

    return PreparedFloorplanPage(
        page_number=page_number,
        page_count=page_count,
        render_bgr=render_bgr,
        cleaned_bgr=cleaned_bgr,
        model_view_rgb=model_view_rgb,
        model_input_512=model_input_512,
        model_input_metadata=model_input_metadata,
        vector_analysis=vector_outcome.evidence,
        scale_calibration=scale_calibration,
        vector_cleanup=vector_cleanup,
        inference_roi=inference_roi,
        cleanup_mask=cleanup_mask,
        structural_support_mask=structural_support_mask,
        artifacts=artifacts,
    )
