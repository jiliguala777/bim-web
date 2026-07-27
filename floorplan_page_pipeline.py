"""Deterministic preparation of one PDF page for floorplan recognition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
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


@dataclass(frozen=True)
class PreparedFloorplanPage:
    page_number: int
    page_count: int
    render_bgr: np.ndarray
    cleaned_bgr: np.ndarray
    model_view_rgb: np.ndarray
    model_input_512: np.ndarray
    model_input_metadata: dict[str, Any]
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


def _write_image(path: Path, image: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"Cannot write image artifact: {path.name}")
    os.replace(temporary, path)
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
    page_data = extract_vector_page(source, page_index=page_number - 1, dpi=vector_dpi)
    page_data["page_count"] = page_count
    render_dpi = vector_dpi if page_data["is_vector_pdf"] else 200
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
    page_data["render_size_px"] = [render_bgr.shape[1], render_bgr.shape[0]]

    scale_calibration = _default_scale_calibration(
        "vector_pdf_overall_dimensions" if page_data["is_vector_pdf"] else "scanned_pdf_manual"
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
        scale_calibration=scale_calibration,
        vector_cleanup=vector_cleanup,
        inference_roi=inference_roi,
        cleanup_mask=cleanup_mask,
        structural_support_mask=structural_support_mask,
        artifacts=artifacts,
    )
