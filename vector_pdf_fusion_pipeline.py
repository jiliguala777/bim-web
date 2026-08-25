"""Independent end-to-end pipeline for native vector-PDF fusion diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import cv2
import numpy as np
from pdf2image import convert_from_path

from vector_pdf_fusion import FusionThresholds, build_line_candidates, fuse_line_candidates
from vector_pdf_model import (
    VectorModelConfig,
    VectorModelContractError,
    VectorModelError,
    VectorModelTimeoutError,
    VectorModelUnavailableError,
    VectorProbabilityResult,
    run_vector_probabilities,
)
from vector_pdf_native import (
    build_dimension_mask,
    crop_native_page_data,
    extract_native_pdf_page,
)
from vector_pdf_rooms import RoomClosureThresholds, find_room_candidates


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(path.suffix, image)
    if not encoded:
        raise OSError(f"cannot encode image artifact: {path.name}")
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


def _remove_masked_annotations(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    border = np.concatenate((
        image_bgr[0, :, :], image_bgr[-1, :, :],
        image_bgr[:, 0, :], image_bgr[:, -1, :],
    ))
    background = np.median(border, axis=0).astype(np.uint8)
    cleaned = image_bgr.copy()
    cleaned[mask > 0] = background
    return cleaned


def _draw_overlay(
    base_bgr: np.ndarray,
    candidates: list[dict],
    rooms: list[dict],
) -> np.ndarray:
    overlay = base_bgr.copy()
    fill = overlay.copy()
    for room in rooms:
        points = np.asarray(room["polygon_px"], dtype=np.int32).reshape(-1, 1, 2)
        colour = (230, 140, 40) if room["decision"] == "accepted_room_candidate" else (180, 40, 180)
        cv2.fillPoly(fill, [points], colour)
    overlay = cv2.addWeighted(fill, 0.25, overlay, 0.75, 0)
    colours = {
        "accepted_wall_candidate": (40, 190, 40),
        "rejected_nonstructural": (40, 40, 220),
        "uncertain": (40, 210, 230),
    }
    for item in candidates:
        cv2.line(
            overlay,
            tuple(int(value) for value in item["start_px"]),
            tuple(int(value) for value in item["end_px"]),
            colours[item["decision"]],
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return overlay


def analyze_vector_pdf_page(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path,
    model_config: VectorModelConfig,
    crop_bbox_page_px: list[int] | None = None,
    *,
    model_runner: Callable[..., VectorProbabilityResult] = run_vector_probabilities,
) -> dict:
    """Analyze one vector-PDF page and publish non-energy diagnostic artifacts."""
    if page_number < 1:
        raise ValueError("page_number must be positive")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    page_data = extract_native_pdf_page(pdf_path, page_index=page_number - 1, dpi=100)
    if not page_data["is_vector_pdf"]:
        _write_json(output / "pdf_native_candidates.json", page_data)
        rejected = {
            "format": "pdf-vector-fusion/1",
            "status": "rejected",
            "load_geometry_ready": False,
            "calibration_status": "uncalibrated",
            "page": {
                "page_number": page_number,
                "page_size_pt": page_data["page_size_pt"],
                "analysis_size_px": page_data["render_size_px"],
                "crop_bbox_page_px": None,
            },
            "model": None,
            "thresholds": {},
            "line_candidates": [],
            "topology": {"merged_lines": [], "snaps": []},
            "room_candidates": [],
            "summary": {
                "accepted_wall_count": 0,
                "rejected_line_count": 0,
                "uncertain_line_count": 0,
                "accepted_room_count": 0,
                "suspicious_room_count": 0,
                "ignored_diagonal_count": int(page_data.get("ignored_diagonal_count") or 0),
            },
            "reason_codes": ["not_vector_pdf"],
        }
        _write_json(output / "pdf_vector_fusion.json", rejected)
        return rejected
    rendered = convert_from_path(
        str(Path(pdf_path).resolve()),
        dpi=100,
        first_page=page_number,
        last_page=page_number,
        poppler_path=os.environ.get("POPPLER_PATH") or None,
    )
    if len(rendered) != 1:
        raise ValueError("PDF renderer did not return exactly one selected page")
    render_bgr = cv2.cvtColor(np.asarray(rendered[0]), cv2.COLOR_RGB2BGR)
    dimension_mask = build_dimension_mask(page_data)
    cleaned_bgr = _remove_masked_annotations(render_bgr, dimension_mask)
    if crop_bbox_page_px is not None:
        left, top, right, bottom = (int(value) for value in crop_bbox_page_px)
        page_data = crop_native_page_data(page_data, [left, top, right, bottom])
        render_bgr = render_bgr[top:bottom, left:right].copy()
        cleaned_bgr = cleaned_bgr[top:bottom, left:right].copy()

    _write_json(output / "pdf_native_candidates.json", page_data)
    model_input_path = output / "pdf_vector_model_input.png"
    _write_image(model_input_path, cleaned_bgr)
    height, width = cleaned_bgr.shape[:2]
    fusion_thresholds = FusionThresholds()
    candidates = build_line_candidates(page_data, fusion_thresholds)
    roi_record = page_data.get("building_roi") or {}
    roi = roi_record.get("bbox_px") if roi_record.get("enabled") else None
    room_thresholds = RoomClosureThresholds()
    try:
        model_result = model_runner(
            model_input_path,
            output / "model-artifacts",
            model_config,
            expected_size=(width, height),
        )
    except VectorModelContractError as exc:
        status, reason = "rejected", "model_contract_invalid"
        detail = str(exc)
    except VectorModelTimeoutError as exc:
        status, reason = "partial", "model_timeout"
        detail = str(exc)
    except VectorModelUnavailableError as exc:
        status, reason = "partial", "model_unavailable"
        detail = str(exc)
    except VectorModelError as exc:
        status, reason = "partial", "model_failed"
        detail = str(exc)
    else:
        status = reason = detail = None
    if status is not None:
        summary = {
            "accepted_wall_count": 0,
            "rejected_line_count": sum(item["decision"] == "rejected_nonstructural" for item in candidates),
            "uncertain_line_count": sum(item["decision"] == "uncertain" for item in candidates),
            "accepted_room_count": 0,
            "suspicious_room_count": 0,
            "ignored_diagonal_count": int(page_data.get("ignored_diagonal_count") or 0),
        }
        diagnostic = {
            "format": "pdf-vector-fusion/1",
            "status": status,
            "load_geometry_ready": False,
            "calibration_status": "uncalibrated",
            "page": {
                "page_number": page_number,
                "page_size_pt": page_data["page_size_pt"],
                "analysis_size_px": [width, height],
                "crop_bbox_page_px": page_data.get("crop_bbox_page_px"),
            },
            "model": {"error": detail},
            "thresholds": {
                "fusion": asdict(fusion_thresholds),
                "room_closure": asdict(room_thresholds),
            },
            "line_candidates": candidates,
            "topology": {"merged_lines": [], "snaps": []},
            "room_candidates": [],
            "summary": summary,
            "reason_codes": [reason],
        }
        _write_json(output / "pdf_vector_fusion.json", diagnostic)
        _write_image(
            output / "pdf_vector_fusion_overlay.png",
            _draw_overlay(render_bgr, candidates, []),
        )
        return diagnostic
    candidates = fuse_line_candidates(
        candidates,
        model_result.probabilities,
        (width, height),
        roi,
        fusion_thresholds,
    )
    topology = find_room_candidates(
        candidates,
        model_result.probabilities,
        (width, height),
        roi,
        room_thresholds,
    )
    summary = {
        "accepted_wall_count": sum(item["decision"] == "accepted_wall_candidate" for item in candidates),
        "rejected_line_count": sum(item["decision"] == "rejected_nonstructural" for item in candidates),
        "uncertain_line_count": sum(item["decision"] == "uncertain" for item in candidates),
        **topology["summary"],
        "ignored_diagonal_count": int(page_data.get("ignored_diagonal_count") or 0),
    }
    payload = {
        "format": "pdf-vector-fusion/1",
        "status": "evaluable",
        "load_geometry_ready": False,
        "calibration_status": "uncalibrated",
        "page": {
            "page_number": page_number,
            "page_size_pt": page_data["page_size_pt"],
            "analysis_size_px": [width, height],
            "crop_bbox_page_px": page_data.get("crop_bbox_page_px"),
        },
        "model": {
            "inference": model_result.inference,
            "probabilities_path": os.path.relpath(
                model_result.artifact_dir / "probabilities.npz", output
            ).replace("\\", "/"),
            "probabilities_sha256": model_result.probabilities_sha256,
        },
        "thresholds": {
            "fusion": asdict(fusion_thresholds),
            "room_closure": asdict(room_thresholds),
        },
        "line_candidates": candidates,
        "topology": {
            "merged_lines": topology["merged_lines"],
            "snaps": topology["snaps"],
        },
        "room_candidates": topology["room_candidates"],
        "summary": summary,
        "reason_codes": [],
    }
    _write_json(output / "pdf_vector_fusion.json", payload)
    _write_image(
        output / "pdf_vector_fusion_overlay.png",
        _draw_overlay(render_bgr, candidates, topology["room_candidates"]),
    )
    return payload
