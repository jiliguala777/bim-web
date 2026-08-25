"""Independent end-to-end pipeline for native vector-PDF fusion diagnostics."""

from __future__ import annotations

import copy
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
from vector_pdf_exterior import (
    build_exterior_topology,
    enumerate_exterior_gaps,
    select_exterior_walls,
)
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
from vector_pdf_openings import classify_exterior_openings
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


def _draw_dashed_line(
    image: np.ndarray,
    start: list[int] | tuple[int, int],
    end: list[int] | tuple[int, int],
    colour: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash_px: int = 8,
) -> None:
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    vector = second - first
    length = float(np.linalg.norm(vector))
    if length == 0:
        return
    direction = vector / length
    for offset in np.arange(0.0, length, dash_px * 2):
        segment_end = min(length, offset + dash_px)
        cv2.line(
            image,
            tuple(np.rint(first + direction * offset).astype(int)),
            tuple(np.rint(first + direction * segment_end).astype(int)),
            colour,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_exterior_overlay(
    base_bgr: np.ndarray,
    exterior_walls: list[dict],
    opening_result: dict,
    exterior_topology: dict,
) -> np.ndarray:
    """Render review-only exterior evidence without changing source geometry."""
    overlay = base_bgr.copy()
    polygon = exterior_topology.get("polygon_px") or []
    if len(polygon) >= 3:
        footprint = overlay.copy()
        cv2.fillPoly(
            footprint,
            [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)],
            (180, 0, 180),
        )
        overlay = cv2.addWeighted(footprint, 0.25, overlay, 0.75, 0)
        cv2.polylines(
            overlay,
            [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=(180, 0, 180),
            thickness=4,
            lineType=cv2.LINE_AA,
        )
    for wall in exterior_walls:
        cv2.line(
            overlay,
            tuple(int(value) for value in wall["start_px"]),
            tuple(int(value) for value in wall["end_px"]),
            (0, 180, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    opening_colours = {"door": (255, 0, 0), "window": (255, 255, 0)}
    for opening in opening_result.get("accepted_openings", []):
        colour = opening_colours.get(opening.get("kind"))
        if colour is None:
            continue
        cv2.line(
            overlay,
            tuple(int(value) for value in opening["start_px"]),
            tuple(int(value) for value in opening["end_px"]),
            colour,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    for bridge in exterior_topology.get("bridges", []):
        colour = (255, 0, 0) if bridge.get("bridge_type") == "opening_bridge" else (0, 165, 255)
        _draw_dashed_line(overlay, bridge["start_px"], bridge["end_px"], colour)
    for gap in exterior_topology.get("unresolved_gaps", []):
        _draw_dashed_line(overlay, gap["start_px"], gap["end_px"], (0, 0, 255))
    return overlay


def _exterior_provenance(
    page_data: dict,
    page_number: int,
    analysis_size: tuple[int, int],
    roi: list[int] | None,
) -> dict:
    crop_bbox = copy.deepcopy(page_data.get("crop_bbox_page_px"))
    return {
        "coordinate_space": "crop-local-px" if crop_bbox is not None else "page-local-px",
        "page_number": page_number,
        "page_size_pt": copy.deepcopy(page_data["page_size_pt"]),
        "analysis_size_px": list(analysis_size),
        "crop_bbox_page_px": crop_bbox,
        "building_roi_px": copy.deepcopy(roi),
    }


def _empty_exterior_topology(status: str) -> dict:
    return {
        "format": "pdf-exterior-topology/1",
        "status": status,
        "confirmed": False,
        "polygon_px": [],
        "area_px2": 0.0,
        "perimeter_px": 0.0,
        "area_m2": None,
        "perimeter_m": None,
        "source_wall_ids": [],
        "bridge_ids": [],
        "opening_ids": [],
        "real_wall_segments": [],
        "bridges": [],
        "unresolved_gaps": [],
        "load_geometry_ready": False,
    }


def _publish_exterior_artifacts(
    output: Path,
    base_bgr: np.ndarray,
    page_data: dict,
    page_number: int,
    image_size: tuple[int, int],
    roi: list[int] | None,
    exterior_walls: list[dict],
    gaps: list[dict],
    opening_result: dict,
    exterior: dict,
) -> tuple[dict, dict, dict]:
    """Atomically publish all unconfirmed exterior-review artifacts."""
    provenance = _exterior_provenance(page_data, page_number, image_size, roi)
    opening_candidates = {
        "format": "pdf-opening-candidates/1",
        "status": exterior["status"],
        "confirmed": False,
        "load_geometry_ready": False,
        "provenance": provenance,
        "gaps": copy.deepcopy(gaps),
        "accepted_openings": copy.deepcopy(opening_result["accepted_openings"]),
        "ambiguous_openings": copy.deepcopy(opening_result["ambiguous_openings"]),
        "unclassified_gaps": copy.deepcopy(opening_result["unclassified_gaps"]),
    }
    exterior_topology = {
        **copy.deepcopy(exterior),
        "confirmed": False,
        "load_geometry_ready": False,
        "unresolved": copy.deepcopy(exterior.get("unresolved_gaps", [])),
        "provenance": provenance,
    }
    exterior_summary = {
        "exterior_wall_count": len(exterior_walls),
        "accepted_gap_count": sum(gap.get("decision") == "accepted_gap" for gap in gaps),
        "rejected_gap_count": sum(gap.get("decision") != "accepted_gap" for gap in gaps),
        "accepted_opening_count": len(opening_result["accepted_openings"]),
        "ambiguous_opening_count": len(opening_result["ambiguous_openings"]),
        "unclassified_gap_count": len(opening_result["unclassified_gaps"]),
        "bridge_count": len(exterior_topology["bridges"]),
        "unresolved_gap_count": len(exterior_topology["unresolved"]),
        "footprint_status": exterior_topology["status"],
        "confirmed": False,
        "load_geometry_ready": False,
    }
    _write_json(output / "pdf_opening_candidates.json", opening_candidates)
    _write_json(output / "pdf_exterior_topology.json", exterior_topology)
    _write_image(
        output / "pdf_exterior_overlay.png",
        _draw_exterior_overlay(base_bgr, exterior_walls, opening_result, exterior_topology),
    )
    return opening_candidates, exterior_topology, exterior_summary


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
        render_width, render_height = (int(value) for value in page_data["render_size_px"])
        exterior_openings, exterior_topology, exterior_summary = _publish_exterior_artifacts(
            output,
            np.full((render_height, render_width, 3), 255, dtype=np.uint8),
            page_data,
            page_number,
            (render_width, render_height),
            None,
            [],
            [],
            {"accepted_openings": [], "ambiguous_openings": [], "unclassified_gaps": []},
            _empty_exterior_topology("not_vector_pdf"),
        )
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
            "opening_candidates": exterior_openings,
            "exterior_topology": exterior_topology,
            "exterior_summary": exterior_summary,
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
    actual_height, actual_width = render_bgr.shape[:2]
    actual_render_size = [actual_width, actual_height]
    if page_data["render_size_px"] != actual_render_size:
        page_data["render_size_px"] = actual_render_size
        roi_record = page_data.get("building_roi") or {}
        if roi_record.get("enabled") and roi_record.get("bbox_pt"):
            page_width, page_height = (float(value) for value in page_data["page_size_pt"])
            x0, y0, x1, y1 = (float(value) for value in roi_record["bbox_pt"])
            roi_record["bbox_px"] = [
                round(x0 * actual_width / page_width),
                round(y0 * actual_height / page_height),
                round(x1 * actual_width / page_width),
                round(y1 * actual_height / page_height),
            ]
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
        exterior_openings, exterior_topology, exterior_summary = _publish_exterior_artifacts(
            output,
            render_bgr,
            page_data,
            page_number,
            (width, height),
            roi,
            [],
            [],
            {"accepted_openings": [], "ambiguous_openings": [], "unclassified_gaps": []},
            _empty_exterior_topology(reason),
        )
        diagnostic.update({
            "opening_candidates": exterior_openings,
            "exterior_topology": exterior_topology,
            "exterior_summary": exterior_summary,
        })
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
    exterior_walls = select_exterior_walls(
        candidates,
        model_result.probabilities,
        (width, height),
        roi,
    )
    gaps = enumerate_exterior_gaps(exterior_walls, (width, height), roi)
    opening_result = classify_exterior_openings(
        gaps,
        model_result.probabilities,
        (width, height),
    )
    exterior = build_exterior_topology(
        exterior_walls,
        gaps,
        opening_result["accepted_openings"],
        model_result.probabilities,
        (width, height),
        roi,
    )
    exterior_openings, exterior_topology, exterior_summary = _publish_exterior_artifacts(
        output,
        render_bgr,
        page_data,
        page_number,
        (width, height),
        roi,
        exterior_walls,
        gaps,
        opening_result,
        exterior,
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
        "opening_candidates": exterior_openings,
        "exterior_topology": exterior_topology,
        "exterior_summary": exterior_summary,
        "summary": summary,
        "reason_codes": [],
    }
    _write_json(output / "pdf_vector_fusion.json", payload)
    _write_image(
        output / "pdf_vector_fusion_overlay.png",
        _draw_overlay(render_bgr, candidates, topology["room_candidates"]),
    )
    return payload
