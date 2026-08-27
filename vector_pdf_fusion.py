"""Fuse PDF-native orthogonal lines with aligned model probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class FusionThresholds:
    axis_tolerance_pt: float = 0.5
    roi_inside_ratio_min: float = 0.70
    roi_outside_reject_ratio: float = 0.80
    dimension_overlap_reject_ratio: float = 0.50
    wall_mean_min: float = 0.35
    wall_p90_min: float = 0.55
    room_side_mean_min: float = 0.35
    footprint_mean_min: float = 0.50

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name != "axis_tolerance_pt" and value > 1:
                raise ValueError(f"{name} must not exceed one")


def _point_to_pixel(point: list[float], page_data: dict) -> list[int]:
    page_width, page_height = (float(value) for value in page_data["page_size_pt"])
    render_width, render_height = (int(value) for value in page_data["render_size_px"])
    return [
        round(float(point[0]) * render_width / page_width),
        round(float(point[1]) * render_height / page_height),
    ]


def _inside_roi_ratio(start: list[int], end: list[int], roi: list[int]) -> float:
    left, top, right, bottom = (int(value) for value in roi)
    if start[1] == end[1]:
        y = start[1]
        if y < top or y > bottom:
            return 0.0
        x0, x1 = sorted((start[0], end[0]))
        overlap = max(0, min(x1, right) - max(x0, left))
        return overlap / max(1, x1 - x0)
    x = start[0]
    if x < left or x > right:
        return 0.0
    y0, y1 = sorted((start[1], end[1]))
    overlap = max(0, min(y1, bottom) - max(y0, top))
    return overlap / max(1, y1 - y0)


def build_line_candidates(
    page_data: dict,
    thresholds: FusionThresholds,
) -> list[dict]:
    """Create traceable candidates and apply native hard exclusions."""
    dimension_ids = {
        native_id
        for candidate in page_data.get("dimension_candidates", [])
        for native_id in candidate.get("member_native_ids", [])
    }
    roi_record = page_data.get("building_roi") or {}
    roi = roi_record.get("bbox_px") if roi_record.get("enabled") else None
    candidates = []
    for index, source in enumerate(page_data.get("orthogonal_segments", []), 1):
        start_px = list(source.get("start_px") or _point_to_pixel(source["start_pt"], page_data))
        end_px = list(source.get("end_px") or _point_to_pixel(source["end_pt"], page_data))
        inside_ratio = _inside_roi_ratio(start_px, end_px, roi) if roi is not None else None
        dimension_overlap = 1.0 if source["native_id"] in dimension_ids else 0.0
        brightness = source.get("neutral_brightness")
        native_structural = brightness is not None and float(brightness) <= 0.25
        reasons = []
        if dimension_overlap >= thresholds.dimension_overlap_reject_ratio:
            reasons.append("dimension_overlap")
        if bool(source.get("is_page_border")):
            reasons.append("page_border")
        if inside_ratio is not None and 1.0 - inside_ratio >= thresholds.roi_outside_reject_ratio:
            reasons.append("outside_building_roi")
        decision = "rejected_nonstructural" if reasons else "uncertain"
        candidates.append({
            "candidate_id": f"line-{index:05d}",
            "source_native_id": source["native_id"],
            "orientation": source["orientation"],
            "start_pt": list(source["start_pt"]),
            "end_pt": list(source["end_pt"]),
            "start_px": start_px,
            "end_px": end_px,
            "native_evidence": {
                "length_pt": float(source["length_pt"]),
                "width_pt": float(source.get("width_pt") or 0.0),
                "stroke_rgb": source.get("stroke_rgb"),
                "neutral_brightness": brightness,
                "native_structural": native_structural,
                "roi_inside_ratio": inside_ratio,
                "dimension_overlap_ratio": dimension_overlap,
                "page_border": bool(source.get("is_page_border")),
            },
            "model_evidence": None,
            "decision": decision,
            "reason_codes": reasons or ["model_evidence_pending"],
        })
    return candidates


def _local_line_mask(
    window_shape: tuple[int, int],
    start: list[int],
    end: list[int],
    thickness: int,
    origin: tuple[int, int],
) -> np.ndarray:
    height, width = window_shape
    origin_x, origin_y = origin
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(
        mask,
        (int(start[0]) - origin_x, int(start[1]) - origin_y),
        (int(end[0]) - origin_x, int(end[1]) - origin_y),
        255,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_8,
    )
    return mask > 0


def _sample(channel: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    values = channel[mask]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.percentile(values, 90)),
        float(np.max(values)),
    )


def fuse_line_candidates(
    candidates: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
    thresholds: FusionThresholds,
) -> list[dict]:
    """Sample aligned probability bands and apply conservative three-state rules."""
    width, height = (int(value) for value in image_size)
    values = np.asarray(probabilities)
    if values.shape != (10, height, width):
        raise ValueError("probabilities must have shape (10, height, width)")
    radius = min(12, max(2, round(min(width, height) * 0.003)))
    results = []
    for original in candidates:
        item = copy.deepcopy(original)
        start = item["start_px"]
        end = item["end_px"]
        offset = radius * 2 + 1
        if item["orientation"] == "horizontal":
            first_start, first_end = [start[0], start[1] - offset], [end[0], end[1] - offset]
            second_start, second_end = [start[0], start[1] + offset], [end[0], end[1] + offset]
        else:
            first_start, first_end = [start[0] - offset, start[1]], [end[0] - offset, end[1]]
            second_start, second_end = [start[0] + offset, start[1]], [end[0] + offset, end[1]]
        sample_points = (
            start, end, first_start, first_end, second_start, second_end,
        )
        padding = radius * 2 + 3
        left = min(width, max(0, min(point[0] for point in sample_points) - padding))
        top = min(height, max(0, min(point[1] for point in sample_points) - padding))
        right = min(width, max(0, max(point[0] for point in sample_points) + padding + 1))
        bottom = min(height, max(0, max(point[1] for point in sample_points) + padding + 1))
        window_shape = (max(0, bottom - top), max(0, right - left))
        origin = (left, top)
        if window_shape[0] and window_shape[1]:
            center = _local_line_mask(window_shape, start, end, radius * 2 + 1, origin)
            first_side = _local_line_mask(window_shape, first_start, first_end, radius + 1, origin)
            second_side = _local_line_mask(window_shape, second_start, second_end, radius + 1, origin)
        else:
            center = np.zeros(window_shape, dtype=bool)
            first_side = np.zeros(window_shape, dtype=bool)
            second_side = np.zeros(window_shape, dtype=bool)
        local_values = values[:, top:bottom, left:right]

        wall_mean, wall_p90, _ = _sample(local_values[4], center)
        footprint_mean, _, _ = _sample(local_values[0], center | first_side | second_side)
        room_first, _, _ = _sample(local_values[2], first_side)
        room_second, _, _ = _sample(local_values[2], second_side)
        _, _, door_max = _sample(local_values[5], center)
        _, _, window_max = _sample(local_values[6], center)
        _, _, door_endpoint_max = _sample(local_values[8], center)
        _, _, window_endpoint_max = _sample(local_values[9], center)
        model_evidence = {
            "sample_radius_px": radius,
            "wall_mean": wall_mean,
            "wall_p90": wall_p90,
            "room_side_mean": [room_first, room_second],
            "footprint_mean": footprint_mean,
            "door_line_max": door_max,
            "window_line_max": window_max,
            "door_endpoint_max": door_endpoint_max,
            "window_endpoint_max": window_endpoint_max,
        }
        item["model_evidence"] = model_evidence

        if item["decision"] == "rejected_nonstructural":
            results.append(item)
            continue
        native = item["native_evidence"]
        roi_ratio = native.get("roi_inside_ratio")
        roi_supported = roi_ratio is not None and roi_ratio >= thresholds.roi_inside_ratio_min
        wall_supported = wall_mean >= thresholds.wall_mean_min or wall_p90 >= thresholds.wall_p90_min
        room_supported = max(room_first, room_second) >= thresholds.room_side_mean_min
        footprint_supported = footprint_mean >= thresholds.footprint_mean_min
        native_supported = bool(native.get("native_structural"))
        accepted = all((
            native_supported,
            roi_supported,
            wall_supported,
            room_supported,
            footprint_supported,
        ))
        reasons = []
        if wall_supported:
            reasons.append("model_wall_supported")
        if room_supported:
            reasons.append("room_side_supported")
        if footprint_supported:
            reasons.append("footprint_supported")
        if roi_supported:
            reasons.append("inside_building_roi")
        if accepted:
            item["decision"] = "accepted_wall_candidate"
            reasons.append("native_structural_style")
        else:
            item["decision"] = "uncertain"
            if not native_supported:
                reasons.append("weak_native_support")
            if not (wall_supported and room_supported and footprint_supported):
                reasons.append("weak_model_support")
            if roi_ratio is None:
                reasons.append("building_roi_unavailable")
            elif not roi_supported:
                reasons.append("weak_roi_support")
        item["reason_codes"] = reasons
        results.append(item)
    return results
