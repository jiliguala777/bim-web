"""Footprint-guided, traceable exterior edge inference candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceThresholds:
    collinear_tolerance_px: int = 2
    endpoint_connection_tolerance_px: int = 4
    side_offset_fraction: float = 0.006
    side_offset_min_px: int = 4
    side_offset_max_px: int = 16
    footprint_inside_mean_min: float = 0.50
    footprint_side_difference_min: float = 0.25
    footprint_boundary_mean_min: float = 0.35
    footprint_inside_support_fraction_min: float = 0.80
    footprint_outside_support_fraction_max: float = 0.50
    max_edge_short_side_fraction: float = 0.10
    max_corner_options_per_endpoint: int = 12


def _axis_record(record: dict) -> dict:
    orientation = str(record["orientation"])
    start = tuple(int(round(value)) for value in record["start_px"])
    end = tuple(int(round(value)) for value in record["end_px"])
    if orientation == "horizontal" and start[1] == end[1] and start[0] != end[0]:
        first, second = sorted((start, end), key=lambda point: point[0])
        fixed, axis_start, axis_end = first[1], first[0], second[0]
    elif orientation == "vertical" and start[0] == end[0] and start[1] != end[1]:
        first, second = sorted((start, end), key=lambda point: point[1])
        fixed, axis_start, axis_end = first[0], first[1], second[1]
    else:
        raise ValueError("exterior wall must be a strict axis-aligned segment")
    expected = {"horizontal": {"up", "down"}, "vertical": {"left", "right"}}
    inside_direction = record.get("inside_direction")
    if inside_direction not in expected[orientation]:
        raise ValueError("inside direction does not match wall orientation")
    candidate_id = record.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("exterior wall candidate_id must be a non-empty string")
    return {
        "orientation": orientation,
        "fixed": fixed,
        "axis_start": axis_start,
        "axis_end": axis_end,
        "inside_direction": inside_direction,
        "anchor_wall_ids": {candidate_id},
    }


def _merge_anchor_bands(exterior_walls: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for wall in exterior_walls:
        try:
            normalised = _axis_record(wall)
        except (KeyError, TypeError, ValueError):
            continue
        key = (
            normalised["orientation"],
            normalised["fixed"],
            normalised["inside_direction"],
        )
        groups.setdefault(key, []).append(normalised)

    merged = []
    for (orientation, fixed, inside_direction), members in sorted(groups.items()):
        members.sort(key=lambda item: (item["axis_start"], item["axis_end"]))
        current = dict(members[0], anchor_wall_ids=set(members[0]["anchor_wall_ids"]))
        for member in members[1:]:
            if member["axis_start"] <= current["axis_end"]:
                current["axis_end"] = max(current["axis_end"], member["axis_end"])
                current["anchor_wall_ids"].update(member["anchor_wall_ids"])
                continue
            merged.append(current)
            current = dict(member, anchor_wall_ids=set(member["anchor_wall_ids"]))
        merged.append(current)
    for index, band in enumerate(merged, 1):
        band["band_id"] = f"anchor-band-{index:04d}"
    return merged


def _endpoint_point(band: dict, side: str) -> tuple[int, int]:
    axis = band["axis_start"] if side == "start" else band["axis_end"]
    return (
        (axis, band["fixed"])
        if band["orientation"] == "horizontal"
        else (band["fixed"], axis)
    )


def _point_near_band(point: tuple[int, int], band: dict, tolerance: int) -> bool:
    x, y = point
    if band["orientation"] == "horizontal":
        return (
            abs(y - band["fixed"]) <= tolerance
            and band["axis_start"] - tolerance <= x <= band["axis_end"] + tolerance
        )
    return (
        abs(x - band["fixed"]) <= tolerance
        and band["axis_start"] - tolerance <= y <= band["axis_end"] + tolerance
    )


def _dangling_endpoints(bands: list[dict], tolerance: int) -> list[dict]:
    endpoints = []
    for band in bands:
        for side in ("start", "end"):
            point = _endpoint_point(band, side)
            connected = any(
                other["band_id"] != band["band_id"]
                and _point_near_band(point, other, tolerance)
                for other in bands
            )
            if not connected:
                endpoints.append({"band": band, "side": side, "point": point})
    return endpoints


def _line_sample_values(
    footprint: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    count = max(2, length + 1)
    xs = np.rint(np.linspace(start[0], end[0], count)).astype(np.int32)
    ys = np.rint(np.linspace(start[1], end[1], count)).astype(np.int32)
    valid = (
        (xs >= 0) & (xs < footprint.shape[1])
        & (ys >= 0) & (ys < footprint.shape[0])
    )
    if not np.any(valid):
        return np.empty(0, dtype=np.float32)
    return np.asarray(footprint[ys[valid], xs[valid]], dtype=np.float32)


def _line_samples(
    footprint: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> float:
    values = _line_sample_values(footprint, start, end)
    return float(np.mean(values)) if values.size else 0.0


def _shifted_edge(
    orientation: str,
    start: tuple[int, int],
    end: tuple[int, int],
    direction: str,
    offset: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    shifts = {
        "up": (0, -offset),
        "down": (0, offset),
        "left": (-offset, 0),
        "right": (offset, 0),
    }
    expected = {"horizontal": {"up", "down"}, "vertical": {"left", "right"}}
    if direction not in expected[orientation]:
        raise ValueError("inside direction does not match inferred edge")
    dx, dy = shifts[direction]
    return (start[0] + dx, start[1] + dy), (end[0] + dx, end[1] + dy)


def _opposite(direction: str) -> str:
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[direction]


def _edge_evidence(
    edge: dict,
    footprint: np.ndarray,
    offset: int,
    thresholds: InferenceThresholds,
) -> tuple[dict, list[str]]:
    start = tuple(edge["start_px"])
    end = tuple(edge["end_px"])
    inside = _shifted_edge(
        edge["orientation"], start, end, edge["inside_direction"], offset,
    )
    outside = _shifted_edge(
        edge["orientation"], start, end, _opposite(edge["inside_direction"]), offset,
    )
    inside_values = _line_sample_values(footprint, *inside)
    outside_values = _line_sample_values(footprint, *outside)
    inside_mean = float(np.mean(inside_values)) if inside_values.size else 0.0
    outside_mean = float(np.mean(outside_values)) if outside_values.size else 0.0
    boundary_mean = _line_samples(footprint, start, end)
    difference = inside_mean - outside_mean
    inside_support_fraction = (
        float(np.mean(inside_values >= thresholds.footprint_inside_mean_min))
        if inside_values.size else 0.0
    )
    outside_support_fraction = (
        float(np.mean(outside_values >= thresholds.footprint_inside_mean_min))
        if outside_values.size else 0.0
    )
    reasons = []
    if (
        inside_mean < thresholds.footprint_inside_mean_min
        or difference < thresholds.footprint_side_difference_min
        or boundary_mean < thresholds.footprint_boundary_mean_min
    ):
        reasons.append("insufficient_inside_footprint_support")
    if inside_support_fraction < thresholds.footprint_inside_support_fraction_min:
        reasons.append("discontinuous_inside_footprint_support")
    if outside_support_fraction >= thresholds.footprint_outside_support_fraction_max:
        reasons.append("inferred_edge_crosses_building_interior")
    return {
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "boundary_mean": boundary_mean,
        "inside_outside_difference": difference,
        "inside_support_fraction": inside_support_fraction,
        "outside_support_fraction": outside_support_fraction,
    }, reasons


def _inside_roi(point: tuple[int, int], roi: list[int]) -> bool:
    left, top, right, bottom = roi
    return left <= point[0] <= right and top <= point[1] <= bottom


def _edge(
    start: tuple[int, int],
    end: tuple[int, int],
    orientation: str,
    inside_direction: str,
    inference_type: str,
    anchors: set[str],
) -> dict:
    first, second = (
        sorted((start, end), key=lambda point: point[0])
        if orientation == "horizontal"
        else sorted((start, end), key=lambda point: point[1])
    )
    return {
        "inference_type": inference_type,
        "orientation": orientation,
        "start_px": list(first),
        "end_px": list(second),
        "length_px": float(math.dist(first, second)),
        "inside_direction": inside_direction,
        "anchor_wall_ids": sorted(anchors),
    }


def _edge_crosses_unanchored_band(
    edge: dict,
    bands: list[dict],
    tolerance: int,
) -> bool:
    anchors = set(edge["anchor_wall_ids"])
    start = tuple(edge["start_px"])
    end = tuple(edge["end_px"])
    axis_start = start[0] if edge["orientation"] == "horizontal" else start[1]
    axis_end = end[0] if edge["orientation"] == "horizontal" else end[1]
    fixed = start[1] if edge["orientation"] == "horizontal" else start[0]
    for band in bands:
        if anchors.intersection(band["anchor_wall_ids"]):
            continue
        if band["orientation"] == edge["orientation"]:
            if abs(band["fixed"] - fixed) > tolerance:
                continue
            overlap_start = max(axis_start, band["axis_start"])
            overlap_end = min(axis_end, band["axis_end"])
            if overlap_end - overlap_start > tolerance:
                return True
            continue
        crossing_axis = band["fixed"]
        if not axis_start + tolerance < crossing_axis < axis_end - tolerance:
            continue
        if band["axis_start"] - tolerance <= fixed <= band["axis_end"] + tolerance:
            return True
    return False


def _corner_compatible(horizontal: dict, vertical: dict) -> bool:
    horizontal_band = horizontal["band"]
    vertical_band = vertical["band"]
    required_vertical_inside = "right" if horizontal["side"] == "start" else "left"
    required_horizontal_inside = "down" if vertical["side"] == "start" else "up"
    horizontal_toward_corner = (
        vertical_band["fixed"] <= horizontal["point"][0]
        if horizontal["side"] == "start"
        else vertical_band["fixed"] >= horizontal["point"][0]
    )
    vertical_toward_corner = (
        horizontal_band["fixed"] <= vertical["point"][1]
        if vertical["side"] == "start"
        else horizontal_band["fixed"] >= vertical["point"][1]
    )
    return (
        horizontal_toward_corner
        and vertical_toward_corner
        and vertical_band["inside_direction"] == required_vertical_inside
        and horizontal_band["inside_direction"] == required_horizontal_inside
    )


def _group_candidates(bands: list[dict], thresholds: InferenceThresholds) -> list[dict]:
    dangling = _dangling_endpoints(bands, thresholds.endpoint_connection_tolerance_px)
    endpoint_by_band_side = {
        (item["band"]["band_id"], item["side"]): item for item in dangling
    }
    groups = []

    by_key: dict[tuple[str, int, str], list[dict]] = {}
    for band in bands:
        key = (band["orientation"], band["fixed"], band["inside_direction"])
        by_key.setdefault(key, []).append(band)
    for members in by_key.values():
        ordered = sorted(members, key=lambda item: (item["axis_start"], item["axis_end"]))
        for left, right in zip(ordered, ordered[1:]):
            left_endpoint = endpoint_by_band_side.get((left["band_id"], "end"))
            right_endpoint = endpoint_by_band_side.get((right["band_id"], "start"))
            gap = right["axis_start"] - left["axis_end"]
            if (
                left_endpoint is None or right_endpoint is None
                or gap <= thresholds.endpoint_connection_tolerance_px
            ):
                continue
            groups.append({
                "inference_type": "collinear_extension",
                "intersection": None,
                "edges": [_edge(
                    left_endpoint["point"], right_endpoint["point"], left["orientation"],
                    left["inside_direction"], "collinear_extension",
                    left["anchor_wall_ids"] | right["anchor_wall_ids"],
                )],
                "group_reasons": [],
            })

    horizontal = [item for item in dangling if item["band"]["orientation"] == "horizontal"]
    vertical = [item for item in dangling if item["band"]["orientation"] == "vertical"]
    for horizontal_endpoint in horizontal:
        corner_options = []
        for vertical_endpoint in vertical:
            horizontal_band = horizontal_endpoint["band"]
            vertical_band = vertical_endpoint["band"]
            intersection = (vertical_band["fixed"], horizontal_band["fixed"])
            horizontal_length = abs(intersection[0] - horizontal_endpoint["point"][0])
            vertical_length = abs(intersection[1] - vertical_endpoint["point"][1])
            if max(horizontal_length, vertical_length) <= thresholds.endpoint_connection_tolerance_px:
                continue
            compatible = _corner_compatible(horizontal_endpoint, vertical_endpoint)
            corner_options.append((
                0 if compatible else 1,
                max(horizontal_length, vertical_length),
                horizontal_length + vertical_length,
                str(vertical_band["band_id"]),
                str(vertical_endpoint["side"]),
                vertical_endpoint,
                intersection,
                horizontal_length,
                vertical_length,
                compatible,
            ))
        for (
            _, _, _, _, _, vertical_endpoint, intersection,
            horizontal_length, vertical_length, compatible,
        ) in sorted(corner_options)[:thresholds.max_corner_options_per_endpoint]:
            horizontal_band = horizontal_endpoint["band"]
            vertical_band = vertical_endpoint["band"]
            reasons = [] if compatible else ["incompatible_corner_inside_directions"]
            anchors = horizontal_band["anchor_wall_ids"] | vertical_band["anchor_wall_ids"]
            edges = []
            if horizontal_length:
                edges.append(_edge(
                    horizontal_endpoint["point"], intersection, "horizontal",
                    horizontal_band["inside_direction"], "orthogonal_corner", anchors,
                ))
            if vertical_length:
                edges.append(_edge(
                    intersection, vertical_endpoint["point"], "vertical",
                    vertical_band["inside_direction"], "orthogonal_corner", anchors,
                ))
            if len(edges) != 2:
                reasons.append("incomplete_orthogonal_corner")
            groups.append({
                "inference_type": "orthogonal_corner",
                "intersection": intersection,
                "edges": edges,
                "group_reasons": reasons,
            })
    return groups


def generate_exterior_inference_candidates(
    exterior_walls: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
) -> list[dict]:
    """Return accepted and rejected, stable footprint-guided edge candidates."""
    width, height = (int(value) for value in image_size)
    values = np.asarray(probabilities)
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if values.shape != (10, height, width):
        raise ValueError("probabilities must have shape (10, height, width)")
    roi = (
        [0, 0, width, height]
        if building_roi is None else [int(value) for value in building_roi]
    )
    thresholds = InferenceThresholds()
    bands = _merge_anchor_bands(exterior_walls)
    if len(bands) < 2:
        return []
    offset = min(
        thresholds.side_offset_max_px,
        max(thresholds.side_offset_min_px, round(min(width, height) * thresholds.side_offset_fraction)),
    )
    max_edge_length = min(width, height) * thresholds.max_edge_short_side_fraction
    raw_groups = _group_candidates(bands, thresholds)

    unique_groups: dict[tuple[Any, ...], dict] = {}
    for group in raw_groups:
        edge_key = tuple(sorted(
            (
                edge["orientation"], tuple(edge["start_px"]), tuple(edge["end_px"]),
                edge["inside_direction"],
            )
            for edge in group["edges"]
        ))
        key = (group["inference_type"], edge_key)
        unique_groups.setdefault(key, group)

    candidates = []
    ordered_groups = sorted(unique_groups.values(), key=lambda group: (
        group["inference_type"],
        tuple(group["intersection"] or (-1, -1)),
        tuple((edge["orientation"], edge["start_px"], edge["end_px"]) for edge in group["edges"]),
    ))
    for group_index, group in enumerate(ordered_groups, 1):
        reasons = list(group["group_reasons"])
        intersection = group["intersection"]
        if intersection is not None and not _inside_roi(intersection, roi):
            reasons.append("inferred_intersection_outside_roi")
        enriched_edges = []
        for edge in group["edges"]:
            edge_reasons = []
            if not all(_inside_roi(tuple(point), roi) for point in (edge["start_px"], edge["end_px"])):
                edge_reasons.append("inferred_edge_outside_roi")
            if edge["length_px"] > max_edge_length:
                edge_reasons.append("inferred_edge_exceeds_short_side_limit")
            if _edge_crosses_unanchored_band(
                edge, bands, thresholds.endpoint_connection_tolerance_px,
            ):
                edge_reasons.append("inferred_edge_crosses_nearer_wall_band")
            evidence, evidence_reasons = _edge_evidence(edge, values[0], offset, thresholds)
            enriched_edges.append((dict(edge, **evidence), edge_reasons + evidence_reasons))
            reasons.extend(edge_reasons)
            reasons.extend(evidence_reasons)
        reasons = list(dict.fromkeys(reasons))
        group_id = f"inference-group-{group_index:04d}"
        group_size = len(enriched_edges)
        for edge_index, (edge, _) in enumerate(enriched_edges, 1):
            edge.update({
                "inference_id": f"inferred-{group_index:04d}-{edge_index:02d}",
                "edge_group_id": group_id,
                "edge_group_size": group_size,
                "decision": "rejected_candidate" if reasons else "accepted_candidate",
                "reason_codes": reasons or ["footprint_guided_exterior_edge"],
            })
            candidates.append(edge)
    return candidates
