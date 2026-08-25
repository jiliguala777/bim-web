"""Select exterior wall anchors and conservatively enumerate their gaps."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExteriorThresholds:
    side_offset_fraction: float = 0.006
    side_offset_min_px: int = 4
    side_offset_max_px: int = 16
    footprint_inside_mean_min: float = 0.50
    footprint_side_difference_min: float = 0.25
    footprint_boundary_mean_min: float = 0.35
    collinear_tolerance_px: int = 2


def _local_line_mask(
    window_shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int,
    origin: tuple[int, int],
) -> np.ndarray:
    height, width = window_shape
    origin_x, origin_y = origin
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(
        mask,
        (start[0] - origin_x, start[1] - origin_y),
        (end[0] - origin_x, end[1] - origin_y),
        255,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_8,
    )
    return mask > 0


def _mean(channel: np.ndarray, mask: np.ndarray) -> float:
    samples = channel[mask]
    return float(np.mean(samples)) if samples.size else 0.0


def _axis_line(item: dict) -> tuple[str, tuple[int, int], tuple[int, int]]:
    orientation = item.get("orientation")
    start = tuple(int(round(value)) for value in item["start_px"])
    end = tuple(int(round(value)) for value in item["end_px"])
    if orientation == "horizontal":
        if start[1] != end[1]:
            raise ValueError("horizontal wall is not axis aligned")
        return orientation, min(start, end, key=lambda point: point[0]), max(start, end, key=lambda point: point[0])
    if orientation == "vertical":
        if start[0] != end[0]:
            raise ValueError("vertical wall is not axis aligned")
        return orientation, min(start, end, key=lambda point: point[1]), max(start, end, key=lambda point: point[1])
    raise ValueError("wall orientation must be horizontal or vertical")


def _sample_footprint_sides(
    item: dict,
    footprint: np.ndarray,
    image_size: tuple[int, int],
    offset: int,
) -> tuple[tuple[str, float], tuple[str, float], float]:
    width, height = (int(value) for value in image_size)
    orientation, start, end = _axis_line(item)
    if orientation == "horizontal":
        first_direction, second_direction = "up", "down"
        first_start, first_end = (start[0], start[1] - offset), (end[0], end[1] - offset)
        second_start, second_end = (start[0], start[1] + offset), (end[0], end[1] + offset)
    else:
        first_direction, second_direction = "left", "right"
        first_start, first_end = (start[0] - offset, start[1]), (end[0] - offset, end[1])
        second_start, second_end = (start[0] + offset, start[1]), (end[0] + offset, end[1])

    points = (start, end, first_start, first_end, second_start, second_end)
    padding = max(2, offset // 2 + 1)
    left = min(width, max(0, min(point[0] for point in points) - padding))
    top = min(height, max(0, min(point[1] for point in points) - padding))
    right = min(width, max(0, max(point[0] for point in points) + padding + 1))
    bottom = min(height, max(0, max(point[1] for point in points) + padding + 1))
    window_shape = (max(0, bottom - top), max(0, right - left))
    origin = (left, top)
    if not window_shape[0] or not window_shape[1]:
        return (first_direction, 0.0), (second_direction, 0.0), 0.0
    thickness = max(1, offset // 2)
    center = _local_line_mask(window_shape, start, end, thickness, origin)
    first = _local_line_mask(window_shape, first_start, first_end, thickness, origin)
    second = _local_line_mask(window_shape, second_start, second_end, thickness, origin)
    values = footprint[top:bottom, left:right]
    return (
        (first_direction, _mean(values, first)),
        (second_direction, _mean(values, second)),
        _mean(values, center),
    )


def select_exterior_walls(
    candidates: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
) -> list[dict]:
    """Return accepted walls with a supported building-inside side."""
    del building_roi  # Fusion has already applied the ROI admission rule.
    width, height = (int(value) for value in image_size)
    values = np.asarray(probabilities)
    if values.shape != (10, height, width):
        raise ValueError("probabilities must have shape (10, height, width)")
    thresholds = ExteriorThresholds()
    offset = min(
        thresholds.side_offset_max_px,
        max(
            thresholds.side_offset_min_px,
            round(min(width, height) * thresholds.side_offset_fraction),
        ),
    )
    exterior = []
    for original in candidates:
        if original.get("decision") != "accepted_wall_candidate":
            continue
        try:
            first, second, boundary_mean = _sample_footprint_sides(
                original, values[0], (width, height), offset,
            )
        except ValueError:
            continue
        inside = max((first, second), key=lambda side: side[1])
        outside = second if inside == first else first
        difference = inside[1] - outside[1]
        direct_support = (
            inside[1] >= thresholds.footprint_inside_mean_min
            and difference >= thresholds.footprint_side_difference_min
        )
        boundary_support = (
            boundary_mean >= thresholds.footprint_boundary_mean_min
            and difference >= thresholds.footprint_side_difference_min
        )
        if not (direct_support or boundary_support):
            continue
        wall = copy.deepcopy(original)
        wall["inside_direction"] = inside[0]
        wall["footprint_inside_mean"] = inside[1]
        wall["footprint_outside_mean"] = outside[1]
        exterior.append(wall)
    return exterior


def _normalise_exterior_wall(item: dict) -> dict:
    orientation, start, end = _axis_line(item)
    inside_direction = item.get("inside_direction")
    expected = {"horizontal": {"up", "down"}, "vertical": {"left", "right"}}
    if inside_direction not in expected[orientation]:
        raise ValueError("inside_direction does not match wall orientation")
    fixed = start[1] if orientation == "horizontal" else start[0]
    axis_start = start[0] if orientation == "horizontal" else start[1]
    axis_end = end[0] if orientation == "horizontal" else end[1]
    return {
        "orientation": orientation,
        "inside_direction": inside_direction,
        "fixed": fixed,
        "axis_start": axis_start,
        "axis_end": axis_end,
        "source_wall_ids": {str(item["candidate_id"])},
    }


def _merge_overlapping_collinear_walls(walls: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for wall in walls:
        groups[(wall["orientation"], wall["inside_direction"], wall["fixed"])].append(wall)
    merged = []
    for (orientation, inside_direction, fixed), members in sorted(groups.items()):
        intervals = sorted(
            (
                (member["axis_start"], member["axis_end"], index, member)
                for index, member in enumerate(members)
            ),
            key=lambda interval: (interval[0], interval[1], interval[2]),
        )
        start, end, _, first = intervals[0]
        source_ids = set(first["source_wall_ids"])
        for next_start, next_end, _, member in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
                source_ids.update(member["source_wall_ids"])
                continue
            merged.append({
                "orientation": orientation,
                "inside_direction": inside_direction,
                "fixed": fixed,
                "axis_start": start,
                "axis_end": end,
                "source_wall_ids": source_ids,
            })
            start, end = next_start, next_end
            source_ids = set(member["source_wall_ids"])
        merged.append({
            "orientation": orientation,
            "inside_direction": inside_direction,
            "fixed": fixed,
            "axis_start": start,
            "axis_end": end,
            "source_wall_ids": source_ids,
        })
    return merged


def _gap_points(left: dict, right: dict) -> tuple[list[int], list[int]]:
    if left["orientation"] == "horizontal":
        return [left["axis_end"], left["fixed"]], [right["axis_start"], right["fixed"]]
    return [left["fixed"], left["axis_end"]], [right["fixed"], right["axis_start"]]


def _perpendicular_crosses_gap(
    left: dict,
    right: dict,
    walls: list[dict],
    tolerance: int,
) -> bool:
    orientation = left["orientation"]
    for wall in walls:
        if wall["orientation"] == orientation:
            continue
        if orientation == "horizontal":
            crosses_axis = left["axis_end"] < wall["fixed"] < right["axis_start"]
            crosses_fixed = wall["axis_start"] - tolerance <= left["fixed"] <= wall["axis_end"] + tolerance
        else:
            crosses_axis = left["axis_end"] < wall["fixed"] < right["axis_start"]
            crosses_fixed = wall["axis_start"] - tolerance <= left["fixed"] <= wall["axis_end"] + tolerance
        if crosses_axis and crosses_fixed:
            return True
    return False


def _inside_roi(start: list[int], end: list[int], roi: list[int]) -> bool:
    left, top, right, bottom = (int(value) for value in roi)
    return all(left <= x <= right and top <= y <= bottom for x, y in (start, end))


def enumerate_exterior_gaps(
    exterior_walls: list[dict],
    image_size: tuple[int, int],
    building_roi: list[int] | None,
) -> list[dict]:
    """Merge overlapping collinear walls, preserve IDs, and enumerate adjacent endpoint gaps."""
    width, height = (int(value) for value in image_size)
    thresholds = ExteriorThresholds()
    walls = _merge_overlapping_collinear_walls([
        _normalise_exterior_wall(item) for item in exterior_walls
    ])
    roi = [0, 0, width, height] if building_roi is None else [int(value) for value in building_roi]

    proposed = []
    for left_index, left in enumerate(walls):
        compatible = [
            (right_index, right)
            for right_index, right in enumerate(walls)
            if right_index != left_index
            and right["orientation"] == left["orientation"]
            and abs(right["fixed"] - left["fixed"]) <= thresholds.collinear_tolerance_px
            and right["axis_start"] > left["axis_end"]
        ]
        if not compatible:
            continue
        nearest_start = min(item[1]["axis_start"] for item in compatible)
        nearest = [
            item for item in compatible
            if item[1]["axis_start"] == nearest_start
        ]
        proposed.append((left_index, nearest))

    records = []
    for left_index, nearest in proposed:
        left = walls[left_index]
        if len(nearest) != 1:
            right = min(nearest, key=lambda item: item[0])[1]
            start, end = _gap_points(left, right)
            competing_ids = set(left["source_wall_ids"])
            for _, competing_wall in nearest:
                competing_ids.update(competing_wall["source_wall_ids"])
            records.append({
                "orientation": left["orientation"],
                "start_px": start,
                "end_px": end,
                "width_px": right["axis_start"] - left["axis_end"],
                "host_wall_ids": sorted(competing_ids),
                "inside_direction": left["inside_direction"],
                "decision": "rejected_gap",
                "reason_codes": ["multiple_competing_endpoints"],
            })
            continue
        _, right = nearest[0]
        start, end = _gap_points(left, right)
        reasons = []
        if left["fixed"] != right["fixed"]:
            reasons.append("not_strictly_collinear")
        if left["inside_direction"] != right["inside_direction"]:
            reasons.append("different_inside_directions")
        if _perpendicular_crosses_gap(left, right, walls, thresholds.collinear_tolerance_px):
            reasons.append("perpendicular_structural_crossing")
        if not _inside_roi(start, end, roi):
            reasons.append("outside_building_roi")
        records.append({
            "orientation": left["orientation"],
            "start_px": start,
            "end_px": end,
            "width_px": right["axis_start"] - left["axis_end"],
            "host_wall_ids": sorted(left["source_wall_ids"] | right["source_wall_ids"]),
            "inside_direction": (
                left["inside_direction"]
                if left["inside_direction"] == right["inside_direction"] else None
            ),
            "decision": "rejected_gap" if reasons else "accepted_gap",
            "reason_codes": reasons or ["unambiguous_collinear_exterior_gap"],
        })

    records.sort(key=lambda item: (
        0 if item["orientation"] == "horizontal" else 1,
        item["start_px"][1], item["start_px"][0],
        item["end_px"][1], item["end_px"][0],
    ))
    for index, record in enumerate(records, 1):
        record["gap_id"] = f"gap-{index:04d}"
    return records
