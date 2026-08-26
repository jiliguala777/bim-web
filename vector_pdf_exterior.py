"""Select exterior wall anchors and conservatively enumerate their gaps."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
import math

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
    continuity_connection_tolerance_px: int = 36
    continuity_boundary_band_px: int = 8
    continuity_min_length_px: int = 40


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


def _point_near_segment(
    point: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    tolerance_px: int,
) -> bool:
    if start[0] == end[0]:
        return (
            abs(point[0] - start[0]) <= tolerance_px
            and min(start[1], end[1]) - tolerance_px <= point[1] <= max(start[1], end[1]) + tolerance_px
        )
    return (
        abs(point[1] - start[1]) <= tolerance_px
        and min(start[0], end[0]) - tolerance_px <= point[0] <= max(start[0], end[0]) + tolerance_px
    )


def rescue_connected_exterior_walls(
    candidates: list[dict],
    exterior_walls: list[dict],
    image_size: tuple[int, int],
) -> list[dict]:
    """Promote only native wall candidates that close a supported exterior side.

    A candidate must already be a model-supported structural line and connect at
    both endpoints to distinct exterior anchors.  This intentionally cannot
    invent a long wall where the PDF itself supplies no line evidence.
    """
    if len(exterior_walls) < 2:
        return []
    width, height = (int(value) for value in image_size)
    thresholds = ExteriorThresholds()
    tolerance = thresholds.continuity_connection_tolerance_px
    existing_ids = {str(wall.get("candidate_id")) for wall in exterior_walls}
    anchors = []
    for wall in exterior_walls:
        try:
            _, start, end = _axis_line(wall)
        except (KeyError, TypeError, ValueError):
            continue
        anchors.append((str(wall.get("candidate_id")), start, end))
    if len(anchors) < 2:
        return []

    midpoints = []
    for _, start, end in anchors:
        midpoints.append(((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0))
    centre_x = float(np.mean([point[0] for point in midpoints]))
    centre_y = float(np.mean([point[1] for point in midpoints]))

    rescued = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        native = candidate.get("native_evidence") or {}
        if (
            candidate_id in existing_ids
            or candidate.get("decision") != "accepted_wall_candidate"
            or not native.get("native_structural")
        ):
            continue
        try:
            orientation, start, end = _axis_line(candidate)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(0 <= point[0] < width and 0 <= point[1] < height for point in (start, end)):
            continue
        if math.dist(start, end) < thresholds.continuity_min_length_px:
            continue

        # A rescued segment must lie on an already-supported outside band, not
        # merely cross the building interior between two exterior walls.
        fixed = start[0] if orientation == "vertical" else start[1]
        anchor_fixed = [
            anchor_start[0] if orientation == "vertical" else anchor_start[1]
            for _, anchor_start, anchor_end in anchors
            if (anchor_start[0] == anchor_end[0]) == (orientation == "vertical")
        ]
        if (
            anchor_fixed
            and min(abs(fixed - value) for value in anchor_fixed)
            > thresholds.continuity_boundary_band_px
        ):
            continue

        start_connections = {
            anchor_id for anchor_id, anchor_start, anchor_end in anchors
            if _point_near_segment(start, anchor_start, anchor_end, tolerance)
        }
        end_connections = {
            anchor_id for anchor_id, anchor_start, anchor_end in anchors
            if _point_near_segment(end, anchor_start, anchor_end, tolerance)
        }
        connected_anchor_ids = start_connections | end_connections
        if not start_connections or not end_connections or len(connected_anchor_ids) < 2:
            continue

        wall = copy.deepcopy(candidate)
        if orientation == "vertical":
            wall["inside_direction"] = "right" if fixed <= centre_x else "left"
        else:
            wall["inside_direction"] = "down" if fixed <= centre_y else "up"
        wall["footprint_inside_mean"] = None
        wall["footprint_outside_mean"] = None
        wall["reason_codes"] = [
            *wall.get("reason_codes", []),
            "connected_exterior_continuity_rescue",
        ]
        rescued.append(wall)
    return rescued


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


def _repair_limit_px(
    image_size: tuple[int, int],
    scale_m_per_px: float | None,
) -> float:
    if scale_m_per_px is not None:
        scale = float(scale_m_per_px)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale_m_per_px must be finite and positive")
        return 0.6 / scale
    return float(max(4, min(32, round(min(image_size) * 0.01))))


def _gap_within_repair_limit(
    width_px: float,
    repair_limit_px: float,
    scale_m_per_px: float | None,
) -> bool:
    if scale_m_per_px is None:
        return width_px <= repair_limit_px
    width_m = width_px * float(scale_m_per_px)
    return width_m <= math.nextafter(0.6, math.inf)


def _strict_axis_segment(item: dict) -> dict:
    orientation = item.get("orientation")
    start = tuple(float(value) for value in item["start_px"])
    end = tuple(float(value) for value in item["end_px"])
    if orientation == "horizontal" and start[1] == end[1] and start[0] != end[0]:
        first, second = sorted((start, end), key=lambda point: point[0])
        return {
            "orientation": orientation,
            "fixed": first[1],
            "axis_start": first[0],
            "axis_end": second[0],
        }
    if orientation == "vertical" and start[0] == end[0] and start[1] != end[1]:
        first, second = sorted((start, end), key=lambda point: point[1])
        return {
            "orientation": orientation,
            "fixed": first[0],
            "axis_start": first[1],
            "axis_end": second[1],
        }
    raise ValueError("segment must use strict horizontal or vertical endpoints")


def _segment_points(segment: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    if segment["orientation"] == "horizontal":
        return (
            (segment["axis_start"], segment["fixed"]),
            (segment["axis_end"], segment["fixed"]),
        )
    return (
        (segment["fixed"], segment["axis_start"]),
        (segment["fixed"], segment["axis_end"]),
    )


def _canonical_endpoints(item: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    segment = _strict_axis_segment(item)
    return _segment_points(segment)


def _opening_matches_gap(opening: dict, gap: dict) -> bool:
    try:
        return (
            opening.get("exterior") is True
            and opening.get("kind") in {"door", "window"}
            and opening.get("orientation") == gap.get("orientation")
            and _canonical_endpoints(opening) == _canonical_endpoints(gap)
            and sorted(str(value) for value in opening.get("host_wall_ids", []))
            == sorted(str(value) for value in gap.get("host_wall_ids", []))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _pending_opening_matches_gap(pending: dict, gap: dict) -> bool:
    try:
        return (
            pending.get("exterior") is True
            and pending.get("kind") == "pending_opening"
            and pending.get("orientation") == gap.get("orientation")
            and _canonical_endpoints(pending) == _canonical_endpoints(gap)
            and sorted(str(value) for value in pending.get("host_wall_ids", []))
            == sorted(str(value) for value in gap.get("host_wall_ids", []))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _unresolved_gap(gap: dict, reason: str | None = None) -> dict:
    unresolved = copy.deepcopy(gap)
    if reason is not None:
        unresolved["reason_codes"] = [reason]
    return unresolved


def _bridge_segment(bridge: dict) -> dict:
    segment = _strict_axis_segment(bridge)
    segment.update({
        "source_wall_ids": set(),
        "bridge_ids": {bridge["bridge_id"]},
        "opening_ids": (
            {bridge["opening_id"]} if bridge.get("opening_id") is not None else set()
        ),
        "pending_opening_ids": (
            {bridge["pending_opening_id"]}
            if bridge.get("pending_opening_id") is not None else set()
        ),
    })
    return segment


def _wall_segment(wall: dict) -> dict:
    segment = _strict_axis_segment(wall)
    candidate_id = wall.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("wall candidate_id must be a non-empty string")
    segment.update({
        "source_wall_ids": {candidate_id},
        "bridge_ids": set(),
        "opening_ids": set(),
        "pending_opening_ids": set(),
    })
    return segment


def _merge_topology_segments(segments: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for segment in segments:
        groups[(segment["orientation"], segment["fixed"])].append(segment)
    merged = []
    for (orientation, fixed), members in sorted(groups.items()):
        members.sort(key=lambda item: (item["axis_start"], item["axis_end"]))
        current = copy.deepcopy(members[0])
        for member in members[1:]:
            if member["axis_start"] <= current["axis_end"]:
                current["axis_end"] = max(current["axis_end"], member["axis_end"])
                for field in ("source_wall_ids", "bridge_ids", "opening_ids", "pending_opening_ids"):
                    current[field].update(member[field])
                continue
            merged.append(current)
            current = copy.deepcopy(member)
        merged.append(current)
    return merged


def _edge_key(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (start, end) if start <= end else (end, start)


def _split_at_intersections(
    segments: list[dict],
) -> tuple[dict[tuple[float, float], set[tuple[float, float]]], dict]:
    cuts = [{segment["axis_start"], segment["axis_end"]} for segment in segments]
    horizontal = [
        (index, segment) for index, segment in enumerate(segments)
        if segment["orientation"] == "horizontal"
    ]
    vertical = [
        (index, segment) for index, segment in enumerate(segments)
        if segment["orientation"] == "vertical"
    ]
    for horizontal_index, horizontal_segment in horizontal:
        for vertical_index, vertical_segment in vertical:
            x = vertical_segment["fixed"]
            y = horizontal_segment["fixed"]
            if (
                horizontal_segment["axis_start"] <= x <= horizontal_segment["axis_end"]
                and vertical_segment["axis_start"] <= y <= vertical_segment["axis_end"]
            ):
                cuts[horizontal_index].add(x)
                cuts[vertical_index].add(y)

    adjacency = defaultdict(set)
    edge_evidence = {}
    for segment, positions in zip(segments, cuts):
        ordered = sorted(positions)
        for axis_start, axis_end in zip(ordered, ordered[1:]):
            if axis_start == axis_end:
                continue
            piece = dict(segment, axis_start=axis_start, axis_end=axis_end)
            start, end = _segment_points(piece)
            adjacency[start].add(end)
            adjacency[end].add(start)
            key = _edge_key(start, end)
            evidence = edge_evidence.setdefault(key, {
                "source_wall_ids": set(),
                "bridge_ids": set(),
                "opening_ids": set(),
                "pending_opening_ids": set(),
            })
            for field in evidence:
                evidence[field].update(segment[field])
    return dict(adjacency), edge_evidence


_CLOCKWISE_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _direction_index(
    start: tuple[float, float],
    end: tuple[float, float],
) -> int:
    dx, dy = end[0] - start[0], end[1] - start[1]
    direction = (
        (1 if dx > 0 else -1, 0)
        if dx else (0, 1 if dy > 0 else -1)
    )
    return _CLOCKWISE_DIRECTIONS.index(direction)


def _next_face_edge(
    previous: tuple[float, float],
    current: tuple[float, float],
    adjacency: dict[tuple[float, float], set[tuple[float, float]]],
) -> tuple[float, float]:
    incoming = _direction_index(previous, current)
    by_direction = {
        _direction_index(current, neighbour): neighbour
        for neighbour in adjacency[current]
    }
    for direction in (
        (incoming + 1) % 4,
        incoming,
        (incoming - 1) % 4,
        (incoming + 2) % 4,
    ):
        if direction in by_direction:
            return by_direction[direction]
    raise ValueError("graph vertex has no outgoing edge")


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon, polygon[1:] + polygon[:1])
    )


def _simplify_polygon(
    polygon: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    simplified = []
    for index, point in enumerate(polygon):
        previous = polygon[index - 1]
        following = polygon[(index + 1) % len(polygon)]
        if (
            (previous[0] == point[0] == following[0])
            or (previous[1] == point[1] == following[1])
        ):
            continue
        simplified.append(point)
    return simplified


def _enumerate_bounded_faces(adjacency: dict, edge_evidence: dict) -> list[dict]:
    visited = set()
    faces = []
    directed_edges = sorted(
        (start, end) for start, neighbours in adjacency.items() for end in neighbours
    )
    for initial in directed_edges:
        if initial in visited:
            continue
        polygon = []
        face_edges = []
        edge = initial
        while edge not in visited:
            visited.add(edge)
            previous, current = edge
            polygon.append(previous)
            face_edges.append(_edge_key(previous, current))
            edge = (current, _next_face_edge(previous, current, adjacency))
        if edge != initial or len(polygon) < 4:
            continue
        area = _signed_area(polygon)
        if area <= 0:
            continue
        polygon = _simplify_polygon(polygon)
        evidence = {
            "source_wall_ids": set(),
            "bridge_ids": set(),
            "opening_ids": set(),
            "pending_opening_ids": set(),
        }
        for face_edge in face_edges:
            for field in evidence:
                evidence[field].update(edge_evidence[face_edge][field])
        faces.append({
            "polygon": polygon,
            "area": float(area),
            "perimeter": float(sum(
                abs(start[0] - end[0]) + abs(start[1] - end[1])
                for start, end in zip(polygon, polygon[1:] + polygon[:1])
            )),
            **evidence,
        })
    return faces


def _face_inside_roi(face: dict, roi: list[int]) -> bool:
    left, top, right, bottom = (float(value) for value in roi)
    return all(
        left <= x <= right and top <= y <= bottom
        for x, y in face["polygon"]
    )


def _face_footprint_mean(face: dict, footprint: np.ndarray) -> float:
    mask = np.zeros(footprint.shape, dtype=np.uint8)
    polygon = np.rint(np.asarray(face["polygon"], dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return _mean(footprint, mask > 0)


def _vertex_components(adjacency: dict) -> dict[tuple[float, float], int]:
    component_by_vertex = {}
    for vertex in sorted(adjacency):
        if vertex in component_by_vertex:
            continue
        component_id = len(set(component_by_vertex.values()))
        pending = [vertex]
        component_by_vertex[vertex] = component_id
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour in component_by_vertex:
                    continue
                component_by_vertex[neighbour] = component_id
                pending.append(neighbour)
    return component_by_vertex


def _polygon_strictly_contains(
    outer: list[tuple[float, float]],
    inner: list[tuple[float, float]],
) -> bool:
    contour = np.asarray(outer, dtype=np.float32)
    return all(
        cv2.pointPolygonTest(contour, (float(x), float(y)), False) > 0
        for x, y in inner
    )


def _has_nested_faces(faces: list[dict]) -> bool:
    for index, first in enumerate(faces):
        for second in faces[index + 1:]:
            if (
                _polygon_strictly_contains(first["polygon"], second["polygon"])
                or _polygon_strictly_contains(second["polygon"], first["polygon"])
            ):
                return True
    return False


def _empty_exterior_topology(status: str, unresolved_gaps: list[dict], bridges: list[dict]) -> dict:
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
        "pending_opening_ids": [],
        "real_wall_segments": [],
        "bridges": bridges,
        "unresolved_gaps": unresolved_gaps,
        "load_geometry_ready": False,
    }


def _real_wall_segments_for_face(
    exterior_walls: list[dict],
    polygon: list[tuple[float, float]],
    bridges: list[dict],
    selected_bridge_ids: set[str],
) -> list[dict]:
    wall_segments = []
    for wall in exterior_walls:
        try:
            wall_segments.append(_wall_segment(wall))
        except (KeyError, TypeError, ValueError):
            continue
    bridge_segments = []
    for bridge in bridges:
        if bridge.get("bridge_id") not in selected_bridge_ids:
            continue
        try:
            bridge_segments.append(_strict_axis_segment(bridge))
        except (KeyError, TypeError, ValueError):
            continue

    real_segments = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        edge = _strict_axis_segment({
            "orientation": "horizontal" if start[1] == end[1] else "vertical",
            "start_px": start,
            "end_px": end,
        })
        pieces = []
        cuts = {edge["axis_start"], edge["axis_end"]}
        for wall in wall_segments:
            if wall["orientation"] != edge["orientation"] or wall["fixed"] != edge["fixed"]:
                continue
            overlap_start = max(edge["axis_start"], wall["axis_start"])
            overlap_end = min(edge["axis_end"], wall["axis_end"])
            if overlap_start < overlap_end:
                pieces.append((overlap_start, overlap_end, wall["source_wall_ids"]))
                cuts.update((overlap_start, overlap_end))
        edge_bridges = []
        for bridge in bridge_segments:
            if bridge["orientation"] != edge["orientation"] or bridge["fixed"] != edge["fixed"]:
                continue
            overlap_start = max(edge["axis_start"], bridge["axis_start"])
            overlap_end = min(edge["axis_end"], bridge["axis_end"])
            if overlap_start < overlap_end:
                edge_bridges.append((overlap_start, overlap_end))
                cuts.update((overlap_start, overlap_end))

        atomic = []
        ordered = sorted(cuts)
        for axis_start, axis_end in zip(ordered, ordered[1:]):
            if any(
                bridge_start <= axis_start and axis_end <= bridge_end
                for bridge_start, bridge_end in edge_bridges
            ):
                continue
            source_ids = {
                source_id
                for piece_start, piece_end, piece_sources in pieces
                if piece_start <= axis_start and axis_end <= piece_end
                for source_id in piece_sources
            }
            if source_ids:
                atomic.append([axis_start, axis_end, source_ids])
        merged = []
        for item in atomic:
            if merged and merged[-1][1] == item[0] and merged[-1][2] == item[2]:
                merged[-1][1] = item[1]
            else:
                merged.append(item)
        for axis_start, axis_end, source_ids in merged:
            segment = dict(edge, axis_start=axis_start, axis_end=axis_end)
            segment_start, segment_end = _segment_points(segment)
            real_segments.append({
                "segment_id": f"real-wall-{len(real_segments) + 1:04d}",
                "orientation": edge["orientation"],
                "start_px": list(segment_start),
                "end_px": list(segment_end),
                "length_px": float(axis_end - axis_start),
                "source_wall_ids": sorted(source_ids),
            })
    return real_segments


def _discard_unused_small_gap_repairs(
    bridges: list[dict],
    unresolved_gaps: list[dict],
    gaps: list[dict],
    used_bridge_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    gaps_by_id = {str(gap.get("gap_id")): gap for gap in gaps}
    kept = []
    for bridge in bridges:
        if (
            bridge["bridge_type"] in {"small_gap_repair", "pending_opening_bridge"}
            and bridge["bridge_id"] not in used_bridge_ids
        ):
            gap = gaps_by_id.get(bridge["gap_id"])
            if gap is not None:
                unresolved_gaps.append(_unresolved_gap(
                    gap,
                    "pending_opening_does_not_close_supported_exterior"
                    if bridge["bridge_type"] == "pending_opening_bridge"
                    else "gap_does_not_close_supported_exterior",
                ))
            continue
        kept.append(bridge)
    unresolved_gaps.sort(key=lambda item: str(item.get("gap_id", "")))
    return kept, unresolved_gaps


def build_exterior_topology(
    exterior_walls: list[dict],
    gaps: list[dict],
    openings: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
    *,
    scale_m_per_px: float | None = None,
    pending_openings: list[dict] | None = None,
) -> dict:
    """Build traceable bridges and select the largest supported orthogonal footprint."""
    width, height = (int(value) for value in image_size)
    values = np.asarray(probabilities)
    if values.shape != (10, height, width):
        raise ValueError("probabilities must have shape (10, height, width)")
    roi = [0, 0, width, height] if building_roi is None else [int(value) for value in building_roi]
    repair_limit = _repair_limit_px((width, height), scale_m_per_px)
    pending_openings = [] if pending_openings is None else pending_openings

    wall_segments = []
    for wall in exterior_walls:
        try:
            wall_segments.append(_wall_segment(wall))
        except (KeyError, TypeError, ValueError):
            continue

    bridges = []
    unresolved_gaps = []
    for gap in gaps:
        if gap.get("decision") != "accepted_gap":
            unresolved_gaps.append(_unresolved_gap(gap))
            continue
        try:
            gap_segment = _strict_axis_segment(gap)
        except (KeyError, TypeError, ValueError):
            unresolved_gaps.append(_unresolved_gap(gap, "corner_gap_requires_turn"))
            continue
        matching_openings = [opening for opening in openings if _opening_matches_gap(opening, gap)]
        matching_pending = [
            pending for pending in pending_openings
            if _pending_opening_matches_gap(pending, gap)
        ]
        if len(matching_openings) > 1:
            unresolved_gaps.append(_unresolved_gap(gap, "multiple_matching_openings"))
            continue
        width_px = float(gap_segment["axis_end"] - gap_segment["axis_start"])
        if matching_openings:
            opening = matching_openings[0]
            bridge_type = "opening_bridge"
            opening_id = str(opening["opening_id"])
            confidence = float(opening.get("confidence", 0.0))
            reason_codes = ["accepted_opening_exact_gap_match"]
            pending_opening_id = None
        elif len(matching_pending) == 1:
            pending = matching_pending[0]
            bridge_type = "pending_opening_bridge"
            opening_id = None
            pending_opening_id = str(pending["pending_opening_id"])
            confidence = float(pending.get("confidence", 0.0))
            reason_codes = ["pending_opening_exact_gap_match"]
        elif len(matching_pending) > 1:
            unresolved_gaps.append(_unresolved_gap(gap, "multiple_matching_pending_openings"))
            continue
        elif _gap_within_repair_limit(width_px, repair_limit, scale_m_per_px):
            bridge_type = "small_gap_repair"
            opening_id = None
            pending_opening_id = None
            confidence = 1.0
            reason_codes = ["gap_within_repair_limit"]
        else:
            unresolved_gaps.append(_unresolved_gap(gap, "gap_exceeds_repair_limit"))
            continue
        start, end = _segment_points(gap_segment)
        bridge = {
            "bridge_id": f"bridge-{len(bridges) + 1:04d}",
            "bridge_type": bridge_type,
            "orientation": gap_segment["orientation"],
            "start_px": list(start),
            "end_px": list(end),
            "source_endpoints_px": [list(start), list(end)],
            "width_px": width_px,
            "gap_id": str(gap["gap_id"]),
            "host_wall_ids": sorted(str(value) for value in gap.get("host_wall_ids", [])),
            "opening_id": opening_id,
            "pending_opening_id": pending_opening_id,
            "confidence": confidence,
            "reason_codes": reason_codes,
        }
        bridges.append(bridge)

    if not wall_segments:
        bridges, unresolved_gaps = _discard_unused_small_gap_repairs(
            bridges, unresolved_gaps, gaps, set(),
        )
        return _empty_exterior_topology("no_exterior_wall_evidence", unresolved_gaps, bridges)

    graph_segments = [*wall_segments, *(_bridge_segment(bridge) for bridge in bridges)]
    merged = _merge_topology_segments(graph_segments)
    adjacency, edge_evidence = _split_at_intersections(merged)
    component_by_vertex = _vertex_components(adjacency)
    roi_faces = [
        face for face in _enumerate_bounded_faces(adjacency, edge_evidence)
        if _face_inside_roi(face, roi)
    ]
    for face in roi_faces:
        face["component_id"] = component_by_vertex[face["polygon"][0]]
    faces = [
        face for face in roi_faces
        if _face_footprint_mean(face, values[0]) >= ExteriorThresholds().footprint_inside_mean_min
    ]
    structurally_ambiguous = (
        len({face["component_id"] for face in roi_faces}) > 1
        or _has_nested_faces(roi_faces)
    )
    used_bridge_ids = {
        bridge_id for face in faces for bridge_id in face["bridge_ids"]
    }
    bridges, unresolved_gaps = _discard_unused_small_gap_repairs(
        bridges, unresolved_gaps, gaps, used_bridge_ids,
    )
    if structurally_ambiguous:
        return _empty_exterior_topology("ambiguous_exterior", unresolved_gaps, bridges)
    if not faces:
        return _empty_exterior_topology("exterior_not_closed", unresolved_gaps, bridges)

    faces.sort(key=lambda face: (-face["area"], face["polygon"]))
    largest = faces[0]
    if len(faces) > 1 and faces[1]["area"] >= largest["area"] * 0.90:
        return _empty_exterior_topology("ambiguous_exterior", unresolved_gaps, bridges)

    scale = float(scale_m_per_px) if scale_m_per_px is not None else None
    real_wall_segments = _real_wall_segments_for_face(
        exterior_walls, largest["polygon"], bridges, set(largest["bridge_ids"]),
    )
    return {
        "format": "pdf-exterior-topology/1",
        "status": "review_required",
        "confirmed": False,
        "polygon_px": [list(point) for point in largest["polygon"]],
        "area_px2": largest["area"],
        "perimeter_px": largest["perimeter"],
        "area_m2": largest["area"] * scale ** 2 if scale is not None else None,
        "perimeter_m": largest["perimeter"] * scale if scale is not None else None,
        "source_wall_ids": sorted(largest["source_wall_ids"]),
        "bridge_ids": sorted(largest["bridge_ids"]),
        "opening_ids": sorted(largest["opening_ids"]),
        "pending_opening_ids": sorted(largest["pending_opening_ids"]),
        "real_wall_segments": real_wall_segments,
        "bridges": bridges,
        "unresolved_gaps": unresolved_gaps,
        "load_geometry_ready": False,
    }
