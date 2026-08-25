"""Closed orthogonal room candidates derived from accepted native wall lines."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
import math

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class RoomClosureThresholds:
    snap_fraction: float = 0.003
    snap_min_px: int = 2
    snap_max_px: int = 8
    min_area_fraction: float = 0.0001
    min_area_px2: float = 400.0
    max_roi_area_fraction: float = 0.80
    room_probability_min: float = 0.35
    footprint_probability_min: float = 0.50


def _normalise_line(item: dict) -> dict:
    start = tuple(int(round(value)) for value in item["start_px"])
    end = tuple(int(round(value)) for value in item["end_px"])
    if item["orientation"] == "horizontal":
        if start[1] != end[1]:
            raise ValueError("horizontal accepted line is not axis aligned")
        start, end = (start, end) if start[0] <= end[0] else (end, start)
    elif item["orientation"] == "vertical":
        if start[0] != end[0]:
            raise ValueError("vertical accepted line is not axis aligned")
        start, end = (start, end) if start[1] <= end[1] else (end, start)
    else:
        raise ValueError("accepted line orientation must be horizontal or vertical")
    return {
        "orientation": item["orientation"],
        "start": start,
        "end": end,
        "source_wall_ids": {str(item["candidate_id"])},
    }


def _merge_collinear(lines: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for line in lines:
        fixed = line["start"][1] if line["orientation"] == "horizontal" else line["start"][0]
        groups[(line["orientation"], fixed)].append(line)
    merged = []
    for (orientation, fixed), records in sorted(groups.items()):
        if orientation == "horizontal":
            intervals = [(item["start"][0], item["end"][0], item) for item in records]
        else:
            intervals = [(item["start"][1], item["end"][1], item) for item in records]
        intervals.sort(key=lambda interval: (interval[0], interval[1]))
        first_start, first_end, first = intervals[0]
        current_start, current_end = first_start, first_end
        current_ids = set(first["source_wall_ids"])
        for interval_start, interval_end, item in intervals[1:]:
            if interval_start <= current_end:
                current_end = max(current_end, interval_end)
                current_ids.update(item["source_wall_ids"])
                continue
            start = (current_start, fixed) if orientation == "horizontal" else (fixed, current_start)
            end = (current_end, fixed) if orientation == "horizontal" else (fixed, current_end)
            merged.append({
                "orientation": orientation,
                "start": start,
                "end": end,
                "source_wall_ids": set(current_ids),
            })
            current_start, current_end = interval_start, interval_end
            current_ids = set(item["source_wall_ids"])
        start = (current_start, fixed) if orientation == "horizontal" else (fixed, current_start)
        end = (current_end, fixed) if orientation == "horizontal" else (fixed, current_end)
        merged.append({
            "orientation": orientation,
            "start": start,
            "end": end,
            "source_wall_ids": current_ids,
        })
    return merged


def _snap_endpoints(lines: list[dict], snap_px: int) -> tuple[list[dict], list[dict]]:
    snapped = copy.deepcopy(lines)
    records = []
    for index, line in enumerate(snapped):
        for endpoint_name in ("start", "end"):
            point = line[endpoint_name]
            candidates = []
            for target_index, target in enumerate(snapped):
                if target_index == index or target["orientation"] == line["orientation"]:
                    continue
                if line["orientation"] == "horizontal":
                    target_x = target["start"][0]
                    target_y0, target_y1 = target["start"][1], target["end"][1]
                    if not target_y0 <= point[1] <= target_y1:
                        continue
                    proposed = (target_x, point[1])
                    distance = abs(target_x - point[0])
                else:
                    target_y = target["start"][1]
                    target_x0, target_x1 = target["start"][0], target["end"][0]
                    if not target_x0 <= point[0] <= target_x1:
                        continue
                    proposed = (point[0], target_y)
                    distance = abs(target_y - point[1])
                if 0 < distance <= snap_px:
                    candidates.append((distance, proposed, target_index))
            if not candidates:
                continue
            minimum = min(item[0] for item in candidates)
            nearest = [item for item in candidates if item[0] == minimum]
            points = {item[1] for item in nearest}
            if len(points) != 1:
                continue
            distance, proposed, target_index = min(nearest, key=lambda item: item[2])
            original = point
            line[endpoint_name] = proposed
            target = snapped[target_index]
            source_ids = sorted(line["source_wall_ids"] | target["source_wall_ids"])
            records.append({
                "snap_id": f"snap-{len(records) + 1:04d}",
                "original_point_px": list(original),
                "snapped_point_px": list(proposed),
                "distance_px": float(distance),
                "source_wall_ids": source_ids,
            })
    return snapped, records


def _split_graph_edges(lines: list[dict]) -> list[tuple[Point, Point, set[str]]]:
    split_points = [{line["start"], line["end"]} for line in lines]
    horizontals = [(index, line) for index, line in enumerate(lines) if line["orientation"] == "horizontal"]
    verticals = [(index, line) for index, line in enumerate(lines) if line["orientation"] == "vertical"]
    for h_index, horizontal in horizontals:
        hx0, hy = horizontal["start"]
        hx1, _ = horizontal["end"]
        for v_index, vertical in verticals:
            vx, vy0 = vertical["start"]
            _, vy1 = vertical["end"]
            if hx0 <= vx <= hx1 and vy0 <= hy <= vy1:
                point = (vx, hy)
                split_points[h_index].add(point)
                split_points[v_index].add(point)
    edges = []
    for line, points in zip(lines, split_points):
        ordered = sorted(points, key=lambda point: point[0] if line["orientation"] == "horizontal" else point[1])
        for start, end in zip(ordered, ordered[1:]):
            if start != end:
                edges.append((start, end, set(line["source_wall_ids"])))
    return edges


def _signed_area(points: list[Point]) -> float:
    return sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    ) / 2.0


def _canonical_polygon(points: list[Point]) -> tuple[Point, ...]:
    choices = []
    for sequence in (points, list(reversed(points))):
        start = min(range(len(sequence)), key=lambda index: sequence[index])
        choices.append(tuple(sequence[start:] + sequence[:start]))
    return min(choices)


def _bounded_faces(
    edges: list[tuple[Point, Point, set[str]]],
) -> list[tuple[list[Point], set[str]]]:
    adjacency = defaultdict(set)
    sources = {}
    for start, end, wall_ids in edges:
        adjacency[start].add(end)
        adjacency[end].add(start)
        key = frozenset((start, end))
        sources.setdefault(key, set()).update(wall_ids)
    ordered = {
        point: sorted(
            neighbors,
            key=lambda neighbor: math.atan2(-(neighbor[1] - point[1]), neighbor[0] - point[0]),
        )
        for point, neighbors in adjacency.items()
    }
    visited = set()
    faces = []
    seen = set()
    directed_edges = [(start, end) for start, neighbors in adjacency.items() for end in neighbors]
    for initial in directed_edges:
        if initial in visited:
            continue
        start, end = initial
        polygon = []
        wall_ids = set()
        current = initial
        for _ in range(len(directed_edges) + 1):
            if current in visited and current != initial:
                polygon = []
                break
            visited.add(current)
            first, second = current
            polygon.append(first)
            wall_ids.update(sources[frozenset((first, second))])
            neighbors = ordered[second]
            reverse_index = neighbors.index(first)
            following = neighbors[(reverse_index + 1) % len(neighbors)]
            current = (second, following)
            if current == initial:
                break
        else:
            polygon = []
        if len(polygon) < 4 or _signed_area(polygon) <= 0:
            continue
        canonical = _canonical_polygon(polygon)
        if canonical in seen:
            continue
        seen.add(canonical)
        faces.append((polygon, wall_ids))
    return faces


def _mean_in_polygon(channel: np.ndarray, polygon: list[Point]) -> float:
    mask = np.zeros(channel.shape, dtype=np.uint8)
    contour = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 255)
    values = channel[mask > 0]
    return float(np.mean(values)) if values.size else 0.0


def find_room_candidates(
    accepted_lines: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
    thresholds: RoomClosureThresholds,
) -> dict:
    """Find minimal bounded orthogonal faces and score their interiors."""
    width, height = (int(value) for value in image_size)
    values = np.asarray(probabilities)
    if values.shape != (10, height, width):
        raise ValueError("probabilities must have shape (10, height, width)")
    normalised = [
        _normalise_line(item)
        for item in accepted_lines
        if item.get("decision") == "accepted_wall_candidate"
    ]
    snap_px = min(
        int(thresholds.snap_max_px),
        max(int(thresholds.snap_min_px), round(min(width, height) * thresholds.snap_fraction)),
    )
    snapped, snaps = _snap_endpoints(normalised, snap_px) if normalised else ([], [])
    merged = _merge_collinear(snapped) if snapped else []
    edges = _split_graph_edges(merged)
    faces = _bounded_faces(edges)
    image_area = width * height
    minimum_area = max(float(thresholds.min_area_px2), image_area * thresholds.min_area_fraction)
    if building_roi is None:
        roi = [0, 0, width, height]
    else:
        roi = [int(value) for value in building_roi]
    roi_area = max(1, roi[2] - roi[0]) * max(1, roi[3] - roi[1])
    rooms = []
    for polygon, wall_ids in faces:
        area = abs(_signed_area(polygon))
        inside_roi = all(roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3] for x, y in polygon)
        if not inside_roi or area < minimum_area or area > roi_area * thresholds.max_roi_area_fraction:
            continue
        room_mean = _mean_in_polygon(values[2], polygon)
        footprint_mean = _mean_in_polygon(values[0], polygon)
        accepted = (
            room_mean >= thresholds.room_probability_min
            and footprint_mean >= thresholds.footprint_probability_min
        )
        closure_applied = any(set(record["source_wall_ids"]) <= wall_ids for record in snaps)
        rooms.append({
            "room_id": f"room-{len(rooms) + 1:04d}",
            "polygon_px": [[int(x), int(y)] for x, y in polygon],
            "area_px2": float(area),
            "source_wall_ids": sorted(wall_ids),
            "closure_applied": closure_applied,
            "model_evidence": {
                "room_interior_mean": room_mean,
                "footprint_interior_mean": footprint_mean,
            },
            "decision": "accepted_room_candidate" if accepted else "suspicious_closed_region",
            "reason_codes": (
                ["closed_wall_face", "room_probability_supported", "footprint_supported"]
                if accepted else ["closed_wall_face", "weak_room_model_support"]
            ),
        })
    return {
        "merged_lines": [{
            "merged_id": f"merged-{index:05d}",
            "orientation": item["orientation"],
            "start_px": list(item["start"]),
            "end_px": list(item["end"]),
            "source_wall_ids": sorted(item["source_wall_ids"]),
        } for index, item in enumerate(merged, 1)],
        "snaps": snaps,
        "room_candidates": rooms,
        "summary": {
            "accepted_room_count": sum(item["decision"] == "accepted_room_candidate" for item in rooms),
            "suspicious_room_count": sum(item["decision"] == "suspicious_closed_region" for item in rooms),
        },
    }
