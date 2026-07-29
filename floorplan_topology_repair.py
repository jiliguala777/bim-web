"""Conservative topology repair for vector-PDF floor-plan masks."""

from __future__ import annotations

import math

import cv2
import numpy as np

from floorplan_rooms import extract_room_topology


DOMINANT_SPAN_MIN_VECTOR_SIDE_SUPPORT = 0.60
DOMINANT_SPAN_MIN_MODEL_SIDE_SUPPORT = 0.15


def _normalize_roi(roi, width: int, height: int) -> list[int] | None:
    if roi is None or len(roi) != 4:
        return None
    x0, y0, x1, y1 = (int(round(value)) for value in roi)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _room_topology(mask: np.ndarray, min_room_area_px: float) -> dict:
    return extract_room_topology(
        mask,
        max_gap_px=0,
        min_room_area_px=min_room_area_px,
    )


def _room_areas(topology: dict) -> list[float]:
    return sorted(float(room["area_px2"]) for room in topology.get("rooms", []))


def _bbox_iou(first: list[int], second: list[int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def _rooms_stable(before: dict, after: dict, tolerance: float = 0.03) -> bool:
    before_areas = _room_areas(before)
    after_areas = _room_areas(after)
    if len(before_areas) != len(after_areas):
        return False
    for old, new in zip(before_areas, after_areas):
        allowed = max(12.0, old * tolerance)
        if abs(old - new) > allowed:
            return False
    return True


def _new_room_is_plausible(before: dict, after: dict, roi: list[int]) -> bool:
    if int(after.get("room_count") or 0) <= int(before.get("room_count") or 0):
        return False
    before_rooms = before.get("rooms", [])
    roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
    for room in after.get("rooms", []):
        area = float(room["area_px2"])
        if any(
            _bbox_iou(room["bbox_px"], old["bbox_px"]) >= 0.8
            and abs(area - float(old["area_px2"]))
            <= max(12.0, float(old["area_px2"]) * 0.05)
            for old in before_rooms
        ):
            continue
        x, y, width, height = room["bbox_px"]
        aspect = max(width, height) / max(1.0, min(width, height))
        if (
            roi[0] <= x
            and roi[1] <= y
            and x + width <= roi[2]
            and y + height <= roi[3]
            and area <= roi_area * 0.95
            and aspect <= 12.0
        ):
            return True
    return False


def _closed_free_regions(mask: np.ndarray, min_area_px: float) -> list[dict]:
    free = (mask == 0).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    height, width = mask.shape
    regions = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        region_width = int(stats[label, cv2.CC_STAT_WIDTH])
        region_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = float(stats[label, cv2.CC_STAT_AREA])
        if x <= 0 or y <= 0 or x + region_width >= width or y + region_height >= height:
            continue
        if area >= min_area_px:
            regions.append({
                "area_px2": area,
                "bbox_px": [x, y, region_width, region_height],
            })
    return regions


def _new_closed_region_is_plausible(
    before_regions: list[dict],
    after_regions: list[dict],
    roi: list[int],
) -> bool:
    if len(after_regions) <= len(before_regions):
        return False
    roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
    for region in after_regions:
        if any(
            _bbox_iou(region["bbox_px"], old["bbox_px"]) >= 0.8
            for old in before_regions
        ):
            continue
        x, y, width, height = region["bbox_px"]
        aspect = max(width, height) / max(1.0, min(width, height))
        if (
            roi[0] <= x
            and roi[1] <= y
            and x + width <= roi[2]
            and y + height <= roi[3]
            and region["area_px2"] <= roi_area * 0.95
            and aspect <= 12.0
        ):
            return True
    return False


def _lower_entrance_candidates(
    barrier: np.ndarray,
    support: np.ndarray,
    roi: list[int],
    max_short_gap_px: int,
) -> list[dict]:
    """Find one logical long horizontal boundary across a recessed entrance."""
    x0, y0, x1, y1 = roi
    roi_width = x1 - x0
    roi_height = y1 - y0
    search_top = round(y0 + roi_height * 0.6)
    search_bottom = round(y0 + roi_height * 0.94)
    support_vertical = cv2.dilate(
        (support > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)),
    )
    row_coverage_values = np.mean(
        support_vertical[search_top:search_bottom, x0:x1] > 0,
        axis=1,
    )
    supported_rows = []
    for offset, coverage_value in enumerate(row_coverage_values):
        y = search_top + offset
        coverage = float(coverage_value)
        if coverage >= 0.25:
            supported_rows.append((y, coverage))

    row_clusters: list[list[tuple[int, float]]] = []
    for row in supported_rows:
        if not row_clusters or row[0] > row_clusters[-1][-1][0] + 1:
            row_clusters.append([])
        row_clusters[-1].append(row)

    candidates = []
    vertical_half_window = max(20, round(roi_height * 0.035))
    barrier_horizontal = cv2.dilate(
        (barrier > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
    )
    support_for_lines = cv2.dilate(
        (support > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13)),
    )
    for cluster in row_clusters:
        y, row_coverage = max(cluster, key=lambda item: item[1])
        vertical_scores = np.mean(
            barrier_horizontal[
                max(0, y - vertical_half_window):min(barrier.shape[0], y + vertical_half_window + 1),
                x0:x1,
            ] > 0,
            axis=0,
        )
        anchor_columns = (np.flatnonzero(vertical_scores >= 0.28) + x0).tolist()

        anchor_clusters: list[list[int]] = []
        for x in anchor_columns:
            if not anchor_clusters or x > anchor_clusters[-1][-1] + 1:
                anchor_clusters.append([])
            anchor_clusters[-1].append(x)
        anchors = [
            round(sum(group) / len(group))
            for group in anchor_clusters
            if len(group) >= 2
        ]

        supported_pairs = []
        for index, start in enumerate(anchors):
            for end in anchors[index + 1:]:
                span = end - start
                if span <= max_short_gap_px or span > roi_width * 0.8:
                    continue
                support_ratio = float(np.mean(support_for_lines[y, start:end + 1] > 0))
                if support_ratio >= 0.8:
                    supported_pairs.append((span, start, end, support_ratio))
        if not supported_pairs:
            continue
        _, start, end, support_ratio = max(supported_pairs)
        candidates.append({
            "orientation": "horizontal",
            "line_px": [int(start), int(y), int(end), int(y)],
            "bbox_px": [int(start), int(y), int(end - start + 1), 1],
            "kind": "lower_entrance",
            "support_ratio": round(support_ratio, 4),
            "row_coverage": round(row_coverage, 4),
        })
    return candidates


def _candidate_segments(barrier: np.ndarray, max_gap_px: int) -> list[dict]:
    if max_gap_px <= 0:
        return []
    kernel_size = max(3, int(max_gap_px) + 1)
    additions = []
    for orientation, kernel in (
        ("horizontal", cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))),
        ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))),
    ):
        closed = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kernel)
        added = ((closed > 0) & (barrier == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
        for label in range(1, count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            if orientation == "horizontal":
                if width < 2 or height > max(5, width // 2):
                    continue
                line = [x - 1, y + height // 2, x + width, y + height // 2]
            else:
                if height < 2 or width > max(5, height // 2):
                    continue
                line = [x + width // 2, y - 1, x + width // 2, y + height]
            line[0] = max(0, line[0])
            line[1] = max(0, line[1])
            line[2] = min(barrier.shape[1] - 1, line[2])
            line[3] = min(barrier.shape[0] - 1, line[3])
            additions.append({
                "orientation": orientation,
                "line_px": line,
                "bbox_px": [x, y, width, height],
            })

    unique = {}
    for candidate in additions:
        unique[tuple(candidate["line_px"])] = candidate
    return list(unique.values())


def _u_shape_candidates(component: np.ndarray, max_gap_px: int) -> list[dict]:
    """Return a missing bbox side only when the other three sides are present."""
    points = cv2.findNonZero(component.astype(np.uint8))
    if points is None:
        return []
    x, y, width, height = cv2.boundingRect(points)
    if width < 6 or height < 6 or max(width, height) > max_gap_px + 5:
        return []

    band = max(2, min(3, min(width, height) // 4))
    crop = component[y:y + height, x:x + width]
    side_occupancy = {
        "top": float(np.mean(np.any(crop[:band, :], axis=0))),
        "bottom": float(np.mean(np.any(crop[-band:, :], axis=0))),
        "left": float(np.mean(np.any(crop[:, :band], axis=1))),
        "right": float(np.mean(np.any(crop[:, -band:], axis=1))),
    }
    missing = [name for name, ratio in side_occupancy.items() if ratio < 0.5]
    present = [name for name, ratio in side_occupancy.items() if ratio >= 0.65]
    if len(missing) != 1 or len(present) != 3:
        return []

    side = missing[0]
    if side == "top":
        line = [x + 1, y + 1, x + width - 2, y + 1]
        orientation = "horizontal"
    elif side == "bottom":
        line = [x + 1, y + height - 2, x + width - 2, y + height - 2]
        orientation = "horizontal"
    elif side == "left":
        line = [x + 1, y + 1, x + 1, y + height - 2]
        orientation = "vertical"
    else:
        line = [x + width - 2, y + 1, x + width - 2, y + height - 2]
        orientation = "vertical"
    return [{
        "orientation": orientation,
        "line_px": line,
        "bbox_px": [x, y, width, height],
    }]


def _line_mask(shape: tuple[int, int], line: list[int], thickness: int = 3) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, tuple(line[:2]), tuple(line[2:]), 255, thickness=thickness)
    return mask


def _support_ratio(support: np.ndarray | None, line: list[int]) -> float:
    if support is None:
        return 0.0
    probe = _line_mask(support.shape, line, thickness=3)
    pixels = probe > 0
    if not np.any(pixels):
        return 0.0
    return float(np.count_nonzero((support > 0) & pixels)) / float(np.count_nonzero(pixels))


def _line_band_coverage(
    evidence: np.ndarray,
    line: list[int],
    half_band: int = 6,
) -> float:
    x0, y0, x1, y1 = (int(value) for value in line)
    binary = evidence > 0
    if y0 == y1:
        left, right = sorted((x0, x1))
        top = max(0, y0 - half_band)
        bottom = min(binary.shape[0], y0 + half_band + 1)
        if right < left or bottom <= top:
            return 0.0
        return float(np.mean(np.any(binary[top:bottom, left:right + 1], axis=0)))
    if x0 == x1:
        top, bottom = sorted((y0, y1))
        left = max(0, x0 - half_band)
        right = min(binary.shape[1], x0 + half_band + 1)
        if bottom < top or right <= left:
            return 0.0
        return float(np.mean(np.any(binary[top:bottom + 1, left:right], axis=1)))
    return 0.0


def _candidate_near_roi_edge(candidate: dict, roi: list[int]) -> bool:
    x0, y0, x1, y1 = roi
    line = candidate["line_px"]
    center_x = (line[0] + line[2]) / 2.0
    center_y = (line[1] + line[3]) / 2.0
    edge_band = max(12.0, min(x1 - x0, y1 - y0) * 0.12)
    return min(
        abs(center_x - x0),
        abs(center_x - x1),
        abs(center_y - y0),
        abs(center_y - y1),
    ) <= edge_band


def _main_component_label(barrier: np.ndarray) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(barrier, connectivity=8)
    if count <= 1:
        return labels, 0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels, main


def _endpoint_touches_label(labels: np.ndarray, point: tuple[int, int], label: int) -> bool:
    x, y = point
    x0, x1 = max(0, x - 3), min(labels.shape[1], x + 4)
    y0, y1 = max(0, y - 3), min(labels.shape[0], y + 4)
    return bool(label and np.any(labels[y0:y1, x0:x1] == label))


def _endpoint_touches_barrier(barrier: np.ndarray, point: tuple[int, int]) -> bool:
    x, y = point
    x0, x1 = max(0, x - 3), min(barrier.shape[1], x + 4)
    y0, y1 = max(0, y - 3), min(barrier.shape[0], y + 4)
    return bool(np.any(barrier[y0:y1, x0:x1] > 0))


def _apply_wall_line(mask: np.ndarray, line: list[int]) -> np.ndarray:
    result = mask.copy()
    cv2.line(result, tuple(line[:2]), tuple(line[2:]), 1, thickness=3)
    return result


def _estimate_wall_thickness(barrier: np.ndarray, roi: list[int]) -> int:
    x0, y0, x1, y1 = roi
    crop = (barrier[y0:y1, x0:x1] > 0).astype(np.uint8)
    distances = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
    samples = distances[(distances > 0) & (distances <= 8)]
    if samples.size == 0:
        return 1
    return int(np.clip(round(float(np.median(samples)) * 2), 1, 8))


def _multi_gap_limits(
    barrier: np.ndarray,
    roi: list[int],
    configured_gap: int,
) -> dict:
    wall = _estimate_wall_thickness(barrier, roi)
    short_side = min(roi[2] - roi[0], roi[3] - roi[1])
    return {
        "wall_thickness_px": wall,
        "max_gap_px": int(np.clip(
            max(configured_gap, 8 * wall, round(short_side * 0.015)),
            16,
            64,
        )),
        "alignment_tolerance_px": min(max(2 * wall, 4), 12),
        "max_candidates_per_round": 24,
        "beam_width": 16,
        "max_rounds": 6,
        "max_lines": 16,
    }


def _baseline_rooms_preserved(before: dict, after: dict) -> bool:
    for old in before.get("rooms", []):
        old_area = float(old["area_px2"])
        if not any(
            _bbox_iou(old["bbox_px"], room["bbox_px"]) >= 0.8
            and abs(float(room["area_px2"]) - old_area)
            <= max(12.0, old_area * 0.05)
            for room in after.get("rooms", [])
        ):
            return False
    return True


def _ranked_gap_candidates(
    barrier: np.ndarray,
    support: np.ndarray,
    roi: list[int],
    limits: dict,
) -> list[dict]:
    candidates = _candidate_segments(barrier, limits["max_gap_px"])
    candidates.extend(_lower_entrance_candidates(
        barrier,
        support,
        roi,
        limits["max_gap_px"],
    ))
    unique = {}
    for candidate in candidates:
        line = [int(value) for value in candidate["line_px"]]
        if not (
            roi[0] <= line[0] <= roi[2]
            and roi[0] <= line[2] <= roi[2]
            and roi[1] <= line[1] <= roi[3]
            and roi[1] <= line[3] <= roi[3]
        ):
            continue
        if not (
            _endpoint_touches_barrier(barrier, tuple(line[:2]))
            and _endpoint_touches_barrier(barrier, tuple(line[2:]))
        ):
            continue
        length = float(math.hypot(line[2] - line[0], line[3] - line[1]))
        support_ratio = _support_ratio(support, line)
        score = support_ratio * 2.0 + 1.0 - length / max(1.0, limits["max_gap_px"])
        record = {
            **candidate,
            "line_px": line,
            "length_px": round(length, 3),
            "support_ratio": round(support_ratio, 4),
            "alignment_error_px": 0.0,
            "score": round(score, 4),
        }
        key = tuple(line)
        if key not in unique or record["score"] > unique[key]["score"]:
            unique[key] = record
    ranked = sorted(
        unique.values(),
        key=lambda item: (-item["score"], item["length_px"], item["line_px"]),
    )
    return ranked[:limits["max_candidates_per_round"]]


def _dominant_axes(
    support: np.ndarray,
    roi: list[int],
    orientation: str,
    min_coverage: float = 0.70,
) -> list[dict]:
    x0, y0, x1, y1 = roi
    binary = (support > 0).astype(np.uint8)
    if orientation == "vertical":
        expanded = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)),
        )
        values = np.mean(expanded[y0:y1, x0:x1] > 0, axis=0)
        origin = x0
    else:
        expanded = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13)),
        )
        values = np.mean(expanded[y0:y1, x0:x1] > 0, axis=1)
        origin = y0

    coordinates = (np.flatnonzero(values >= min_coverage) + origin).tolist()
    clusters = []
    for coordinate in coordinates:
        if not clusters or coordinate > clusters[-1][-1] + 1:
            clusters.append([])
        clusters[-1].append(int(coordinate))
    axes = []
    for cluster in clusters:
        peak = max(values[value - origin] for value in cluster)
        peak_coordinates = [
            value
            for value in cluster
            if values[value - origin] >= peak - 1e-9
        ]
        coordinate = round(sum(peak_coordinates) / len(peak_coordinates))
        axes.append({
            "coordinate": int(coordinate),
            "coverage": round(float(peak), 4),
            "band": [cluster[0], cluster[-1]],
        })
    axes.sort(key=lambda item: (-item["coverage"], item["coordinate"]))
    return axes[:12]


def _border_connected_free_mask(mask: np.ndarray) -> np.ndarray:
    free = (mask == 0).astype(np.uint8)
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    if count <= 1:
        return np.zeros_like(free, dtype=bool)
    border_labels = np.unique(np.concatenate((
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1],
    )))
    border_labels = border_labels[border_labels > 0]
    return np.isin(labels, border_labels)


def _dominant_span_rectangles(
    mask: np.ndarray,
    support: np.ndarray,
    roi: list[int],
    min_room_area_px: float,
) -> list[dict]:
    vertical = _dominant_axes(support, roi, "vertical")
    horizontal = _dominant_axes(support, roi, "horizontal")
    if len(vertical) < 2 or len(horizontal) < 2:
        return []
    roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
    exterior = _border_connected_free_mask(mask).astype(np.uint8)
    integral = cv2.integral(exterior)
    barrier = (mask != 0).astype(np.uint8)
    support_binary = support > 0
    barrier_binary = barrier > 0
    vertical_profiles = {}
    for axis in vertical:
        x = axis["coordinate"]
        vertical_profiles[x] = (
            np.any(
                support_binary[:, max(0, x - 8):min(support.shape[1], x + 9)],
                axis=1,
            ),
            np.any(
                barrier_binary[:, max(0, x - 4):min(barrier.shape[1], x + 5)],
                axis=1,
            ),
        )
    horizontal_profiles = {}
    for axis in horizontal:
        y = axis["coordinate"]
        horizontal_profiles[y] = (
            np.any(
                support_binary[max(0, y - 8):min(support.shape[0], y + 9), :],
                axis=0,
            ),
            np.any(
                barrier_binary[max(0, y - 4):min(barrier.shape[0], y + 5), :],
                axis=0,
            ),
        )
    candidates = []
    for left_index, left_axis in enumerate(vertical):
        for right_axis in vertical[left_index + 1:]:
            left, right = sorted((left_axis["coordinate"], right_axis["coordinate"]))
            width = right - left
            if width < 20:
                continue
            for top_index, top_axis in enumerate(horizontal):
                for bottom_axis in horizontal[top_index + 1:]:
                    top, bottom = sorted((top_axis["coordinate"], bottom_axis["coordinate"]))
                    height = bottom - top
                    if height < 20:
                        continue
                    area = float(width * height)
                    if area < max(min_room_area_px * 4.0, roi_area * 0.05):
                        continue
                    if area > roi_area * 0.65:
                        continue
                    aspect = max(width, height) / max(1.0, min(width, height))
                    if aspect > 6.0:
                        continue
                    lines = [
                        [left, top, right, top],
                        [left, bottom, right, bottom],
                        [left, top, left, bottom],
                        [right, top, right, bottom],
                    ]
                    vector_ratios = [
                        float(np.mean(horizontal_profiles[top][0][left:right + 1])),
                        float(np.mean(horizontal_profiles[bottom][0][left:right + 1])),
                        float(np.mean(vertical_profiles[left][0][top:bottom + 1])),
                        float(np.mean(vertical_profiles[right][0][top:bottom + 1])),
                    ]
                    side_names = ("top", "bottom", "left", "right")
                    vector_support_by_side = dict(zip(side_names, vector_ratios))
                    if (
                        min(vector_support_by_side.values())
                        < DOMINANT_SPAN_MIN_VECTOR_SIDE_SUPPORT
                    ):
                        continue
                    model_ratios = [
                        float(np.mean(horizontal_profiles[top][1][left:right + 1])),
                        float(np.mean(horizontal_profiles[bottom][1][left:right + 1])),
                        float(np.mean(vertical_profiles[left][1][top:bottom + 1])),
                        float(np.mean(vertical_profiles[right][1][top:bottom + 1])),
                    ]
                    model_support_by_side = dict(zip(side_names, model_ratios))
                    if (
                        min(model_support_by_side.values())
                        < DOMINANT_SPAN_MIN_MODEL_SIDE_SUPPORT
                    ):
                        continue
                    leak_area = float(
                        integral[bottom, right]
                        - integral[top, right]
                        - integral[bottom, left]
                        + integral[top, left]
                    )
                    if leak_area < min_room_area_px:
                        continue
                    perimeter = float(2 * (width + height))
                    score = (
                        leak_area / max(1.0, perimeter)
                        * float(np.mean(vector_ratios))
                        * (0.5 + float(np.mean(model_ratios)))
                    )
                    candidates.append({
                        "lines_px": lines,
                        "bbox_px": [left, top, width, height],
                        "leak_area_px": round(leak_area, 1),
                        "vector_support_ratio": round(float(np.mean(vector_ratios)), 4),
                        "model_support_ratio": round(float(np.mean(model_ratios)), 4),
                        "vector_support_by_side": {
                            name: round(float(value), 4)
                            for name, value in vector_support_by_side.items()
                        },
                        "model_support_by_side": {
                            name: round(float(value), 4)
                            for name, value in model_support_by_side.items()
                        },
                        "score": round(score, 4),
                    })
    candidates.sort(
        key=lambda item: (-item["score"], item["bbox_px"]),
    )
    return candidates[:24]


def _search_dominant_span_rectangle(
    mask: np.ndarray,
    support: np.ndarray,
    roi: list[int],
    min_room_area_px: float,
) -> tuple[np.ndarray, dict | None]:
    baseline = _room_topology(mask, min_room_area_px)
    baseline_area = float(baseline.get("total_area_px2") or 0.0)
    solutions = []
    for candidate in _dominant_span_rectangles(
        mask,
        support,
        roi,
        min_room_area_px,
    ):
        left, top, width, height = candidate["bbox_px"]
        right, bottom = left + width, top + height
        overlaps_existing_room = False
        for room in baseline.get("rooms", []):
            room_x, room_y, room_width, room_height = room["bbox_px"]
            overlap_width = max(0, min(right, room_x + room_width) - max(left, room_x))
            overlap_height = max(0, min(bottom, room_y + room_height) - max(top, room_y))
            if overlap_width * overlap_height > max(12.0, float(room["area_px2"]) * 0.02):
                overlaps_existing_room = True
                break
        if overlaps_existing_room:
            continue
        simulated = mask.copy()
        for line in candidate["lines_px"]:
            simulated = _apply_wall_line(simulated, line)
        topology = _room_topology(simulated, min_room_area_px)
        if not _baseline_rooms_preserved(baseline, topology):
            continue
        gain = float(topology.get("total_area_px2") or 0.0) - baseline_area
        if gain < min_room_area_px or not _new_room_is_plausible(
            baseline,
            topology,
            roi,
        ):
            continue
        perimeter = sum(
            math.hypot(line[2] - line[0], line[3] - line[1])
            for line in candidate["lines_px"]
        )
        solutions.append((
            candidate["vector_support_ratio"],
            candidate["model_support_ratio"],
            gain / max(1.0, perimeter),
            gain,
            simulated,
            candidate,
        ))
    if not solutions:
        return mask, None
    solutions.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
        ),
        reverse=True,
    )
    _vector, _model, _efficiency, gain, simulated, candidate = solutions[0]
    candidate = {**candidate, "closed_area_gain_px2": round(gain, 1)}
    return simulated, candidate


def _search_multi_gap_solution(
    mask: np.ndarray,
    candidates: list[dict],
    roi: list[int],
    min_room_area_px: float,
    limits: dict,
) -> tuple[np.ndarray, list[dict]]:
    baseline = _room_topology(mask, min_room_area_px)
    baseline_area = float(baseline.get("total_area_px2") or 0.0)
    states = [(mask, [], 0.0, 0.0)]
    solutions = []
    for candidate in candidates:
        expanded = list(states)
        for state_mask, accepted, _gain, evidence_score in states:
            if len(accepted) >= limits["max_lines"]:
                continue
            simulated = _apply_wall_line(state_mask, candidate["line_px"])
            topology = _room_topology(simulated, min_room_area_px)
            if not _baseline_rooms_preserved(baseline, topology):
                continue
            gain = max(
                0.0,
                float(topology.get("total_area_px2") or 0.0) - baseline_area,
            )
            next_accepted = accepted + [candidate]
            next_state = (
                simulated,
                next_accepted,
                gain,
                evidence_score + float(candidate["score"]),
            )
            expanded.append(next_state)
            if gain >= min_room_area_px and _new_room_is_plausible(
                baseline,
                topology,
                roi,
            ):
                solutions.append(next_state)
        expanded.sort(
            key=lambda state: (
                state[2] > 0,
                state[2],
                state[3],
                -sum(item["length_px"] for item in state[1]),
            ),
            reverse=True,
        )
        states = expanded[:limits["beam_width"]]

    if not solutions:
        return mask, []
    solutions.sort(
        key=lambda state: (
            state[2],
            -len(state[1]),
            -sum(item["length_px"] for item in state[1]),
        ),
        reverse=True,
    )
    best_mask, accepted, _gain, _evidence = solutions[0]
    return best_mask, accepted


def _component_record(stats_row: np.ndarray, label: int) -> dict:
    return {
        "component": int(label),
        "bbox_px": [
            int(stats_row[cv2.CC_STAT_LEFT]),
            int(stats_row[cv2.CC_STAT_TOP]),
            int(stats_row[cv2.CC_STAT_WIDTH]),
            int(stats_row[cv2.CC_STAT_HEIGHT]),
        ],
        "area_px": int(stats_row[cv2.CC_STAT_AREA]),
    }


def _is_internal_candidate(
    stats_row: np.ndarray,
    roi: list[int],
    max_component_area_px: int,
) -> bool:
    width = int(stats_row[cv2.CC_STAT_WIDTH])
    height = int(stats_row[cv2.CC_STAT_HEIGHT])
    area = int(stats_row[cv2.CC_STAT_AREA])
    if area <= max_component_area_px:
        return True
    aspect = max(width, height) / max(1.0, min(width, height))
    roi_width = roi[2] - roi[0]
    roi_height = roi[3] - roi[1]
    slender_thickness_limit = max(5, round(min(roi_width, roi_height) * 0.04))
    slender_area_limit = max(
        max_component_area_px * 20,
        round(roi_width * roi_height * 0.005),
    )
    return bool(
        aspect >= 6.0
        and min(width, height) <= slender_thickness_limit
        and area <= slender_area_limit
    )


def repair_vector_floorplan_topology(
    class_mask: np.ndarray,
    *,
    building_roi: list[int] | None,
    structural_support_mask: np.ndarray | None,
    max_exterior_gap_px: int,
    max_internal_component_area_px: int,
    min_room_area_px: float,
) -> dict:
    """Return a conservatively repaired final mask and JSON-safe diagnostics."""
    source = np.asarray(class_mask)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("class_mask must be a non-empty 2D array")
    if max_exterior_gap_px <= 0 or max_internal_component_area_px <= 0:
        raise ValueError("repair limits must be positive")
    if not math.isfinite(float(min_room_area_px)) or min_room_area_px <= 0:
        raise ValueError("min_room_area_px must be positive and finite")

    height, width = source.shape
    roi = _normalize_roi(building_roi, width, height)
    support = None
    if structural_support_mask is not None:
        support = np.asarray(structural_support_mask)
        if support.shape != source.shape:
            raise ValueError("structural_support_mask must match class_mask")

    result = source.copy()
    initial_topology = _room_topology(result, min_room_area_px)
    diagnostics = {
        "calculation_mask": result,
        "exterior_repair": {
            "status": "not_applicable",
            "candidate_count": 0,
            "accepted_line_px": None,
            "accepted_lines_px": [],
            "reason": "missing_vector_roi_or_support",
        },
        "internal_fragments": {"ignored": [], "repaired": [], "ambiguous": []},
        "manual_exterior_wall_required": False,
        "manual_review_reasons": [],
        "closure_status": "not_applicable",
    }
    if roi is None or support is None or not np.any(support):
        return diagnostics

    barrier = (result != 0).astype(np.uint8)
    limits = _multi_gap_limits(barrier, roi, max_exterior_gap_px)
    result, span_rectangle = _search_dominant_span_rectangle(
        result,
        support,
        roi,
        min_room_area_px,
    )
    exterior_candidates = []
    accepted_candidates = []
    if span_rectangle is None:
        exterior_candidates = _ranked_gap_candidates(barrier, support, roi, limits)
        result, accepted_candidates = _search_multi_gap_solution(
            result,
            exterior_candidates,
            roi,
            min_room_area_px,
            limits,
        )

    if span_rectangle is not None:
        accepted_lines = span_rectangle["lines_px"]
        diagnostics["exterior_repair"] = {
            "status": "repaired",
            "candidate_count": 1,
            "accepted_line_px": accepted_lines[0],
            "accepted_lines_px": accepted_lines,
            "accepted_candidates": [{
                "kind": "dominant_span_rectangle",
                **span_rectangle,
            }],
            "round_count": 1,
            "limits": limits,
            "reason": "dominant_span_rectangle",
        }
    elif accepted_candidates:
        accepted_lines = [candidate["line_px"] for candidate in accepted_candidates]
        diagnostics["exterior_repair"] = {
            "status": "repaired",
            "candidate_count": len(exterior_candidates),
            "accepted_line_px": accepted_lines[0],
            "accepted_lines_px": accepted_lines,
            "accepted_candidates": accepted_candidates,
            "round_count": 1,
            "limits": limits,
            "reason": "multi_gap_solution",
        }
    else:
        diagnostics["exterior_repair"]["candidate_count"] = len(exterior_candidates)
        diagnostics["exterior_repair"].update({
            "status": "manual_exterior_wall_required",
            "accepted_lines_px": [],
            "round_count": 1,
            "limits": limits,
            "reason": "no_valid_multi_gap_solution",
        })
        diagnostics["manual_exterior_wall_required"] = True
        diagnostics["manual_review_reasons"].append("exterior_not_uniquely_repairable")

    # Internal fragments are handled after the exterior decision. First give a
    # small disconnected U-shaped component one chance to close with one line.
    barrier = (result != 0).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(barrier, 8)
    main_label = 0
    if count > 1:
        main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    repaired_components = set()
    internal_baseline = None
    for label in range(1, count):
        if label == main_label or not _is_internal_candidate(
            stats[label], roi, max_internal_component_area_px
        ):
            continue
        record = _component_record(stats[label], label)
        x, y, component_width, component_height = record["bbox_px"]
        component_mask = component_labels == label
        internal_candidates = _u_shape_candidates(component_mask, max_exterior_gap_px)
        if not internal_candidates:
            continue
        matching = []
        local_pad = max(20, max_exterior_gap_px * 2)
        local_x0 = max(0, x - local_pad)
        local_y0 = max(0, y - local_pad)
        local_x1 = min(width, x + component_width + local_pad)
        local_y1 = min(height, y + component_height + local_pad)
        local_source = result[local_y0:local_y1, local_x0:local_x1]
        local_before = _room_topology(local_source, min_room_area_px)
        local_roi = [0, 0, local_source.shape[1], local_source.shape[0]]
        for candidate in internal_candidates:
            line = candidate["line_px"]
            center_x = (line[0] + line[2]) / 2.0
            center_y = (line[1] + line[3]) / 2.0
            if not (x - 2 <= center_x <= x + component_width + 2 and y - 2 <= center_y <= y + component_height + 2):
                continue
            if _support_ratio(support, line) < 0.35:
                continue
            if not (
                _endpoint_touches_label(component_labels, tuple(line[:2]), label)
                and _endpoint_touches_label(component_labels, tuple(line[2:]), label)
            ):
                continue
            local_line = [
                line[0] - local_x0,
                line[1] - local_y0,
                line[2] - local_x0,
                line[3] - local_y0,
            ]
            local_simulated = _apply_wall_line(local_source, local_line)
            local_after = _room_topology(local_simulated, min_room_area_px)
            if not _new_room_is_plausible(local_before, local_after, local_roi):
                continue
            if internal_baseline is None:
                internal_baseline = _room_topology(result, min_room_area_px)
            simulated = _apply_wall_line(result, line)
            after = _room_topology(simulated, min_room_area_px)
            if _new_room_is_plausible(internal_baseline, after, roi):
                matching.append((candidate, simulated))
        if len(matching) == 1:
            candidate, result = matching[0]
            repaired_components.add(label)
            diagnostics["internal_fragments"]["repaired"].append({
                **record,
                "line_px": candidate["line_px"],
            })
            internal_baseline = _room_topology(result, min_room_area_px)

    # Re-label after accepted repairs, then safely omit unsupported fragments.
    barrier = (result != 0).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(barrier, 8)
    main_label = 0
    if count > 1:
        main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    free_count, free_labels, free_stats, _ = cv2.connectedComponentsWithStats(
        (barrier == 0).astype(np.uint8),
        connectivity=8,
    )
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label == main_label or not _is_internal_candidate(
            stats[label], roi, max_internal_component_area_px
        ):
            continue
        record = _component_record(stats[label], label)
        x, y, component_width, component_height = record["bbox_px"]
        edge_band = max(4, round(min(roi[2] - roi[0], roi[3] - roi[1]) * 0.05))
        if (
            x <= roi[0] + edge_band
            or y <= roi[1] + edge_band
            or x + component_width >= roi[2] - edge_band
            or y + component_height >= roi[3] - edge_band
        ):
            diagnostics["internal_fragments"]["ambiguous"].append({**record, "reason": "near_exterior"})
            continue
        component = component_labels == label
        support_ratio = float(np.count_nonzero(component & (support > 0))) / max(1.0, float(area))
        if support_ratio >= 0.15:
            diagnostics["internal_fragments"]["ambiguous"].append({**record, "reason": "vector_supported"})
            continue
        neighbor_ring = cv2.dilate(
            component.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool) & ~component
        adjacent_labels = {
            int(value)
            for value in np.unique(free_labels[neighbor_ring])
            if int(value) > 0
            and int(free_stats[int(value), cv2.CC_STAT_AREA]) >= min_room_area_px
        }
        if len(adjacent_labels) >= 2:
            diagnostics["internal_fragments"]["ambiguous"].append({**record, "reason": "changes_rooms"})
            continue
        if adjacent_labels:
            free_area = float(free_stats[next(iter(adjacent_labels)), cv2.CC_STAT_AREA])
            if area > max(12.0, free_area * 0.05):
                diagnostics["internal_fragments"]["ambiguous"].append({**record, "reason": "changes_room_area"})
                continue

        result[component] = 0
        diagnostics["internal_fragments"]["ignored"].append(record)

    final_topology = _room_topology(result, min_room_area_px)
    closed_area_gain = max(
        0.0,
        float(final_topology.get("total_area_px2") or 0.0)
        - float(initial_topology.get("total_area_px2") or 0.0),
    )
    roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
    major_closure_threshold = max(min_room_area_px * 2.0, roi_area * 0.08)
    if int(final_topology.get("room_count") or 0) <= 0:
        closure_status = "failed"
    elif closed_area_gain >= major_closure_threshold:
        closure_status = "complete"
    else:
        closure_status = "partial"
    diagnostics.update({
        "closure_status": closure_status,
        "closed_area_gain_px2": round(closed_area_gain, 1),
        "major_closure_threshold_px2": round(major_closure_threshold, 1),
        "room_count_before": int(initial_topology.get("room_count") or 0),
        "room_count_after": int(final_topology.get("room_count") or 0),
    })
    if closure_status != "complete":
        diagnostics["manual_exterior_wall_required"] = True
        if "major_space_still_unclosed" not in diagnostics["manual_review_reasons"]:
            diagnostics["manual_review_reasons"].append("major_space_still_unclosed")
    diagnostics["calculation_mask"] = result
    return diagnostics
