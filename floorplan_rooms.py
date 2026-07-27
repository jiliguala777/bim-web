"""从语义分割掩膜中修补边界并提取闭合房间多边形。"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np


def apply_scale_to_room_topology(topology: dict, scale_m_per_px: float | None) -> dict:
    """Return a scaled copy; an absent or invalid scale never fabricates square metres."""
    result = copy.deepcopy(topology)
    valid_scale = None
    if scale_m_per_px is not None:
        candidate = float(scale_m_per_px)
        if math.isfinite(candidate) and candidate > 0:
            valid_scale = candidate

    for room in result.get("rooms", []):
        area_px2 = float(room.get("area_px2") or 0.0)
        room["area_m2"] = area_px2 * valid_scale ** 2 if valid_scale is not None else None
    total_area_px2 = float(result.get("total_area_px2") or 0.0)
    result["total_area_m2"] = (
        total_area_px2 * valid_scale ** 2 if valid_scale is not None else None
    )
    result["scale_m_per_px"] = valid_scale
    result["load_geometry_ready"] = bool(
        valid_scale is not None and int(result.get("room_count") or 0) > 0
    )
    return result


def _close_short_gaps(barrier: np.ndarray, max_gap_px: int) -> np.ndarray:
    """只沿水平、垂直方向连接短缺口，避免大面积膨胀墙体。"""
    if max_gap_px <= 0:
        return barrier.copy()

    span = max_gap_px * 2 + 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, span))
    horizontal = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, horizontal_kernel)
    vertical = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, vertical_kernel)
    return cv2.bitwise_or(horizontal, vertical)


def _component_touches_border(stats_row: np.ndarray, width: int, height: int) -> bool:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x <= 0 or y <= 0 or x + w >= width or y + h >= height


def extract_room_topology(
    class_mask: np.ndarray,
    *,
    max_gap_px: int = 6,
    min_room_area_px: float = 500.0,
    scale_m_per_px: float | None = None,
) -> dict:
    """提取不与图像边缘相通的空白区域，作为闭合房间。

    墙、窗、门以及其他非背景类别都统一视为边界。这样类别不确定时
    默认按墙处理，不会因为门窗分类错误而主动打断房间闭合。
    """
    mask = np.asarray(class_mask)
    if mask.ndim != 2:
        raise ValueError("class_mask must be a 2D array")
    if mask.size == 0:
        raise ValueError("class_mask must not be empty")
    if max_gap_px < 0:
        raise ValueError("max_gap_px must be non-negative")
    if min_room_area_px <= 0 or not math.isfinite(float(min_room_area_px)):
        raise ValueError("min_room_area_px must be positive and finite")
    if scale_m_per_px is not None:
        scale_m_per_px = float(scale_m_per_px)
        if not math.isfinite(scale_m_per_px) or scale_m_per_px <= 0:
            raise ValueError("scale_m_per_px must be positive and finite")

    # 所有非背景识别结果都参与闭合；不确定构件等价于不透明墙体。
    original_barrier = (mask != 0).astype(np.uint8) * 255
    closed_barrier = _close_short_gaps(original_barrier, int(max_gap_px))
    closure_applied = bool(np.any((closed_barrier > 0) & (original_barrier == 0)))

    free_space = (closed_barrier == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=8)
    height, width = mask.shape
    rooms = []

    for label in range(1, count):
        if _component_touches_border(stats[label], width, height):
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area_px2 = float(cv2.contourArea(contour))
        if area_px2 < min_room_area_px:
            continue

        epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        points = [[float(point[0][0]), float(point[0][1])] for point in polygon]
        x, y, w, h = cv2.boundingRect(contour)
        room = {
            "id": f"room-{len(rooms) + 1}",
            "polygon_px": points,
            "area_px2": area_px2,
            "bbox_px": [int(x), int(y), int(w), int(h)],
            "area_m2": None,
        }
        if scale_m_per_px is not None:
            room["area_m2"] = area_px2 * scale_m_per_px * scale_m_per_px
        rooms.append(room)

    rooms.sort(key=lambda room: (room["bbox_px"][1], room["bbox_px"][0]))
    for index, room in enumerate(rooms, start=1):
        room["id"] = f"room-{index}"

    total_area_px2 = float(sum(room["area_px2"] for room in rooms))
    total_area_m2 = None
    if scale_m_per_px is not None:
        total_area_m2 = total_area_px2 * scale_m_per_px * scale_m_per_px

    return {
        "status": "closed" if rooms else "no_closed_rooms",
        "room_count": len(rooms),
        "closure_applied": closure_applied,
        "max_gap_px": int(max_gap_px),
        "min_room_area_px": float(min_room_area_px),
        "rooms": rooms,
        "total_area_px2": total_area_px2,
        "total_area_m2": total_area_m2,
        "scale_m_per_px": scale_m_per_px,
        "load_geometry_ready": bool(rooms and scale_m_per_px is not None),
    }
