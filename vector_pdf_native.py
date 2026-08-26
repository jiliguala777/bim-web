"""Independent native-vector extraction for the new PDF fusion path.

This module intentionally does not import the legacy floor-plan recognizer or
its PDF helpers.  It exposes only JSON-compatible page evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
import re

import cv2
import numpy as np
import pdfplumber


_NUMBER_RE = re.compile(r"^\d{2,7}(?:\.\d+)?$")


@dataclass(frozen=True)
class NativePdfThresholds:
    axis_tolerance_pt: float = 0.5
    border_tolerance_pt: float = 2.0
    border_span_fraction: float = 0.80

    def __post_init__(self) -> None:
        values = (
            self.axis_tolerance_pt,
            self.border_tolerance_pt,
            self.border_span_fraction,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("native PDF thresholds must be finite and positive")
        if self.border_span_fraction > 1:
            raise ValueError("border_span_fraction must not exceed one")


def _stroke_rgb(value) -> list[float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        channels = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(channel) for channel in channels):
        return None
    return channels


def _neutral_brightness(rgb: list[float] | None) -> float | None:
    if rgb is None or max(rgb) - min(rgb) > 0.06:
        return None
    return float(sum(rgb) / 3.0)


def _orthogonal_segment(
    item: dict,
    tolerance: float,
) -> tuple[dict | None, bool]:
    x0, x1 = float(item["x0"]), float(item["x1"])
    top, bottom = float(item["top"]), float(item["bottom"])
    dx, dy = abs(x1 - x0), abs(bottom - top)
    if dy <= tolerance < dx:
        start = [min(x0, x1), (top + bottom) / 2.0]
        end = [max(x0, x1), (top + bottom) / 2.0]
        orientation = "horizontal"
        length = dx
    elif dx <= tolerance < dy:
        start = [(x0 + x1) / 2.0, min(top, bottom)]
        end = [(x0 + x1) / 2.0, max(top, bottom)]
        orientation = "vertical"
        length = dy
    else:
        return None, bool(dx > tolerance and dy > tolerance)
    rgb = _stroke_rgb(item.get("stroking_color"))
    return {
        "start_pt": start,
        "end_pt": end,
        "orientation": orientation,
        "length_pt": float(length),
        "width_pt": float(item.get("linewidth") or 0.0),
        "stroke_rgb": rgb,
        "neutral_brightness": _neutral_brightness(rgb),
    }, False


def _page_objects(page) -> list[dict]:
    objects = list(page.lines)
    objects.extend(getattr(page, "rect_edges", []))
    objects.extend(getattr(page, "curve_edges", []))
    unique = []
    seen = set()
    for item in objects:
        key = (
            round(float(item["x0"]), 3),
            round(float(item["top"]), 3),
            round(float(item["x1"]), 3),
            round(float(item["bottom"]), 3),
            round(float(item.get("linewidth") or 0.0), 3),
            tuple(round(value, 4) for value in (_stroke_rgb(item.get("stroking_color")) or [])),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _extract_text_spans(page) -> list[dict]:
    spans = []
    for word in page.extract_words(extra_attrs=["size", "upright"]):
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        x0, x1 = float(word["x0"]), float(word["x1"])
        top, bottom = float(word["top"]), float(word["bottom"])
        upright = bool(word.get("upright", True))
        spans.append({
            "text": text,
            "bbox_pt": [x0, top, x1, bottom],
            "center_pt": [(x0 + x1) / 2.0, (top + bottom) / 2.0],
            "direction": "horizontal" if upright else "vertical",
            "font_size": float(word.get("size") or 0.0),
        })
    return spans


def _is_page_border(
    segment: dict,
    page_size: tuple[float, float],
    thresholds: NativePdfThresholds,
) -> bool:
    page_width, page_height = page_size
    (x0, y0), (x1, y1) = segment["start_pt"], segment["end_pt"]
    if segment["orientation"] == "horizontal":
        near_edge = min(abs(y0), abs(page_height - y0)) <= thresholds.border_tolerance_pt
        return near_edge and abs(x1 - x0) >= page_width * thresholds.border_span_fraction
    near_edge = min(abs(x0), abs(page_width - x0)) <= thresholds.border_tolerance_pt
    return near_edge and abs(y1 - y0) >= page_height * thresholds.border_span_fraction


def _detect_dimension_candidates(
    spans: list[dict],
    segments: list[dict],
) -> list[dict]:
    results = []
    for span in spans:
        if not _NUMBER_RE.fullmatch(span["text"]):
            continue
        orientation = span["direction"]
        baselines = [item for item in segments if item["orientation"] == orientation]
        perpendicular = [item for item in segments if item["orientation"] != orientation]
        cx, cy = span["center_pt"]
        ranked = []
        for baseline in baselines:
            (x0, y0), (x1, y1) = baseline["start_pt"], baseline["end_pt"]
            axis_center = (x0 + x1) / 2.0 if orientation == "horizontal" else (y0 + y1) / 2.0
            text_axis = cx if orientation == "horizontal" else cy
            cross_distance = abs((y0 if orientation == "horizontal" else x0) - (cy if orientation == "horizontal" else cx))
            if cross_distance > 30.0 or abs(axis_center - text_axis) > baseline["length_pt"] * 0.65:
                continue
            endpoints = []
            for marker in perpendicular:
                (mx0, my0), (mx1, my1) = marker["start_pt"], marker["end_pt"]
                if marker["length_pt"] > 40.0:
                    continue
                if orientation == "horizontal":
                    marker_axis = mx0
                    crosses = min(my0, my1) - 1.0 <= y0 <= max(my0, my1) + 1.0
                    distance = min(abs(marker_axis - x0), abs(marker_axis - x1))
                else:
                    marker_axis = my0
                    crosses = min(mx0, mx1) - 1.0 <= x0 <= max(mx0, mx1) + 1.0
                    distance = min(abs(marker_axis - y0), abs(marker_axis - y1))
                if crosses and distance <= 2.0:
                    endpoints.append(marker)
            distinct_ends = []
            for marker in endpoints:
                coordinate = marker["start_pt"][0 if orientation == "horizontal" else 1]
                if all(abs(coordinate - existing[0]) > 1.0 for existing in distinct_ends):
                    distinct_ends.append((coordinate, marker))
            if len(distinct_ends) >= 2:
                selected = [min(distinct_ends, key=lambda pair: pair[0])[1], max(distinct_ends, key=lambda pair: pair[0])[1]]
                ranked.append((cross_distance, -baseline["length_pt"], baseline, selected))
        if not ranked:
            continue
        _, _, baseline, endpoints = min(ranked, key=lambda item: (item[0], item[1]))
        results.append({
            "dimension_id": f"dimension-{len(results) + 1:04d}",
            "text": span["text"],
            "text_bbox_pt": span["bbox_pt"],
            "orientation": orientation,
            "dimension_line_id": baseline["native_id"],
            "endpoint_line_ids": [item["native_id"] for item in endpoints],
            "member_native_ids": [baseline["native_id"], *(item["native_id"] for item in endpoints)],
        })
    return results


def _building_roi(
    segments: list[dict],
    dimension_candidates: list[dict],
    page_size: tuple[float, float],
    render_size: tuple[int, int],
) -> dict:
    excluded = {
        native_id
        for candidate in dimension_candidates
        for native_id in candidate["member_native_ids"]
    }
    structural = [
        item for item in segments
        if item["native_id"] not in excluded
        and not item["is_page_border"]
        and item["neutral_brightness"] is not None
        and item["neutral_brightness"] <= 0.25
    ]
    disabled = {
        "enabled": False,
        "bbox_px": None,
        "confidence": 0.0,
        "reason": "insufficient_dark_geometry",
        "dark_edge_count": len(structural),
    }
    if len(structural) < 4:
        return disabled
    xs = [point[0] for item in structural for point in (item["start_pt"], item["end_pt"])]
    ys = [point[1] for item in structural for point in (item["start_pt"], item["end_pt"])]
    page_width, page_height = page_size
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    area_fraction = max(0.0, x1 - x0) * max(0.0, y1 - y0) / max(1.0, page_width * page_height)
    if area_fraction < 0.05 or area_fraction > 0.95:
        result = dict(disabled)
        result.update({"reason": "implausible_roi_area", "area_fraction": area_fraction})
        return result
    render_width, render_height = render_size
    margin_x, margin_y = page_width * 0.02, page_height * 0.02
    bbox_pt = [
        max(0.0, x0 - margin_x), max(0.0, y0 - margin_y),
        min(page_width, x1 + margin_x), min(page_height, y1 + margin_y),
    ]
    bbox_px = [
        round(bbox_pt[0] * render_width / page_width),
        round(bbox_pt[1] * render_height / page_height),
        round(bbox_pt[2] * render_width / page_width),
        round(bbox_pt[3] * render_height / page_height),
    ]
    return {
        "enabled": True,
        "bbox_px": bbox_px,
        "bbox_pt": bbox_pt,
        "confidence": round(min(0.99, len(structural) / 12.0), 4),
        "reason": "dark_native_geometry_bounds",
        "area_fraction": area_fraction,
        "dark_edge_count": len(structural),
    }


def extract_native_pdf_page(
    pdf_path: str | Path,
    page_index: int = 0,
    dpi: int = 100,
    *,
    thresholds: NativePdfThresholds | None = None,
) -> dict:
    """Extract one vector-PDF page into the new fusion evidence contract."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    config = thresholds or NativePdfThresholds()
    source = Path(pdf_path).resolve(strict=True)
    with pdfplumber.open(str(source)) as document:
        if page_index < 0 or page_index >= len(document.pages):
            raise IndexError("page_index is outside the PDF page range")
        page = document.pages[page_index]
        page_size = (float(page.width), float(page.height))
        render_size = (
            round(page_size[0] * dpi / 72.0),
            round(page_size[1] * dpi / 72.0),
        )
        spans = _extract_text_spans(page)
        segments = []
        styled_edges = []
        ignored_diagonal_count = 0
        for item in _page_objects(page):
            rgb = _stroke_rgb(item.get("stroking_color"))
            width_pt = float(item.get("linewidth") or 0.0)
            if rgb is not None and width_pt > 0:
                styled_edges.append({
                    "start_pt": [float(item["x0"]), float(item["top"])],
                    "end_pt": [float(item["x1"]), float(item["bottom"])],
                    "stroke_rgb": rgb,
                    "width_pt": width_pt,
                })
            segment, is_diagonal = _orthogonal_segment(item, config.axis_tolerance_pt)
            ignored_diagonal_count += int(is_diagonal)
            if segment is not None:
                segments.append(segment)

    segments.sort(key=lambda item: (
        item["orientation"],
        *(round(value, 4) for value in (*item["start_pt"], *item["end_pt"])),
        round(item["width_pt"], 4),
        tuple(round(value, 4) for value in (item["stroke_rgb"] or [])),
    ))
    for index, segment in enumerate(segments, 1):
        segment["native_id"] = f"native-{index:05d}"
        segment["is_page_border"] = _is_page_border(segment, page_size, config)

    dimensions = _detect_dimension_candidates(spans, segments)
    opening_curve_edges = _opening_curve_edges(
        page, page_size, render_size, config,
    )
    opening_short_segments = _opening_short_segments(
        segments, dimensions, page_size, render_size,
    )
    roi = _building_roi(segments, dimensions, page_size, render_size)
    has_vector_geometry = len(styled_edges) >= 4 or len(segments) >= 4
    return {
        "format": "native-vector-pdf-page/1",
        "page_index": int(page_index),
        "page_size_pt": [page_size[0], page_size[1]],
        "render_size_px": [render_size[0], render_size[1]],
        "dpi": int(dpi),
        "text_spans": spans,
        "orthogonal_segments": segments,
        "opening_curve_edges": opening_curve_edges,
        "opening_short_segments": opening_short_segments,
        "styled_edges": styled_edges,
        "dimension_candidates": dimensions,
        "building_roi": roi,
        "has_vector_geometry": has_vector_geometry,
        "is_vector_pdf": has_vector_geometry,
        "ignored_diagonal_count": ignored_diagonal_count,
    }


def _point_to_pixel(
    point: list[float],
    page_size: list[float],
    render_size: list[int],
) -> list[int]:
    return [
        round(float(point[0]) * int(render_size[0]) / float(page_size[0])),
        round(float(point[1]) * int(render_size[1]) / float(page_size[1])),
    ]


def _bbox_to_pixel(
    bbox: list[float],
    page_size: tuple[float, float] | list[float],
    render_size: tuple[int, int] | list[int],
) -> list[int]:
    first = _point_to_pixel(bbox[:2], page_size, render_size)
    second = _point_to_pixel(bbox[2:], page_size, render_size)
    return [first[0], first[1], second[0], second[1]]


def _opening_curve_edges(page, page_size, render_size, thresholds) -> list[dict]:
    curves = []
    page_width, page_height = page_size
    for item in getattr(page, "curves", []):
        path_ops = tuple(str(command[0]).lower() for command in item.get("path", []))
        has_bezier = "c" in path_ops
        is_closed = "h" in path_ops
        # Door swing arcs are open Bezier paths. Closed outlines are commonly
        # columns, furniture or glyph shapes and must not become door evidence.
        if not has_bezier or is_closed:
            continue
        bbox_pt = [
            float(item["x0"]), float(item["top"]),
            float(item["x1"]), float(item["bottom"]),
        ]
        width = abs(bbox_pt[2] - bbox_pt[0])
        height = abs(bbox_pt[3] - bbox_pt[1])
        if (
            width <= thresholds.axis_tolerance_pt
            or height <= thresholds.axis_tolerance_pt
            or (width >= page_width * thresholds.border_span_fraction
                and height >= page_height * thresholds.border_span_fraction)
        ):
            continue
        curves.append({
            "bbox_pt": bbox_pt,
            "bbox_px": _bbox_to_pixel(bbox_pt, page_size, render_size),
            "start_px": _point_to_pixel([bbox_pt[0], bbox_pt[1]], page_size, render_size),
            "end_px": _point_to_pixel([bbox_pt[2], bbox_pt[3]], page_size, render_size),
            "stroke_rgb": _stroke_rgb(item.get("stroking_color")),
            "width_pt": float(item.get("linewidth") or 0.0),
            "path_ops": list(path_ops),
            "has_bezier": has_bezier,
            "is_closed": is_closed,
        })
    curves.sort(key=lambda item: tuple(item["bbox_pt"]))
    for index, curve in enumerate(curves, 1):
        curve["curve_id"] = f"curve-{index:04d}"
    return curves


def _opening_short_segments(
    segments: list[dict],
    dimensions: list[dict],
    page_size: tuple[float, float],
    render_size: tuple[int, int],
) -> list[dict]:
    excluded = {
        native_id
        for dimension in dimensions
        for native_id in dimension["member_native_ids"]
    }
    short_segments = []
    for segment in segments:
        if (
            segment["native_id"] in excluded
            or segment["is_page_border"]
            or not 4.0 <= segment["length_pt"] <= 80.0
        ):
            continue
        item = copy.deepcopy(segment)
        item["start_px"] = _point_to_pixel(
            item["start_pt"], page_size, render_size,
        )
        item["end_px"] = _point_to_pixel(
            item["end_pt"], page_size, render_size,
        )
        short_segments.append(item)
    return short_segments


def build_dimension_mask(page_data: dict) -> np.ndarray:
    """Rasterize only dimension text and explicitly matched native members."""
    render_width, render_height = (int(value) for value in page_data["render_size_px"])
    mask = np.zeros((render_height, render_width), dtype=np.uint8)
    by_id = {
        item["native_id"]: item
        for item in page_data.get("orthogonal_segments", [])
    }
    page_size = page_data["page_size_pt"]
    render_size = page_data["render_size_px"]
    for candidate in page_data.get("dimension_candidates", []):
        x0, top, x1, bottom = candidate["text_bbox_pt"]
        first = _point_to_pixel([x0, top], page_size, render_size)
        second = _point_to_pixel([x1, bottom], page_size, render_size)
        cv2.rectangle(
            mask,
            (max(0, first[0] - 2), max(0, first[1] - 2)),
            (min(render_width - 1, second[0] + 2), min(render_height - 1, second[1] + 2)),
            255,
            thickness=-1,
        )
        for native_id in candidate["member_native_ids"]:
            segment = by_id.get(native_id)
            if segment is None:
                continue
            start = _point_to_pixel(segment["start_pt"], page_size, render_size)
            end = _point_to_pixel(segment["end_pt"], page_size, render_size)
            scale = max(
                render_width / float(page_size[0]),
                render_height / float(page_size[1]),
            )
            thickness = max(2, round(float(segment.get("width_pt") or 0.0) * scale) + 1)
            cv2.line(mask, start, end, 255, thickness=thickness, lineType=cv2.LINE_8)
    if page_data.get("dimension_candidates"):
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def crop_native_page_data(page_data: dict, bbox_px: list[int]) -> dict:
    """Clip orthogonal candidates into a crop while retaining page coordinates."""
    if not isinstance(bbox_px, list) or len(bbox_px) != 4:
        raise ValueError("bbox_px must contain four integers")
    left, top, right, bottom = (int(value) for value in bbox_px)
    page_width, page_height = (int(value) for value in page_data["render_size_px"])
    if left < 0 or top < 0 or right > page_width or bottom > page_height:
        raise ValueError("bbox_px must stay inside the rendered page")
    if right <= left or bottom <= top:
        raise ValueError("bbox_px must have positive area")

    result = copy.deepcopy(page_data)
    cropped_segments = []
    for segment in page_data.get("orthogonal_segments", []):
        page_start = _point_to_pixel(
            segment["start_pt"], page_data["page_size_pt"], page_data["render_size_px"]
        )
        page_end = _point_to_pixel(
            segment["end_pt"], page_data["page_size_pt"], page_data["render_size_px"]
        )
        if segment["orientation"] == "horizontal":
            y = page_start[1]
            x0, x1 = sorted((page_start[0], page_end[0]))
            if y < top or y > bottom or x1 < left or x0 > right:
                continue
            clipped_start = [max(x0, left), y]
            clipped_end = [min(x1, right), y]
        else:
            x = page_start[0]
            y0, y1 = sorted((page_start[1], page_end[1]))
            if x < left or x > right or y1 < top or y0 > bottom:
                continue
            clipped_start = [x, max(y0, top)]
            clipped_end = [x, min(y1, bottom)]
        if clipped_start == clipped_end:
            continue
        item = copy.deepcopy(segment)
        item.update({
            "page_start_px": page_start,
            "page_end_px": page_end,
            "start_px": [clipped_start[0] - left, clipped_start[1] - top],
            "end_px": [clipped_end[0] - left, clipped_end[1] - top],
        })
        cropped_segments.append(item)

    result["orthogonal_segments"] = cropped_segments
    cropped_curves = []
    for curve in page_data.get("opening_curve_edges", []):
        x0, y0, x1, y1 = (int(value) for value in curve["bbox_px"])
        ix0, iy0 = max(left, x0), max(top, y0)
        ix1, iy1 = min(right, x1), min(bottom, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        item = copy.deepcopy(curve)
        item["page_bbox_px"] = [x0, y0, x1, y1]
        item["bbox_px"] = [ix0 - left, iy0 - top, ix1 - left, iy1 - top]
        item["page_start_px"] = copy.deepcopy(curve["start_px"])
        item["page_end_px"] = copy.deepcopy(curve["end_px"])
        item["start_px"] = [
            min(max(0, curve["start_px"][0] - left), right - left),
            min(max(0, curve["start_px"][1] - top), bottom - top),
        ]
        item["end_px"] = [
            min(max(0, curve["end_px"][0] - left), right - left),
            min(max(0, curve["end_px"][1] - top), bottom - top),
        ]
        cropped_curves.append(item)

    cropped_short_segments = []
    for segment in page_data.get("opening_short_segments", []):
        start, end = segment["start_px"], segment["end_px"]
        if segment["orientation"] == "horizontal":
            y = int(start[1])
            x0, x1 = sorted((int(start[0]), int(end[0])))
            if y < top or y > bottom or x1 < left or x0 > right:
                continue
            local_start, local_end = [max(x0, left) - left, y - top], [min(x1, right) - left, y - top]
        else:
            x = int(start[0])
            y0, y1 = sorted((int(start[1]), int(end[1])))
            if x < left or x > right or y1 < top or y0 > bottom:
                continue
            local_start, local_end = [x - left, max(y0, top) - top], [x - left, min(y1, bottom) - top]
        if local_start == local_end:
            continue
        item = copy.deepcopy(segment)
        item["page_start_px"] = copy.deepcopy(start)
        item["page_end_px"] = copy.deepcopy(end)
        item["start_px"] = local_start
        item["end_px"] = local_end
        cropped_short_segments.append(item)

    result["opening_curve_edges"] = cropped_curves
    result["opening_short_segments"] = cropped_short_segments
    result["render_size_px"] = [right - left, bottom - top]
    result["crop_bbox_page_px"] = [left, top, right, bottom]
    roi = copy.deepcopy(page_data.get("building_roi") or {"enabled": False, "bbox_px": None})
    if roi.get("enabled") and roi.get("bbox_px"):
        rx0, ry0, rx1, ry1 = (int(value) for value in roi["bbox_px"])
        ix0, iy0 = max(left, rx0), max(top, ry0)
        ix1, iy1 = min(right, rx1), min(bottom, ry1)
        if ix1 > ix0 and iy1 > iy0:
            roi["bbox_page_px"] = [rx0, ry0, rx1, ry1]
            roi["bbox_px"] = [ix0 - left, iy0 - top, ix1 - left, iy1 - top]
        else:
            roi.update({"enabled": False, "bbox_px": None, "reason": "roi_outside_crop"})
    result["building_roi"] = roi
    return result
