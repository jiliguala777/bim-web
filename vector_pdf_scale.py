"""Extract vector-PDF text and line geometry for architectural scale calibration."""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

import cv2
import numpy as np
import pdfplumber


def _parse_dimension_metres(text: str) -> float | None:
    normalized = text.strip().lower().replace(" ", "")
    if ":" in normalized or "²" in normalized or "㎡" in normalized:
        return None
    if re.search(r"[a-z]", normalized) and not normalized.endswith(("mm", "m")):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(mm|m)?", normalized)
    if not match:
        return None
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0:
        return None
    unit = match.group(2) or "mm"
    if unit == "mm" and "." in match.group(1) and value < 100:
        return None
    return value / 1000.0 if unit == "mm" else value


def _perpendicular_distance(span: dict, segment: dict) -> float:
    if segment["orientation"] == "horizontal":
        return abs(span["center_pt"][1] - segment["start_pt"][1])
    return abs(span["center_pt"][0] - segment["start_pt"][0])


def _segment_overlaps_text_axis(span: dict, segment: dict, tolerance: float = 2.0) -> bool:
    center_x, center_y = span["center_pt"]
    if segment["orientation"] == "horizontal":
        return segment["start_pt"][0] - tolerance <= center_x <= segment["end_pt"][0] + tolerance
    return segment["start_pt"][1] - tolerance <= center_y <= segment["end_pt"][1] + tolerance


def _endpoint_segments(main: dict, all_segments: list[dict]) -> list[dict]:
    evidence = []
    coordinate_tolerance = 1.5
    if main["orientation"] == "horizontal":
        main_y = main["start_pt"][1]
        for endpoint_x in (main["start_pt"][0], main["end_pt"][0]):
            matches = [
                segment for segment in all_segments
                if segment["orientation"] == "vertical"
                and abs(segment["start_pt"][0] - endpoint_x) <= coordinate_tolerance
                and segment["start_pt"][1] - coordinate_tolerance <= main_y <= segment["end_pt"][1] + coordinate_tolerance
            ]
            if matches:
                evidence.append(min(matches, key=lambda item: item["length_pt"]))
    else:
        main_x = main["start_pt"][0]
        for endpoint_y in (main["start_pt"][1], main["end_pt"][1]):
            matches = [
                segment for segment in all_segments
                if segment["orientation"] == "horizontal"
                and abs(segment["start_pt"][1] - endpoint_y) <= coordinate_tolerance
                and segment["start_pt"][0] - coordinate_tolerance <= main_x <= segment["end_pt"][0] + coordinate_tolerance
            ]
            if matches:
                evidence.append(min(matches, key=lambda item: item["length_pt"]))
    return evidence


def _candidate_position(segment: dict, page_size: list[float]) -> str:
    if segment["orientation"] == "horizontal":
        return "top" if segment["start_pt"][1] < page_size[1] / 2 else "bottom"
    return "left" if segment["start_pt"][0] < page_size[0] / 2 else "right"


def _is_page_border(segment: dict, page_size: list[float], margin: float = 3.0) -> bool:
    width, height = page_size
    if segment["orientation"] == "horizontal":
        y = segment["start_pt"][1]
        return (y <= margin or y >= height - margin) and segment["length_pt"] >= width * 0.8
    x = segment["start_pt"][0]
    return (x <= margin or x >= width - margin) and segment["length_pt"] >= height * 0.8


def _dimension_text(span: dict) -> str:
    text = str(span.get("text", "")).strip()
    if span.get("direction") == "vertical" and text.startswith("0"):
        reversed_text = text[::-1]
        if reversed_text and not reversed_text.startswith("0"):
            return reversed_text
    return text


def _merge_split_dimension_segments(span: dict, segments: list[dict]) -> list[dict]:
    """Bridge two collinear dimension fragments separated by the text box."""
    orientation = span.get("direction")
    bbox = span["bbox_pt"]
    center_x, center_y = span["center_pt"]
    max_distance = max(24.0, float(span.get("font_size") or 0.0) * 3.0)
    if orientation == "horizontal":
        same_axis = [
            item for item in segments
            if item["orientation"] == orientation
            and abs(item["start_pt"][1] - center_y) <= max_distance
        ]
        first_items = [
            item for item in same_axis
            if bbox[0] - 12.0 <= item["end_pt"][0] <= bbox[0] + 3.0
            and item["start_pt"][0] < bbox[0]
        ]
        second_items = [
            item for item in same_axis
            if bbox[2] - 3.0 <= item["start_pt"][0] <= bbox[2] + 12.0
            and item["end_pt"][0] > bbox[2]
        ]
    else:
        same_axis = [
            item for item in segments
            if item["orientation"] == orientation
            and abs(item["start_pt"][0] - center_x) <= max_distance
        ]
        first_items = [
            item for item in same_axis
            if bbox[1] - 12.0 <= item["end_pt"][1] <= bbox[1] + 3.0
            and item["start_pt"][1] < bbox[1]
        ]
        second_items = [
            item for item in same_axis
            if bbox[3] - 3.0 <= item["start_pt"][1] <= bbox[3] + 12.0
            and item["end_pt"][1] > bbox[3]
        ]
    merged = []
    for first in first_items:
        for second in second_items:
            if first is second:
                continue
            if orientation == "horizontal":
                if abs(first["start_pt"][1] - second["start_pt"][1]) > 1.0:
                    continue
                left, right = sorted((first, second), key=lambda item: item["start_pt"][0])
                gap_start, gap_end = left["end_pt"][0], right["start_pt"][0]
                text_start, text_end = bbox[0], bbox[2]
                start = [left["start_pt"][0], left["start_pt"][1]]
                end = [right["end_pt"][0], right["end_pt"][1]]
            else:
                if abs(first["start_pt"][0] - second["start_pt"][0]) > 1.0:
                    continue
                top, bottom = sorted((first, second), key=lambda item: item["start_pt"][1])
                gap_start, gap_end = top["end_pt"][1], bottom["start_pt"][1]
                text_start, text_end = bbox[1], bbox[3]
                start = [top["start_pt"][0], top["start_pt"][1]]
                end = [bottom["end_pt"][0], bottom["end_pt"][1]]
            text_size = text_end - text_start
            gap_size = gap_end - gap_start
            if gap_size < 0 or gap_size > text_size + 12.0:
                continue
            if gap_start > text_start + 3.0 or gap_end < text_end - 3.0:
                continue
            merged.append({
                "start_pt": start,
                "end_pt": end,
                "orientation": orientation,
                "length_pt": (end[0] - start[0]) if orientation == "horizontal" else (end[1] - start[1]),
                "width_pt": max(float(first.get("width_pt") or 0.0), float(second.get("width_pt") or 0.0)),
                "merged_from": [first, second],
            })
    return merged


def detect_dimension_candidates(page_data: dict) -> list[dict]:
    """Match numeric dimension text to a parallel line and two endpoint markers."""
    if not page_data.get("is_vector_pdf"):
        return []
    segments = page_data.get("segments") or []
    dpi = float(page_data.get("dpi") or 200)
    zoom = dpi / 72.0
    candidates = []
    for span in page_data.get("text_spans") or []:
        dimension_text = _dimension_text(span)
        actual_length_m = _parse_dimension_metres(dimension_text)
        if actual_length_m is None:
            continue
        orientation = span.get("direction")
        match_segments = [*segments, *_merge_split_dimension_segments(span, segments)]
        parallel = [
            segment for segment in match_segments
            if segment["orientation"] == orientation
            and not _is_page_border(segment, page_data["page_size_pt"])
            and _segment_overlaps_text_axis(span, segment)
            and segment["length_pt"] >= max(12.0, span.get("font_size", 0.0) * 2.0)
        ]
        max_distance = max(24.0, float(span.get("font_size") or 0.0) * 3.0)
        parallel = [
            segment for segment in parallel
            if _perpendicular_distance(span, segment) <= max_distance
        ]
        if not parallel:
            continue
        main = min(parallel, key=lambda item: (_perpendicular_distance(span, item), -item["length_pt"]))
        endpoint_segments = _endpoint_segments(main, segments)
        if len(endpoint_segments) != 2:
            continue
        span_px = main["length_pt"] * zoom
        candidates.append({
            "text": dimension_text,
            "text_source": span.get("source", "pdf_text"),
            "text_bbox_pt": list(span["bbox_pt"]),
            "orientation": orientation,
            "actual_length_m": actual_length_m,
            "span_pt": float(main["length_pt"]),
            "span_px": float(span_px),
            "scale_m_per_px": actual_length_m / span_px,
            "position": _candidate_position(main, page_data["page_size_pt"]),
            "endpoint_evidence": len(endpoint_segments),
            "dimension_line": main,
            "endpoint_segments": endpoint_segments,
        })
    return candidates


def calibrate_from_overall_dimensions(page_data: dict) -> dict:
    """Select overall dimensions and validate horizontal/vertical scale agreement."""
    candidates = detect_dimension_candidates(page_data)
    selection_candidates = candidates
    if len(candidates) >= 3 and any(item.get("text_source") == "rapidocr" for item in candidates):
        clusters = []
        for seed in candidates:
            seed_scale = seed["scale_m_per_px"]
            cluster = [
                item for item in candidates
                if abs(item["scale_m_per_px"] - seed_scale)
                / ((item["scale_m_per_px"] + seed_scale) / 2.0) <= 0.02
            ]
            clusters.append(cluster)
        consensus = max(
            clusters,
            key=lambda items: (len(items), sum(item["span_pt"] for item in items)),
        )
        if (
            len(consensus) >= 2
            and {item["orientation"] for item in consensus} == {"horizontal", "vertical"}
        ):
            selection_candidates = consensus
    horizontal_items = [item for item in selection_candidates if item["orientation"] == "horizontal"]
    vertical_items = [item for item in selection_candidates if item["orientation"] == "vertical"]
    horizontal = max(horizontal_items, key=lambda item: item["actual_length_m"], default=None)
    vertical = max(vertical_items, key=lambda item: item["actual_length_m"], default=None)
    selected = [item for item in (horizontal, vertical) if item is not None]

    if not selected:
        return {
            "status": "manual_required",
            "method": "vector_pdf_overall_dimensions",
            "scale_m_per_px": None,
            "confidence": 0.0,
            "horizontal": None,
            "vertical": None,
            "axis_difference_percent": None,
            "evidence": candidates,
        }

    if len(selected) == 1:
        return {
            "status": "confirmation_required",
            "method": "vector_pdf_overall_dimensions",
            "scale_m_per_px": selected[0]["scale_m_per_px"],
            "confidence": 0.5,
            "horizontal": horizontal,
            "vertical": vertical,
            "axis_difference_percent": None,
            "evidence": candidates,
        }

    scale_x = horizontal["scale_m_per_px"]
    scale_y = vertical["scale_m_per_px"]
    difference = abs(scale_x - scale_y) / ((scale_x + scale_y) / 2.0)
    epsilon = 1e-9
    if difference <= 0.02 + epsilon:
        status, confidence = "confirmed", 0.95
    elif difference <= 0.05 + epsilon:
        status, confidence = "confirmation_required", 0.65
    else:
        status, confidence = "conflict", 0.0
    return {
        "status": status,
        "method": "vector_pdf_overall_dimensions",
        "scale_m_per_px": statistics.median((scale_x, scale_y)) if status != "conflict" else None,
        "confidence": confidence,
        "horizontal": horizontal,
        "vertical": vertical,
        "axis_difference_percent": difference * 100.0,
        "evidence": candidates,
    }


def _point_to_pixel(point: list[float], scale_x: float, scale_y: float) -> tuple[int, int]:
    return round(point[0] * scale_x), round(point[1] * scale_y)


def build_dimension_annotation_mask(page_data: dict, candidates: list[dict]) -> np.ndarray:
    """Rasterize accepted dimension text, baselines, and endpoint markers."""
    render_width, render_height = page_data["render_size_px"]
    page_width, page_height = page_data["page_size_pt"]
    scale_x = render_width / page_width
    scale_y = render_height / page_height
    mask = np.zeros((render_height, render_width), dtype=np.uint8)

    for candidate in candidates:
        x0, top, x1, bottom = candidate["text_bbox_pt"]
        pad = 2
        cv2.rectangle(
            mask,
            (max(0, round(x0 * scale_x) - pad), max(0, round(top * scale_y) - pad)),
            (min(render_width - 1, round(x1 * scale_x) + pad), min(render_height - 1, round(bottom * scale_y) + pad)),
            255,
            thickness=-1,
        )
        annotation_segments = [candidate["dimension_line"], *candidate["endpoint_segments"]]
        for segment in annotation_segments:
            start = _point_to_pixel(segment["start_pt"], scale_x, scale_y)
            end = _point_to_pixel(segment["end_pt"], scale_x, scale_y)
            thickness = max(2, round(float(segment.get("width_pt") or 0.0) * max(scale_x, scale_y)) + 1)
            cv2.line(mask, start, end, 255, thickness=thickness)

    if candidates:
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def _neutral_brightness(edge: dict) -> float | None:
    rgb = edge.get("stroke_rgb")
    if not isinstance(rgb, list) or len(rgb) < 3:
        return None
    channels = [float(value) for value in rgb[:3]]
    if max(channels) - min(channels) > 0.06:
        return None
    return sum(channels) / 3.0


def _vector_layer_counts(page_data: dict) -> tuple[list[dict], list[dict]]:
    black_edges = []
    gray_edges = []
    for edge in page_data.get("styled_edges") or []:
        brightness = _neutral_brightness(edge)
        if brightness is None:
            continue
        if brightness <= 0.18:
            black_edges.append(edge)
        elif 0.35 <= brightness <= 0.75:
            gray_edges.append(edge)
    return black_edges, gray_edges


def _draw_edges(
    edges: list[dict],
    size: tuple[int, int],
    page_size: list[float],
    *,
    extra_thickness: int = 1,
) -> np.ndarray:
    width, height = size
    page_width, page_height = page_size
    scale_x = width / float(page_width)
    scale_y = height / float(page_height)
    mask = np.zeros((height, width), dtype=np.uint8)
    for edge in edges:
        start = _point_to_pixel(edge["start_pt"], scale_x, scale_y)
        end = _point_to_pixel(edge["end_pt"], scale_x, scale_y)
        thickness = max(
            1,
            round(float(edge.get("width_pt") or 0.0) * max(scale_x, scale_y))
            + extra_thickness,
        )
        cv2.line(mask, start, end, 255, thickness=thickness)
    return mask


def build_nonstructural_vector_mask(page_data: dict) -> tuple[np.ndarray, dict]:
    """Rasterize a neutral-gray auxiliary layer when a black structure layer exists."""
    render_size = tuple(int(value) for value in page_data["render_size_px"])
    black_edges, gray_edges = _vector_layer_counts(page_data)
    enabled = bool(
        page_data.get("has_vector_geometry")
        and len(black_edges) >= 8
        and len(gray_edges) >= 3
    )
    evidence = {
        "enabled": enabled,
        "black_edge_count": len(black_edges),
        "gray_edge_count": len(gray_edges),
        "masked_pixels": 0,
    }
    if not enabled:
        return np.zeros((render_size[1], render_size[0]), dtype=np.uint8), evidence

    mask = _draw_edges(
        gray_edges,
        render_size,
        page_data["page_size_pt"],
        extra_thickness=2,
    )
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    evidence["masked_pixels"] = int(np.count_nonzero(mask))
    return mask, evidence


def build_structural_vector_mask(page_data: dict) -> tuple[np.ndarray, dict]:
    """Rasterize sufficiently dark neutral vector edges as structural evidence."""
    render_size = tuple(int(value) for value in page_data["render_size_px"])
    dark_edges = []
    for edge in page_data.get("styled_edges") or []:
        brightness = _neutral_brightness(edge)
        if brightness is not None and brightness <= 0.25:
            dark_edges.append(edge)

    enabled = bool(page_data.get("has_vector_geometry") and len(dark_edges) >= 8)
    evidence = {
        "enabled": enabled,
        "dark_edge_count": len(dark_edges),
        "masked_pixels": 0,
    }
    if not enabled:
        return np.zeros((render_size[1], render_size[0]), dtype=np.uint8), evidence

    mask = _draw_edges(
        dark_edges,
        render_size,
        page_data["page_size_pt"],
        extra_thickness=1,
    )
    evidence["masked_pixels"] = int(np.count_nonzero(mask))
    return mask, evidence


def detect_building_roi(page_data: dict) -> dict:
    """Estimate a conservative building-body ROI from dense black vector geometry."""
    black_edges, _ = _vector_layer_counts(page_data)
    disabled = {
        "enabled": False,
        "bbox_px": None,
        "confidence": 0.0,
        "reason": "insufficient_black_geometry",
    }
    if not page_data.get("has_vector_geometry") or len(black_edges) < 8:
        return disabled

    page_width, page_height = (float(value) for value in page_data["page_size_pt"])
    analysis_scale = 512.0 / max(page_width, page_height)
    analysis_width = max(1, round(page_width * analysis_scale))
    analysis_height = max(1, round(page_height * analysis_scale))
    ink = _draw_edges(
        black_edges,
        (analysis_width, analysis_height),
        [page_width, page_height],
        extra_thickness=0,
    )
    density = cv2.boxFilter(
        (ink > 0).astype(np.float32),
        ddepth=-1,
        ksize=(11, 11),
        normalize=True,
    )
    dense = None
    labels = None
    candidates = []
    min_area = analysis_width * analysis_height * 0.005
    for density_threshold in (0.12, 0.10, 0.08):
        dense = (density > density_threshold).astype(np.uint8) * 255
        dense = cv2.morphologyEx(
            dense,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), dtype=np.uint8),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
        candidates = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            aspect = max(width, height) / max(1, min(width, height))
            if area < min_area or aspect > 8.0:
                continue
            candidates.append((area, label))
        if candidates:
            break
    if not candidates:
        result = dict(disabled)
        result["reason"] = "no_dense_structural_cluster"
        return result

    component_area, component_label = max(candidates)
    ys, xs = np.where(labels == component_label)
    x0 = float(np.quantile(xs, 0.05))
    x1 = float(np.quantile(xs, 0.95))
    y0 = float(np.quantile(ys, 0.05))
    y1 = float(np.quantile(ys, 0.98))
    x_margin = analysis_width * 0.02
    y_margin = analysis_height * 0.02
    x0 = max(0.0, x0 - x_margin)
    x1 = min(float(analysis_width - 1), x1 + x_margin)
    y0 = max(0.0, y0 - y_margin)
    y1 = min(float(analysis_height - 1), y1 + y_margin)

    render_width, render_height = (int(value) for value in page_data["render_size_px"])
    bbox_px = [
        round(x0 * render_width / analysis_width),
        round(y0 * render_height / analysis_height),
        round(x1 * render_width / analysis_width),
        round(y1 * render_height / analysis_height),
    ]
    bbox_area = max(0, bbox_px[2] - bbox_px[0]) * max(0, bbox_px[3] - bbox_px[1])
    page_area = render_width * render_height
    area_fraction = bbox_area / max(1, page_area)
    if area_fraction < 0.10 or area_fraction > 0.95:
        result = dict(disabled)
        result.update({"reason": "implausible_roi_area", "area_fraction": area_fraction})
        return result

    dense_total = int(np.count_nonzero(dense))
    confidence = min(0.99, component_area / max(1, dense_total))
    return {
        "enabled": True,
        "bbox_px": bbox_px,
        "confidence": round(float(confidence), 4),
        "reason": "dense_black_vector_cluster",
        "area_fraction": area_fraction,
        "black_edge_count": len(black_edges),
    }


def remove_dimension_annotations(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill detected PDF annotations using the median page-border colour."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a BGR image")
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("annotation mask must match image dimensions")
    border_pixels = np.concatenate((
        image_bgr[0, :, :],
        image_bgr[-1, :, :],
        image_bgr[:, 0, :],
        image_bgr[:, -1, :],
    ), axis=0)
    background = np.median(border_pixels, axis=0).astype(np.uint8)
    cleaned = image_bgr.copy()
    cleaned[mask > 0] = background
    return cleaned


def _extract_text_spans(page) -> list[dict]:
    spans = []
    for word in page.extract_words(extra_attrs=["size", "upright"]):
        x0 = float(word["x0"])
        x1 = float(word["x1"])
        top = float(word["top"])
        bottom = float(word["bottom"])
        upright = bool(word.get("upright", True))
        spans.append({
            "text": str(word.get("text", "")).strip(),
            "bbox_pt": [x0, top, x1, bottom],
            "center_pt": [(x0 + x1) / 2.0, (top + bottom) / 2.0],
            "direction": "horizontal" if upright else "vertical",
            "font_size": float(word.get("size") or 0.0),
        })
    return [span for span in spans if span["text"]]


def _line_to_segment(line) -> dict | None:
    x0 = float(line["x0"])
    x1 = float(line["x1"])
    top = float(line["top"])
    bottom = float(line["bottom"])
    dx = abs(x1 - x0)
    dy = abs(bottom - top)
    tolerance = 0.5
    if dy <= tolerance and dx > tolerance:
        orientation = "horizontal"
        start, end = [min(x0, x1), top], [max(x0, x1), top]
        length = dx
    elif dx <= tolerance and dy > tolerance:
        orientation = "vertical"
        start, end = [x0, min(top, bottom)], [x0, max(top, bottom)]
        length = dy
    else:
        return None
    return {
        "start_pt": start,
        "end_pt": end,
        "orientation": orientation,
        "length_pt": float(length),
        "width_pt": float(line.get("linewidth") or 0.0),
    }


def _extract_axis_aligned_segments(page) -> list[dict]:
    segments = []
    seen = set()
    objects = list(page.lines)
    objects.extend(getattr(page, "rect_edges", []))
    objects.extend(getattr(page, "curve_edges", []))
    for line in objects:
        segment = _line_to_segment(line)
        if segment is None:
            continue
        key = (
            segment["orientation"],
            *(round(value, 3) for value in (*segment["start_pt"], *segment["end_pt"])),
        )
        if key not in seen:
            seen.add(key)
            segments.append(segment)
    return segments


def _stroke_rgb(value) -> list[float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        rgb = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(channel) for channel in rgb):
        return None
    return rgb


def _extract_styled_edges(page) -> list[dict]:
    """Keep coloured drawing edges while dropping unstyled glyph-outline noise."""
    edges = []
    seen = set()
    objects = list(page.lines)
    objects.extend(getattr(page, "rect_edges", []))
    objects.extend(getattr(page, "curve_edges", []))
    for item in objects:
        stroke_rgb = _stroke_rgb(item.get("stroking_color"))
        width_pt = float(item.get("linewidth") or 0.0)
        if stroke_rgb is None or width_pt <= 0:
            continue
        start = [float(item["x0"]), float(item["top"])]
        end = [float(item["x1"]), float(item["bottom"])]
        if start == end:
            continue
        key = (
            *(round(value, 3) for value in (*start, *end)),
            *(round(value, 4) for value in stroke_rgb),
            round(width_pt, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "start_pt": start,
            "end_pt": end,
            "stroke_rgb": stroke_rgb,
            "width_pt": width_pt,
        })
    return edges


def extract_vector_page(
    pdf_path: str | Path,
    page_index: int = 0,
    dpi: int = 200,
) -> dict:
    """Return JSON-compatible positioned text and axis-aligned vector segments."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with pdfplumber.open(str(pdf_path)) as document:
        if page_index < 0 or page_index >= len(document.pages):
            raise IndexError("page_index is outside the PDF page range")
        page = document.pages[page_index]
        text_spans = _extract_text_spans(page)
        segments = _extract_axis_aligned_segments(page)
        styled_edges = _extract_styled_edges(page)
        zoom = dpi / 72.0
        has_vector_text = bool(text_spans)
        has_vector_geometry = len(styled_edges) >= 4 or len(segments) >= 4
        return {
            "page_size_pt": [float(page.width), float(page.height)],
            "render_size_px": [round(float(page.width) * zoom), round(float(page.height) * zoom)],
            "dpi": int(dpi),
            "text_spans": text_spans,
            "segments": segments,
            "styled_edges": styled_edges,
            "vector_text_count": len(text_spans),
            "vector_segment_count": len(segments),
            "styled_edge_count": len(styled_edges),
            "has_vector_text": has_vector_text,
            "has_vector_geometry": has_vector_geometry,
            "is_vector_pdf": has_vector_geometry,
        }
