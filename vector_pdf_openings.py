"""Classify accepted exterior wall gaps as door, window, or ambiguous."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OpeningThresholds:
    line_mean_min: float = 0.30
    endpoint_max_min: float = 0.35
    endpoint_radius_px: int = 8
    ambiguity_margin: float = 0.08


def _axis_line(gap: dict) -> tuple[str, tuple[int, int], tuple[int, int]]:
    orientation = gap.get("orientation")
    raw_start = tuple(gap["start_px"])
    raw_end = tuple(gap["end_px"])
    if orientation == "horizontal" and raw_start[1] == raw_end[1]:
        start = tuple(int(round(value)) for value in raw_start)
        end = tuple(int(round(value)) for value in raw_end)
        return orientation, start, end
    if orientation == "vertical" and raw_start[0] == raw_end[0]:
        start = tuple(int(round(value)) for value in raw_start)
        end = tuple(int(round(value)) for value in raw_end)
        return orientation, start, end
    raise ValueError("gap orientation must match an axis-aligned segment")


def _line_mean(channel: np.ndarray, orientation: str, start: tuple[int, int], end: tuple[int, int]) -> float:
    height, width = channel.shape
    if orientation == "horizontal":
        y = start[1]
        if not 0 <= y < height:
            return 0.0
        left, right = sorted((start[0], end[0]))
        left, right = max(0, left), min(width - 1, right)
        samples = channel[y, left:right + 1] if left <= right else np.empty(0)
    else:
        x = start[0]
        if not 0 <= x < width:
            return 0.0
        top, bottom = sorted((start[1], end[1]))
        top, bottom = max(0, top), min(height - 1, bottom)
        samples = channel[top:bottom + 1, x] if top <= bottom else np.empty(0)
    return float(np.mean(samples)) if samples.size else 0.0


def _endpoint_max(channel: np.ndarray, point: tuple[int, int], radius: int) -> float:
    height, width = channel.shape
    x, y = point
    left, right = (
        min(width, max(0, x - radius)),
        min(width, max(0, x + radius + 1)),
    )
    top, bottom = (
        min(height, max(0, y - radius)),
        min(height, max(0, y + radius + 1)),
    )
    if left >= right or top >= bottom:
        return 0.0
    samples = channel[top:bottom, left:right]
    return float(np.max(samples)) if samples.size else 0.0


def _class_evidence(
    probabilities: np.ndarray,
    orientation: str,
    start: tuple[int, int],
    end: tuple[int, int],
    line_channel: int,
    endpoint_channel: int,
    thresholds: OpeningThresholds,
) -> dict:
    line_mean = _line_mean(probabilities[line_channel], orientation, start, end)
    start_max = _endpoint_max(probabilities[endpoint_channel], start, thresholds.endpoint_radius_px)
    end_max = _endpoint_max(probabilities[endpoint_channel], end, thresholds.endpoint_radius_px)
    confidence = float(np.clip(min(line_mean, start_max, end_max), 0.0, 1.0))
    return {
        "line_mean": line_mean,
        "start_endpoint_max": start_max,
        "end_endpoint_max": end_max,
        "confidence": confidence,
        "supported": (
            line_mean >= thresholds.line_mean_min
            and start_max >= thresholds.endpoint_max_min
            and end_max >= thresholds.endpoint_max_min
        ),
    }


def _reason_codes(gap: dict, reason: str) -> list[str]:
    return [*gap.get("reason_codes", []), reason]


def _opening(
    gap: dict,
    kind: str,
    width_px: float,
    confidence: float,
    evidence: dict,
    reason: str,
) -> dict:
    return {
        "opening_id": f"opening-{gap['gap_id']}",
        "kind": kind,
        "orientation": gap["orientation"],
        "start_px": copy.deepcopy(gap["start_px"]),
        "end_px": copy.deepcopy(gap["end_px"]),
        "width_px": width_px,
        "width_m": None,
        "host_wall_ids": copy.deepcopy(gap["host_wall_ids"]),
        "exterior": True,
        "confidence": confidence,
        "evidence": evidence,
        "reason_codes": _reason_codes(gap, reason),
    }


def _unclassified(gap: dict, reason: str) -> dict:
    record = copy.deepcopy(gap)
    record["reason_codes"] = _reason_codes(gap, reason)
    return record


def _gap_width_px(gap: dict) -> float:
    try:
        return float(gap["width_px"])
    except (KeyError, TypeError, ValueError):
        raw_start = gap["start_px"]
        raw_end = gap["end_px"]
        return float(abs(raw_end[0] - raw_start[0]) + abs(raw_end[1] - raw_start[1]))


def classify_exterior_openings(
    gaps: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    """Classify only gap-anchored exterior door/window candidates."""
    width, height = (int(value) for value in image_size)
    values = np.asarray(probabilities)
    if values.shape != (10, height, width):
        raise ValueError("probabilities must have shape (10, height, width)")

    thresholds = OpeningThresholds()
    result = {
        "accepted_openings": [],
        "ambiguous_openings": [],
        "unclassified_gaps": [],
    }
    for gap in gaps:
        if gap.get("decision") != "accepted_gap":
            continue
        try:
            orientation, start, end = _axis_line(gap)
        except (KeyError, TypeError, ValueError):
            result["unclassified_gaps"].append(_unclassified(gap, "invalid_gap_geometry"))
            continue

        door = _class_evidence(values, orientation, start, end, 5, 8, thresholds)
        window = _class_evidence(values, orientation, start, end, 6, 9, thresholds)
        evidence = {"door": door, "window": window}
        width_px = _gap_width_px(gap)
        supported = [
            ("door", door),
            ("window", window),
        ]
        supported = [candidate for candidate in supported if candidate[1]["supported"]]
        if not supported:
            result["unclassified_gaps"].append(_unclassified(gap, "no_door_or_window_support"))
            continue
        if len(supported) == 2 and abs(door["confidence"] - window["confidence"]) < thresholds.ambiguity_margin:
            result["ambiguous_openings"].append(_opening(
                gap,
                "ambiguous",
                width_px,
                max(door["confidence"], window["confidence"]),
                evidence,
                "door_window_score_within_ambiguity_margin",
            ))
            continue
        kind, selected = max(supported, key=lambda candidate: candidate[1]["confidence"])
        result["accepted_openings"].append(_opening(
            gap,
            kind,
            width_px,
            selected["confidence"],
            evidence,
            f"{kind}_line_and_endpoints_supported",
        ))
    return result
