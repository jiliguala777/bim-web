"""Convert confirmed exterior-plan measurements into energy-model geometry."""

from __future__ import annotations

import copy
import math
from numbers import Real


def _require_positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _require_positive_count(value, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} cannot exceed floors")
    return value


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    return value


def _scaled_opening_widths(openings: list[dict], source_key: str) -> tuple[float, float]:
    door_width_m = 0.0
    window_width_m = 0.0
    if not isinstance(openings, list):
        raise ValueError("openings must be a list")
    for index, opening in enumerate(openings):
        _require_mapping(opening, f"opening {index}")
        kind = opening.get("kind")
        if kind not in {"door", "window"}:
            raise ValueError(f"opening {index} kind must be door or window")
        width_m = _require_positive_number(opening.get(source_key), f"opening {index} {source_key}")
        if kind == "door":
            door_width_m += width_m
        else:
            window_width_m += width_m
    return door_width_m, window_width_m


def apply_scale_to_exterior(
    topology: dict, openings: list[dict], scale_m_per_px: float,
) -> tuple[dict, list[dict]]:
    """Deep-copy and add metric area, perimeter, and opening widths."""
    _require_mapping(topology, "topology")
    scale = _require_positive_number(scale_m_per_px, "scale_m_per_px")
    area_px2 = _require_positive_number(topology.get("area_px2"), "topology area_px2")
    perimeter_px = _require_positive_number(
        topology.get("perimeter_px"), "topology perimeter_px",
    )
    _scaled_opening_widths(openings, "width_px")

    scaled_topology = copy.deepcopy(topology)
    scaled_openings = copy.deepcopy(openings)
    scaled_topology["area_m2"] = area_px2 * scale ** 2
    scaled_topology["perimeter_m"] = perimeter_px * scale
    for opening in scaled_openings:
        opening["width_m"] = float(opening["width_px"]) * scale
    return scaled_topology, scaled_openings


def build_exterior_energy_geometry(
    topology: dict,
    openings: list[dict],
    *,
    storey_height_m: float,
    floors: int,
    door_height_m: float,
    window_height_m: float,
    door_repeat_count: int = 1,
    window_repeat_count: int | None = None,
) -> dict:
    """Return net opaque wall and separate opening areas for energy_calc.py."""
    _require_mapping(topology, "topology")
    footprint_area_m2 = _require_positive_number(
        topology.get("area_m2"), "topology area_m2",
    )
    perimeter_m = _require_positive_number(
        topology.get("perimeter_m"), "topology perimeter_m",
    )
    storey_height = _require_positive_number(storey_height_m, "storey_height_m")
    floor_count = _require_positive_count(floors, "floors")
    door_height = _require_positive_number(door_height_m, "door_height_m")
    window_height = _require_positive_number(window_height_m, "window_height_m")
    door_repeats = _require_positive_count(
        door_repeat_count, "door_repeat_count", maximum=floor_count,
    )
    if window_repeat_count is None:
        window_repeat_count = floor_count
    window_repeats = _require_positive_count(
        window_repeat_count, "window_repeat_count", maximum=floor_count,
    )
    door_width_m, window_width_m = _scaled_opening_widths(openings, "width_m")

    gross_wall_area_m2 = perimeter_m * storey_height * floor_count
    door_area_m2 = door_width_m * door_height * door_repeats
    window_area_m2 = window_width_m * window_height * window_repeats
    wall_area_m2 = gross_wall_area_m2 - door_area_m2 - window_area_m2
    if wall_area_m2 < 0:
        raise ValueError("opening area exceeds gross exterior wall area")

    return {
        "per_floor_footprint_area_m2": footprint_area_m2,
        "total_floor_area_m2": footprint_area_m2 * floor_count,
        "exterior_perimeter_m": perimeter_m,
        "gross_exterior_wall_area_m2": gross_wall_area_m2,
        "wall_area_m2": wall_area_m2,
        "door_area_m2": door_area_m2,
        "window_area_m2": window_area_m2,
        "door_total_width_m": door_width_m,
        "window_total_width_m": window_width_m,
    }
