"""Convert confirmed exterior-plan measurements into energy-model geometry."""

from __future__ import annotations

import copy
import math
from numbers import Real


def _require_positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _require_finite_result(value, name: str, *, allow_zero: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _multiply(name: str, *values: float, allow_zero: bool) -> float:
    try:
        product = 1.0
        for value in values:
            product *= value
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    return _require_finite_result(product, name, allow_zero=allow_zero)


def _add_nonnegative(left: float, right: float, name: str) -> float:
    try:
        result = left + right
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    return _require_finite_result(result, name, allow_zero=True)


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
            door_width_m = _add_nonnegative(
                door_width_m, width_m, f"door total {source_key}",
            )
        else:
            window_width_m = _add_nonnegative(
                window_width_m, width_m, f"window total {source_key}",
            )
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
    scaled_topology["area_m2"] = _multiply(
        "topology area_m2", area_px2, scale, scale, allow_zero=False,
    )
    scaled_topology["perimeter_m"] = _multiply(
        "topology perimeter_m", perimeter_px, scale, allow_zero=False,
    )
    for index, opening in enumerate(scaled_openings):
        width_px = _require_positive_number(
            opening.get("width_px"), f"opening {index} width_px",
        )
        opening["width_m"] = _multiply(
            f"opening {index} width_m", width_px, scale, allow_zero=False,
        )
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

    gross_wall_area_m2 = _multiply(
        "gross exterior wall area", perimeter_m, storey_height, floor_count,
        allow_zero=False,
    )
    door_area_m2 = _multiply(
        "door area", door_width_m, door_height, door_repeats, allow_zero=True,
    )
    window_area_m2 = _multiply(
        "window area", window_width_m, window_height, window_repeats,
        allow_zero=True,
    )
    try:
        wall_area_m2 = gross_wall_area_m2 - door_area_m2 - window_area_m2
    except OverflowError:
        raise ValueError("wall area must be finite") from None
    if not math.isfinite(wall_area_m2):
        raise ValueError("wall area must be finite")
    if wall_area_m2 < 0:
        raise ValueError("opening area exceeds gross exterior wall area")
    wall_area_m2 = _require_finite_result(
        wall_area_m2, "wall area", allow_zero=True,
    )
    total_floor_area_m2 = _multiply(
        "total floor area", footprint_area_m2, floor_count, allow_zero=False,
    )

    return {
        "per_floor_footprint_area_m2": footprint_area_m2,
        "total_floor_area_m2": total_floor_area_m2,
        "exterior_perimeter_m": perimeter_m,
        "gross_exterior_wall_area_m2": gross_wall_area_m2,
        "wall_area_m2": wall_area_m2,
        "door_area_m2": door_area_m2,
        "window_area_m2": window_area_m2,
        "door_total_width_m": door_width_m,
        "window_total_width_m": window_width_m,
    }
