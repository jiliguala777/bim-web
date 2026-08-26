"""Conservative exterior wall-chain recovery using native PDF door arcs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DoorRecoveryThresholds:
    boundary_band_px: int = 16
    endpoint_tolerance_px: int = 14
    chain_gap_tolerance_px: int = 12
    min_opening_span_px: int = 12
    max_opening_span_px: int = 140


def _axis_line(record: dict) -> tuple[str, tuple[int, int], tuple[int, int]]:
    orientation = str(record["orientation"])
    start = tuple(int(round(value)) for value in record["start_px"])
    end = tuple(int(round(value)) for value in record["end_px"])
    if orientation == "horizontal" and start[1] == end[1]:
        return orientation, min(start, end), max(start, end)
    if orientation == "vertical" and start[0] == end[0]:
        return orientation, min(start, end, key=lambda point: point[1]), max(
            start, end, key=lambda point: point[1],
        )
    raise ValueError("record must be an axis-aligned line")


def _fixed_interval(record: dict) -> tuple[str, int, int, int]:
    orientation, start, end = _axis_line(record)
    if orientation == "vertical":
        return orientation, start[0], start[1], end[1]
    return orientation, start[1], start[0], end[0]


def _native_uncertain_wall(candidate: dict) -> bool:
    native = candidate.get("native_evidence") or {}
    return (
        candidate.get("decision") == "uncertain"
        and bool(candidate.get("source_native_id"))
        and not native.get("page_border")
        and float(native.get("roi_inside_ratio") or 0.0) >= 0.95
    )


def _project_arc_to_wall_band(
    curve: dict,
    wall: dict,
    thresholds: DoorRecoveryThresholds,
) -> dict | None:
    """Return a projected opening node or None when the arc is not local to the wall."""
    if curve.get("has_bezier") is not True or curve.get("is_closed") is True:
        return None
    try:
        orientation, fixed, _, _ = _fixed_interval(wall)
        raw_start = curve.get("path_start_px", curve["start_px"])
        raw_end = curve.get("path_end_px", curve["end_px"])
        path_start = tuple(int(round(value)) for value in raw_start)
        path_end = tuple(int(round(value)) for value in raw_end)
        bbox = [int(round(value)) for value in curve["bbox_px"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(path_start) != 2 or len(path_end) != 2 or len(bbox) != 4:
        return None
    if max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])) > thresholds.max_opening_span_px:
        return None
    if orientation == "vertical":
        endpoint_distance = min(abs(path_start[0] - fixed), abs(path_end[0] - fixed))
        axis_start, axis_end = sorted((path_start[1], path_end[1]))
        projected_start, projected_end = [fixed, axis_start], [fixed, axis_end]
    else:
        endpoint_distance = min(abs(path_start[1] - fixed), abs(path_end[1] - fixed))
        axis_start, axis_end = sorted((path_start[0], path_end[0]))
        projected_start, projected_end = [axis_start, fixed], [axis_end, fixed]
    span = axis_end - axis_start
    if endpoint_distance > thresholds.endpoint_tolerance_px:
        return None
    if not thresholds.min_opening_span_px <= span <= thresholds.max_opening_span_px:
        return None
    projected = copy.deepcopy(curve)
    projected.update({
        "orientation": orientation,
        "fixed_px": fixed,
        "interval_start": axis_start,
        "interval_end": axis_end,
        "projected_start_px": projected_start,
        "projected_end_px": projected_end,
        "inside_direction": wall.get("inside_direction"),
        "wall_band_distance_px": endpoint_distance,
    })
    return projected


def _axis_components(
    nodes: list[dict],
    tolerance_px: int,
) -> list[list[dict]]:
    """Return deterministic connected components along one wall band."""
    ordered = sorted(nodes, key=lambda node: (
        int(node["interval_start"]),
        int(node["interval_end"]),
        str(node.get("node_type")),
        str(node.get("node_id")),
    ))
    components: list[list[dict]] = []
    component_end: int | None = None
    for node in ordered:
        start = int(node["interval_start"])
        end = int(node["interval_end"])
        if not components or component_end is None or start > component_end + tolerance_px:
            components.append([node])
            component_end = end
        else:
            components[-1].append(node)
            component_end = max(component_end, end)
    return components


def _wall_node(wall: dict, node_type: str, band: dict) -> dict | None:
    try:
        orientation, _, interval_start, interval_end = _fixed_interval(wall)
    except (KeyError, TypeError, ValueError):
        return None
    if orientation != band["orientation"]:
        return None
    return {
        "node_type": node_type,
        "node_id": str(wall.get("candidate_id")),
        "interval_start": interval_start,
        "interval_end": interval_end,
        "record": wall,
    }


def _build_bands(exterior_walls: list[dict], thresholds: DoorRecoveryThresholds) -> list[dict]:
    bands: list[dict] = []
    for wall in sorted(exterior_walls, key=lambda item: str(item.get("candidate_id"))):
        try:
            orientation, fixed, _, _ = _fixed_interval(wall)
        except (KeyError, TypeError, ValueError):
            continue
        inside_direction = wall.get("inside_direction")
        matches = [
            band for band in bands
            if band["orientation"] == orientation
            and band["inside_direction"] == inside_direction
            and abs(int(band["fixed_px"]) - fixed) <= thresholds.boundary_band_px
        ]
        if matches:
            band = min(matches, key=lambda item: abs(int(item["fixed_px"]) - fixed))
        else:
            band = {
                "band_id": f"band-{len(bands) + 1:04d}",
                "orientation": orientation,
                "fixed_px": fixed,
                "inside_direction": inside_direction,
                "anchors": [],
                "uncertain": [],
                "arcs": [],
            }
            bands.append(band)
        band["anchors"].append(wall)
    return bands


def _nearest_band_for_wall(candidate: dict, bands: list[dict], tolerance: int) -> dict | None:
    try:
        orientation, fixed, _, _ = _fixed_interval(candidate)
    except (KeyError, TypeError, ValueError):
        return None
    matches = [
        band for band in bands
        if band["orientation"] == orientation
        and abs(int(band["fixed_px"]) - fixed) <= tolerance
    ]
    if not matches:
        return None
    return min(matches, key=lambda band: (
        abs(int(band["fixed_px"]) - fixed), str(band["band_id"]),
    ))


def _normalized_recovered_wall(wall: dict, band: dict, arc_ids: list[str], anchor_ids: list[str]) -> dict:
    recovered = copy.deepcopy(wall)
    recovered["original_start_px"] = copy.deepcopy(wall["start_px"])
    recovered["original_end_px"] = copy.deepcopy(wall["end_px"])
    orientation, _, interval_start, interval_end = _fixed_interval(wall)
    fixed = int(band["fixed_px"])
    if orientation == "vertical":
        recovered["start_px"] = [fixed, interval_start]
        recovered["end_px"] = [fixed, interval_end]
    else:
        recovered["start_px"] = [interval_start, fixed]
        recovered["end_px"] = [interval_end, fixed]
    recovered.update({
        "inside_direction": band["inside_direction"],
        "recovery_method": "exterior_door_arc_chain",
        "recovery_arc_ids": arc_ids,
        "recovery_anchor_ids": anchor_ids,
        "reason_codes": list(dict.fromkeys([
            *(wall.get("reason_codes") or []),
            "native_wall_connected_through_exterior_door_arc",
        ])),
    })
    return recovered


def recover_exterior_walls_from_door_arcs(
    candidates: list[dict],
    exterior_walls: list[dict],
    curve_edges: list[dict],
    image_size: tuple[int, int],
    building_roi: list[int] | None,
) -> dict:
    """Return recovered walls and confirmed/pending exterior door arcs."""
    del building_roi
    width, height = (int(value) for value in image_size)
    thresholds = DoorRecoveryThresholds()
    bands = _build_bands(exterior_walls, thresholds)
    empty = {
        "recovered_walls": [],
        "confirmed_door_arcs": [],
        "pending_door_arcs": [],
        "recovery_components": [],
    }
    if not bands or width <= 0 or height <= 0:
        return empty

    for candidate in candidates:
        if not _native_uncertain_wall(candidate):
            continue
        band = _nearest_band_for_wall(candidate, bands, thresholds.boundary_band_px)
        if band is not None:
            band["uncertain"].append(candidate)

    for curve in curve_edges:
        projections = []
        for band in bands:
            representative = band["anchors"][0]
            projected = _project_arc_to_wall_band(curve, representative, thresholds)
            if projected is not None:
                projections.append((projected["wall_band_distance_px"], band, projected))
        if projections:
            _, band, projected = min(
                projections, key=lambda item: (item[0], str(item[1]["band_id"])),
            )
            band["arcs"].append(projected)

    recovered_by_id: dict[str, dict] = {}
    confirmed_by_id: dict[str, dict] = {}
    pending_by_id: dict[str, dict] = {}
    recovery_components = []
    for band in bands:
        nodes = []
        for wall in band["anchors"]:
            node = _wall_node(wall, "anchor", band)
            if node is not None:
                nodes.append(node)
        for wall in band["uncertain"]:
            node = _wall_node(wall, "uncertain_wall", band)
            if node is not None:
                nodes.append(node)
        for arc in band["arcs"]:
            nodes.append({
                "node_type": "door_arc",
                "node_id": str(arc.get("curve_id")),
                "interval_start": int(arc["interval_start"]),
                "interval_end": int(arc["interval_end"]),
                "record": arc,
            })
        for component in _axis_components(nodes, thresholds.chain_gap_tolerance_px):
            arcs = [node["record"] for node in component if node["node_type"] == "door_arc"]
            if not arcs:
                continue
            anchors = [node["record"] for node in component if node["node_type"] == "anchor"]
            uncertain = [
                node["record"] for node in component if node["node_type"] == "uncertain_wall"
            ]
            anchor_ids = sorted({str(wall.get("candidate_id")) for wall in anchors})
            arc_ids = sorted({str(arc.get("curve_id")) for arc in arcs})
            status = "confirmed" if len(anchor_ids) >= 2 else "pending" if anchor_ids else "ignored"
            recovery_components.append({
                "component_id": f"recovery-{len(recovery_components) + 1:04d}",
                "band_id": band["band_id"],
                "orientation": band["orientation"],
                "inside_direction": band["inside_direction"],
                "status": status,
                "anchor_ids": anchor_ids,
                "arc_ids": arc_ids,
                "candidate_wall_ids": sorted({
                    str(wall.get("candidate_id")) for wall in uncertain
                }),
            })
            if status == "confirmed":
                for wall in uncertain:
                    candidate_id = str(wall.get("candidate_id"))
                    recovered_by_id[candidate_id] = _normalized_recovered_wall(
                        wall, band, arc_ids, anchor_ids,
                    )
                for arc in arcs:
                    curve_id = str(arc.get("curve_id"))
                    confirmed = copy.deepcopy(arc)
                    confirmed.update({
                        "status": "confirmed",
                        "exterior_recovery_approved": True,
                        "host_wall_ids": anchor_ids,
                        "reason_codes": ["exterior_door_arc_connects_two_wall_anchors"],
                    })
                    confirmed_by_id[curve_id] = confirmed
            elif status == "pending":
                for arc in arcs:
                    curve_id = str(arc.get("curve_id"))
                    pending = copy.deepcopy(arc)
                    pending.update({
                        "status": "pending",
                        "exterior_recovery_approved": False,
                        "host_wall_ids": anchor_ids,
                        "reason_codes": ["exterior_door_arc_has_single_wall_anchor"],
                    })
                    pending_by_id[curve_id] = pending

    return {
        "recovered_walls": [recovered_by_id[key] for key in sorted(recovered_by_id)],
        "confirmed_door_arcs": [confirmed_by_id[key] for key in sorted(confirmed_by_id)],
        "pending_door_arcs": [pending_by_id[key] for key in sorted(pending_by_id)],
        "recovery_components": recovery_components,
    }
