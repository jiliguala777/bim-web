"""Bridge vector-floorplan inference artifacts into the energy platform contract.

The platform deliberately runs the approved vector predictor in its separate
training environment.  This keeps the Flask runtime free of PyTorch while the
legacy ONNX recogniser remains available unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

import cv2
import numpy as np


VECTOR_BACKEND = "vector_pytorch"


@dataclass(frozen=True)
class VectorPlatformConfig:
    training_root: Path
    python_executable: Path
    checkpoint_path: Path
    device: str = "cpu"
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_root", Path(self.training_root).resolve())
        object.__setattr__(self, "python_executable", Path(self.python_executable).resolve())
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path).resolve())
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("vector device must be exactly cpu or cuda")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("vector timeout must be finite and positive")

    @classmethod
    def from_environment(cls) -> "VectorPlatformConfig | None":
        values = {
            "training_root": os.environ.get("VECTOR_TRAINING_ROOT", "").strip(),
            "python_executable": os.environ.get("VECTOR_PYTHON", "").strip(),
            "checkpoint_path": os.environ.get("VECTOR_MODEL_PATH", "").strip(),
        }
        if not all(values.values()):
            return None
        return cls(
            **values,
            device=os.environ.get("VECTOR_DEVICE", "cpu").strip(),
            timeout_seconds=float(os.environ.get("VECTOR_TIMEOUT_SECONDS", "600")),
        )

    def require_available(self) -> None:
        if not self.training_root.is_dir():
            raise RuntimeError("VECTOR_TRAINING_ROOT is not a directory")
        if not self.python_executable.is_file():
            raise RuntimeError("VECTOR_PYTHON is not an executable file")
        if not self.checkpoint_path.is_file():
            raise RuntimeError("VECTOR_MODEL_PATH is not an existing model file")


class VectorPlatformAdapter:
    """Run one vector prediction externally and return platform-shaped geometry."""

    def __init__(self, config: VectorPlatformConfig) -> None:
        self.config = config

    def predict(self, image_path: Path, artifact_parent: Path) -> dict:
        self.config.require_available()
        source = Path(image_path).resolve(strict=True)
        parent = Path(artifact_parent).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        output = parent / f"vector-prediction-{uuid.uuid4().hex}"
        command = [
            str(self.config.python_executable), "-m", "training.predict_vector",
            "--checkpoint", str(self.config.checkpoint_path),
            "--input", str(source),
            "--output", str(output),
            "--device", self.config.device,
        ]
        completed = subprocess.run(
            command, cwd=self.config.training_root, capture_output=True, text=True,
            timeout=self.config.timeout_seconds, check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "vector inference failed").strip()
            raise RuntimeError(f"vector inference failed: {detail}")
        return self._read_prediction(source, output)

    def _read_prediction(self, image_path: Path, output: Path) -> dict:
        geometry_path = output / "prediction.geometry.json"
        measurements_path = output / "measurements.json"
        inference_path = output / "inference.json"
        overlay_path = output / "overlay.png"
        if not all(path.is_file() for path in (geometry_path, measurements_path, inference_path, overlay_path)):
            raise RuntimeError("vector inference did not publish its required artifacts")
        geometry = _read_json(geometry_path)
        measurements = _read_json(measurements_path)
        inference = _read_json(inference_path)
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if source is None or overlay is None:
            raise RuntimeError("vector inference image artifact cannot be decoded")
        return adapt_vector_prediction(geometry, measurements, inference, source, overlay, output)


def adapt_vector_prediction(
    geometry: dict,
    measurements: dict,
    inference: dict,
    source_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    artifact_dir: Path,
) -> dict:
    """Convert canonical vector geometry to the existing recognition.json shape."""
    target = geometry.get("target") if isinstance(geometry, dict) else None
    if not isinstance(target, dict):
        raise ValueError("vector prediction geometry has no target")
    width, height = target.get("width"), target.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("vector prediction target dimensions are invalid")
    if source_bgr.shape[:2] != (height, width) or overlay_bgr.shape[:2] != (height, width):
        raise ValueError("vector prediction artifacts do not match source dimensions")

    walls = [_legacy_line(item, "wall") for item in _items(geometry, "walls")]
    openings = _items(geometry, "openings")
    windows = [_legacy_line(item, "window") for item in openings if item.get("opening_type") == "window"]
    doors = [_legacy_line(item, "door") for item in openings if item.get("opening_type") == "door"]
    rooms = [_legacy_room(item, index + 1) for index, item in enumerate(_items(geometry, "rooms"))]
    room_area_px2 = float(sum(room["area_px2"] for room in rooms))
    scale = measurements.get("scale_m_per_px") if isinstance(measurements, dict) else None
    valid_scale = float(scale) if isinstance(scale, (int, float)) and math.isfinite(float(scale)) and float(scale) > 0 else None
    room_topology = {
        "status": "closed" if rooms else "no_closed_rooms",
        "room_count": len(rooms),
        "closure_applied": False,
        "rooms": rooms,
        "total_area_px2": room_area_px2,
        "total_area_m2": room_area_px2 * valid_scale ** 2 if valid_scale is not None else None,
        "scale_m_per_px": valid_scale,
        "load_geometry_ready": bool(rooms and valid_scale is not None),
    }
    for room in rooms:
        room["area_m2"] = room["area_px2"] * valid_scale ** 2 if valid_scale is not None else None
    mask = _preview_mask(width, height, walls, windows, doors)
    scale_calibration = {
        "status": "confirmed" if valid_scale is not None else "manual_required",
        "method": "vector_dimensions" if valid_scale is not None else "raster_manual",
        "scale_m_per_px": valid_scale,
        "confidence": 1.0 if valid_scale is not None else 0.0,
        "horizontal": None,
        "vertical": None,
        "axis_difference_percent": None,
        "evidence": measurements.get("candidates", []) if isinstance(measurements, dict) else [],
    }
    return {
        "backend": VECTOR_BACKEND,
        "model": {
            "backend": VECTOR_BACKEND,
            "name": "vector-resnet34-unet",
            "version": str(inference.get("checkpoint_sha256") or "unknown"),
            "path": str(artifact_dir),
            "classes": ["footprint", "room", "wall", "door", "window"],
        },
        "image_size": [width, height],
        "mask": mask,
        "overlay": overlay_bgr,
        "source": source_bgr,
        "stats": {
            "footprints": len(_items(geometry, "footprints")), "rooms": len(rooms),
            "walls": len(walls), "windows": len(windows), "doors": len(doors),
        },
        "geometry": {"walls": walls, "windows": windows, "doors": doors},
        "room_topology": room_topology,
        "scale_calibration": scale_calibration,
        "vector_geometry": geometry,
        "vector_measurements": measurements,
        "vector_inference": inference,
        "artifact_dir": str(artifact_dir),
    }


def _items(payload: dict, name: str) -> list[dict]:
    value = payload.get(name, []) if isinstance(payload, dict) else []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"vector prediction {name} must be a list of objects")
    return value


def _points(item: dict, *, minimum: int) -> list[list[float]]:
    points = item.get("points")
    if not isinstance(points, list) or len(points) < minimum:
        raise ValueError("vector prediction contains an invalid point sequence")
    result = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("vector prediction contains an invalid point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("vector prediction point must be finite")
        result.append([x, y])
    return result


def _legacy_line(item: dict, kind: str) -> dict:
    points = _points(item, minimum=2)
    length = sum(math.dist(first, second) for first, second in zip(points, points[1:]))
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return {
        "id": str(item.get("id") or kind), "pts": points, "area": 0.0,
        "length_px": length, "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
    }


def _legacy_room(item: dict, index: int) -> dict:
    points = _points(item, minimum=3)
    area = abs(sum(
        point[0] * points[(offset + 1) % len(points)][1]
        - points[(offset + 1) % len(points)][0] * point[1]
        for offset, point in enumerate(points)
    ) / 2.0)
    return {"id": str(item.get("id") or f"room-{index}"), "polygon_px": points, "area_px2": area, "area_m2": None}


def _preview_mask(width: int, height: int, walls: list[dict], windows: list[dict], doors: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for class_id, records in ((1, walls), (2, windows), (3, doors)):
        for record in records:
            points = np.rint(np.asarray(record["pts"], dtype=np.float64)).astype(np.int32)
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, class_id, thickness=2, lineType=cv2.LINE_8)
    return mask


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vector inference artifact is invalid: {path.name}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"vector inference artifact root must be an object: {path.name}")
    return result
