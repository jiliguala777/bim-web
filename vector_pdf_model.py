"""Strict probability evidence bridge for the new vector-PDF path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np


CHANNEL_NAMES = (
    "footprint_interior",
    "footprint_boundary",
    "room_interior",
    "room_boundary",
    "wall_centerline",
    "door_line",
    "window_line",
    "wall_keypoint",
    "door_endpoint",
    "window_endpoint",
)


class VectorModelError(RuntimeError):
    """Base error for the external vector probability provider."""


class VectorModelUnavailableError(VectorModelError):
    """Raised when configured inference files are unavailable."""


class VectorModelTimeoutError(VectorModelError):
    """Raised when external CPU inference exceeds its configured limit."""


class VectorModelContractError(VectorModelError):
    """Raised when external artifacts violate the probability contract."""


@dataclass(frozen=True)
class VectorProbabilityResult:
    probabilities: np.ndarray
    inference: dict
    artifact_dir: Path
    probabilities_sha256: str


@dataclass(frozen=True)
class VectorModelConfig:
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
            raise ValueError("vector model device must be cpu or cuda")
        if not math.isfinite(float(self.timeout_seconds)) or float(self.timeout_seconds) <= 0:
            raise ValueError("vector model timeout must be finite and positive")

    @classmethod
    def from_environment(cls) -> "VectorModelConfig | None":
        training_root = os.environ.get("VECTOR_TRAINING_ROOT", "").strip()
        python_executable = os.environ.get("VECTOR_PYTHON", "").strip()
        checkpoint_path = os.environ.get("VECTOR_MODEL_PATH", "").strip()
        if not all((training_root, python_executable, checkpoint_path)):
            return None
        return cls(
            training_root=Path(training_root),
            python_executable=Path(python_executable),
            checkpoint_path=Path(checkpoint_path),
            device=os.environ.get("VECTOR_DEVICE", "cpu").strip(),
            timeout_seconds=float(os.environ.get("VECTOR_TIMEOUT_SECONDS", "600")),
        )

    def require_available(self) -> None:
        if not self.training_root.is_dir():
            raise VectorModelUnavailableError("VECTOR_TRAINING_ROOT is not a directory")
        if not self.python_executable.is_file():
            raise VectorModelUnavailableError("VECTOR_PYTHON is not an existing file")
        if not self.checkpoint_path.is_file():
            raise VectorModelUnavailableError("VECTOR_MODEL_PATH is not an existing file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_probability_artifacts(
    output_dir: str | Path,
    expected_size: tuple[int, int],
) -> VectorProbabilityResult:
    """Load source-aligned ten-channel probabilities under a strict contract."""
    output = Path(output_dir).resolve()
    probability_path = output / "probabilities.npz"
    inference_path = output / "inference.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    width, height = (int(value) for value in expected_size)
    if inference.get("channel_names") != list(CHANNEL_NAMES):
        raise VectorModelContractError("vector probability channel contract is incompatible")
    if inference.get("image_size") != [width, height]:
        raise VectorModelContractError("inference image size does not match the analysis image")
    with np.load(probability_path, allow_pickle=False) as archive:
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32).copy()
    if probabilities.shape != (len(CHANNEL_NAMES), height, width):
        raise VectorModelContractError("probability array shape does not match the analysis image")
    if not np.isfinite(probabilities).all():
        raise VectorModelContractError("probability array contains non-finite values")
    if float(probabilities.min(initial=0.0)) < 0.0 or float(probabilities.max(initial=0.0)) > 1.0:
        raise VectorModelContractError("probability values must stay between zero and one")
    probabilities.setflags(write=False)
    return VectorProbabilityResult(
        probabilities=probabilities,
        inference=inference,
        artifact_dir=output,
        probabilities_sha256=_sha256(probability_path),
    )


def run_vector_probabilities(
    image_path: str | Path,
    artifact_parent: str | Path,
    config: VectorModelConfig,
    *,
    expected_size: tuple[int, int],
) -> VectorProbabilityResult:
    """Run the approved external predictor and load only probability evidence."""
    config.require_available()
    source = Path(image_path).resolve(strict=True)
    parent = Path(artifact_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / f"vector-probabilities-{uuid.uuid4().hex}"
    command = [
        str(config.python_executable),
        "-m",
        "training.predict_vector",
        "--checkpoint",
        str(config.checkpoint_path),
        "--input",
        str(source),
        "--output",
        str(output),
        "--device",
        config.device,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=config.training_root,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VectorModelTimeoutError(
            f"vector probability inference exceeded {config.timeout_seconds:g} seconds"
        ) from exc
    except OSError as exc:
        raise VectorModelUnavailableError("vector probability process could not start") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "vector inference failed").strip()
        raise VectorModelError(f"vector probability inference failed: {detail}")
    return load_probability_artifacts(output, expected_size)
