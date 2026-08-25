"""Opt-in CPU smoke check for the external vector probability model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Callable, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vector_pdf_model import (  # noqa: E402
    CHANNEL_NAMES,
    VectorModelConfig,
    run_vector_probabilities,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real vector model once on a temporary orthogonal room image.",
    )
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--python", required=True, dest="python_executable", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable = run_vector_probabilities,
) -> int:
    args = _parser().parse_args(argv)
    config = VectorModelConfig(
        training_root=args.training_root,
        python_executable=args.python_executable,
        checkpoint_path=args.checkpoint,
        device=args.device,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        config.require_available()
        with tempfile.TemporaryDirectory(prefix="vector-pdf-smoke-") as temp_dir:
            owned_dir = Path(temp_dir)
            image_path = owned_dir / "orthogonal-room.png"
            image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (192, 192), (832, 832), (0, 0, 0), thickness=8)
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError("could not create the temporary smoke image")
            result = runner(
                image_path,
                owned_dir,
                config,
                expected_size=(1024, 1024),
            )
            summary = {
                "shape": list(result.probabilities.shape),
                "channel_names": list(CHANNEL_NAMES),
                "probabilities_sha256": result.probabilities_sha256,
                "runtime_metadata": result.inference,
                "device": config.device,
            }
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        print(f"vector model smoke failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
