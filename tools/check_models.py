"""Check ONNX input/output contracts, checksum, and CPU zero-input inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def inspect_model(path, backend):
    import numpy as np
    import onnxruntime as ort
    path = Path(path)
    if not path.is_file():
        raise ValueError("Model file not found; see docs/models.md")
    size = {"legacy_onnx": 512, "image_onnx": 256}[backend]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    expected = [1, 3, size, size]
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
        raise ValueError("Expected one float32 NCHW input")
    if any(isinstance(a, int) and a != b for a, b in zip(inputs[0].shape, expected)):
        raise ValueError("Unexpected input shape")
    outputs = session.run(None, {inputs[0].name: np.zeros(expected, dtype=np.float32)})
    channels = [4] if backend == "legacy_onnx" else [4, 1]
    if len(outputs) != len(channels) or any(
        out.shape != (1, c, size, size) or not np.isfinite(out).all()
        for out, c in zip(outputs, channels)
    ):
        raise ValueError("Unexpected output heads, shapes or non-finite logits")
    return {"backend": backend, "sha256": digest.hexdigest(), "input_shape": expected,
            "output_shapes": [list(out.shape) for out in outputs], "zero_input_inference": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("legacy_onnx", "image_onnx"), required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_model(args.path, args.backend), indent=2))
    except Exception as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
