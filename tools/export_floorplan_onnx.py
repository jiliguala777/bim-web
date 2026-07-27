"""Export the four-class floorplan U-Net checkpoint and verify ONNX parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODEL_INPUT_SHAPE = (1, 3, 512, 512)
MODEL_OUTPUT_SHAPE = (1, 4, 512, 512)
DEFAULT_MAX_ABS_ERROR = 1e-4
DEFAULT_MIN_ARGMAX_AGREEMENT = 0.99999
ONNX_EXPORT_OPTIONS = {"dynamo": False}


def check_onnx_in_subprocess(
    model_path: str | Path,
    *,
    python_executable: str | Path | None = None,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Run the native ONNX checker out of process so DLL crashes are reportable."""
    executable = str(python_executable or sys.executable)
    code = (
        "import onnx, sys; "
        "onnx.checker.check_model(sys.argv[1]); "
        "print('checker passed')"
    )
    result = runner(
        [executable, "-c", code, str(Path(model_path).resolve())],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode == 0:
        return {"status": "passed", "returncode": 0}
    return {
        "status": "failed",
        "returncode": int(result.returncode),
        "stderr": (result.stderr or result.stdout or "checker failed without output")[-4000:],
    }


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the named state dictionary or reject an incompatible checkpoint."""
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain a 'model_state_dict' entry")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint 'model_state_dict' must be a mapping")
    return state_dict


def compare_outputs(torch_output: np.ndarray, onnx_output: np.ndarray) -> dict[str, Any]:
    """Calculate numerical and semantic agreement for two NCHW logits arrays."""
    torch_array = np.asarray(torch_output, dtype=np.float32)
    onnx_array = np.asarray(onnx_output, dtype=np.float32)
    if torch_array.shape != onnx_array.shape:
        raise ValueError(
            f"output shape mismatch: PyTorch {torch_array.shape}, ONNX {onnx_array.shape}"
        )
    if torch_array.ndim != 4:
        raise ValueError(f"expected four-dimensional NCHW output, got {torch_array.shape}")

    absolute_error = np.abs(torch_array - onnx_array)
    torch_classes = torch_array.argmax(axis=1)
    onnx_classes = onnx_array.argmax(axis=1)
    return {
        "output_shape": list(torch_array.shape),
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "mean_abs_error": float(absolute_error.mean()),
        "argmax_pixel_agreement": float(np.mean(torch_classes == onnx_classes)),
    }


def require_equivalent(
    metrics: Mapping[str, Any],
    *,
    max_abs_error: float = DEFAULT_MAX_ABS_ERROR,
    min_argmax_agreement: float = DEFAULT_MIN_ARGMAX_AGREEMENT,
) -> None:
    """Raise when runtime comparison metrics do not meet the acceptance limits."""
    actual_error = float(metrics["max_abs_error"])
    if actual_error > max_abs_error:
        raise RuntimeError(
            f"maximum absolute error {actual_error:.8g} exceeds {max_abs_error:.8g}"
        )
    actual_agreement = float(metrics["argmax_pixel_agreement"])
    if actual_agreement < min_argmax_agreement:
        raise RuntimeError(
            f"argmax pixel agreement {actual_agreement:.8g} is below "
            f"{min_argmax_agreement:.8g}"
        )


def build_model():
    """Construct the exact architecture used by the annotation tool."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch is required to build the floorplan model"
        ) from exc
    return smp.Unet(
        "resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )


def load_checkpoint_model(checkpoint_path: str | Path):
    """Load the checkpoint into the exact CPU model using strict key matching."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load the checkpoint") from exc

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    model = build_model()
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 17,
    max_abs_error: float = DEFAULT_MAX_ABS_ERROR,
    min_argmax_agreement: float = DEFAULT_MIN_ARGMAX_AGREEMENT,
    allow_checker_failure: bool = False,
) -> dict[str, Any]:
    """Export, check, execute, and compare the real floorplan model."""
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite ONNX output: {destination}")
    if Path(checkpoint_path).resolve() == destination:
        raise ValueError("checkpoint and ONNX output paths must differ")
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "export requires torch and onnxruntime in the active environment"
        ) from exc

    model = load_checkpoint_model(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(20260714)
    dummy_input = torch.randn(MODEL_INPUT_SHAPE, generator=generator, dtype=torch.float32)
    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()
    if tuple(torch_output.shape) != MODEL_OUTPUT_SHAPE:
        raise RuntimeError(
            f"unexpected PyTorch output shape {torch_output.shape}; expected {MODEL_OUTPUT_SHAPE}"
        )

    torch.onnx.export(
        model,
        dummy_input,
        destination,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        **ONNX_EXPORT_OPTIONS,
    )

    checker = check_onnx_in_subprocess(destination)
    if checker["status"] != "passed" and not allow_checker_failure:
        raise RuntimeError(
            "ONNX checker failed in its isolated subprocess with return code "
            f"{checker['returncode']}"
        )
    session = ort.InferenceSession(
        str(destination), providers=["CPUExecutionProvider"]
    )
    model_inputs = session.get_inputs()
    model_outputs = session.get_outputs()
    if len(model_inputs) != 1 or model_inputs[0].name != "input":
        raise RuntimeError("exported ONNX graph must expose one input named 'input'")
    if len(model_outputs) != 1 or model_outputs[0].name != "logits":
        raise RuntimeError("exported ONNX graph must expose one output named 'logits'")

    onnx_output = session.run(["logits"], {"input": dummy_input.numpy()})[0]
    metrics = compare_outputs(torch_output, onnx_output)
    require_equivalent(
        metrics,
        max_abs_error=max_abs_error,
        min_argmax_agreement=min_argmax_agreement,
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "onnx": str(destination.resolve()),
        "opset": opset,
        "checker": checker,
        "input": {"name": model_inputs[0].name, "shape": model_inputs[0].shape},
        "output": {"name": model_outputs[0].name, "shape": model_outputs[0].shape},
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--allow-checker-failure",
        action="store_true",
        help="continue parity validation while recording a native checker failure",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = export_onnx(
        args.checkpoint,
        args.output,
        opset=args.opset,
        allow_checker_failure=args.allow_checker_failure,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
