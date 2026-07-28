"""Train, compare, export, and verify a single-page overfit experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tools.export_floorplan_onnx import (
    DEFAULT_MIN_ARGMAX_AGREEMENT,
    compare_outputs,
    export_onnx,
)

from .dataset import ExportedFloorplanDataset
from .evaluate import compare_checkpoints
from .train_floorplan import (
    TrainingConfig,
    TrainingResult,
    build_floorplan_model,
    load_model_strict,
    train,
)


@dataclass(frozen=True)
class ExperimentResult:
    training: TrainingResult
    onnx_path: Path
    export_report_path: Path
    comparison_report_path: Path
    sample_parity: dict


def require_class_mask_equivalent(
    metrics: dict,
    *,
    min_argmax_agreement: float = DEFAULT_MIN_ARGMAX_AGREEMENT,
) -> None:
    agreement = float(metrics["argmax_pixel_agreement"])
    if agreement < min_argmax_agreement:
        raise RuntimeError(
            f"class-mask agreement {agreement:.8g} is below "
            f"{min_argmax_agreement:.8g}"
        )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_sample_parity(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    dataset_path: str | Path,
    *,
    model_factory: Callable[[], torch.nn.Module] = build_floorplan_model,
    device: str = "cpu",
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for sample parity verification") from exc
    dataset = ExportedFloorplanDataset(dataset_path, augment=False, seed=0)
    sample = dataset[0]
    inputs = sample["image"].unsqueeze(0)
    model = load_model_strict(
        checkpoint_path,
        model_factory,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        torch_logits = model(inputs.to(device)).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_logits = session.run([output_name], {input_name: inputs.numpy()})[0]
    metrics = compare_outputs(torch_logits, np.asarray(onnx_logits))
    require_class_mask_equivalent(metrics)
    return {
        "status": "passed",
        "sample_id": sample["sample_id"],
        **metrics,
    }


def run_experiment(
    config: TrainingConfig,
    *,
    model_factory: Callable[[], torch.nn.Module] = build_floorplan_model,
    export_function=export_onnx,
    sample_parity_function=verify_sample_parity,
) -> ExperimentResult:
    training = train(config, model_factory=model_factory)
    training_metrics = json.loads(training.metrics_path.read_text(encoding="utf-8"))
    experiment_type = training_metrics["experiment_type"]
    comparison = compare_checkpoints(
        config.checkpoint_path,
        training.best_checkpoint,
        config.dataset_path,
        config.output_dir,
        model_factory=model_factory,
        device=config.device,
    )
    comparison_path = config.output_dir / "comparison_metrics.json"
    onnx_path = config.output_dir / "best.onnx"
    export_report = export_function(training.best_checkpoint, onnx_path)
    sample_parity = sample_parity_function(
        training.best_checkpoint,
        onnx_path,
        config.dataset_path,
        model_factory=model_factory,
        device=config.device,
    )
    report = {
        "experiment_type": experiment_type,
        "checker_and_random_input_parity": export_report,
        "confirmed_sample_parity": sample_parity,
        "comparison_report": str(comparison_path),
        "has_independent_validation": False,
        "generalization_claim": False,
    }
    export_report_path = config.output_dir / "export_report.json"
    _atomic_json(export_report_path, report)
    return ExperimentResult(
        training=training,
        onnx_path=onnx_path,
        export_report_path=export_report_path,
        comparison_report_path=comparison_path,
        sample_parity=sample_parity,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentation-count", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = TrainingConfig(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        augmentation_count=args.augmentation_count,
    )
    result = run_experiment(config)
    print(
        json.dumps(
            {
                "best_checkpoint": str(result.training.best_checkpoint),
                "onnx": str(result.onnx_path),
                "export_report": str(result.export_report_path),
                "comparison_report": str(result.comparison_report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
