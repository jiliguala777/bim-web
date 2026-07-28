"""Deterministic fine-tuning loop for the four-class floorplan model."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Callable

import numpy as np
import torch

from tools.export_floorplan_onnx import build_model, extract_state_dict

from .dataset import ExportedFloorplanDataset, validate_export
from .losses import CombinedSegmentationLoss


DEFAULT_SEED = 20260727


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    checkpoint_path: Path
    output_dir: Path
    seed: int = DEFAULT_SEED
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    device: str = "cpu"
    augmentation_count: int = 4
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        for field_name in ("dataset_path", "checkpoint_path", "output_dir"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.augmentation_count < 1:
            raise ValueError("augmentation_count must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")


@dataclass(frozen=True)
class TrainingResult:
    last_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    log_path: Path
    epochs_completed: int
    best_loss: float


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "training_loss", "smoothed_training_loss", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_floorplan_model():
    """Build exactly the same U-Net architecture used by the ONNX exporter."""
    return build_model()


def load_model_strict(
    checkpoint_path: str | Path,
    model_factory: Callable[[], torch.nn.Module] = build_floorplan_model,
    *,
    device: str | torch.device,
) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    model = model_factory()
    try:
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"strict state_dict loading failed: {exc}") from exc
    return model.to(device)


def _class_weights(dataset_path: Path) -> list[float]:
    report_path = dataset_path / "dataset_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pixels = report.get("class_pixels")
        if isinstance(pixels, dict):
            return CombinedSegmentationLoss.weights_from_class_pixels(pixels)
    validated = validate_export(dataset_path)
    pixels = {str(class_id): 0 for class_id in range(4)}
    for sample in validated["samples"]:
        mask = np.load(sample["_mask_path"], allow_pickle=False)
        for class_id in range(4):
            pixels[str(class_id)] += int(np.sum(mask == class_id))
    return CombinedSegmentationLoss.weights_from_class_pixels(pixels)


def _serialized_config(config: TrainingConfig) -> dict:
    payload = asdict(config)
    for key in ("dataset_path", "checkpoint_path", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def _validate_experiment(validated: dict) -> str:
    experiment_type = validated["manifest"].get("experiment_type")
    sample_count = int(validated["sample_count"])
    if experiment_type == "single_page_overfit":
        if sample_count != 1:
            raise ValueError("single_page_overfit requires exactly one sample")
        return experiment_type
    if experiment_type == "fine_tune":
        if sample_count < 2:
            raise ValueError("fine_tune requires at least two samples")
        return experiment_type
    raise ValueError(f"unsupported experiment_type: {experiment_type}")


def train(
    config: TrainingConfig,
    *,
    model_factory: Callable[[], torch.nn.Module] = build_floorplan_model,
) -> TrainingResult:
    _set_deterministic(config.seed)
    validated = validate_export(config.dataset_path)
    experiment_type = _validate_experiment(validated)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty for a new experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_strict(
        config.checkpoint_path,
        model_factory,
        device=device,
    )
    model.train()
    weights = _class_weights(config.dataset_path)
    criterion = CombinedSegmentationLoss(weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    log_rows = []
    best_loss = float("inf")
    smoothed_loss = None
    epochs_without_improvement = 0
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    log_path = output_dir / "training_log.csv"
    metrics_path = output_dir / "metrics.json"

    for epoch in range(1, config.epochs + 1):
        losses = []
        for repetition in range(config.augmentation_count):
            dataset = ExportedFloorplanDataset(
                config.dataset_path,
                augment=True,
                seed=config.seed + epoch * 1000 + repetition,
            )
            for sample in dataset:
                inputs = sample["image"].unsqueeze(0).to(device)
                targets = sample["mask"].unsqueeze(0).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("training loss became non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.gradient_clip_norm,
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
        training_loss = float(np.mean(losses))
        smoothed_loss = (
            training_loss
            if smoothed_loss is None
            else 0.8 * smoothed_loss + 0.2 * training_loss
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "training_loss": training_loss,
            "smoothed_training_loss": smoothed_loss,
            "config": _serialized_config(config),
            "class_weights": weights,
        }
        _atomic_torch_save(last_path, checkpoint)
        improved = smoothed_loss < best_loss - 1e-8
        if improved:
            best_loss = smoothed_loss
            epochs_without_improvement = 0
            _atomic_torch_save(best_path, checkpoint)
        else:
            epochs_without_improvement += 1
        log_rows.append(
            {
                "epoch": epoch,
                "training_loss": f"{training_loss:.10f}",
                "smoothed_training_loss": f"{smoothed_loss:.10f}",
                "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
            }
        )
        _write_log(log_path, log_rows)
        metrics = {
            "experiment_type": experiment_type,
            "config": _serialized_config(config),
            "class_weights": weights,
            "epochs_completed": epoch,
            "best_smoothed_training_loss": best_loss,
            "selection_basis": "smoothed_training_loss",
            "has_independent_validation": False,
            "generalization_claim": False,
            "stopped_early": epochs_without_improvement >= config.patience,
        }
        _atomic_json(metrics_path, metrics)
        if epochs_without_improvement >= config.patience:
            break

    return TrainingResult(
        last_checkpoint=last_path,
        best_checkpoint=best_path,
        metrics_path=metrics_path,
        log_path=log_path,
        epochs_completed=len(log_rows),
        best_loss=best_loss,
    )
