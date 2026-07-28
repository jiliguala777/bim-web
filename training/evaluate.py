"""Checkpoint evaluation and visual comparison against a confirmed mask."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ExportedFloorplanDataset,
    validate_export,
)
from .losses import dice_per_class
from .train_floorplan import build_floorplan_model, load_model_strict


CLASS_NAMES = ("background", "wall", "window", "door")
CLASS_COLOURS_RGB = np.asarray(
    [(40, 40, 40), (231, 76, 60), (52, 152, 219), (46, 204, 113)],
    dtype=np.uint8,
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_image(path: Path, image: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(path.suffix, image)
    if not encoded:
        raise ValueError(f"image could not be encoded: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(buffer.tobytes())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_image(path: Path) -> np.ndarray | None:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _safe_sample_id(sample_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(sample_id)).strip("-")
    if not safe:
        raise ValueError("sample_id must contain a safe filename character")
    return safe


def _confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    encoded = target.reshape(-1) * 4 + prediction.reshape(-1)
    return np.bincount(encoded, minlength=16).reshape(4, 4)


def _metrics(confusion: np.ndarray) -> dict:
    classes = {}
    for class_id, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[class_id, class_id])
        target_pixels = int(confusion[class_id, :].sum())
        predicted_pixels = int(confusion[:, class_id].sum())
        union = target_pixels + predicted_pixels - true_positive
        present = target_pixels > 0
        classes[str(class_id)] = {
            "name": name,
            "present": present,
            "iou": true_positive / union if present and union else None,
            "dice": (
                2.0 * true_positive / (target_pixels + predicted_pixels)
                if present and target_pixels + predicted_pixels
                else None
            ),
            "target_pixels": target_pixels,
            "prediction_pixels": predicted_pixels,
        }
    return classes


def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image_rgb.copy()
    foreground = mask > 0
    colours = CLASS_COLOURS_RGB[mask]
    result[foreground] = (
        result[foreground].astype(np.float32) * 0.35
        + colours[foreground].astype(np.float32) * 0.65
    ).astype(np.uint8)
    return result


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    model_factory: Callable[[], torch.nn.Module] = build_floorplan_model,
    device: str = "cpu",
) -> dict:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    validated = validate_export(dataset_path)
    experiment_type = validated["manifest"].get("experiment_type")
    dataset = ExportedFloorplanDataset(dataset_path, augment=False, seed=0)
    model = load_model_strict(checkpoint_path, model_factory, device=device)
    model.eval()
    total_confusion = np.zeros((4, 4), dtype=np.int64)
    samples = []
    first_visual = None
    with torch.no_grad():
        for sample in dataset:
            inputs = sample["image"].unsqueeze(0).to(device)
            logits = model(inputs)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            target = sample["mask"].numpy().astype(np.uint8)
            total_confusion += _confusion(prediction, target)
            class_report = dice_per_class(
                logits.cpu(),
                sample["mask"].unsqueeze(0),
            )
            safe_sample_id = _safe_sample_id(sample["sample_id"])
            relative_sample_dir = Path("samples") / safe_sample_id
            sample_dir = destination / relative_sample_dir
            sample_dir.mkdir(parents=True, exist_ok=False)
            image_rgb = _denormalize(sample["image"])
            _write_image(
                sample_dir / "overlay.png",
                cv2.cvtColor(_overlay(image_rgb, prediction), cv2.COLOR_RGB2BGR),
            )
            _write_image(
                sample_dir / "ground_truth.png",
                cv2.cvtColor(_overlay(image_rgb, target), cv2.COLOR_RGB2BGR),
            )
            np.save(sample_dir / "prediction.npy", prediction, allow_pickle=False)
            samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "classes": {str(key): value for key, value in class_report.items()},
                    "artifact_directory": relative_sample_dir.as_posix(),
                }
            )
            if first_visual is None:
                first_visual = (image_rgb, prediction, target)
    report = {
        "experiment_type": experiment_type,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sample_count": len(dataset),
        "confusion_matrix": total_confusion.tolist(),
        "classes": _metrics(total_confusion),
        "samples": samples,
        "has_independent_validation": False,
    }
    _atomic_json(destination / "metrics.json", report)
    if first_visual is not None:
        image_rgb, prediction, target = first_visual
        _write_image(
            destination / "overlay.png",
            cv2.cvtColor(_overlay(image_rgb, prediction), cv2.COLOR_RGB2BGR),
        )
        _write_image(
            destination / "ground_truth.png",
            cv2.cvtColor(_overlay(image_rgb, target), cv2.COLOR_RGB2BGR),
        )
        np.save(destination / "prediction.npy", prediction, allow_pickle=False)
    return report


def compare_checkpoints(
    old_checkpoint: str | Path,
    new_checkpoint: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    model_factory: Callable[[], torch.nn.Module] = build_floorplan_model,
    device: str = "cpu",
) -> dict:
    destination = Path(output_dir).resolve()
    old_dir = destination / "old"
    new_dir = destination / "new"
    old_report = evaluate_checkpoint(
        old_checkpoint,
        dataset_path,
        old_dir,
        model_factory=model_factory,
        device=device,
    )
    new_report = evaluate_checkpoint(
        new_checkpoint,
        dataset_path,
        new_dir,
        model_factory=model_factory,
        device=device,
    )
    old_overlay = _read_image(old_dir / "overlay.png")
    new_overlay = _read_image(new_dir / "overlay.png")
    ground_truth = _read_image(new_dir / "ground_truth.png")
    panels = []
    for label, image in (
        ("OLD MODEL", old_overlay),
        ("NEW MODEL", new_overlay),
        ("MANUAL LABEL", ground_truth),
    ):
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (18, 25, 34), -1)
        cv2.putText(
            panel,
            label,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    destination.mkdir(parents=True, exist_ok=True)
    _write_image(destination / "old_model_overlay.png", old_overlay)
    _write_image(destination / "new_model_overlay.png", new_overlay)
    _write_image(destination / "ground_truth.png", ground_truth)
    _write_image(destination / "comparison.png", np.hstack(panels))
    report = {
        "experiment_type": old_report["experiment_type"],
        "has_independent_validation": False,
        "old_model": old_report,
        "new_model": new_report,
    }
    _atomic_json(destination / "comparison_metrics.json", report)
    return report
