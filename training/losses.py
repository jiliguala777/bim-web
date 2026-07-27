"""Losses and explicit per-class metrics for four-class segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


NUM_CLASSES = 4


def dice_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[int, dict[str, float | bool | int | None]]:
    if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
        raise ValueError("logits must have shape [N,4,H,W]")
    if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("target dimensions must match logits")
    prediction = logits.argmax(dim=1)
    report = {}
    for class_id in range(NUM_CLASSES):
        target_class = target == class_id
        prediction_class = prediction == class_id
        target_pixels = int(target_class.sum().item())
        prediction_pixels = int(prediction_class.sum().item())
        present = target_pixels > 0
        if present:
            intersection = int((target_class & prediction_class).sum().item())
            dice = (2.0 * intersection) / max(target_pixels + prediction_pixels, 1)
        else:
            dice = None
        report[class_id] = {
            "present": present,
            "dice": dice,
            "target_pixels": target_pixels,
            "prediction_pixels": prediction_pixels,
        }
    return report


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        *,
        cross_entropy_weight: float = 0.6,
        dice_weight: float = 0.4,
        smooth: float = 1.0,
    ):
        super().__init__()
        weights = (
            torch.ones(NUM_CLASSES, dtype=torch.float32)
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32)
        )
        if tuple(weights.shape) != (NUM_CLASSES,):
            raise ValueError("class_weights must contain four values")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("class_weights must be finite and positive")
        self.register_buffer("class_weights", weights)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    @staticmethod
    def weights_from_class_pixels(
        class_pixels: Mapping[str | int, int],
        *,
        minimum: float = 0.1,
        maximum: float = 10.0,
    ) -> list[float]:
        counts = torch.tensor(
            [float(class_pixels.get(str(index), class_pixels.get(index, 0))) for index in range(4)],
            dtype=torch.float64,
        )
        present = counts > 0
        weights = torch.ones(4, dtype=torch.float64)
        if present.any():
            frequencies = counts[present] / counts[present].sum()
            inverse = torch.rsqrt(frequencies)
            inverse = inverse / inverse.mean()
            weights[present] = inverse
        weights = weights.clamp(min=minimum, max=maximum)
        return [float(value) for value in weights.tolist()]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
            raise ValueError("logits must have shape [N,4,H,W]")
        if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError("target dimensions must match logits")
        cross_entropy = F.cross_entropy(logits, target, weight=self.class_weights)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=NUM_CLASSES).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        present = one_hot.sum(dim=dimensions) > 0
        soft_dice_loss = 1.0 - dice[present].mean()
        return (
            self.cross_entropy_weight * cross_entropy
            + self.dice_weight * soft_dice_loss
        )
