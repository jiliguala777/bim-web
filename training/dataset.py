"""Validated deterministic loader for exported floorplan annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_SIZE = 512
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _safe_file(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the export directory") from exc
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {relative}")
    return path


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("manifest.json is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json is not valid JSON") from exc
    if not isinstance(manifest.get("samples"), list) or not manifest["samples"]:
        raise ValueError("manifest must contain at least one sample")
    return manifest


def _read_image(path: Path) -> np.ndarray | None:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def validate_export(root: str | Path) -> dict[str, Any]:
    export_root = Path(root).resolve()
    manifest = _load_manifest(export_root)
    validated = []
    for index, sample in enumerate(manifest["samples"]):
        if sample.get("status") != "confirmed":
            raise ValueError(f"sample {index} must have confirmed status")
        image_path = _safe_file(export_root, sample.get("image", ""), "image")
        mask_path = _safe_file(export_root, sample.get("mask", ""), "mask")
        image = _read_image(image_path)
        if image is None:
            raise ValueError(f"image cannot be decoded: {sample.get('image')}")
        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("training image must be exactly 512x512")
        try:
            mask = np.load(mask_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"mask cannot be loaded: {sample.get('mask')}") from exc
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("training mask must be exactly 512x512")
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("training mask must contain integer class values")
        if mask.size and (int(mask.min()) < 0 or int(mask.max()) > 3):
            raise ValueError("training mask values must be within 0..3")
        validated.append(
            {
                **sample,
                "sample_id": sample.get("sample_id")
                or f"page-{sample.get('page_number', index + 1)}",
                "_image_path": image_path,
                "_mask_path": mask_path,
            }
        )
    return {
        "root": export_root,
        "manifest": manifest,
        "samples": validated,
        "sample_count": len(validated),
    }


def _morphological_width(mask: np.ndarray, operation: int) -> np.ndarray:
    if operation == 0:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = np.zeros_like(mask)
    for class_id in (1, 2, 3):
        binary = (mask == class_id).astype(np.uint8)
        if operation > 0:
            changed = cv2.dilate(binary, kernel, iterations=1)
        else:
            changed = cv2.erode(binary, kernel, iterations=1)
        result[changed > 0] = class_id
    return result


def _augment_pair(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    image = image_rgb
    target = mask
    if rng.random() < 0.5:
        image, target = np.fliplr(image), np.fliplr(target)
    if rng.random() < 0.25:
        image, target = np.flipud(image), np.flipud(target)
    rotations = int(rng.integers(0, 4))
    if rotations:
        image, target = np.rot90(image, rotations), np.rot90(target, rotations)

    if rng.random() < 0.65:
        scale = float(rng.uniform(0.94, 1.06))
        tx = float(rng.uniform(-14, 14))
        ty = float(rng.uniform(-14, 14))
        matrix = cv2.getRotationMatrix2D((255.5, 255.5), 0, scale)
        matrix[:, 2] += [tx, ty]
        image = cv2.warpAffine(
            image,
            matrix,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        target = cv2.warpAffine(
            target,
            matrix,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.9, 1.1))
        beta = float(rng.uniform(-12, 12))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if rng.random() < 0.2:
        quality = int(rng.integers(75, 96))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rng.random() < 0.35:
        target = _morphological_width(target, int(rng.choice([-1, 1])))
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


class ExportedFloorplanDataset(Dataset):
    def __init__(self, root: str | Path, *, augment: bool, seed: int):
        validated = validate_export(root)
        self.root = validated["root"]
        self.samples = validated["samples"]
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_bgr = _read_image(sample["_image_path"])
        if image_bgr is None:
            raise ValueError(f"image cannot be decoded: {sample['image']}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = np.load(sample["_mask_path"], allow_pickle=False).astype(np.uint8)
        if self.augment:
            image_rgb, mask = _augment_pair(
                image_rgb,
                mask,
                np.random.default_rng(self.seed + index),
            )
        normalized = image_rgb.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(normalized.transpose(2, 0, 1))
        ).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample["sample_id"],
        }
