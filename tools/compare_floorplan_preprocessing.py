"""Compare annotation-tool and current-website preprocessing on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES = ("background", "wall", "window", "door")
CLASS_COLORS_BGR = (
    (40, 40, 40),
    (60, 76, 231),
    (219, 152, 52),
    (113, 204, 46),
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_nchw(image_rgb: np.ndarray) -> np.ndarray:
    normalized = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def prepare_annotation_tool_input(
    image_bgr: np.ndarray, *, image_size: int = 512
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match the annotation tool's aspect-preserving black letterbox input."""
    height, width = image_bgr.shape[:2]
    scale = image_size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        (resized_width, resized_height),
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return _normalize_nchw(canvas), {
        "original_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "padding": {"top": top, "left": left},
        "model_size": [image_size, image_size],
    }


def restore_annotation_tool_mask(
    model_mask: np.ndarray, metadata: dict[str, Any]
) -> np.ndarray:
    """Crop letterbox padding and restore class IDs to original dimensions."""
    width, height = metadata["original_size"]
    resized_width, resized_height = metadata["resized_size"]
    top = metadata["padding"]["top"]
    left = metadata["padding"]["left"]
    cropped = model_mask[top : top + resized_height, left : left + resized_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST).astype(
        np.uint8
    )


def _remove_annotations(image_bgr: np.ndarray) -> np.ndarray:
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    annotation_ranges = (
        ((50, 60, 60), (85, 255, 255)),
        ((155, 60, 60), (175, 255, 255)),
        ((130, 60, 60), (155, 255, 255)),
        ((10, 80, 80), (35, 255, 255)),
        ((0, 120, 100), (10, 255, 255)),
        ((170, 120, 100), (180, 255, 255)),
    )
    combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for low, high in annotation_ranges:
        mask = cv2.inRange(image_hsv, np.array(low), np.array(high))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    cleaned = image_bgr.copy()
    cleaned[combined_mask > 0] = 0
    return cleaned


def _preprocess_dark_cad(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def prepare_website_input(
    image_bgr: np.ndarray, *, image_size: int = 512
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match the current website's cleanup, dark-CAD transform, and stretching."""
    height, width = image_bgr.shape[:2]
    transformed = _preprocess_dark_cad(_remove_annotations(image_bgr))
    resized = cv2.resize(transformed, (image_size, image_size))
    return _normalize_nchw(resized), {
        "original_size": [width, height],
        "model_size": [image_size, image_size],
    }


def restore_stretched_mask(model_mask: np.ndarray, original_size: list[int]) -> np.ndarray:
    """Restore a directly stretched class mask to original width and height."""
    width, height = original_size
    return cv2.resize(
        model_mask, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)


def compute_class_stats(mask: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Return pixels, share, and external contour count for every class."""
    total = int(mask.size)
    stats: dict[str, dict[str, float | int]] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        binary = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pixels = int(binary.sum())
        stats[name] = {
            "pixels": pixels,
            "percentage": round(pixels / total * 100.0, 4),
            "contours": len(contours),
        }
    return stats


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render website-compatible colors and outlines over the source image."""
    segmentation = np.zeros_like(image_bgr)
    for class_id, color in enumerate(CLASS_COLORS_BGR):
        segmentation[mask == class_id] = color
    overlay = image_bgr.copy()
    foreground = mask > 0
    blended = cv2.addWeighted(image_bgr, 0.4, segmentation, 0.6, 0)
    overlay[foreground] = blended[foreground]
    for class_id in (1, 2, 3):
        binary = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, CLASS_COLORS_BGR[class_id], 2)
    return overlay


def _run_logits(session, tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0]
    if tuple(logits.shape) != (1, 4, 512, 512):
        raise RuntimeError(f"unexpected ONNX output shape: {logits.shape}")
    return logits.argmax(axis=1)[0].astype(np.uint8)


def compare_preprocessing(
    model_path: str | Path, image_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run both inputs through one ONNX session and save comparison artifacts."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for preprocessing comparison") from exc

    source_path = Path(image_path)
    source = cv2.imread(str(source_path))
    if source is None:
        raise ValueError(f"cannot read source image: {source_path}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    annotation_tensor, annotation_metadata = prepare_annotation_tool_input(source)
    annotation_model_mask = _run_logits(session, annotation_tensor)
    annotation_mask = restore_annotation_tool_mask(
        annotation_model_mask, annotation_metadata
    )

    website_tensor, website_metadata = prepare_website_input(source)
    website_model_mask = _run_logits(session, website_tensor)
    website_mask = restore_stretched_mask(
        website_model_mask, website_metadata["original_size"]
    )

    results: dict[str, Any] = {
        "source_image": str(source_path.resolve()),
        "source_size": [source.shape[1], source.shape[0]],
        "model": str(Path(model_path).resolve()),
        "pipelines": {},
    }
    for name, mask, metadata in (
        ("annotation_tool", annotation_mask, annotation_metadata),
        ("website_current", website_mask, website_metadata),
    ):
        mask_path = destination / f"{name}_mask.png"
        overlay_path = destination / f"{name}_overlay.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"failed to write mask: {mask_path}")
        if not cv2.imwrite(str(overlay_path), make_overlay(source, mask)):
            raise RuntimeError(f"failed to write overlay: {overlay_path}")
        results["pipelines"][name] = {
            "metadata": metadata,
            "stats": compute_class_stats(mask),
            "mask": str(mask_path.resolve()),
            "overlay": str(overlay_path.resolve()),
        }

    report_path = destination / "comparison.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    results["report"] = str(report_path.resolve())
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare_preprocessing(args.model, args.image, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
