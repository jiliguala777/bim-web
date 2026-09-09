"""ONNX inference for the 256px dual-head raster-floorplan model."""

from pathlib import Path

import cv2
import numpy as np

from floorplan_onnx import FloorplanSegmenterONNX, HAS_ONNX, ort


IMAGE_MODEL_SIZE = 256


def prepare_image_tensor(image_rgb, size=IMAGE_MODEL_SIZE):
    """Match the training dataset's black letterbox and [0, 1] RGB tensor."""
    height, width = image_rgb.shape[:2]
    scale = size / max(width, height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top, left = (size - resized_height) // 2, (size - resized_width) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    tensor = canvas.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
    return tensor, {
        "original_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "padding": [top, left, size - resized_height - top, size - resized_width - left],
    }


def remap_element_classes(training_mask):
    """Map training [background, wall, door, window] to platform wall/window/door."""
    return np.array([0, 1, 3, 2], dtype=np.uint8)[np.asarray(training_mask, dtype=np.uint8)]


def footprint_mask_from_logits(logits, output_size):
    """Restore the binary footprint head to source-image coordinates."""
    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)[0, 0]))
    mask = (cv2.resize(probability, output_size, interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return result


def exterior_element_lengths(mask, footprint, boundary_band_px=None):
    """Estimate exterior wall/window/door lengths on the four page sides in pixels."""
    footprint = np.asarray(footprint, dtype=np.uint8)
    height, width = footprint.shape
    band_px = boundary_band_px or max(6, min(24, round(min(height, width) * 0.015)))
    edge = cv2.morphologyEx(footprint, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    edge = cv2.dilate(edge, np.ones((band_px * 2 + 1, band_px * 2 + 1), dtype=np.uint8))
    moments = cv2.moments(footprint)
    center_x = moments["m10"] / moments["m00"] if moments["m00"] else width / 2
    center_y = moments["m01"] / moments["m00"] if moments["m00"] else height / 2
    output = {side: {"wall_px": 0.0, "window_px": 0.0, "door_px": 0.0} for side in ("top", "right", "bottom", "left")}
    y_grid, x_grid = np.indices(footprint.shape)
    dx, dy = x_grid - center_x, y_grid - center_y
    side_masks = {
        "top": (dy < 0) & (np.abs(dy) >= np.abs(dx)),
        "right": (dx >= 0) & (np.abs(dx) > np.abs(dy)),
        "bottom": (dy >= 0) & (np.abs(dy) >= np.abs(dx)),
        "left": (dx < 0) & (np.abs(dx) > np.abs(dy)),
    }
    for class_id, name in ((1, "wall_px"), (2, "window_px"), (3, "door_px")):
        exterior = (np.asarray(mask) == class_id) & (edge > 0)
        for side, side_mask in side_masks.items():
            coordinates = np.where(exterior & side_mask)
            output[side][name] = float(len(np.unique(coordinates[1 if side in ("top", "bottom") else 0])))
    return output


class FloorplanImageSegmenterONNX(FloorplanSegmenterONNX):
    """Keep the established geometry and energy contract for the new image model."""

    def __init__(self, model_path):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required")
        self.session = ort.InferenceSession(
            str(Path(model_path)), providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = IMAGE_MODEL_SIZE
        self.num_classes = 4

    def prepare_model_input(self, image_rgb):
        tensor, metadata = prepare_image_tensor(image_rgb, self.img_size)
        return image_rgb, tensor, metadata

    def _run_inference(self, image_rgb):
        height, width = image_rgb.shape[:2]
        _, tensor, metadata = self.prepare_model_input(image_rgb)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 2 or outputs[0].shape[:2] != (1, self.num_classes):
            raise ValueError("image ONNX model must return element and footprint logits")
        logits = np.asarray(outputs[0], dtype=np.float32)[0]
        top, left, _, _ = metadata["padding"]
        resized_width, resized_height = metadata["resized_size"]
        logits = logits[:, top:top + resized_height, left:left + resized_width]
        probabilities = np.empty((self.num_classes, height, width), dtype=np.float32)
        for index in range(self.num_classes):
            probabilities[index] = cv2.resize(logits[index], (width, height), interpolation=cv2.INTER_LINEAR)
        probabilities = np.exp(probabilities - probabilities.max(axis=0, keepdims=True))
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        probabilities = probabilities[[0, 1, 3, 2]]
        footprint_logits = np.asarray(outputs[1], dtype=np.float32)[:, :, top:top + resized_height, left:left + resized_width]
        footprint = footprint_mask_from_logits(footprint_logits, (width, height))
        roi = cv2.dilate(footprint, np.ones((9, 9), dtype=np.uint8))
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        prediction[roi == 0] = 0
        return prediction, probabilities, {"footprint_mask": footprint}

    def predict(self, image_input, use_preprocessing=False, **kwargs):
        result = super().predict(image_input, use_preprocessing=False, **kwargs)
        footprint = result["footprint_mask"].astype(bool)
        result["exterior_elements"] = exterior_element_lengths(result["mask"], result["footprint_mask"])
        purple = np.zeros_like(result["overlay"])
        purple[:] = (180, 80, 180)
        result["overlay"][footprint] = cv2.addWeighted(result["overlay"], 0.55, purple, 0.45, 0)[footprint]
        return result


def get_image_segmenter(model_path):
    return FloorplanImageSegmenterONNX(model_path)
