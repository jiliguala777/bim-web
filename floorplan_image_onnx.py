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
        return probabilities.argmax(axis=0).astype(np.uint8), probabilities

    def predict(self, image_input, use_preprocessing=False, **kwargs):
        return super().predict(image_input, use_preprocessing=False, **kwargs)


def get_image_segmenter(model_path):
    return FloorplanImageSegmenterONNX(model_path)
