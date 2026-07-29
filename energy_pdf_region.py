import json
import math

import numpy as np
from pdf2image import convert_from_path


def render_pdf_page_preview(pdf_path, page_number, page_count, poppler_path):
    if not 1 <= page_number <= page_count:
        raise ValueError("page_number must be within the document")
    images = convert_from_path(
        str(pdf_path),
        dpi=100,
        first_page=page_number,
        last_page=page_number,
        poppler_path=poppler_path,
    )
    if len(images) != 1:
        raise ValueError("renderer must return exactly one image")
    return np.asarray(images[0].convert("RGB"))[:, :, ::-1].copy()


def _parse_numeric_array(raw, *, length, field_name):
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be a JSON array") from exc
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must be a JSON array of {length} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field_name} values must be numbers")
    if not all(math.isfinite(item) for item in value):
        raise ValueError(f"{field_name} values must be finite")
    return value


def parse_crop_region_request(mode_raw, bbox_raw, preview_size_raw):
    if mode_raw not in {"full_page", "crop_region"}:
        raise ValueError("mode must be full_page or crop_region")
    if mode_raw == "full_page":
        return {"mode": "full_page", "crop_bbox_px": None, "preview_size": None}

    crop_bbox_px = _parse_numeric_array(bbox_raw, length=4, field_name="crop_bbox_px")
    preview_size = _parse_numeric_array(preview_size_raw, length=2, field_name="preview_size")
    left, top, right, bottom = crop_bbox_px
    preview_width, preview_height = preview_size
    if preview_width <= 0 or preview_height <= 0:
        raise ValueError("preview_size must be positive")
    if not (0 <= left < right <= preview_width and 0 <= top < bottom <= preview_height):
        raise ValueError("crop_bbox_px must be within preview bounds")
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < 128 or crop_height < 128:
        raise ValueError("crop_bbox_px dimensions must be at least 128 pixels")
    if crop_width * crop_height < preview_width * preview_height * 0.01:
        raise ValueError("crop_bbox_px area must be at least 1% of preview")
    return {
        "mode": "crop_region",
        "crop_bbox_px": crop_bbox_px,
        "preview_size": preview_size,
    }


def map_crop_bbox_to_page(crop_bbox_px, crop_preview_size, page_size):
    try:
        left, top, right, bottom = crop_bbox_px
        preview_width, preview_height = crop_preview_size
        page_width, page_height = page_size
    except (TypeError, ValueError) as exc:
        raise ValueError("crop and image sizes must have the required values") from exc
    values = [
        left,
        top,
        right,
        bottom,
        preview_width,
        preview_height,
        page_width,
        page_height,
    ]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("crop and image sizes must be numbers")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("crop and image sizes must be finite")
    if preview_width <= 0 or preview_height <= 0 or page_width <= 0 or page_height <= 0:
        raise ValueError("image sizes must be positive")
    if not (0 <= left < right <= preview_width and 0 <= top < bottom <= preview_height):
        raise ValueError("crop_bbox_px must be within preview bounds")
    mapped_left = max(0, math.floor(left * page_width / preview_width))
    mapped_top = max(0, math.floor(top * page_height / preview_height))
    mapped_right = min(page_width, math.ceil(right * page_width / preview_width))
    mapped_bottom = min(page_height, math.ceil(bottom * page_height / preview_height))
    if mapped_left >= mapped_right or mapped_top >= mapped_bottom:
        raise ValueError("mapped crop_bbox_px must have positive dimensions")
    return [mapped_left, mapped_top, mapped_right, mapped_bottom]


def crop_page_inputs(
    render_bgr,
    cleaned_bgr,
    cleanup_mask,
    structural_support_mask,
    inference_roi,
    crop_bbox_page_px,
):
    left, top, right, bottom = crop_bbox_page_px
    page_height, page_width = render_bgr.shape[:2]
    if not (0 <= left < right <= page_width and 0 <= top < bottom <= page_height):
        raise ValueError("crop_bbox_page_px must be within page bounds")

    def crop(array):
        if array is None:
            return None
        return array[top:bottom, left:right].copy()

    crop_width = right - left
    crop_height = bottom - top
    crop_roi = [0, 0, crop_width, crop_height]
    if inference_roi is not None:
        roi_left, roi_top, roi_right, roi_bottom = inference_roi
        intersection_left = max(left, roi_left)
        intersection_top = max(top, roi_top)
        intersection_right = min(right, roi_right)
        intersection_bottom = min(bottom, roi_bottom)
        if intersection_left < intersection_right and intersection_top < intersection_bottom:
            crop_roi = [
                intersection_left - left,
                intersection_top - top,
                intersection_right - left,
                intersection_bottom - top,
            ]

    return {
        "render_bgr": crop(render_bgr),
        "cleaned_bgr": crop(cleaned_bgr),
        "cleanup_mask": crop(cleanup_mask),
        "structural_support_mask": crop(structural_support_mask),
        "inference_roi": crop_roi,
    }
