"""Local OCR adapter for numeric architectural dimension labels."""

from __future__ import annotations

import re

import numpy as np


MIN_CONFIDENCE = 0.60
_DIMENSION_TEXT = re.compile(r"(?:\d{3,7}|\d+(?:\.\d+)?(?:mm|m))", re.IGNORECASE)
_rapidocr_engine = None


def _load_rapidocr_engine():
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import RapidOCR

        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def extract_numeric_text_spans(
    image_bgr: np.ndarray,
    page_size_pt: list[float],
    engine=None,
) -> tuple[list[dict], dict]:
    """Return high-confidence numeric OCR boxes in PDF point coordinates."""
    evidence = {
        "status": "completed",
        "candidate_count": 0,
        "accepted_count": 0,
        "min_confidence": MIN_CONFIDENCE,
    }
    try:
        active_engine = engine if engine is not None else _load_rapidocr_engine()
        image_height, image_width = image_bgr.shape[:2]
        page_width, page_height = (float(value) for value in page_size_pt)
        scale_x = page_width / image_width
        scale_y = page_height / image_height
        spans = []
        regions = [[0, 0, image_width, image_height]]
        seen = set()
        for region_x0, region_y0, region_x1, region_y1 in regions:
            output = active_engine(image_bgr[region_y0:region_y1, region_x0:region_x1])
            boxes = output.boxes if output.boxes is not None else []
            texts = output.txts if output.txts is not None else []
            scores = output.scores if output.scores is not None else []
            evidence["candidate_count"] += min(len(boxes), len(texts), len(scores))
            for box, raw_text, raw_score in zip(boxes, texts, scores):
                confidence = float(raw_score)
                text = re.sub(r"\s+", "", str(raw_text)).lower()
                if confidence < MIN_CONFIDENCE or not _DIMENSION_TEXT.fullmatch(text):
                    continue
                points = np.asarray(box, dtype=float)
                points[:, 0] += region_x0
                points[:, 1] += region_y0
                x0 = float(points[:, 0].min() * scale_x)
                x1 = float(points[:, 0].max() * scale_x)
                top = float(points[:, 1].min() * scale_y)
                bottom = float(points[:, 1].max() * scale_y)
                key = (text, round(x0, 2), round(top, 2), round(x1, 2), round(bottom, 2))
                if key in seen:
                    continue
                seen.add(key)
                width = x1 - x0
                height = bottom - top
                spans.append({
                    "text": text,
                    "bbox_pt": [x0, top, x1, bottom],
                    "center_pt": [(x0 + x1) / 2.0, (top + bottom) / 2.0],
                    "direction": "horizontal" if width >= height else "vertical",
                    "font_size": min(width, height),
                    "source": "rapidocr",
                    "confidence": confidence,
                })
        evidence["accepted_count"] = len(spans)
        return spans, evidence
    except Exception as exc:
        evidence.update({
            "status": "failed",
            "accepted_count": 0,
            "reason": str(exc),
        })
        return [], evidence
