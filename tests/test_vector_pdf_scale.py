import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas

from vector_pdf_scale import (
    _merge_split_dimension_segments,
    build_dimension_annotation_mask,
    build_nonstructural_vector_mask,
    build_structural_vector_mask,
    calibrate_from_overall_dimensions,
    detect_building_roi,
    detect_dimension_candidates,
    extract_vector_page,
    remove_dimension_annotations,
)


class VectorPdfExtractionTests(unittest.TestCase):
    def _make_vector_pdf(self, directory):
        path = Path(directory) / "vector-dimensions.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(400, 300))
        pdf.setLineWidth(0.5)
        pdf.line(40, 260, 360, 260)
        pdf.line(40, 250, 40, 280)
        pdf.line(360, 250, 360, 280)
        pdf.drawString(178, 266, "40600")
        pdf.line(25, 40, 25, 240)
        pdf.line(15, 40, 35, 40)
        pdf.line(15, 240, 35, 240)
        pdf.saveState()
        pdf.translate(12, 124)
        pdf.rotate(90)
        pdf.drawString(0, 0, "18600")
        pdf.restoreState()
        pdf.drawString(60, 150, "1:100")
        pdf.drawString(170, 150, "68.80m2")
        pdf.drawString(280, 150, "13.500")
        pdf.drawString(170, 110, "C1376")
        pdf.rect(2, 2, 396, 296)
        pdf.save()
        return path

    def _make_image_only_pdf(self, directory):
        image_path = Path(directory) / "scan.png"
        Image.new("RGB", (200, 100), "white").save(image_path)
        path = Path(directory) / "scan.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(400, 300))
        pdf.drawImage(str(image_path), 50, 50, width=300, height=150)
        pdf.save()
        return path

    def _make_geometry_only_pdf(self, directory):
        path = Path(directory) / "geometry-only.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(400, 300))
        pdf.setLineWidth(0.72)
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.rect(80, 60, 240, 180, stroke=1, fill=0)
        pdf.line(200, 60, 200, 240)
        pdf.setStrokeColorRGB(0.5, 0.5, 0.5)
        for y in (100, 120, 140, 160, 180):
            pdf.line(120, y, 280, y)
        pdf.save()
        return path

    def test_extracts_positioned_text_and_axis_aligned_segments_from_vector_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            page = extract_vector_page(self._make_vector_pdf(directory), dpi=200)

        self.assertTrue(page["is_vector_pdf"])
        self.assertEqual(page["page_size_pt"], [400.0, 300.0])
        self.assertEqual(page["render_size_px"], [1111, 833])
        self.assertIn("40600", {span["text"] for span in page["text_spans"]})
        self.assertTrue(any(line["orientation"] == "horizontal" for line in page["segments"]))
        self.assertTrue(any(line["orientation"] == "vertical" for line in page["segments"]))

    def test_image_only_pdf_is_not_reported_as_vector_drawing(self):
        with tempfile.TemporaryDirectory() as directory:
            page = extract_vector_page(self._make_image_only_pdf(directory), dpi=200)

        self.assertFalse(page["is_vector_pdf"])
        self.assertEqual(page["text_spans"], [])
        self.assertEqual(page["segments"], [])

    def test_geometry_only_pdf_is_vector_even_without_extractable_text(self):
        with tempfile.TemporaryDirectory() as directory:
            page = extract_vector_page(self._make_geometry_only_pdf(directory), dpi=200)

        self.assertEqual(page["text_spans"], [])
        self.assertFalse(page["has_vector_text"])
        self.assertTrue(page["has_vector_geometry"])
        self.assertTrue(page["is_vector_pdf"])

    def test_styled_edges_preserve_neutral_gray_colour_and_line_width(self):
        with tempfile.TemporaryDirectory() as directory:
            page = extract_vector_page(self._make_geometry_only_pdf(directory), dpi=200)

        gray_edges = [
            edge for edge in page["styled_edges"]
            if all(abs(channel - 0.5) < 0.01 for channel in edge["stroke_rgb"])
        ]
        self.assertGreaterEqual(len(gray_edges), 5)
        self.assertTrue(all(abs(edge["width_pt"] - 0.72) < 0.01 for edge in gray_edges))

    def test_rotated_dimension_text_and_page_border_are_handled(self):
        with tempfile.TemporaryDirectory() as directory:
            page = extract_vector_page(self._make_vector_pdf(directory), dpi=200)
            candidates = detect_dimension_candidates(page)

        by_text = {item["text"]: item for item in candidates}
        self.assertIn("18600", by_text)
        self.assertNotIn("00681", by_text)
        self.assertAlmostEqual(by_text["18600"]["span_pt"], 200.0)


class VectorPdfCalibrationTests(unittest.TestCase):
    @staticmethod
    def _page_data(vertical_scale_multiplier=1.0, include_horizontal=True, include_vertical=True):
        horizontal_scale = 0.1
        vertical_scale = horizontal_scale * vertical_scale_multiplier
        horizontal_span = 40.0 / horizontal_scale
        vertical_span = 20.0 / vertical_scale
        texts = [
            {"text": "1:100", "bbox_pt": [180, 140, 220, 150], "center_pt": [200, 145], "direction": "horizontal", "font_size": 10},
            {"text": "68.80m2", "bbox_pt": [170, 160, 230, 170], "center_pt": [200, 165], "direction": "horizontal", "font_size": 10},
            {"text": "13.500", "bbox_pt": [170, 180, 230, 190], "center_pt": [200, 185], "direction": "horizontal", "font_size": 10},
            {"text": "C1376", "bbox_pt": [170, 200, 230, 210], "center_pt": [200, 205], "direction": "horizontal", "font_size": 10},
        ]
        segments = [
            {"start_pt": [0, 0], "end_pt": [500, 0], "orientation": "horizontal", "length_pt": 500, "width_pt": 1},
            {"start_pt": [0, 300], "end_pt": [500, 300], "orientation": "horizontal", "length_pt": 500, "width_pt": 1},
        ]
        if include_horizontal:
            texts.append({"text": "40000", "bbox_pt": [225, 22, 275, 32], "center_pt": [250, 27], "direction": "horizontal", "font_size": 10})
            segments.extend([
                {"start_pt": [50, 40], "end_pt": [50 + horizontal_span, 40], "orientation": "horizontal", "length_pt": horizontal_span, "width_pt": 0.5},
                {"start_pt": [50, 32], "end_pt": [50, 48], "orientation": "vertical", "length_pt": 16, "width_pt": 0.5},
                {"start_pt": [50 + horizontal_span, 32], "end_pt": [50 + horizontal_span, 48], "orientation": "vertical", "length_pt": 16, "width_pt": 0.5},
            ])
        if include_vertical:
            texts.append({"text": "20000", "bbox_pt": [12, 125, 22, 175], "center_pt": [17, 150], "direction": "vertical", "font_size": 10})
            segments.extend([
                {"start_pt": [35, 150 - vertical_span / 2], "end_pt": [35, 150 + vertical_span / 2], "orientation": "vertical", "length_pt": vertical_span, "width_pt": 0.5},
                {"start_pt": [27, 150 - vertical_span / 2], "end_pt": [43, 150 - vertical_span / 2], "orientation": "horizontal", "length_pt": 16, "width_pt": 0.5},
                {"start_pt": [27, 150 + vertical_span / 2], "end_pt": [43, 150 + vertical_span / 2], "orientation": "horizontal", "length_pt": 16, "width_pt": 0.5},
            ])
        return {
            "page_size_pt": [500.0, 300.0],
            "render_size_px": [500, 300],
            "dpi": 72,
            "text_spans": texts,
            "segments": segments,
            "is_vector_pdf": True,
        }

    def test_selects_overall_dimensions_and_rejects_non_dimension_numbers(self):
        candidates = detect_dimension_candidates(self._page_data())

        self.assertEqual({item["text"] for item in candidates}, {"40000", "20000"})
        self.assertTrue(all(item["endpoint_evidence"] == 2 for item in candidates))

        calibration = calibrate_from_overall_dimensions(self._page_data())
        self.assertEqual(calibration["status"], "confirmed")
        self.assertEqual(calibration["horizontal"]["text"], "40000")
        self.assertEqual(calibration["vertical"]["text"], "20000")
        self.assertAlmostEqual(calibration["scale_m_per_px"], 0.1)

    def test_ocr_calibration_uses_supported_scale_consensus_instead_of_largest_misread(self):
        page_data = self._page_data()
        for span in page_data["text_spans"]:
            span["source"] = "rapidocr"
        page_data["text_spans"].append({
            "text": "1200700",
            "bbox_pt": [230, 92, 270, 102],
            "center_pt": [250, 97],
            "direction": "horizontal",
            "font_size": 10,
            "source": "rapidocr",
        })
        page_data["segments"].extend([
            {"start_pt": [233, 110], "end_pt": [267, 110], "orientation": "horizontal", "length_pt": 34, "width_pt": 0.5},
            {"start_pt": [233, 102], "end_pt": [233, 118], "orientation": "vertical", "length_pt": 16, "width_pt": 0.5},
            {"start_pt": [267, 102], "end_pt": [267, 118], "orientation": "vertical", "length_pt": 16, "width_pt": 0.5},
        ])

        calibration = calibrate_from_overall_dimensions(page_data)

        self.assertEqual(calibration["status"], "confirmed")
        self.assertEqual(calibration["horizontal"]["text"], "40000")
        self.assertEqual(calibration["vertical"]["text"], "20000")

    def test_two_and_five_percent_axis_difference_thresholds(self):
        ratio_at_two_percent = (2.0 + 0.02) / (2.0 - 0.02)
        ratio_at_five_percent = (2.0 + 0.05) / (2.0 - 0.05)

        at_two = calibrate_from_overall_dimensions(self._page_data(ratio_at_two_percent))
        at_five = calibrate_from_overall_dimensions(self._page_data(ratio_at_five_percent))
        over_five = calibrate_from_overall_dimensions(self._page_data(ratio_at_five_percent + 0.002))

        self.assertEqual(at_two["status"], "confirmed")
        self.assertEqual(at_five["status"], "confirmation_required")
        self.assertEqual(over_five["status"], "conflict")

    def test_single_axis_requires_confirmation_and_no_dimensions_require_manual_input(self):
        horizontal_only = calibrate_from_overall_dimensions(
            self._page_data(include_vertical=False)
        )
        no_dimensions = calibrate_from_overall_dimensions(
            self._page_data(include_horizontal=False, include_vertical=False)
        )

        self.assertEqual(horizontal_only["status"], "confirmation_required")
        self.assertEqual(horizontal_only["horizontal"]["text"], "40000")
        self.assertIsNone(horizontal_only["vertical"])
        self.assertEqual(no_dimensions["status"], "manual_required")
        self.assertIsNone(no_dimensions["scale_m_per_px"])

    def test_dimension_line_split_around_text_is_merged_before_matching(self):
        page_data = self._page_data(include_horizontal=False, include_vertical=False)
        page_data["text_spans"].append({
            "text": "40000",
            "bbox_pt": [225, 22, 275, 32],
            "center_pt": [250, 27],
            "direction": "horizontal",
            "font_size": 10,
        })
        page_data["segments"].extend([
            {"start_pt": [50, 40], "end_pt": [225, 40], "orientation": "horizontal", "length_pt": 175, "width_pt": 0.5},
            {"start_pt": [275, 40], "end_pt": [450, 40], "orientation": "horizontal", "length_pt": 175, "width_pt": 0.5},
            {"start_pt": [50, 32], "end_pt": [50, 48], "orientation": "vertical", "length_pt": 16, "width_pt": 0.5},
            {"start_pt": [450, 32], "end_pt": [450, 48], "orientation": "vertical", "length_pt": 16, "width_pt": 0.5},
        ])

        candidates = detect_dimension_candidates(page_data)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["text"], "40000")
        self.assertAlmostEqual(candidates[0]["span_pt"], 400.0)

    def test_split_dimension_matching_does_not_compare_every_unrelated_line_pair(self):
        class CountingSegment(dict):
            accesses = 0

            def __getitem__(self, key):
                type(self).accesses += 1
                if type(self).accesses > 50000:
                    raise AssertionError("dimension merge scanned unrelated line pairs")
                return super().__getitem__(key)

        span = {
            "text": "40000",
            "bbox_pt": [225, 22, 275, 32],
            "center_pt": [250, 27],
            "direction": "horizontal",
            "font_size": 10,
        }
        segments = [
            CountingSegment({
                "start_pt": [float(index), 200.0],
                "end_pt": [float(index) + 0.5, 200.0],
                "orientation": "horizontal",
                "length_pt": 0.5,
                "width_pt": 0.5,
            })
            for index in range(1000)
        ]
        segments.extend([
            CountingSegment({"start_pt": [50, 40], "end_pt": [225, 40], "orientation": "horizontal", "length_pt": 175, "width_pt": 0.5}),
            CountingSegment({"start_pt": [275, 40], "end_pt": [450, 40], "orientation": "horizontal", "length_pt": 175, "width_pt": 0.5}),
        ])

        merged = _merge_split_dimension_segments(span, segments)

        self.assertEqual(len(merged), 1)
        self.assertLess(CountingSegment.accesses, 50000)


class VectorPdfAnnotationMaskTests(unittest.TestCase):
    @staticmethod
    def _layered_page_data():
        edges = []
        black = [0.0, 0.0, 0.0]
        gray = [0.5, 0.5, 0.5]
        for x in (80, 120, 160, 200, 240, 280, 320):
            edges.append({"start_pt": [x, 50], "end_pt": [x, 240], "stroke_rgb": black, "width_pt": 0.72})
        for y in (50, 80, 110, 140, 170, 200, 240):
            edges.append({"start_pt": [80, y], "end_pt": [320, y], "stroke_rgb": black, "width_pt": 0.72})
        for y in (100, 120, 140, 160, 180):
            edges.append({"start_pt": [130, y], "end_pt": [270, y], "stroke_rgb": gray, "width_pt": 0.72})
        # Sparse exterior dimensions and a dense but shallow title strip.
        edges.extend([
            {"start_pt": [20, 30], "end_pt": [380, 30], "stroke_rgb": black, "width_pt": 0.72},
            {"start_pt": [20, 20], "end_pt": [20, 270], "stroke_rgb": black, "width_pt": 0.72},
        ])
        for x in range(10, 391, 20):
            edges.append({"start_pt": [x, 275], "end_pt": [x, 295], "stroke_rgb": black, "width_pt": 0.72})
        for y in (275, 285, 295):
            edges.append({"start_pt": [10, y], "end_pt": [390, y], "stroke_rgb": black, "width_pt": 0.72})
        return {
            "page_size_pt": [400.0, 300.0],
            "render_size_px": [800, 600],
            "styled_edges": edges,
            "has_vector_geometry": True,
            "is_vector_pdf": True,
        }

    def test_nonstructural_mask_removes_gray_furniture_but_not_black_walls(self):
        mask, evidence = build_nonstructural_vector_mask(self._layered_page_data())

        self.assertTrue(evidence["enabled"])
        self.assertGreater(evidence["gray_edge_count"], 0)
        self.assertEqual(mask[240, 400], 255)
        self.assertEqual(mask[100, 400], 0)

    def test_nonstructural_mask_stays_disabled_without_black_structure_evidence(self):
        page = self._layered_page_data()
        page["styled_edges"] = [
            edge for edge in page["styled_edges"] if edge["stroke_rgb"][0] > 0.3
        ]

        mask, evidence = build_nonstructural_vector_mask(page)

        self.assertFalse(evidence["enabled"])
        self.assertEqual(int(mask.max()), 0)

    def test_structural_mask_rasterizes_dark_edges_but_excludes_gray_furniture(self):
        mask, evidence = build_structural_vector_mask(self._layered_page_data())

        self.assertTrue(evidence["enabled"])
        self.assertGreaterEqual(evidence["dark_edge_count"], 8)
        self.assertEqual(mask[100, 400], 255)
        self.assertEqual(mask[240, 300], 0)

    def test_structural_mask_stays_disabled_without_enough_dark_edges(self):
        page = self._layered_page_data()
        page["styled_edges"] = page["styled_edges"][:7]

        mask, evidence = build_structural_vector_mask(page)

        self.assertFalse(evidence["enabled"])
        self.assertEqual(evidence["dark_edge_count"], 7)
        self.assertEqual(int(mask.max()), 0)

    def test_building_roi_excludes_sparse_dimensions_and_shallow_title_strip(self):
        roi = detect_building_roi(self._layered_page_data())

        self.assertTrue(roi["enabled"], roi)
        x0, y0, x1, y1 = roi["bbox_px"]
        self.assertGreater(x0, 40)
        self.assertGreater(y0, 40)
        self.assertLess(x1, 760)
        self.assertLess(y1, 550)
        self.assertLessEqual(x0, 160)
        self.assertGreaterEqual(x1, 640)
    def test_masks_dimension_text_baseline_and_endpoints_but_not_wall_geometry(self):
        page_data = VectorPdfCalibrationTests._page_data()
        candidates = detect_dimension_candidates(page_data)

        mask = build_dimension_annotation_mask(page_data, candidates)

        self.assertEqual(mask.shape, (300, 500))
        self.assertEqual(mask[40, 250], 255)
        self.assertEqual(mask[27, 250], 255)
        self.assertEqual(mask[40, 50], 255)
        self.assertEqual(mask[100, 250], 0)

    def test_annotation_removal_fills_mask_with_page_border_colour(self):
        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        image[40, 50:451] = 0
        page_data = VectorPdfCalibrationTests._page_data()
        mask = build_dimension_annotation_mask(
            page_data,
            detect_dimension_candidates(page_data),
        )

        cleaned = remove_dimension_annotations(image, mask)

        self.assertTrue(np.all(cleaned[40, 250] == 255))
        self.assertTrue(np.all(cleaned[100, 250] == 255))


if __name__ == "__main__":
    unittest.main()
