import unittest
import time

import numpy as np


class NativeLineCandidateTests(unittest.TestCase):
    @staticmethod
    def _segment(native_id, start, end, orientation, brightness, *, border=False):
        return {
            "native_id": native_id,
            "start_pt": list(start),
            "end_pt": list(end),
            "orientation": orientation,
            "length_pt": abs(end[0] - start[0]) + abs(end[1] - start[1]),
            "width_pt": 1.0,
            "stroke_rgb": [brightness, brightness, brightness],
            "neutral_brightness": brightness,
            "is_page_border": border,
        }

    def _page_data(self):
        return {
            "page_size_pt": [100.0, 100.0],
            "render_size_px": [100, 100],
            "building_roi": {"enabled": True, "bbox_px": [10, 10, 90, 90]},
            "orthogonal_segments": [
                self._segment("wall-1", (10, 20), (90, 20), "horizontal", 0.0),
                self._segment("dimension-1", (10, 5), (90, 5), "horizontal", 0.0),
                self._segment("border-1", (0, 0), (100, 0), "horizontal", 0.0, border=True),
                self._segment("outside-1", (95, 10), (95, 90), "vertical", 0.0),
                self._segment("gray-1", (10, 50), (90, 50), "horizontal", 0.5),
            ],
            "dimension_candidates": [{
                "dimension_id": "dimension-0001",
                "member_native_ids": ["dimension-1"],
            }],
        }

    def test_hard_exclusions_are_traceable_and_stable(self):
        from vector_pdf_fusion import FusionThresholds, build_line_candidates

        candidates = build_line_candidates(self._page_data(), FusionThresholds())
        by_source = {item["source_native_id"]: item for item in candidates}

        self.assertEqual(by_source["dimension-1"]["decision"], "rejected_nonstructural")
        self.assertIn("dimension_overlap", by_source["dimension-1"]["reason_codes"])
        self.assertEqual(by_source["border-1"]["decision"], "rejected_nonstructural")
        self.assertIn("page_border", by_source["border-1"]["reason_codes"])
        self.assertEqual(by_source["outside-1"]["decision"], "rejected_nonstructural")
        self.assertIn("outside_building_roi", by_source["outside-1"]["reason_codes"])
        self.assertEqual(by_source["wall-1"]["decision"], "uncertain")
        self.assertEqual(by_source["gray-1"]["decision"], "uncertain")
        self.assertEqual(
            [item["candidate_id"] for item in candidates],
            [f"line-{index:05d}" for index in range(1, 6)],
        )


class ModelEvidenceFusionTests(unittest.TestCase):
    @staticmethod
    def _page_data(brightness=0.0, *, dimension=False):
        native_id = "dimension-1" if dimension else "wall-1"
        return {
            "page_size_pt": [100.0, 100.0],
            "render_size_px": [100, 100],
            "building_roi": {"enabled": True, "bbox_px": [5, 5, 95, 95]},
            "orthogonal_segments": [{
                "native_id": native_id,
                "start_pt": [10.0, 50.0],
                "end_pt": [90.0, 50.0],
                "orientation": "horizontal",
                "length_pt": 80.0,
                "width_pt": 1.0,
                "stroke_rgb": [brightness, brightness, brightness],
                "neutral_brightness": brightness,
                "is_page_border": False,
            }],
            "dimension_candidates": ([{
                "dimension_id": "dimension-0001",
                "member_native_ids": [native_id],
            }] if dimension else []),
        }

    @staticmethod
    def _supported_probabilities():
        probabilities = np.zeros((10, 100, 100), dtype=np.float32)
        probabilities[0, :, :] = 0.8
        probabilities[2, :, :] = 0.7
        probabilities[4, 47:54, 10:91] = 0.8
        return probabilities

    def test_accepts_dark_native_line_with_wall_room_and_footprint_support(self):
        from vector_pdf_fusion import (
            FusionThresholds,
            build_line_candidates,
            fuse_line_candidates,
        )

        thresholds = FusionThresholds()
        candidates = build_line_candidates(self._page_data(), thresholds)
        result = fuse_line_candidates(
            candidates,
            self._supported_probabilities(),
            (100, 100),
            [5, 5, 95, 95],
            thresholds,
        )

        self.assertEqual(result[0]["decision"], "accepted_wall_candidate")
        self.assertGreaterEqual(result[0]["model_evidence"]["wall_p90"], 0.55)
        self.assertIn("model_wall_supported", result[0]["reason_codes"])
        self.assertIn("room_side_supported", result[0]["reason_codes"])

    def test_keeps_weak_model_or_gray_native_lines_uncertain(self):
        from vector_pdf_fusion import (
            FusionThresholds,
            build_line_candidates,
            fuse_line_candidates,
        )

        thresholds = FusionThresholds()
        weak_candidates = build_line_candidates(self._page_data(), thresholds)
        weak = fuse_line_candidates(
            weak_candidates,
            np.zeros((10, 100, 100), dtype=np.float32),
            (100, 100),
            [5, 5, 95, 95],
            thresholds,
        )
        gray_candidates = build_line_candidates(self._page_data(brightness=0.5), thresholds)
        gray = fuse_line_candidates(
            gray_candidates,
            self._supported_probabilities(),
            (100, 100),
            [5, 5, 95, 95],
            thresholds,
        )

        self.assertEqual(weak[0]["decision"], "uncertain")
        self.assertIn("weak_model_support", weak[0]["reason_codes"])
        self.assertEqual(gray[0]["decision"], "uncertain")
        self.assertIn("weak_native_support", gray[0]["reason_codes"])

    def test_hard_rejection_cannot_be_overridden_by_model_probability(self):
        from vector_pdf_fusion import (
            FusionThresholds,
            build_line_candidates,
            fuse_line_candidates,
        )

        thresholds = FusionThresholds()
        candidates = build_line_candidates(self._page_data(dimension=True), thresholds)
        result = fuse_line_candidates(
            candidates,
            self._supported_probabilities(),
            (100, 100),
            [5, 5, 95, 95],
            thresholds,
        )

        self.assertEqual(result[0]["decision"], "rejected_nonstructural")
        self.assertIn("dimension_overlap", result[0]["reason_codes"])
        self.assertIsInstance(result[0]["model_evidence"], dict)
        self.assertGreater(result[0]["model_evidence"]["wall_mean"], 0.0)

    def test_clipped_and_outside_lines_have_stable_empty_sampling(self):
        from vector_pdf_fusion import FusionThresholds, fuse_line_candidates

        def candidate(line_id, orientation, start, end):
            return {
                "candidate_id": line_id,
                "source_native_id": f"native-{line_id}",
                "orientation": orientation,
                "start_px": list(start),
                "end_px": list(end),
                "native_evidence": {
                    "native_structural": True,
                    "roi_inside_ratio": 1.0,
                },
                "model_evidence": None,
                "decision": "uncertain",
                "reason_codes": ["model_evidence_pending"],
            }

        candidates = [
            candidate("top-clipped", "horizontal", (-20, 0), (20, 0)),
            candidate("left-clipped", "vertical", (0, -20), (0, 20)),
            candidate("above-image", "horizontal", (10, -100), (20, -100)),
            candidate("right-of-image", "vertical", (200, 10), (200, 20)),
        ]
        result = fuse_line_candidates(
            candidates,
            np.ones((10, 100, 100), dtype=np.float32),
            (100, 100),
            [0, 0, 100, 100],
            FusionThresholds(),
        )
        by_id = {item["candidate_id"]: item for item in result}

        self.assertEqual(by_id["top-clipped"]["model_evidence"]["wall_mean"], 1.0)
        self.assertEqual(by_id["left-clipped"]["model_evidence"]["wall_mean"], 1.0)
        self.assertEqual(by_id["above-image"]["model_evidence"]["wall_mean"], 0.0)
        self.assertEqual(by_id["right-of-image"]["model_evidence"]["wall_mean"], 0.0)
        self.assertEqual(by_id["above-image"]["decision"], "uncertain")
        self.assertEqual(by_id["right-of-image"]["decision"], "uncertain")

    def test_many_short_lines_do_not_allocate_full_page_masks_per_candidate(self):
        from vector_pdf_fusion import FusionThresholds, fuse_line_candidates

        width, height = 1600, 1200
        probabilities = np.broadcast_to(
            np.zeros((10, 1, 1), dtype=np.float32),
            (10, height, width),
        )
        candidates = []
        for index in range(500):
            x = 20 + (index % 50) * 30
            y = 20 + (index // 50) * 30
            candidates.append({
                "candidate_id": f"line-{index:05d}",
                "source_native_id": f"native-{index:05d}",
                "orientation": "horizontal",
                "start_px": [x, y],
                "end_px": [x + 20, y],
                "native_evidence": {
                    "native_structural": True,
                    "roi_inside_ratio": 1.0,
                },
                "model_evidence": None,
                "decision": "uncertain",
                "reason_codes": ["model_evidence_pending"],
            })

        started = time.perf_counter()
        result = fuse_line_candidates(
            candidates,
            probabilities,
            (width, height),
            [0, 0, width, height],
            FusionThresholds(),
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(len(result), 500)
        self.assertTrue(all(item["decision"] == "uncertain" for item in result))
        self.assertTrue(all(item["model_evidence"]["wall_mean"] == 0.0 for item in result))
        self.assertLess(elapsed, 1.5, f"500 short-line samples took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
