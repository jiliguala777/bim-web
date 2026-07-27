import unittest

import numpy as np

from floorplan_ocr import extract_numeric_text_spans


class _FakeOutput:
    def __init__(self):
        self.boxes = np.asarray([
            [[100, 20], [300, 20], [300, 40], [100, 40]],
            [[20, 50], [40, 50], [40, 170], [20, 170]],
            [[40, 80], [140, 80], [140, 100], [40, 100]],
            [[40, 110], [140, 110], [140, 130], [40, 130]],
            [[40, 140], [140, 140], [140, 160], [40, 160]],
        ], dtype=np.float32)
        self.txts = ("40600", "18600", "99999", "Room 101", "1:100")
        self.scores = (0.98, 0.96, 0.30, 0.99, 0.99)


class _FakeEngine:
    def __call__(self, image):
        self.image = image
        return _FakeOutput()


class FloorplanOcrTests(unittest.TestCase):
    def test_extracts_only_high_confidence_dimension_numbers_and_maps_boxes(self):
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        engine = _FakeEngine()

        spans, evidence = extract_numeric_text_spans(
            image,
            [200.0, 100.0],
            engine=engine,
        )

        self.assertTrue(np.shares_memory(engine.image, image))
        self.assertEqual(engine.image.shape, image.shape)
        self.assertEqual({span["text"] for span in spans}, {"40600", "18600"})
        by_text = {span["text"]: span for span in spans}
        self.assertEqual(by_text["40600"]["bbox_pt"], [50.0, 10.0, 150.0, 20.0])
        self.assertEqual(by_text["40600"]["center_pt"], [100.0, 15.0])
        self.assertEqual(by_text["40600"]["direction"], "horizontal")
        self.assertEqual(by_text["18600"]["direction"], "vertical")
        self.assertEqual(by_text["40600"]["source"], "rapidocr")
        self.assertAlmostEqual(by_text["40600"]["confidence"], 0.98)
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["candidate_count"], 5)
        self.assertEqual(evidence["accepted_count"], 2)
        self.assertEqual(evidence["min_confidence"], 0.60)

    def test_engine_failure_returns_diagnostics_instead_of_raising(self):
        class FailingEngine:
            def __call__(self, image):
                raise RuntimeError("model unavailable")

        spans, evidence = extract_numeric_text_spans(
            np.full((20, 20, 3), 255, dtype=np.uint8),
            [20.0, 20.0],
            engine=FailingEngine(),
        )

        self.assertEqual(spans, [])
        self.assertEqual(evidence["status"], "failed")
        self.assertIn("model unavailable", evidence["reason"])


if __name__ == "__main__":
    unittest.main()
