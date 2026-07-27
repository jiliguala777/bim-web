import unittest
from subprocess import CompletedProcess

import numpy as np

from tools.export_floorplan_onnx import (
    check_onnx_in_subprocess,
    compare_outputs,
    extract_state_dict,
    require_equivalent,
)


class ExportFloorplanOnnxTests(unittest.TestCase):
    def test_checker_subprocess_reports_native_crash_without_killing_parent(self):
        def fake_run(*args, **kwargs):
            return CompletedProcess(args=args[0], returncode=-1073741819, stdout="", stderr="access violation")

        result = check_onnx_in_subprocess("model.onnx", runner=fake_run)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], -1073741819)
        self.assertIn("access violation", result["stderr"])

    def test_checker_subprocess_reports_success(self):
        def fake_run(*args, **kwargs):
            return CompletedProcess(args=args[0], returncode=0, stdout="checker passed\n", stderr="")

        result = check_onnx_in_subprocess("model.onnx", runner=fake_run)

        self.assertEqual(result, {"status": "passed", "returncode": 0})

    def test_extract_state_dict_requires_named_checkpoint_entry(self):
        with self.assertRaisesRegex(ValueError, "model_state_dict"):
            extract_state_dict({"weights": {}})

    def test_extract_state_dict_returns_checkpoint_weights(self):
        weights = {"layer.weight": np.ones((2, 2), dtype=np.float32)}
        self.assertIs(extract_state_dict({"model_state_dict": weights}), weights)

    def test_compare_outputs_reports_numeric_and_argmax_metrics(self):
        torch_output = np.array(
            [[[[2.0, 0.0]], [[0.0, 3.0]], [[-1.0, -1.0]], [[-2.0, -2.0]]]],
            dtype=np.float32,
        )
        onnx_output = torch_output.copy()
        onnx_output[0, 0, 0, 0] += 0.00001

        metrics = compare_outputs(torch_output, onnx_output)

        self.assertEqual(metrics["output_shape"], [1, 4, 1, 2])
        self.assertAlmostEqual(metrics["max_abs_error"], 0.00001, places=6)
        self.assertGreater(metrics["mean_abs_error"], 0.0)
        self.assertEqual(metrics["argmax_pixel_agreement"], 1.0)

    def test_compare_outputs_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_outputs(
                np.zeros((1, 4, 2, 2), dtype=np.float32),
                np.zeros((1, 4, 2, 3), dtype=np.float32),
            )

    def test_require_equivalent_rejects_excess_error(self):
        metrics = {
            "max_abs_error": 0.001,
            "argmax_pixel_agreement": 1.0,
        }
        with self.assertRaisesRegex(RuntimeError, "maximum absolute error"):
            require_equivalent(metrics, max_abs_error=0.0001, min_argmax_agreement=0.99999)

    def test_require_equivalent_rejects_argmax_disagreement(self):
        metrics = {
            "max_abs_error": 0.00001,
            "argmax_pixel_agreement": 0.99,
        }
        with self.assertRaisesRegex(RuntimeError, "argmax"):
            require_equivalent(metrics, max_abs_error=0.0001, min_argmax_agreement=0.99999)


if __name__ == "__main__":
    unittest.main()
