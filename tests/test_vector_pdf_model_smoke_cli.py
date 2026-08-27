import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vector_pdf_model import CHANNEL_NAMES, VectorProbabilityResult


class VectorPdfModelSmokeCliTests(unittest.TestCase):
    def test_main_runs_injected_probability_runner_and_prints_contract_summary(self):
        from tools.smoke_vector_pdf_model import main

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_root = root / "training-root"
            training_root.mkdir()
            python_path = root / "python.exe"
            python_path.write_bytes(b"")
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            runner_calls = []

            def fake_runner(image_path, artifact_parent, config, *, expected_size):
                runner_calls.append((Path(image_path), config, expected_size))
                probabilities = np.zeros((10, 1024, 1024), dtype=np.float32)
                probabilities.setflags(write=False)
                return VectorProbabilityResult(
                    probabilities=probabilities,
                    inference={"runtime_seconds": 1.25},
                    artifact_dir=Path(artifact_parent),
                    probabilities_sha256="a" * 64,
                )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "--training-root", str(training_root),
                    "--python", str(python_path),
                    "--checkpoint", str(checkpoint),
                    "--device", "cpu",
                ], runner=fake_runner)

        self.assertEqual(status, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["shape"], [10, 1024, 1024])
        self.assertEqual(summary["channel_names"], list(CHANNEL_NAMES))
        self.assertEqual(summary["probabilities_sha256"], "a" * 64)
        self.assertEqual(summary["runtime_metadata"], {"runtime_seconds": 1.25})
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(runner_calls[0][2], (1024, 1024))
        self.assertFalse(runner_calls[0][0].exists())

    def test_main_rejects_missing_paths_without_running_model(self):
        from tools.smoke_vector_pdf_model import main

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main([
                "--training-root", "missing-training-root",
                "--python", "missing-python",
                "--checkpoint", "missing-best.pt",
                "--device", "cpu",
            ], runner=lambda *args, **kwargs: self.fail("runner must not execute"))

        self.assertNotEqual(status, 0)
        self.assertIn("not a directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
