import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class VectorProbabilityArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "prediction"
        self.output.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_artifacts(self, probabilities, channel_names):
        np.savez_compressed(self.output / "probabilities.npz", probabilities=probabilities)
        (self.output / "inference.json").write_text(
            json.dumps({
                "format": "vector-floorplan-inference/1",
                "channel_names": channel_names,
                "image_size": [probabilities.shape[2], probabilities.shape[1]],
                "runtime_seconds": 1.25,
            }),
            encoding="utf-8",
        )

    def test_loads_ten_aligned_finite_probability_channels(self):
        from vector_pdf_model import CHANNEL_NAMES, load_probability_artifacts

        probabilities = np.zeros((10, 20, 30), dtype=np.float32)
        self._write_artifacts(probabilities, list(CHANNEL_NAMES))

        result = load_probability_artifacts(self.output, expected_size=(30, 20))

        self.assertEqual(result.probabilities.shape, (10, 20, 30))
        self.assertFalse(result.probabilities.flags.writeable)
        self.assertEqual(len(result.probabilities_sha256), 64)
        self.assertEqual(result.inference["runtime_seconds"], 1.25)

    def test_rejects_wrong_channel_order(self):
        from vector_pdf_model import (
            CHANNEL_NAMES,
            VectorModelContractError,
            load_probability_artifacts,
        )

        probabilities = np.zeros((10, 20, 30), dtype=np.float32)
        self._write_artifacts(probabilities, list(reversed(CHANNEL_NAMES)))

        with self.assertRaisesRegex(VectorModelContractError, "channel"):
            load_probability_artifacts(self.output, expected_size=(30, 20))

    def test_rejects_probability_shape_that_does_not_match_analysis_image(self):
        from vector_pdf_model import (
            CHANNEL_NAMES,
            VectorModelContractError,
            load_probability_artifacts,
        )

        probabilities = np.zeros((10, 20, 30), dtype=np.float32)
        self._write_artifacts(probabilities, list(CHANNEL_NAMES))

        with self.assertRaisesRegex(VectorModelContractError, "size|shape"):
            load_probability_artifacts(self.output, expected_size=(31, 20))

    def test_rejects_nonfinite_and_out_of_range_probabilities(self):
        from vector_pdf_model import (
            CHANNEL_NAMES,
            VectorModelContractError,
            load_probability_artifacts,
        )

        for invalid in (float("nan"), -0.01, 1.01):
            with self.subTest(invalid=invalid):
                probabilities = np.zeros((10, 20, 30), dtype=np.float32)
                probabilities[4, 5, 5] = invalid
                self._write_artifacts(probabilities, list(CHANNEL_NAMES))
                with self.assertRaisesRegex(VectorModelContractError, "finite|between"):
                    load_probability_artifacts(self.output, expected_size=(30, 20))


class VectorProbabilityRunnerTests(unittest.TestCase):
    def test_environment_configuration_requires_all_three_paths(self):
        from vector_pdf_model import VectorModelConfig

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(VectorModelConfig.from_environment())
        with mock.patch.dict(os.environ, {"VECTOR_TRAINING_ROOT": "training"}, clear=True):
            self.assertIsNone(VectorModelConfig.from_environment())

    def test_environment_configuration_defaults_to_cpu_and_parses_timeout(self):
        from vector_pdf_model import VectorModelConfig

        values = {
            "VECTOR_TRAINING_ROOT": "training",
            "VECTOR_PYTHON": "python.exe",
            "VECTOR_MODEL_PATH": "best.pt",
            "VECTOR_TIMEOUT_SECONDS": "42.5",
        }
        with mock.patch.dict(os.environ, values, clear=True):
            config = VectorModelConfig.from_environment()

        self.assertIsNotNone(config)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.timeout_seconds, 42.5)

    def test_preserves_symlinked_virtualenv_python_executable(self):
        from vector_pdf_model import VectorModelConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_root = root / "training-root"
            training_root.mkdir()
            system_python = root / "system-python"
            system_python.write_bytes(b"")
            virtualenv_python = root / "venv" / "bin" / "python"
            virtualenv_python.parent.mkdir(parents=True)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"weights")

            original_resolve = Path.resolve

            def resolve_virtualenv_symlink(path, *args, **kwargs):
                if path == virtualenv_python:
                    return system_python
                return original_resolve(path, *args, **kwargs)

            with mock.patch.object(Path, "resolve", resolve_virtualenv_symlink):
                config = VectorModelConfig(
                    training_root=training_root,
                    python_executable=virtualenv_python,
                    checkpoint_path=checkpoint,
                )

        self.assertEqual(config.python_executable, virtualenv_python.absolute())

    def test_runs_external_predictor_with_array_arguments_and_cpu_timeout(self):
        from vector_pdf_model import (
            CHANNEL_NAMES,
            VectorModelConfig,
            run_vector_probabilities,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_root = root / "training-root"
            training_root.mkdir()
            python_executable = root / "python.exe"
            python_executable.write_bytes(b"")
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"weights")
            image = root / "input.png"
            image.write_bytes(b"image")
            artifact_parent = root / "artifacts"
            config = VectorModelConfig(
                training_root=training_root,
                python_executable=python_executable,
                checkpoint_path=checkpoint,
                device="cpu",
                timeout_seconds=12.0,
            )

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                probabilities = np.zeros((10, 20, 30), dtype=np.float32)
                np.savez_compressed(output / "probabilities.npz", probabilities=probabilities)
                (output / "inference.json").write_text(
                    json.dumps({
                        "format": "vector-floorplan-inference/1",
                        "channel_names": list(CHANNEL_NAMES),
                        "image_size": [30, 20],
                    }),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0, stdout="{}", stderr="")

            with mock.patch("vector_pdf_model.subprocess.run", side_effect=fake_run) as invoked:
                result = run_vector_probabilities(
                    image,
                    artifact_parent,
                    config,
                    expected_size=(30, 20),
                )

        command = invoked.call_args.args[0]
        self.assertEqual(
            command[:3],
            [str(python_executable.resolve()), "-m", "training.predict_vector"],
        )
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(invoked.call_args.kwargs["cwd"], training_root.resolve())
        self.assertEqual(invoked.call_args.kwargs["timeout"], 12.0)
        self.assertNotIn("shell", invoked.call_args.kwargs)
        self.assertEqual(result.probabilities.shape, (10, 20, 30))


if __name__ == "__main__":
    unittest.main()
