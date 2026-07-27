import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("training tests require .venv-train") from exc
from torch import nn

from training.evaluate import evaluate_checkpoint
from training.run_experiment import require_class_mask_equivalent, run_experiment
from training.train_floorplan import (
    TrainingConfig,
    build_floorplan_model,
    load_model_strict,
    train,
)


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=1),
        )

    def forward(self, inputs):
        return self.network(inputs)


class TrainingExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.export = self.root / "export"
        (self.export / "images").mkdir(parents=True)
        (self.export / "masks").mkdir()
        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (80, 80), (430, 430), (0, 0, 0), 5)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[78:83, 80:431] = 1
        cv2.imwrite(str(self.export / "images" / "page-1.png"), image)
        np.save(self.export / "masks" / "page-1.npy", mask, allow_pickle=False)
        (self.export / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_type": "single_page_overfit",
                    "class_map": {
                        "0": "background",
                        "1": "wall",
                        "2": "window",
                        "3": "door",
                    },
                    "samples": [
                        {
                            "sample_id": "page-1",
                            "page_number": 1,
                            "status": "confirmed",
                            "image": "images/page-1.png",
                            "mask": "masks/page-1.npy",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.export / "dataset_report.json").write_text(
            json.dumps(
                {
                    "class_pixels": {
                        "0": int((mask == 0).sum()),
                        "1": int((mask == 1).sum()),
                        "2": 0,
                        "3": 0,
                    }
                }
            ),
            encoding="utf-8",
        )
        self.initial = self.root / "initial.pt"
        torch.save({"model_state_dict": TinySegmentationModel().state_dict()}, self.initial)

    def tearDown(self):
        self.temporary.cleanup()

    def test_two_epoch_cpu_training_writes_reproducible_artifacts(self):
        output = self.root / "experiment"
        config = TrainingConfig(
            dataset_path=self.export,
            checkpoint_path=self.initial,
            output_dir=output,
            epochs=2,
            learning_rate=1e-3,
            patience=2,
            device="cpu",
            augmentation_count=1,
        )

        result = train(config, model_factory=TinySegmentationModel)

        self.assertTrue(result.last_checkpoint.is_file())
        self.assertTrue(result.best_checkpoint.is_file())
        self.assertTrue((output / "training_log.csv").is_file())
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["config"]["seed"], 20260727)
        self.assertEqual(metrics["epochs_completed"], 2)
        self.assertEqual(metrics["selection_basis"], "smoothed_training_loss")
        self.assertFalse(metrics["has_independent_validation"])

        evaluation = evaluate_checkpoint(
            result.best_checkpoint,
            self.export,
            output / "evaluation",
            model_factory=TinySegmentationModel,
            device="cpu",
        )
        self.assertTrue((output / "evaluation" / "metrics.json").is_file())
        self.assertTrue((output / "evaluation" / "overlay.png").is_file())
        self.assertEqual(len(evaluation["confusion_matrix"]), 4)

    def test_checkpoint_loading_is_strict_for_injected_model(self):
        incompatible = self.root / "incompatible.pt"
        torch.save({"model_state_dict": {"wrong.weight": torch.ones(1)}}, incompatible)

        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            load_model_strict(incompatible, TinySegmentationModel, device="cpu")

    def test_training_refuses_to_reuse_a_nonempty_output_directory(self):
        output = self.root / "existing-output"
        output.mkdir()
        sentinel = output / "best.pt"
        sentinel.write_bytes(b"do-not-overwrite")
        config = TrainingConfig(
            dataset_path=self.export,
            checkpoint_path=self.initial,
            output_dir=output,
            epochs=1,
            device="cpu",
            augmentation_count=1,
        )

        with self.assertRaisesRegex(ValueError, "empty"):
            train(config, model_factory=TinySegmentationModel)

        self.assertEqual(sentinel.read_bytes(), b"do-not-overwrite")

    def test_real_factory_builds_the_four_class_resnet34_unet(self):
        model = build_floorplan_model()

        self.assertEqual(type(model).__name__, "Unet")
        self.assertEqual(type(model.encoder).__name__, "ResNetEncoder")
        self.assertEqual(model.segmentation_head[0].out_channels, 4)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 24436804)

    def test_experiment_runner_exports_and_compares_old_and_new_models(self):
        output = self.root / "orchestrated"
        config = TrainingConfig(
            dataset_path=self.export,
            checkpoint_path=self.initial,
            output_dir=output,
            epochs=1,
            learning_rate=1e-3,
            patience=1,
            device="cpu",
            augmentation_count=1,
        )

        def fake_export(checkpoint, destination):
            Path(destination).write_bytes(b"test-onnx")
            return {
                "checker": {"status": "passed", "returncode": 0},
                "metrics": {"argmax_pixel_agreement": 1.0, "max_abs_error": 0.0},
            }

        result = run_experiment(
            config,
            model_factory=TinySegmentationModel,
            export_function=fake_export,
            sample_parity_function=lambda *args, **kwargs: {
                "argmax_pixel_agreement": 1.0,
                "status": "passed",
            },
        )

        self.assertTrue(result.onnx_path.is_file())
        self.assertTrue((output / "export_report.json").is_file())
        self.assertTrue((output / "comparison.png").is_file())
        self.assertEqual(result.sample_parity["status"], "passed")

    def test_confirmed_sample_gate_uses_class_mask_agreement(self):
        require_class_mask_equivalent(
            {"max_abs_error": 0.001, "argmax_pixel_agreement": 1.0}
        )
        with self.assertRaisesRegex(RuntimeError, "class-mask"):
            require_class_mask_equivalent(
                {"max_abs_error": 0.0, "argmax_pixel_agreement": 0.99}
            )


if __name__ == "__main__":
    unittest.main()
