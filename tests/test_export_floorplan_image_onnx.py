from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "export_floorplan_image_onnx.py"
SPEC = importlib.util.spec_from_file_location("export_floorplan_image_onnx", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExportFloorplanImageOnnxTests(unittest.TestCase):
    def test_parse_args_accepts_explicit_training_checkpoint_and_output_paths(self):
        args = MODULE.parse_args([
            "--training-root", "D:/training",
            "--checkpoint", "D:/training/runs/best.pt",
            "--output", "G:/bim-web/models/image.onnx",
        ])

        self.assertEqual(args.training_root, Path("D:/training"))
        self.assertEqual(args.checkpoint, Path("D:/training/runs/best.pt"))
        self.assertEqual(args.output, Path("G:/bim-web/models/image.onnx"))


if __name__ == "__main__":
    unittest.main()
