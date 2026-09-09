"""Export the trained raster-floorplan checkpoint for ONNX Runtime deployment."""

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def export_checkpoint(training_root, checkpoint, output):
    import onnx
    import torch

    training_root = Path(training_root).resolve()
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    sys.path.insert(0, str(training_root))
    try:
        from model import DualHeadUNet
    finally:
        sys.path.pop(0)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("config", {}).get("image_size") != 256:
        raise ValueError("only 256px image-model checkpoints are supported")
    model = DualHeadUNet().eval()
    model.load_state_dict(state["model_state_dict"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.onnx.export(
        model, torch.zeros((1, 3, 256, 256), dtype=torch.float32), temporary,
        input_names=["image"], output_names=["element_logits", "footprint_logits"],
        opset_version=17, dynamo=False,
    )
    try:
        onnx.checker.check_model(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    export_checkpoint(args.training_root, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
