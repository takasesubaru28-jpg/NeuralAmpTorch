from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    block_size = config["export"]["block_size"]
    audio = torch.zeros(1, block_size, 1)
    params = torch.full((1, config["model"].get("param_dim", 4)), 0.5)
    state = model.initial_state(1)
    output_path = Path(
        args.output
        or Path("exports") / f"{config['model']['onnx_name']}.onnx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "audio": {0: "batch"},
        "params": {0: "batch"},
        "state": {1: "batch"},
        "output": {0: "batch"},
        "new_state": {1: "batch"},
    }
    if config["model"]["type"] != "lru":
        dynamic_axes["audio"][1] = "samples"
        dynamic_axes["output"][1] = "samples"

    torch.onnx.export(
        model,
        (audio, params, state),
        str(output_path),
        input_names=["audio", "params", "state"],
        output_names=["output", "new_state"],
        dynamic_axes=dynamic_axes,
        opset_version=config["export"]["opset"],
        do_constant_folding=True,
        dynamo=False,
    )

    metadata = {
        "model_type": config["model"]["type"],
        "onnx_name": config["model"]["onnx_name"],
        "block_size": block_size,
        "param_order": ["bass", "middle", "treble", "gain"],
        "state_shape": list(state.shape),
        "sample_rate": config["audio"]["sample_rate"],
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
