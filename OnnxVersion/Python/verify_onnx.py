from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    args = parser.parse_args()

    model_path = Path(args.model)
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    state_input = session.get_inputs()[2]
    state_shape = [1 if not isinstance(value, int) else value for value in state_input.shape]
    state = np.zeros(state_shape, dtype=np.float32)
    params = np.full((1, 4), 0.5, dtype=np.float32)
    audio_input = session.get_inputs()[0]
    block_size = (
        audio_input.shape[1] if isinstance(audio_input.shape[1], int) else 64
    )

    outputs = []
    rng = np.random.default_rng(1234)
    for _ in range(args.blocks):
        audio = rng.normal(0.0, 0.05, (1, block_size, 1)).astype(np.float32)
        output, state = session.run(
            ["output", "new_state"],
            {"audio": audio, "params": params, "state": state},
        )
        if not np.isfinite(output).all() or not np.isfinite(state).all():
            raise RuntimeError("Model produced NaN or infinity")
        outputs.append(output)

    combined = np.concatenate(outputs, axis=1)
    print(
        f"OK: {model_path}, blocks={args.blocks}, samples={combined.shape[1]}, "
        f"peak={np.abs(combined).max():.6f}"
    )


if __name__ == "__main__":
    main()
