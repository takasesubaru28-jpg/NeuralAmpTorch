from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_data import find_wavs, parameters_from_filename, read_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    input_audio = read_audio(args.input, args.sample_rate)
    failures = 0
    fingerprints: dict[tuple[int, float, float], str] = {}
    for path in find_wavs(args.targets):
        try:
            target = read_audio(path, args.sample_rate)
            parameters_from_filename(path)
            if len(target) != len(input_audio):
                raise ValueError(
                    f"length {len(target)} does not match input {len(input_audio)}"
                )
            rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
            peak = float(np.max(np.abs(target)))
            if rms < 1.0e-6:
                raise ValueError(f"target is effectively silent (RMS={rms:.3g})")
            fingerprint = (len(target), round(rms, 10), round(peak, 10))
            if fingerprint in fingerprints:
                print(
                    f"WARNING {path}: same length/RMS/peak as "
                    f"{fingerprints[fingerprint]}"
                )
            else:
                fingerprints[fingerprint] = path
            print(f"OK {path}: RMS={rms:.6f}, peak={peak:.6f}")
        except Exception as exception:
            failures += 1
            print(f"ERROR {path}: {exception}")
    if failures:
        raise SystemExit(f"{failures} invalid target file(s)")


if __name__ == "__main__":
    main()
