from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset


def read_audio(
    path: str | Path, expected_sample_rate: int, mmap: bool = False
) -> np.ndarray:
    sample_rate, data = wavfile.read(path, mmap=mmap)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path}: expected {expected_sample_rate} Hz, got {sample_rate} Hz"
        )
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: audio contains NaN or infinity")
    return np.ascontiguousarray(data.reshape(-1, 1))


def read_audio_mmap(path: str | Path, expected_sample_rate: int) -> np.ndarray:
    """Open a WAV without loading the complete recording into RAM."""
    sample_rate, data = wavfile.read(path, mmap=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path}: expected {expected_sample_rate} Hz, got {sample_rate} Hz"
        )
    return data


def audio_slice(data: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Read and normalize only the requested section of a memory-mapped WAV."""
    section = data[start:stop]
    if section.ndim == 2:
        section = section.astype(np.float32).mean(axis=1)
    elif np.issubdtype(section.dtype, np.integer):
        section = section.astype(np.float32)
    else:
        section = np.asarray(section, dtype=np.float32)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        section /= float(max(abs(info.min), info.max))
    if not np.isfinite(section).all():
        raise ValueError("Audio slice contains NaN or infinity")
    return np.ascontiguousarray(section.reshape(-1, 1))


def parameters_from_filename(path: str | Path, param_dim: int = 4) -> np.ndarray:
    encoded = Path(path).stem.rsplit("_", 1)[-1]
    values = np.asarray([float(value) for value in encoded.split("&")])
    if len(values) != param_dim:
        raise ValueError(f"{path}: expected {param_dim} parameters, got {len(values)}")
    return (values / 10.0).astype(np.float32)


class StatefulAmpDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        target_paths: Sequence[str],
        sequence_length: int,
        sample_rate: int,
        param_dim: int = 4,
        samples_per_epoch: int = 10000,
    ):
        self.input_audio = read_audio_mmap(input_path, sample_rate)
        self.targets = []
        for target_path in target_paths:
            target = read_audio_mmap(target_path, sample_rate)
            if len(target) != len(self.input_audio):
                raise ValueError(
                    f"{target_path}: target length {len(target)} does not match "
                    f"input length {len(self.input_audio)}"
                )
            self.targets.append(
                (target, parameters_from_filename(target_path, param_dim))
            )
        if not self.targets:
            raise ValueError("No target WAV files were found")
        self.sequence_length = sequence_length
        self.samples_per_epoch = samples_per_epoch
        if len(self.input_audio) < sequence_length:
            raise ValueError("Input audio is shorter than sequence_length")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _):
        target, params = random.choice(self.targets)
        start = random.randint(0, len(self.input_audio) - self.sequence_length)
        stop = start + self.sequence_length
        return (
            torch.from_numpy(audio_slice(self.input_audio, start, stop)),
            torch.from_numpy(audio_slice(target, start, stop)),
            torch.from_numpy(params),
        )


def find_wavs(directory: str) -> list[str]:
    return sorted(
        str(path)
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
