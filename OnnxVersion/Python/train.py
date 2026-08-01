from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_data import (
    StatefulAmpDataset,
    audio_slice,
    find_wavs,
    parameters_from_filename,
    read_audio_mmap,
)
from discriminators import (
    AmpDiscriminator,
    adversarial_loss,
    discriminator_loss,
    feature_matching_loss,
)
from losses import AmpLoss
from models import create_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


class PreventSystemSleep(AbstractContextManager):
    """Keep Windows awake while a long training process is alive."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __enter__(self):
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )
            print("Windows automatic sleep prevention: enabled", flush=True)
        return self

    def __exit__(self, *args):
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def choose_device(training: dict) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training device: CUDA ({torch.cuda.get_device_name(0)})", flush=True)
        fraction = training.get("max_cuda_memory_fraction")
        if fraction is not None:
            torch.cuda.set_per_process_memory_fraction(float(fraction))
        torch.backends.cudnn.benchmark = True
    else:
        print("Training device: CPU", flush=True)
    return device


def split_target_paths(paths: list[str], validation_count: int):
    if validation_count <= 0 or validation_count >= len(paths):
        raise ValueError("validation_target_count must be between 1 and target_count - 1")
    # Evenly spaced validation settings provide better parameter coverage than a tail split.
    indices = {
        round(index * (len(paths) - 1) / (validation_count - 1))
        if validation_count > 1
        else len(paths) // 2
        for index in range(validation_count)
    }
    validation = [path for index, path in enumerate(paths) if index in indices]
    training = [path for index, path in enumerate(paths) if index not in indices]
    return training, validation


def fixed_validation_batches(
    input_path: str,
    target_paths: list[str],
    sample_rate: int,
    sequence_length: int,
    batch_count: int,
):
    input_audio = read_audio_mmap(input_path, sample_rate)
    targets = [(read_audio_mmap(path, sample_rate), parameters_from_filename(path)) for path in target_paths]
    maximum_start = len(input_audio) - sequence_length
    batches = []
    for index in range(batch_count):
        target, params = targets[index % len(targets)]
        start = round((index + 1) * maximum_start / (batch_count + 1))
        stop = start + sequence_length
        batches.append(
            (
                torch.from_numpy(audio_slice(input_audio, start, stop)).unsqueeze(0),
                torch.from_numpy(audio_slice(target, start, stop)).unsqueeze(0),
                torch.from_numpy(params).unsqueeze(0),
            )
        )
    return batches


@torch.inference_mode()
def evaluate(model, criterion, batches, device):
    model.eval()
    totals: dict[str, float] = {}
    for audio, target, params in batches:
        audio, target, params = audio.to(device), target.to(device), params.to(device)
        state = model.initial_state(audio.shape[0], device=device)
        prediction, _ = model(audio, params, state)
        loss, components = criterion(prediction, target)
        totals["total"] = totals.get("total", 0.0) + loss.item()
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + value.item()
    return {name: value / len(batches) for name, value in totals.items()}


@torch.inference_mode()
def render_evaluation(model, config, target_path: str, device):
    sample_rate = config["audio"]["sample_rate"]
    input_audio = read_audio_mmap(config["paths"]["input_wav"], sample_rate)
    target_audio = read_audio_mmap(target_path, sample_rate)
    seconds = config["monitoring"].get("evaluation_seconds", 10)
    sample_count = min(int(seconds * sample_rate), len(input_audio))
    start = max(0, (len(input_audio) - sample_count) // 2)
    stop = start + sample_count
    audio_np = audio_slice(input_audio, start, stop)
    target_np = audio_slice(target_audio, start, stop)
    params = torch.from_numpy(parameters_from_filename(target_path)).unsqueeze(0).to(device)
    block_size = config["training"]["sequence_length"]
    state = model.initial_state(1, device=device)
    predictions = []
    model.eval()
    for offset in range(0, sample_count, block_size):
        block = torch.from_numpy(audio_np[offset : offset + block_size]).unsqueeze(0).to(device)
        valid = block.shape[1]
        if valid < block_size:
            block = F.pad(block, (0, 0, 0, block_size - valid))
        prediction, state = model(block, params, state)
        predictions.append(prediction[:, :valid].cpu())
    prediction_np = torch.cat(predictions, dim=1).squeeze(0).numpy()
    return audio_np, target_np, prediction_np


def spectrogram_image(audio: Tensor, fft_size: int = 1024):
    window = torch.hann_window(fft_size)
    spectrum = torch.stft(
        audio.flatten().cpu(), fft_size, fft_size // 4,
        window=window, return_complex=True
    ).abs().log1p()
    spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1.0e-8)
    return spectrum.flip(0).unsqueeze(0)


def log_and_save_evaluation(writer, model, config, target_path, epoch, device):
    input_audio, target_audio, prediction = render_evaluation(
        model, config, target_path, device
    )
    sample_rate = config["audio"]["sample_rate"]
    experiment = config["experiment_name"]
    output = Path("evaluations") / experiment / f"epoch_{epoch:06d}"
    output.mkdir(parents=True, exist_ok=True)
    for name, audio in (
        ("input", input_audio), ("target", target_audio), ("prediction", prediction)
    ):
        clipped = np.clip(audio.squeeze(), -1.0, 1.0).astype(np.float32)
        wavfile.write(output / f"{name}.wav", sample_rate, clipped)
        tensor = torch.from_numpy(clipped)
        writer.add_audio(f"evaluation/{name}", tensor.unsqueeze(0), epoch, sample_rate)
        writer.add_image(f"spectrogram/{name}", spectrogram_image(tensor), epoch)
    prediction_tensor = torch.from_numpy(prediction.squeeze())
    input_tensor = torch.from_numpy(input_audio.squeeze())
    target_tensor = torch.from_numpy(target_audio.squeeze())
    writer.add_scalar(
        "evaluation/prediction_input_correlation",
        F.cosine_similarity(prediction_tensor, input_tensor, dim=0).item(), epoch,
    )
    writer.add_scalar(
        "evaluation/target_input_correlation",
        F.cosine_similarity(target_tensor, input_tensor, dim=0).item(), epoch,
    )
    print(f"Evaluation audio: {output}", flush=True)


def checkpoint_payload(
    epoch, global_step, config, model, optimizer, scheduler, scaler,
    best_loss, no_improve, discriminator=None, discriminator_optimizer=None,
    discriminator_scaler=None,
):
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_loss": best_loss,
        "no_improve_epochs": no_improve,
    }
    if discriminator is not None:
        payload.update(
            discriminator_state_dict=discriminator.state_dict(),
            discriminator_optimizer_state_dict=discriminator_optimizer.state_dict(),
            discriminator_scaler_state_dict=discriminator_scaler.state_dict(),
        )
    return payload


def train(config: dict):
    training = config["training"]
    monitoring = config["monitoring"]
    convergence = config["convergence"]
    gan = config.get("gan", {"enabled": False})
    device = choose_device(training)
    use_amp = device.type == "cuda"

    all_targets = find_wavs(config["paths"]["target_dir"])
    train_targets, validation_targets = split_target_paths(
        all_targets, monitoring["validation_target_count"]
    )
    dataset = StatefulAmpDataset(
        input_path=config["paths"]["input_wav"],
        target_paths=train_targets,
        sequence_length=training["sequence_length"],
        sample_rate=config["audio"]["sample_rate"],
        param_dim=config["model"].get("param_dim", 4),
        samples_per_epoch=training["samples_per_epoch"],
    )
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
        pin_memory=use_amp and training.get("pin_memory", False),
        drop_last=True,
    )
    validation_batches = fixed_validation_batches(
        config["paths"]["input_wav"], validation_targets,
        config["audio"]["sample_rate"], training["sequence_length"],
        monitoring["validation_batches"],
    )

    model = create_model(config).to(device)
    criterion = AmpLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=convergence["lr_factor"],
        patience=convergence["lr_patience"],
        threshold=convergence["min_delta_ratio"],
        threshold_mode="rel", min_lr=convergence["min_learning_rate"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    discriminator = discriminator_optimizer = discriminator_scaler = None
    if gan["enabled"]:
        discriminator = AmpDiscriminator().to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=gan["discriminator_learning_rate"],
            betas=(0.5, 0.9)
        )
        discriminator_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    experiment = config["experiment_name"]
    checkpoint_dir = Path("checkpoints") / experiment
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path("runs") / experiment)
    writer.add_text("configuration", f"```json\n{json.dumps(config, indent=2)}\n```", 0)
    writer.add_text("validation_targets", "\n".join(validation_targets), 0)

    epoch = global_step = no_improve = 0
    best_loss = math.inf
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists() and training.get("resume", True):
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", epoch * len(loader))
        best_loss = checkpoint.get("best_loss", checkpoint.get("loss", math.inf))
        no_improve = checkpoint.get("no_improve_epochs", 0)
        if discriminator is not None and "discriminator_state_dict" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer_state_dict"]
            )
            discriminator_scaler.load_state_dict(
                checkpoint.get("discriminator_scaler_state_dict", {})
            )
        print(f"Resumed {experiment} from epoch {epoch}", flush=True)

    interactive = sys.stderr.isatty()
    maximum_epochs = convergence.get("max_epochs", 0)
    print(
        f"Training {experiment}: GAN={gan['enabled']}, "
        f"stop=validation convergence, max_epochs={maximum_epochs or 'unlimited'}",
        flush=True,
    )

    try:
        while not maximum_epochs or epoch < maximum_epochs:
            epoch += 1
            model.train()
            totals: dict[str, float] = {}
            progress = tqdm(loader, desc=f"{experiment} epoch {epoch}", disable=not interactive)
            for step, (audio, target, params) in enumerate(progress, start=1):
                audio = audio.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                params = params.to(device, non_blocking=True)
                state = model.initial_state(audio.shape[0], device=device)

                discriminator_value = None
                if discriminator is not None and step % gan["discriminator_interval"] == 0:
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        with torch.no_grad():
                            generated, _ = model(audio, params, state)
                        real_scores, _ = discriminator(target.transpose(1, 2))
                        generated_scores, _ = discriminator(generated.detach().transpose(1, 2))
                        discriminator_value = discriminator_loss(real_scores, generated_scores)
                    discriminator_scaler.scale(discriminator_value).backward()
                    discriminator_scaler.unscale_(discriminator_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    discriminator_scaler.step(discriminator_optimizer)
                    discriminator_scaler.update()

                optimizer.zero_grad(set_to_none=True)
                if discriminator is not None:
                    discriminator.requires_grad_(False)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    prediction, _ = model(audio, params, state)
                    loss, components = criterion(prediction, target)
                    adversarial_value = feature_value = None
                    if discriminator is not None:
                        real_scores, real_features = discriminator(target.transpose(1, 2))
                        generated_scores, generated_features = discriminator(
                            prediction.transpose(1, 2)
                        )
                        adversarial_value = adversarial_loss(generated_scores)
                        feature_value = feature_matching_loss(
                            real_features, generated_features
                        )
                        loss = (
                            loss
                            + gan["adversarial_weight"] * adversarial_value
                            + gan["feature_matching_weight"] * feature_value
                        )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["clip_grad"])
                scaler.step(optimizer)
                scaler.update()
                if discriminator is not None:
                    discriminator.requires_grad_(True)

                values = {"total": loss.item(), **{k: v.item() for k, v in components.items()}}
                if adversarial_value is not None:
                    values.update(
                        adversarial=adversarial_value.item(),
                        feature_matching=feature_value.item(),
                    )
                if discriminator_value is not None:
                    values["discriminator"] = discriminator_value.item()
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value
                global_step += 1
                if global_step % monitoring["log_every_steps"] == 0:
                    for name, value in values.items():
                        writer.add_scalar(f"train_step/{name}", value, global_step)
                progress.set_postfix(loss=f"{loss.item():.5f}")
                pause_ms = training.get("step_pause_ms", 0)
                if pause_ms:
                    time.sleep(pause_ms / 1000.0)

            train_values = {name: value / len(loader) for name, value in totals.items()}
            validation = evaluate(model, criterion, validation_batches, device)
            scheduler.step(validation["total"])
            learning_rate = optimizer.param_groups[0]["lr"]
            for name, value in train_values.items():
                writer.add_scalar(f"train_epoch/{name}", value, epoch)
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, epoch)
            writer.add_scalar("optimization/learning_rate", learning_rate, epoch)

            threshold = best_loss * (1.0 - convergence["min_delta_ratio"])
            improved = validation["total"] < threshold
            if improved:
                best_loss = validation["total"]
                no_improve = 0
            else:
                no_improve += 1

            payload = checkpoint_payload(
                epoch, global_step, config, model, optimizer, scheduler, scaler,
                best_loss, no_improve, discriminator, discriminator_optimizer,
                discriminator_scaler,
            )
            torch.save(payload, latest_path)
            if improved:
                torch.save(payload, checkpoint_dir / "best.pt")
            if epoch == 1 or epoch % monitoring["evaluation_interval_epochs"] == 0:
                log_and_save_evaluation(
                    writer, model, config, validation_targets[0], epoch, device
                )
            writer.flush()
            print(
                f"{experiment} epoch={epoch} train={train_values['total']:.6f} "
                f"validation={validation['total']:.6f} best={best_loss:.6f} "
                f"lr={learning_rate:.2e} plateau={no_improve}/"
                f"{convergence['patience']}",
                flush=True,
            )

            converged = (
                epoch >= convergence["minimum_epochs"]
                and no_improve >= convergence["patience"]
                and learning_rate <= convergence["min_learning_rate"] * 1.01
            )
            if converged:
                print(f"Converged after {epoch} epochs: {experiment}", flush=True)
                break
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        config["experiment_name"] += "_smoke"
        config["training"]["samples_per_epoch"] = config["training"]["batch_size"]
        config["training"]["resume"] = False
        config["convergence"]["minimum_epochs"] = 1
        config["convergence"]["max_epochs"] = 1
        config["convergence"]["patience"] = 1
        config["monitoring"]["validation_target_count"] = 2
        config["monitoring"]["validation_batches"] = 1
        config["monitoring"]["evaluation_interval_epochs"] = 1
        config["monitoring"]["evaluation_seconds"] = 0.1
    with PreventSystemSleep():
        train(config)
    # This marker is consumed by the background runner.  It is deliberately
    # printed only after SummaryWriter has flushed/closed and the sleep guard
    # has been released, so it means that all durable training work completed.
    print("TRAINING_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
