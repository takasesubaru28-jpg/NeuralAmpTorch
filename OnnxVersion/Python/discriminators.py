from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm, weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 16, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(16, 64, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 128, (5, 1), 1, (2, 0))),
            ]
        )
        self.output = weight_norm(nn.Conv2d(128, 1, (3, 1), 1, (1, 0)))

    def forward(self, audio: Tensor):
        batch, channels, samples = audio.shape
        remainder = samples % self.period
        if remainder:
            audio = F.pad(audio, (0, self.period - remainder), mode="reflect")
            samples = audio.shape[-1]
        features = []
        value = audio.reshape(batch, channels, samples // self.period, self.period)
        for layer in self.layers:
            value = F.leaky_relu(layer(value), 0.1)
            features.append(value)
        value = self.output(value)
        features.append(value)
        return value.flatten(1), features


class ScaleDiscriminator(nn.Module):
    def __init__(self, spectral: bool = False):
        super().__init__()
        normalize = spectral_norm if spectral else weight_norm
        self.layers = nn.ModuleList(
            [
                normalize(nn.Conv1d(1, 64, 15, 1, 7)),
                normalize(nn.Conv1d(64, 64, 41, 2, 20, groups=4)),
                normalize(nn.Conv1d(64, 128, 41, 2, 20, groups=16)),
                normalize(nn.Conv1d(128, 256, 41, 4, 20, groups=16)),
                normalize(nn.Conv1d(256, 256, 5, 1, 2)),
            ]
        )
        self.output = normalize(nn.Conv1d(256, 1, 3, 1, 1))

    def forward(self, audio: Tensor):
        features = []
        value = audio
        for layer in self.layers:
            value = F.leaky_relu(layer(value), 0.1)
            features.append(value)
        value = self.output(value)
        features.append(value)
        return value.flatten(1), features


class AmpDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = nn.ModuleList(PeriodDiscriminator(p) for p in (2, 3, 5))
        self.scales = nn.ModuleList(
            [ScaleDiscriminator(spectral=True), ScaleDiscriminator(), ScaleDiscriminator()]
        )
        self.pools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, audio: Tensor):
        scores = []
        features = []
        for discriminator in self.periods:
            score, feature = discriminator(audio)
            scores.append(score)
            features.append(feature)
        scaled = audio
        for index, discriminator in enumerate(self.scales):
            if index:
                scaled = self.pools[index - 1](scaled)
            score, feature = discriminator(scaled)
            scores.append(score)
            features.append(feature)
        return scores, features


def discriminator_loss(real_scores, generated_scores):
    return sum(
        (1.0 - real).square().mean() + generated.square().mean()
        for real, generated in zip(real_scores, generated_scores)
    )


def adversarial_loss(generated_scores):
    return sum((1.0 - generated).square().mean() for generated in generated_scores)


def feature_matching_loss(real_features, generated_features):
    return 2.0 * sum(
        F.l1_loss(generated, real.detach())
        for real_group, generated_group in zip(real_features, generated_features)
        for real, generated in zip(real_group, generated_group)
    )
