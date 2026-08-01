from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    state_rank: int
    state_size: int


class ParameterEncoder(nn.Module):
    def __init__(self, param_dim: int, embedding_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(param_dim, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),
        )

    def forward(self, params: Tensor) -> Tensor:
        return self.network(params)


class StatefulGRU(nn.Module):
    def __init__(
        self,
        param_dim: int = 4,
        hidden_size: int = 48,
        num_layers: int = 1,
        param_embedding: int = 16,
        output_mode: str = "residual",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_mode = output_mode
        self.param_encoder = ParameterEncoder(param_dim, param_embedding)
        self.recurrent = nn.GRU(
            1 + param_embedding,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def initial_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

    def forward(self, audio: Tensor, params: Tensor, state: Tensor):
        embedding = self.param_encoder(params)
        condition = embedding.unsqueeze(1).expand(-1, audio.shape[1], -1)
        recurrent_input = torch.cat((audio, condition), dim=-1)
        features, new_state = self.recurrent(recurrent_input, state)
        residual = torch.tanh(self.output(features))
        output = (
            residual
            if self.output_mode == "direct"
            else audio + self.residual_scale * residual
        )
        return output, new_state


class StatefulLSTM(nn.Module):
    """LSTM with h and c packed into one ONNX state tensor."""

    def __init__(
        self,
        param_dim: int = 4,
        hidden_size: int = 48,
        num_layers: int = 1,
        param_embedding: int = 16,
        output_mode: str = "residual",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_mode = output_mode
        self.param_encoder = ParameterEncoder(param_dim, param_embedding)
        self.recurrent = nn.LSTM(
            1 + param_embedding,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def initial_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(
            self.num_layers * 2, batch_size, self.hidden_size, device=device
        )

    def forward(self, audio: Tensor, params: Tensor, state: Tensor):
        h, c = state[: self.num_layers], state[self.num_layers :]
        embedding = self.param_encoder(params)
        condition = embedding.unsqueeze(1).expand(-1, audio.shape[1], -1)
        recurrent_input = torch.cat((audio, condition), dim=-1)
        features, (new_h, new_c) = self.recurrent(recurrent_input, (h, c))
        residual = torch.tanh(self.output(features))
        output = (
            residual
            if self.output_mode == "direct"
            else audio + self.residual_scale * residual
        )
        return output, torch.cat((new_h, new_c), dim=0)


class StatefulLRU(nn.Module):
    """A lightweight diagonal linear recurrent unit with a nonlinear readout.

    This real-valued variant keeps poles inside the unit circle and provides
    explicit block-to-block state. The recurrence is exported as a fixed-size
    ONNX graph for the configured VST processing block.
    """

    def __init__(
        self,
        param_dim: int = 4,
        hidden_size: int = 48,
        param_embedding: int = 16,
        output_mode: str = "residual",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_mode = output_mode
        self.param_encoder = ParameterEncoder(param_dim, param_embedding)
        self.input_projection = nn.Linear(1 + param_embedding, hidden_size)
        self.logit_decay = nn.Parameter(torch.full((hidden_size,), 2.0))
        self.state_drive = nn.Parameter(torch.ones(hidden_size))
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def initial_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, audio: Tensor, params: Tensor, state: Tensor):
        embedding = self.param_encoder(params)
        condition = embedding.unsqueeze(1).expand(-1, audio.shape[1], -1)
        projected = self.input_projection(torch.cat((audio, condition), dim=-1))
        decay = torch.sigmoid(self.logit_decay)
        current = state[0]
        frames = []
        for sample in projected.unbind(dim=1):
            current = decay * current + (1.0 - decay) * self.state_drive * sample
            frames.append(current)
        features = torch.stack(frames, dim=1)
        residual = torch.tanh(self.readout(features))
        output = (
            residual
            if self.output_mode == "direct"
            else audio + self.residual_scale * residual
        )
        return output, current.unsqueeze(0)


class StatefulWaveNetLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.history = (kernel_size - 1) * dilation
        self.dilated = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(
        self,
        features: Tensor,
        cache: Tensor,
        film_scale: Tensor,
        film_bias: Tensor,
    ):
        # features/cache use [batch, channels, samples].
        convolution_input = torch.cat((cache, features), dim=-1)
        combined = self.dilated(convolution_input)
        tanh_part, sigmoid_part = combined.chunk(2, dim=1)
        gated = torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part)
        gated = gated * (1.0 + film_scale.unsqueeze(-1))
        gated = gated + film_bias.unsqueeze(-1)
        new_features = features + self.residual(gated)
        skip = self.skip(gated)
        new_cache = convolution_input[:, :, -self.history :]
        return new_features, skip, new_cache


class StatefulWaveNet(nn.Module):
    """Causal WaveNet with every layer cache packed into one ONNX state."""

    def __init__(
        self,
        param_dim: int = 4,
        channels: int = 32,
        num_layers: int = 9,
        kernel_size: int = 3,
        param_embedding: int = 32,
        output_mode: str = "residual",
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.output_mode = output_mode
        self.histories = [
            (kernel_size - 1) * (2**layer) for layer in range(num_layers)
        ]
        self.total_state_size = channels * sum(self.histories)
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.param_encoder = ParameterEncoder(param_dim, param_embedding)
        self.film = nn.Linear(param_embedding, num_layers * channels * 2)
        self.layers = nn.ModuleList(
            StatefulWaveNetLayer(channels, kernel_size, 2**layer)
            for layer in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, 1, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def initial_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(1, batch_size, self.total_state_size, device=device)

    def forward(self, audio: Tensor, params: Tensor, state: Tensor):
        features = self.input_projection(audio.transpose(1, 2))
        embedding = self.param_encoder(params)
        film = self.film(embedding).reshape(
            audio.shape[0], self.num_layers, 2, self.channels
        )

        packed_state = state[0]
        offset = 0
        skips = []
        new_caches = []
        for layer_index, layer in enumerate(self.layers):
            cache_elements = self.channels * self.histories[layer_index]
            cache = packed_state[:, offset : offset + cache_elements].reshape(
                audio.shape[0], self.channels, self.histories[layer_index]
            )
            offset += cache_elements
            features, skip, new_cache = layer(
                features,
                cache,
                film[:, layer_index, 0],
                film[:, layer_index, 1],
            )
            skips.append(skip)
            new_caches.append(new_cache.flatten(start_dim=1))

        combined_skip = torch.stack(skips, dim=0).sum(dim=0)
        residual = torch.tanh(self.output(combined_skip).transpose(1, 2))
        output = (
            residual
            if self.output_mode == "direct"
            else audio + self.residual_scale * residual
        )
        new_state = torch.cat(new_caches, dim=1).unsqueeze(0)
        return output, new_state


MODEL_TYPES: Dict[str, type[nn.Module]] = {
    "gru": StatefulGRU,
    "lstm": StatefulLSTM,
    "lru": StatefulLRU,
    "wavenet": StatefulWaveNet,
}


def create_model(config: dict) -> nn.Module:
    model_config = config["model"]
    model_type = model_config["type"].lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model type {model_type!r}; choose from {sorted(MODEL_TYPES)}"
        )
    kwargs = {
        key: value
        for key, value in model_config.items()
        if key not in {"type", "onnx_name"}
    }
    return MODEL_TYPES[model_type](**kwargs)
