import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch import Tensor, nn


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(256, 512, 1024), eps=1.0e-7):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.eps = eps

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction = prediction.squeeze(-1)
        target = target.squeeze(-1)
        total = prediction.new_zeros(())
        for fft_size in self.fft_sizes:
            hop = fft_size // 4
            window = torch.hann_window(
                fft_size, device=prediction.device, dtype=prediction.dtype
            )
            pred_stft = torch.stft(
                prediction,
                fft_size,
                hop_length=hop,
                window=window,
                return_complex=True,
                center=False,
            ).abs()
            target_stft = torch.stft(
                target,
                fft_size,
                hop_length=hop,
                window=window,
                return_complex=True,
                center=False,
            ).abs()
            spectral_convergence = torch.linalg.vector_norm(
                target_stft - pred_stft
            ) / (torch.linalg.vector_norm(target_stft) + self.eps)
            log_magnitude = F.l1_loss(
                torch.log(pred_stft + self.eps),
                torch.log(target_stft + self.eps),
            )
            total = total + spectral_convergence + log_magnitude
        return total / len(self.fft_sizes)


def error_to_signal_ratio(prediction: Tensor, target: Tensor) -> Tensor:
    error = (target - prediction).square().mean(dim=1)
    energy = target.square().mean(dim=1).clamp_min(1.0e-4)
    return (error / energy).mean()


class AmpLoss(nn.Module):
    def __init__(self, time_weight=1.0, esr_weight=0.1, stft_weight=1.0):
        super().__init__()
        self.time_weight = time_weight
        self.esr_weight = esr_weight
        self.stft_weight = stft_weight
        self.stft = MultiResolutionSTFTLoss()

    def forward(self, prediction: Tensor, target: Tensor):
        time_loss = F.l1_loss(prediction, target)
        esr_loss = error_to_signal_ratio(prediction, target)
        stft_loss = self.stft(prediction, target)
        total = (
            self.time_weight * time_loss
            + self.esr_weight * esr_loss
            + self.stft_weight * stft_loss
        )
        return total, {
            "time": time_loss.detach(),
            "esr": esr_loss.detach(),
            "stft": stft_loss.detach(),
        }
