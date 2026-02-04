"""
Vocoder integration for converting mel spectrograms to audio waveforms.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ensure_mel_channels_first(mel: np.ndarray, n_mels: int) -> np.ndarray:
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel spectrogram, got shape {mel.shape}")
    if mel.shape[0] == n_mels:
        return mel
    if mel.shape[1] == n_mels:
        return mel.T
    raise ValueError(
        f"Mel spectrogram shape {mel.shape} does not match n_mels={n_mels}."
    )


class BaseVocoder:
    """Base interface for mel-to-waveform vocoders."""

    def __init__(self, n_mels: int, device: torch.device):
        self.n_mels = n_mels
        self.device = device

    def synthesize(self, mel_spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError


class GriffinLimVocoder(BaseVocoder):
    """Baseline Griffin-Lim reconstruction for mel spectrograms."""

    def __init__(self, n_mels: int, device: torch.device, n_fft: int, hop_length: int, win_length: int):
        super().__init__(n_mels=n_mels, device=device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self._librosa = None

    def _load_librosa(self):
        if self._librosa is None and _module_available("librosa"):
            self._librosa = importlib.import_module("librosa")
        return self._librosa

    def synthesize(self, mel_spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
        mel_spectrogram = _ensure_mel_channels_first(mel_spectrogram, self.n_mels)
        librosa = self._load_librosa()
        if librosa is None:
            logger.warning("librosa not available, returning noise placeholder audio")
            return np.random.randn(mel_spectrogram.shape[1] * self.hop_length).astype(np.float32)

        mel_spectrogram = np.exp(mel_spectrogram)
        linear_spec = librosa.feature.inverse.mel_to_stft(
            mel_spectrogram,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        audio = librosa.griffinlim(
            linear_spec,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio.astype(np.float32)


@dataclass
class TorchScriptConfig:
    model_path: Path
    sigma: Optional[float] = None


class TorchScriptVocoder(BaseVocoder):
    """Generic TorchScript vocoder wrapper (HiFi-GAN/WaveGlow)."""

    def __init__(self, n_mels: int, device: torch.device, config: TorchScriptConfig):
        super().__init__(n_mels=n_mels, device=device)
        self.config = config
        self.model = torch.jit.load(str(config.model_path), map_location=device)
        self.model.eval()

    def synthesize(self, mel_spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
        mel_spectrogram = _ensure_mel_channels_first(mel_spectrogram, self.n_mels)
        mel_tensor = torch.from_numpy(mel_spectrogram).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "infer"):
                if self.config.sigma is None:
                    audio_tensor = self.model.infer(mel_tensor)
                else:
                    audio_tensor = self.model.infer(mel_tensor, sigma=self.config.sigma)
            else:
                audio_tensor = self.model(mel_tensor)

        if isinstance(audio_tensor, (tuple, list)):
            audio_tensor = audio_tensor[0]

        audio = audio_tensor.squeeze().cpu().numpy()
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio.astype(np.float32)


def build_vocoder(config: "Config", device: torch.device) -> BaseVocoder:
    backend = config.get("vocoder.backend", "griffinlim").lower()
    n_mels = config.get("features.n_mels", 80)

    if backend == "griffinlim":
        return GriffinLimVocoder(
            n_mels=n_mels,
            device=device,
            n_fft=config.get("features.n_fft", 1024),
            hop_length=config.get("features.hop_length", 256),
            win_length=config.get("features.win_length", 1024),
        )

    if backend == "hifigan":
        model_path = config.get("vocoder.hifigan.model_path")
        if not model_path:
            raise ValueError("HiFi-GAN vocoder requested but vocoder.hifigan.model_path is not set.")
        return TorchScriptVocoder(
            n_mels=n_mels,
            device=device,
            config=TorchScriptConfig(model_path=Path(model_path)),
        )

    if backend == "waveglow":
        model_path = config.get("vocoder.waveglow.model_path")
        if not model_path:
            raise ValueError("WaveGlow vocoder requested but vocoder.waveglow.model_path is not set.")
        sigma = config.get("vocoder.waveglow.sigma", 0.6)
        return TorchScriptVocoder(
            n_mels=n_mels,
            device=device,
            config=TorchScriptConfig(model_path=Path(model_path), sigma=sigma),
        )

    raise ValueError(f"Unknown vocoder backend: {backend}")
