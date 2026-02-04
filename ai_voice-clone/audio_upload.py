"""
Audio upload and recording module for AI Voice Clone.

Provides utilities to ingest uploaded audio files and record microphone input
into a consistent format for downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import io
import logging

import numpy as np
import soundfile as sf

from .audio_input import AudioInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioAsset:
    """Metadata for a stored audio asset."""

    path: Path
    sample_rate: int
    channels: int
    duration: float
    frames: int


class AudioUploadRecorder:
    """Handle audio file uploads and microphone recordings."""

    def __init__(
        self,
        storage_dir: str | Path = "audio_assets",
        sample_rate: int = 22050,
        channels: int = 1,
        allowed_extensions: Optional[Iterable[str]] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.audio_input = AudioInput(sample_rate=sample_rate, channels=channels)
        self.allowed_extensions = {
            ext.lower() for ext in (allowed_extensions or {".wav", ".flac", ".ogg", ".aiff", ".aif"})
        }

    def save_upload(self, file_bytes: bytes, filename: str) -> AudioAsset:
        """
        Save uploaded audio bytes to storage after validation and normalization.

        Args:
            file_bytes: Raw uploaded file bytes.
            filename: Original filename from upload.

        Returns:
            AudioAsset with metadata for the saved file.
        """
        if not filename:
            raise ValueError("Filename must be provided for uploaded audio.")

        extension = Path(filename).suffix.lower()
        if extension not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported audio extension '{extension}'. "
                f"Allowed: {', '.join(sorted(self.allowed_extensions))}"
            )

        safe_name = Path(filename).name
        destination = self.storage_dir / safe_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        audio_data, sample_rate = self._load_from_bytes(file_bytes)
        audio_data = self._prepare_audio(audio_data, sample_rate)

        self.audio_input.save_audio(audio_data, str(destination), sample_rate=self.audio_input.sample_rate)
        return self._build_asset(destination)

    def record_to_file(
        self,
        duration: int = 5,
        filename: Optional[str] = None,
        device: Optional[int] = None,
    ) -> AudioAsset:
        """
        Record microphone input and save it to storage.

        Args:
            duration: Recording duration in seconds.
            filename: Optional filename override.
            device: Optional audio device index.

        Returns:
            AudioAsset with metadata for the saved recording.
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            safe_name = Path(filename).name
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_name = f"recording_{timestamp}.wav"

        destination = self.storage_dir / safe_name
        audio_data = self.audio_input.record_audio(duration=duration, device=device)
        self.audio_input.save_audio(audio_data, str(destination), sample_rate=self.audio_input.sample_rate)
        return self._build_asset(destination)

    def get_asset_info(self, file_path: str | Path) -> AudioAsset:
        """Return metadata for a stored audio asset."""
        return self._build_asset(Path(file_path))

    def _load_from_bytes(self, file_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode audio bytes into a numpy array and sample rate."""
        try:
            with io.BytesIO(file_bytes) as buffer:
                audio_data, sample_rate = sf.read(buffer, dtype="float32")
            return audio_data, sample_rate
        except Exception as exc:
            logger.error("Failed to decode uploaded audio: %s", exc)
            raise

    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize, mix down, and resample audio for consistent storage."""
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != self.audio_input.sample_rate:
            logger.info(
                "Resampling uploaded audio from %sHz to %sHz",
                sample_rate,
                self.audio_input.sample_rate,
            )
            audio_data = self.audio_input._resample_audio(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.audio_input.sample_rate,
            )

        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak

        return audio_data

    def _build_asset(self, path: Path) -> AudioAsset:
        """Build AudioAsset metadata from a file path."""
        info = sf.info(str(path))
        return AudioAsset(
            path=path,
            sample_rate=info.samplerate,
            channels=info.channels,
            duration=info.duration,
            frames=info.frames,
        )
