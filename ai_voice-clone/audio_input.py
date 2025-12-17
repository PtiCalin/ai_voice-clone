"""
Audio input handling for AI Voice Clone.
Handles recording from microphone and loading audio files.
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioInput:
    """Handles audio input operations: recording and file loading."""

    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        """
        Initialize audio input handler.

        Args:
            sample_rate: Audio sampling rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels

    def record_audio(self, duration: int = 5, device: Optional[int] = None) -> np.ndarray:
        """
        Record audio from microphone.

        Args:
            duration: Recording duration in seconds
            device: Audio device index (None for default)

        Returns:
            Recorded audio as numpy array
        """
        logger.info(f"Recording audio for {duration} seconds...")

        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                device=device,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to finish

            # Normalize to [-1, 1] range
            if np.max(np.abs(recording)) > 0:
                recording = recording / np.max(np.abs(recording))

            logger.info(f"Recording completed. Shape: {recording.shape}")
            return recording.flatten() if self.channels == 1 else recording.T[0]

        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            raise

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logger.info(f"Loading audio from: {file_path}")

        try:
            # Load audio file
            audio_data, sample_rate = sf.read(str(file_path), dtype='float32')

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                logger.info(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)

            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            logger.info(f"Audio loaded. Shape: {audio_data.shape}, Sample rate: {self.sample_rate}Hz")
            return audio_data, self.sample_rate

        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise

    def save_audio(self, audio_data: np.ndarray, file_path: str,
                   sample_rate: Optional[int] = None) -> None:
        """
        Save audio data to file.

        Args:
            audio_data: Audio data as numpy array
            file_path: Output file path
            sample_rate: Sample rate (uses instance default if None)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = sample_rate or self.sample_rate

        logger.info(f"Saving audio to: {file_path}")

        try:
            # Ensure audio data is in correct format
            audio_data = np.asarray(audio_data, dtype=np.float32)

            # Save audio file
            sf.write(str(file_path), audio_data, sample_rate)

            logger.info(f"Audio saved successfully. Shape: {audio_data.shape}")

        except Exception as e:
            logger.error(f"Error saving audio file: {str(e)}")
            raise

    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio_data: Input audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio data
        """
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            logger.warning("librosa not available, using basic resampling")
            # Simple resampling (not recommended for production)
            ratio = target_sr / orig_sr
            import scipy.signal
            return scipy.signal.resample(audio_data, int(len(audio_data) * ratio))

    def list_audio_devices(self) -> None:
        """List available audio devices."""
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, "
                  f"outputs: {device['max_output_channels']})")

    def get_audio_info(self, file_path: str) -> dict:
        """
        Get information about an audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio file information
        """
        try:
            info = sf.info(str(file_path))
            return {
                'samplerate': info.samplerate,
                'channels': info.channels,
                'duration': info.duration,
                'frames': info.frames,
                'subtype': info.subtype,
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {}