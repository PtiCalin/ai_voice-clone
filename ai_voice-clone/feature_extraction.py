"""
Feature extraction for AI Voice Clone.
Extracts audio features like MFCC, spectrograms for voice cloning.
"""

import numpy as np
import torch
import torchaudio
import librosa
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts audio features for voice cloning."""

    def __init__(self, config: 'Config'):
        """
        Initialize feature extractor.

        Args:
            config: Configuration object
        """
        self.config = config

        # Audio parameters
        self.sample_rate = config.get('audio.sample_rate', 22050)

        # Feature parameters
        self.n_mels = config.get('features.n_mels', 80)
        self.n_fft = config.get('features.n_fft', 1024)
        self.hop_length = config.get('features.hop_length', 256)
        self.win_length = config.get('features.win_length', 1024)
        self.fmin = config.get('features.fmin', 0)
        self.fmax = config.get('features.fmax', 8000)

        # Initialize transforms
        self._init_transforms()

    def _init_transforms(self):
        """Initialize audio transforms."""
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
        )

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'n_mels': self.n_mels,
                'f_min': self.fmin,
                'f_max': self.fmax,
            }
        )

    def extract_features(self, audio: np.ndarray,
                        feature_type: str = 'mel') -> torch.Tensor:
        """
        Extract audio features.

        Args:
            audio: Audio waveform as numpy array
            feature_type: Type of features ('mel', 'mfcc', 'spectrogram')

        Returns:
            Extracted features as torch tensor
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure correct shape (batch_size, channels, length)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        logger.debug(f"Extracting {feature_type} features. Audio shape: {audio.shape}")

        try:
            if feature_type == 'mel':
                features = self.mel_transform(audio)
                # Convert to log scale
                features = torch.log(features + 1e-9)

            elif feature_type == 'mfcc':
                features = self.mfcc_transform(audio)

            elif feature_type == 'spectrogram':
                # Compute spectrogram
                spec_transform = torchaudio.transforms.Spectrogram(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                )
                features = spec_transform(audio)
                features = torch.log(features + 1e-9)

            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            # Remove batch and channel dimensions for single audio
            if features.dim() == 3 and features.shape[0] == 1:
                features = features.squeeze(0)

            logger.debug(f"Features extracted. Shape: {features.shape}")
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def extract_multiple_features(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Extract multiple types of features.

        Args:
            audio: Audio waveform

        Returns:
            Dictionary of feature types and their tensors
        """
        features = {}

        for feature_type in ['mel', 'mfcc', 'spectrogram']:
            try:
                features[feature_type] = self.extract_features(audio, feature_type)
            except Exception as e:
                logger.warning(f"Failed to extract {feature_type} features: {str(e)}")

        return features

    def normalize_features(self, features: torch.Tensor,
                          mean: Optional[torch.Tensor] = None,
                          std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize features using mean and standard deviation.

        Args:
            features: Input features
            mean: Pre-computed mean (computed if None)
            std: Pre-computed std (computed if None)

        Returns:
            Tuple of (normalized_features, mean, std)
        """
        if mean is None or std is None:
            # Compute statistics
            mean = features.mean(dim=-1, keepdim=True)
            std = features.std(dim=-1, keepdim=True) + 1e-9  # Avoid division by zero

        # Normalize
        normalized = (features - mean) / std

        return normalized, mean, std

    def denormalize_features(self, features: torch.Tensor,
                           mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Denormalize features.

        Args:
            features: Normalized features
            mean: Mean used for normalization
            std: Std used for normalization

        Returns:
            Denormalized features
        """
        return features * std + mean

    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch (fundamental frequency) from audio.

        Args:
            audio: Audio waveform

        Returns:
            Pitch values over time
        """
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax
            )

            # Get the pitch with highest magnitude at each frame
            pitch = np.zeros(pitches.shape[1])
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch[i] = pitches[index, i]

            return pitch

        except Exception as e:
            logger.error(f"Error extracting pitch: {str(e)}")
            return np.array([])

    def extract_voice_quality_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract voice quality features.

        Args:
            audio: Audio waveform

        Returns:
            Dictionary of voice quality metrics
        """
        features = {}

        try:
            # Zero-crossing rate
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))

            # RMS energy
            features['rms'] = np.mean(librosa.feature.rms(y=audio))

            # Spectral centroid
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate))

            # Spectral bandwidth
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate))

            # Spectral rolloff
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate))

        except Exception as e:
            logger.error(f"Error extracting voice quality features: {str(e)}")

        return features

    def get_feature_dimensions(self, feature_type: str) -> Tuple[int, int]:
        """
        Get expected dimensions for a feature type.

        Args:
            feature_type: Type of features

        Returns:
            Tuple of (num_features, time_steps)
        """
        if feature_type == 'mel':
            return self.n_mels, None  # Variable time steps
        elif feature_type == 'mfcc':
            return 13, None  # 13 MFCC coefficients
        elif feature_type == 'spectrogram':
            return self.n_fft // 2 + 1, None
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")