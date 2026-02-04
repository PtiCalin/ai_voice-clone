"""
Inference module for AI Voice Clone.
Handles voice generation from trained models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from ai_voice_clone.vocoder import build_vocoder

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles inference for voice cloning."""

    def __init__(self, model: torch.nn.Module, feature_extractor: 'FeatureExtractor',
                 config: 'Config', device: Optional[str] = None):
        """
        Initialize inference engine.

        Args:
            model: Trained voice cloning model
            feature_extractor: Feature extractor
            config: Configuration object
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Inference on device: {self.device}")

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Inference parameters
        self.temperature = config.get('inference.temperature', 0.8)
        self.top_k = config.get('inference.top_k', 40)
        self.max_length = config.get('inference.max_length', 1000)

        # Vocab for text processing (simplified)
        self.vocab = self._create_vocab()

        # Vocoder for mel-to-audio conversion
        self.vocoder = build_vocoder(config, self.device)

    def _create_vocab(self) -> dict:
        """Create a simple vocabulary for text processing."""
        # Basic character-level vocabulary
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"
        vocab = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2}
        for i, char in enumerate(chars, start=3):
            vocab[char] = i
        vocab['<UNK>'] = len(vocab)
        return vocab

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert text to token sequence.

        Args:
            text: Input text

        Returns:
            Token tensor
        """
        tokens = [self.vocab.get('<SOS>', 0)]  # Start token

        for char in text.lower():
            token = self.vocab.get(char, self.vocab.get('<UNK>', 2))
            tokens.append(token)

        tokens.append(self.vocab.get('<EOS>', 1))  # End token

        return torch.tensor(tokens, dtype=torch.long)

    def load_model(self, model_path: str) -> None:
        """
        Load trained model.

        Args:
            model_path: Path to saved model
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        checkpoint = torch.load(model_file, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logger.info(f"Model loaded from: {model_file}")

    def generate_voice(self, text: str, reference_audio_path: str) -> np.ndarray:
        """
        Generate cloned voice from text and reference audio.

        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio file

        Returns:
            Generated audio waveform
        """
        logger.info(f"Generating voice for text: '{text}'")

        # Load and process reference audio
        audio_data, sample_rate = self._load_reference_audio(reference_audio_path)

        # Extract voice features
        voice_features = self.feature_extractor.extract_features(audio_data)

        # Convert text to tokens
        text_tokens = self.text_to_tokens(text)

        # Generate mel spectrograms
        with torch.no_grad():
            # Move to device
            voice_features = voice_features.unsqueeze(0).to(self.device)  # Add batch dimension
            text_tokens = text_tokens.unsqueeze(0).to(self.device)  # Add batch dimension

            # Generate
            generated_mel = self.model(voice_features, text_tokens)

            # Remove batch dimension
            generated_mel = generated_mel.squeeze(0).cpu().numpy()

        # Convert mel spectrograms to audio
        generated_audio = self._mel_to_audio(generated_mel, sample_rate)

        logger.info(f"Voice generation completed. Audio length: {len(generated_audio)} samples")

        return generated_audio

    def _load_reference_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load reference audio for voice cloning.

        Args:
            audio_path: Path to reference audio

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        from ai_voice_clone.audio_input import AudioInput
        audio_input = AudioInput()
        audio_data, sample_rate = audio_input.load_audio(audio_path)
        return audio_data, sample_rate

    def _mel_to_audio(self, mel_spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel_spectrogram: Mel spectrogram (n_mels, time_steps)
            sample_rate: Target sample rate

        Returns:
            Audio waveform
        """
        return self.vocoder.synthesize(mel_spectrogram, sample_rate)

    def generate_with_sampling(self, text: str, reference_audio_path: str,
                             temperature: float = 1.0, top_k: int = 40) -> np.ndarray:
        """
        Generate voice with sampling strategies.

        Args:
            text: Text to synthesize
            reference_audio_path: Reference audio path
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated audio
        """
        # For now, use basic generation
        # In a more advanced implementation, this would include sampling strategies
        return self.generate_voice(text, reference_audio_path)

    def batch_generate(self, text_list: list, reference_audio_path: str) -> list:
        """
        Generate multiple voice samples.

        Args:
            text_list: List of texts to synthesize
            reference_audio_path: Reference audio path

        Returns:
            List of generated audio arrays
        """
        results = []

        for text in text_list:
            try:
                audio = self.generate_voice(text, reference_audio_path)
                results.append(audio)
            except Exception as e:
                logger.error(f"Failed to generate voice for text '{text}': {str(e)}")
                results.append(np.array([]))

        return results

    def get_voice_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extract voice embedding from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Voice embedding tensor
        """
        # Load audio
        audio_data, _ = self._load_reference_audio(audio_path)

        # Extract features
        features = self.feature_extractor.extract_features(audio_data)

        # Get voice embedding
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.device)
            embedding = self.model.encode_voice(features)
            embedding = embedding.squeeze(0).cpu()

        return embedding

    def interpolate_voices(self, audio_path1: str, audio_path2: str,
                          alpha: float = 0.5) -> torch.Tensor:
        """
        Interpolate between two voice embeddings.

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file
            alpha: Interpolation factor (0.0 = voice1, 1.0 = voice2)

        Returns:
            Interpolated voice embedding
        """
        embedding1 = self.get_voice_embedding(audio_path1)
        embedding2 = self.get_voice_embedding(audio_path2)

        # Linear interpolation
        interpolated = (1 - alpha) * embedding1 + alpha * embedding2

        return interpolated
