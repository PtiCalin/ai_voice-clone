"""
AI Voice Clone - A deep learning system for voice cloning and synthesis.

This package provides tools for:
- Recording and processing audio
- Training voice cloning models
- Generating synthetic speech with cloned voices
"""

__version__ = "0.1.0"
__author__ = "AI Voice Clone Team"
__description__ = "Deep learning voice cloning and synthesis system"

from .audio_input import AudioInput
from .audio_upload import AudioAsset, AudioUploadRecorder
from .feature_extraction import FeatureExtractor
from .model import VoiceCloner
from .training import Trainer
from .inference import InferenceEngine
from .config import Config

__all__ = [
    'AudioInput',
    'AudioAsset',
    'AudioUploadRecorder',
    'FeatureExtractor',
    'VoiceCloner',
    'Trainer',
    'InferenceEngine',
    'Config'
]
