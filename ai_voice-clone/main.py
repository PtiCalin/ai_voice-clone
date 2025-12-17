#!/usr/bin/env python3
"""
AI Voice Clone - Main Entry Point

This program provides voice cloning capabilities using deep learning.
It can record audio, process it, and generate cloned voice outputs.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ai_voice_clone.audio_input import AudioInput
from ai_voice_clone.feature_extraction import FeatureExtractor
from ai_voice_clone.model import VoiceCloner
from ai_voice_clone.training import Trainer
from ai_voice_clone.inference import InferenceEngine
from ai_voice_clone.config import Config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('voice_clone.log')
        ]
    )


def record_voice_sample(output_path: str, duration: int = 5):
    """Record a voice sample from microphone."""
    print(f"Recording voice sample for {duration} seconds...")
    print("Speak clearly into the microphone...")

    audio_input = AudioInput()
    audio_data = audio_input.record_audio(duration=duration)
    audio_input.save_audio(audio_data, output_path)

    print(f"Voice sample saved to: {output_path}")
    return output_path


def clone_voice(reference_audio: str, text: str, output_path: str):
    """Clone voice from reference audio and generate new speech."""
    print("Loading voice cloning model...")

    # Load configuration
    config = Config()
    config.load()

    # Initialize components
    feature_extractor = FeatureExtractor(config)
    model = VoiceCloner(config)
    inference_engine = InferenceEngine(model, feature_extractor, config)

    # Load or train model
    model_path = Path("models/voice_clone_model.pt")
    if model_path.exists():
        print("Loading pre-trained model...")
        inference_engine.load_model(str(model_path))
    else:
        print("No pre-trained model found. Training new model...")
        train_model(reference_audio)

    # Generate cloned voice
    print(f"Generating voice for text: '{text}'")
    output_audio = inference_engine.generate_voice(text, reference_audio)

    # Save output
    audio_input = AudioInput()
    audio_input.save_audio(output_audio, output_path)

    print(f"Cloned voice saved to: {output_path}")


def train_model(reference_audio: str):
    """Train the voice cloning model."""
    print("Training voice cloning model...")

    config = Config()
    config.load()

    # Initialize components
    feature_extractor = FeatureExtractor(config)
    model = VoiceCloner(config)
    trainer = Trainer(model, feature_extractor, config)

    # Train model
    trainer.train(reference_audio)

    # Save model
    model_path = "models/voice_clone_model.pt"
    Path(model_path).parent.mkdir(exist_ok=True)
    trainer.save_model(model_path)

    print(f"Model trained and saved to: {model_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Voice Clone")
    parser.add_argument("--mode", choices=["record", "clone", "train"],
                       default="record", help="Operation mode")
    parser.add_argument("--input", "-i", help="Input audio file path")
    parser.add_argument("--output", "-o", default="output.wav",
                       help="Output audio file path")
    parser.add_argument("--text", "-t", default="Hello, this is my cloned voice!",
                       help="Text to generate for voice cloning")
    parser.add_argument("--duration", "-d", type=int, default=5,
                       help="Recording duration in seconds")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        if args.mode == "record":
            # Record voice sample
            output_path = args.output or "voice_sample.wav"
            record_voice_sample(output_path, args.duration)

        elif args.mode == "clone":
            # Clone voice
            if not args.input:
                logger.error("Input audio file required for cloning mode")
                sys.exit(1)

            clone_voice(args.input, args.text, args.output)

        elif args.mode == "train":
            # Train model
            if not args.input:
                logger.error("Input audio file required for training mode")
                sys.exit(1)

            train_model(args.input)

        logger.info("Operation completed successfully!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()