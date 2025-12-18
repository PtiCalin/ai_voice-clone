# AI Voice Clone

## Overview

AI Voice Clone is an open-source project aimed at developing advanced voice synthesis and cloning technologies using artificial intelligence. The project focuses on creating realistic voice replicas from audio samples, enabling applications in entertainment, accessibility, education, and more.

## Technologies

- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch
- **Audio Processing**: Torchaudio, Librosa
- **Voice Synthesis**: Tacotron2, WaveGlow, or similar TTS models
- **Machine Learning**: Scikit-learn for preprocessing
- **Web Framework** (future): FastAPI or Flask for API deployment

## Project Scope

### Current Features

- Basic voice recording and preprocessing
- Audio feature extraction (MFCC, spectrograms)
- Model training pipeline setup

### Development Roadmap

- **Phase 1**: Implement basic voice cloning with pre-trained models
- **Phase 2**: Custom model training from user audio samples
- **Phase 3**: Real-time voice conversion
- **Phase 4**: Multi-speaker voice cloning
- **Phase 5**: Web interface and API deployment

### Key Components

- Data collection and preprocessing pipeline
- Neural network architectures for voice synthesis
- Training scripts and utilities
- Evaluation metrics and testing framework
- Deployment and inference scripts

## Installation

```bash
git clone https://github.com/PtiCalin/ai_voice-clone.git
cd ai_voice-clone
pip install -r requirements.txt
```

## Usage

### Graphical User Interface (Recommended)

The easiest way to use AI Voice Clone is through the graphical interface:

```bash
# Launch the GUI
python ai_voice-clone/main.py --mode gui

# Or use the launcher script
python launch_gui.py
```

The GUI provides four main tabs:
- **🎤 Record Voice**: Record audio from your microphone
- **🧠 Train Model**: Train the voice cloning model with your recordings
- **🔊 Clone Voice**: Generate cloned voice from text and reference audio
- **⚙️ Settings**: Configure audio and model parameters

### Command Line Interface

```bash
# Record a voice sample
python ai_voice-clone/main.py --mode record --duration 5 --output my_voice.wav

# Train a model with your voice
python ai_voice-clone/main.py --mode train --input my_voice.wav

# Generate cloned voice
python ai_voice-clone/main.py --mode clone --input my_voice.wav --text "Hello, this is my cloned voice!" --output cloned_voice.wav
```

### Python API

```python
from ai_voice_clone import AudioInput, VoiceCloner, Trainer, InferenceEngine, Config

# Initialize components
config = Config()
config.load()

audio_input = AudioInput()
feature_extractor = FeatureExtractor(config)
model = VoiceCloner(config)
trainer = Trainer(model, feature_extractor, config)
inference_engine = InferenceEngine(model, feature_extractor, config)

# Record or load audio
audio_data = audio_input.record_audio(duration=5)
# or
audio_data, sr = audio_input.load_audio("path/to/audio.wav")

# Train model (if needed)
trainer.train("path/to/training/audio.wav")

# Generate cloned voice
cloned_audio = inference_engine.generate_voice("Hello, world!", "path/to/reference/audio.wav")

# Save result
audio_input.save_audio(cloned_audio, "output.wav")
```

## Project Structure

```
ai_voice-clone/
├── main.py                 # CLI entry point and GUI launcher
├── UI/
│   └── gui.py             # Graphical user interface
├── config.py              # Configuration management
├── audio_input.py         # Audio recording and loading
├── feature_extraction.py  # Audio feature extraction
├── model.py               # Neural network models
├── training.py            # Model training logic
├── inference.py           # Voice generation
├── __init__.py            # Package initialization
├── requirements.txt       # Dependencies
├── launch_gui.py          # GUI launcher script
├── config.yaml            # Default configuration
└── ...
```

## Configuration

The system uses a YAML configuration file (`config.yaml`) with the following main sections:

- `audio`: Audio processing parameters
- `features`: Feature extraction settings
- `model`: Neural network architecture
- `training`: Training hyperparameters
- `inference`: Generation parameters

## Contributing

See [CONTRIBUTION.md](CONTRIBUTION.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details.

## Testing

See [TESTING.md](TESTING.md) for testing procedures.

## Update Log

See [UPDATE-LOG.md](UPDATE-LOG.md) for version history.