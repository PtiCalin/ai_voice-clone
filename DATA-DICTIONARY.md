# Data Dictionary - AI Voice Clone

This document serves as a comprehensive reference for all data elements, functions, modules, and structures used in the AI Voice Clone project.

## Overview

The AI Voice Clone project implements voice synthesis and cloning using deep learning techniques. This data dictionary documents all code elements to ensure consistency, maintainability, and clear understanding across the development team.

## Table of Contents

- [Variables](#variables)
- [User-Defined Functions](#user-defined-functions)
- [Modules and Classes](#modules-and-classes)
- [Data Structures](#data-structures)
- [Constants](#constants)
- [Audio Data Formats](#audio-data-formats)
- [Model Parameters](#model-parameters)

---

## Variables

### Global Variables

| Variable Name | Type | Description | Default Value | Usage Context |
|---------------|------|-------------|---------------|---------------|
| `SAMPLE_RATE` | int | Audio sampling rate in Hz | 22050 | Audio processing |
| `AUDIO_CHANNELS` | int | Number of audio channels | 1 | Mono audio |
| `MAX_AUDIO_LENGTH` | float | Maximum audio length in seconds | 10.0 | Input validation |
| `MODEL_PATH` | str | Path to trained model file | "models/voice_clone.pt" | Model loading |

### Local Variables

| Variable Name | Type | Description | Scope | Example Usage |
|---------------|------|-------------|-------|---------------|
| `audio_data` | numpy.ndarray | Raw audio waveform data | Function-level | `audio_data = load_audio(file_path)` |
| `spectrogram` | torch.Tensor | Audio spectrogram | Function-level | `spectrogram = extract_features(audio_data)` |
| `text_input` | str | Text to be synthesized | Function-level | `text_input = "Hello, world!"` |
| `output_audio` | numpy.ndarray | Generated audio output | Function-level | `output_audio = model.generate(text_input)` |

---

## User-Defined Functions

### Audio Processing Functions

| Function Name | Parameters | Return Type | Description | File Location |
|---------------|------------|-------------|-------------|---------------|
| `record_audio` | duration: int = 5, device: Optional[int] = None | numpy.ndarray | Record audio from microphone | `audio_input.py` |
| `load_audio` | file_path: str | Tuple[numpy.ndarray, int] | Load audio from file | `audio_input.py` |
| `save_audio` | audio: numpy.ndarray, file_path: str, sample_rate: Optional[int] = None | None | Save audio to file | `audio_input.py` |
| `extract_features` | audio: numpy.ndarray, feature_type: str = 'mel' | torch.Tensor | Extract audio features | `feature_extraction.py` |
| `extract_multiple_features` | audio: numpy.ndarray | Dict[str, torch.Tensor] | Extract multiple feature types | `feature_extraction.py` |
| `normalize_features` | features: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None | Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Normalize features | `feature_extraction.py` |
| `extract_pitch` | audio: numpy.ndarray | numpy.ndarray | Extract pitch from audio | `feature_extraction.py` |
| `extract_voice_quality_features` | audio: numpy.ndarray | Dict[str, float] | Extract voice quality metrics | `feature_extraction.py` |

### Model Functions

| Function Name | Parameters | Return Type | Description | File Location |
|---------------|------------|-------------|-------------|---------------|
| `encode_voice` | audio_features: torch.Tensor | torch.Tensor | Encode voice characteristics | `model.py` |
| `decode_speech` | text_tokens: torch.Tensor, voice_embedding: torch.Tensor, max_length: int = 100 | torch.Tensor | Decode speech from text and voice | `model.py` |
| `prepare_data` | audio_file: str | Tuple[DataLoader, DataLoader] | Prepare training and validation data | `training.py` |
| `train_epoch` | train_loader: DataLoader | float | Train for one epoch | `training.py` |
| `validate` | val_loader: DataLoader | float | Validate model performance | `training.py` |
| `train` | audio_file: str | Dict[str, any] | Complete training process | `training.py` |
| `generate_voice` | text: str, reference_audio_path: str | numpy.ndarray | Generate cloned voice | `inference.py` |
| `text_to_tokens` | text: str | torch.Tensor | Convert text to token sequence | `inference.py` |
| `load_model` | model_path: str | None | Load trained model | `inference.py` |

### Data Processing Functions

| Function Name | Parameters | Return Type | Description | File Location |
|---------------|------------|-------------|-------------|---------------|
| `create_dataset` | audio_dir: str, text_file: str | Dataset | Create PyTorch dataset | `data_loader.py` |
| `collate_fn` | batch: list | dict | Collate batch for DataLoader | `data_loader.py` |
| `split_dataset` | dataset: Dataset, train_ratio: float = 0.8 | tuple | Split dataset into train/val | `data_utils.py` |
| `preprocess_text` | text: str | str | Clean and normalize text | `text_utils.py` |
| `tokenize_text` | text: str | list | Convert text to token IDs | `text_utils.py` |

### Utility Functions

| Function Name | Parameters | Return Type | Description | File Location |
|---------------|------------|-------------|-------------|---------------|
| `setup_logging` | log_level: str = "INFO", log_file: str = None | None | Configure logging | `utils.py` |
| `calculate_metrics` | predicted: numpy.ndarray, target: numpy.ndarray | dict | Calculate audio quality metrics | `metrics.py` |
| `plot_spectrogram` | spectrogram: numpy.ndarray, save_path: str = None | matplotlib.figure.Figure | Visualize spectrogram | `visualization.py` |
| `time_function` | func: callable | callable | Decorator to time function execution | `utils.py` |

---

## Modules and Classes

### Core Modules

| Module Name | Description | Key Classes/Functions | Dependencies |
|-------------|-------------|----------------------|--------------|
| `main.py` | CLI entry point and main program logic | `main()`, `record_voice_sample()`, `clone_voice()`, `train_model()` | All other modules |
| `config.py` | Configuration management | `Config` class, `load()`, `save()`, `get()` | yaml, pathlib |
| `audio_input.py` | Audio recording and file I/O | `AudioInput` class, `record_audio()`, `load_audio()`, `save_audio()` | numpy, soundfile, sounddevice |
| `feature_extraction.py` | Audio feature extraction | `FeatureExtractor` class, `extract_features()`, `normalize_features()` | torch, torchaudio, librosa, numpy |
| `model.py` | Neural network architectures | `VoiceCloner`, `VoiceEncoder`, `VoiceDecoder`, `BahdanauAttention` | torch, torch.nn |
| `training.py` | Model training logic | `Trainer` class, `train()`, `train_epoch()`, `validate()` | torch, torch.nn, tqdm |
| `inference.py` | Voice generation and inference | `InferenceEngine` class, `generate_voice()`, `text_to_tokens()` | torch, numpy, librosa |

### Utility Modules

| Module Name | Description | Key Classes/Functions | Dependencies |
|-------------|-------------|----------------------|--------------|
| `config.py` | Configuration management | `Config`, `load_config()` | yaml, argparse |
| `utils.py` | General utility functions | `setup_logging()`, `time_function()` | logging, time, functools |
| `metrics.py` | Evaluation metrics | `calculate_metrics()`, `PESQ`, `STOI` | numpy, pesq, pystoi |
| `visualization.py` | Plotting and visualization | `plot_spectrogram()`, `plot_loss()` | matplotlib, seaborn |
| `text_utils.py` | Text processing utilities | `preprocess_text()`, `tokenize_text()` | re, nltk |

### Main Classes

| Class Name | Module | Description | Key Methods | Inheritance |
|------------|--------|-------------|-------------|-------------|
| `Config` | `config.py` | Configuration manager | `load()`, `save()`, `get()`, `set()` | object |
| `AudioInput` | `audio_input.py` | Audio recording and loading | `record_audio()`, `load_audio()`, `save_audio()` | object |
| `FeatureExtractor` | `feature_extraction.py` | Audio feature extraction | `extract_features()`, `normalize_features()`, `extract_pitch()` | object |
| `VoiceEncoder` | `model.py` | Voice characteristics encoder | `forward()` | torch.nn.Module |
| `VoiceDecoder` | `model.py` | Speech generation decoder | `forward()` | torch.nn.Module |
| `BahdanauAttention` | `model.py` | Attention mechanism | `forward()` | torch.nn.Module |
| `VoiceCloner` | `model.py` | Complete voice cloning model | `encode_voice()`, `decode_speech()`, `forward()` | torch.nn.Module |
| `Trainer` | `training.py` | Model training manager | `train()`, `train_epoch()`, `validate()`, `save_checkpoint()` | object |
| `InferenceEngine` | `inference.py` | Voice generation engine | `generate_voice()`, `load_model()`, `text_to_tokens()` | object |

---

## Data Structures

### Audio Data Structures

| Structure Name | Type | Description | Fields/Properties |
|----------------|------|-------------|-------------------|
| `AudioSample` | dict | Single audio sample with metadata | `waveform`: numpy.ndarray, `sample_rate`: int, `duration`: float, `speaker_id`: str |
| `AudioBatch` | dict | Batch of audio samples | `waveforms`: torch.Tensor, `sample_rates`: list, `texts`: list, `speaker_ids`: list |
| `SpectrogramData` | dict | Spectrogram representation | `spectrogram`: torch.Tensor, `mel_spectrogram`: torch.Tensor, `phase`: torch.Tensor |

### Model Data Structures

| Structure Name | Type | Description | Fields/Properties |
|----------------|------|-------------|-------------------|
| `ModelConfig` | dict | Model hyperparameters | `hidden_size`: int, `num_layers`: int, `dropout`: float, `learning_rate`: float |
| `TrainingState` | dict | Current training state | `epoch`: int, `loss`: float, `best_loss`: float, `model_state`: dict |
| `InferenceInput` | dict | Input for inference | `text`: str, `reference_audio`: numpy.ndarray, `speaker_embedding`: torch.Tensor |

### Dataset Structures

| Structure Name | Type | Description | Fields/Properties |
|----------------|------|-------------|-------------------|
| `DatasetItem` | dict | Single dataset item | `audio_path`: str, `text`: str, `speaker_id`: str, `duration`: float |
| `DataLoaderConfig` | dict | DataLoader configuration | `batch_size`: int, `shuffle`: bool, `num_workers`: int, `pin_memory`: bool |

---

## Constants

### Audio Constants

| Constant Name | Value | Type | Description |
|---------------|-------|------|-------------|
| `DEFAULT_SAMPLE_RATE` | 22050 | int | Default audio sampling rate |
| `MIN_AUDIO_LENGTH` | 1.0 | float | Minimum audio length in seconds |
| `MAX_AUDIO_LENGTH` | 30.0 | float | Maximum audio length in seconds |
| `N_MELS` | 80 | int | Number of mel bins for spectrograms |
| `N_FFT` | 1024 | int | FFT size for STFT |
| `HOP_LENGTH` | 256 | int | Hop length for STFT |

### Model Constants

| Constant Name | Value | Type | Description |
|---------------|-------|------|-------------|
| `HIDDEN_SIZE` | 512 | int | Default hidden layer size |
| `NUM_LAYERS` | 3 | int | Default number of layers |
| `DROPOUT_RATE` | 0.1 | float | Default dropout rate |
| `LEARNING_RATE` | 0.001 | float | Default learning rate |
| `BATCH_SIZE` | 16 | int | Default batch size |

### File Path Constants

| Constant Name | Value | Type | Description |
|---------------|-------|------|-------------|
| `DATA_DIR` | "data/" | str | Data directory path |
| `MODEL_DIR` | "models/" | str | Model directory path |
| `LOG_DIR` | "logs/" | str | Log directory path |
| `CONFIG_FILE` | "config.yaml" | str | Configuration file path |

---

## Audio Data Formats

### Input Formats

| Format | Description | Supported | Conversion Required |
|--------|-------------|-----------|---------------------|
| WAV | Uncompressed audio | Yes | No |
| MP3 | Compressed audio | Yes | To WAV |
| FLAC | Lossless compressed | Yes | To WAV |
| OGG | Compressed audio | Yes | To WAV |
| M4A | AAC compressed | Yes | To WAV |

### Internal Formats

| Format | Description | Data Type | Shape |
|--------|-------------|-----------|-------|
| Raw Waveform | Time-domain audio | numpy.ndarray | (samples,) or (channels, samples) |
| Spectrogram | Frequency-domain | torch.Tensor | (batch, freq_bins, time_steps) |
| Mel Spectrogram | Mel-scale spectrogram | torch.Tensor | (batch, n_mels, time_steps) |
| MFCC | Mel-frequency cepstral coefficients | torch.Tensor | (batch, n_mfcc, time_steps) |

### Output Formats

| Format | Description | Quality | Use Case |
|--------|-------------|---------|----------|
| WAV | Uncompressed output | High | Development/Testing |
| MP3 | Compressed output | Medium | Distribution |
| OGG | Compressed output | Medium | Web deployment |

---

## Model Parameters

### Encoder Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `encoder_hidden_size` | int | 256 | 128-1024 | Encoder hidden dimension |
| `encoder_num_layers` | int | 2 | 1-6 | Number of encoder layers |
| `encoder_dropout` | float | 0.1 | 0.0-0.5 | Encoder dropout rate |
| `encoder_bidirectional` | bool | True | - | Bidirectional encoder |

### Decoder Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `decoder_hidden_size` | int | 512 | 256-2048 | Decoder hidden dimension |
| `decoder_num_layers` | int | 3 | 1-6 | Number of decoder layers |
| `decoder_dropout` | float | 0.1 | 0.0-0.5 | Decoder dropout rate |
| `attention_dim` | int | 128 | 64-512 | Attention mechanism dimension |

### Training Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `learning_rate` | float | 0.001 | 1e-5 - 1e-2 | Optimizer learning rate |
| `batch_size` | int | 16 | 1-128 | Training batch size |
| `epochs` | int | 100 | 1-1000 | Maximum training epochs |
| `gradient_clip` | float | 1.0 | 0.1-10.0 | Gradient clipping threshold |
| `weight_decay` | float | 1e-6 | 1e-8 - 1e-4 | L2 regularization |

---

## Maintenance Notes

- Update this document when adding new functions, variables, or modules
- Include type hints and docstrings in code for automatic documentation
- Use consistent naming conventions across the project
- Document any changes to data structures or API interfaces
- Review and update periodically to ensure accuracy

## Version History

- v1.0.0: Initial data dictionary setup (December 2025)
- v1.1.0: Updated with actual implementation details including main.py, config.py, audio_input.py, feature_extraction.py, model.py, training.py, inference.py, and supporting files (December 2025)