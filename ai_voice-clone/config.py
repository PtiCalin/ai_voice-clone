"""
Configuration management for AI Voice Clone.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the voice cloning system."""

    def __init__(self):
        """Initialize default configuration."""
        self.config = {
            # Audio parameters
            "audio": {
                "sample_rate": 22050,
                "channels": 1,
                "max_length": 10.0,
                "min_length": 1.0,
            },

            # Feature extraction parameters
            "features": {
                "n_mels": 80,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "fmin": 0,
                "fmax": 8000,
            },

            # Model parameters
            "model": {
                "encoder_hidden_size": 256,
                "encoder_num_layers": 2,
                "decoder_hidden_size": 512,
                "decoder_num_layers": 3,
                "attention_dim": 128,
                "dropout": 0.1,
                "bidirectional": True,
            },

            # Training parameters
            "training": {
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 100,
                "gradient_clip": 1.0,
                "weight_decay": 1e-6,
                "validation_split": 0.1,
            },

            # Inference parameters
            "inference": {
                "temperature": 0.8,
                "top_k": 40,
                "max_length": 1000,
            },

            # File paths
            "paths": {
                "data_dir": "data",
                "model_dir": "models",
                "log_dir": "logs",
                "config_file": "config.yaml",
            }
        }

    def load(self, config_file: str = None) -> None:
        """Load configuration from YAML file."""
        if config_file is None:
            config_file = self.config["paths"]["config_file"]

        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self._merge_config(self.config, loaded_config)

    def save(self, config_file: str = None) -> None:
        """Save configuration to YAML file."""
        if config_file is None:
            config_file = self.config["paths"]["config_file"]

        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dictionary access."""
        self.set(key, value)