"""
Training module for AI Voice Clone.
Handles model training, validation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training of the voice cloning model."""

    def __init__(self, model: nn.Module, feature_extractor: 'FeatureExtractor',
                 config: 'Config', device: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            model: Voice cloning model
            feature_extractor: Feature extractor
            config: Configuration object
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Training on device: {self.device}")

        # Move model to device
        self.model.to(self.device)

        # Training parameters
        self.learning_rate = config.get('training.learning_rate', 0.001)
        self.batch_size = config.get('training.batch_size', 16)
        self.epochs = config.get('training.epochs', 100)
        self.gradient_clip = config.get('training.gradient_clip', 1.0)
        self.weight_decay = config.get('training.weight_decay', 1e-6)

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Loss functions
        self.mel_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []

    def prepare_data(self, audio_file: str) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data from audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info(f"Preparing data from: {audio_file}")

        # Load audio
        from ai_voice_clone.audio_input import AudioInput
        audio_input = AudioInput()
        audio_data, sample_rate = audio_input.load_audio(audio_file)

        # Extract features
        features = self.feature_extractor.extract_features(audio_data)

        # For now, create synthetic text tokens (this would be replaced with actual text data)
        # In a real implementation, you'd have paired audio-text data
        seq_length = features.shape[1]
        text_tokens = torch.randint(0, 100, (seq_length,))  # Random tokens for demo

        # Create dataset
        # Features: (seq_len, feature_dim) -> (batch_size, seq_len, feature_dim)
        features = features.unsqueeze(0)  # Add batch dimension
        text_tokens = text_tokens.unsqueeze(0)  # Add batch dimension

        # Split into train/val
        val_split = self.config.get('training.validation_split', 0.1)
        split_idx = int(seq_length * (1 - val_split))

        train_features = features[:, :split_idx, :]
        train_text = text_tokens[:, :split_idx]
        val_features = features[:, split_idx:, :]
        val_text = text_tokens[:, split_idx:]

        # Create data loaders
        train_dataset = TensorDataset(train_features, train_text)
        val_dataset = TensorDataset(val_features, val_text)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Data prepared. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_features, batch_text in progress_bar:
            # Move to device
            batch_features = batch_features.to(self.device)
            batch_text = batch_text.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features, batch_text)

            # Compute loss (simplified - in practice you'd compare to target mel spectrograms)
            # For now, using reconstruction loss
            target_mel = batch_features  # Simplified: target is input features
            loss = self.mel_loss(outputs, target_mel) + 0.1 * self.l1_loss(outputs, target_mel)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Optimizer step
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {self.current_epoch + 1} training loss: {avg_loss:.4f}")

        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_features, batch_text in val_loader:
                # Move to device
                batch_features = batch_features.to(self.device)
                batch_text = batch_text.to(self.device)

                # Forward pass
                outputs = self.model(batch_features, batch_text)

                # Compute loss
                target_mel = batch_features
                loss = self.mel_loss(outputs, target_mel) + 0.1 * self.l1_loss(outputs, target_mel)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def train(self, audio_file: str) -> Dict[str, any]:
        """
        Train the model.

        Args:
            audio_file: Path to training audio file

        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")

        # Prepare data
        train_loader, val_loader = self.prepare_data(audio_file)

        # Training loop
        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f"best_model.pt")

            # Log progress
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_info)

            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        # Save final model
        self.save_checkpoint("final_model.pt")

        results = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': self.best_loss,
            'training_history': self.training_history
        }

        logger.info(f"Training completed. Best validation loss: {self.best_loss:.4f}")

        return results

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = Path("models") / filename
        checkpoint_path.parent.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config.config
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = Path("models") / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def save_model(self, filename: str) -> None:
        """
        Save model for inference.

        Args:
            filename: Model filename
        """
        model_path = Path("models") / filename
        model_path.parent.mkdir(exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.config
        }, model_path)

        logger.info(f"Model saved: {model_path}")

    def load_model(self, filename: str) -> None:
        """
        Load model for inference.

        Args:
            filename: Model filename
        """
        model_path = Path("models") / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded: {model_path}")