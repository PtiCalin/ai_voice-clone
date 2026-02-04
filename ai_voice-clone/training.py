"""
Training module for AI Voice Clone.
Handles model training, validation, and checkpointing.
"""

import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

from ai_voice_clone.text_processing import TextTokenizer

logger = logging.getLogger(__name__)


class PairedAudioTextDataset(Dataset):
    """Dataset of paired audio features and text tokens."""

    def __init__(self, examples: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


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
        self.tokenizer = TextTokenizer()

    def prepare_data(self, audio_file: Optional[str] = None,
                     transcript_file: Optional[str] = None,
                     manifest_path: Optional[str] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data from paired audio/text data.

        Args:
            audio_file: Path to audio file
            transcript_file: Path to transcript file
            manifest_path: Path to dataset manifest file

        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Preparing data from paired audio/text transcripts.")

        manifest_path = manifest_path or self.config.get('training.manifest_path')

        pairs = self._resolve_pairs(audio_file, transcript_file, manifest_path)
        examples = self._build_examples(pairs)

        dataset = PairedAudioTextDataset(examples)

        if len(dataset) < 2:
            logger.warning("Dataset too small for train/val split. Using the same data for both.")
            train_dataset = dataset
            val_dataset = dataset
        else:
            val_split = self.config.get('training.validation_split', 0.1)
            val_size = max(1, int(len(dataset) * val_split))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch
        )

        logger.info(f"Data prepared. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    def _resolve_pairs(self, audio_file: Optional[str],
                       transcript_file: Optional[str],
                       manifest_path: Optional[str]) -> List[Tuple[str, str]]:
        if manifest_path:
            return self._load_manifest(manifest_path)

        if not audio_file:
            raise ValueError("Audio file path or manifest path must be provided for training.")

        if transcript_file:
            transcript = self._load_transcript(transcript_file)
            return [(audio_file, transcript)]

        inferred = self._infer_transcript_path(audio_file)
        transcript = self._load_transcript(inferred)
        return [(audio_file, transcript)]

    def _infer_transcript_path(self, audio_file: str) -> str:
        extension = self.config.get('training.transcript_extension', '.txt')
        return str(Path(audio_file).with_suffix(extension))

    def _load_transcript(self, transcript_file: str) -> str:
        transcript_path = Path(transcript_file)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        return transcript_path.read_text(encoding='utf-8').strip()

    def _load_manifest(self, manifest_path: str) -> List[Tuple[str, str]]:
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest}")

        if manifest.suffix in {'.jsonl', '.json'}:
            return self._load_json_manifest(manifest)

        delimiter = self.config.get('training.manifest_delimiter', ',')
        pairs: List[Tuple[str, str]] = []
        with manifest.open('r', encoding='utf-8') as handle:
            sample = handle.read(1024)
            handle.seek(0)
            if '|' in sample and delimiter == ',':
                delimiter = '|'
            reader = csv.reader(handle, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            return pairs

        header = [cell.strip().lower() for cell in rows[0]]
        if 'audio_path' in header or 'transcript' in header or 'text' in header:
            audio_idx = header.index('audio_path') if 'audio_path' in header else 0
            text_idx = header.index('transcript') if 'transcript' in header else header.index('text')
            for row in rows[1:]:
                if len(row) <= max(audio_idx, text_idx):
                    continue
                audio_path = self._resolve_audio_path(row[audio_idx].strip(), manifest.parent)
                pairs.append((audio_path, row[text_idx].strip()))
        else:
            for row in rows:
                if len(row) < 2:
                    continue
                audio_path = self._resolve_audio_path(row[0].strip(), manifest.parent)
                pairs.append((audio_path, row[1].strip()))

        return pairs

    def _load_json_manifest(self, manifest: Path) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if manifest.suffix == '.jsonl':
            with manifest.open('r', encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    audio_path = record.get('audio_path') or record.get('audio')
                    transcript = record.get('transcript') or record.get('text')
                    if audio_path and transcript:
                        resolved_path = self._resolve_audio_path(audio_path, manifest.parent)
                        pairs.append((resolved_path, transcript))
        else:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            items = data.get('items', data) if isinstance(data, dict) else data
            for record in items:
                audio_path = record.get('audio_path') or record.get('audio')
                transcript = record.get('transcript') or record.get('text')
                if audio_path and transcript:
                    resolved_path = self._resolve_audio_path(audio_path, manifest.parent)
                    pairs.append((resolved_path, transcript))
        return pairs

    def _resolve_audio_path(self, audio_path: str, base_dir: Path) -> str:
        path = Path(audio_path)
        if not path.is_absolute():
            path = base_dir / path
        return str(path)

    def _build_examples(self, pairs: List[Tuple[str, str]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        from ai_voice_clone.audio_input import AudioInput

        audio_input = AudioInput()
        examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for audio_path, transcript in pairs:
            audio_data, _ = audio_input.load_audio(audio_path)
            features = self.feature_extractor.extract_features(audio_data)
            if features.dim() == 2:
                features = features.transpose(0, 1)
            tokens = self.tokenizer.text_to_tokens(transcript)
            examples.append((features, tokens))

        return examples

    def _collate_batch(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        features, tokens = zip(*batch)
        padded_features = pad_sequence(features, batch_first=True)
        padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded_features, padded_tokens

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

    def train(self, audio_file: Optional[str] = None,
              transcript_file: Optional[str] = None,
              manifest_path: Optional[str] = None) -> Dict[str, any]:
        """
        Train the model.

        Args:
            audio_file: Path to training audio file
            transcript_file: Path to transcript file for audio_file
            manifest_path: Path to manifest file for paired data

        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")

        # Prepare data
        train_loader, val_loader = self.prepare_data(
            audio_file=audio_file,
            transcript_file=transcript_file,
            manifest_path=manifest_path
        )

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
