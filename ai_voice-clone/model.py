"""
Neural network model for AI Voice Clone.
Implements encoder-decoder architecture for voice cloning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VoiceEncoder(nn.Module):
    """Encoder for voice characteristics extraction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.1, bidirectional: bool = True):
        """
        Initialize voice encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(VoiceEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output dimension depends on bidirectional setting
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention mechanism for voice embedding
        self.attention = nn.Linear(lstm_output_dim, 1)

        # Voice embedding projection
        self.voice_embedding = nn.Linear(lstm_output_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, seq_len, input_dim)

        Returns:
            Voice embedding (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)

        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        attention_weights = torch.tanh(self.attention(lstm_out))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum for voice embedding
        voice_embedding = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim * 2)

        # Project to final embedding
        voice_embedding = self.voice_embedding(voice_embedding)  # (batch_size, hidden_dim)

        return voice_embedding


class VoiceDecoder(nn.Module):
    """Decoder for generating speech from text and voice embedding."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 attention_dim: int, dropout: float = 0.1):
        """
        Initialize voice decoder.

        Args:
            vocab_size: Size of text vocabulary
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            attention_dim: Attention mechanism dimension
            dropout: Dropout rate
        """
        super(VoiceDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Text embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Concatenated with attention context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim, attention_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_input: torch.Tensor, voice_embedding: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one time step.

        Args:
            text_input: Text input tokens (batch_size,)
            voice_embedding: Voice embedding (batch_size, hidden_dim)
            hidden: Previous hidden state

        Returns:
            Tuple of (output_logits, new_hidden_state)
        """
        # Embed text input
        embedded = self.embedding(text_input).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Get attention context
        context = self.attention(voice_embedding, embedded.squeeze(1))  # (batch_size, hidden_dim)

        # Concatenate embedded input with context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch_size, 1, hidden_dim * 2)

        # LSTM step
        lstm_out, hidden = self.lstm(lstm_input, hidden)

        # Output projection
        output = self.output_projection(lstm_out.squeeze(1))  # (batch_size, vocab_size)

        return output, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""

    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism.

        Args:
            hidden_dim: Hidden dimension
            attention_dim: Attention dimension
        """
        super(BahdanauAttention, self).__init__()

        self.attention = nn.Linear(hidden_dim * 2, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, voice_embedding: torch.Tensor, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights.

        Args:
            voice_embedding: Voice embedding (batch_size, hidden_dim)
            decoder_hidden: Decoder hidden state (batch_size, hidden_dim)

        Returns:
            Context vector (batch_size, hidden_dim)
        """
        # Concatenate voice embedding and decoder hidden
        combined = torch.cat([voice_embedding, decoder_hidden], dim=1)  # (batch_size, hidden_dim * 2)

        # Compute attention energy
        energy = torch.tanh(self.attention(combined))  # (batch_size, attention_dim)
        attention = self.v(energy)  # (batch_size, 1)

        # Since we only have one "encoder" output (voice embedding), attention is always 1
        # This is a simplified attention for single voice embedding
        return voice_embedding


class VoiceCloner(nn.Module):
    """Complete voice cloning model."""

    def __init__(self, config: 'Config'):
        """
        Initialize voice cloning model.

        Args:
            config: Configuration object
        """
        super(VoiceCloner, self).__init__()

        # Model parameters from config
        self.feature_dim = config.get('features.n_mels', 80)
        self.vocab_size = 100  # Basic vocabulary size, can be expanded
        self.encoder_hidden = config.get('model.encoder_hidden_size', 256)
        self.decoder_hidden = config.get('model.decoder_hidden_size', 512)
        self.encoder_layers = config.get('model.encoder_num_layers', 2)
        self.decoder_layers = config.get('model.decoder_num_layers', 3)
        self.attention_dim = config.get('model.attention_dim', 128)
        self.dropout = config.get('model.dropout', 0.1)
        self.bidirectional = config.get('model.bidirectional', True)

        # Encoder
        self.encoder = VoiceEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.encoder_hidden,
            num_layers=self.encoder_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        # Decoder
        self.decoder = VoiceDecoder(
            vocab_size=self.vocab_size,
            hidden_dim=self.decoder_hidden,
            num_layers=self.decoder_layers,
            attention_dim=self.attention_dim,
            dropout=self.dropout
        )

        # Output projection to mel spectrograms
        self.mel_projection = nn.Linear(self.decoder_hidden, self.feature_dim)

        logger.info(f"VoiceCloner model initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def encode_voice(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode voice characteristics.

        Args:
            audio_features: Audio features (batch_size, seq_len, feature_dim)

        Returns:
            Voice embedding (batch_size, encoder_hidden)
        """
        return self.encoder(audio_features)

    def decode_speech(self, text_tokens: torch.Tensor, voice_embedding: torch.Tensor,
                     max_length: int = 100) -> torch.Tensor:
        """
        Decode speech from text and voice embedding.

        Args:
            text_tokens: Text tokens (batch_size, seq_len)
            voice_embedding: Voice embedding (batch_size, encoder_hidden)
            max_length: Maximum output length

        Returns:
            Generated mel spectrograms (batch_size, max_length, feature_dim)
        """
        batch_size = text_tokens.shape[0]
        outputs = []

        # Initialize decoder hidden state with voice embedding
        decoder_hidden = self._init_decoder_hidden(voice_embedding, batch_size)

        # Start with SOS token (assuming 0 is SOS)
        current_token = torch.zeros(batch_size, dtype=torch.long, device=text_tokens.device)

        for t in range(max_length):
            # Decoder step
            output, decoder_hidden = self.decoder(current_token, voice_embedding, decoder_hidden)

            # Project to mel spectrograms
            mel_frame = self.mel_projection(output)  # (batch_size, feature_dim)
            outputs.append(mel_frame.unsqueeze(1))

            # Get next token (greedy decoding for now)
            current_token = output.argmax(dim=1)

            # Stop if EOS token (assuming 1 is EOS)
            if (current_token == 1).all():
                break

        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, feature_dim)

        return outputs

    def _init_decoder_hidden(self, voice_embedding: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize decoder hidden state.

        Args:
            voice_embedding: Voice embedding
            batch_size: Batch size

        Returns:
            Initial hidden state for decoder
        """
        # Project voice embedding to decoder hidden size
        hidden = voice_embedding.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        cell = torch.zeros_like(hidden)

        return (hidden, cell)

    def forward(self, audio_features: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio_features: Audio features (batch_size, seq_len, feature_dim)
            text_tokens: Text tokens (batch_size, seq_len)

        Returns:
            Generated mel spectrograms (batch_size, seq_len, feature_dim)
        """
        # Encode voice
        voice_embedding = self.encode_voice(audio_features)

        # Decode speech
        output = self.decode_speech(text_tokens, voice_embedding)

        return output