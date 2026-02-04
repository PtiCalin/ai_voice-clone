"""
Text processing utilities for AI Voice Clone.
Provides shared tokenization logic for training and inference.
"""

from typing import Dict, Optional
import torch


class TextTokenizer:
    """Simple character-level tokenizer shared across training and inference."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or self._create_vocab()
        self.pad_token_id = self.vocab.get('<PAD>', 2)

    def _create_vocab(self) -> Dict[str, int]:
        """Create a simple vocabulary for text processing."""
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
        tokens = [self.vocab.get('<SOS>', 0)]

        for char in text.lower():
            token = self.vocab.get(char, self.vocab.get('<UNK>', self.pad_token_id))
            tokens.append(token)

        tokens.append(self.vocab.get('<EOS>', 1))
        return torch.tensor(tokens, dtype=torch.long)
