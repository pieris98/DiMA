"""
Abstract base class for all decoders in the DiMA pipeline.

A decoder takes latent embeddings produced by the diffusion model
(after denormalization) and produces either logits over a vocabulary
or decoded sequences.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List


class BaseDecoder(nn.Module, ABC):
    """
    Interface that all DiMA decoders must implement.

    Subclasses must implement:
        - forward(x, mask) -> logits tensor
        - load_checkpoint(path) -> None

    Optionally override:
        - decode_to_sequences(logits, tokenizer, attention_mask) -> List[str]
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent embeddings to logits.

        Args:
            x: Latent embeddings of shape (batch, seq_len, embed_dim).
            mask: Optional attention mask of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load decoder weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file (.pth).
        """
        ...

    def decode_to_sequences(
        self,
        logits: torch.Tensor,
        tokenizer,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """
        Convert logits to decoded string sequences.

        Default implementation: argmax -> tokenizer.batch_decode.
        Override for custom decoding strategies.

        Args:
            logits: Logits of shape (batch, seq_len, vocab_size).
            tokenizer: Tokenizer with a batch_decode method.
            attention_mask: Optional mask of shape (batch, seq_len).

        Returns:
            List of decoded strings.
        """
        token_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        if attention_mask is not None:
            for i, t in enumerate(token_ids):
                seq_len = int(attention_mask[i].sum().item())
                token_ids[i] = t[:seq_len]

        decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return ["".join(s.split()) for s in decoded]
