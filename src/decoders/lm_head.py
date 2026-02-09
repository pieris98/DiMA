"""
LM Head decoder — wraps an encoder's built-in language model head.

Used when no separate decoder training is needed (e.g., ESM2 with its
own lm_head, or CHEAP with LatentToSequence).
"""

import torch
import torch.nn as nn
from typing import Optional

from src.decoders.base import BaseDecoder
from pipeline.registry import registry


@registry.register_decoder("lm_head")
class LMHeadDecoder(BaseDecoder):
    """
    Wraps an arbitrary nn.Module (typically an encoder's lm_head)
    as a BaseDecoder.
    """

    def __init__(self, lm_head: nn.Module):
        """
        Args:
            lm_head: The language model head module to wrap.
        """
        super().__init__()
        self.lm_head = lm_head

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Most lm_heads don't use a mask argument
        return self.lm_head(x)

    def load_checkpoint(self, path: str) -> None:
        # LM heads are typically loaded as part of the encoder, not separately
        pass
