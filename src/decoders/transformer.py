"""
Transformer-based decoder wrapper.

Wraps the existing TransformerDecoder (from src/encoders/transformer_decoder.py)
to implement the BaseDecoder interface and integrate with the registry.
"""

import os
import torch
from typing import Optional

from src.decoders.base import BaseDecoder
from src.encoders.transformer_decoder import TransformerDecoder
from pipeline.registry import registry


@registry.register_decoder("transformer")
class TransformerDecoderWrapper(BaseDecoder):
    """
    Wraps the existing TransformerDecoder architecture to conform
    to the BaseDecoder interface.
    """

    def __init__(self, config):
        """
        Args:
            config: The main Hydra DictConfig (needs encoder and decoder sub-configs).
        """
        super().__init__()
        self.inner = TransformerDecoder(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner(x, mask)

    def load_checkpoint(self, path: str) -> None:
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            self.inner.load_state_dict(state["decoder"])
        else:
            print(f"Warning: Decoder checkpoint not found at {path}")
