"""
Family-specific classifier for noisy protein encodings.

This module implements the lightweight classifier described in §3.6.2 of the paper:
"For classifier guidance, we train a lightweight classifier (3 transformer blocks)
 on noisy protein encodings to predict family membership."

The classifier operates on noisy latent vectors z_t at various noise levels
and predicts which protein family the underlying clean protein belongs to.
During inference, gradients from this classifier can steer the diffusion
process toward generating family-specific proteins.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from src.models.blocks import BertBlock, timestep_embedding


class FamilyClassifierConfig:
    """Configuration for the family classifier."""
    def __init__(
        self,
        num_families: int = 8,
        embedding_size: int = 1280,    # Should match encoder dim
        hidden_size: int = 320,
        num_hidden_layers: int = 3,    # "3 transformer blocks" per paper §3.6.2
        num_attention_heads: int = 16,
        attention_head_size: int = 20,
        intermediate_size: int = 1280,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        norm_type: str = "layernorm",
        qk_norm: bool = False,
        add_cross_attention: bool = False,
        use_self_cond: bool = False,
    ):
        self.num_families = num_families
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.norm_type = norm_type
        self.qk_norm = qk_norm
        self.add_cross_attention = add_cross_attention
        self.use_self_cond = use_self_cond


class FamilyClassifier(nn.Module):
    """
    Lightweight transformer classifier for predicting protein family 
    membership from noisy latent encodings.
    
    Architecture: input projection -> 3 transformer blocks -> mean pooling -> classification head
    
    The classifier takes as input:
        - z_t: noisy protein encodings at timestep t
        - t: diffusion timestep  
        - attention_mask: sequence mask
    
    And outputs logits over protein families.
    """
    
    def __init__(self, config: FamilyClassifierConfig):
        super().__init__()
        self.config = config
        
        # Input projection (encoder dim -> hidden dim)
        if config.embedding_size != config.hidden_size:
            self.input_proj = nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.input_proj = nn.Identity()
        
        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        
        # Time integration layers
        self.time_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BertBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Positional embeddings
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_families),
        )
    
    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        extended = attention_mask[:, None, None, :]
        extended = (1.0 - extended) * torch.finfo(dtype).min
        return extended
    
    def forward(
        self,
        z_t: torch.Tensor,                           # (B, S, embedding_size)
        time_t: torch.Tensor,                         # (B,)
        attention_mask: Optional[torch.Tensor] = None, # (B, S)
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z_t: Noisy protein encodings [B, S, D].
            time_t: Diffusion timestep [B].
            attention_mask: Attention mask [B, S].
            
        Returns:
            Logits over families [B, num_families].
        """
        # Project input
        x = self.input_proj(z_t)
        
        # Add positional embeddings
        seq_len = x.size(1)
        pos_ids = self.position_ids[:, :seq_len]
        x = x + self.position_embeddings(pos_ids)
        
        # Time embedding
        t_emb = self.time_emb(timestep_embedding(time_t, self.config.hidden_size))
        t_emb = t_emb[:, None, :]  # (B, 1, D)
        
        # Extended attention mask
        if attention_mask is not None:
            ext_mask = self.get_extended_attention_mask(attention_mask, x.dtype)
        else:
            ext_mask = None
        
        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x = x + self.time_layers[i](t_emb)
            x = block(hidden_states=x, attention_mask=ext_mask)
        
        # Mean pooling over sequence dimension (masked)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            x_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x_pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x_pooled)
        return logits
