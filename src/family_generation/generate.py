"""
Family-specific conditional generation using DiMA.

Implements two approaches described in §3.6.2:

1. **Classifier Guidance**: Uses gradients from a separately trained family
   classifier to steer the diffusion denoising process toward a target family.
   No fine-tuning of the diffusion model is needed.

2. **Conditional Fine-tuning (CFG)**: Augments DiMA with family class label
   embeddings and fine-tunes on all families simultaneously using
   classifier-free guidance.

Both approaches modify the reverse diffusion sampling to condition on a
target protein family.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.family_classifier import FamilyClassifier, FamilyClassifierConfig
from src.diffusion.dima import DiMAModel
from src.utils.hydra_utils import setup_config


def load_family_classifier(
    checkpoint_path: str, 
    device: torch.device,
) -> tuple:
    """
    Load a trained family classifier from checkpoint.
    
    Returns:
        (classifier, family_to_idx) tuple.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    clf_config_dict = ckpt["config"]
    clf_config = FamilyClassifierConfig(**clf_config_dict)
    
    classifier = FamilyClassifier(clf_config)
    classifier.load_state_dict(ckpt["model"])
    classifier = classifier.to(device).eval()
    
    family_to_idx = ckpt["family_to_idx"]
    
    return classifier, family_to_idx


class ClassifierGuidedSampler:
    """
    Classifier-guided diffusion sampling for family-specific generation.
    
    At each denoising step, computes the gradient of log p(family | z_t)
    with respect to z_t and adds it (scaled) to the denoising prediction.
    
    This follows the classifier guidance approach from Dhariwal & Nichol (2021),
    adapted to the protein latent diffusion setting.
    """
    
    def __init__(
        self,
        model: DiMAModel,
        classifier: FamilyClassifier,
        family_to_idx: Dict[str, int],
        guidance_scale: float = 5.0,
    ):
        """
        Args:
            model: Pre-trained DiMA model.
            classifier: Trained family classifier on noisy encodings.
            family_to_idx: Mapping from family name to class index.
            guidance_scale: Strength of classifier guidance (higher = stronger).
        """
        self.model = model
        self.classifier = classifier
        self.family_to_idx = family_to_idx
        self.idx_to_family = {v: k for k, v in family_to_idx.items()}
        self.guidance_scale = guidance_scale
    
    def compute_classifier_gradient(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        target_family_idx: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ∇_{z_t} log p(family | z_t, t).
        
        Args:
            z_t: Noisy latent vectors [B, S, D].
            t: Timestep [B].
            target_family_idx: Index of target family.
            attention_mask: Attention mask [B, S].
            
        Returns:
            Gradient tensor of shape [B, S, D].
        """
        z_t_in = z_t.detach().requires_grad_(True)
        
        logits = self.classifier(z_t_in, t, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Select log probability of target family
        target = torch.full((z_t_in.shape[0],), target_family_idx, 
                           dtype=torch.long, device=z_t_in.device)
        log_prob = log_probs[torch.arange(z_t_in.shape[0]), target].sum()
        
        # Compute gradient
        grad = torch.autograd.grad(log_prob, z_t_in)[0]
        
        return grad
    
    @torch.no_grad()
    def generate_family_sequences(
        self,
        family_name: str,
        num_sequences: int = 100,
        guidance_scale: Optional[float] = None,
    ) -> List[str]:
        """
        Generate protein sequences conditioned on a target family
        using classifier guidance.
        
        Args:
            family_name: Name of target family (must be in family_to_idx).
            num_sequences: Number of sequences to generate.
            guidance_scale: Override guidance scale. None = use default.
            
        Returns:
            List of generated amino acid sequences.
        """
        if family_name not in self.family_to_idx:
            raise ValueError(
                f"Unknown family '{family_name}'. "
                f"Available: {list(self.family_to_idx.keys())}"
            )
        
        target_idx = self.family_to_idx[family_name]
        scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        
        config = self.model.config
        results = []
        
        while len(results) < num_sequences:
            batch_size = min(config.generation.batch_size, num_sequences - len(results))
            
            # Sample lengths and create attention mask
            lens = self.model.length_sampler.sample(batch_size)
            attention_mask = self.model.encoder.get_attention_mask_for_lens(
                lens, max_sequence_len=config.datasets.max_sequence_len
            )
            
            # Generate embeddings with classifier guidance
            pred_embeddings = self._guided_pred_embeddings(
                attention_mask=attention_mask,
                target_family_idx=target_idx,
                scale=scale,
            )
            
            # Decode to sequences
            sequences = self.model.pred_logits(pred_embeddings, attention_mask=attention_mask)
            results.extend(sequences)
        
        return results[:num_sequences]
    
    def _guided_pred_embeddings(
        self,
        attention_mask: torch.Tensor,
        target_family_idx: int,
        scale: float,
    ) -> torch.Tensor:
        """
        Run reverse diffusion with classifier guidance at each step.
        """
        config = self.model.config
        device = self.model.device
        
        shape = (
            attention_mask.shape[0],
            attention_mask.shape[1],
            config.model.config.embedding_size,
        )
        
        x = torch.randn(shape, device=device)
        x_0_self_cond = torch.zeros_like(x)
        eps_t = config.generation.t_min
        
        timesteps = torch.linspace(
            self.model.dynamic.T, eps_t, config.generation.N_steps + 1, device=device
        )
        
        for idx in tqdm(range(config.generation.N_steps), desc="Guided sampling", leave=False):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]
            
            input_t = t * torch.ones(shape[0], device=device)
            next_input_t = next_t * torch.ones(shape[0], device=device)
            
            # Standard denoising step (in no_grad context)
            with torch.no_grad():
                output = self.model.solver.step(
                    x_t=x, t=input_t, next_t=next_input_t,
                    mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                )
                x_denoised = output["x"]
                x_0_self_cond = output["x_0"]
            
            # Classifier guidance: add gradient of log p(family | x_t)
            # We need to enable grad temporarily for this
            with torch.enable_grad():
                grad = self.compute_classifier_gradient(
                    z_t=x_denoised, 
                    t=input_t,
                    target_family_idx=target_family_idx,
                    attention_mask=attention_mask,
                )
            
            # Apply guidance
            x = x_denoised + scale * grad
        
        return x


class ConditionalDiMAModel(DiMAModel):
    """
    DiMA model augmented with family label conditioning for classifier-free guidance.
    
    Extends the base DiMA model by:
    1. Adding a learnable family embedding table.
    2. Injecting family embeddings into each transformer block.
    3. Supporting classifier-free guidance during inference.
    
    During training, the family label is dropped (replaced with a null label)
    with probability `cfg_drop_rate` to enable classifier-free guidance.
    """
    
    def __init__(
        self, 
        config_path: str, 
        device: torch.device,
        num_families: int = 8,
        cfg_drop_rate: float = 0.1,
        overrides=None,
    ):
        super().__init__(config_path, device, overrides=overrides)
        
        self.num_families = num_families
        self.cfg_drop_rate = cfg_drop_rate
        hidden_size = self.config.model.config.hidden_size
        
        # Family embedding: num_families + 1 (the +1 is null/unconditional class)
        self.family_embedding = nn.Embedding(num_families + 1, hidden_size).to(device)
        self.null_family_idx = num_families  # Last index = unconditional
        
        # Projection for integrating family conditioning
        self.family_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(device)
    
    def generate_family_samples(
        self,
        family_idx: int,
        num_texts: int,
        cfg_scale: float = 3.0,
    ) -> List[str]:
        """
        Generate sequences conditioned on a family label using
        classifier-free guidance.
        
        Args:
            family_idx: Family class index.
            num_texts: Number of sequences to generate.
            cfg_scale: Classifier-free guidance scale.
            
        Returns:
            List of amino acid sequences.
        """
        results = []
        
        while len(results) < num_texts:
            batch_size = min(self.config.generation.batch_size, num_texts - len(results))
            
            lens = self.length_sampler.sample(batch_size)
            attention_mask = self.encoder.get_attention_mask_for_lens(
                lens, max_sequence_len=self.config.datasets.max_sequence_len
            )
            
            with torch.no_grad():
                pred_emb = self._cfg_pred_embeddings(
                    attention_mask=attention_mask,
                    family_idx=family_idx,
                    cfg_scale=cfg_scale,
                )
                sequences = self.pred_logits(pred_emb, attention_mask=attention_mask)
            
            results.extend(sequences)
        
        return results[:num_texts]
    
    def _cfg_pred_embeddings(
        self,
        attention_mask: torch.Tensor,
        family_idx: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Reverse diffusion with classifier-free guidance.
        
        CFG formula: eps_guided = (1 + scale) * eps_cond - scale * eps_uncond
        Equivalently for x_0 prediction:
            x_0_guided = (1 + scale) * x_0_cond - scale * x_0_uncond
        """
        config = self.config
        device = self.device
        
        shape = (
            attention_mask.shape[0],
            attention_mask.shape[1],
            config.model.config.embedding_size,
        )
        
        x = torch.randn(shape, device=device)
        x_0_self_cond = torch.zeros_like(x)
        eps_t = config.generation.t_min
        
        timesteps = torch.linspace(
            self.dynamic.T, eps_t, config.generation.N_steps + 1, device=device
        )
        
        # Family conditioning embeddings
        cond_idx = torch.full((shape[0],), family_idx, dtype=torch.long, device=device)
        uncond_idx = torch.full((shape[0],), self.null_family_idx, dtype=torch.long, device=device)
        
        cond_emb = self.family_proj(self.family_embedding(cond_idx))[:, None, :]  # (B, 1, D)
        uncond_emb = self.family_proj(self.family_embedding(uncond_idx))[:, None, :]
        
        for idx in tqdm(range(config.generation.N_steps), desc="CFG sampling", leave=False):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]
            input_t = t * torch.ones(shape[0], device=device)
            
            # Conditional prediction
            x_cond = x + cond_emb
            output_cond = self.solver.step(
                x_t=x_cond, t=input_t,
                next_t=next_t * torch.ones(shape[0], device=device),
                mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )
            
            # Unconditional prediction
            x_uncond = x + uncond_emb
            output_uncond = self.solver.step(
                x_t=x_uncond, t=input_t,
                next_t=next_t * torch.ones(shape[0], device=device),
                mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )
            
            # CFG combination
            x_0_cond = output_cond["x_0"]
            x_0_uncond = output_uncond["x_0"]
            x_0_guided = (1 + cfg_scale) * x_0_cond - cfg_scale * x_0_uncond
            
            x = output_cond["x"] + cfg_scale * (output_cond["x"] - output_uncond["x"])
            x_0_self_cond = x_0_guided
        
        return x


# ──────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Family-specific generation with DiMA")
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--mode", choices=["classifier_guidance", "cfg"], required=True,
                        help="Generation mode: classifier_guidance or cfg (classifier-free guidance)")
    parser.add_argument("--family", type=str, required=True,
                        help="Target protein family name")
    parser.add_argument("--num_sequences", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Path to classifier checkpoint (required for classifier_guidance mode)")
    parser.add_argument("--output_path", type=str, default="generated_family_sequences.json")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "classifier_guidance":
        if args.classifier_ckpt is None:
            raise ValueError("--classifier_ckpt is required for classifier_guidance mode")
        
        # Load DiMA model
        model = DiMAModel(config_path=args.config_path, device=device)
        model.load_pretrained()
        model.score_estimator.eval()
        
        # Load classifier
        classifier, family_to_idx = load_family_classifier(args.classifier_ckpt, device)
        
        # Generate
        sampler = ClassifierGuidedSampler(
            model=model,
            classifier=classifier,
            family_to_idx=family_to_idx,
            guidance_scale=args.guidance_scale,
        )
        
        sequences = sampler.generate_family_sequences(
            family_name=args.family,
            num_sequences=args.num_sequences,
            guidance_scale=args.guidance_scale,
        )
    
    elif args.mode == "cfg":
        # For CFG mode, the model would need to have been fine-tuned
        # with family embeddings. This requires a separate training run.
        raise NotImplementedError(
            "CFG mode requires a fine-tuned ConditionalDiMAModel. "
            "Use train_conditional_dima.py first."
        )
    
    # Save results
    output = {
        "family": args.family,
        "mode": args.mode,
        "guidance_scale": args.guidance_scale,
        "num_sequences": len(sequences),
        "sequences": sequences,
    }
    
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Generated {len(sequences)} sequences for family '{args.family}'")
    print(f"Saved to {args.output_path}")
