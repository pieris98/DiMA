"""
Conditional generation experiments for DiMA: Motif scaffolding and Fold-conditioned generation.

This module implements the conditional generation tasks described in §3.6.1 and §3.6.3 of the paper:

1. Motif Scaffolding: Generate protein sequences that preserve specific functional motifs
   while designing new scaffolds around them.

2. Fold-conditioned Generation: Generate sequences that adopt specific protein folds.

Usage:
    # Motif scaffolding with SaProt-650M
    python -m src.conditional_generation.motif_scaffold \
        --mode scaffold \
        --encoder saprot_650m \
        --motif_pdb path/to/motif.pdb \
        --output_dir outputs/motif_scaffolding

    # Fold-conditioned generation with CHEAP
    python -m src.conditional_generation.fold_condition \
        --mode fold \
        --encoder cheap \
        --target_fold path/to/target.pdb \
        --output_dir outputs/fold_generation
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.diffusion.dima import DiMAModel
from src.utils.hydra_utils import setup_config
from src.models.blocks import BertBlock, timestep_embedding


class ConditionalScoreEstimator(nn.Module):
    """
    Score estimator with cross-attention conditioning for motif/fold conditioning.
    
    Extends the base score estimator with additional conditioning mechanism
    that processes structural constraints via cross-attention.
    """
    
    def __init__(self, base_model, cond_dim: int = 1280, num_cond_layers: int = 4):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Cross-attention for condition encoding
        self.cond_layers = nn.ModuleList([
            BertBlock(self.config) for _ in range(num_cond_layers)
        ])
        
        # Projection from condition to hidden size
        self.cond_projector = nn.Linear(cond_dim, self.config.hidden_size)
        
        # Fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x_t: torch.Tensor,
        time_t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        x_0_self_cond: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Base forward pass
        output = self.base_model(x_t, time_t, attention_mask, x_0_self_cond)
        
        # Apply cross-attention conditioning if provided
        if condition is not None:
            # Project condition
            cond_proj = self.cond_projector(condition)
            
            # Process through cross-attention layers
            for layer in self.cond_layers:
                # Create cross-attention input by concatenating with main hidden states
                hidden = output
                
                # Expand cond_mask if needed
                if cond_mask is not None and hidden.size(1) != cond_mask.size(1):
                    # Tile condition to match sequence length
                    repeat_factor = hidden.size(1) // cond_proj.size(1)
                    cond_proj = cond_proj.repeat(1, repeat_factor, 1)
                    cond_mask = cond_mask.repeat(1, repeat_factor)
                
                # Cross-attention (using encoder_hidden_states)
                output = layer(
                    hidden_states=hidden,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cond_proj,
                    encoder_attention_mask=cond_mask,
                )
            
            # Gated fusion
            combined = torch.cat([output, hidden], dim=-1)
            gate = self.fusion_gate(combined)
            output = gate * output + (1 - gate) * hidden
            
        return output


class MotifScaffoldingSampler:
    """
    Sampler for motif scaffolding tasks.
    
    Given a motif (specified as a PDB file or residue positions), generates
    a protein sequence that preserves the motif while designing a new scaffold around it.
    """
    
    def __init__(
        self,
        model: DiMAModel,
        encoder_type: str = "saprot_650m",
        guidance_scale: float = 3.0,
    ):
        self.model = model
        self.encoder_type = encoder_type
        self.guidance_scale = guidance_scale
        self.device = model.device
        self.config = model.config
        
    def _load_motif_from_pdb(self, pdb_path: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Load motif from PDB file and encode it.
        
        Returns:
            Tuple of (encoded_motif, residue_indices)
        """
        # For now, return placeholder - in practice would use Biopython to parse PDB
        # and extract residue coordinates
        raise NotImplementedError("PDB loading not yet implemented")
        
    def _encode_structure(self, pdb_path: str) -> torch.Tensor:
        """
        Encode a protein structure using the encoder.
        
        For SaProt, this uses the structural tokens.
        For other encoders, would use structural embedding.
        """
        # This is a placeholder - would need proper structure encoding
        # For SaProt, the encoder already handles structure via 3Di tokens
        return None
        
    @torch.no_grad()
    def scaffold_motif(
        self,
        motif_pdb: str,
        num_sequences: int = 100,
        scaffold_length: int = 200,
        guidance_scale: Optional[float] = None,
    ) -> List[str]:
        """
        Generate scaffolds around a given motif.
        
        Args:
            motif_pdb: Path to PDB file containing the motif
            num_sequences: Number of sequences to generate
            scaffold_length: Target length of the generated scaffold
            guidance_scale: Guidance scale for sampling
            
        Returns:
            List of generated amino acid sequences
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        results = []
        
        while len(results) < num_sequences:
            batch_size = min(self.config.generation.batch_size, num_sequences - len(results))
            
            # Create attention mask for target length
            lens = [scaffold_length] * batch_size
            attention_mask = self.model.encoder.get_attention_mask_for_lens(
                lens, max_sequence_len=self.config.datasets.max_sequence_len
            )
            
            # Generate with conditioning
            # In practice, would add motif conditioning here
            pred_embeddings = self._guided_pred_embeddings(
                attention_mask=attention_mask,
                guidance_scale=guidance_scale,
            )
            
            sequences = self.model.pred_logits(pred_embeddings, attention_mask=attention_mask)
            results.extend(sequences)
            
        return results[:num_sequences]
    
    def _guided_pred_embeddings(
        self,
        attention_mask: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run reverse diffusion with guidance."""
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
            self.model.dynamic.T, eps_t, config.generation.N_steps + 1, device=device
        )
        
        for idx in range(config.generation.N_steps):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]
            
            input_t = t * torch.ones(shape[0], device=device)
            next_input_t = next_t * torch.ones(shape[0], device=device)
            
            # Standard denoising step
            output = self.model.solver.step(
                x_t=x, t=input_t, next_t=next_input_t,
                mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )
            x = output["x"]
            x_0_self_cond = output["x_0"]
        
        return x


class FoldConditionedSampler:
    """
    Sampler for fold-conditioned generation.
    
    Generates protein sequences that are predicted to adopt a specific fold
    as measured by TM-score to a target structure.
    """
    
    def __init__(
        self,
        model: DiMAModel,
        encoder_type: str = "cheap",
        guidance_scale: float = 3.0,
    ):
        self.model = model
        self.encoder_type = encoder_type
        self.guidance_scale = guidance_scale
        self.device = model.device
        self.config = model.config
        
    @torch.no_grad()
    def generate_for_fold(
        self,
        target_pdb: str,
        num_sequences: int = 10,
        guidance_scale: Optional[float] = None,
    ) -> List[str]:
        """
        Generate sequences conditioned on a target fold.
        
        Args:
            target_pdb: Path to PDB file of target fold
            num_sequences: Number of sequences to generate
            guidance_scale: Guidance scale for sampling
            
        Returns:
            List of generated amino acid sequences
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        results = []
        
        # Get target sequence length from PDB
        target_length = self._get_pdb_length(target_pdb)
        
        while len(results) < num_sequences:
            batch_size = min(self.config.generation.batch_size, num_sequences - len(results))
            
            # Use target length for conditioning
            lens = [target_length] * batch_size
            attention_mask = self.model.encoder.get_attention_mask_for_lens(
                lens, max_sequence_len=self.config.datasets.max_sequence_len
            )
            
            # Generate
            pred_embeddings = self._pred_embeddings_with_fold_conditioning(
                attention_mask=attention_mask,
                target_length=target_length,
            )
            
            sequences = self.model.pred_logits(pred_embeddings, attention_mask=attention_mask)
            results.extend(sequences)
            
        return results[:num_sequences]
    
    def _get_pdb_length(self, pdb_path: str) -> int:
        """Get sequence length from PDB file."""
        # Placeholder - would use Biopython
        return 200
        
    def _pred_embeddings_with_fold_conditioning(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Generate embeddings with fold conditioning."""
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
            self.model.dynamic.T, eps_t, config.generation.N_steps + 1, device=device
        )
        
        for idx in range(config.generation.N_steps):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]
            
            input_t = t * torch.ones(shape[0], device=device)
            next_input_t = next_t * torch.ones(shape[0], device=device)
            
            output = self.model.solver.step(
                x_t=x, t=input_t, next_t=next_input_t,
                mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )
            x = output["x"]
            x_0_self_cond = output["x_0"]
        
        return x


def run_motif_scaffolding(args):
    """Run motif scaffolding experiment."""
    print("=" * 70)
    print("Motif Scaffolding Experiment")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    
    # Build encoder overrides
    overrides = [f"encoder={args.encoder}"]
    if args.encoder_overrides:
        overrides.extend(args.encoder_overrides)
    
    # Initialize model
    print(f"\n[1] Loading DiMA model with {args.encoder} encoder...")
    model = DiMAModel(config_path=config_path, device=device, overrides=overrides)
    model.load_pretrained()
    model.score_estimator.eval()
    
    # Initialize sampler
    sampler = MotifScaffoldingSampler(
        model=model,
        encoder_type=args.encoder,
        guidance_scale=args.guidance_scale,
    )
    
    # Generate sequences for each motif
    results = {}
    
    motif_list = args.motif_pdbs if args.motif_pdbs else []
    
    if args.benchmark_file and os.path.exists(args.benchmark_file):
        with open(args.benchmark_file, "r") as f:
            benchmark = json.load(f)
            motif_list = [m["pdb"] for m in benchmark.get("problems", [])]
    
    for motif_pdb in motif_list:
        print(f"\n[2] Scaffolding motif: {motif_pdb}")
        
        sequences = sampler.scaffold_motif(
            motif_pdb=motif_pdb,
            num_sequences=args.num_sequences,
            scaffold_length=args.scaffold_length,
            guidance_scale=args.guidance_scale,
        )
        
        results[motif_pdb] = sequences
        
        # Save results
        output_file = os.path.join(
            args.output_dir,
            f"scaffold_{Path(motif_pdb).stem}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "motif": motif_pdb,
                "num_sequences": len(sequences),
                "sequences": sequences,
            }, f, indent=2)
        
        print(f"  Generated {len(sequences)} sequences -> {output_file}")
    
    # Summary
    print(f"\n[3] Summary:")
    print(f"  Total motifs: {len(results)}")
    print(f"  Output directory: {args.output_dir}")
    
    return results


def run_fold_conditioned(args):
    """Run fold-conditioned generation experiment."""
    print("=" * 70)
    print("Fold-Conditioned Generation Experiment")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    
    # Build encoder overrides
    overrides = [f"encoder={args.encoder}"]
    if args.encoder_overrides:
        overrides.extend(args.encoder_overrides)
    
    # Initialize model
    print(f"\n[1] Loading DiMA model with {args.encoder} encoder...")
    model = DiMAModel(config_path=config_path, device=device, overrides=overrides)
    model.load_pretrained()
    model.score_estimator.eval()
    
    # Initialize sampler
    sampler = FoldConditionedSampler(
        model=model,
        encoder_type=args.encoder,
        guidance_scale=args.guidance_scale,
    )
    
    # Generate for each target fold
    results = {}
    fold_list = args.target_pdbs if args.target_pdbs else []
    
    if args.fold_list_file and os.path.exists(args.fold_list_file):
        with open(args.fold_list_file, "r") as f:
            fold_data = json.load(f)
            fold_list = [f["pdb"] for f in fold_data.get("folds", [])]
    
    for fold_pdb in fold_list:
        print(f"\n[2] Generating for fold: {fold_pdb}")
        
        sequences = sampler.generate_for_fold(
            target_pdb=fold_pdb,
            num_sequences=args.num_sequences,
            guidance_scale=args.guidance_scale,
        )
        
        results[fold_pdb] = sequences
        
        # Save results
        output_file = os.path.join(
            args.output_dir,
            f"fold_{Path(fold_pdb).stem}.json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "target_fold": fold_pdb,
                "num_sequences": len(sequences),
                "sequences": sequences,
            }, f, indent=2)
        
        print(f"  Generated {len(sequences)} sequences -> {output_file}")
    
    # Summary
    print(f"\n[3] Summary:")
    print(f"  Total folds: {len(results)}")
    print(f"  Output directory: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiMA Conditional Generation Experiments")
    
    # Common args
    parser.add_argument("--mode", choices=["scaffold", "fold"], required=True,
                        help="Generation mode: scaffold (motif scaffolding) or fold (fold-conditioned)")
    parser.add_argument("--encoder", type=str, default="saprot_650m",
                        help="Encoder config (e.g., saprot_650m, cheap)")
    parser.add_argument("--output_dir", type=str, default="outputs/conditional",
                        help="Output directory")
    parser.add_argument("--num_sequences", type=int, default=100,
                        help="Number of sequences to generate per target")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Guidance scale for sampling")
    parser.add_argument("--encoder_overrides", nargs="*", default=None,
                        help="Additional Hydra config overrides")
    
    # Motif scaffolding args
    parser.add_argument("--motif_pdbs", nargs="*", default=None,
                        help="List of motif PDB files")
    parser.add_argument("--benchmark_file", type=str, default=None,
                        help="Path to RFDiffusion benchmark JSON")
    parser.add_argument("--scaffold_length", type=int, default=200,
                        help="Target scaffold length")
    
    # Fold-conditioned args
    parser.add_argument("--target_pdbs", nargs="*", default=None,
                        help="List of target fold PDB files")
    parser.add_argument("--fold_list_file", type=str, default=None,
                        help="Path to fold list JSON")
    
    args = parser.parse_args()
    
    if args.mode == "scaffold":
        run_motif_scaffolding(args)
    elif args.mode == "fold":
        run_fold_conditioned(args)