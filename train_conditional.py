"""
Conditional generation training script for DiMA.

Supports training conditional variants with:
- Motif scaffolding (conditioning on PDB structure)
- Fold-conditioned generation (conditioning on fold type)
- Family-conditioned generation (conditioning on protein family)

Usage:
    # Train fold-conditioned model with ESM-8M
    pixi run -e dima-env python train_conditional.py \
        mode=fold \
        encoder=esm2_8m

    # Train with custom config
    pixi run -e dima-env python train_conditional.py \
        mode=motif \
        encoder=esm2_35m \
        training.batch_size=128
"""

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.utils import seed_everything, setup_ddp, print_config
from src.utils.logging_utils import config_to_wandb


class ConditionalDiffusionTrainer(BaseDiffusionTrainer):
    """
    Extended trainer for conditional generation tasks.
    
    Adds cross-attention conditioning support for:
    - Motif scaffolding: condition on PDB structure
    - Fold-conditioned: condition on fold type (alpha, beta, alpha+beta, etc.)
    - Family-conditioned: condition on protein family
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        self.cond_type = config.get('conditional', {}).get('type', 'none')
        self.cond_dim = config.get('conditional', {}).get('dim', 0)
        
        super().__init__(config, device)
        
        if self.cond_type != 'none':
            self._init_conditioning_modules()
    
    def _init_conditioning_modules(self):
        """Initialize conditioning modules based on conditional type."""
        if self.cond_type == 'fold':
            self.fold_embed = torch.nn.Embedding(
                num_embeddings=10,
                embedding_dim=self.cond_dim
            ).to(self.device)
        elif self.cond_type == 'family':
            num_families = self.config.get('conditional', {}).get('num_families', 8)
            self.family_embed = torch.nn.Embedding(
                num_embeddings=num_families,
                embedding_dim=self.cond_dim
            ).to(self.device)
        elif self.cond_type == 'motif':
            pass
    
    def get_conditioning(self, batch, cond_type=None):
        """Get conditioning embeddings for the batch."""
        cond_type = cond_type or self.cond_type
        
        if cond_type == 'none':
            return None
        elif cond_type == 'fold':
            fold_labels = batch.get('fold_labels', torch.zeros(len(batch), dtype=torch.long, device=self.device))
            return self.fold_embed(fold_labels)
        elif cond_type == 'family':
            family_labels = batch.get('family_labels', torch.zeros(len(batch), dtype=torch.long, device=self.device))
            return self.family_embed(family_labels)
        elif cond_type == 'motif':
            return batch.get('motif_embeddings', None)
        return None
    
    def calc_loss(self, batch, condition=None):
        """
        Calculate denoising loss with optional conditioning.
        
        If condition is provided, the score estimator should use cross-attention
        to condition on the conditioning embeddings.
        """
        lens = batch["lens"]
        attention_mask = self.encoder.get_attention_mask_for_lens(
            lens, max_sequence_len=self.config.datasets.max_sequence_len
        )
        
        noisy_embeddings, noise, t = self.dynamic.add_noise(
            batch["embeddings"],
            attention_mask
        )
        
        if condition is not None:
            condition = condition.to(self.device)
        
        pred = self.score_estimator(
            noisy_embeddings,
            t,
            attention_mask=attention_mask,
            condition=condition
        )
        
        loss = self.mse_loss(pred, noise, attention_mask)
        
        return loss


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config: DictConfig):
    if config.ddp.enabled:
        config.ddp.local_rank, config.ddp.global_rank = setup_ddp()
        config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
        config.dataloader.batch_size = config.training.batch_size_per_gpu
    
    config.model.config.embedding_size = config.encoder.config.embedding_dim
    
    if config.ddp.global_rank == 0:
        print_config(config)
    
    seed = config.project.seed + config.ddp.global_rank
    seed_everything(seed)
    
    if not config.ddp.enabled or config.ddp.global_rank == 0:
        name = f"{config.project.checkpoints_prefix}-{config.conditional.type}"
        wandb.init(
            project=config.project.wandb_project,
            name=name,
            mode="online"
        )
        config_to_wandb(config)
    
    device = torch.device(f"cuda:{config.ddp.local_rank}") if config.ddp.enabled else torch.device("cuda")
    trainer = ConditionalDiffusionTrainer(config, device)
    trainer.train()
    
    if config.ddp.global_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
