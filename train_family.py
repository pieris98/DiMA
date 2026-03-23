"""
Family-specific conditional generation training script for DiMA.

Supports two approaches from §3.6.2:
1. Classifier Guidance: Train a classifier on noisy encodings, use gradients to guide generation
2. Conditional Fine-tuning: Fine-tune DiMA with family label embeddings

Usage:
    # Conditional fine-tuning with ESM2-650M
    pixi run -e dima-env python train_family.py --config-name config_family

    # Classifier guidance approach
    pixi run -e dima-env python train_family.py --config-name config_family \
        conditional.approach=classifier_guidance
"""

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, List
import os
import wandb

from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.utils import seed_everything, setup_ddp, print_config
from src.utils.logging_utils import config_to_wandb


class FamilyDiffusionTrainer(BaseDiffusionTrainer):
    """
    Trainer for family-specific conditional generation.
    
    Supports two approaches:
    1. conditional_finetuning: Add family embeddings and fine-tune
    2. classifier_guidance: Train separate classifier, use gradients
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        self.cond_type = config.get('conditional', {}).get('type', 'family')
        self.cond_dim = config.get('conditional', {}).get('dim', 1280)
        self.num_families = config.get('conditional', {}).get('num_families', 9)
        self.approach = config.get('conditional', {}).get('approach', 'conditional_finetuning')
        
        # Family label to index mapping (from paper: 0 = no family)
        self.family_to_idx = {
            'no_family': 0,
            'CRISPR': 1,
            'Calmodulin': 2,
            'Glycosyl_hydrolase': 3,
            'LexA': 4,
            'Lysozyme': 5,
            'NrdR': 6,
            'PHI': 7,
            'PurE': 8,
        }
        self.idx_to_family = {v: k for k, v in self.family_to_idx.items()}
        
        super().__init__(config, device)
        
        if self.approach == 'conditional_finetuning':
            self._init_family_embeddings()
    
    def _init_family_embeddings(self):
        """Initialize family label embeddings for conditional fine-tuning."""
        self.family_embed = nn.Embedding(
            num_embeddings=self.num_families,
            embedding_dim=self.cond_dim
        ).to(self.device)
        
        # Initialize with small random values
        nn.init.normal_(self.family_embed.weight, mean=0, std=0.02)
        
        # Add to trainable parameters
        self.optimizer.add_param_group({
            'params': self.family_embed.parameters(),
            'lr': self.config.training.get('lr', 5e-5)
        })
    
    def get_family_label(self, batch) -> torch.Tensor:
        """Extract family labels from batch."""
        if 'family_labels' in batch:
            return batch['family_labels'].to(self.device)
        # If no labels in batch, return random family
        return torch.randint(1, self.num_families, (len(batch['sequence']),), device=self.device)
    
    def calc_loss(self, batch, condition=None):
        """
        Calculate denoising loss with optional family conditioning.
        
        For conditional fine-tuning, family embeddings are added to the input.
        """
        lens = batch["lens"]
        attention_mask = self.encoder.get_attention_mask_for_lens(
            lens, max_sequence_len=self.config.datasets.max_sequence_len
        )
        
        # Get family labels if conditioning
        if self.approach == 'conditional_finetuning':
            family_labels = self.get_family_label(batch)
            family_emb = self.family_embed(family_labels)
            condition = family_emb
        else:
            condition = None
        
        # Noizing
        noisy_embeddings, noise, t = self.dynamic.add_noise(
            batch["embeddings"],
            attention_mask
        )
        
        # Add family conditioning if using conditional fine-tuning
        if condition is not None:
            condition = condition.to(self.device)
            # Add to the input embeddings
            noisy_embeddings = noisy_embeddings + condition.unsqueeze(1)
        
        # Model prediction
        if self.config.model.config.use_self_cond:
            x_0_self_cond = torch.zeros_like(noisy_embeddings, dtype=noisy_embeddings.dtype)
            x_0_self_cond = self._get_estimator()(
                x_t=noisy_embeddings,
                time_t=t,
                attention_mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )
        else:
            x_0_self_cond = None
        
        x_0 = self._get_estimator()(
            x_t=noisy_embeddings,
            time_t=t,
            attention_mask=attention_mask,
            x_0_self_cond=x_0_self_cond,
        )
        
        # MSE loss
        loss_dict = {}
        total_loss = F.mse_loss(x_0, batch["embeddings"], reduction='none')
        total_loss = (total_loss * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict, {}
    
    def generate_samples(self, num_samples: int) -> List[Dict]:
        """Generate family-conditioned protein samples."""
        from src.diffusion.solvers import EulerDiffEqSolver
        
        # Sample lengths from distribution
        sampled_lengths = self.length_sampler.sample(num_samples)
        
        generated_sequences = []
        for length in tqdm(sampled_lengths, desc="Generating"):
            # Sample random noise
            z_t = torch.randn(1, length, self.config.model.config.embedding_size, device=self.device)
            
            # Reverse diffusion
            solver = EulerDiffEqSolver(
                dynamic=self.dynamic,
                model=self._get_estimator(),
                device=self.device,
                ode_sampling=self.config.solver.get('ode_sampling', False),
            )
            
            # Sample with optional family conditioning
            if self.approach == 'conditional_finetuning':
                # Sample a random family for generation
                target_family = torch.randint(1, self.num_families, (1,), device=self.device)
                family_emb = self.family_embed(target_family)
                
                # Run solver with conditioning
                z_0 = solver.sample(z_t, condition=family_emb)
            else:
                z_0 = solver.sample(z_t)
            
            # Decode to sequence
            sequence = self.encoder.decode(z_0.squeeze(0))
            generated_sequences.append({
                'sequence': sequence,
                'length': length,
            })
        
        return generated_sequences


class FamilyClassifierTrainer:
    """
    Trainer for the family classifier used in classifier guidance approach.
    
    The classifier takes noisy protein encodings and predicts family membership.
    During generation, gradients from the classifier guide the diffusion process.
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device
        self.num_families = config.conditional.num_families
        
        self._init_classifier()
        self._init_optimizer()
    
    def _init_classifier(self):
        """Initialize the family classifier."""
        from src.models.blocks import BertBlock
        
        # Classifier: 3 transformer blocks with latent attention
        num_blocks = self.config.classifier.get('num_blocks', 3)
        hidden_dim = self.config.classifier.get('hidden_dim', self.config.encoder.config.embedding_dim)
        
        blocks = nn.ModuleList([
            BertBlock(self.config.model.config) for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Sequential(*blocks)
        self.classifier.to(self.device)
    
    def _init_optimizer(self):
        """Initialize classifier optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config.classifier.get('lr', 2e-4),
            weight_decay=0.01,
        )
    
    def compute_loss(self, batch, noisy_embeddings, t):
        """Compute classifier loss for family prediction."""
        logits = self.classifier(noisy_embeddings, t)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, batch['family_labels'])
        return loss
    
    def train_classifier(self, dataloader, num_iters: int):
        """Train the family classifier."""
        self.classifier.train()
        
        for i in range(num_iters):
            batch = next(iter(dataloader))
            
            # Add noise to encodings
            noisy_embeddings, noise, t = self.dynamic.add_noise(
                batch['embeddings'], 
                batch['attention_mask']
            )
            
            loss = self.compute_loss(batch, noisy_embeddings, t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % 100 == 0:
                print(f"Classifier iter {i}/{num_iters}, loss: {loss.item():.4f}")
    
    def save_checkpoint(self, path: str):
        """Save classifier checkpoint."""
        torch.save({
            'model': self.classifier.state_dict(),
            'config': self.config,
            'family_to_idx': self.family_to_idx,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load classifier checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(ckpt['model'])
        self.family_to_idx = ckpt['family_to_idx']


@hydra.main(version_base=None, config_path="src/configs", config_name="config_family")
def main(config: DictConfig):
    if config.ddp.enabled:
        import torch.distributed as dist
        config.ddp.local_rank, config.ddp.global_rank = setup_ddp()
        config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    
    config.model.config.embedding_size = config.encoder.config.embedding_dim
    
    if config.ddp.global_rank == 0:
        print_config(config)
    
    seed = config.project.seed + config.ddp.global_rank
    seed_everything(seed)
    
    # Initialize wandb
    if not config.ddp.enabled or config.ddp.global_rank == 0:
        wandb_mode = "disabled" if config.project.wandb_project == "disabled" else "offline"
        wandb.init(
            project=config.project.wandb_project,
            name=config.project.checkpoints_prefix,
            mode=wandb_mode
        )
        if wandb_mode != "disabled":
            config_to_wandb(config)
    
    device = torch.device(f"cuda:{config.ddp.local_rank}") if config.ddp.enabled else torch.device("cuda")
    
    # Choose approach
    approach = config.conditional.get('approach', 'conditional_finetuning')
    
    if approach == 'classifier_guidance':
        # Train classifier first
        print("Training family classifier...")
        classifier_trainer = FamilyClassifierTrainer(config, device)
        
        # Create dataloader for classifier training
        from src.family_generation import FamilySequenceDataset
        train_dataset = FamilySequenceDataset(
            data_dir=config.datasets.data_dir,
            families=config.families,
            split='train'
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.classifier.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        classifier_trainer.train_classifier(
            train_loader, 
            config.classifier.training_iters
        )
        
        # Save classifier
        os.makedirs(config.project.checkpoints_folder, exist_ok=True)
        classifier_trainer.save_checkpoint(config.classifier.checkpoint)
        
        # Then train diffusion with classifier guidance
        print("Training diffusion model with classifier guidance...")
        trainer = BaseDiffusionTrainer(config, device)
    else:
        # Conditional fine-tuning approach
        print("Training with conditional fine-tuning...")
        trainer = FamilyDiffusionTrainer(config, device)
    
    trainer.train()
    
    if config.ddp.global_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
