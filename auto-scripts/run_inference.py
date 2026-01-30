import hydra
import torch
import os
import json
from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.utils import seed_everything, setup_ddp, print_config
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../src/configs", config_name="config")
def main(config):
    # Disable DDP for simple inference script if running on single GPU/cpu for testing
    # But code expects ddp config
    
    # Adjust config for inference
    if not hasattr(config.ddp, "enabled"):
         config.ddp.enabled = False
    
    config.ddp.global_rank = 0
    config.ddp.local_rank = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set embedding size based on encoder
    config.model.config.embedding_size = config.encoder.config.embedding_dim

    seed_everything(config.project.seed)
    
    print("Initializing trainer...")
    trainer = BaseDiffusionTrainer(config, device)
    
    # Load checkpoint if provided
    # pass 'checkpoint_path' via command line override or config
    # e.g. ++checkpoint_path=/path/to/ckpt.pth
    if hasattr(config, "checkpoint_path") and config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}...")
        trainer.restore_checkpoint(config.checkpoint_path)
    elif config.training.init_se:
         print(f"Loading init_se from {config.training.init_se}...")
         trainer.init_checkpoint()
    
    print(f"Generating {config.generation.num_gen_samples} samples...")
    trainer.ddp_score_estimator.eval()
    if hasattr(trainer, "switch_to_ema"):
        trainer.switch_to_ema()
        
    sequences = trainer.generate_samples(config.generation.num_gen_samples)
    
    output_path = os.path.join("auto-scripts", "generated_samples.json")
    with open(output_path, "w") as f:
        json.dump(sequences, f, indent=4)
    print(f"Saved generated sequences to {output_path}")

if __name__ == "__main__":
    main()
