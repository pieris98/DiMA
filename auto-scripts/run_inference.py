import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
import torch
import json
from src.diffusion.dima import DiMAModel
from src.utils.ddp_utils import seed_everything
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../src/configs", config_name="config")
def main(config):
    # Force single-GPU inference (no DDP needed)
    config.ddp.enabled = False
    config.ddp.global_rank = 0
    config.ddp.local_rank = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_everything(config.project.seed)

    print("Initializing model...")
    model = DiMAModel.__new__(DiMAModel)
    model.config = config
    model.device = device
    # Bootstrap base trainer without re-loading config
    from src.diffusion.base_trainer import BaseDiffusionTrainer
    BaseDiffusionTrainer.__init__(model, config, device)

    # Load checkpoint: explicit path > pretrained S3 download
    if hasattr(config, "checkpoint_path") and config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}...")
        model.restore_checkpoint(config.checkpoint_path)
        model.switch_to_ema()
    else:
        print("No checkpoint_path provided – loading pretrained weights via load_pretrained()...")
        from src.utils.pretrained_utils import PRETRAINED_MODELS_PATHS
        encoder_name = config.encoder.config.encoder_type
        if encoder_name in PRETRAINED_MODELS_PATHS:
            model.load_pretrained()
        else:
            raise ValueError(
                f"No pretrained weights for encoder '{encoder_name}'. "
                f"Pass ++checkpoint_path=/path/to/ckpt.pth on the command line."
            )

    print(f"Generating {config.generation.num_gen_samples} samples...")
    model.score_estimator.eval()

    sequences = model.generate_samples(config.generation.num_gen_samples)

    os.makedirs("auto-scripts", exist_ok=True)
    output_path = os.path.join("auto-scripts", "generated_samples.json")
    with open(output_path, "w") as f:
        json.dump(sequences, f, indent=4)
    print(f"Saved {len(sequences)} sequences to {output_path}")

if __name__ == "__main__":
    main()
