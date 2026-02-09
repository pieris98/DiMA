"""
Stage: Run inference / sample generation.

Generates protein sequences using a trained diffusion model.
"""

import os
import json
from typing import Dict, Any

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("run_inference")
class RunInferenceStage(BaseStage):
    name = "run_inference"
    description = "Generate protein sequences from the trained diffusion model"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        # Inference needs a checkpoint
        checkpoint_path = context.get("checkpoint_path")
        if checkpoint_path and not os.path.exists(checkpoint_path):
            issues.append(f"Checkpoint not found at {checkpoint_path}")
        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        gen = config.generation if hasattr(config, "generation") else config.get("generation", {})
        num_samples = gen.num_gen_samples if hasattr(gen, "num_gen_samples") else gen.get("num_gen_samples", 2048)
        save_dir = gen.save_dir if hasattr(gen, "save_dir") else gen.get("save_dir", "generated_sequences")

        print(f"[run_inference] Generating {num_samples} samples...")

        # Try to use DiMAModel for inference
        import torch
        from src.diffusion.dima import DiMAModel

        config_path = context.get("config_path", "../configs")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DiMAModel(config_path=config_path, device=device)

        # Load checkpoint
        checkpoint_path = context.get("checkpoint_path")
        if checkpoint_path:
            model.restore_checkpoint(checkpoint_path)
        else:
            # Try to load pretrained
            try:
                model.load_pretrained()
            except Exception as e:
                print(f"[run_inference] Warning: Could not load pretrained: {e}")

        model.score_estimator.eval()
        model.switch_to_ema()

        sequences = model.generate_samples(num_samples)

        # Save
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "generated_samples.json")
        with open(output_path, "w") as f:
            json.dump(sequences, f, indent=4)

        context["generated_sequences"] = sequences
        context["generated_sequences_path"] = output_path
        print(f"[run_inference] Saved {len(sequences)} sequences to {output_path}")
        print("[run_inference] Complete.")
        return context
