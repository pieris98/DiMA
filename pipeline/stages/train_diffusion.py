"""
Stage: Diffusion model training.

Trains the score estimator (denoising model) using the latent diffusion
framework. Supports multi-GPU training via torchrun.
"""

import os
import subprocess
import sys
from typing import Dict, Any

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("train_diffusion")
class TrainDiffusionStage(BaseStage):
    name = "train_diffusion"
    description = "Train the diffusion model (score estimator)"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        # Check data exists
        ds = config.datasets if hasattr(config, "datasets") else config.get("datasets", {})
        data_dir = ds.data_dir if hasattr(ds, "data_dir") else ds.get("data_dir", "")
        if data_dir and not os.path.exists(os.path.join(data_dir, "train")):
            issues.append(f"Training data not found at {data_dir}/train.")

        # Check statistics exist
        stats_path = context.get("statistics_path")
        if stats_path and not os.path.exists(stats_path):
            issues.append(f"Statistics not found at {stats_path}.")

        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        training = config.training if hasattr(config, "training") else config.get("training", {})
        ddp = config.ddp if hasattr(config, "ddp") else config.get("ddp", {})
        ddp_enabled = ddp.enabled if hasattr(ddp, "enabled") else ddp.get("enabled", False)

        project = config.project if hasattr(config, "project") else config["project"]
        ckpt_folder = project.diffusion_checkpoints_folder if hasattr(project, "diffusion_checkpoints_folder") else project["diffusion_checkpoints_folder"]

        # Get number of GPUs from context or default
        num_gpus = context.get("num_gpus", 1)
        master_port = context.get("master_port", 31345)

        print(f"[train_diffusion] Starting diffusion training (GPUs={num_gpus})...")

        if ddp_enabled and num_gpus > 1:
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                f"--master_port={master_port}",
                "train_diffusion.py",
            ]
        else:
            cmd = [sys.executable, "train_diffusion.py"]

        # Add any Hydra overrides from context
        hydra_overrides = context.get("hydra_overrides", [])
        cmd.extend(hydra_overrides)

        print(f"[train_diffusion] Running: {' '.join(cmd)}")
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"[train_diffusion] STDOUT: {result.stdout[-2000:]}")
            print(f"[train_diffusion] STDERR: {result.stderr[-2000:]}")
            raise RuntimeError(f"Diffusion training failed with exit code {result.returncode}")

        print(result.stdout[-1000:])
        context["diffusion_checkpoints_folder"] = ckpt_folder
        print("[train_diffusion] Complete.")
        return context
