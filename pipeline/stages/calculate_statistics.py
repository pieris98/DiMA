"""
Stage: Calculate normalization statistics for encoder embeddings.

Computes mean and std of encoder representations on training data,
used for normalizing latent space during diffusion training.
"""

import os
import subprocess
import sys
from typing import Dict, Any

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("calculate_statistics")
class CalculateStatisticsStage(BaseStage):
    name = "calculate_statistics"
    description = "Compute normalization statistics for encoder embeddings"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        data_dir = context.get("data_dir")
        if data_dir is None:
            # Try config
            ds = config.datasets if hasattr(config, "datasets") else config.get("datasets", {})
            data_dir = ds.data_dir if hasattr(ds, "data_dir") else ds.get("data_dir", "")
        if data_dir and not os.path.exists(os.path.join(data_dir, "train")):
            issues.append(f"Training data not found at {data_dir}/train. Run setup_data first.")
        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        project = config.project if hasattr(config, "project") else config["project"]
        stats_folder = project.statistics_folder if hasattr(project, "statistics_folder") else project["statistics_folder"]
        enc = config.encoder if hasattr(config, "encoder") else config["encoder"]
        enc_cfg = enc.config if hasattr(enc, "config") else enc["config"]
        encoder_type = enc_cfg.encoder_type if hasattr(enc_cfg, "encoder_type") else enc_cfg["encoder_type"]
        stats_path = os.path.join(stats_folder, f"encodings-{encoder_type}.pth")

        if os.path.exists(stats_path):
            print(f"[calculate_statistics] Statistics already exist at {stats_path}, skipping.")
            context["statistics_path"] = stats_path
            return context

        print(f"[calculate_statistics] Computing statistics for encoder '{encoder_type}'...")
        os.makedirs(stats_folder, exist_ok=True)

        # Delegate to existing script
        config_path = context.get("config_path", "../configs")
        cmd = [
            sys.executable, "-m", "src.preprocessing.calculate_statistics",
            f"--config_path={config_path}",
        ]
        print(f"[calculate_statistics] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[calculate_statistics] STDERR: {result.stderr}")
            raise RuntimeError(f"Statistics calculation failed: {result.stderr}")

        print(result.stdout)
        context["statistics_path"] = stats_path
        print("[calculate_statistics] Complete.")
        return context
