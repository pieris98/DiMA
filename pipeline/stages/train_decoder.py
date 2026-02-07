"""
Stage: Decoder training.

Trains a transformer decoder for improved reconstruction from
noisy latent embeddings. Optional — can be skipped if the encoder
has a built-in decoder (e.g., CHEAP).
"""

import os
import subprocess
import sys
from typing import Dict, Any

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("train_decoder")
class TrainDecoderStage(BaseStage):
    name = "train_decoder"
    description = "Train decoder for improved reconstruction (optional)"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        dec = config.decoder if hasattr(config, "decoder") else config.get("decoder", {})
        decoder_type = dec.decoder_type if hasattr(dec, "decoder_type") else dec.get("decoder_type", "default")

        if decoder_type == "default":
            issues.append("Decoder type is 'default' (lm_head). Decoder training not applicable — skip this stage.")

        # Check that statistics exist
        stats_path = context.get("statistics_path")
        if stats_path and not os.path.exists(stats_path):
            issues.append(f"Statistics not found at {stats_path}. Run calculate_statistics first.")

        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        dec = config.decoder if hasattr(config, "decoder") else config.get("decoder", {})
        decoder_type = dec.decoder_type if hasattr(dec, "decoder_type") else dec.get("decoder_type", "default")
        decoder_path = dec.decoder_path if hasattr(dec, "decoder_path") else dec.get("decoder_path", "")

        if decoder_type == "default":
            print("[train_decoder] Decoder type is 'default', skipping training.")
            context["decoder_path"] = None
            return context

        if decoder_path and os.path.exists(decoder_path):
            print(f"[train_decoder] Decoder checkpoint already exists at {decoder_path}, skipping.")
            context["decoder_path"] = decoder_path
            return context

        print(f"[train_decoder] Training {decoder_type} decoder...")

        config_path = context.get("config_path", "../configs")
        cmd = [
            sys.executable, "-m", "src.preprocessing.train_decoder",
            f"--config_path={config_path}",
        ]
        print(f"[train_decoder] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[train_decoder] STDERR: {result.stderr}")
            raise RuntimeError(f"Decoder training failed: {result.stderr}")

        print(result.stdout)
        context["decoder_path"] = decoder_path
        print("[train_decoder] Complete.")
        return context
