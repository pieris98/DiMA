"""
Stage: Model/encoder download and caching.

Ensures that encoder models (ESM2, SaProt, CHEAP, etc.) are downloaded
and cached locally before training begins.
"""

import os
from typing import Dict, Any

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("setup_models")
class SetupModelsStage(BaseStage):
    name = "setup_models"
    description = "Download and cache encoder models"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        if not hasattr(config, "encoder") and "encoder" not in config:
            issues.append("No 'encoder' section in config")
        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        encoder_config = config.encoder if hasattr(config, "encoder") else config["encoder"]
        enc_cfg = encoder_config.config if hasattr(encoder_config, "config") else encoder_config["config"]
        encoder_type = enc_cfg.encoder_type if hasattr(enc_cfg, "encoder_type") else enc_cfg["encoder_type"]
        model_name = enc_cfg.encoder_model_name if hasattr(enc_cfg, "encoder_model_name") else enc_cfg.get("encoder_model_name", "")

        print(f"[setup_models] Setting up encoder: {encoder_type}")

        if "ESM2" in encoder_type:
            self._setup_esm2(model_name)
        elif "SaProt" in encoder_type.lower() or "saprot" in encoder_type.lower():
            self._setup_saprot(model_name)
        elif "CHEAP" in encoder_type:
            self._setup_cheap(enc_cfg)
        else:
            # Try registry for custom encoders
            if registry.has("encoder", encoder_type):
                print(f"[setup_models] Custom encoder '{encoder_type}' found in registry. Skipping auto-setup.")
            else:
                print(f"[setup_models] Warning: Unknown encoder type '{encoder_type}', skipping setup.")

        context["encoder_type"] = encoder_type
        context["encoder_model_name"] = model_name
        print("[setup_models] Complete.")
        return context

    def _setup_esm2(self, model_name: str):
        from transformers import EsmTokenizer, EsmForMaskedLM
        print(f"[setup_models] Downloading/caching ESM2: {model_name}")
        EsmTokenizer.from_pretrained(model_name)
        EsmForMaskedLM.from_pretrained(model_name)
        print("[setup_models] ESM2 ready.")

    def _setup_saprot(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel
        print(f"[setup_models] Downloading/caching SaProt: {model_name}")
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("[setup_models] SaProt ready.")

    def _setup_cheap(self, enc_cfg):
        from cheap.pretrained import load_pretrained_model, CHECKPOINT_DIR_PATH
        encoder_type = enc_cfg.encoder_type if hasattr(enc_cfg, "encoder_type") else enc_cfg["encoder_type"]
        parts = encoder_type.split("_")
        shorten_factor = int(parts[2])
        channel_dimension = int(parts[4])
        print(f"[setup_models] Loading CHEAP model (shorten={shorten_factor}, dim={channel_dimension})")
        load_pretrained_model(
            shorten_factor=shorten_factor,
            channel_dimension=channel_dimension,
            infer_mode=True,
            model_dir=CHECKPOINT_DIR_PATH,
        )
        print("[setup_models] CHEAP ready.")
