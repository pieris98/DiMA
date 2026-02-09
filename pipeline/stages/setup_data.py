"""
Stage: Dataset download and preparation.

Downloads datasets from HuggingFace Hub, filters by sequence length,
splits into train/val/test, and saves to disk.
"""

import os
from typing import Dict, Any

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("setup_data")
class SetupDataStage(BaseStage):
    name = "setup_data"
    description = "Download and prepare datasets"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        if not hasattr(config, "datasets") and "datasets" not in config:
            issues.append("No 'datasets' section in config")
        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        from datasets import load_from_disk, load_dataset

        datasets_config = config.datasets if hasattr(config, "datasets") else config["datasets"]
        data_dir = datasets_config.data_dir if hasattr(datasets_config, "data_dir") else datasets_config["data_dir"]
        os.makedirs(data_dir, exist_ok=True)

        dataset_name = getattr(datasets_config, "name", None) or datasets_config.get("name", "afdb")
        group_name = getattr(datasets_config, "group_name", None) or datasets_config.get("group_name", "bayes-group-diffusion")
        hub_name = getattr(datasets_config, "hub_name", None) or datasets_config.get("hub_name", "AFDB-v2")
        min_len = getattr(datasets_config, "min_sequence_len", None) or datasets_config.get("min_sequence_len", 64)
        max_len = getattr(datasets_config, "max_sequence_len", None) or datasets_config.get("max_sequence_len", 510)

        train_path = os.path.join(data_dir, "train")
        if os.path.exists(train_path):
            print(f"[setup_data] Dataset already prepared at {data_dir}, skipping download.")
            context["data_dir"] = data_dir
            return context

        print(f"[setup_data] Downloading {group_name}/{hub_name}...")
        dataset = load_dataset(f"{group_name}/{hub_name}")

        if hasattr(dataset, "keys") and "train" in dataset.keys():
            data = dataset["train"]
        else:
            data = dataset

        print(f"[setup_data] Filtering sequences to length [{min_len}, {max_len}]...")
        filtered = data.filter(lambda x: min_len <= len(x["sequence"]) <= max_len)
        total = len(filtered)
        print(f"[setup_data] {total} sequences after filtering.")

        # Split
        test_size = min(50000, total // 10)
        val_size = min(50000, total // 10)

        split1 = filtered.train_test_split(test_size=test_size, seed=42)
        test_data = split1["test"]
        remaining = split1["train"]

        split2 = remaining.train_test_split(test_size=val_size, seed=42)
        val_data = split2["test"]
        train_data = split2["train"]

        print(f"[setup_data] Saving splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        train_data.save_to_disk(os.path.join(data_dir, "train"))
        val_data.save_to_disk(os.path.join(data_dir, "val"))
        test_data.save_to_disk(os.path.join(data_dir, "test"))

        context["data_dir"] = data_dir
        print("[setup_data] Complete.")
        return context
