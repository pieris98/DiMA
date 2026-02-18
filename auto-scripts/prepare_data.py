import os
import argparse
import sys
from datasets import load_from_disk, load_dataset, Dataset

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.hydra_utils import setup_config

def filter_by_length(dataset, min_len, max_len):
    """Filter dataset by sequence length."""
    print(f"Filtering dataset: keeping sequences between {min_len} and {max_len}...")
    return dataset.filter(lambda x: min_len <= len(x["sequence"]) <= max_len)

def prepare_afdb(data_root, output_dir):
    """Download and prepare AFDB dataset."""
    print("Preparing AFDB dataset...")
    group_name = "bayes-group-diffusion"
    dataset_name = "AFDB-v2"

    raw_path = os.path.join(data_root, "raw", dataset_name)
    if not os.path.exists(raw_path):
        print(f"Downloading {dataset_name} to {raw_path}...")
        dataset = load_dataset(f"{group_name}/{dataset_name}")
        dataset.save_to_disk(raw_path)
    else:
        print(f"Loading raw {dataset_name} from {raw_path}...")
        dataset = load_from_disk(raw_path)

    # If dataset is a dict, iterate? Usually unconditional training uses one big train set.
    # checking src/datasets/load_hub.py, it seems it expects a single disk save.
    
    if hasattr(dataset, "keys") and "train" in dataset.keys():
        full_data = dataset["train"]
    else:
        full_data = dataset

    # Filter
    filtered_data = filter_by_length(full_data, 64, 510)
    
    # Split: 2.1M train / 50K val / 50K test (approx)
    # Total size check
    total_size = len(filtered_data)
    print(f"Total sequences after filtering: {total_size}")
    
    test_size = 50000
    val_size = 50000
    train_size = total_size - test_size - val_size
    
    if train_size <= 0:
         raise ValueError("Dataset too small for requested split sizes.")

    # Deterministic split
    split_1 = filtered_data.train_test_split(test_size=test_size, seed=42)
    test_data = split_1["test"]
    remaining = split_1["train"]
    
    split_2 = remaining.train_test_split(test_size=val_size, seed=42)
    val_data = split_2["test"]
    train_data = split_2["train"]
    
    # Save
    print(f"Saving splits to {output_dir}...")
    train_data.save_to_disk(os.path.join(output_dir, "train"))
    val_data.save_to_disk(os.path.join(output_dir, "val"))
    test_data.save_to_disk(os.path.join(output_dir, "test"))
    
    print("AFDB preparation complete.")

def prepare_swissprot(data_root, output_dir):
    """Download and prepare SwissProt dataset."""
    print("Preparing SwissProt dataset...")
    group_name = "bayes-group-diffusion"
    dataset_name = "swissprot"

    raw_path = os.path.join(data_root, "raw", dataset_name)
    if not os.path.exists(raw_path):
        print(f"Downloading {dataset_name} to {raw_path}...")
        dataset = load_dataset(f"{group_name}/{dataset_name}")
        dataset.save_to_disk(raw_path)
    else:
        print(f"Loading raw {dataset_name} from {raw_path}...")
        dataset = load_from_disk(raw_path)

    if hasattr(dataset, "keys") and "train" in dataset.keys():
        data = dataset["train"]
    else:
        data = dataset

    # Filter 128-254
    filtered_data = filter_by_length(data, 128, 254)
    total_size = len(filtered_data)
    print(f"SwissProt sequences after filtering: {total_size}")

    # Split into train/val/test
    test_size = min(5000, total_size // 10)
    val_size = min(5000, total_size // 10)
    train_size = total_size - test_size - val_size

    if train_size <= 0:
        raise ValueError("SwissProt dataset too small for requested split sizes.")

    split_1 = filtered_data.train_test_split(test_size=test_size, seed=42)
    test_data = split_1["test"]
    remaining = split_1["train"]

    split_2 = remaining.train_test_split(test_size=val_size, seed=42)
    val_data = split_2["test"]
    train_data = split_2["train"]

    print(f"Saving SwissProt splits to {output_dir}...")
    train_data.save_to_disk(os.path.join(output_dir, "train"))
    val_data.save_to_disk(os.path.join(output_dir, "val"))
    test_data.save_to_disk(os.path.join(output_dir, "test"))
    print("SwissProt preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs",
                        help="Path to the configs directory (not the yaml file)")
    parser.add_argument("--dataset", type=str, choices=["afdb", "swissprot", "all"], default="all")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for data (default: <project_path>/data)")
    args = parser.parse_args()

    # Accept both "src/configs" (directory) and "src/configs/config.yaml" (file)
    config_path = args.config_path
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        config_path = os.path.dirname(config_path)

    config = setup_config(config_path=config_path)

    data_root = args.data_root if args.data_root else os.path.join(config.project.path, "data")
    os.makedirs(os.path.join(data_root, "raw"), exist_ok=True)

    if args.dataset in ["afdb", "all"]:
        afdb_output = os.path.join(data_root, "AFDB-v2")
        os.makedirs(afdb_output, exist_ok=True)
        prepare_afdb(data_root, afdb_output)

    if args.dataset in ["swissprot", "all"]:
        sp_output = os.path.join(data_root, "swissprot")
        os.makedirs(sp_output, exist_ok=True)
        prepare_swissprot(data_root, sp_output)
