import os
import argparse
import sys
from datasets import load_from_disk, load_dataset, Dataset

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.hydra_utils import setup_config
from src.datasets.load_hub import load_from_hub

def filter_by_length(dataset, min_len, max_len):
    """Filter dataset by sequence length."""
    print(f"Filtering dataset: keeping sequences between {min_len} and {max_len}...")
    return dataset.filter(lambda x: min_len <= len(x["sequence"]) <= max_len)

def prepare_afdb(config, output_dir):
    """Download and prepare AFDB dataset."""
    print("Prearing AFDB dataset...")
    # Using the dataset logic from load_hub.py but adapted for this script
    group_name = "bayes-group-diffusion"
    dataset_name = "AFDB-v2"
    
    # We load from hub, but we might want to cache it locally in a raw folder first
    raw_path = os.path.join(config.datasets.data_dir, "raw", dataset_name)
    if not os.path.exists(raw_path):
        print(f"Downloading {dataset_name} to {raw_path}...")
        dataset = load_dataset(f"{group_name}/{dataset_name}")
        # Assuming it's a DatasetDict or we take 'train' split if it's the only one
        # But load_dataset usually returns DatasetDict if splits exist
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

def prepare_swissprot(config, output_dir):
    """Prepare SwissProt dataset."""
    print("Preparing SwissProt dataset...")
    group_name = "bayes-group-diffusion"
    dataset_name = "swissprot"
    
    raw_path = os.path.join(config.datasets.data_dir, "raw", dataset_name)
    if not os.path.exists(raw_path):
        print(f"Downloading {dataset_name} to {raw_path}...")
        dataset = load_dataset(f"{group_name}/{dataset_name}")
        dataset.save_to_disk(raw_path)
    else:
        dataset = load_from_disk(raw_path)

    if hasattr(dataset, "keys") and "train" in dataset.keys():
        data = dataset["train"]
    else:
        data = dataset

    # Filter 128-254
    filtered_data = filter_by_length(data, 128, 254)
    print(f"SwissProt sequences after filtering: {len(filtered_data)}")
    
    # Save as specific validation set
    save_path = os.path.join(output_dir, "swissprot_val")
    filtered_data.save_to_disk(save_path)
    print(f"Saved SwissProt to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/configs/config.yaml")
    parser.add_argument("--dataset", type=str, choices=["afdb", "swissprot", "all"], default="all")
    args = parser.parse_args()

    # Load config to get data_dir
    # We might need to mock or setup config if not running via hydra directly
    # create a minimal config object or load it
    config = setup_config(config_path=args.config_path)
    
    output_dir = config.datasets.data_dir
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    
    if args.dataset in ["afdb", "all"]:
        prepare_afdb(config, output_dir)
    
    if args.dataset in ["swissprot", "all"]:
        prepare_swissprot(config, output_dir)
