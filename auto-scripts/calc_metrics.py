import argparse
import json
import torch
import sys
import os
import hydra
from omegaconf import OmegaConf

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import plddt, esmpppl, fid
from src.utils.hydra_utils import setup_config

def calculate_metrics(generated_sequences, config, device, metrics_list):
    results = {}
    
    if "esmpppl" in metrics_list:
        print("Calculating ESM Perplexity...")
        # Fix: use calculate_pppl, assume max_len=512 for now or based on config
        # We need to know max_len or just pick large enough
        val = esmpppl.calculate_pppl(generated_sequences, max_len=512, device=device)
        results["esmpppl"] = val
        print(f"ESM PPPL (mean): {sum(val)/len(val) if val else 0}")

    if "plddt" in metrics_list:
        print("Calculating pLDDT (requires ESMFold)...")
        # Fix: use calculate_plddt w/ index_list
        pdb_path = "auto-scripts/generated_pdbs"
        os.makedirs(pdb_path, exist_ok=True)
        indices = list(range(len(generated_sequences)))
        val_dict = plddt.calculate_plddt(generated_sequences, index_list=indices, device=device, pdb_path=pdb_path)
        # Convert dict to list of values or keep dict
        val = list(val_dict.values())
        results["plddt"] = val
        print(f"pLDDT (mean): {sum(val)/len(val) if val else 0}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to generated json")
    parser.add_argument("--metrics", nargs="+", default=["esmpppl", "plddt"], help="Metrics to calc")
    parser.add_argument("--config_path", type=str, default="src/configs/config.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.json_path, "r") as f:
        sequences = json.load(f)
        
    # config needed? maybe for some paths
    config = setup_config(config_path=args.config_path)
    
    results = calculate_metrics(sequences, config, device, args.metrics)
    
    # Save results
    output_path = args.json_path.replace(".json", "_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {output_path}")
