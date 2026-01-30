import torch
from transformers import EsmForMaskedLM, EsmTokenizer, AutoModel, AutoTokenizer
from cheap.pretrained import load_pretrained_model, CHECKPOINT_DIR_PATH
import os
import argparse

def setup_esm2(model_name="facebook/esm2_t36_3B_UR50D"):
    print(f"Setting up ESM2: {model_name}...")
    try:
        EsmTokenizer.from_pretrained(model_name)
        EsmForMaskedLM.from_pretrained(model_name)
        print("ESM2 setup complete.")
    except Exception as e:
        print(f"Error setting up ESM2: {e}")

def setup_saprot(model_name="westlake-repl/SaProt_35M_AF2"):
    print(f"Setting up SaProt: {model_name}...")
    try:
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print("SaProt setup complete.")
    except Exception as e:
        print(f"Error setting up SaProt: {e}")

def setup_cheap():
    print("Setting up CHEAP...")
    # Based on src/encoders/cheap.py, it uses load_pretrained_model
    # We'll try to trigger a download if possible, or just verify it runs
    try:
        # These params match what's in cheap.yaml
        shorten_factor = 1
        channel_dimension = 1024
        load_pretrained_model(
            shorten_factor=shorten_factor,
            channel_dimension=channel_dimension,
            infer_mode=True,
            model_dir=CHECKPOINT_DIR_PATH,
        )
        print("CHEAP setup complete.")
    except Exception as e:
        print(f"Error setting up CHEAP: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["esm2", "saprot", "cheap"], help="Models to setup")
    args = parser.parse_args()

    if "esm2" in args.models:
        setup_esm2()
    
    if "saprot" in args.models:
        setup_saprot()
        
    if "cheap" in args.models:
        setup_cheap()
