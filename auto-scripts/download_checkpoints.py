import os
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

# Hardcoded metadata based on src/utils/pretrained_utils.py and config exploration
CHECKPOINT_VARIANTS = {
    "SaProt-35M": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-SaProt-35M-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-SaProt-35M.pth",
        "stats": "checkpoints/statistics/encodings-SaProt-35M.pth"
    },
    "ESM2-3B": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-ESM2_3B-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-ESM2-3B.pth",
        "stats": "checkpoints/statistics/encodings-ESM2-3B.pth"
    },
    "ESM2-8M": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-ESM2_8M-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-ESM2-8M.pth",
        "stats": "checkpoints/statistics/encodings-ESM2-8M.pth"
    },
    "CHEAP_shorten_1_dim_1024": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-CHEAP_shorten_1_dim_1024-AFDB/500000.pth",
        "stats": "checkpoints/statistics/encodings-CHEAP_shorten_1_dim_1024.pth"
    },
    "esmc_300m": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-esmc_300m-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-esmc_300m.pth",
        "stats": "checkpoints/statistics/encodings-esmc_300m.pth"
    }
}

S3_BUCKET = "dima-protein-diffusion"
S3_REGION = "eu-north-1"
BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/"

def download_file(url, local_path):
    if local_path.exists():
        print(f"File already exists: {local_path}")
        return

    print(f"Downloading {url} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=local_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def main():
    parser = argparse.ArgumentParser(description="Download DiMA checkpoints from S3")
    parser.add_argument("--variants", nargs="+", help="Specific variants to download (default: all)")
    parser.add_argument("--project_root", type=str, default=".", help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root).absolute()
    
    variants_to_download = args.variants if args.variants else CHECKPOINT_VARIANTS.keys()

    for variant in variants_to_download:
        if variant not in CHECKPOINT_VARIANTS:
            print(f"Warning: Variant {variant} not found in metadata. Skipping.")
            continue
        
        print(f"\nProcessing variant: {variant}")
        paths = CHECKPOINT_VARIANTS[variant]
        
        for key, relative_path in paths.items():
            url = BASE_URL + relative_path
            local_path = project_root / relative_path
            try:
                download_file(url, local_path)
            except Exception as e:
                print(f"Error downloading {key} for {variant}: {e}")

if __name__ == "__main__":
    main()
