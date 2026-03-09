import torch
import os
import sys
import argparse
import ssl

if os.environ.get("SSL_NO_VERIFY") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context

# Allow running from the project root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.diffusion.dima import DiMAModel

# Encoder presets: name → (encoder_type, hf_model_name, embedding_dim)
ENCODER_PRESETS = {
    "esm2-8m":   ("ESM2-8M",  "facebook/esm2_t6_8M_UR50D",   320),
    "esm2-35m":  ("ESM2-35M", "facebook/esm2_t12_35M_UR50D",  480),
    "esm2-150m": ("ESM2-150M","facebook/esm2_t30_150M_UR50D", 640),
    "esm2-650m": ("ESM2-650M","facebook/esm2_t33_650M_UR50D", 1280),
    "esm2-3b":   ("ESM2-3B",  "facebook/esm2_t36_3B_UR50D",  2560),
}

parser = argparse.ArgumentParser(description="DiMA unconditional protein sequence generation")
parser.add_argument(
    "--encoder",
    choices=list(ENCODER_PRESETS.keys()),
    default="esm2-3b",
    help="Encoder model size. Use esm2-8m on low-VRAM machines (default: esm2-3b for HPC)",
)
parser.add_argument("--num_samples", type=int, default=10)
args = parser.parse_args()

encoder_type, hf_name, emb_dim = ENCODER_PRESETS[args.encoder]
overrides = [
    f"encoder.config.encoder_type={encoder_type}",
    f"encoder.config.encoder_model_name={hf_name}",
    f"encoder.config.embedding_dim={emb_dim}",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "configs")

print(f"Encoder: {encoder_type} | Device: {device}")
model = DiMAModel(config_path=config_path, device=device, overrides=overrides)
model.load_pretrained()

sequences = model.generate_samples(num_texts=args.num_samples)
print(f"Generated {len(sequences)} sequences")
for i, seq in enumerate(sequences):
    print(f"  [{i}] len={len(seq)}: {seq}")
