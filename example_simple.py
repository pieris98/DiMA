import torch
import os
import sys

# Allow running from the project root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.diffusion.dima import DiMAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config path relative to this file's directory
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "configs")
model = DiMAModel(config_path=config_path, device=device)
model.load_pretrained()

sequences = model.generate_samples(num_texts=10)
print(f"Generated {len(sequences)} sequences")
for i, seq in enumerate(sequences):
    print(f"  [{i}] len={len(seq)}: {seq}")
