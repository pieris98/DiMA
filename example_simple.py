import torch
from src.diffusion.dima import DiMAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiMAModel(config_path="../configs", device=device)
model.load_pretrained()

sequences = model.generate_samples(num_texts=10)
