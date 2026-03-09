"""
Family-specific classifier training script.

Trains the lightweight family classifier on noisy protein encodings as 
described in §3.6.2 of the paper. The classifier is trained on encodings
corrupted at various noise levels so that it can guide the diffusion
process at any timestep during inference.

Training procedure:
1. Load protein sequences with family labels from a dataset.
2. Encode sequences using the pre-trained encoder (e.g., ESM2-650M).
3. Apply normalization (same as diffusion training).
4. Corrupt encodings with noise at random timesteps t ~ U[0, 1].
5. Train the classifier with cross-entropy loss on the noisy encodings.

Usage:
    python -m src.family_generation.train_classifier \
        --config_path src/configs \
        --families_json path/to/families.json \
        --output_dir checkpoints/family_classifier \
        --epochs 50 \
        --batch_size 32
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.family_classifier import FamilyClassifier, FamilyClassifierConfig
from src.utils.hydra_utils import setup_config
from hydra.utils import instantiate


# ──────────────────────────────────────────────────
# The eight protein families from §3.6.2
# ──────────────────────────────────────────────────
DEFAULT_FAMILIES = [
    "CRISPR-associated protein",
    "Calmodulin",
    "Glycosyl hydrolase",
    "Kinase",
    "Lipase",
    "Lysozyme",
    "Protease",
    "Thioredoxin",
]


class FamilySequenceDataset(Dataset):
    """
    Dataset of protein sequences with family labels.
    
    Expects a JSON file with the format:
    {
        "sequences": ["MKLA...", "MVTL...", ...],
        "families": ["Kinase", "Lysozyme", ...],
        "family_names": ["CRISPR-associated protein", "Calmodulin", ...]
    }
    
    Or alternatively, a list of dicts:
    [
        {"sequence": "MKLA...", "family": "Kinase"},
        ...
    ]
    """
    def __init__(self, data_path: str, family_to_idx: dict = None):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, list):
            # List of dicts format
            self.sequences = [d["sequence"] for d in raw_data]
            family_labels = [d["family"] for d in raw_data]
        elif isinstance(raw_data, dict):
            self.sequences = raw_data["sequences"]
            family_labels = raw_data["families"]
        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")
        
        # Build family-to-index mapping
        if family_to_idx is None:
            unique_families = sorted(set(family_labels))
            self.family_to_idx = {name: i for i, name in enumerate(unique_families)}
        else:
            self.family_to_idx = family_to_idx
        
        self.idx_to_family = {v: k for k, v in self.family_to_idx.items()}
        self.labels = [self.family_to_idx[f] for f in family_labels]
        self.num_families = len(self.family_to_idx)
        
        print(f"Loaded {len(self.sequences)} sequences across {self.num_families} families")
        for name, idx in sorted(self.family_to_idx.items(), key=lambda x: x[1]):
            count = sum(1 for l in self.labels if l == idx)
            print(f"  [{idx}] {name}: {count} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "label": self.labels[idx],
        }


def collate_fn(batch):
    """Simple collation for the family dataset."""
    sequences = [b["sequence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"sequence": sequences, "label": labels}


def train_classifier(
    config,
    dataset: FamilySequenceDataset,
    val_dataset: FamilySequenceDataset = None,
    output_dir: str = "checkpoints/family_classifier",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: torch.device = None,
):
    """
    Train the family classifier on noisy protein encodings.
    
    Args:
        config: Hydra config with encoder settings.
        dataset: Training dataset with family labels.
        val_dataset: Optional validation dataset.
        output_dir: Directory to save checkpoints.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Torch device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Initialize encoder (for encoding sequences) ──
    encoder_partial = instantiate(config.encoder)
    encoder = encoder_partial(
        device=device,
        main_config=config,
        add_enc_normalizer=True,
    )
    
    # Load normalization stats
    stats_path = config.encoder.config.statistics_path
    if os.path.exists(stats_path):
        encoder.enc_normalizer._load_state_dict(stats_path)
        print(f"Loaded encoder normalization stats from {stats_path}")
    else:
        print(f"WARNING: Stats not found at {stats_path}. Running without normalization.")
    
    # ── Initialize diffusion scheduler (for noise sampling) ──
    scheduler = instantiate(config.scheduler)
    dynamic = instantiate(config.dynamic, scheduler=scheduler)
    
    # ── Initialize classifier ──
    clf_config = FamilyClassifierConfig(
        num_families=dataset.num_families,
        embedding_size=config.encoder.config.embedding_dim,
        hidden_size=320,
        num_hidden_layers=3,
        num_attention_heads=16,
        attention_head_size=20,
        intermediate_size=1280,
    )
    classifier = FamilyClassifier(clf_config).to(device)
    
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Classifier parameters: {total_params:,}")
    
    # ── Training setup ──
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    
    best_val_acc = 0.0
    
    for epoch in trange(epochs, desc="Training classifier"):
        classifier.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            sequences = batch["sequence"]
            labels = batch["label"].to(device)
            
            # Encode sequences
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_dict = {"sequence": sequences}
                encodings, attention_mask, _ = encoder.batch_encode(
                    batch_dict, max_sequence_len=config.datasets.max_sequence_len
                )
                attention_mask = attention_mask.float()
            
            # Sample random timesteps for noise
            t = torch.cuda.FloatTensor(encodings.size(0)).uniform_() * dynamic.T
            
            # Add noise
            marg = dynamic.marginal(encodings, t)
            z_t = marg["x_t"]
            
            # Forward through classifier
            logits = classifier(z_t, t, attention_mask)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_samples += labels.size(0)
        
        scheduler_lr.step()
        
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # ── Validation ──
        if val_dataset is not None:
            val_acc = evaluate_classifier(classifier, val_dataset, encoder, dynamic, config, device, batch_size)
            print(f"  Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(output_dir, "best_classifier.pth")
                torch.save({
                    "model": classifier.state_dict(),
                    "config": clf_config.__dict__,
                    "family_to_idx": dataset.family_to_idx,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                }, save_path)
                print(f"  Saved best model (val_acc={val_acc:.4f}) to {save_path}")
        
        # Periodic save
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_path = os.path.join(output_dir, f"classifier_epoch_{epoch+1}.pth")
            torch.save({
                "model": classifier.state_dict(),
                "config": clf_config.__dict__,
                "family_to_idx": dataset.family_to_idx,
                "epoch": epoch + 1,
            }, save_path)
    
    # Save final model
    save_path = os.path.join(output_dir, "final_classifier.pth")
    torch.save({
        "model": classifier.state_dict(),
        "config": clf_config.__dict__,
        "family_to_idx": dataset.family_to_idx,
        "epoch": epochs,
    }, save_path)
    print(f"Final classifier saved to {save_path}")
    
    return classifier


@torch.no_grad()
def evaluate_classifier(classifier, dataset, encoder, dynamic, config, device, batch_size=32):
    """Evaluate classifier accuracy on a dataset with noisy encodings."""
    classifier.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        sequences = batch["sequence"]
        labels = batch["label"].to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_dict = {"sequence": sequences}
            encodings, attention_mask, _ = encoder.batch_encode(
                batch_dict, max_sequence_len=config.datasets.max_sequence_len
            )
            attention_mask = attention_mask.float()
        
        # Use a fixed moderate noise level for evaluation
        t = torch.full((encodings.size(0),), 0.5, device=device)
        marg = dynamic.marginal(encodings, t)
        z_t = marg["x_t"]
        
        logits = classifier(z_t, t, attention_mask)
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_samples += labels.size(0)
    
    classifier.train()
    return total_correct / total_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train family classifier for DiMA")
    parser.add_argument("--config_path", type=str, default="src/configs")
    parser.add_argument("--families_json", type=str, required=True,
                        help="Path to JSON file with family-labeled sequences")
    parser.add_argument("--val_json", type=str, default=None,
                        help="Optional validation JSON file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/family_classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    config = setup_config(config_path=args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = FamilySequenceDataset(args.families_json)
    val_dataset = FamilySequenceDataset(args.val_json, dataset.family_to_idx) if args.val_json else None
    
    train_classifier(
        config=config,
        dataset=dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
