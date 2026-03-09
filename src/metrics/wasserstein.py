"""
1-Wasserstein Optimal Transport (OT) metric for protein sequence distribution matching.

Computes the 1-Wasserstein (Earth Mover's Distance) between distributions of 
generated and reference protein sequences in the ProtT5 embedding space.

This corresponds to the "1-Wasserstein optimal transport" metric mentioned in
§3.1 of the DiMA paper as part of the comprehensive evaluation framework.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import List, Dict

from src.metrics.util import create_embeds


def wasserstein_per_dimension(embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> float:
    """
    Compute the average 1-Wasserstein distance across all embedding dimensions.
    
    Since the full multi-dimensional Wasserstein distance is computationally
    expensive for high-dimensional embeddings, we use the sliced/per-dimension
    approach: compute the 1D Wasserstein distance for each feature dimension
    and average them.
    
    Args:
        embeddings_1: (N, D) array of embeddings for set 1.
        embeddings_2: (M, D) array of embeddings for set 2.
        
    Returns:
        Average 1-Wasserstein distance across dimensions.
    """
    assert embeddings_1.shape[1] == embeddings_2.shape[1], \
        "Embedding dimensions must match"
    
    dim = embeddings_1.shape[1]
    w_distances = []
    for d in range(dim):
        w_d = wasserstein_distance(embeddings_1[:, d], embeddings_2[:, d])
        w_distances.append(w_d)
    
    return float(np.mean(w_distances))


def sliced_wasserstein(embeddings_1: np.ndarray, embeddings_2: np.ndarray, 
                       n_projections: int = 100, seed: int = 42) -> float:
    """
    Compute the sliced Wasserstein distance between two sets of embeddings.
    
    Projects the embeddings onto random 1D directions and computes the
    average 1-Wasserstein distance over those projections.
    
    Args:
        embeddings_1: (N, D) array.
        embeddings_2: (M, D) array.
        n_projections: Number of random projections.
        seed: Random seed for reproducibility.
        
    Returns:
        Sliced Wasserstein distance.
    """
    rng = np.random.RandomState(seed)
    dim = embeddings_1.shape[1]
    
    # Random unit directions
    projections = rng.randn(n_projections, dim)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    sw_distances = []
    for proj in projections:
        # Project onto 1D
        proj_1 = embeddings_1 @ proj  # (N,)
        proj_2 = embeddings_2 @ proj  # (M,)
        w_d = wasserstein_distance(proj_1, proj_2)
        sw_distances.append(w_d)
    
    return float(np.mean(sw_distances))


def calculate_wasserstein_for_embs(embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> Dict[str, float]:
    """
    Compute both per-dimension and sliced 1-Wasserstein distances.
    
    Args:
        embeddings_1: (N, D) embeddings for generated sequences.
        embeddings_2: (M, D) embeddings for reference sequences.
        
    Returns:
        Dictionary with 'w1_per_dim' and 'w1_sliced'.
    """
    w1_dim = wasserstein_per_dimension(embeddings_1, embeddings_2)
    w1_sliced = sliced_wasserstein(embeddings_1, embeddings_2)
    
    return {
        "w1_per_dim": w1_dim,
        "w1_sliced": w1_sliced,
    }


def calculate_wasserstein_for_lists(
    predictions: List[str], 
    references: List[str], 
    max_len: int, 
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    Compute 1-Wasserstein distance between generated and reference protein sequences.
    
    Uses ProtT5 embeddings (same as FID computation) for consistent comparison.
    
    Args:
        predictions: Generated protein sequences.
        references: Reference protein sequences.
        max_len: Maximum sequence length for tokenization.
        device: Torch device string.
        
    Returns:
        Dictionary with Wasserstein distance metrics.
    """
    embeddings_1, embeddings_2 = create_embeds(predictions, references, max_len, device)
    return calculate_wasserstein_for_embs(embeddings_1, embeddings_2)
