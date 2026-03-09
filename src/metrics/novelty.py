"""
Novelty metric for protein sequence generation.

Novelty evaluates similarity to training data to detect potential memorization.
For each generated sequence, we compute the maximum sequence identity against
the training/reference set. Novelty is reported as (1 - max_identity) * 100,
so higher values indicate more novel sequences.

This corresponds to the "Novelty" column in Tables 1-3 of the DiMA paper.
"""

import numpy as np
from typing import List, Optional
from tqdm import tqdm


def sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute pairwise sequence identity between two protein sequences.
    Identity = number of matching positions / length of the longer sequence.
    
    For efficiency, we use a simple ungapped alignment (position-wise comparison
    up to the length of the shorter sequence, divided by the longer).
    """
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / max_len


def compute_novelty_single(query: str, references: List[str]) -> float:
    """
    Compute the novelty score for a single generated sequence.
    Novelty = (1 - max sequence identity to any reference) * 100.
    """
    if not references:
        return 100.0
    max_identity = max(sequence_identity(query, ref) for ref in references)
    return (1.0 - max_identity) * 100.0


def calculate_novelty(
    generated_sequences: List[str],
    reference_sequences: List[str],
    max_references: Optional[int] = None,
) -> dict:
    """
    Calculate novelty scores for a set of generated protein sequences.
    
    For each generated sequence, computes the maximum sequence identity
    against the reference (training) set. Returns per-sequence novelty
    scores and summary statistics.
    
    Args:
        generated_sequences: List of generated amino acid strings.
        reference_sequences: List of reference (training) amino acid strings.
        max_references: If set, subsample references for speed. None = use all.
        
    Returns:
        Dictionary with:
            - 'per_sequence': list of novelty scores (one per generated seq)
            - 'mean': mean novelty across all generated sequences
            - 'std': standard deviation of novelty
            - 'median': median novelty
    """
    refs = reference_sequences
    if max_references is not None and len(refs) > max_references:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(refs), size=max_references, replace=False)
        refs = [refs[i] for i in indices]

    novelty_scores = []
    for gen_seq in tqdm(generated_sequences, desc="Computing novelty"):
        score = compute_novelty_single(gen_seq, refs)
        novelty_scores.append(score)

    return {
        "per_sequence": novelty_scores,
        "mean": float(np.mean(novelty_scores)),
        "std": float(np.std(novelty_scores)),
        "median": float(np.median(novelty_scores)),
    }
