"""
Cluster Diversity (CD) metric for protein sequence generation.

Measures the fraction of distinct protein clusters in generated samples at
a given sequence identity threshold. CD0.5 (50% identity) is the primary
metric used in the DiMA paper (Tables 1-3).

CD = (number of clusters) / (number of sequences)

A value of 1.0 means all sequences are unique clusters (maximum diversity).
A low value indicates many sequences cluster together (low diversity / repetition).

The clustering uses a greedy incremental approach similar to CD-HIT:
sequences are sorted by length (descending), and each sequence either joins
an existing cluster (if identity >= threshold to the representative) or
starts a new cluster.
"""

import numpy as np
from typing import List, Optional, Dict
from tqdm import tqdm


def sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute ungapped pairwise sequence identity.
    Identity = matching positions / length of longer sequence.
    """
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / max_len


def greedy_cluster(sequences: List[str], identity_threshold: float = 0.5) -> List[List[int]]:
    """
    Greedy incremental clustering of protein sequences.
    
    Sequences are sorted by length (descending). Each sequence is compared
    against existing cluster representatives. If identity >= threshold,
    it joins that cluster; otherwise it starts a new cluster.
    
    Args:
        sequences: List of amino acid strings.
        identity_threshold: Minimum identity to join a cluster (default: 0.5).
        
    Returns:
        List of clusters, where each cluster is a list of original indices.
    """
    if not sequences:
        return []

    # Sort by length descending (CD-HIT convention)
    indexed_seqs = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)
    
    clusters = []  # Each cluster: (representative_seq, [indices])
    
    for orig_idx, seq in tqdm(indexed_seqs, desc=f"Clustering (thresh={identity_threshold})", 
                               disable=len(sequences) < 100):
        assigned = False
        for cluster_rep, cluster_indices in clusters:
            ident = sequence_identity(seq, cluster_rep)
            if ident >= identity_threshold:
                cluster_indices.append(orig_idx)
                assigned = True
                break
        
        if not assigned:
            clusters.append((seq, [orig_idx]))
    
    return [indices for _, indices in clusters]


def calculate_cluster_diversity(
    sequences: List[str],
    identity_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate cluster diversity at a given sequence identity threshold.
    
    CD = num_clusters / num_sequences
    
    Args:
        sequences: List of generated amino acid strings.
        identity_threshold: Clustering identity threshold (default: 0.5 for CD0.5).
        
    Returns:
        Dictionary with:
            - 'num_clusters': Number of clusters formed.
            - 'num_sequences': Total number of sequences.
            - 'cluster_diversity': CD = num_clusters / num_sequences.
            - 'identity_threshold': The threshold used.
    """
    if not sequences:
        return {
            "num_clusters": 0,
            "num_sequences": 0,
            "cluster_diversity": 0.0,
            "identity_threshold": identity_threshold,
        }
    
    clusters = greedy_cluster(sequences, identity_threshold)
    num_clusters = len(clusters)
    num_sequences = len(sequences)
    
    return {
        "num_clusters": num_clusters,
        "num_sequences": num_sequences,
        "cluster_diversity": num_clusters / num_sequences,
        "identity_threshold": identity_threshold,
    }


def calculate_multi_threshold_diversity(
    sequences: List[str],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Dict]:
    """
    Calculate cluster diversity at multiple sequence identity thresholds.
    
    The paper mentions "diversity assessment through multiple sequence identity
    thresholds" in §3.1.
    
    Args:
        sequences: List of generated amino acid strings.
        thresholds: List of identity thresholds. Default: [0.3, 0.5, 0.7, 0.9].
        
    Returns:
        Dictionary mapping threshold string to diversity metrics.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]
    
    results = {}
    for thresh in thresholds:
        key = f"CD_{thresh:.1f}"
        results[key] = calculate_cluster_diversity(sequences, identity_threshold=thresh)
    
    return results
