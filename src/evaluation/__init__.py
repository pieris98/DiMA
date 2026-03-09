"""
Comprehensive evaluation pipeline for DiMA protein sequence generation.

This script implements the full evaluation framework described in §3.1 of the paper,
computing all sequence-modality metrics in a single unified run:

Quality metrics:
  - pLDDT: Structural plausibility (via ESMFold)
  - ESM Pseudo-Perplexity: Language model quality score

Distribution matching:
  - FD-seq (Fréchet Distance): Distribution similarity on ProtT5 embeddings
  - MMD: Maximum Mean Discrepancy on ProtT5 embeddings
  - W1 (1-Wasserstein): Optimal transport distance on ProtT5 embeddings

Diversity:
  - CD0.5: Cluster diversity at 50% sequence identity
  - CD0.3, CD0.7, CD0.9: Multi-threshold cluster diversity

Novelty:
  - Novelty: (1 - max sequence identity to training data) × 100

Additional:
  - Sequence length statistics
  - Amino acid composition analysis
  - Repetition rate (fraction of duplicated sequences)

Usage:
    python -m src.evaluation.evaluate_sequences \
        --generated_json path/to/generated.json \
        --reference_json path/to/reference.json \
        --output_dir evaluation_results/ \
        --metrics all \
        --device cuda:0
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from typing import List, Dict, Optional, Set
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ──────────────────────────────────────────────────
# Helper: Amino Acid composition analysis
# ──────────────────────────────────────────────────
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def analyze_sequences(sequences: List[str]) -> Dict:
    """Compute basic statistics about a set of protein sequences."""
    lengths = [len(s) for s in sequences]
    
    # Amino acid composition
    total_residues = sum(lengths)
    aa_counts = Counter()
    for seq in sequences:
        aa_counts.update(seq)
    
    aa_freq = {aa: aa_counts.get(aa, 0) / total_residues for aa in sorted(STANDARD_AA)}
    
    # Non-standard residues
    non_standard = sum(1 for seq in sequences for c in seq if c not in STANDARD_AA)
    non_standard_rate = non_standard / total_residues if total_residues > 0 else 0
    
    # Repetition rate: fraction of sequences that appear more than once
    seq_counter = Counter(sequences)
    duplicated = sum(count - 1 for count in seq_counter.values() if count > 1)
    repetition_rate = duplicated / len(sequences) if sequences else 0
    
    return {
        "num_sequences": len(sequences),
        "length_mean": float(np.mean(lengths)) if lengths else 0,
        "length_std": float(np.std(lengths)) if lengths else 0,
        "length_min": int(min(lengths)) if lengths else 0,
        "length_max": int(max(lengths)) if lengths else 0,
        "length_median": float(np.median(lengths)) if lengths else 0,
        "aa_frequency": aa_freq,
        "non_standard_rate": non_standard_rate,
        "unique_sequences": len(seq_counter),
        "repetition_rate": repetition_rate,
    }


def run_evaluation(
    generated_sequences: List[str],
    reference_sequences: Optional[List[str]] = None,
    training_sequences: Optional[List[str]] = None,
    metrics: Set[str] = None,
    max_len: int = 254,
    device: str = "cuda:0",
    pdb_path: str = "evaluation_pdbs",
    output_dir: str = "evaluation_results",
    num_samples_heavy: int = 512,
    num_samples_light: int = 2048,
) -> Dict:
    """
    Run comprehensive evaluation on generated protein sequences.
    
    Args:
        generated_sequences: List of generated amino acid strings.
        reference_sequences: Reference (test set) sequences for distribution metrics.
        training_sequences: Training sequences for novelty computation.
        metrics: Set of metric names to compute. None or {"all"} = all metrics.
        max_len: Maximum sequence length for embedding computation.
        device: Torch device string.
        pdb_path: Directory to store predicted PDB files (for pLDDT).
        output_dir: Directory to save results.
        num_samples_heavy: Number of sequences for heavy metrics (pLDDT, pppl).
        num_samples_light: Number of sequences for light metrics (FID, MMD, etc.).
        
    Returns:
        Dictionary of all computed metrics.
    """
    if metrics is None:
        metrics = {"all"}
    
    compute_all = "all" in metrics
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("DiMA Comprehensive Evaluation")
    print("=" * 70)
    print(f"Generated sequences: {len(generated_sequences)}")
    if reference_sequences:
        print(f"Reference sequences: {len(reference_sequences)}")
    if training_sequences:
        print(f"Training sequences:  {len(training_sequences)}")
    print()
    
    # ── Basic statistics ──
    print("─" * 50)
    print("[1/7] Computing sequence statistics...")
    results["generated_stats"] = analyze_sequences(generated_sequences)
    if reference_sequences:
        results["reference_stats"] = analyze_sequences(reference_sequences)
    print(f"  Generated: {results['generated_stats']['num_sequences']} sequences, "
          f"mean length: {results['generated_stats']['length_mean']:.1f}")
    print(f"  Repetition rate: {results['generated_stats']['repetition_rate']:.4f}")
    
    # ── Cluster Diversity (CD) ──
    if compute_all or "diversity" in metrics or "cd" in metrics:
        print("─" * 50)
        print("[2/7] Computing cluster diversity (CD0.5 and multi-threshold)...")
        from src.metrics.diversity import calculate_multi_threshold_diversity
        
        seqs_for_cd = generated_sequences[:num_samples_light]
        cd_results = calculate_multi_threshold_diversity(
            seqs_for_cd, thresholds=[0.3, 0.5, 0.7, 0.9]
        )
        results["cluster_diversity"] = {
            k: v["cluster_diversity"] for k, v in cd_results.items()
        }
        results["cluster_diversity_details"] = {
            k: {kk: vv for kk, vv in v.items()} for k, v in cd_results.items()
        }
        
        cd05 = results["cluster_diversity"].get("CD_0.5", "N/A")
        print(f"  CD0.5: {cd05}")
        for key, val in results["cluster_diversity"].items():
            print(f"  {key}: {val:.4f}")
    
    # ── Novelty ──
    if (compute_all or "novelty" in metrics) and training_sequences:
        print("─" * 50)
        print("[3/7] Computing novelty (vs training data)...")
        from src.metrics.novelty import calculate_novelty
        
        seqs_for_novelty = generated_sequences[:num_samples_light]
        novelty_result = calculate_novelty(
            seqs_for_novelty, 
            training_sequences,
            max_references=5000,  # Subsample for speed
        )
        results["novelty"] = {
            "mean": novelty_result["mean"],
            "std": novelty_result["std"],
            "median": novelty_result["median"],
        }
        print(f"  Novelty (mean): {novelty_result['mean']:.2f}")
    elif compute_all or "novelty" in metrics:
        print("─" * 50)
        print("[3/7] Skipping novelty (no training sequences provided)")
    
    # ── Fréchet Distance (FD-seq) ──
    if (compute_all or "fid" in metrics or "fd_seq" in metrics) and reference_sequences:
        print("─" * 50)
        print("[4/7] Computing Fréchet Distance (FD-seq) on ProtT5 embeddings...")
        from src.metrics.fid import calculate_fid_for_lists
        
        gen_subset = generated_sequences[:num_samples_light]
        ref_subset = reference_sequences[:num_samples_light]
        fid_val = calculate_fid_for_lists(gen_subset, ref_subset, max_len=max_len, device=device)
        results["fd_seq"] = float(fid_val)
        print(f"  FD-seq: {fid_val:.4f}")
    
    # ── MMD ──
    if (compute_all or "mmd" in metrics) and reference_sequences:
        print("─" * 50)
        print("[5/7] Computing MMD on ProtT5 embeddings...")
        from src.metrics.mmd import calculate_mmd_for_lists
        
        gen_subset = generated_sequences[:num_samples_light]
        ref_subset = reference_sequences[:num_samples_light]
        mmd_val = calculate_mmd_for_lists(gen_subset, ref_subset, max_len=max_len, device=device)
        results["mmd"] = float(mmd_val)
        print(f"  MMD: {mmd_val:.6f}")
    
    # ── 1-Wasserstein ──
    if (compute_all or "wasserstein" in metrics or "w1" in metrics) and reference_sequences:
        print("─" * 50)
        print("[5.5/7] Computing 1-Wasserstein distance...")
        from src.metrics.wasserstein import calculate_wasserstein_for_lists
        
        gen_subset = generated_sequences[:num_samples_light]
        ref_subset = reference_sequences[:num_samples_light]
        w1_result = calculate_wasserstein_for_lists(gen_subset, ref_subset, max_len=max_len, device=device)
        results["wasserstein"] = w1_result
        print(f"  W1 (per-dim): {w1_result['w1_per_dim']:.6f}")
        print(f"  W1 (sliced):  {w1_result['w1_sliced']:.6f}")
    
    # ── ESM Pseudo-Perplexity ──
    if compute_all or "esmpppl" in metrics or "pppl" in metrics:
        print("─" * 50)
        print("[6/7] Computing ESM pseudo-perplexity...")
        from src.metrics.esmpppl import calculate_pppl
        
        seqs_for_pppl = generated_sequences[:num_samples_heavy]
        pppl_vals = calculate_pppl(seqs_for_pppl, max_len=max_len, device=device)
        results["esm_pppl"] = {
            "mean": float(np.mean(pppl_vals)),
            "std": float(np.std(pppl_vals)),
            "median": float(np.median(pppl_vals)),
        }
        print(f"  ESM PPPL (mean): {results['esm_pppl']['mean']:.4f}")
    
    # ── pLDDT ──
    if compute_all or "plddt" in metrics:
        print("─" * 50)
        print("[7/7] Computing pLDDT (requires ESMFold - this may be slow)...")
        try:
            from src.metrics.plddt import calculate_plddt
            
            seqs_for_plddt = generated_sequences[:num_samples_heavy]
            indices = list(range(len(seqs_for_plddt)))
            plddt_dir = os.path.join(output_dir, pdb_path)
            os.makedirs(plddt_dir, exist_ok=True)
            
            plddt_result = calculate_plddt(
                predictions=seqs_for_plddt,
                index_list=indices,
                device=device,
                pdb_path=plddt_dir,
            )
            plddt_values = list(plddt_result.values())
            results["plddt"] = {
                "mean": float(np.mean(plddt_values)),
                "std": float(np.std(plddt_values)),
                "median": float(np.median(plddt_values)),
            }
            print(f"  pLDDT (mean): {results['plddt']['mean']:.2f}")
        except ImportError as e:
            print(f"  WARNING: pLDDT computation skipped (missing dependency: {e})")
            results["plddt"] = {"error": str(e)}
    
    # ── Summary ──
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    summary_keys = [
        ("FD-seq ↓", results.get("fd_seq")),
        ("MMD ↓", results.get("mmd")),
        ("W1 (sliced) ↓", results.get("wasserstein", {}).get("w1_sliced")),
        ("pLDDT ↑", results.get("plddt", {}).get("mean")),
        ("ESM-PPPL ↓", results.get("esm_pppl", {}).get("mean")),
        ("CD0.5 ↑", results.get("cluster_diversity", {}).get("CD_0.5")),
        ("Novelty ↑", results.get("novelty", {}).get("mean")),
        ("Repetition rate ↓", results.get("generated_stats", {}).get("repetition_rate")),
    ]
    
    for name, val in summary_keys:
        if val is not None:
            if isinstance(val, float):
                print(f"  {name:25s}: {val:.4f}")
            else:
                print(f"  {name:25s}: {val}")
    
    print("=" * 70)
    
    # Save results
    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of generated protein sequences")
    parser.add_argument("--generated_json", type=str, required=True,
                        help="Path to JSON file with generated sequences (list of strings)")
    parser.add_argument("--reference_json", type=str, default=None,
                        help="Path to JSON file with reference (test) sequences")
    parser.add_argument("--training_json", type=str, default=None,
                        help="Path to JSON file with training sequences (for novelty)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        help="Metrics to compute: all, fid, mmd, wasserstein, plddt, esmpppl, diversity, novelty")
    parser.add_argument("--max_len", type=int, default=254)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples_heavy", type=int, default=512,
                        help="Number of samples for heavy metrics (pLDDT, PPPL)")
    parser.add_argument("--num_samples_light", type=int, default=2048,
                        help="Number of samples for light metrics (FID, MMD, etc.)")
    args = parser.parse_args()
    
    # Load generated sequences
    with open(args.generated_json, "r") as f:
        data = json.load(f)
    
    # Handle both formats: list of strings, or dict with "sequences" key
    if isinstance(data, list):
        generated_sequences = data
    elif isinstance(data, dict):
        generated_sequences = data.get("sequences", data.get("generated_sequences", []))
    else:
        raise ValueError(f"Unsupported format in {args.generated_json}")
    
    # Load reference sequences
    reference_sequences = None
    if args.reference_json:
        with open(args.reference_json, "r") as f:
            ref_data = json.load(f)
        if isinstance(ref_data, list):
            reference_sequences = ref_data
        elif isinstance(ref_data, dict):
            reference_sequences = ref_data.get("sequences", ref_data.get("reference_sequences", []))
    
    # Load training sequences (for novelty)
    training_sequences = None
    if args.training_json:
        with open(args.training_json, "r") as f:
            train_data = json.load(f)
        if isinstance(train_data, list):
            training_sequences = train_data
        elif isinstance(train_data, dict):
            training_sequences = train_data.get("sequences", [])
    
    results = run_evaluation(
        generated_sequences=generated_sequences,
        reference_sequences=reference_sequences,
        training_sequences=training_sequences,
        metrics=set(args.metrics),
        max_len=args.max_len,
        device=args.device,
        output_dir=args.output_dir,
        num_samples_heavy=args.num_samples_heavy,
        num_samples_light=args.num_samples_light,
    )
