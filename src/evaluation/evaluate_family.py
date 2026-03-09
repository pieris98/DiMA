"""
Evaluation of family-specific conditional generation.

Implements the evaluation protocol from §3.6.2:
- Fidelity: InterProScan-based family membership verification
- Quality: pLDDT structural quality assessment
- Diversity: CD0.5 cluster diversity at 50% sequence identity

Since InterProScan requires an external installation, this module provides:
1. HMMER-based fidelity (using Pfam HMM profiles, if available)
2. Sequence-similarity based fidelity (using known family members as reference)
3. Wrapper for InterProScan CLI (if installed)

Usage:
    python -m src.evaluation.evaluate_family \
        --generated_json path/to/family_generated.json \
        --family_reference_json path/to/family_members.json \
        --output_dir evaluation_results/family/
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def write_fasta(sequences: List[str], filepath: str, prefix: str = "gen"):
    """Write protein sequences to a FASTA file."""
    with open(filepath, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i}\n")
            # Write in 80-character lines
            for j in range(0, len(seq), 80):
                f.write(seq[j:j+80] + "\n")


def sequence_identity(seq1: str, seq2: str) -> float:
    """Simple ungapped sequence identity."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / max_len


# ──────────────────────────────────────────────────
# Fidelity Methods
# ──────────────────────────────────────────────────

def fidelity_by_sequence_similarity(
    generated_sequences: List[str],
    family_reference_sequences: List[str],
    identity_threshold: float = 0.3,
) -> Dict:
    """
    Estimate fidelity by checking if generated sequences are similar enough
    to known family members.
    
    A generated sequence is considered "faithful" to the family if its
    maximum sequence identity to any reference family member exceeds
    the threshold.
    
    Args:
        generated_sequences: Generated sequences.
        family_reference_sequences: Known members of the target family.
        identity_threshold: Minimum identity to be considered a family member.
        
    Returns:
        Dictionary with fidelity metrics.
    """
    from tqdm import tqdm
    
    faithful_count = 0
    max_identities = []
    
    for gen_seq in tqdm(generated_sequences, desc="Computing fidelity"):
        max_id = max(
            sequence_identity(gen_seq, ref) for ref in family_reference_sequences
        )
        max_identities.append(max_id)
        if max_id >= identity_threshold:
            faithful_count += 1
    
    fidelity = faithful_count / len(generated_sequences)
    
    return {
        "fidelity": fidelity,
        "faithful_count": faithful_count,
        "total_count": len(generated_sequences),
        "identity_threshold": identity_threshold,
        "mean_max_identity": float(np.mean(max_identities)),
        "std_max_identity": float(np.std(max_identities)),
    }


def fidelity_by_interproscan(
    generated_sequences: List[str],
    target_family_accession: str,
    interproscan_path: str = "interproscan.sh",
) -> Dict:
    """
    Compute fidelity using InterProScan for family membership verification.
    
    This is the primary method used in the paper (§3.6.2):
    "InterProScan for family membership verification (Fidelity)"
    
    Requires InterProScan to be installed and accessible.
    
    Args:
        generated_sequences: Generated sequences.
        target_family_accession: InterPro/Pfam accession (e.g., "PF00069").
        interproscan_path: Path to interproscan.sh.
        
    Returns:
        Dictionary with fidelity metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sequences to FASTA
        fasta_path = os.path.join(tmpdir, "generated.fasta")
        write_fasta(generated_sequences, fasta_path)
        
        # Run InterProScan
        output_path = os.path.join(tmpdir, "results.tsv")
        cmd = [
            interproscan_path,
            "-i", fasta_path,
            "-f", "tsv",
            "-o", output_path,
            "--disable-precalc",
            "-dp",
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"InterProScan not found at '{interproscan_path}'. "
                "Please install InterProScan or provide the correct path. "
                "Alternatively, use fidelity_by_sequence_similarity()."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"InterProScan failed: {e.stderr}")
        
        # Parse results
        annotated = set()
        family_matches = set()
        all_annotations = []
        
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 12:
                        seq_id = parts[0]
                        accession = parts[11] if len(parts) > 11 else ""
                        annotated.add(seq_id)
                        all_annotations.append({
                            "seq_id": seq_id,
                            "accession": accession,
                            "description": parts[12] if len(parts) > 12 else "",
                        })
                        if target_family_accession in accession:
                            family_matches.add(seq_id)
        
        fidelity = len(family_matches) / len(generated_sequences) if generated_sequences else 0
        annotation_rate = len(annotated) / len(generated_sequences) if generated_sequences else 0
        
        return {
            "fidelity": fidelity,
            "family_matches": len(family_matches),
            "annotated_sequences": len(annotated),
            "total_sequences": len(generated_sequences),
            "annotation_rate": annotation_rate,
            "target_accession": target_family_accession,
        }


# ──────────────────────────────────────────────────
# Combined family evaluation
# ──────────────────────────────────────────────────

def evaluate_family_generation(
    generated_sequences: List[str],
    family_reference_sequences: Optional[List[str]] = None,
    target_family_accession: Optional[str] = None,
    interproscan_path: Optional[str] = None,
    device: str = "cuda:0",
    output_dir: str = "evaluation_results/family",
) -> Dict:
    """
    Run the complete family-specific evaluation from §3.6.2.
    
    Computes:
    1. Fidelity (via sequence similarity or InterProScan)
    2. Quality (pLDDT)
    3. Diversity (CD0.5)
    
    Args:
        generated_sequences: Generated sequences for the target family.
        family_reference_sequences: Known family members for similarity-based fidelity.
        target_family_accession: InterPro accession for InterProScan-based fidelity.
        interproscan_path: Path to interproscan.sh (None = skip InterProScan).
        device: Torch device.
        output_dir: Output directory for results.
        
    Returns:
        Combined evaluation results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print("=" * 60)
    print("Family-Specific Generation Evaluation")
    print("=" * 60)
    print(f"Generated sequences: {len(generated_sequences)}")
    
    # ── 1. Fidelity ──
    print("\n[1/3] Computing Fidelity...")
    
    if interproscan_path and target_family_accession:
        try:
            fidelity_result = fidelity_by_interproscan(
                generated_sequences, target_family_accession, interproscan_path
            )
            results["fidelity_interproscan"] = fidelity_result
            print(f"  InterProScan Fidelity: {fidelity_result['fidelity']:.4f}")
        except RuntimeError as e:
            print(f"  InterProScan failed: {e}")
    
    if family_reference_sequences:
        sim_fidelity = fidelity_by_sequence_similarity(
            generated_sequences, family_reference_sequences
        )
        results["fidelity_similarity"] = sim_fidelity
        print(f"  Sequence similarity Fidelity: {sim_fidelity['fidelity']:.4f}")
        print(f"  Mean max identity to family: {sim_fidelity['mean_max_identity']:.4f}")
    
    # ── 2. Quality (pLDDT) ──
    print("\n[2/3] Computing pLDDT...")
    try:
        from src.metrics.plddt import calculate_plddt
        
        seqs_for_plddt = generated_sequences[:min(100, len(generated_sequences))]
        plddt_dir = os.path.join(output_dir, "pdbs")
        os.makedirs(plddt_dir, exist_ok=True)
        
        plddt_result = calculate_plddt(
            predictions=seqs_for_plddt,
            index_list=list(range(len(seqs_for_plddt))),
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
        print(f"  pLDDT skipped: {e}")
        results["plddt"] = {"error": str(e)}
    
    # ── 3. Diversity (CD0.5) ──
    print("\n[3/3] Computing Cluster Diversity (CD0.5)...")
    from src.metrics.diversity import calculate_cluster_diversity
    
    cd_result = calculate_cluster_diversity(generated_sequences, identity_threshold=0.5)
    results["cluster_diversity"] = cd_result
    print(f"  CD0.5: {cd_result['cluster_diversity']:.4f}")
    print(f"  Clusters: {cd_result['num_clusters']} / {cd_result['num_sequences']}")
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("FAMILY EVALUATION SUMMARY")
    print("=" * 60)
    
    if "fidelity_similarity" in results:
        print(f"  Fidelity:  {results['fidelity_similarity']['fidelity']:.4f}")
    if "plddt" in results and "mean" in results["plddt"]:
        print(f"  pLDDT:     {results['plddt']['mean']:.2f}")
    print(f"  CD0.5:     {cd_result['cluster_diversity']:.4f}")
    print("=" * 60)
    
    # Save
    output_path = os.path.join(output_dir, "family_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate family-specific generation")
    parser.add_argument("--generated_json", type=str, required=True)
    parser.add_argument("--family_reference_json", type=str, default=None,
                        help="JSON with known family member sequences")
    parser.add_argument("--family_accession", type=str, default=None,
                        help="InterPro/Pfam accession for the target family")
    parser.add_argument("--interproscan_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/family")
    args = parser.parse_args()
    
    with open(args.generated_json, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        generated_sequences = data
    elif isinstance(data, dict):
        generated_sequences = data.get("sequences", [])
    
    family_reference = None
    if args.family_reference_json:
        with open(args.family_reference_json, "r") as f:
            ref_data = json.load(f)
        if isinstance(ref_data, list):
            family_reference = ref_data
        elif isinstance(ref_data, dict):
            family_reference = ref_data.get("sequences", [])
    
    evaluate_family_generation(
        generated_sequences=generated_sequences,
        family_reference_sequences=family_reference,
        target_family_accession=args.family_accession,
        interproscan_path=args.interproscan_path,
        device=args.device,
        output_dir=args.output_dir,
    )
