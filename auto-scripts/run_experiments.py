"""
End-to-end experiment runner for DiMA evaluation and family-specific generation.

This script orchestrates the full experiment pipeline:

1. Unconditional Generation + Comprehensive Evaluation
   - Generate sequences with the pretrained DiMA model
   - Run all sequence-modality metrics (FD-seq, pLDDT, CD0.5, Novelty, etc.)

2. Family-Specific Generation
   a. Prepare family datasets
   b. Train family classifier
   c. Generate family-conditioned sequences via classifier guidance
   d. Evaluate family generation (Fidelity, pLDDT, CD0.5)

Usage:
    # Run unconditional evaluation only
    python auto-scripts/run_experiments.py --mode unconditional --encoder esm2

    # Run family-specific experiments
    python auto-scripts/run_experiments.py --mode family --encoder esm2

    # Run everything
    python auto-scripts/run_experiments.py --mode all --encoder esm2
"""

import os
import sys
import json
import argparse
import time

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_unconditional_evaluation(args):
    """
    Step 1: Generate sequences unconditionally and evaluate them.
    """
    import torch
    from src.diffusion.dima import DiMAModel
    
    print("=" * 70)
    print("STEP 1: Unconditional Generation + Comprehensive Evaluation")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "configs")
    
    # Build encoder overrides if needed
    overrides = []
    if args.encoder_overrides:
        overrides = args.encoder_overrides
    
    # Initialize model
    print("\n[1.1] Loading DiMA model...")
    model = DiMAModel(config_path=config_path, device=device, overrides=overrides or None)
    model.load_pretrained()
    model.score_estimator.eval()
    
    # Generate sequences
    num_gen = args.num_samples
    print(f"\n[1.2] Generating {num_gen} sequences...")
    t_start = time.time()
    generated_sequences = model.generate_samples(num_texts=num_gen)
    gen_time = time.time() - t_start
    print(f"  Generated {len(generated_sequences)} sequences in {gen_time:.1f}s")
    
    # Save generated sequences
    output_dir = os.path.join(args.output_dir, "unconditional")
    os.makedirs(output_dir, exist_ok=True)
    
    gen_path = os.path.join(output_dir, "generated_sequences.json")
    with open(gen_path, "w") as f:
        json.dump(generated_sequences, f, indent=2)
    print(f"  Saved to {gen_path}")
    
    # Load reference sequences if available
    reference_sequences = None
    training_sequences = None
    
    if args.reference_json:
        with open(args.reference_json, "r") as f:
            ref_data = json.load(f)
        reference_sequences = ref_data if isinstance(ref_data, list) else ref_data.get("sequences", [])
        print(f"\n  Reference sequences: {len(reference_sequences)}")
    
    if args.training_json:
        with open(args.training_json, "r") as f:
            train_data = json.load(f)
        training_sequences = train_data if isinstance(train_data, list) else train_data.get("sequences", [])
        print(f"  Training sequences: {len(training_sequences)}")
    
    # Run comprehensive evaluation
    print(f"\n[1.3] Running comprehensive evaluation...")
    
    # Free up GPU memory from the DiMA model before evaluation
    del model
    torch.cuda.empty_cache()
    
    from src.evaluation import run_evaluation
    
    results = run_evaluation(
        generated_sequences=generated_sequences,
        reference_sequences=reference_sequences,
        training_sequences=training_sequences,
        metrics=set(args.metrics),
        max_len=254,
        device=str(device),
        output_dir=output_dir,
        num_samples_heavy=min(args.num_samples_heavy, len(generated_sequences)),
        num_samples_light=min(args.num_samples_light, len(generated_sequences)),
    )
    
    results["generation_time_seconds"] = gen_time
    results["num_generated"] = len(generated_sequences)
    
    return results


def run_family_experiments(args):
    """
    Step 2: Family-specific conditional generation experiments.
    """
    import torch
    
    print("\n" + "=" * 70)
    print("STEP 2: Family-Specific Conditional Generation")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "configs")
    
    family_output_dir = os.path.join(args.output_dir, "family_generation")
    os.makedirs(family_output_dir, exist_ok=True)
    
    # ── Step 2.1: Prepare family data ──
    print("\n[2.1] Preparing family dataset...")
    family_data_dir = os.path.join(args.output_dir, "family_data")
    
    if args.families_json and os.path.exists(args.families_json):
        print(f"  Using provided family data: {args.families_json}")
        train_path = args.families_json
        val_path = args.families_val_json
    else:
        from src.family_generation.prepare_family_data import (
            FAMILY_DEFINITIONS, save_family_dataset
        )
        import numpy as np
        
        print("  Generating test family dataset...")
        rng = np.random.RandomState(42)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        
        family_seqs = {}
        for fname in FAMILY_DEFINITIONS:
            seqs = []
            for _ in range(50):
                length = rng.randint(64, 200)
                seq = "".join(rng.choice(list(aa)) for _ in range(length))
                seqs.append(seq)
            family_seqs[fname] = seqs
        
        train_path, val_path, _ = save_family_dataset(family_seqs, family_data_dir)
    
    # ── Step 2.2: Train family classifier ──
    print("\n[2.2] Training family classifier...")
    from src.family_generation import (
        FamilySequenceDataset, train_classifier
    )
    from src.utils.hydra_utils import setup_config
    
    config = setup_config(config_path=config_path, overrides=args.encoder_overrides or None)
    
    train_dataset = FamilySequenceDataset(train_path)
    val_dataset = FamilySequenceDataset(val_path, train_dataset.family_to_idx) if val_path else None
    
    classifier_dir = os.path.join(family_output_dir, "classifier")
    
    classifier = train_classifier(
        config=config,
        dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=classifier_dir,
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        device=device,
    )
    
    # ── Step 2.3: Generate family-conditioned sequences ──
    print("\n[2.3] Generating family-conditioned sequences...")
    from src.diffusion.dima import DiMAModel
    from src.family_generation.generate import ClassifierGuidedSampler, load_family_classifier
    
    # Load model
    model = DiMAModel(config_path=config_path, device=device, overrides=args.encoder_overrides or None)
    model.load_pretrained()
    model.score_estimator.eval()
    
    # Load best classifier
    clf_path = os.path.join(classifier_dir, "final_classifier.pth")
    classifier_loaded, family_to_idx = load_family_classifier(clf_path, device)
    
    sampler = ClassifierGuidedSampler(
        model=model,
        classifier=classifier_loaded,
        family_to_idx=family_to_idx,
        guidance_scale=args.guidance_scale,
    )
    
    all_family_results = {}
    
    for family_name in family_to_idx.keys():
        print(f"\n  Generating for family: {family_name}")
        family_seqs = sampler.generate_family_sequences(
            family_name=family_name,
            num_sequences=args.num_family_samples,
            guidance_scale=args.guidance_scale,
        )
        
        # Save
        family_file = os.path.join(
            family_output_dir, 
            f"generated_{family_name.replace(' ', '_').lower()}.json"
        )
        with open(family_file, "w") as f:
            json.dump({
                "family": family_name,
                "guidance_scale": args.guidance_scale,
                "sequences": family_seqs,
            }, f, indent=2)
        print(f"  Saved {len(family_seqs)} sequences to {family_file}")
        
        all_family_results[family_name] = family_seqs
    
    # ── Step 2.4: Evaluate family generation ──
    print("\n[2.4] Evaluating family-specific generation...")
    
    # Free model memory
    del model
    del sampler
    torch.cuda.empty_cache()
    
    from src.evaluation.evaluate_family import evaluate_family_generation
    from src.metrics.diversity import calculate_cluster_diversity
    
    family_eval_results = {}
    for family_name, seqs in all_family_results.items():
        print(f"\n  Evaluating family: {family_name}")
        
        eval_dir = os.path.join(family_output_dir, "evaluation", 
                               family_name.replace(" ", "_").lower())
        
        # Get reference sequences for this family from training data
        family_refs = [d["sequence"] for d in json.load(open(train_path)) 
                      if d["family"] == family_name]
        
        eval_result = evaluate_family_generation(
            generated_sequences=seqs,
            family_reference_sequences=family_refs if family_refs else None,
            device=str(device),
            output_dir=eval_dir,
        )
        
        family_eval_results[family_name] = eval_result
    
    # Save combined results
    combined_path = os.path.join(family_output_dir, "all_family_results.json")
    with open(combined_path, "w") as f:
        json.dump(family_eval_results, f, indent=2, default=str)
    print(f"\nCombined family results saved to {combined_path}")
    
    return family_eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiMA Experiment Runner")
    parser.add_argument("--mode", choices=["unconditional", "family", "all"], default="unconditional",
                        help="Which experiments to run")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Base output directory")
    
    # Unconditional generation args
    parser.add_argument("--num_samples", type=int, default=2048,
                        help="Number of sequences to generate for unconditional evaluation")
    parser.add_argument("--num_samples_heavy", type=int, default=512,
                        help="Number of samples for heavy metrics (pLDDT, PPPL)")
    parser.add_argument("--num_samples_light", type=int, default=2048,
                        help="Number of samples for light metrics")
    parser.add_argument("--reference_json", type=str, default=None,
                        help="Reference sequences JSON for distribution metrics")
    parser.add_argument("--training_json", type=str, default=None,
                        help="Training sequences JSON for novelty")
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        help="Metrics to compute")
    
    # Family generation args
    parser.add_argument("--families_json", type=str, default=None,
                        help="Pre-prepared family training data JSON")
    parser.add_argument("--families_val_json", type=str, default=None,
                        help="Pre-prepared family validation data JSON")
    parser.add_argument("--classifier_epochs", type=int, default=30,
                        help="Number of epochs for classifier training")
    parser.add_argument("--num_family_samples", type=int, default=100,
                        help="Number of sequences per family")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance scale")
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Encoder args
    parser.add_argument("--encoder", type=str, default="esm2",
                        help="Encoder config name (esm2, cheap, saprot)")
    parser.add_argument("--encoder_overrides", nargs="*", default=None,
                        help="Additional Hydra config overrides")
    
    args = parser.parse_args()
    
    print("DiMA Experiment Runner")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    print()
    
    if args.mode in ["unconditional", "all"]:
        run_unconditional_evaluation(args)
    
    if args.mode in ["family", "all"]:
        run_family_experiments(args)
    
    print("\n✓ All experiments complete!")
