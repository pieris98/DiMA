#!/usr/bin/env python
"""
DiMA Pipeline Runner

Single entry point for running the modular ML pipeline.

Usage:
    # Full pipeline with defaults
    python run_pipeline.py --pipeline pipeline/configs/full_pipeline.yaml

    # Training only
    python run_pipeline.py --pipeline pipeline/configs/train_only.yaml

    # Inference only with checkpoint
    python run_pipeline.py --pipeline pipeline/configs/inference_only.yaml \
        --checkpoint /path/to/checkpoint.pth

    # With Hydra config overrides
    python run_pipeline.py --pipeline pipeline/configs/full_pipeline.yaml \
        --hydra-overrides encoder=cheap datasets=swissprot

    # Dry run (validate without executing)
    python run_pipeline.py --pipeline pipeline/configs/full_pipeline.yaml --dry-run

    # List available components
    python run_pipeline.py --list-components
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def list_components():
    """Print all registered components."""
    from pipeline.registry import registry

    # Trigger registration by importing stages
    import pipeline.stages  # noqa: F401

    print("\nRegistered Components:")
    print("=" * 40)
    components = registry.list_components()
    for category, names in components.items():
        if names:
            print(f"\n  {category}:")
            for name in names:
                print(f"    - {name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="DiMA Modular ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="pipeline/configs/full_pipeline.yaml",
        help="Path to pipeline YAML config (default: pipeline/configs/full_pipeline.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate stages without executing them",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to diffusion model checkpoint (for inference)",
    )
    parser.add_argument(
        "--hydra-overrides",
        nargs="*",
        default=None,
        help="Hydra config overrides (e.g., encoder=cheap datasets=swissprot)",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="List all registered components and exit",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed training (default: 1)",
    )

    args = parser.parse_args()

    if args.list_components:
        list_components()
        return

    if not os.path.exists(args.pipeline):
        print(f"Error: Pipeline config not found: {args.pipeline}")
        sys.exit(1)

    from pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator.from_yaml(
        yaml_path=args.pipeline,
        hydra_config=None,  # Will be loaded by stages as needed
        dry_run=args.dry_run,
    )

    # Inject CLI args into context
    if args.checkpoint:
        orchestrator.context["checkpoint_path"] = args.checkpoint
    if args.hydra_overrides:
        orchestrator.context["hydra_overrides"] = args.hydra_overrides
        # Also update the pipeline config
        existing = orchestrator.pipeline_config.get("hydra_overrides", [])
        orchestrator.pipeline_config["hydra_overrides"] = existing + args.hydra_overrides
    if args.num_gpus:
        orchestrator.context["num_gpus"] = args.num_gpus

    context = orchestrator.run()

    # Print final results
    results = context.get("pipeline_results", {})
    failed = [name for name, r in results.items() if r["status"] == "failed"]
    if failed:
        print(f"\nPipeline completed with {len(failed)} failure(s).")
        sys.exit(1)
    else:
        print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
