"""
Stage: Evaluate metrics on generated sequences.

Computes quality and diversity metrics using the pluggable
metric registry.
"""

import os
import json
from typing import Dict, Any, List

from pipeline.stages.base_stage import BaseStage
from pipeline.registry import registry


@registry.register_stage("evaluate_metrics")
class EvaluateMetricsStage(BaseStage):
    name = "evaluate_metrics"
    description = "Compute evaluation metrics on generated sequences"

    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        issues = []
        # Need either generated sequences in context or a path to load from
        if "generated_sequences" not in context and "generated_sequences_path" not in context:
            issues.append("No generated sequences available. Run run_inference first.")
        return issues

    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Get generated sequences
        sequences = context.get("generated_sequences")
        if sequences is None:
            seq_path = context.get("generated_sequences_path", "")
            if seq_path and os.path.exists(seq_path):
                with open(seq_path) as f:
                    sequences = json.load(f)
            else:
                raise RuntimeError("No generated sequences available for metrics.")

        # Determine which metrics to compute
        stage_params = context.get("stage_params", {}).get("evaluate_metrics", {})
        metric_names = stage_params.get("metrics", None)

        if metric_names is None:
            # Fall back to config
            metrics_cfg = config.metrics if hasattr(config, "metrics") else config.get("metrics", {})
            if hasattr(metrics_cfg, "keys"):
                metric_names = list(metrics_cfg.keys()) if callable(metrics_cfg.keys) else list(metrics_cfg)
            else:
                metric_names = list(metrics_cfg)

        # Get reference sequences if available
        reference_sequences = context.get("reference_sequences", [])

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds = config.datasets if hasattr(config, "datasets") else config.get("datasets", {})
        max_len = ds.max_sequence_len if hasattr(ds, "max_sequence_len") else ds.get("max_sequence_len", 512)

        results = {}
        for metric_name in metric_names:
            print(f"[evaluate_metrics] Computing {metric_name}...")

            # Try registry first
            if registry.has("metric", metric_name):
                metric_cls = registry.get("metric", metric_name)
                metric = metric_cls()
                num_samples_cfg = {}
                if hasattr(config, "metrics"):
                    m_cfg = getattr(config.metrics, metric_name, None)
                    if m_cfg and hasattr(m_cfg, "num_samples"):
                        num_samples_cfg["num_samples"] = m_cfg.num_samples

                n = num_samples_cfg.get("num_samples", len(sequences))
                preds = sequences[:n]
                refs = reference_sequences[:n] if reference_sequences else None

                value = metric.compute(
                    predictions=preds,
                    references=refs,
                    device=device,
                    max_len=max_len,
                    pdb_path=context.get("pdb_path", "generated_pdbs"),
                )
                results[metric_name] = value
                print(f"[evaluate_metrics] {metric_name} = {value:.5f}")
            else:
                # Fall back to compute_ddp_metric for unregistered metrics
                from src.metrics.metric import compute_ddp_metric
                value = compute_ddp_metric(
                    metric_name=metric_name,
                    predictions=sequences,
                    references=reference_sequences,
                    max_len=max_len,
                    device=device,
                )
                results[metric_name] = value
                print(f"[evaluate_metrics] {metric_name} = {value:.5f}")

        # Save results
        output_path = context.get("metrics_output_path", "metrics_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        context["metrics_results"] = results
        context["metrics_output_path"] = output_path
        print(f"[evaluate_metrics] Results saved to {output_path}")
        print("[evaluate_metrics] Complete.")
        return context
