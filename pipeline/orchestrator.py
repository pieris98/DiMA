"""
Pipeline orchestrator for DiMA.

Reads a pipeline YAML configuration and runs stages in sequence,
passing a shared context between them. Supports:
  - Plugin loading
  - Stage validation before execution
  - Dry-run mode
  - Selective stage enable/disable
  - Stage-specific parameter overrides
"""

import os
import sys
import time
import traceback
from typing import Dict, Any, List, Optional

import yaml

from pipeline.registry import registry
from pipeline.plugin_loader import load_plugins


class PipelineOrchestrator:
    """
    Orchestrates the execution of a sequence of pipeline stages.

    Usage:
        orchestrator = PipelineOrchestrator(pipeline_config_path, hydra_config)
        orchestrator.run()
    """

    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        hydra_config: Any = None,
        dry_run: bool = False,
    ):
        """
        Args:
            pipeline_config: Pipeline config dict (loaded from YAML).
            hydra_config: The Hydra DictConfig for the project.
            dry_run: If True, validate but don't execute stages.
        """
        self.pipeline_config = pipeline_config
        self.hydra_config = hydra_config
        self.dry_run = dry_run
        self.context: Dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, yaml_path: str, hydra_config: Any = None, dry_run: bool = False):
        """Load pipeline config from a YAML file."""
        with open(yaml_path) as f:
            pipeline_config = yaml.safe_load(f)
        return cls(pipeline_config, hydra_config, dry_run)

    def _load_plugins(self):
        """Load plugins specified in the pipeline config."""
        plugins = self.pipeline_config.get("plugins", [])
        if plugins:
            print(f"[orchestrator] Loading {len(plugins)} plugin(s)...")
            loaded = load_plugins(plugins)
            print(f"[orchestrator] Loaded plugins: {loaded}")
        return plugins

    def _get_enabled_stages(self) -> List[Dict[str, Any]]:
        """Get the list of enabled stages from the pipeline config."""
        stages_config = self.pipeline_config.get("stages", [])
        enabled = []
        for stage_cfg in stages_config:
            if stage_cfg.get("enabled", True):
                enabled.append(stage_cfg)
        return enabled

    def _resolve_stage(self, stage_name: str):
        """Look up a stage class from the registry and instantiate it."""
        if not registry.has("stage", stage_name):
            raise ValueError(
                f"Stage '{stage_name}' not found in registry. "
                f"Available stages: {registry.list_components('stage')['stage']}"
            )
        stage_cls = registry.get("stage", stage_name)
        return stage_cls()

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate all enabled stages.

        Returns:
            Dict mapping stage name -> list of validation issues.
            Empty lists mean the stage is valid.
        """
        enabled_stages = self._get_enabled_stages()
        all_issues = {}

        for stage_cfg in enabled_stages:
            name = stage_cfg["name"]
            try:
                stage = self._resolve_stage(name)
                issues = stage.validate(self.hydra_config or {}, self.context)
                if issues:
                    all_issues[name] = issues
            except Exception as e:
                all_issues[name] = [f"Error resolving stage: {e}"]

        return all_issues

    def run(self) -> Dict[str, Any]:
        """
        Execute the pipeline.

        Returns:
            The final context dict after all stages have run.
        """
        print("=" * 60)
        print("  DiMA Pipeline Orchestrator")
        print("=" * 60)

        # Load plugins
        self._load_plugins()

        # Load stages from the pipeline config
        # Import stages module to trigger registration
        import pipeline.stages  # noqa: F401

        enabled_stages = self._get_enabled_stages()
        if not enabled_stages:
            print("[orchestrator] No stages enabled. Nothing to do.")
            return self.context

        print(f"\n[orchestrator] Pipeline stages ({len(enabled_stages)}):")
        for i, s in enumerate(enabled_stages, 1):
            print(f"  {i}. {s['name']}")

        # Inject pipeline-level context
        self.context["config_path"] = self.pipeline_config.get("config_path", "../configs")
        self.context["stage_params"] = {}
        for stage_cfg in enabled_stages:
            params = stage_cfg.get("params", {})
            if params:
                self.context["stage_params"][stage_cfg["name"]] = params

        # Inject Hydra overrides if present
        hydra_overrides = self.pipeline_config.get("hydra_overrides", [])
        if hydra_overrides:
            self.context["hydra_overrides"] = hydra_overrides

        # Validate
        if not self.dry_run:
            print("\n[orchestrator] Validating stages...")
            issues = self.validate()
            if issues:
                print("[orchestrator] Validation warnings:")
                for name, issue_list in issues.items():
                    for issue in issue_list:
                        print(f"  [{name}] {issue}")
                # Warnings don't block execution; stages handle their own errors

        if self.dry_run:
            print("\n[orchestrator] DRY RUN — showing what would execute:")
            for i, stage_cfg in enumerate(enabled_stages, 1):
                name = stage_cfg["name"]
                params = stage_cfg.get("params", {})
                param_str = f" (params: {params})" if params else ""
                print(f"  {i}. {name}{param_str}")
            print("\n[orchestrator] Dry run complete. No stages were executed.")
            return self.context

        # Execute stages
        print("\n[orchestrator] Starting pipeline execution...")
        results = {}

        for i, stage_cfg in enumerate(enabled_stages, 1):
            name = stage_cfg["name"]
            print(f"\n{'─' * 50}")
            print(f"  Stage {i}/{len(enabled_stages)}: {name}")
            print(f"{'─' * 50}")

            start_time = time.time()
            try:
                stage = self._resolve_stage(name)
                self.context = stage.run(self.hydra_config or {}, self.context)
                elapsed = time.time() - start_time
                results[name] = {"status": "success", "elapsed_seconds": round(elapsed, 2)}
                print(f"  [{name}] completed in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                results[name] = {
                    "status": "failed",
                    "error": str(e),
                    "elapsed_seconds": round(elapsed, 2),
                }
                print(f"  [{name}] FAILED after {elapsed:.1f}s: {e}")
                traceback.print_exc()

                on_failure = stage_cfg.get("on_failure", "abort")
                if on_failure == "abort":
                    print(f"\n[orchestrator] Aborting pipeline due to failure in '{name}'.")
                    break
                elif on_failure == "continue":
                    print(f"  [{name}] on_failure=continue, proceeding to next stage.")
                    continue

        # Summary
        print(f"\n{'=' * 60}")
        print("  Pipeline Summary")
        print(f"{'=' * 60}")
        for name, res in results.items():
            status = res["status"]
            elapsed = res["elapsed_seconds"]
            marker = "OK" if status == "success" else "FAIL"
            print(f"  [{marker}] {name} ({elapsed}s)")

        self.context["pipeline_results"] = results
        return self.context
