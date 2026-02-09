"""Tests for the pipeline orchestrator."""

import unittest
import os
import sys
import tempfile
import yaml
from unittest.mock import MagicMock, patch

# Mock heavy ML deps
for mod in ["torch", "torch.nn", "torch.distributed", "transformers", "datasets",
            "hydra", "hydra.utils", "hydra.core.global_hydra", "omegaconf",
            "cheap", "cheap.pretrained", "cheap.proteins", "cheap.esmfold",
            "src.diffusion.dima", "src.diffusion.base_trainer", "src.metrics.metric",
            "src.utils.hydra_utils", "wandb", "tqdm"]:
    sys.modules.setdefault(mod, MagicMock())

from pipeline.registry import ComponentRegistry, registry
from pipeline.orchestrator import PipelineOrchestrator

# Register test stages in the global registry for orchestrator tests
import pipeline.stages  # noqa: triggers stage registration


class TestOrchestrator(unittest.TestCase):

    def _create_pipeline_yaml(self, content: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(content, f)
        return path

    def test_from_yaml(self):
        config = {
            "stages": [
                {"name": "setup_data", "enabled": True},
            ]
        }
        path = self._create_pipeline_yaml(config)
        try:
            orch = PipelineOrchestrator.from_yaml(path)
            self.assertEqual(len(orch.pipeline_config["stages"]), 1)
        finally:
            os.unlink(path)

    def test_get_enabled_stages(self):
        config = {
            "stages": [
                {"name": "setup_data", "enabled": True},
                {"name": "setup_models", "enabled": False},
                {"name": "train_diffusion", "enabled": True},
            ]
        }
        orch = PipelineOrchestrator(config)
        enabled = orch._get_enabled_stages()
        self.assertEqual(len(enabled), 2)
        self.assertEqual(enabled[0]["name"], "setup_data")
        self.assertEqual(enabled[1]["name"], "train_diffusion")

    def test_get_enabled_stages_default(self):
        """Stages without 'enabled' key should default to True."""
        config = {
            "stages": [
                {"name": "setup_data"},
            ]
        }
        orch = PipelineOrchestrator(config)
        enabled = orch._get_enabled_stages()
        self.assertEqual(len(enabled), 1)

    def test_resolve_stage(self):
        """Should resolve a registered stage."""
        orch = PipelineOrchestrator({"stages": []})
        stage = orch._resolve_stage("setup_data")
        self.assertEqual(stage.name, "setup_data")

    def test_resolve_unknown_stage(self):
        orch = PipelineOrchestrator({"stages": []})
        with self.assertRaises(ValueError) as ctx:
            orch._resolve_stage("nonexistent_stage")
        self.assertIn("not found", str(ctx.exception))

    def test_dry_run(self):
        config = {
            "stages": [
                {"name": "setup_data", "enabled": True},
                {"name": "setup_models", "enabled": True},
            ]
        }
        orch = PipelineOrchestrator(config, dry_run=True)
        context = orch.run()
        # Dry run shouldn't produce pipeline_results (no stages executed)
        self.assertNotIn("pipeline_results", context)

    def test_validate(self):
        """Validate should return issues for stages with unmet preconditions."""
        config = {
            "stages": [
                {"name": "evaluate_metrics", "enabled": True},
            ]
        }
        orch = PipelineOrchestrator(config)
        issues = orch.validate()
        # evaluate_metrics should complain about missing sequences
        self.assertIn("evaluate_metrics", issues)

    def test_empty_pipeline(self):
        config = {"stages": []}
        orch = PipelineOrchestrator(config)
        context = orch.run()
        # Empty pipeline returns early with empty context
        self.assertEqual(context, {})

    def test_plugin_loading(self):
        """Test that plugins section is processed."""
        plugin_code = '''
class DummyMetric:
    name = "dummy"
    def compute(self, predictions, references=None, **kwargs):
        return 0.0

def register(registry):
    registry.register("metric", "dummy_test", DummyMetric)
'''
        fd, plugin_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(plugin_code)

        config = {
            "plugins": [{"path": plugin_path}],
            "stages": [],
        }
        try:
            orch = PipelineOrchestrator(config)
            orch._load_plugins()
            self.assertTrue(registry.has("metric", "dummy_test"))
        finally:
            os.unlink(plugin_path)
            registry.unregister("metric", "dummy_test")

    def test_stage_params_in_context(self):
        """When a stage is enabled, its params should be in context."""
        from pipeline.stages.base_stage import BaseStage

        class ParamCheckStage(BaseStage):
            name = "param_check"
            def validate(self, config, context):
                return []
            def run(self, config, context):
                context["saw_params"] = context.get("stage_params", {})
                return context

        registry.register("stage", "param_check", ParamCheckStage, overwrite=True)
        config = {
            "stages": [
                {
                    "name": "param_check",
                    "enabled": True,
                    "params": {"metrics": ["fid", "mmd"]},
                },
            ]
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        self.assertIn("param_check", context.get("stage_params", {}))
        self.assertEqual(context["stage_params"]["param_check"]["metrics"], ["fid", "mmd"])
        registry.unregister("stage", "param_check")


class TestOrchestratorWithCustomStages(unittest.TestCase):
    """Test the orchestrator with lightweight custom stages."""

    def setUp(self):
        # Register custom test stages
        from pipeline.stages.base_stage import BaseStage

        class PassStage(BaseStage):
            name = "test_pass"
            def validate(self, config, context):
                return []
            def run(self, config, context):
                context["test_pass_ran"] = True
                return context

        class FailStage(BaseStage):
            name = "test_fail"
            def validate(self, config, context):
                return []
            def run(self, config, context):
                raise RuntimeError("Intentional failure")

        registry.register("stage", "test_pass", PassStage, overwrite=True)
        registry.register("stage", "test_fail", FailStage, overwrite=True)

    def tearDown(self):
        registry.unregister("stage", "test_pass")
        registry.unregister("stage", "test_fail")

    def test_successful_stage_execution(self):
        config = {
            "stages": [{"name": "test_pass", "enabled": True}],
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        self.assertTrue(context.get("test_pass_ran"))
        self.assertEqual(context["pipeline_results"]["test_pass"]["status"], "success")

    def test_stage_failure_abort(self):
        config = {
            "stages": [
                {"name": "test_fail", "enabled": True, "on_failure": "abort"},
                {"name": "test_pass", "enabled": True},
            ],
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        self.assertEqual(context["pipeline_results"]["test_fail"]["status"], "failed")
        # test_pass should not have run because test_fail aborted
        self.assertNotIn("test_pass", context["pipeline_results"])

    def test_stage_failure_continue(self):
        config = {
            "stages": [
                {"name": "test_fail", "enabled": True, "on_failure": "continue"},
                {"name": "test_pass", "enabled": True},
            ],
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        self.assertEqual(context["pipeline_results"]["test_fail"]["status"], "failed")
        # test_pass should have still run despite test_fail
        self.assertEqual(context["pipeline_results"]["test_pass"]["status"], "success")
        self.assertTrue(context.get("test_pass_ran"))


class TestRunPipeline(unittest.TestCase):
    """Test the run_pipeline.py entry point."""

    def test_run_pipeline_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../../run_pipeline.py")
        self.assertTrue(os.path.exists(path))

    def test_pipeline_configs_exist(self):
        configs_dir = os.path.join(os.path.dirname(__file__), "../configs")
        for name in ["full_pipeline.yaml", "inference_only.yaml", "train_only.yaml"]:
            path = os.path.join(configs_dir, name)
            self.assertTrue(os.path.exists(path), f"Missing config: {name}")

    def test_full_pipeline_config_valid(self):
        path = os.path.join(os.path.dirname(__file__), "../configs/full_pipeline.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        self.assertIn("stages", config)
        stage_names = [s["name"] for s in config["stages"]]
        self.assertIn("setup_data", stage_names)
        self.assertIn("train_diffusion", stage_names)
        self.assertIn("evaluate_metrics", stage_names)

    def test_list_components(self):
        """Test --list-components functionality."""
        from pipeline.registry import registry
        components = registry.list_components()
        # Should have at least the stage category populated
        self.assertIn("stage", components)
        self.assertTrue(len(components["stage"]) >= 7)


if __name__ == "__main__":
    unittest.main()
