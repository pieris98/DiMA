"""
Integration tests for the full pipeline system.

Tests end-to-end workflows: plugin loading, registry, stage execution,
orchestrator with custom components.
"""

import unittest
import os
import sys
import tempfile
import yaml
from unittest.mock import MagicMock

# Mock heavy ML deps
for mod in ["torch", "torch.nn", "torch.distributed", "transformers", "datasets",
            "hydra", "hydra.utils", "hydra.core.global_hydra", "omegaconf",
            "cheap", "cheap.pretrained", "cheap.proteins", "cheap.esmfold",
            "src.diffusion.dima", "src.diffusion.base_trainer", "src.metrics.metric",
            "src.utils.hydra_utils", "wandb", "tqdm"]:
    sys.modules.setdefault(mod, MagicMock())

from pipeline.registry import registry
from pipeline.plugin_loader import load_plugin_from_path
from pipeline.orchestrator import PipelineOrchestrator
import pipeline.stages  # trigger registration


class TestPluginIntegration(unittest.TestCase):
    """Test loading the example plugin and using its components."""

    def setUp(self):
        self.plugin_path = os.path.join(
            os.path.dirname(__file__), "example_plugin.py"
        )

    def test_load_example_plugin(self):
        load_plugin_from_path(self.plugin_path)
        self.assertTrue(registry.has("metric", "avg_seq_length"))
        self.assertTrue(registry.has("stage", "custom_hello"))

    def test_use_custom_metric(self):
        load_plugin_from_path(self.plugin_path)
        metric_cls = registry.get("metric", "avg_seq_length")
        metric = metric_cls()
        result = metric.compute(["ACDEFG", "HIJKLMNOP"])
        expected = (6 + 9) / 2
        self.assertAlmostEqual(result, expected)

    def test_use_custom_stage(self):
        load_plugin_from_path(self.plugin_path)
        stage_cls = registry.get("stage", "custom_hello")
        stage = stage_cls()
        context = stage.run({}, {})
        self.assertTrue(context.get("custom_hello_ran"))

    def tearDown(self):
        # Clean up registrations from plugin
        registry.unregister("metric", "avg_seq_length")
        registry.unregister("stage", "custom_hello")


class TestPipelineWithPlugin(unittest.TestCase):
    """Test running the orchestrator with a plugin-provided stage."""

    def test_pipeline_with_plugin_stage(self):
        plugin_path = os.path.join(
            os.path.dirname(__file__), "example_plugin.py"
        )

        pipeline_config = {
            "plugins": [{"path": plugin_path}],
            "stages": [
                {"name": "custom_hello", "enabled": True},
            ],
        }
        orch = PipelineOrchestrator(pipeline_config)
        context = orch.run()

        self.assertTrue(context.get("custom_hello_ran"))
        self.assertEqual(context["pipeline_results"]["custom_hello"]["status"], "success")

        # Clean up
        registry.unregister("metric", "avg_seq_length")
        registry.unregister("stage", "custom_hello")


class TestMultiStageOrchestration(unittest.TestCase):
    """Test orchestrating multiple custom stages in sequence."""

    def setUp(self):
        from pipeline.stages.base_stage import BaseStage

        class StageA(BaseStage):
            name = "stage_a"
            def validate(self, config, context):
                return []
            def run(self, config, context):
                context["step_a"] = "done"
                context["counter"] = context.get("counter", 0) + 1
                return context

        class StageB(BaseStage):
            name = "stage_b"
            def validate(self, config, context):
                if "step_a" not in context:
                    return ["stage_a must run first"]
                return []
            def run(self, config, context):
                context["step_b"] = "done"
                context["counter"] = context.get("counter", 0) + 1
                return context

        class StageC(BaseStage):
            name = "stage_c"
            def validate(self, config, context):
                return []
            def run(self, config, context):
                context["step_c"] = "done"
                context["counter"] = context.get("counter", 0) + 1
                return context

        registry.register("stage", "stage_a", StageA, overwrite=True)
        registry.register("stage", "stage_b", StageB, overwrite=True)
        registry.register("stage", "stage_c", StageC, overwrite=True)

    def tearDown(self):
        registry.unregister("stage", "stage_a")
        registry.unregister("stage", "stage_b")
        registry.unregister("stage", "stage_c")

    def test_sequential_stages(self):
        config = {
            "stages": [
                {"name": "stage_a", "enabled": True},
                {"name": "stage_b", "enabled": True},
                {"name": "stage_c", "enabled": True},
            ],
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        self.assertEqual(context.get("step_a"), "done")
        self.assertEqual(context.get("step_b"), "done")
        self.assertEqual(context.get("step_c"), "done")
        self.assertEqual(context.get("counter"), 3)

    def test_selective_stages(self):
        config = {
            "stages": [
                {"name": "stage_a", "enabled": True},
                {"name": "stage_b", "enabled": False},
                {"name": "stage_c", "enabled": True},
            ],
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        self.assertEqual(context.get("step_a"), "done")
        self.assertIsNone(context.get("step_b"))
        self.assertEqual(context.get("step_c"), "done")
        self.assertEqual(context.get("counter"), 2)

    def test_context_flows_between_stages(self):
        """Verify that context written by stage_a is visible to stage_b."""
        config = {
            "stages": [
                {"name": "stage_a", "enabled": True},
                {"name": "stage_b", "enabled": True},
            ],
        }
        orch = PipelineOrchestrator(config)
        context = orch.run()
        # stage_b should validate successfully because stage_a ran first
        self.assertEqual(context["pipeline_results"]["stage_b"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
