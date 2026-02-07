"""Tests for pipeline stages."""

import unittest
import os
import ast
import sys
from unittest.mock import MagicMock

# Mock heavy deps
for mod in ["torch", "torch.nn", "torch.distributed", "transformers", "datasets",
            "hydra", "hydra.utils", "hydra.core.global_hydra", "omegaconf",
            "cheap", "cheap.pretrained", "cheap.proteins", "cheap.esmfold",
            "src.diffusion.dima", "src.diffusion.base_trainer", "src.metrics.metric",
            "src.utils.hydra_utils", "wandb", "tqdm"]:
    sys.modules.setdefault(mod, MagicMock())

from pipeline.registry import ComponentRegistry, registry


class TestBaseStage(unittest.TestCase):

    def test_base_stage_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../stages/base_stage.py")
        self.assertTrue(os.path.exists(path))

    def test_base_stage_defines_interface(self):
        path = os.path.join(os.path.dirname(__file__), "../stages/base_stage.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        self.assertIn("BaseStage", classes)

        base = classes["BaseStage"]
        methods = [n.name for n in base.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertIn("validate", methods)
        self.assertIn("run", methods)


class TestStageRegistration(unittest.TestCase):

    def test_all_stage_files_exist(self):
        stages_dir = os.path.join(os.path.dirname(__file__), "../stages")
        expected_files = [
            "base_stage.py",
            "setup_data.py",
            "setup_models.py",
            "calculate_statistics.py",
            "train_decoder.py",
            "train_diffusion.py",
            "run_inference.py",
            "evaluate_metrics.py",
        ]
        for fname in expected_files:
            path = os.path.join(stages_dir, fname)
            self.assertTrue(os.path.exists(path), f"Missing stage file: {fname}")

    def test_stages_define_classes(self):
        """Verify each stage file defines a class that extends BaseStage."""
        stages_dir = os.path.join(os.path.dirname(__file__), "../stages")
        stage_files = {
            "setup_data.py": "SetupDataStage",
            "setup_models.py": "SetupModelsStage",
            "calculate_statistics.py": "CalculateStatisticsStage",
            "train_decoder.py": "TrainDecoderStage",
            "train_diffusion.py": "TrainDiffusionStage",
            "run_inference.py": "RunInferenceStage",
            "evaluate_metrics.py": "EvaluateMetricsStage",
        }
        for fname, class_name in stage_files.items():
            path = os.path.join(stages_dir, fname)
            with open(path) as f:
                tree = ast.parse(f.read())
            class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            self.assertIn(class_name, class_names, f"Class {class_name} not found in {fname}")

    def test_stages_implement_validate_and_run(self):
        stages_dir = os.path.join(os.path.dirname(__file__), "../stages")
        stage_files = [
            "setup_data.py",
            "setup_models.py",
            "calculate_statistics.py",
            "train_decoder.py",
            "train_diffusion.py",
            "run_inference.py",
            "evaluate_metrics.py",
        ]
        for fname in stage_files:
            path = os.path.join(stages_dir, fname)
            with open(path) as f:
                tree = ast.parse(f.read())
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            for cls in classes:
                if cls.name == "BaseStage":
                    continue
                methods = [n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                self.assertIn("validate", methods, f"{cls.name} missing validate()")
                self.assertIn("run", methods, f"{cls.name} missing run()")

    def test_stages_registered_in_registry(self):
        """Stages use @registry.register_stage decorator."""
        # Since we can't import the stages directly (torch not available),
        # we verify by parsing the source for the decorator pattern
        stages_dir = os.path.join(os.path.dirname(__file__), "../stages")
        expected = {
            "setup_data.py": "setup_data",
            "setup_models.py": "setup_models",
            "calculate_statistics.py": "calculate_statistics",
            "train_decoder.py": "train_decoder",
            "train_diffusion.py": "train_diffusion",
            "run_inference.py": "run_inference",
            "evaluate_metrics.py": "evaluate_metrics",
        }
        for fname, stage_name in expected.items():
            path = os.path.join(stages_dir, fname)
            with open(path) as f:
                content = f.read()
            self.assertIn(f'@registry.register_stage("{stage_name}")', content,
                         f"Stage '{stage_name}' not registered in {fname}")


class TestStageValidation(unittest.TestCase):

    def test_custom_stage_registration(self):
        """Verify a custom stage can be registered and retrieved."""
        reg = ComponentRegistry()

        @reg.register_stage("custom_stage")
        class CustomStage:
            name = "custom_stage"
            def validate(self, config, context):
                return []
            def run(self, config, context):
                context["custom_ran"] = True
                return context

        self.assertTrue(reg.has("stage", "custom_stage"))
        stage_cls = reg.get("stage", "custom_stage")
        stage = stage_cls()
        ctx = stage.run({}, {})
        self.assertTrue(ctx["custom_ran"])

    def test_setup_data_validate_no_datasets(self):
        """SetupDataStage.validate should flag missing datasets config."""
        from pipeline.stages.setup_data import SetupDataStage
        stage = SetupDataStage()
        issues = stage.validate({}, {})
        self.assertTrue(len(issues) > 0)

    def test_evaluate_metrics_validate_no_sequences(self):
        """EvaluateMetricsStage.validate should flag missing sequences."""
        from pipeline.stages.evaluate_metrics import EvaluateMetricsStage
        stage = EvaluateMetricsStage()
        issues = stage.validate({}, {})
        self.assertTrue(len(issues) > 0)
        self.assertIn("generated sequences", issues[0].lower())


if __name__ == "__main__":
    unittest.main()
