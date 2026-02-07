"""Tests for the metrics base class and registry integration."""

import unittest
import sys
import os
import ast
from unittest.mock import MagicMock

# Mock heavy ML deps
for mod in ["torch", "torch.nn", "transformers", "numpy", "scipy", "scipy.linalg",
            "biotite", "biotite.structure", "biotite.structure.io", "cheap", "cheap.esmfold",
            "src.metrics.util", "src.metrics.fid", "src.metrics.mmd",
            "src.metrics.esmpppl", "src.metrics.plddt"]:
    sys.modules.setdefault(mod, MagicMock())

from pipeline.registry import ComponentRegistry


class TestMetricsBase(unittest.TestCase):

    def test_base_metric_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/metrics/base.py")
        self.assertTrue(os.path.exists(path))

    def test_base_metric_defines_interface(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/metrics/base.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        self.assertIn("BaseMetric", classes)

        base = classes["BaseMetric"]
        methods = [n.name for n in base.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertIn("compute", methods)

    def test_concrete_metric_classes_exist(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/metrics/base.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("FIDMetric", class_names)
        self.assertIn("MMDMetric", class_names)
        self.assertIn("ESMPPPLMetric", class_names)
        self.assertIn("PLDDTMetric", class_names)

    def test_registry_setup_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/metrics/registry_setup.py")
        self.assertTrue(os.path.exists(path))

    def test_metric_registry_decorator(self):
        reg = ComponentRegistry()

        @reg.register_metric("test_metric")
        class TestMetric:
            name = "test_metric"
            def compute(self, predictions, references=None, **kwargs):
                return 0.0

        self.assertTrue(reg.has("metric", "test_metric"))
        cls = reg.get("metric", "test_metric")
        instance = cls()
        self.assertEqual(instance.compute(["SEQ"]), 0.0)

    def test_third_party_metric_pattern(self):
        """Verify that a third party can register a custom metric."""
        reg = ComponentRegistry()

        class CustomMetric:
            name = "custom"
            requires_references = False

            def compute(self, predictions, references=None, **kwargs):
                return len(predictions) * 1.0

        reg.register("metric", "custom", CustomMetric)
        cls = reg.get("metric", "custom")
        instance = cls()
        result = instance.compute(["A", "B", "C"])
        self.assertEqual(result, 3.0)

    def test_metric_properties(self):
        """Verify that metric wrapper classes define expected properties."""
        path = os.path.join(os.path.dirname(__file__), "../../src/metrics/base.py")
        with open(path) as f:
            content = f.read()

        # Check that the concrete classes set these attributes
        self.assertIn("requires_references = True", content)  # FID/MMD
        self.assertIn("is_per_sample = True", content)  # pLDDT/ESM-pPPL
        self.assertIn("is_per_sample = False", content)  # FID/MMD


if __name__ == "__main__":
    unittest.main()
