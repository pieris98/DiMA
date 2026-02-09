"""Tests for the decoder abstraction layer.

These tests verify the decoder module structure and registry integration
without requiring torch or heavy ML dependencies.
"""

import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock torch and related modules before importing our code
_mock_torch = MagicMock()
_mock_torch.nn = MagicMock()
_mock_torch.nn.Module = type("Module", (), {"__init__": lambda self: None})
_mock_torch.Tensor = MagicMock
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.nn", _mock_torch.nn)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("omegaconf", MagicMock())

# Now import — these should work with mocked deps
from pipeline.registry import ComponentRegistry


class TestDecoderRegistry(unittest.TestCase):

    def test_decoder_base_file_exists(self):
        base_path = os.path.join(os.path.dirname(__file__), "../../src/decoders/base.py")
        self.assertTrue(os.path.exists(base_path), "src/decoders/base.py should exist")

    def test_decoder_transformer_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/decoders/transformer.py")
        self.assertTrue(os.path.exists(path), "src/decoders/transformer.py should exist")

    def test_decoder_lm_head_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/decoders/lm_head.py")
        self.assertTrue(os.path.exists(path), "src/decoders/lm_head.py should exist")

    def test_decoder_init_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "../../src/decoders/__init__.py")
        self.assertTrue(os.path.exists(path), "src/decoders/__init__.py should exist")

    def test_registry_decorator_registers_decoders(self):
        """Test that the decorator pattern works for decoder registration."""
        reg = ComponentRegistry()

        @reg.register_decoder("test_decoder")
        class TestDecoder:
            pass

        self.assertTrue(reg.has("decoder", "test_decoder"))
        self.assertIs(reg.get("decoder", "test_decoder"), TestDecoder)

    def test_base_decoder_defines_interface(self):
        """Verify base.py defines the expected class structure."""
        import ast
        base_path = os.path.join(os.path.dirname(__file__), "../../src/decoders/base.py")
        with open(base_path) as f:
            tree = ast.parse(f.read())

        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        class_names = [c.name for c in classes]
        self.assertIn("BaseDecoder", class_names)

        # Check BaseDecoder has the expected methods
        base_cls = [c for c in classes if c.name == "BaseDecoder"][0]
        method_names = [n.name for n in base_cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertIn("forward", method_names)
        self.assertIn("load_checkpoint", method_names)
        self.assertIn("decode_to_sequences", method_names)

    def test_transformer_wrapper_defines_interface(self):
        """Verify transformer.py wraps properly."""
        import ast
        path = os.path.join(os.path.dirname(__file__), "../../src/decoders/transformer.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        class_names = [c.name for c in classes]
        self.assertIn("TransformerDecoderWrapper", class_names)

    def test_lm_head_decoder_defines_interface(self):
        """Verify lm_head.py wraps properly."""
        import ast
        path = os.path.join(os.path.dirname(__file__), "../../src/decoders/lm_head.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        class_names = [c.name for c in classes]
        self.assertIn("LMHeadDecoder", class_names)


if __name__ == "__main__":
    unittest.main()
