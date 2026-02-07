"""Tests for the component registry."""

import unittest
from pipeline.registry import ComponentRegistry, VALID_CATEGORIES


class TestComponentRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ComponentRegistry()

    def test_register_and_get(self):
        class DummyEncoder:
            pass

        self.registry.register("encoder", "dummy", DummyEncoder)
        result = self.registry.get("encoder", "dummy")
        self.assertIs(result, DummyEncoder)

    def test_register_invalid_category(self):
        with self.assertRaises(ValueError) as ctx:
            self.registry.register("invalid_cat", "foo", object)
        self.assertIn("Invalid category", str(ctx.exception))

    def test_register_empty_name(self):
        with self.assertRaises(ValueError):
            self.registry.register("encoder", "", object)

    def test_register_duplicate_raises(self):
        self.registry.register("encoder", "dup", object)
        with self.assertRaises(ValueError) as ctx:
            self.registry.register("encoder", "dup", object)
        self.assertIn("already registered", str(ctx.exception))

    def test_register_duplicate_overwrite(self):
        class V1:
            pass

        class V2:
            pass

        self.registry.register("encoder", "versioned", V1)
        self.registry.register("encoder", "versioned", V2, overwrite=True)
        self.assertIs(self.registry.get("encoder", "versioned"), V2)

    def test_get_missing_component(self):
        with self.assertRaises(KeyError) as ctx:
            self.registry.get("encoder", "nonexistent")
        self.assertIn("No component", str(ctx.exception))

    def test_get_invalid_category(self):
        with self.assertRaises(ValueError):
            self.registry.get("bad_cat", "foo")

    def test_has(self):
        self.assertFalse(self.registry.has("encoder", "test"))
        self.registry.register("encoder", "test", object)
        self.assertTrue(self.registry.has("encoder", "test"))

    def test_has_invalid_category(self):
        self.assertFalse(self.registry.has("invalid", "test"))

    def test_list_components_single_category(self):
        self.registry.register("metric", "fid", object)
        self.registry.register("metric", "mmd", object)
        result = self.registry.list_components("metric")
        self.assertEqual(result, {"metric": ["fid", "mmd"]})

    def test_list_components_all(self):
        self.registry.register("encoder", "e1", object)
        self.registry.register("decoder", "d1", object)
        result = self.registry.list_components()
        self.assertIn("encoder", result)
        self.assertIn("decoder", result)
        self.assertIn("e1", result["encoder"])
        self.assertIn("d1", result["decoder"])

    def test_list_components_invalid_category(self):
        with self.assertRaises(ValueError):
            self.registry.list_components("invalid")

    def test_unregister(self):
        self.registry.register("encoder", "to_remove", object)
        self.assertTrue(self.registry.has("encoder", "to_remove"))
        self.registry.unregister("encoder", "to_remove")
        self.assertFalse(self.registry.has("encoder", "to_remove"))

    def test_unregister_nonexistent_silent(self):
        # Should not raise
        self.registry.unregister("encoder", "nonexistent")

    def test_clear_category(self):
        self.registry.register("encoder", "e1", object)
        self.registry.register("encoder", "e2", object)
        self.registry.register("decoder", "d1", object)
        self.registry.clear("encoder")
        self.assertFalse(self.registry.has("encoder", "e1"))
        self.assertFalse(self.registry.has("encoder", "e2"))
        self.assertTrue(self.registry.has("decoder", "d1"))

    def test_clear_all(self):
        self.registry.register("encoder", "e1", object)
        self.registry.register("decoder", "d1", object)
        self.registry.clear()
        self.assertFalse(self.registry.has("encoder", "e1"))
        self.assertFalse(self.registry.has("decoder", "d1"))

    def test_decorator_register_encoder(self):
        @self.registry.register_encoder("decorated_enc")
        class MyEncoder:
            pass

        self.assertIs(self.registry.get("encoder", "decorated_enc"), MyEncoder)

    def test_decorator_register_decoder(self):
        @self.registry.register_decoder("decorated_dec")
        class MyDecoder:
            pass

        self.assertIs(self.registry.get("decoder", "decorated_dec"), MyDecoder)

    def test_decorator_register_metric(self):
        @self.registry.register_metric("decorated_metric")
        class MyMetric:
            pass

        self.assertIs(self.registry.get("metric", "decorated_metric"), MyMetric)

    def test_decorator_register_stage(self):
        @self.registry.register_stage("decorated_stage")
        class MyStage:
            pass

        self.assertIs(self.registry.get("stage", "decorated_stage"), MyStage)

    def test_decorator_register_dataset(self):
        @self.registry.register_dataset("decorated_ds")
        class MyDataset:
            pass

        self.assertIs(self.registry.get("dataset", "decorated_ds"), MyDataset)

    def test_valid_categories_are_complete(self):
        expected = {"encoder", "decoder", "metric", "stage", "dataset"}
        self.assertEqual(set(VALID_CATEGORIES), expected)


if __name__ == "__main__":
    unittest.main()
