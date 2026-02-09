"""Tests for the plugin loader."""

import unittest
import os
import tempfile
from pipeline.registry import ComponentRegistry
from pipeline.plugin_loader import (
    load_plugin_from_path,
    load_plugin_from_package,
    load_plugins,
    PluginLoadError,
)


class TestPluginLoaderFromPath(unittest.TestCase):

    def setUp(self):
        self.registry = ComponentRegistry()

    def _write_plugin(self, content: str) -> str:
        """Write plugin content to a temp .py file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_load_valid_plugin(self):
        plugin_code = '''
class MyEncoder:
    pass

def register(registry):
    registry.register("encoder", "test_enc", MyEncoder)
'''
        path = self._write_plugin(plugin_code)
        try:
            name = load_plugin_from_path(path, self.registry)
            self.assertTrue(self.registry.has("encoder", "test_enc"))
            self.assertIn("dima_plugin_", name)
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        with self.assertRaises(PluginLoadError) as ctx:
            load_plugin_from_path("/nonexistent/plugin.py", self.registry)
        self.assertIn("not found", str(ctx.exception))

    def test_load_non_python_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with self.assertRaises(PluginLoadError) as ctx:
                load_plugin_from_path(path, self.registry)
            self.assertIn(".py", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_plugin_without_register(self):
        plugin_code = '''
x = 42
'''
        path = self._write_plugin(plugin_code)
        try:
            with self.assertRaises(PluginLoadError) as ctx:
                load_plugin_from_path(path, self.registry)
            self.assertIn("register(registry)", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_plugin_with_syntax_error(self):
        plugin_code = '''
def register(registry)
    pass  # missing colon
'''
        path = self._write_plugin(plugin_code)
        try:
            with self.assertRaises(PluginLoadError) as ctx:
                load_plugin_from_path(path, self.registry)
            self.assertIn("Error executing", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_plugin_register_raises(self):
        plugin_code = '''
def register(registry):
    raise RuntimeError("Intentional failure")
'''
        path = self._write_plugin(plugin_code)
        try:
            with self.assertRaises(PluginLoadError) as ctx:
                load_plugin_from_path(path, self.registry)
            self.assertIn("Error in register()", str(ctx.exception))
        finally:
            os.unlink(path)


class TestPluginLoaderFromPackage(unittest.TestCase):

    def setUp(self):
        self.registry = ComponentRegistry()

    def test_load_nonexistent_package(self):
        with self.assertRaises(PluginLoadError) as ctx:
            load_plugin_from_package("nonexistent_dima_plugin_xyz", self.registry)
        self.assertIn("Could not import", str(ctx.exception))


class TestLoadPlugins(unittest.TestCase):

    def setUp(self):
        self.registry = ComponentRegistry()

    def test_load_multiple_path_plugins(self):
        plugin1 = '''
class Enc1:
    pass
def register(registry):
    registry.register("encoder", "enc1", Enc1)
'''
        plugin2 = '''
class Met1:
    pass
def register(registry):
    registry.register("metric", "met1", Met1)
'''
        fd1, path1 = tempfile.mkstemp(suffix=".py")
        fd2, path2 = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd1, "w") as f:
            f.write(plugin1)
        with os.fdopen(fd2, "w") as f:
            f.write(plugin2)

        try:
            configs = [{"path": path1}, {"path": path2}]
            loaded = load_plugins(configs, self.registry)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(self.registry.has("encoder", "enc1"))
            self.assertTrue(self.registry.has("metric", "met1"))
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_load_invalid_config(self):
        with self.assertRaises(PluginLoadError) as ctx:
            load_plugins([{"invalid_key": "value"}], self.registry)
        self.assertIn("path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
