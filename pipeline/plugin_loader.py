"""
Plugin loader for DiMA pipeline.

Discovers and loads third-party plugins from:
  1. Python files specified by path
  2. Installed Python packages

Each plugin module must define a `register(registry)` function that
registers its components into the global registry.
"""

import importlib
import importlib.util
import sys
import os
from typing import List, Optional, Union

from pipeline.registry import ComponentRegistry, registry as global_registry


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""
    pass


def load_plugin_from_path(path: str, reg: Optional[ComponentRegistry] = None) -> str:
    """
    Load a plugin from a Python file path.

    The module must define a `register(registry)` function.

    Args:
        path: Absolute or relative path to a .py file.
        reg: Registry to pass to the plugin. Defaults to global registry.

    Returns:
        The module name that was loaded.

    Raises:
        PluginLoadError: If the file doesn't exist, can't be loaded,
                         or doesn't define a register() function.
    """
    reg = reg or global_registry

    if not os.path.isfile(path):
        raise PluginLoadError(f"Plugin file not found: {path}")

    if not path.endswith(".py"):
        raise PluginLoadError(f"Plugin file must be a .py file: {path}")

    module_name = os.path.splitext(os.path.basename(path))[0]
    # Avoid collisions with existing modules
    full_module_name = f"dima_plugin_{module_name}"

    spec = importlib.util.spec_from_file_location(full_module_name, path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Could not create module spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[full_module_name]
        raise PluginLoadError(f"Error executing plugin {path}: {e}") from e

    if not hasattr(module, "register") or not callable(module.register):
        del sys.modules[full_module_name]
        raise PluginLoadError(
            f"Plugin {path} must define a callable 'register(registry)' function."
        )

    try:
        module.register(reg)
    except Exception as e:
        raise PluginLoadError(
            f"Error in register() of plugin {path}: {e}"
        ) from e

    return full_module_name


def load_plugin_from_package(package_name: str, reg: Optional[ComponentRegistry] = None) -> str:
    """
    Load a plugin from an installed Python package.

    The package must define a `register(registry)` function at its top level.

    Args:
        package_name: Importable package name (e.g., "dima_plugin_esmfold").
        reg: Registry to pass to the plugin. Defaults to global registry.

    Returns:
        The package name.

    Raises:
        PluginLoadError: If the package can't be imported or doesn't define register().
    """
    reg = reg or global_registry

    try:
        module = importlib.import_module(package_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Could not import plugin package '{package_name}': {e}"
        ) from e

    if not hasattr(module, "register") or not callable(module.register):
        raise PluginLoadError(
            f"Plugin package '{package_name}' must define a callable "
            f"'register(registry)' function."
        )

    try:
        module.register(reg)
    except Exception as e:
        raise PluginLoadError(
            f"Error in register() of plugin package '{package_name}': {e}"
        ) from e

    return package_name


def load_plugins(
    plugin_configs: List[dict],
    reg: Optional[ComponentRegistry] = None,
) -> List[str]:
    """
    Load multiple plugins from a list of plugin configuration dicts.

    Each dict should have either:
      - {"path": "/path/to/plugin.py"}
      - {"package": "installed_package_name"}

    Args:
        plugin_configs: List of plugin config dicts.
        reg: Registry to use. Defaults to global registry.

    Returns:
        List of loaded module/package names.

    Raises:
        PluginLoadError: If any plugin fails to load.
    """
    reg = reg or global_registry
    loaded = []

    for plugin_cfg in plugin_configs:
        if "path" in plugin_cfg:
            name = load_plugin_from_path(plugin_cfg["path"], reg)
        elif "package" in plugin_cfg:
            name = load_plugin_from_package(plugin_cfg["package"], reg)
        else:
            raise PluginLoadError(
                f"Plugin config must have 'path' or 'package' key: {plugin_cfg}"
            )
        loaded.append(name)

    return loaded
