"""
Component registry for DiMA modular pipeline.

Provides a central registry where encoders, decoders, metrics, datasets,
and pipeline stages can be registered by name and retrieved at runtime.
Third-party plugins use the same registration mechanism.
"""

from typing import Dict, Type, Optional, List, Any


# Valid component categories
VALID_CATEGORIES = ("encoder", "decoder", "metric", "stage", "dataset")


class ComponentRegistry:
    """
    A registry that maps (category, name) -> component class.

    Usage:
        registry = ComponentRegistry()
        registry.register("encoder", "esm2", ESM2EncoderModel)
        cls = registry.get("encoder", "esm2")
    """

    def __init__(self):
        self._components: Dict[str, Dict[str, Type]] = {cat: {} for cat in VALID_CATEGORIES}

    def register(self, category: str, name: str, cls: Type, overwrite: bool = False) -> None:
        """Register a component class under a category and name."""
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(f"Name must be a non-empty string, got: {name!r}")
        if name in self._components[category] and not overwrite:
            raise ValueError(
                f"Component '{name}' already registered under '{category}'. "
                f"Pass overwrite=True to replace it."
            )
        self._components[category][name] = cls

    def get(self, category: str, name: str) -> Type:
        """Retrieve a registered component class."""
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}"
            )
        if name not in self._components[category]:
            available = list(self._components[category].keys())
            raise KeyError(
                f"No component '{name}' registered under '{category}'. "
                f"Available: {available}"
            )
        return self._components[category][name]

    def has(self, category: str, name: str) -> bool:
        """Check if a component is registered."""
        if category not in VALID_CATEGORIES:
            return False
        return name in self._components[category]

    def list_components(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List registered components.
        If category is given, return names for that category.
        Otherwise, return all categories with their names.
        """
        if category is not None:
            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}"
                )
            return {category: list(self._components[category].keys())}
        return {cat: list(names.keys()) for cat, names in self._components.items()}

    def unregister(self, category: str, name: str) -> None:
        """Remove a component from the registry."""
        if category in self._components and name in self._components[category]:
            del self._components[category][name]

    def clear(self, category: Optional[str] = None) -> None:
        """Clear all registrations, or just one category."""
        if category is not None:
            if category in self._components:
                self._components[category].clear()
        else:
            for cat in self._components:
                self._components[cat].clear()

    # --- Decorator API ---

    def register_encoder(self, name: str):
        """Decorator to register an encoder class."""
        def decorator(cls):
            self.register("encoder", name, cls)
            return cls
        return decorator

    def register_decoder(self, name: str):
        """Decorator to register a decoder class."""
        def decorator(cls):
            self.register("decoder", name, cls)
            return cls
        return decorator

    def register_metric(self, name: str):
        """Decorator to register a metric class."""
        def decorator(cls):
            self.register("metric", name, cls)
            return cls
        return decorator

    def register_stage(self, name: str):
        """Decorator to register a pipeline stage class."""
        def decorator(cls):
            self.register("stage", name, cls)
            return cls
        return decorator

    def register_dataset(self, name: str):
        """Decorator to register a dataset class."""
        def decorator(cls):
            self.register("dataset", name, cls)
            return cls
        return decorator


# Global singleton registry
registry = ComponentRegistry()
