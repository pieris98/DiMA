"""
Abstract base class for all pipeline stages.

Each stage represents a discrete step in the ML pipeline (data setup,
training, inference, evaluation, etc.). Stages communicate through a
shared context dictionary.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStage(ABC):
    """
    Interface for pipeline stages.

    Subclasses must implement:
        - name (class attribute): Unique identifier for the stage
        - validate(config, context) -> list of issues (empty = ok)
        - run(config, context) -> updated context

    The context dict is shared across stages and carries:
        - File paths (checkpoints, statistics, generated samples)
        - Stage outputs (e.g., sequences produced by inference)
        - Configuration overrides
    """

    name: str = "base_stage"
    description: str = ""

    @abstractmethod
    def validate(self, config: Dict[str, Any], context: Dict[str, Any]) -> list:
        """
        Check preconditions before running the stage.

        Args:
            config: Hydra config or dict with pipeline configuration.
            context: Shared context dict from prior stages.

        Returns:
            List of validation error strings. Empty list means valid.
        """
        ...

    @abstractmethod
    def run(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the stage.

        Args:
            config: Hydra config or dict with pipeline configuration.
            context: Shared context dict from prior stages.

        Returns:
            Updated context dict.
        """
        ...

    def __repr__(self):
        return f"<Stage: {self.name}>"
