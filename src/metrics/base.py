"""
Abstract base class for all metrics in the DiMA pipeline.

All metrics implement a common interface so they can be discovered
via the registry and dispatched uniformly.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict


class BaseMetric(ABC):
    """
    Interface that all DiMA metrics must implement.

    Subclasses must implement:
        - compute(predictions, references, **kwargs) -> float

    Properties:
        - name: Human-readable metric name
        - requires_references: Whether the metric needs reference sequences
        - is_per_sample: Whether the metric is computed per-sample (True)
                         or over the full distribution (False)
    """

    name: str = "base_metric"
    requires_references: bool = False
    is_per_sample: bool = False

    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> float:
        """
        Compute the metric.

        Args:
            predictions: List of generated sequences.
            references: List of reference sequences (if required).
            **kwargs: Additional arguments (device, max_len, pdb_path, etc.)

        Returns:
            Scalar metric value.
        """
        ...


class FIDMetric(BaseMetric):
    """FID (Frechet Inception Distance) metric wrapper."""
    name = "fid"
    requires_references = True
    is_per_sample = False

    def compute(self, predictions, references=None, **kwargs):
        from src.metrics.fid import calculate_fid_for_lists
        device = kwargs.get("device", "cuda:0")
        max_len = kwargs.get("max_len", 512)
        return calculate_fid_for_lists(predictions, references, max_len, device)


class MMDMetric(BaseMetric):
    """MMD (Maximum Mean Discrepancy) metric wrapper."""
    name = "mmd"
    requires_references = True
    is_per_sample = False

    def compute(self, predictions, references=None, **kwargs):
        from src.metrics.mmd import calculate_mmd_for_lists
        device = kwargs.get("device", "cuda:0")
        max_len = kwargs.get("max_len", 512)
        return calculate_mmd_for_lists(predictions, references, max_len, device)


class ESMPPPLMetric(BaseMetric):
    """ESM pseudo-perplexity metric wrapper."""
    name = "esm_pppl"
    requires_references = False
    is_per_sample = True

    def compute(self, predictions, references=None, **kwargs):
        import numpy as np
        from src.metrics.esmpppl import calculate_pppl
        device = kwargs.get("device", "cuda:0")
        max_len = kwargs.get("max_len", 512)
        pppl_result = calculate_pppl(predictions, max_len, device)
        return float(np.mean(pppl_result))


class PLDDTMetric(BaseMetric):
    """pLDDT (predicted confidence) metric wrapper."""
    name = "plddt"
    requires_references = False
    is_per_sample = True

    def compute(self, predictions, references=None, **kwargs):
        import numpy as np
        from src.metrics.plddt import calculate_plddt
        device = kwargs.get("device", "cuda:0")
        pdb_path = kwargs.get("pdb_path", "")
        index_list = kwargs.get("index_list", list(range(len(predictions))))
        result = calculate_plddt(predictions, index_list, device, pdb_path)
        return float(np.mean(list(result.values())))
