"""
Register all built-in metrics with the pipeline registry.

Import this module to ensure all metrics are available in the registry.
"""

from pipeline.registry import registry
from src.metrics.base import FIDMetric, MMDMetric, ESMPPPLMetric, PLDDTMetric

registry.register("metric", "fid", FIDMetric)
registry.register("metric", "mmd", MMDMetric)
registry.register("metric", "esm_pppl", ESMPPPLMetric)
registry.register("metric", "plddt", PLDDTMetric)
