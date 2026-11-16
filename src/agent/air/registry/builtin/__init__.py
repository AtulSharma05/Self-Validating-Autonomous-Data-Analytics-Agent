"""
Built-in Trusted Analytical Capabilities for the AIR Capability Registry.

Phase 4.0 Core Built-ins.
"""

from src.agent.air.registry.builtin.temporal_filter import (
    TEMPORAL_FILTER_CAPABILITY_ID,
    TEMPORAL_FILTER_VERSION,
    TemporalFilterCompilerHandler,
    TemporalFilterVerifierHandler,
    TemporalFilterValidatorHandler,
    register_temporal_filter_capability,
)
from src.agent.air.registry.builtin.grouped_sum import (
    GROUPED_SUM_CAPABILITY_ID,
    GROUPED_SUM_VERSION,
    GroupedSumCompilerHandler,
    GroupedSumVerifierHandler,
    GroupedSumValidatorHandler,
    register_grouped_sum_capability,
)

__all__ = [
    "TEMPORAL_FILTER_CAPABILITY_ID",
    "TEMPORAL_FILTER_VERSION",
    "TemporalFilterCompilerHandler",
    "TemporalFilterVerifierHandler",
    "TemporalFilterValidatorHandler",
    "register_temporal_filter_capability",
    "GROUPED_SUM_CAPABILITY_ID",
    "GROUPED_SUM_VERSION",
    "GroupedSumCompilerHandler",
    "GroupedSumVerifierHandler",
    "GroupedSumValidatorHandler",
    "register_grouped_sum_capability",
]

