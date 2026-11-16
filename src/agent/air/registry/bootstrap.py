"""
Capability Registry Bootstrap for AIR Phase 4.0.

Initializes the process-scoped AnalyticalCapabilityRegistry with trusted
built-in capabilities and locks the catalog permanently.
"""

from __future__ import annotations
from typing import Optional

from src.agent.air.registry.registry import AnalyticalCapabilityRegistry
from src.agent.air.registry.builtin.temporal_filter import register_temporal_filter_capability
from src.agent.air.registry.builtin.grouped_sum import register_grouped_sum_capability


def bootstrap_registry(
    registry: Optional[AnalyticalCapabilityRegistry] = None,
    freeze: bool = True,
) -> AnalyticalCapabilityRegistry:
    """
    Bootstraps the registry with all trusted application-owned capabilities.
    If no registry is provided, retrieves or initializes the process singleton.
    Permanently freezes the registry after registration if freeze=True.
    """
    target_registry = registry if registry is not None else AnalyticalCapabilityRegistry.get_instance()

    # Register Phase 4.0B & 4.0C atomic capabilities
    register_temporal_filter_capability(target_registry)
    register_grouped_sum_capability(target_registry)

    if freeze and not target_registry.is_frozen:
        target_registry.freeze()

    return target_registry

