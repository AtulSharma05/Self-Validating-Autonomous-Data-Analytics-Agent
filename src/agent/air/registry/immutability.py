"""
Recursive Structural Immutability Validator for the AIR Capability Registry.

Phase 4.0 Core Security & Immutability Safeguards.
"""

from __future__ import annotations
import dataclasses
import decimal
from types import MappingProxyType
from typing import Any, Dict, Set
from pydantic import BaseModel


class RegistryError(Exception):
    """Base exception for capability registry errors."""
    pass


class RegistrySecurityError(RegistryError):
    """Raised when an object fails trusted allowlist or immutability verification."""
    pass


class RegistryFrozenError(RegistryError):
    """Raised when mutation is attempted on a frozen registry."""
    pass


PRIMITIVE_IMMUTABLE_TYPES = (
    int, float, str, bool, bytes, type(None), decimal.Decimal,
)

TRUSTED_MODULE_PREFIXES = (
    "src.agent.air.registry.",
    "tests.",
)


def assert_deeply_immutable(
    obj: Any,
    path: str = "root",
    active_stack: Set[int] = None,
    completed_nodes: Set[int] = None,
) -> None:
    """
    Recursively inspects an object across its entire object graph.
    Uses two-phase cycle detection (active_stack for recursion path, completed_nodes for memoization)
    to ensure cyclic structures are validated without infinite loops or verification bypass.
    """
    if active_stack is None:
        active_stack = set()
    if completed_nodes is None:
        completed_nodes = set()

    obj_id = id(obj)
    if obj_id in completed_nodes:
        return
    if obj_id in active_stack:
        return  # Cycle detected; ancestor currently under active validation

    active_stack.add(obj_id)

    try:
        # 1. Primitives
        if isinstance(obj, PRIMITIVE_IMMUTABLE_TYPES):
            completed_nodes.add(obj_id)
            return

        # 2. Immutable Containers
        if isinstance(obj, tuple):
            for idx, item in enumerate(obj):
                assert_deeply_immutable(item, f"{path}[{idx}]", active_stack, completed_nodes)
            completed_nodes.add(obj_id)
            return

        if isinstance(obj, frozenset):
            for item in obj:
                assert_deeply_immutable(item, f"{path}{{{item}}}", active_stack, completed_nodes)
            completed_nodes.add(obj_id)
            return

        if isinstance(obj, MappingProxyType):
            for key, value in obj.items():
                assert_deeply_immutable(key, f"{path}.key({key})", active_stack, completed_nodes)
                assert_deeply_immutable(value, f"{path}[{key}]", active_stack, completed_nodes)
            completed_nodes.add(obj_id)
            return

        # 3. Explicit Rejection of Mutable Built-ins
        if isinstance(obj, (list, dict, set, bytearray)):
            raise RegistrySecurityError(
                f"Mutable container of type '{type(obj).__name__}' detected at '{path}'. "
                "All nested handler state must be immutable."
            )

        # 4. Allowlist Check for Complex Objects
        obj_module = getattr(type(obj), "__module__", "")
        if not any(obj_module.startswith(prefix) for prefix in TRUSTED_MODULE_PREFIXES):
            raise RegistrySecurityError(
                f"Handler at '{path}' of class '{type(obj).__name__}' belongs to unapproved module '{obj_module}'. "
                "Only application-owned built-ins and verified test handlers may be registered."
            )

        # 5. Mandatory Frozen Dataclasses (including slots-based frozen dataclasses)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            params = getattr(obj, "__dataclass_params__", None)
            if not (params and params.frozen):
                raise RegistrySecurityError(f"Dataclass at '{path}' ({type(obj).__name__}) must have frozen=True.")
            for f in dataclasses.fields(obj):
                assert_deeply_immutable(getattr(obj, f.name), f"{path}.{f.name}", active_stack, completed_nodes)
            completed_nodes.add(obj_id)
            return

        # 6. Mandatory Frozen Pydantic Models
        if isinstance(obj, BaseModel):
            if not obj.model_config.get("frozen", False):
                raise RegistrySecurityError(f"Pydantic model at '{path}' ({type(obj).__name__}) must have frozen=True.")
            for field_name in type(obj).model_fields:
                assert_deeply_immutable(getattr(obj, field_name), f"{path}.{field_name}", active_stack, completed_nodes)
            for priv_attr in ("__pydantic_private__", "__pydantic_extra__"):
                priv_val = getattr(obj, priv_attr, None)
                if priv_val:
                    assert_deeply_immutable(priv_val, f"{path}.{priv_attr}", active_stack, completed_nodes)
            completed_nodes.add(obj_id)
            return

        # 7. Explicit Rejection of Non-Dataclass Slots Classes
        if hasattr(type(obj), "__slots__"):
            raise RegistrySecurityError(
                f"Non-dataclass slots-based class at '{path}' ({type(obj).__name__}) is prohibited. "
                "All capability handlers must be frozen dataclasses or frozen Pydantic models."
            )

        raise RegistrySecurityError(
            f"Unsupported object of type '{type(obj).__name__}' at '{path}'. "
            "Handlers must be frozen dataclasses or frozen Pydantic models."
        )

    finally:
        active_stack.remove(obj_id)
