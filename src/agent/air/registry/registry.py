"""
Process-Scoped, Thread-Safe Analytical Capability Registry.

Phase 4.0 Core Registry Catalog.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

from src.agent.air.registry.models import (
    CapabilityAuthorizationPolicy,
    CapabilityKind,
    CapabilityManifestItem,
    PortDefinition,
    ResourceConstraints,
)
from src.agent.air.registry.protocols import (
    CapabilityCompilerHandler,
    CapabilityVerifierHandler,
    IndependentValidatorHandler,
    PrerequisiteEvaluator,
)
from src.agent.air.registry.immutability import (
    RegistryError,
    RegistryFrozenError,
    assert_deeply_immutable,
)


def create_isolated_mapping_proxy(source_dict: Dict[str, PortDefinition]) -> MappingProxyType[str, PortDefinition]:
    """
    Creates an independently owned, deeply copied MappingProxyType.
    Guarantees that external callers retain zero references to the backing dictionary.
    """
    isolated_copy: Dict[str, PortDefinition] = {}
    for k, v in source_dict.items():
        assert_deeply_immutable(k, path="mapping_key")
        assert_deeply_immutable(v, path=f"mapping_val[{k}]")
        isolated_copy[str(k)] = copy.deepcopy(v)

    return MappingProxyType(isolated_copy)


@dataclass(frozen=True)
class RegisteredCapability:
    """Immutable runtime record for a capability, wrapping metadata and trusted handlers."""
    capability_id: str
    version: str
    kind: CapabilityKind
    description: str
    input_ports: MappingProxyType[str, PortDefinition]
    output_ports: MappingProxyType[str, PortDefinition]
    compiler_handler: CapabilityCompilerHandler
    validator_handler: IndependentValidatorHandler
    verifier_handler: Optional[CapabilityVerifierHandler]
    prerequisite_evaluators: Tuple[PrerequisiteEvaluator, ...]
    authorization_policy: CapabilityAuthorizationPolicy
    resource_constraints: ResourceConstraints


class AnalyticalCapabilityRegistry:
    """Thread-safe, process-scoped, deeply immutable catalog of validated analytical capabilities."""

    _instance: Optional[AnalyticalCapabilityRegistry] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self):
        self._storage: Dict[Tuple[str, str], RegisteredCapability] = {}
        self._frozen: bool = False

    @classmethod
    def get_instance(cls) -> AnalyticalCapabilityRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance_for_testing(cls) -> None:
        """Isolated test reset only."""
        with cls._lock:
            cls._instance = None

    def register(
        self,
        capability_id: str,
        version: str,
        kind: CapabilityKind,
        description: str,
        input_ports: List[PortDefinition],
        output_ports: List[PortDefinition],
        compiler_handler: CapabilityCompilerHandler,
        validator_handler: IndependentValidatorHandler,
        verifier_handler: Optional[CapabilityVerifierHandler] = None,
        prerequisite_evaluators: Optional[List[PrerequisiteEvaluator]] = None,
        authorization_policy: Optional[CapabilityAuthorizationPolicy] = None,
        resource_constraints: Optional[ResourceConstraints] = None,
    ) -> None:
        """Registers a trusted capability. Validates deep structural immutability."""
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(
                    f"Security Boundary Violation: Cannot register capability '{capability_id}' "
                    "after registry freeze. Runtime mutation is strictly prohibited."
                )

            key = (capability_id.strip(), version.strip())
            if key in self._storage:
                raise RegistryError(f"Duplicate capability registration: '{key[0]}' version '{key[1]}'.")

            # 1. Structural Immutability Verification on All Handlers
            assert_deeply_immutable(compiler_handler, f"{key[0]}.compiler_handler")
            assert_deeply_immutable(validator_handler, f"{key[0]}.validator_handler")
            if verifier_handler:
                assert_deeply_immutable(verifier_handler, f"{key[0]}.verifier_handler")
            for idx, pe in enumerate(prerequisite_evaluators or []):
                assert_deeply_immutable(pe, f"{key[0]}.prerequisite_evaluators[{idx}]")

            # 2. Deep Ownership Copy for Ports (Caller cannot retain backing dictionary reference)
            in_dict = {p.name: p for p in input_ports}
            out_dict = {p.name: p for p in output_ports}
            frozen_inputs = create_isolated_mapping_proxy(in_dict)
            frozen_outputs = create_isolated_mapping_proxy(out_dict)

            rc = RegisteredCapability(
                capability_id=key[0],
                version=key[1],
                kind=kind,
                description=description,
                input_ports=frozen_inputs,
                output_ports=frozen_outputs,
                compiler_handler=compiler_handler,
                validator_handler=validator_handler,
                verifier_handler=verifier_handler,
                prerequisite_evaluators=tuple(prerequisite_evaluators or []),
                authorization_policy=authorization_policy or CapabilityAuthorizationPolicy(),
                resource_constraints=resource_constraints or ResourceConstraints(),
            )
            self._storage[key] = rc

    def freeze(self) -> None:
        """Permanently locks the registry against further mutation."""
        with self._lock:
            self._frozen = True

    @property
    def is_frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def get(self, capability_id: str, version: str) -> RegisteredCapability:
        """Retrieves exact capability by capability_id and exact version."""
        key = (capability_id.strip(), version.strip())
        if key not in self._storage:
            raise RegistryError(f"Capability '{capability_id}' version '{version}' is not registered.")
        return self._storage[key]

    def get_catalog_view(self) -> MappingProxyType[Tuple[str, str], RegisteredCapability]:
        """Returns read-only view of internal catalog preventing dictionary mutation."""
        with self._lock:
            return MappingProxyType(self._storage)

    def get_manifest_for_prompt(self) -> Tuple[CapabilityManifestItem, ...]:
        """Returns read-only, sanitized metadata items for the LLM prompt."""
        with self._lock:
            manifest_items = []
            for (cap_id, ver), cap in sorted(self._storage.items()):
                in_ports = tuple(
                    {
                        "name": p.name,
                        "role": p.semantic_role.value,
                        "cardinality": p.cardinality.value,
                        "grain": p.grain.value,
                    }
                    for p in cap.input_ports.values()
                    if p.is_required
                )
                out_ports = tuple(
                    {
                        "name": p.name,
                        "role": p.semantic_role.value,
                        "cardinality": p.cardinality.value,
                        "grain": p.grain.value,
                    }
                    for p in cap.output_ports.values()
                )
                manifest_items.append(
                    CapabilityManifestItem(
                        capability_id=cap_id,
                        version=ver,
                        kind=cap.kind,
                        description=cap.description,
                        required_input_ports=in_ports,
                        output_ports=out_ports,
                    )
                )
            return tuple(manifest_items)

