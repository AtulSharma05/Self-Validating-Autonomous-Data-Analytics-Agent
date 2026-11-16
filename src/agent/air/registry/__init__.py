"""
AIR Capability Registry Package.

Phase 4.0 Core Exports.
"""

from src.agent.air.registry.models import (
    CapabilityAuthorizationPolicy,
    CapabilityKind,
    CapabilityManifestItem,
    DataGrain,
    ExecutionMode,
    PortCardinality,
    PortDefinition,
    PortPayloadType,
    ResourceConstraints,
    SemanticRole,
)
from src.agent.air.registry.protocols import (
    CapabilityCompilerHandler,
    CapabilityVerifierHandler,
    ContractFragment,
    IndependentValidatorHandler,
    PrerequisiteEvaluator,
)
from src.agent.air.registry.immutability import (
    RegistryError,
    RegistryFrozenError,
    RegistrySecurityError,
    assert_deeply_immutable,
)
from src.agent.air.registry.fingerprint import (
    SerializationError,
    canonical_serialize,
    compute_canonical_fingerprint,
)
from src.agent.air.registry.registry import (
    AnalyticalCapabilityRegistry,
    RegisteredCapability,
    create_isolated_mapping_proxy,
)
from src.agent.air.registry.bootstrap import bootstrap_registry
from src.agent.air.registry.composition import (
    CapabilityDAG,
    CapabilityDAGNode,
    CapabilityResolver,
    ComposedExecutionContract,
    ComposedPipelineCompiler,
    ComposedPipelineVerifier,
    ComposedPipelineValidator,
)

__all__ = [
    "ExecutionMode",
    "CapabilityKind",
    "SemanticRole",
    "DataGrain",
    "PortCardinality",
    "PortPayloadType",
    "PortDefinition",
    "ResourceConstraints",
    "CapabilityAuthorizationPolicy",
    "CapabilityManifestItem",
    "ContractFragment",
    "CapabilityCompilerHandler",
    "CapabilityVerifierHandler",
    "IndependentValidatorHandler",
    "PrerequisiteEvaluator",
    "RegistryError",
    "RegistrySecurityError",
    "RegistryFrozenError",
    "assert_deeply_immutable",
    "SerializationError",
    "canonical_serialize",
    "compute_canonical_fingerprint",
    "AnalyticalCapabilityRegistry",
    "RegisteredCapability",
    "create_isolated_mapping_proxy",
    "bootstrap_registry",
    "CapabilityDAG",
    "CapabilityDAGNode",
    "CapabilityResolver",
    "ComposedExecutionContract",
    "ComposedPipelineCompiler",
    "ComposedPipelineVerifier",
    "ComposedPipelineValidator",
]

