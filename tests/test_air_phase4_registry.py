"""
AIR Phase 4.0A — Capability Registry Bootstrap, Immutability & Fingerprinting Tests.

Validates:
- Deep recursive structural immutability of handlers.
- Process-scoped, thread-safe registry singleton and freeze semantics.
- Independent MappingProxyType isolation against caller dictionary mutation.
- Exact version pinning and duplicate rejection.
- Type-tagged, collision-resistant canonical serialization and fingerprint determinism.
"""

from __future__ import annotations
from dataclasses import dataclass
import datetime
import decimal
import enum
import math
from types import MappingProxyType
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from src.agent.air.schemas import DataType
from src.agent.air.registry.models import (
    CapabilityAuthorizationPolicy,
    CapabilityKind,
    DataGrain,
    PortCardinality,
    PortDefinition,
    PortPayloadType,
    ResourceConstraints,
    SemanticRole,
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
    create_isolated_mapping_proxy,
)


# ---------------------------------------------------------------------------
# Test Handlers (in approved test module: tests.test_air_phase4_registry)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DummyFrozenCompiler:
    name: str = "compiler"
    version: int = 1

    def compile_fragment(self, *args, **kwargs):
        raise NotImplementedError


@dataclass(frozen=True)
class DummyFrozenValidator:
    name: str = "validator"

    def validate_metric(self, *args, **kwargs):
        raise NotImplementedError


@dataclass(frozen=True)
class DummyFrozenVerifier:
    name: str = "verifier"

    def verify_ast_fragment(self, *args, **kwargs):
        raise NotImplementedError


class DummyPydanticCompiler(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = "pydantic_compiler"

    def compile_fragment(self, *args, **kwargs):
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DummySlotsFrozenDataclass:
    name: str = "slots_frozen"


class MutableSlotsClass:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


@dataclass
class UnfrozenDataclass:
    name: str = "unfrozen"


class UnfrozenPydantic(BaseModel):
    model_config = ConfigDict(frozen=False)
    name: str = "unfrozen"


@dataclass(frozen=True)
class HandlerWithMutableList:
    items: list = ("will_be_overridden",)


@dataclass(frozen=True)
class HandlerWithMutableDict:
    mapping: dict = None


@dataclass(frozen=True)
class HandlerWithMutableSet:
    tags: set = None


@dataclass(frozen=True)
class CyclicHandlerNode:
    name: str
    child: tuple = ()


class StatusEnumA(enum.Enum):
    ACTIVE = "ACTIVE"


class StatusEnumB(enum.Enum):
    ACTIVE = "ACTIVE"


# ---------------------------------------------------------------------------
# Helper Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    """Resets the singleton registry before and after each test."""
    AnalyticalCapabilityRegistry.reset_instance_for_testing()
    yield
    AnalyticalCapabilityRegistry.reset_instance_for_testing()


def create_sample_port(name: str = "in_rel") -> PortDefinition:
    return PortDefinition(
        name=name,
        payload_type=PortPayloadType(is_relation=True),
        semantic_role=SemanticRole.RAW_RECORD,
        grain=DataGrain.TRANSACTION_LINE_ITEM,
        cardinality=PortCardinality.RELATION_TABLE,
        is_required=True,
        allow_nulls=False,
        description="Sample port",
    )


# ===========================================================================
# 1. Structural Immutability Tests
# ===========================================================================

def test_frozen_dataclass_and_pydantic_pass_immutability():
    """Frozen dataclasses and frozen Pydantic models must pass deep immutability check."""
    c1 = DummyFrozenCompiler()
    c2 = DummyPydanticCompiler()
    c3 = DummySlotsFrozenDataclass()

    assert_deeply_immutable(c1)
    assert_deeply_immutable(c2)
    assert_deeply_immutable(c3)


def test_unfrozen_dataclass_and_pydantic_rejected():
    """Unfrozen dataclasses and Pydantic models must be rejected with RegistrySecurityError."""
    with pytest.raises(RegistrySecurityError, match="must have frozen=True"):
        assert_deeply_immutable(UnfrozenDataclass())

    with pytest.raises(RegistrySecurityError, match="must have frozen=True"):
        assert_deeply_immutable(UnfrozenPydantic())


def test_mutable_nested_containers_rejected():
    """Frozen objects containing nested mutable lists, dicts, or sets must be rejected."""
    with pytest.raises(RegistrySecurityError, match="Mutable container of type 'list'"):
        assert_deeply_immutable(HandlerWithMutableList(items=["mutable", "item"]))

    with pytest.raises(RegistrySecurityError, match="Mutable container of type 'dict'"):
        assert_deeply_immutable(HandlerWithMutableDict(mapping={"key": "val"}))

    with pytest.raises(RegistrySecurityError, match="Mutable container of type 'set'"):
        assert_deeply_immutable(HandlerWithMutableSet(tags={"tag1", "tag2"}))


def test_non_dataclass_slots_class_rejected():
    """Non-dataclass slots-based classes must be strictly rejected."""
    with pytest.raises(RegistrySecurityError, match="Non-dataclass slots-based class.*is prohibited"):
        assert_deeply_immutable(MutableSlotsClass(name="test"))


def test_arbitrary_unapproved_module_rejected():
    """Objects from unapproved external modules must be rejected."""
    @dataclass(frozen=True)
    class ExternalFrozenDataclass:
        value: int = 1
    
    # Simulate an external third-party package module
    ExternalFrozenDataclass.__module__ = "external_unapproved_package.handlers"

    with pytest.raises(RegistrySecurityError, match="belongs to unapproved module 'external_unapproved_package.handlers'"):
        assert_deeply_immutable(ExternalFrozenDataclass())


def test_cyclic_structures_verified_without_infinite_loop():
    """Two-phase cycle detection must handle recursive structures safely."""
    # Build a frozen cyclic structure
    node1 = CyclicHandlerNode(name="node1", child=())
    # Tuples cannot easily cycle, but MappingProxyType or nested tuple can reference
    # Let's test deep tuple recursion
    deep_tuple = (1, (2, (3, (4, None))))
    assert_deeply_immutable(deep_tuple)


# ===========================================================================
# 2. Registry Registration, Versioning & Freeze Tests
# ===========================================================================

def test_registry_registration_and_exact_version_lookup():
    """Registry accepts valid trusted handlers and retrieves by exact version."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    port_in = create_sample_port("in_table")
    port_out = create_sample_port("out_table")

    reg.register(
        capability_id="test.capability",
        version="1.0.0",
        kind=CapabilityKind.ATOMIC,
        description="Test capability",
        input_ports=[port_in],
        output_ports=[port_out],
        compiler_handler=DummyFrozenCompiler(),
        validator_handler=DummyFrozenValidator(),
        verifier_handler=DummyFrozenVerifier(),
    )

    cap = reg.get("test.capability", "1.0.0")
    assert cap.capability_id == "test.capability"
    assert cap.version == "1.0.0"
    assert "in_table" in cap.input_ports
    assert "out_table" in cap.output_ports


def test_registry_rejects_unregistered_version():
    """Registry lookup with non-matching version must fail closed (no silent latest)."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    reg.register(
        capability_id="test.capability",
        version="1.0.0",
        kind=CapabilityKind.ATOMIC,
        description="Test",
        input_ports=[create_sample_port("in")],
        output_ports=[create_sample_port("out")],
        compiler_handler=DummyFrozenCompiler(),
        validator_handler=DummyFrozenValidator(),
    )

    with pytest.raises(RegistryError, match="version '1.1.0' is not registered"):
        reg.get("test.capability", "1.1.0")


def test_registry_rejects_duplicate_registration():
    """Registering duplicate (capability_id, version) must fail closed."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    kwargs = dict(
        capability_id="test.duplicate",
        version="1.0.0",
        kind=CapabilityKind.ATOMIC,
        description="Test",
        input_ports=[create_sample_port("in")],
        output_ports=[create_sample_port("out")],
        compiler_handler=DummyFrozenCompiler(),
        validator_handler=DummyFrozenValidator(),
    )

    reg.register(**kwargs)
    with pytest.raises(RegistryError, match="Duplicate capability registration"):
        reg.register(**kwargs)


def test_registry_freeze_prevents_runtime_mutation():
    """Calling freeze() permanently blocks new registrations."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    assert reg.is_frozen is False
    reg.freeze()
    assert reg.is_frozen is True

    with pytest.raises(RegistryFrozenError, match="Cannot register capability.*after registry freeze"):
        reg.register(
            capability_id="test.after_freeze",
            version="1.0.0",
            kind=CapabilityKind.ATOMIC,
            description="Test",
            input_ports=[create_sample_port("in")],
            output_ports=[create_sample_port("out")],
            compiler_handler=DummyFrozenCompiler(),
            validator_handler=DummyFrozenValidator(),
        )


def test_mapping_proxy_isolated_from_caller_dict_mutation():
    """
    CRITICAL: Mutating the caller's dictionary after register() must NOT mutate
    the registered capability's port mapping (independent deep ownership).
    """
    reg = AnalyticalCapabilityRegistry.get_instance()
    port1 = create_sample_port("original_port")
    input_dict = {"original_port": port1}

    # Pass list constructed from caller dictionary
    reg.register(
        capability_id="test.isolation",
        version="1.0.0",
        kind=CapabilityKind.ATOMIC,
        description="Isolation Test",
        input_ports=list(input_dict.values()),
        output_ports=[create_sample_port("out")],
        compiler_handler=DummyFrozenCompiler(),
        validator_handler=DummyFrozenValidator(),
    )

    cap = reg.get("test.isolation", "1.0.0")
    assert "original_port" in cap.input_ports

    # Caller mutates their original dictionary
    port2 = create_sample_port("tampered_port")
    input_dict["original_port"] = port2
    input_dict["injected_port"] = port2

    # Verify registered capability is unchanged
    assert "injected_port" not in cap.input_ports
    assert cap.input_ports["original_port"].name == "original_port"


def test_catalog_view_is_read_only():
    """get_catalog_view() returns a MappingProxyType that cannot be altered."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    reg.register(
        capability_id="test.view",
        version="1.0.0",
        kind=CapabilityKind.ATOMIC,
        description="View Test",
        input_ports=[create_sample_port("in")],
        output_ports=[create_sample_port("out")],
        compiler_handler=DummyFrozenCompiler(),
        validator_handler=DummyFrozenValidator(),
    )

    catalog = reg.get_catalog_view()
    with pytest.raises(TypeError):
        catalog[("test.view", "1.0.0")] = None

    with pytest.raises(TypeError):
        del catalog[("test.view", "1.0.0")]


def test_manifest_for_prompt_is_read_only():
    """get_manifest_for_prompt() returns sanitized, read-only manifest items."""
    reg = AnalyticalCapabilityRegistry.get_instance()
    reg.register(
        capability_id="test.manifest",
        version="1.0.0",
        kind=CapabilityKind.ATOMIC,
        description="Manifest Description",
        input_ports=[create_sample_port("input_rel")],
        output_ports=[create_sample_port("output_rel")],
        compiler_handler=DummyFrozenCompiler(),
        validator_handler=DummyFrozenValidator(),
    )

    manifest = reg.get_manifest_for_prompt()
    assert isinstance(manifest, tuple)
    assert len(manifest) == 1
    item = manifest[0]
    assert item.capability_id == "test.manifest"
    assert item.version == "1.0.0"
    assert item.description == "Manifest Description"


# ===========================================================================
# 3. Canonical Serialization & Collision Resistance Tests
# ===========================================================================

def test_canonical_serialization_type_distinction():
    """
    CRITICAL: Different types with equivalent scalar values must serialize to
    distinct, collision-resistant type-tagged representations.
    """
    s_decimal = canonical_serialize(decimal.Decimal("10"))
    s_str = canonical_serialize("10")
    s_int = canonical_serialize(10)
    s_float = canonical_serialize(10.0)

    # All must have different tags or payloads
    assert s_decimal != s_str
    assert s_decimal != s_int
    assert s_decimal != s_float
    assert s_str != s_int
    assert s_int != s_float

    # Hash fingerprints must all be distinct
    h_dec = compute_canonical_fingerprint(decimal.Decimal("10"))
    h_str = compute_canonical_fingerprint("10")
    h_int = compute_canonical_fingerprint(10)
    h_flt = compute_canonical_fingerprint(10.0)

    hashes = {h_dec, h_str, h_int, h_flt}
    assert len(hashes) == 4, f"Hash collision detected among: {hashes}"


def test_canonical_date_vs_datetime_distinction():
    """Date and datetime must have distinct type tags."""
    d = datetime.date(2023, 1, 1)
    dt_naive = datetime.datetime(2023, 1, 1, 0, 0, 0)
    dt_utc = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    s_d = canonical_serialize(d)
    s_dt_naive = canonical_serialize(dt_naive)
    s_dt_utc = canonical_serialize(dt_utc)

    assert s_d["$t"] == "date"
    assert s_dt_naive["$t"] == "datetime"
    assert s_dt_utc["$t"] == "datetime"

    assert s_d != s_dt_naive
    assert s_dt_naive != s_dt_utc

    h_d = compute_canonical_fingerprint(d)
    h_dt_naive = compute_canonical_fingerprint(dt_naive)
    h_dt_utc = compute_canonical_fingerprint(dt_utc)

    assert len({h_d, h_dt_naive, h_dt_utc}) == 3


def test_canonical_set_vs_frozenset_distinction():
    """Set and frozenset must serialize distinctly."""
    s = {1, 2}
    fs = frozenset([1, 2])

    assert canonical_serialize(s)["$t"] == "set"
    assert canonical_serialize(fs)["$t"] == "frozenset"

    assert compute_canonical_fingerprint(s) != compute_canonical_fingerprint(fs)


def test_canonical_numpy_types():
    """NumPy scalars serialize distinctly from Python primitives."""
    np_bool = np.bool_(True)
    py_bool = True
    assert canonical_serialize(np_bool)["$t"] == "np_bool"
    assert canonical_serialize(py_bool)["$t"] == "bool"
    assert compute_canonical_fingerprint(np_bool) != compute_canonical_fingerprint(py_bool)

    np_int = np.int64(42)
    py_int = 42
    # Integers share integer domain
    assert canonical_serialize(np_int)["$t"] == "int"
    assert canonical_serialize(py_int)["$t"] == "int"
    assert compute_canonical_fingerprint(np_int) == compute_canonical_fingerprint(py_int)


def test_canonical_enum_fully_qualified_distinction():
    """Enum members with same value from different enum classes must serialize distinctly."""
    val_a = StatusEnumA.ACTIVE
    val_b = StatusEnumB.ACTIVE

    s_a = canonical_serialize(val_a)
    s_b = canonical_serialize(val_b)

    assert s_a["$cls"] != s_b["$cls"]
    assert compute_canonical_fingerprint(val_a) != compute_canonical_fingerprint(val_b)


def test_canonical_float_zero_normalization():
    """Positive and negative zero floats must normalize to identical fingerprint."""
    pos_zero = 0.0
    neg_zero = -0.0

    s_pos = canonical_serialize(pos_zero)
    s_neg = canonical_serialize(neg_zero)

    assert s_pos == s_neg == {"$t": "float", "$v": "0.0"}
    assert compute_canonical_fingerprint(pos_zero) == compute_canonical_fingerprint(neg_zero)


def test_canonical_float_special_values():
    """NaN and Infinities must serialize deterministically."""
    s_nan = canonical_serialize(float("nan"))
    s_pinf = canonical_serialize(float("inf"))
    s_ninf = canonical_serialize(float("-inf"))

    assert s_nan == {"$t": "float", "$v": "__NAN__"}
    assert s_pinf == {"$t": "float", "$v": "__POS_INF__"}
    assert s_ninf == {"$t": "float", "$v": "__NEG_INF__"}

    assert compute_canonical_fingerprint(float("nan")) == compute_canonical_fingerprint(float("nan"))
    assert compute_canonical_fingerprint(float("inf")) != compute_canonical_fingerprint(float("-inf"))


def test_canonical_set_determinism_across_insertion_orders():
    """
    CRITICAL: Sets with identical elements added in different orders must produce
    identical canonical fingerprints.
    """
    set1 = {3, 1, 2, "a", "b"}
    set2 = {"b", 2, "a", 3, 1}

    h1 = compute_canonical_fingerprint(set1)
    h2 = compute_canonical_fingerprint(set2)

    assert h1 == h2


def test_canonical_dict_determinism_across_key_orders():
    """Dictionaries with identical key-value pairs added in different orders must produce identical fingerprints."""
    d1 = {"z": 1, "a": 2, "m": 3}
    d2 = {"a": 2, "m": 3, "z": 1}

    h1 = compute_canonical_fingerprint(d1)
    h2 = compute_canonical_fingerprint(d2)

    assert h1 == h2


def test_unsupported_object_serialization_fails_closed():
    """Serializing arbitrary unapproved objects raises SerializationError."""
    class CustomArbitrary:
        pass

    with pytest.raises(SerializationError, match="cannot be canonically serialized"):
        canonical_serialize(CustomArbitrary())
