"""
Collision-Resistant, Type-Tagged Canonical Serializer & Fingerprinting Engine.

Phase 4.0 Core Canonical Serialization.
"""

from __future__ import annotations
import datetime
import decimal
import enum
import hashlib
import json
import math
from typing import Any
import numpy as np


class SerializationError(Exception):
    """Raised when an object cannot be canonically serialized."""
    pass


def canonical_serialize(val: Any) -> Any:
    """
    Recursively converts values into explicitly type-tagged canonical representations.
    Guarantees zero collision across types (e.g. Decimal("10") vs "10" vs 10 vs 10.0).
    """
    if val is None:
        return {"$t": "null"}

    # Distinguish Python bool from int (since bool inherits from int in Python)
    if type(val) is bool:
        return {"$t": "bool", "$v": val}

    # Distinguish NumPy bool from Python bool
    if isinstance(val, (np.bool_,)):
        return {"$t": "np_bool", "$v": bool(val)}

    # Standard Integers & NumPy Integers (Arbitrary precision integer preserved as exact string)
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return {"$t": "int", "$v": str(int(val))}

    # Exact Decimals (Preserves exact scale and digits without float coercion)
    if isinstance(val, decimal.Decimal):
        return {"$t": "decimal", "$v": str(val)}

    # Floats & NumPy Floats (Normalized Zeros Policy: +0.0 and -0.0 serialize to "0.0")
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if math.isnan(f):
            return {"$t": "float", "$v": "__NAN__"}
        if math.isinf(f):
            return {"$t": "float", "$v": "__POS_INF__" if f > 0 else "__NEG_INF__"}
        if f == 0.0:
            return {"$t": "float", "$v": "0.0"}
        return {"$t": "float", "$v": str(f)}

    # Strings
    if isinstance(val, str):
        return {"$t": "str", "$v": val}

    # Bytes
    if isinstance(val, (bytes, bytearray)):
        return {"$t": "bytes", "$v": bytes(val).hex()}

    # Distinct Date vs DateTime Handling (datetime inherits from date in Python!)
    if type(val) is datetime.datetime:
        tz_str = val.tzinfo.tzname(val) if val.tzinfo else "naive"
        return {"$t": "datetime", "$v": val.isoformat(), "$tz": tz_str}

    if type(val) is datetime.date:
        return {"$t": "date", "$v": val.isoformat()}

    # NumPy datetime64
    if isinstance(val, np.datetime64):
        return {"$t": "np_datetime64", "$v": str(val)}

    # Enums with Fully Qualified Class Names (avoids collision across different enums with same value)
    if isinstance(val, enum.Enum):
        fqcn = f"{val.__class__.__module__}.{val.__class__.__qualname__}"
        return {"$t": "enum", "$cls": fqcn, "$v": val.value}

    # Distinct Set vs Frozenset Handling (sorted deterministically by canonical json representation)
    if type(val) is frozenset:
        canonical_elements = [canonical_serialize(item) for item in val]
        canonical_elements.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return {"$t": "frozenset", "$v": canonical_elements}

    if type(val) is set:
        canonical_elements = [canonical_serialize(item) for item in val]
        canonical_elements.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return {"$t": "set", "$v": canonical_elements}

    # Distinct Tuple vs List Handling (preserves exact sequence order)
    if type(val) is tuple:
        return {"$t": "tuple", "$v": [canonical_serialize(item) for item in val]}

    if type(val) is list:
        return {"$t": "list", "$v": [canonical_serialize(item) for item in val]}

    # Mappings (Canonical Key-Value Pairs Sorted Deterministically by Key)
    if isinstance(val, dict):
        entries = []
        for k in val:
            entries.append([canonical_serialize(k), canonical_serialize(val[k])])
        entries.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {"$t": "map", "$entries": entries}

    if hasattr(val, "model_dump"):
        return canonical_serialize(val.model_dump())

    raise SerializationError(f"Object of type '{type(val).__name__}' cannot be canonically serialized.")


def compute_canonical_fingerprint(payload: Any) -> str:
    """Computes SHA-256 fingerprint over type-tagged canonical serialization."""
    canonical_repr = canonical_serialize(payload)
    encoded = json.dumps(canonical_repr, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

