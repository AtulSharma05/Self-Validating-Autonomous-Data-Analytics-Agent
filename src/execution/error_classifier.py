"""
Error Classifier — maps raw Python exceptions and validation failures into
standardized, actionable error categories that the self-correction loop can act on.

Four canonical error types:
  SYNTAX_ERROR        — SQL parse failure, bad column name, Python NameError/AttributeError
  INVARIANT_ERROR     — Mathematical discrepancy (dual-path mismatch, sum-of-parts violation)
  BOUNDARY_ERROR      — Impossible values (negative revenue, ratio > 100, out-of-range dates)
  EMPTY_RESULT_ERROR  — Query returned 0 rows due to overly restrictive filters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Error type taxonomy
# ---------------------------------------------------------------------------

class ErrorType(str, Enum):
    SYNTAX_ERROR       = "SYNTAX_ERROR"
    INVARIANT_ERROR    = "INVARIANT_ERROR"
    BOUNDARY_ERROR     = "BOUNDARY_ERROR"
    EMPTY_RESULT_ERROR = "EMPTY_RESULT_ERROR"
    TIMEOUT_ERROR      = "TIMEOUT_ERROR"
    UNKNOWN_ERROR      = "UNKNOWN_ERROR"


# Self-correction hint templates keyed by error type
_CORRECTION_HINTS: dict[str, str] = {
    ErrorType.SYNTAX_ERROR: (
        "Re-examine column names against the dataset schema. "
        "Use double-quotes for column identifiers. "
        "Ensure the table name matches exactly what was registered. "
        "Check for missing commas, unclosed parentheses, or wrong SQL dialect keywords."
    ),
    ErrorType.INVARIANT_ERROR: (
        "The sum of segmented values does not match the grand total. "
        "Check for JOIN conditions that drop rows (prefer LEFT JOIN over INNER JOIN). "
        "Ensure GROUP BY columns are correct and avoid accidental Cartesian products."
    ),
    ErrorType.BOUNDARY_ERROR: (
        "Result contains values outside expected business domain. "
        "Add WHERE filters to exclude NULL or negative values before aggregation. "
        "Verify that percentage/ratio columns are expressed as 0–100, not 0–1."
    ),
    ErrorType.EMPTY_RESULT_ERROR: (
        "The query returned zero rows. "
        "Relax date filters — check the dataset's actual date range before filtering. "
        "Verify categorical filter values match exact strings in the data (case-sensitive). "
        "Consider removing one filter at a time to isolate the cause."
    ),
    ErrorType.TIMEOUT_ERROR: (
        "Query exceeded the execution time limit. "
        "Rewrite using aggregations instead of row-level operations. "
        "Add WHERE filters to reduce the working dataset before grouping. "
        "Avoid correlated subqueries — use CTEs or window functions instead."
    ),
    ErrorType.UNKNOWN_ERROR: (
        "An unexpected error occurred. "
        "Simplify the query and test each sub-clause independently."
    ),
}

# Keyword patterns used to classify raw exception messages
_SYNTAX_KEYWORDS = [
    "binder error", "parser error", "catalog error",
    "column not found", "table not found", "syntax error",
    "does not exist", "ambiguous", "nameerror", "attributeerror",
    "keyerror", "typeerror", "invalid input syntax",
]

_BOUNDARY_KEYWORDS = [
    "boundary", "negative", "out of range", "overflow",
    "division by zero", "invalid value",
]

_EMPTY_KEYWORDS = [
    "empty", "no rows", "0 rows", "zero rows",
]

_INVARIANT_KEYWORDS = [
    "invariant", "mismatch", "discrepancy", "dual-path",
    "sum of parts", "does not equal grand total",
]

_TIMEOUT_KEYWORDS = [
    "timeout", "timed out", "time limit", "futures.timeout",
    "concurrent.futures", "executor",
]


# ---------------------------------------------------------------------------
# Classified error dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedError:
    """A structured, actionable error produced by the classifier."""
    error_type: ErrorType
    raw_message: str
    correction_hint: str = field(init=False)
    is_retryable: bool = field(init=False)

    def __post_init__(self):
        self.correction_hint = _CORRECTION_HINTS.get(self.error_type, "")
        # Timeout and unknown errors are still retryable with a simpler query;
        # boundary errors may also be retryable with corrected filters.
        self.is_retryable = self.error_type != ErrorType.UNKNOWN_ERROR

    def to_prompt_injection(self) -> str:
        """Formats the error for injection into a self-correction LLM prompt."""
        return (
            f"**Error Type:** `{self.error_type.value}`\n"
            f"**Error Detail:** {self.raw_message}\n"
            f"**Correction Guidance:** {self.correction_hint}"
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """
    Classifies raw exceptions and validation failure messages into one of the
    four canonical error types, with actionable correction hints.
    """

    @classmethod
    def classify_exception(cls, exc: Exception) -> ClassifiedError:
        """Classify a Python/DuckDB exception."""
        return cls.classify_message(str(exc))

    @classmethod
    def classify_message(cls, message: str) -> ClassifiedError:
        """
        Classify a raw error message string.

        Args:
            message: The raw exception string or validation error message.

        Returns:
            ClassifiedError with the appropriate type and correction hint.
        """
        lower = message.lower()

        if any(kw in lower for kw in _TIMEOUT_KEYWORDS):
            return ClassifiedError(ErrorType.TIMEOUT_ERROR, message)

        if any(kw in lower for kw in _INVARIANT_KEYWORDS):
            return ClassifiedError(ErrorType.INVARIANT_ERROR, message)

        if any(kw in lower for kw in _BOUNDARY_KEYWORDS):
            return ClassifiedError(ErrorType.BOUNDARY_ERROR, message)

        if any(kw in lower for kw in _EMPTY_KEYWORDS):
            return ClassifiedError(ErrorType.EMPTY_RESULT_ERROR, message)

        if any(kw in lower for kw in _SYNTAX_KEYWORDS):
            return ClassifiedError(ErrorType.SYNTAX_ERROR, message)

        return ClassifiedError(ErrorType.UNKNOWN_ERROR, message)

    @classmethod
    def classify_empty_result(cls, query: str) -> ClassifiedError:
        """Convenience constructor for a known empty-result situation."""
        return ClassifiedError(
            ErrorType.EMPTY_RESULT_ERROR,
            f"Query returned 0 rows: {query[:200]}",
        )

    @classmethod
    def classify_validation_failure(cls, error_type: str, message: str) -> ClassifiedError:
        """
        Map a ValidationResult error_type string directly to a ClassifiedError.

        Args:
            error_type: One of the strings from ValidationResult.error_type.
            message:    The validation error_message.
        """
        mapping = {
            "INVARIANT_VIOLATION_SUM_OF_PARTS": ErrorType.INVARIANT_ERROR,
            "INVARIANT_ERROR": ErrorType.INVARIANT_ERROR,
            "BOUNDARY_ERROR": ErrorType.BOUNDARY_ERROR,
            "SYNTAX_ERROR": ErrorType.SYNTAX_ERROR,
            "EMPTY_RESULT_ERROR": ErrorType.EMPTY_RESULT_ERROR,
            "TIMEOUT_ERROR": ErrorType.TIMEOUT_ERROR,
        }
        et = mapping.get(error_type, ErrorType.UNKNOWN_ERROR)
        return ClassifiedError(et, message)

