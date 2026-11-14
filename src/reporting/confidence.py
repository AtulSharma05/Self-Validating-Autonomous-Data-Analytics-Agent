"""
Evidence Sufficiency & Mathematical Confidence Scoring Engine (Feature 11).

Formulates the Evidence Confidence Score (ECS in [0, 100]%):
    ECS = 100 * (w1 * V_exec + w2 * D_cov + w3 * E_attr - w4 * C_contradict)

Components:
  1. V_exec (0.40): Deterministic invariant checks pass rate.
  2. D_cov  (0.25): Data completeness (1.0 - key_column_null_rate).
  3. E_attr (0.25): Variance explained by identified contributing factors.
  4. C_contradict (0.10): Penalty for conflicting sub-trends or unresolved discrepancies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agent.schemas import DatasetProfile, RCAReport, SemanticCoverageReport


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"          # ECS >= 75%
    MODERATE = "MODERATE"  # 50% <= ECS < 75%
    LOW = "LOW"            # ECS < 50%
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # Insufficient data or failed verification


@dataclass
class ECSComponent:
    """Breakdown of an individual ECS mathematical factor."""
    name: str
    raw_value: float
    weight: float
    weighted_value: float
    description: str


@dataclass
class ECSReport:
    """Final calculated Evidence Confidence Score report."""
    score: float
    tier: ConfidenceTier
    badge_color: str
    components: Dict[str, ECSComponent]
    recommendation: str
    formula_expression: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "tier": self.tier.value,
            "badge_color": self.badge_color,
            "components": {k: asdict(v) for k, v in self.components.items()},
            "recommendation": self.recommendation,
            "formula_expression": self.formula_expression,
        }


class ECSCalculator:
    """
    Computes mathematically rigorous evidence confidence scores.
    """

    W_VEXEC = 0.40
    W_DCOV = 0.25
    W_EATTR = 0.25
    W_CCONTRA = 0.10

    HIGH_THRESHOLD = 75.0
    MODERATE_THRESHOLD = 50.0

    @classmethod
    def compute(
        cls,
        checks_passed: int,
        checks_total: int,
        profile: Optional[DatasetProfile] = None,
        key_columns: Optional[List[str]] = None,
        rca_report: Optional[RCAReport] = None,
        unresolved_discrepancy: bool = False,
        has_contradicting_trends: bool = False,
        semantic_coverage: Optional[SemanticCoverageReport] = None,
    ) -> ECSReport:
        """
        Calculates the full multi-factor Evidence Confidence Score.
        """
        # 1. Validation Execution Score (V_exec)
        if checks_total > 0:
            v_exec = max(0.0, min(1.0, checks_passed / checks_total))
        else:
            v_exec = 0.0

        # 2. Data Completeness Score (D_cov)
        d_cov = 1.0
        if profile and key_columns:
            null_percentages = []
            for tbl in profile.tables:
                col_map = {c.name.lower(): c for c in tbl.columns}
                for kc in key_columns:
                    col = col_map.get(kc.lower())
                    if col:
                        null_percentages.append(col.null_percentage)
            if null_percentages:
                avg_null_rate = (sum(null_percentages) / len(null_percentages)) / 100.0
                d_cov = max(0.0, min(1.0, 1.0 - avg_null_rate))

        # 3. Attribution Explanatory Power (E_attr)
        if rca_report is not None:
            if abs(rca_report.total_delta) > 0 and rca_report.primary_factors:
                primary_abs_sum = sum(abs(pf.delta) for pf in rca_report.primary_factors)
                e_attr = min(1.0, primary_abs_sum / abs(rca_report.total_delta))
            else:
                e_attr = 0.5
        else:
            # For non-diagnostic descriptive queries that passed validation, attribution is 1.0
            e_attr = 1.0 if v_exec >= 1.0 else 0.8

        # 4. Contradiction Penalty (C_contradict)
        c_contra = 0.0
        if unresolved_discrepancy or has_contradicting_trends:
            c_contra = 1.0

        # Weighted calculation
        raw_score = 100.0 * (
            cls.W_VEXEC * v_exec
            + cls.W_DCOV * d_cov
            + cls.W_EATTR * e_attr
            - cls.W_CCONTRA * c_contra
        )
        final_score = round(max(0.0, min(100.0, raw_score)), 1)

        # Semantic Coverage Gating: If question coverage failed, gate score to <= 20%
        if semantic_coverage is not None and not semantic_coverage.is_covered:
            tier = ConfidenceTier.LOW
            color = "#EF4444"
            rec = f"Semantic coverage failed: {semantic_coverage.details or 'Analysis substituted or omitted requested metrics.'}"
            final_score = min(final_score, 20.0)
        # Tier assignment
        elif checks_total == 0 or checks_passed == 0 or (v_exec == 0.0):
            tier = ConfidenceTier.INSUFFICIENT_EVIDENCE
            color = "#EF4444"
            rec = "Insufficient evidence to draw an authoritative conclusion. Requires manual review."
            final_score = 0.0
        elif final_score >= cls.HIGH_THRESHOLD:
            tier = ConfidenceTier.HIGH
            color = "#10B981"  # Emerald green
            rec = "High-confidence finding backed by deterministic validation and clean data."
        elif final_score >= cls.MODERATE_THRESHOLD:
            tier = ConfidenceTier.MODERATE
            color = "#F59E0B"  # Amber
            rec = "Moderate confidence; actionable with business context, minor data caveats exist."
        else:
            tier = ConfidenceTier.LOW
            color = "#EF4444"  # Rose red
            rec = "Insufficient evidence to draw an authoritative conclusion. Requires manual review."

        components = {
            "validation_score": ECSComponent(
                name="Validation Score (V_exec)",
                raw_value=round(v_exec, 3),
                weight=cls.W_VEXEC,
                weighted_value=round(cls.W_VEXEC * v_exec * 100, 2),
                description=f"{checks_passed}/{checks_total} deterministic invariant checks passed",
            ),
            "data_coverage": ECSComponent(
                name="Data Completeness (D_cov)",
                raw_value=round(d_cov, 3),
                weight=cls.W_DCOV,
                weighted_value=round(cls.W_DCOV * d_cov * 100, 2),
                description=f"Cleanliness of relevant columns (null rate = {(1.0 - d_cov) * 100:.1f}%)",
            ),
            "attribution_power": ECSComponent(
                name="Attribution Power (E_attr)",
                raw_value=round(e_attr, 3),
                weight=cls.W_EATTR,
                weighted_value=round(cls.W_EATTR * e_attr * 100, 2),
                description="Share of observed metric variance explained by validated factors",
            ),
            "contradiction_penalty": ECSComponent(
                name="Contradiction Penalty (C_contra)",
                raw_value=round(c_contra, 1),
                weight=cls.W_CCONTRA,
                weighted_value=round(-cls.W_CCONTRA * c_contra * 100, 2),
                description="Penalty applied for conflicting trends or unresolved discrepancies",
            ),
        }

        formula_str = (
            f"ECS = 100 * (0.40 * {v_exec:.2f} + 0.25 * {d_cov:.2f} + "
            f"0.25 * {e_attr:.2f} - 0.10 * {c_contra:.1f}) = {final_score:.1f}%"
        )

        return ECSReport(
            score=final_score,
            tier=tier,
            badge_color=color,
            components=components,
            recommendation=rec,
            formula_expression=formula_str,
        )

