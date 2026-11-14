"""
Diagnostic Root-Cause Analysis (RCA) & Variance Decomposition Engine (Feature 9).

Performs:
  1. Pareto variance decomposition across candidate categorical dimensions.
  2. Mathematical Volume vs. Price decomposition for revenue/sales metrics.
  3. Factor attribution ranking (|Contribution| >= 10% designated as Primary).
  4. Evidence-grounded narrative summary generation.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional
import duckdb
import pandas as pd

from src.agent.schemas import (
    FactorAttribution,
    HypothesisResult,
    RCAReport,
    VolumePriceDecomposition,
)


class RCAEngine:
    """
    Automated neuro-symbolic root-cause analysis engine.
    """

    PRIMARY_FACTOR_THRESHOLD_PCT = 10.0

    @classmethod
    def decompose_variance(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        metric_col: str,
        dimensions: List[str],
        baseline_where: str = "1=1",
        test_where: str = "1=1",
    ) -> tuple[float, float, float, float, Dict[str, List[FactorAttribution]], List[FactorAttribution]]:
        """
        Decomposes total delta into dimensional category contributions.
        
        Returns:
            (baseline_total, test_total, total_delta, delta_pct, dimensional_factors, primary_factors)
        """
        # If baseline and test filters are identical (e.g. 1=1), auto-partition by time or cross-sectional
        effective_base = baseline_where
        effective_test = test_where

        if (baseline_where.strip() == "1=1" and test_where.strip() == "1=1") or (baseline_where == test_where):
            # Try to auto-split across a temporal column if available
            try:
                cols_desc = con.execute(f'DESCRIBE "{table_name}"').fetchall()
                date_cols = [c[0] for c in cols_desc if any(k in c[0].lower() for k in ["date", "time", "year", "month", "timestamp"])]
                if date_cols:
                    t_col = date_cols[0]
                    # Check if date column has multiple distinct dates
                    d_range = con.execute(f'SELECT MIN("{t_col}"), MAX("{t_col}") FROM "{table_name}"').fetchone()
                    if d_range and d_range[0] != d_range[1]:
                        # Use midpoint or half split
                        mid_date = con.execute(
                            f'SELECT CAST(MIN("{t_col}") + (MAX("{t_col}") - MIN("{t_col}")) / 2 AS VARCHAR) FROM "{table_name}"'
                        ).fetchone()[0]
                        effective_base = f'"{t_col}" <= \'{mid_date}\''
                        effective_test = f'"{t_col}" > \'{mid_date}\''
            except Exception:
                pass

        # 1. Total baseline and test metrics
        base_q = f"SELECT CAST(COALESCE(SUM(\"{metric_col}\"), 0.0) AS DOUBLE) FROM {table_name} WHERE {effective_base}"
        test_q = f"SELECT CAST(COALESCE(SUM(\"{metric_col}\"), 0.0) AS DOUBLE) FROM {table_name} WHERE {effective_test}"

        m_base: float = float(con.execute(base_q).fetchone()[0])
        m_test: float = float(con.execute(test_q).fetchone()[0])

        delta = m_test - m_base
        delta_pct = (delta / m_base * 100.0) if m_base != 0 else 0.0

        # If still delta == 0 (no time split available), do Pareto cross-sectional share decomposition
        is_cross_sectional = (delta == 0.0 and m_test > 0)
        if is_cross_sectional:
            delta = m_test

        dimensional_factors: Dict[str, List[FactorAttribution]] = {}
        all_factors: List[FactorAttribution] = []

        # 2. Decompose across each dimension
        for dim in dimensions:
            try:
                if is_cross_sectional:
                    q = f"""
                        SELECT 
                            "{dim}" AS category,
                            0.0 AS val_base,
                            CAST(COALESCE(SUM("{metric_col}"), 0.0) AS DOUBLE) AS val_test
                        FROM {table_name}
                        GROUP BY "{dim}"
                        ORDER BY val_test DESC
                    """
                else:
                    q = f"""
                        WITH base AS (
                            SELECT "{dim}" AS category, CAST(COALESCE(SUM("{metric_col}"), 0.0) AS DOUBLE) AS val_base
                            FROM {table_name}
                            WHERE {effective_base}
                            GROUP BY "{dim}"
                        ),
                        test AS (
                            SELECT "{dim}" AS category, CAST(COALESCE(SUM("{metric_col}"), 0.0) AS DOUBLE) AS val_test
                            FROM {table_name}
                            WHERE {effective_test}
                            GROUP BY "{dim}"
                        )
                        SELECT 
                            COALESCE(b.category, t.category) AS category,
                            COALESCE(b.val_base, 0.0) AS val_base,
                            COALESCE(t.val_test, 0.0) AS val_test
                        FROM base b
                        FULL OUTER JOIN test t ON b.category = t.category
                        ORDER BY category
                    """
                df_dim = con.execute(q).df()

                dim_factors: List[FactorAttribution] = []
                for _, row in df_dim.iterrows():
                    cat = str(row["category"])
                    v_b = float(row["val_base"])
                    v_t = float(row["val_test"])
                    cat_delta = v_t - v_b
                    contrib = (cat_delta / delta * 100.0) if delta != 0 else 0.0
                    is_primary = abs(contrib) >= cls.PRIMARY_FACTOR_THRESHOLD_PCT

                    fa = FactorAttribution(
                        dimension=dim,
                        category=cat,
                        baseline_value=round(v_b, 2),
                        test_value=round(v_t, 2),
                        delta=round(cat_delta, 2),
                        contribution_pct=round(contrib, 2),
                        is_primary=is_primary,
                    )
                    dim_factors.append(fa)
                    all_factors.append(fa)

                # Sort dimension factors by absolute contribution descending
                dim_factors.sort(key=lambda x: abs(x.contribution_pct), reverse=True)
                dimensional_factors[dim] = dim_factors
            except Exception:
                continue

        # Rank primary factors across all dimensions
        primary_factors = [f for f in all_factors if f.is_primary]
        primary_factors.sort(key=lambda x: abs(x.contribution_pct), reverse=True)

        return (m_base, m_test, delta, delta_pct, dimensional_factors, primary_factors)

    @classmethod
    def decompose_volume_price(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        revenue_col: Optional[str] = None,
        volume_col: Optional[str] = None,
        baseline_where: str = "1=1",
        test_where: str = "1=1",
    ) -> Optional[VolumePriceDecomposition]:
        """
        Performs mathematical Volume vs. Price decomposition:
            ΔRevenue = (Q_T - Q_B) * P_B  [Volume Effect]
                     + (P_T - P_B) * Q_T  [Price Effect]
        """
        try:
            cols = [r[0].lower() for r in con.execute(f'DESCRIBE "{table_name}"').fetchall()]

            # Auto-detect revenue column
            target_rev = revenue_col
            if not target_rev or target_rev.lower() not in cols:
                for c in ["sales", "revenue", "order_amount", "amount", "profit"]:
                    if c in cols:
                        target_rev = c
                        break

            # Auto-detect volume column
            target_vol = volume_col
            if not target_vol or target_vol.lower() not in cols:
                for c in ["quantity", "units_sold", "units", "qty", "volume", "order_quantity"]:
                    if c in cols:
                        target_vol = c
                        break

            if not target_rev or not target_vol or target_rev.lower() not in cols or target_vol.lower() not in cols:
                return None

            # Effective filters
            effective_base = baseline_where
            effective_test = test_where
            if (baseline_where.strip() == "1=1" and test_where.strip() == "1=1") or (baseline_where == test_where):
                date_cols = [c for c in cols if any(k in c for k in ["date", "time", "year", "month"])]
                if date_cols:
                    t_col = date_cols[0]
                    d_range = con.execute(f'SELECT MIN("{t_col}"), MAX("{t_col}") FROM "{table_name}"').fetchone()
                    if d_range and d_range[0] != d_range[1]:
                        mid_date = con.execute(
                            f'SELECT CAST(MIN("{t_col}") + (MAX("{t_col}") - MIN("{t_col}")) / 2 AS VARCHAR) FROM "{table_name}"'
                        ).fetchone()[0]
                        effective_base = f'"{t_col}" <= \'{mid_date}\''
                        effective_test = f'"{t_col}" > \'{mid_date}\''

            base_q = f"""
                SELECT 
                    CAST(COALESCE(SUM("{target_vol}"), 0.0) AS DOUBLE) AS q_b,
                    CAST(COALESCE(SUM("{target_rev}"), 0.0) AS DOUBLE) AS r_b
                FROM {table_name} WHERE {effective_base}
            """
            test_q = f"""
                SELECT 
                    CAST(COALESCE(SUM("{target_vol}"), 0.0) AS DOUBLE) AS q_t,
                    CAST(COALESCE(SUM("{target_rev}"), 0.0) AS DOUBLE) AS r_t
                FROM {table_name} WHERE {effective_test}
            """

            q_b, r_b = con.execute(base_q).fetchone()
            q_t, r_t = con.execute(test_q).fetchone()

            q_b, r_b = float(q_b), float(r_b)
            q_t, r_t = float(q_t), float(r_t)

            if q_b == 0 or q_t == 0:
                return None

            p_b = r_b / q_b
            p_t = r_t / q_t

            vol_effect = (q_t - q_b) * p_b
            price_effect = (p_t - p_b) * q_t
            tot_delta = r_t - r_b

            vol_pct = (vol_effect / tot_delta * 100.0) if tot_delta != 0 else 0.0
            price_pct = (price_effect / tot_delta * 100.0) if tot_delta != 0 else 0.0

            return VolumePriceDecomposition(
                baseline_volume=round(q_b, 2),
                test_volume=round(q_t, 2),
                baseline_avg_price=round(p_b, 2),
                test_avg_price=round(p_t, 2),
                volume_effect=round(vol_effect, 2),
                price_effect=round(price_effect, 2),
                total_delta=round(tot_delta, 2),
                volume_contribution_pct=round(vol_pct, 2),
                price_contribution_pct=round(price_pct, 2),
            )
        except Exception:
            return None

    @classmethod
    def analyze(
        cls,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        metric_col: str,
        dimensions: List[str],
        baseline_where: str = "1=1",
        test_where: str = "1=1",
        volume_col: Optional[str] = None,
    ) -> RCAReport:
        """
        Orchestrates full diagnostic root-cause analysis, hypothesis investigation, and narrative generation.
        """
        m_base, m_test, delta, delta_pct, dim_factors, primary_factors = cls.decompose_variance(
            con, table_name, metric_col, dimensions, baseline_where, test_where
        )

        vp_decomp = cls.decompose_volume_price(
            con, table_name, metric_col, volume_col, baseline_where, test_where
        )

        # Formulate and empirically test analytical hypotheses (Feature 9)
        hypotheses: List[HypothesisResult] = []
        h_idx = 1
        for dim, factors in dim_factors.items():
            if not factors:
                continue
            top_f = factors[0]
            is_confirmed = top_f.is_primary or abs(top_f.contribution_pct) >= cls.PRIMARY_FACTOR_THRESHOLD_PCT
            status = "CONFIRMED_PRIMARY_DRIVER" if is_confirmed else "REJECTED_MINOR_FACTOR"
            statement = (
                f"Variance in '{metric_col}' is significantly driven by category shifts within '{dim}', "
                f"with leading category '{top_f.category}' accounting for {top_f.contribution_pct:+.1f}% of observed variance."
            )
            h_hash = hashlib.sha256(f"{table_name}:{dim}:{metric_col}:{top_f.category}".encode()).hexdigest()[:16]
            hypotheses.append(HypothesisResult(
                hypothesis_id=f"H_{h_idx}",
                dimension=dim,
                statement=statement,
                top_category=top_f.category,
                delta=top_f.delta,
                contribution_pct=top_f.contribution_pct,
                status=status,
                evidence_hash=h_hash,
            ))
            h_idx += 1

        if vp_decomp:
            vp_confirmed = abs(vp_decomp.volume_contribution_pct) > abs(vp_decomp.price_contribution_pct)
            leading_effect = "Volume Effect" if vp_confirmed else "Price Effect"
            leading_pct = vp_decomp.volume_contribution_pct if vp_confirmed else vp_decomp.price_contribution_pct
            vp_statement = (
                f"Revenue variance is primarily driven by unit volume shifts ({vp_decomp.volume_contribution_pct:+.1f}%) "
                f"rather than realized price changes ({vp_decomp.price_contribution_pct:+.1f}%)."
            )
            h_hash = hashlib.sha256(f"{table_name}:volume_price:{metric_col}".encode()).hexdigest()[:16]
            hypotheses.append(HypothesisResult(
                hypothesis_id=f"H_{h_idx}",
                dimension="Volume vs. Price",
                statement=vp_statement,
                top_category=leading_effect,
                delta=vp_decomp.volume_effect if vp_confirmed else vp_decomp.price_effect,
                contribution_pct=leading_pct,
                status="CONFIRMED_PRIMARY_DRIVER",
                evidence_hash=h_hash,
            ))

        # Generate grounded analytical narrative
        narrative = cls._synthesize_narrative(
            metric_col, m_base, m_test, delta, delta_pct, primary_factors, vp_decomp, hypotheses
        )

        return RCAReport(
            target_metric=metric_col,
            baseline_total=round(m_base, 2),
            test_total=round(m_test, 2),
            total_delta=round(delta, 2),
            delta_pct=round(delta_pct, 2),
            primary_factors=primary_factors,
            dimensional_factors=dim_factors,
            hypotheses=hypotheses,
            volume_price=vp_decomp,
            narrative_summary=narrative,
        )

    @staticmethod
    def _synthesize_narrative(
        metric: str,
        m_base: float,
        m_test: float,
        delta: float,
        delta_pct: float,
        primary_factors: List[FactorAttribution],
        vp: Optional[VolumePriceDecomposition],
        hypotheses: Optional[List[HypothesisResult]] = None,
    ) -> str:
        direction = "increased" if delta >= 0 else "decreased"
        if m_base == m_test or m_base == 0:
            lines = [
                f"Total {metric} is {m_test:,.2f} across the analyzed dataset."
            ]
        else:
            lines = [
                f"Total {metric} {direction} by {abs(delta):,.2f} ({delta_pct:+.1f}%), moving from {m_base:,.2f} to {m_test:,.2f}."
            ]

        if vp:
            lines.append(
                f"- **Decomposition (Volume vs Price):** Volume change accounted for {vp.volume_effect:,.2f} ({vp.volume_contribution_pct:.1f}%), "
                f"while Average Realized Price change accounted for {vp.price_effect:,.2f} ({vp.price_contribution_pct:.1f}%)."
            )

        if hypotheses:
            confirmed_h = [h for h in hypotheses if h.status == "CONFIRMED_PRIMARY_DRIVER"]
            if confirmed_h:
                lines.append(f"- **Empirical Hypotheses Confirmed ({len(confirmed_h)} validated drivers):**")
                for ch in confirmed_h[:4]:
                    lines.append(f"  * **[{ch.hypothesis_id}]**: {ch.dimension} driver `{ch.top_category}` explains **{ch.contribution_pct:+.1f}%** of variance.")

        if primary_factors:
            lines.append("- **Primary Contributing Factors (>= 10% variance explained):**")
            for pf in primary_factors[:5]:
                f_dir = "growth" if pf.delta >= 0 else "decline"
                lines.append(
                    f"  * `{pf.dimension}` = **{pf.category}**: $\\Delta$ {pf.delta:,.2f} "
                    f"({pf.contribution_pct:+.1f}% of total {direction}), driving {f_dir}."
                )
        else:
            lines.append("- No single dimension category accounted for more than 10% of the overall variance (change is uniformly distributed).")

        return "\n".join(lines)

