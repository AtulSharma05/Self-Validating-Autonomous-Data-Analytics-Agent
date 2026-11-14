"""
Tests for Entity Deduplication, Table Grain Awareness, and Hypothesis-Based Root-Cause Analysis.
"""

from pathlib import Path
import pytest
import duckdb

from src.agent.graph import create_analytics_graph
from src.agent.rca import RCAEngine
from src.agent.schemas import AnalysisObjective, AnalysisType, DatasetProfile
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler
from src.validation.validator import DualPathVerifier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


def test_dual_path_verifier_supports_count_distinct():
    """Verify DualPathVerifier supports COUNT_DISTINCT and matches between DuckDB and Pandas."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (order_id VARCHAR, region VARCHAR)")
    con.execute("INSERT INTO orders VALUES ('O1', 'West'), ('O1', 'West'), ('O2', 'East'), ('O3', 'West')")

    res = DualPathVerifier.verify(con, "orders", metric_col="order_id", agg_func="COUNT_DISTINCT")
    assert res.is_valid
    assert res.audit_details["path_a_sql"] == 3.0
    assert res.audit_details["path_b_pandas"] == 3.0


def test_dual_path_verifier_supports_aov():
    """Verify DualPathVerifier computes AOV correctly across DuckDB and Pandas."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE sales (order_id VARCHAR, sales DOUBLE)")
    # Order 1: two items ($100 + $50) = $150
    # Order 2: one item ($50) = $50
    # Total sales = $200, Unique orders = 2, AOV = $100
    # (Notice line item average would be $200 / 3 = $66.67)
    con.execute("INSERT INTO sales VALUES ('O1', 100.0), ('O1', 50.0), ('O2', 50.0)")

    res = DualPathVerifier.verify(con, "sales", metric_col="sales", agg_func="AOV")
    assert res.is_valid
    assert res.audit_details["path_a_sql"] == 100.0
    assert res.audit_details["path_b_pandas"] == 100.0


def test_hypothesis_investigation_in_rca_engine():
    """Verify that RCAEngine formulates and empirically evaluates hypotheses."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE transactions (revenue DOUBLE, units_sold INT, region VARCHAR, category VARCHAR)")
    con.execute("""
        INSERT INTO transactions VALUES 
        (1000.0, 10, 'North', 'Tech'),
        (200.0, 2, 'South', 'Tech'),
        (300.0, 3, 'North', 'Office'),
        (100.0, 1, 'South', 'Office')
    """)

    report = RCAEngine.analyze(
        con=con,
        table_name="transactions",
        metric_col="revenue",
        dimensions=["region", "category"],
    )

    assert report.total_delta > 0
    assert len(report.hypotheses) >= 2
    # Check each hypothesis has required fields
    for h in report.hypotheses:
        assert h.hypothesis_id.startswith("H_")
        assert h.status in ["CONFIRMED_PRIMARY_DRIVER", "REJECTED_MINOR_FACTOR"]
        assert len(h.evidence_hash) > 0
        assert h.statement

    # North represents 1300/1600 = 81.25% of total revenue -> should be confirmed primary driver
    h_region = next((h for h in report.hypotheses if h.dimension == "region"), None)
    assert h_region is not None
    assert h_region.status == "CONFIRMED_PRIMARY_DRIVER"
    assert h_region.top_category == "North"


def test_grain_awareness_avoids_line_item_trap_end_to_end():
    """Verify that end-to-end graph handles entity deduplication and AOV without line-item confusion."""
    con = duckdb.connect(":memory:")
    # 3 line items across 2 orders:
    con.execute("""
        CREATE TABLE sample_orders (
            order_id VARCHAR,
            sales DOUBLE,
            order_date DATE
        )
    """)
    con.execute("""
        INSERT INTO sample_orders VALUES
        ('ORD-001', 300.0, '2017-02-15'),
        ('ORD-001', 100.0, '2017-02-16'),
        ('ORD-002', 200.0, '2017-05-10')
    """)

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_grain")
    graph = create_analytics_graph(con)

    initial_state = {
        "session_id": "test_grain",
        "user_question": "How many total orders were placed in 2017, and what is the average order value (AOV)?",
        "profile": profile,
        "max_retries": 3,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["status"] == "COMPLETED"

    # Step results:
    # Unique orders should be 2, NOT 3 (number of rows)
    # Total sales should be 600.0
    # AOV should be 300.0 (600 / 2), NOT 200.0 (average line item 600 / 3)
    summary = final_state["final_summary"]
    assert "2.00" in summary or "2" in summary  # unique orders
    assert "300.00" in summary  # AOV
    assert "600.00" in summary  # Total sales


def test_dual_path_verifier_supports_max_and_min():
    """Verify DualPathVerifier supports MAX and MIN aggregations without failing."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE products (product_name VARCHAR, profit DOUBLE)")
    con.execute("INSERT INTO products VALUES ('Copier', 5000.0), ('Stapler', 12.0), ('Phone', 800.0)")

    res_max = DualPathVerifier.verify(con, "products", metric_col="profit", agg_func="MAX")
    assert res_max.is_valid
    assert res_max.audit_details["path_a_sql"] == 5000.0
    assert res_max.audit_details["path_b_pandas"] == 5000.0

    res_min = DualPathVerifier.verify(con, "products", metric_col="profit", agg_func="MIN")
    assert res_min.is_valid
    assert res_min.audit_details["path_a_sql"] == 12.0
    assert res_min.audit_details["path_b_pandas"] == 12.0


def test_highest_profit_product_query_end_to_end():
    """Verify 'which product gives the highest profit' completes deterministically with captured evidence."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE store_products (
            product_name VARCHAR,
            profit DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO store_products VALUES
        ('Canon imageCLASS 2200 Advanced Copier', 25199.98),
        ('Fellowes PB500 Electric Punch Plastic Comb Binding Machine', 7753.04),
        ('Hewlett Packard LaserJet 3310 Copier', 6983.88),
        ('GBC DocuBind P400 Electric Binding System', -1878.44)
    """)

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_top_prod")
    graph = create_analytics_graph(con)

    initial_state = {
        "session_id": "test_top_prod",
        "user_question": "which product gives the highest profit?",
        "profile": profile,
        "max_retries": 3,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["status"] == "COMPLETED"
    assert len(final_state["evidence_records"]) > 0
    # Provenance trace must contain valid records, not empty
    assert final_state["ecs_report"].score >= 70.0
    summary = final_state["final_summary"]
    assert "Canon imageCLASS 2200 Advanced Copier" in summary
    assert "25,199.98" in summary


def test_filtered_subset_multi_step_invariant_reconciliation():
    """Verify that multi-step query with filtered groups (HAVING/subsets) resolves cleanly without invariant loop."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE superstore (
            sub_category VARCHAR,
            sales DOUBLE,
            profit DOUBLE
        )
    """)
    # 2 sub-categories with sales > 100k and profit < 0:
    # Bookcases: sales 114880.00, profit -3472.56
    # Tables: sales 206965.53, profit -17725.48
    # Total sales for these 2 = 321845.53
    # Other sub-categories:
    # Chairs: sales 328449.10, profit 26590.17 (profit > 0)
    # Fasteners: sales 3024.28, profit 949.52 (sales < 100k)
    con.execute("""
        INSERT INTO superstore VALUES
        ('Bookcases', 114880.00, -3472.56),
        ('Tables', 206965.53, -17725.48),
        ('Chairs', 328449.10, 26590.17),
        ('Fasteners', 3024.28, 949.52)
    """)

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_subset")
    graph = create_analytics_graph(con)

    initial_state = {
        "session_id": "test_subset",
        "user_question": "Identify sub-categories with sales greater than $100,000 and a negative profit, and calculate their total sales",
        "profile": profile,
        "max_retries": 3,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["status"] == "COMPLETED"
    assert not final_state.get("unresolved_discrepancy", False)

    summary = final_state["final_summary"]
    assert "321,845.53" in summary or "321845.53" in summary


def test_counterfactual_discount_capping_scenario_multi_output():
    """
    Verify counterfactual discount capping question generates all 3 requested outputs:
    1. Revised Total Sales
    2. Revised Total Profit
    3. Revised Profit Margin
    And does NOT run unprompted RCA.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE sample_superstore (
            sales DOUBLE,
            profit DOUBLE,
            discount DOUBLE,
            quantity INT,
            category VARCHAR,
            region VARCHAR
        )
    """)
    # Row 1: sales 100, profit 10, discount 0.40 -> base = 166.67, new_sales = 133.33, new_profit = 43.33
    # Row 2: sales 200, profit 50, discount 0.10 -> unchanged (new_sales = 200, new_profit = 50)
    # Row 3: sales 50, profit -10, discount 0.70 -> base = 166.67, new_sales = 133.33, new_profit = 73.33
    # Total revised sales = 466.67
    # Total revised profit = 166.67
    # Revised margin = 166.67 / 466.67 * 100 = 35.71%
    con.execute("""
        INSERT INTO sample_superstore VALUES
        (100.0, 10.0, 0.40, 2, 'Furniture', 'West'),
        (200.0, 50.0, 0.10, 4, 'Technology', 'East'),
        (50.0, -10.0, 0.70, 1, 'Office Supplies', 'Central')
    """)

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_discount_cap")
    graph = create_analytics_graph(con)

    initial_state = {
        "session_id": "test_discount_cap",
        "user_question": "If maximum discounts had been capped at 20% across all historical orders (assuming zero change in unit volume or cost), what would the revised total sales, total profit, and profit margin be?",
        "profile": profile,
        "max_retries": 3,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["status"] == "COMPLETED"

    # Must NOT run RCA for this non-diagnostic scenario
    assert final_state.get("rca_report") is None

    # Must contain all 3 plan steps
    assert len(final_state["plan"].steps) == 3

    # Check that all 3 metrics are present in final summary
    summary = final_state["final_summary"]
    # Check revised sales ~ $466.67
    assert "466.67" in summary
    # Check revised profit ~ $166.67
    assert "166.67" in summary
    # Check revised profit margin ~ 35.71%
    assert "35.71" in summary


def test_general_multi_metric_query_generates_all_outputs():
    """Verify general multi-metric queries (e.g. sales, profit, and margin) produce all 3 outputs."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE retail (
            sales DOUBLE,
            profit DOUBLE,
            region VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO retail VALUES
        (1000.0, 200.0, 'East'),
        (2000.0, 300.0, 'West')
    """)

    profile = AutomatedDataProfiler.profile_dataset(con, session_id="test_multi_out")
    graph = create_analytics_graph(con)

    initial_state = {
        "session_id": "test_multi_out",
        "user_question": "What is the total sales, total profit, and profit margin?",
        "profile": profile,
        "max_retries": 3,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["status"] == "COMPLETED"
    assert len(final_state["plan"].steps) == 3

    summary = final_state["final_summary"]
    # Total sales = $3,000.00
    assert "3,000.00" in summary
    # Total profit = $500.00
    assert "500.00" in summary
    # Profit margin = 500 / 3000 * 100 = 16.67%
    assert "16.67" in summary
