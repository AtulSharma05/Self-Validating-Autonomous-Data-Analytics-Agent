"""Unit tests for the Automated Data Profiler."""

from pathlib import Path
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


def test_profiler_computes_accurate_metadata():
    engine = DataIngestionEngine()
    engine.load_dataset(str(SAMPLE_CSV))
    
    profile = AutomatedDataProfiler.profile_dataset(engine.con)
    assert profile.total_tables == 1
    
    tbl = profile.tables[0]
    assert tbl.table_name == "sample_retail_sales"
    assert tbl.row_count > 900
    assert tbl.column_count >= 8
    
    col_map = {c.name: c for c in tbl.columns}
    
    # Check numeric columns
    assert "revenue" in col_map
    rev_col = col_map["revenue"]
    assert rev_col.is_numeric is True
    assert rev_col.min_value is not None
    assert rev_col.max_value is not None
    assert rev_col.mean_value > 0
    assert rev_col.q50 is not None
    
    # Check categorical column
    assert "region" in col_map
    region_col = col_map["region"]
    assert region_col.is_categorical is True
    assert region_col.distinct_count == 4
    
    # Check temporal column
    assert "order_date" in col_map
    date_col = col_map["order_date"]
    assert date_col.is_temporal is True
    assert date_col.min_date is not None
    assert date_col.max_date is not None

    # Test compact LLM format generation
    compact_text = profile.to_compact_llm_context()
    assert "sample_retail_sales" in compact_text
    assert "revenue" in compact_text
    assert "Categories:" in compact_text

    engine.close()
