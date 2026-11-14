"""Unit tests for the Data Ingestion Engine."""

import pytest
from pathlib import Path
from src.ingestion.loader import DataIngestionEngine, sanitize_identifier

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_retail_sales.csv"


def test_sanitize_identifier():
    assert sanitize_identifier("Order ID") == "order_id"
    assert sanitize_identifier("Units Sold (2023)") == "units_sold_2023"
    assert sanitize_identifier("123_test") == "t_123_test"
    assert sanitize_identifier("  Region-Name!  ") == "region_name"


def test_load_csv_dataset():
    engine = DataIngestionEngine()
    meta = engine.load_dataset(str(SAMPLE_CSV))
    
    assert meta["table_name"] == "sample_retail_sales"
    assert meta["row_count"] > 900
    assert "sample_retail_sales" in engine.list_tables()
    
    # Test query execution
    df = engine.execute_query("SELECT COUNT(*) as cnt FROM sample_retail_sales")
    assert df["cnt"].iloc[0] == meta["row_count"]
    engine.close()


def test_nonexistent_file_raises_error():
    engine = DataIngestionEngine()
    with pytest.raises(FileNotFoundError):
        engine.load_dataset("data/non_existent.csv")
    engine.close()


def test_load_non_utf8_csv(tmp_path):
    """Verifies that CSVs with latin-1 or non-utf-8 byte sequences are loaded gracefully."""
    latin1_file = tmp_path / "latin1_test.csv"
    # Write Latin-1 encoded bytes (e.g., accent character 0xe9)
    latin1_file.write_bytes(b"id,product_name,sales\n1,Caf\xe9 Espresso,12.50\n2,Na\xefve Bayes,25.00\n")
    
    engine = DataIngestionEngine()
    meta = engine.load_dataset(str(latin1_file))
    
    assert meta["table_name"] == "latin1_test"
    assert meta["row_count"] == 2
    
    df = engine.execute_query("SELECT product_name, sales FROM latin1_test ORDER BY id")
    assert len(df) == 2
    assert "Caf" in df["product_name"].iloc[0]
    engine.close()

