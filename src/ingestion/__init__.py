"""Ingestion and profiling package."""
from src.ingestion.loader import DataIngestionEngine
from src.ingestion.profiler import AutomatedDataProfiler

__all__ = ["DataIngestionEngine", "AutomatedDataProfiler"]
