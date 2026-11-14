"""Configuration settings for the Autonomous Data Analytics Agent."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Model Provider Settings
    model_provider: str = Field(default="ollama", description="'ollama', 'gemini', or 'openai'")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model_name: str = Field(default="qwen2.5-coder:7b")
    
    # Cloud API Keys
    gemini_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # Execution & Sandboxing Controls
    duckdb_max_memory: str = Field(default="2GB")
    execution_timeout_seconds: int = Field(default=10)
    max_self_correction_iterations: int = Field(default=3)

    # Directories
    data_dir: Path = BASE_DIR / "data"
    synthetic_dir: Path = BASE_DIR / "data" / "synthetic"
    benchmarks_dir: Path = BASE_DIR / "data" / "benchmarks"

settings = Settings()
